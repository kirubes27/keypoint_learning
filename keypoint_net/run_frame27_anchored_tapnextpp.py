"""Run frozen TAPNext++ 512 once from frame 27 backward to frame 0."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from material_transport_gate_io import (
    file_record,
    load_json,
    require,
    resolve_rgb_paths,
    write_json,
)
from run_tapnextpp_512_bidirectional_teacher import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE_BYTES,
    EXPECTED_INTERNAL_QUERIES,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TAPNET_COMMIT,
    EXPECTED_WITNESSES,
    IMAGE_SIZE,
    MODEL_COORDINATE_RESOLUTION,
    MODEL_INPUT_RESOLUTION,
    SOURCE_RELPATHS,
    SUPPORT_MODE,
    SUPPORT_POINTS_PER_QUERY,
    SUPPORT_RADIUS_MODEL_INPUT_PX,
    SUPPORT_RADIUS_SPACE,
    _build_internal_queries,
    _coordinate_mapping_smoke,
    _environment_record,
    _git,
    _load_bgr,
    _peak_rss_record,
    _validate_prediction_arrays,
    _write_pip_freeze,
)


EXPECTED_LOCK_SHA256 = (
    "e023344fca7abca6bf9727a409ffc00b7028b994b86b9a24fffc552cef79f4d0"
)
EXPECTED_SOURCE_SCORE_SHA256 = (
    "d5ea9d47cb3c475c7c237a5ee7cd6347af3dedded9685985f83aa504bca92fc2"
)
EXPECTED_WITNESS_IDS = (
    1857,
    2237,
    2241,
    12601,
    12606,
    12980,
    12993,
    13100,
    13868,
    14394,
)
ANCHOR_FRAME = 27
OUTPUT_FRAMES = ANCHOR_FRAME + 1
PREFIX_FRAMES = 5
TRAVERSAL = tuple(range(ANCHOR_FRAME, -1, -1))
PREFIX_TRAVERSAL = TRAVERSAL[:PREFIX_FRAMES]
MODEL_FRAME_CALLS = 2 * PREFIX_FRAMES + OUTPUT_FRAMES
CPU_TIMING_SAFETY_FACTOR = 1.25
CPU_MAX_PROJECTED_SECONDS = 1_200.0
ANCHOR_REPLAY_MAXIMUM_ERROR_PX = (511.0 / 63.0) / np.sqrt(2.0)


def _array_record(array: Any) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256_dtype_shape_bytes": digest.hexdigest(),
    }


def run_anchor_traversal(
    model: Any,
    frames_bgr: Mapping[int, np.ndarray],
    query_xy: np.ndarray,
    traversal: Iterable[int],
    *,
    autocast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a fresh recurrent traversal whose first frame carries the queries."""

    order = [int(value) for value in traversal]
    require(order and order[0] == ANCHOR_FRAME, "traversal must start at frame 27")
    require(len(set(order)) == len(order), "traversal repeats a frame")
    positions: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    state = None
    for step, frame_index in enumerate(order):
        require(frame_index in frames_bgr, f"traversal frame is not loaded: {frame_index}")
        if step == 0:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index],
                query_points_xy=query_xy,
                autocast=autocast,
            )
        else:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index], state=state, autocast=autocast
            )
        current_xy = np.asarray(current_xy, dtype=np.float32)
        current_visible = np.asarray(current_visible, dtype=bool)
        require(
            current_xy.shape == (EXPECTED_INTERNAL_QUERIES, 2),
            "internal query output shape differs",
        )
        require(
            current_visible.shape == (EXPECTED_INTERNAL_QUERIES,),
            "internal visibility output shape differs",
        )
        positions.append(current_xy[:EXPECTED_WITNESSES].copy())
        visibility.append(current_visible[:EXPECTED_WITNESSES].copy())
    return np.stack(positions), np.stack(visibility)


def canonicalize_anchor_traversal(
    traversal: Iterable[int], positions: Any, visibility: Any
) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray(list(traversal), dtype=np.int64)
    position_array = np.asarray(positions)
    visibility_array = np.asarray(visibility)
    require(
        np.array_equal(np.sort(order), np.arange(OUTPUT_FRAMES)),
        "anchor traversal is not a frame-0--27 permutation",
    )
    require(position_array.shape == (OUTPUT_FRAMES, EXPECTED_WITNESSES, 2), "position shape differs")
    require(visibility_array.shape == (OUTPUT_FRAMES, EXPECTED_WITNESSES), "visibility shape differs")
    canonical_positions = np.empty_like(position_array)
    canonical_visibility = np.empty_like(visibility_array)
    canonical_positions[order] = position_array
    canonical_visibility[order] = visibility_array
    return canonical_positions, canonical_visibility


def projected_full_seconds(
    model_load_seconds: float, prefix_seconds_a: float, prefix_seconds_b: float
) -> float:
    values = np.asarray(
        [model_load_seconds, prefix_seconds_a, prefix_seconds_b], dtype=np.float64
    )
    require(bool(np.isfinite(values).all() and (values >= 0.0).all()), "timing input invalid")
    slow_seconds_per_frame = max(prefix_seconds_a, prefix_seconds_b) / PREFIX_FRAMES
    return float(
        CPU_TIMING_SAFETY_FACTOR
        * (model_load_seconds + MODEL_FRAME_CALLS * slow_seconds_per_frame)
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(args.device == "cpu", "frame-27 gate is local CPU only")
    require(not torch.cuda.is_available(), "CPU profile refuses a CUDA-visible runtime")
    require(not args.output_dir.exists(), "output directory already exists")
    require(_git(args.repo_root, "status", "--porcelain") == "", "implementation repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(implementation_head == args.expected_repo_head, "implementation HEAD differs")

    require(args.tapnet_root.is_dir(), "TAPNet checkout is missing")
    require(_git(args.tapnet_root, "status", "--porcelain") == "", "TAPNet checkout is dirty")
    tapnet_commit = _git(args.tapnet_root, "rev-parse", "HEAD")
    require(tapnet_commit == EXPECTED_TAPNET_COMMIT, "TAPNet commit differs")
    tapnet_remote = _git(args.tapnet_root, "remote", "get-url", "origin")
    require(
        tapnet_remote
        in {
            "https://github.com/google-deepmind/tapnet.git",
            "https://github.com/deepmind/tapnet.git",
        },
        "TAPNet origin differs",
    )

    lock_record = file_record(args.semantic_lock)
    manifest_record = file_record(args.manifest)
    source_score_record = file_record(args.source_score_archive)
    checkpoint_record = file_record(args.checkpoint)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic lock differs")
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "RGB manifest differs")
    require(source_score_record["sha256"] == EXPECTED_SOURCE_SCORE_SHA256, "anchor source differs")
    require(checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256, "checkpoint hash differs")
    require(checkpoint_record["size_bytes"] == EXPECTED_CHECKPOINT_SIZE_BYTES, "checkpoint size differs")

    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    with np.load(args.source_score_archive, allow_pickle=False) as archive:
        require(
            {"witness_id", "anchor_frame_index", "anchor_target_coordinate_px"}.issubset(
                archive.files
            ),
            "anchor source omits required arrays",
        )
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        anchor_frame = int(archive["anchor_frame_index"])
        anchor_coordinate = np.asarray(
            archive["anchor_target_coordinate_px"], dtype=np.float32
        )
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "witness identity/order differs")
    require(anchor_frame == ANCHOR_FRAME, "anchor frame differs")
    require(anchor_coordinate.shape == (EXPECTED_WITNESSES, 2), "anchor coordinate shape differs")
    require(bool(np.isfinite(anchor_coordinate).all()), "anchor coordinate is non-finite")
    require(bool(((anchor_coordinate >= 0.0) & (anchor_coordinate < IMAGE_SIZE)).all()), "anchor leaves image")
    internal_query_xy, support_xy = _build_internal_queries(anchor_coordinate)
    frames_bgr = {frame: _load_bgr(rgb_paths[frame]) for frame in range(OUTPUT_FRAMES)}

    sys.path.insert(0, str(args.tapnet_root.resolve()))
    import einops  # pylint: disable=import-outside-toplevel
    import torchvision  # pylint: disable=import-outside-toplevel
    from tapnet.tapnextpp.votsp2026 import utils as tapnextpp_utils  # pylint: disable=import-outside-toplevel
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP  # pylint: disable=import-outside-toplevel

    coordinate_mapping = _coordinate_mapping_smoke(
        internal_query_xy,
        tapnextpp_utils.display_to_model,
        tapnextpp_utils.model_to_display,
    )
    require(TAPNextPP.MODEL_SIZE == MODEL_COORDINATE_RESOLUTION, "model coordinate size differs")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model_load_started = time.perf_counter()
    model = TAPNextPP.from_checkpoint(
        args.checkpoint,
        device=args.device,
        half_precision=False,
        compile_model=False,
        input_resolution=MODEL_INPUT_RESOLUTION,
    )
    model_load_seconds = float(time.perf_counter() - model_load_started)
    require(model.input_resolution == MODEL_INPUT_RESOLUTION, "model input resolution differs")
    require(model.device.type == "cpu", "model execution device differs")

    prefix_started = time.perf_counter()
    prefix_positions_a, prefix_visibility_a = run_anchor_traversal(
        model,
        frames_bgr,
        internal_query_xy,
        PREFIX_TRAVERSAL,
        autocast=False,
    )
    prefix_seconds_a = float(time.perf_counter() - prefix_started)
    prefix_started = time.perf_counter()
    prefix_positions_b, prefix_visibility_b = run_anchor_traversal(
        model,
        frames_bgr,
        internal_query_xy,
        PREFIX_TRAVERSAL,
        autocast=False,
    )
    prefix_seconds_b = float(time.perf_counter() - prefix_started)
    prefix_positions_exact = bool(np.array_equal(prefix_positions_a, prefix_positions_b))
    prefix_visibility_exact = bool(np.array_equal(prefix_visibility_a, prefix_visibility_b))
    require(prefix_positions_exact, "repeated prefix positions differ")
    require(prefix_visibility_exact, "repeated prefix visibility differs")
    _validate_prediction_arrays(prefix_positions_a, prefix_visibility_a)
    projection = projected_full_seconds(model_load_seconds, prefix_seconds_a, prefix_seconds_b)
    require(projection <= CPU_MAX_PROJECTED_SECONDS, "projected CPU runtime exceeds gate")

    full_started = time.perf_counter()
    ordered_positions, ordered_visibility = run_anchor_traversal(
        model, frames_bgr, internal_query_xy, TRAVERSAL, autocast=False
    )
    full_seconds = float(time.perf_counter() - full_started)
    prediction_xy, visible = canonicalize_anchor_traversal(
        TRAVERSAL, ordered_positions, ordered_visibility
    )
    _validate_prediction_arrays(prediction_xy, visible)
    anchor_replay_error = float(
        np.linalg.norm(
            prediction_xy[ANCHOR_FRAME].astype(np.float64) - anchor_coordinate,
            axis=-1,
        ).max()
    )
    require(
        anchor_replay_error <= ANCHOR_REPLAY_MAXIMUM_ERROR_PX + 1e-12,
        "frame-27 output exceeds the half-cell anchor replay tolerance",
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_FRAME27_ANCHORED_TAPNEXTPP_TRACKS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        anchor_frame_index=np.asarray(anchor_frame, dtype=np.int64),
        anchor_coordinate_px=anchor_coordinate,
        frame_index=np.arange(OUTPUT_FRAMES, dtype=np.int64),
        traversal=np.asarray(TRAVERSAL, dtype=np.int64),
        prediction_xy=prediction_xy,
        visible=visible,
        repeated_prefix_traversal=np.asarray(PREFIX_TRAVERSAL, dtype=np.int64),
        repeated_prefix_prediction_xy=prefix_positions_a,
        repeated_prefix_visible=prefix_visibility_a,
    )
    pip_freeze_record = _write_pip_freeze(args.output_dir)
    source_records = {
        relpath: file_record(args.tapnet_root / relpath) for relpath in SOURCE_RELPATHS
    }
    environment = _environment_record(args.device, torchvision, einops)
    receipt = {
        "schema_version": "raw_frame27_anchored_tapnextpp_512_support64.v1",
        "artifact_type": "prediction_only_frame27_anchored_backward_material_tracks",
        "execution_profile": "cpu",
        "decision": {
            "raw_tracker_semantic_pass": True,
            "branch": "authorize_privileged_posthash_evaluation",
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "frame27_anchor_score_archive": source_score_record,
            "tapnextpp_checkpoint": checkpoint_record,
            "tapnet": {
                "origin": tapnet_remote,
                "commit": tapnet_commit,
                "source_files": source_records,
            },
        },
        "anchor": {
            "frame_index": anchor_frame,
            "witness_id": _array_record(witness_id),
            "coordinate_px": _array_record(anchor_coordinate),
            "raw_output_maximum_euclidean_replay_error_px": anchor_replay_error,
            "raw_output_replay_tolerance_px": ANCHOR_REPLAY_MAXIMUM_ERROR_PX,
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
            "official_bridge_helpers": file_record(
                Path(__file__).with_name("run_tapnextpp_512_bidirectional_teacher.py")
            ),
        },
        "environment": environment,
        "pip_freeze": pip_freeze_record,
        "coordinate_mapping": coordinate_mapping,
        "support_configuration": {
            "mode": SUPPORT_MODE,
            "support_radius_space": SUPPORT_RADIUS_SPACE,
            "support_radius_model_input_px": SUPPORT_RADIUS_MODEL_INPUT_PX,
            "support_points_per_real_query": SUPPORT_POINTS_PER_QUERY,
            "real_query_count": EXPECTED_WITNESSES,
            "support_query_count": int(support_xy.shape[0]),
            "internal_query_count": int(internal_query_xy.shape[0]),
            "support_trajectories_saved": False,
        },
        "traversal": {
            "order": list(TRAVERSAL),
            "fresh_state_at_frame": ANCHOR_FRAME,
            "continuous_without_reset": True,
            "saved_frame_order": list(range(OUTPUT_FRAMES)),
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_180_rgb_hashes_rechecked_before_open": True,
            "opened_rgb_frame_indices": list(range(OUTPUT_FRAMES)),
            "only_frame27_anchor_arrays_read_from_score_archive": True,
            "validation_score_map_arrays_read": False,
            "frame0_to_23_truth_opened": False,
            "supplied_masks_opened": False,
            "renderer_angle_or_pivot_opened": False,
            "learned_keypoint_checkpoint_opened": False,
            "operator_or_prior_evaluation_opened": False,
            "local_laptop_gpu_used": False,
            "local_laptop_cpu_only": True,
            "autocast_enabled": False,
            "repeated_prefix_positions_exact": prefix_positions_exact,
            "repeated_prefix_visibility_exact": prefix_visibility_exact,
            "all_coordinates_finite_and_in_image": True,
            "official_coordinate_roundtrip_pass": True,
            "only_ten_real_query_trajectories_saved": True,
            "training_or_weight_update_performed": False,
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "prefix_seconds_a": prefix_seconds_a,
            "prefix_seconds_b": prefix_seconds_b,
            "projected_total_seconds": projection,
            "maximum_authorized_seconds": CPU_MAX_PROJECTED_SECONDS,
            "full_traversal_seconds": full_seconds,
            "model_frame_call_count": MODEL_FRAME_CALLS,
        },
        "visibility_true_count_diagnostic": int(visible.sum()),
        "frame_count": OUTPUT_FRAMES,
        "witness_count": EXPECTED_WITNESSES,
        "peak_memory": _peak_rss_record(),
        "command_argv": list(sys.argv),
        "runtime_seconds": float(time.time() - started),
        "privileged_evaluation_authorized": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over one correlated 28-frame backward continuation; descriptive only",
    }
    receipt_path = args.output_dir / "RAW_FRAME27_ANCHORED_TAPNEXTPP_RECEIPT.json"
    write_json(receipt_path, receipt)
    del model
    gc.collect()
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-score-archive", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--tapnet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
