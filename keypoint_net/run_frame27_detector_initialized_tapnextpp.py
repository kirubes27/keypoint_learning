"""Run frozen TAPNext++ from the detector's frame-27 predictions to frame 0."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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
from run_frame27_anchored_tapnextpp import (
    ANCHOR_FRAME,
    ANCHOR_REPLAY_MAXIMUM_ERROR_PX,
    CPU_MAX_PROJECTED_SECONDS,
    EXPECTED_WITNESS_IDS,
    OUTPUT_FRAMES,
    PREFIX_TRAVERSAL,
    TRAVERSAL,
    _array_record,
    canonicalize_anchor_traversal,
    projected_full_seconds,
    run_anchor_traversal,
)
from run_tapnextpp_512_bidirectional_teacher import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE_BYTES,
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
    "92db01963af3502cee29395221b418e65fc8e4b86ed901d6e2159b402607dd12"
)
EXPECTED_DETECTOR_PREDICTIONS_SHA256 = (
    "ab5f3fb46ff0d7187a88fe06b522d55a3f560ac2927e877f081d17062a270301"
)
EXPECTED_DETECTOR_RECEIPT_SHA256 = (
    "a0bc3137624b53c72e0e7acc194024a0bf7c4c586bd3adf0086ef66339608664"
)
EXPECTED_DETECTOR_CHECKPOINT_SHA256 = (
    "7e5d81241b1251254d46a420022dd1eda60f87c530196ea6063407c9ffb4e6cc"
)
EXPECTED_DETECTOR_RECEIPT_SCHEMA = (
    "leakage_safe_witness_distillation_training_receipt.v1"
)


def select_detector_anchor(
    frame_index: Any, local_prediction_px: Any
) -> np.ndarray:
    frames = np.asarray(frame_index, dtype=np.int64)
    prediction = np.asarray(local_prediction_px, dtype=np.float64)
    require(frames.ndim == 1, "detector frame index shape differs")
    require(
        prediction.shape == (frames.size, EXPECTED_WITNESSES, 2),
        "detector local prediction shape differs",
    )
    selected = np.flatnonzero(frames == ANCHOR_FRAME)
    require(selected.size == 1, "detector archive must contain frame 27 exactly once")
    anchor = prediction[int(selected[0])].astype(np.float32)
    require(bool(np.isfinite(anchor).all()), "detector anchor is non-finite")
    require(
        bool(((anchor >= 0.0) & (anchor < IMAGE_SIZE)).all()),
        "detector anchor leaves image",
    )
    return anchor


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(args.device == "cpu", "detector-initialized gate is local CPU only")
    require(not torch.cuda.is_available(), "CPU profile refuses a CUDA-visible runtime")
    require(not args.output_dir.exists(), "output directory already exists")
    require(
        _git(args.repo_root, "status", "--porcelain") == "",
        "implementation repository is dirty",
    )
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(implementation_head == args.expected_repo_head, "implementation HEAD differs")

    require(args.tapnet_root.is_dir(), "TAPNet checkout is missing")
    require(
        _git(args.tapnet_root, "status", "--porcelain") == "",
        "TAPNet checkout is dirty",
    )
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
    detector_prediction_record = file_record(args.detector_predictions)
    detector_receipt_record = file_record(args.detector_training_receipt)
    detector_checkpoint_record = file_record(args.detector_checkpoint)
    checkpoint_record = file_record(args.checkpoint)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic lock differs")
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "RGB manifest differs")
    require(
        detector_prediction_record["sha256"]
        == EXPECTED_DETECTOR_PREDICTIONS_SHA256,
        "detector prediction archive differs",
    )
    require(
        detector_receipt_record["sha256"] == EXPECTED_DETECTOR_RECEIPT_SHA256,
        "detector training receipt differs",
    )
    require(
        detector_checkpoint_record["sha256"]
        == EXPECTED_DETECTOR_CHECKPOINT_SHA256,
        "detector checkpoint differs",
    )
    require(checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256, "tracker checkpoint hash differs")
    require(
        checkpoint_record["size_bytes"] == EXPECTED_CHECKPOINT_SIZE_BYTES,
        "tracker checkpoint size differs",
    )

    detector_receipt = load_json(args.detector_training_receipt)
    require(
        detector_receipt.get("schema_version") == EXPECTED_DETECTOR_RECEIPT_SCHEMA,
        "detector receipt schema differs",
    )
    require(detector_receipt.get("paired_arm") == "candidate", "detector arm differs")
    require(detector_receipt.get("run_kind") == "full", "detector run kind differs")
    require(
        detector_receipt.get("decision_branch")
        == "freeze_checkpoint_and_run_truth_free_inference",
        "detector selection decision differs",
    )
    require(
        detector_receipt.get("training_predictions") == detector_prediction_record,
        "detector receipt prediction binding differs",
    )
    require(
        detector_receipt.get("selected_model") == detector_checkpoint_record,
        "detector receipt checkpoint binding differs",
    )

    # Read only the two prospectively permitted arrays from the detector archive.
    with np.load(args.detector_predictions, allow_pickle=False) as archive:
        require(
            {"frame_index", "local_3x3_prediction_px"}.issubset(archive.files),
            "detector archive omits required arrays",
        )
        detector_frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        detector_local_prediction = np.asarray(
            archive["local_3x3_prediction_px"], dtype=np.float64
        )
    witness_id = np.asarray(EXPECTED_WITNESS_IDS, dtype=np.int64)
    detector_anchor = select_detector_anchor(
        detector_frame_index, detector_local_prediction
    )
    internal_query_xy, support_xy = _build_internal_queries(detector_anchor)

    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    # The files are opened in the same order in which the model will see them.
    frames_bgr = {frame: _load_bgr(rgb_paths[frame]) for frame in TRAVERSAL}

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
    require(
        TAPNextPP.MODEL_SIZE == MODEL_COORDINATE_RESOLUTION,
        "model coordinate size differs",
    )
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
    projection = projected_full_seconds(
        model_load_seconds, prefix_seconds_a, prefix_seconds_b
    )
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
            prediction_xy[ANCHOR_FRAME].astype(np.float64) - detector_anchor,
            axis=-1,
        ).max()
    )
    require(
        anchor_replay_error <= ANCHOR_REPLAY_MAXIMUM_ERROR_PX + 1e-12,
        "frame-27 output exceeds the half-cell detector-query replay tolerance",
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_FRAME27_DETECTOR_INITIALIZED_TAPNEXTPP_TRACKS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        anchor_frame_index=np.asarray(ANCHOR_FRAME, dtype=np.int64),
        detector_anchor_coordinate_px=detector_anchor,
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
        "schema_version": "raw_frame27_detector_initialized_tapnextpp_512_support64.v1",
        "artifact_type": "prediction_only_detector_initialized_backward_material_tracks",
        "execution_profile": "cpu",
        "decision": {
            "raw_tracker_semantic_pass": True,
            "branch": "authorize_privileged_posthash_evaluation",
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "detector_training_predictions": detector_prediction_record,
            "detector_training_receipt": detector_receipt_record,
            "detector_checkpoint": detector_checkpoint_record,
            "tapnextpp_checkpoint": checkpoint_record,
            "tapnet": {
                "origin": tapnet_remote,
                "commit": tapnet_commit,
                "source_files": source_records,
            },
        },
        "anchor": {
            "source": "selected_augmented_detector_frame27_local_3x3_prediction_px",
            "frame_index": ANCHOR_FRAME,
            "witness_id": _array_record(witness_id),
            "coordinate_px": _array_record(detector_anchor),
            "raw_output_maximum_euclidean_replay_error_px": anchor_replay_error,
            "raw_output_replay_tolerance_px": ANCHOR_REPLAY_MAXIMUM_ERROR_PX,
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
            "anchor_traversal_helpers": file_record(
                Path(__file__).with_name("run_frame27_anchored_tapnextpp.py")
            ),
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
            "rgb_file_open_order": list(TRAVERSAL),
            "fresh_state_at_frame": ANCHOR_FRAME,
            "continuous_without_reset": True,
            "saved_frame_order": list(range(OUTPUT_FRAMES)),
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_180_rgb_hashes_rechecked_before_open": True,
            "opened_rgb_frame_indices_in_order": list(TRAVERSAL),
            "only_frame_index_and_local_prediction_read_from_detector_archive": True,
            "detector_target_coordinate_array_read": False,
            "detector_logits_or_hard_peaks_read": False,
            "frame0_to_23_truth_opened": False,
            "supplied_masks_opened": False,
            "renderer_angle_or_pivot_opened": False,
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
            "model_frame_call_count": 2 * len(PREFIX_TRAVERSAL) + len(TRAVERSAL),
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
    receipt_path = args.output_dir / "RAW_FRAME27_DETECTOR_INITIALIZED_TAPNEXTPP_RECEIPT.json"
    write_json(receipt_path, receipt)
    del model
    gc.collect()
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--detector-predictions", required=True, type=Path)
    parser.add_argument("--detector-training-receipt", required=True, type=Path)
    parser.add_argument("--detector-checkpoint", required=True, type=Path)
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
