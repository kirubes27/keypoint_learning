"""Run the frozen cluster-only TAPNext++ 256 bidirectional teacher gate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch

try:
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_LOCK_SHA256 = "af205b4ba37c58972b6e68d3e4c59b5b31c016440bc23ab9fc774bbf616736d7"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_INITIALS_SHA256 = "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
EXPECTED_TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
EXPECTED_CHECKPOINT_SHA256 = "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"
EXPECTED_CHECKPOINT_SIZE_BYTES = 2_532_282_370
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
MODEL_RESOLUTION = 256
DETERMINISM_PREFIX_FRAMES = 5


SOURCE_RELPATHS = (
    "tapnet/tapnextpp/votsp2026/model.py",
    "tapnet/tapnextpp/votsp2026/utils.py",
    "tapnet/tapnext/tapnext_torch.py",
    "tapnet/tapnext/tapnext_lru_modules.py",
    "tapnet/tapnext/pscan.py",
)


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"RGB shape differs: {path}")
    return np.ascontiguousarray(rgb[..., ::-1])


def _run_traversal(
    model: Any,
    frames_bgr: list[np.ndarray],
    query_xy: np.ndarray,
    traversal: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    order = [int(value) for value in traversal]
    require(order and order[0] == 0, "TAPNext++ traversal must start at frame zero")
    positions: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    state = None
    for step, frame_index in enumerate(order):
        if step == 0:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index],
                query_points_xy=query_xy,
                autocast=True,
            )
        else:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index],
                state=state,
                autocast=True,
            )
        positions.append(np.asarray(current_xy, dtype=np.float32))
        visibility.append(np.asarray(current_visible, dtype=bool))
    return np.stack(positions), np.stack(visibility)


def _canonicalize(
    traversal: Iterable[int],
    positions: np.ndarray,
    visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray([int(value) for value in traversal], dtype=np.int64)
    require(order.ndim == 1 and len(order) == len(positions), "traversal output length differs")
    require(np.array_equal(np.sort(order), np.arange(len(order))), "traversal is not a permutation")
    canonical_positions = np.empty_like(positions)
    canonical_visibility = np.empty_like(visibility)
    canonical_positions[order] = positions
    canonical_visibility[order] = visibility
    return canonical_positions, canonical_visibility


def _validate_prediction_arrays(positions: np.ndarray, visibility: np.ndarray) -> None:
    require(
        positions.shape[1:] == (EXPECTED_WITNESSES, 2),
        "TAPNext++ prediction query shape differs",
    )
    require(visibility.shape == positions.shape[:2], "TAPNext++ visibility shape differs")
    require(bool(np.isfinite(positions).all()), "TAPNext++ produced non-finite coordinates")
    require(
        bool(
            (
                (positions[..., 0] >= 0.0)
                & (positions[..., 0] < IMAGE_SIZE)
                & (positions[..., 1] >= 0.0)
                & (positions[..., 1] < IMAGE_SIZE)
            ).all()
        ),
        "TAPNext++ produced an out-of-image coordinate",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(args.device == "cuda", "this frozen gate is cluster CUDA only")
    require(torch.cuda.is_available(), "CUDA is unavailable; laptop GPU/CPU fallback is forbidden")
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "implementation repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")

    require(args.tapnet_root.is_dir(), "TAPNet checkout is missing")
    require(_git(args.tapnet_root, "status", "--porcelain") == "", "TAPNet checkout is dirty")
    tapnet_commit = _git(args.tapnet_root, "rev-parse", "HEAD")
    require(tapnet_commit == EXPECTED_TAPNET_COMMIT, "TAPNet commit differs")
    tapnet_remote = _git(args.tapnet_root, "remote", "get-url", "origin")
    require(
        tapnet_remote in {
            "https://github.com/google-deepmind/tapnet.git",
            "https://github.com/deepmind/tapnet.git",
        },
        "TAPNet origin differs",
    )

    lock_record = file_record(args.semantic_lock)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic-lock SHA-256 differs")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    initials_record = file_record(args.initials)
    require(initials_record["sha256"] == EXPECTED_INITIALS_SHA256, "initials SHA-256 differs")
    checkpoint_record = file_record(args.checkpoint)
    require(
        checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "TAPNext++ checkpoint SHA-256 differs",
    )
    require(
        checkpoint_record["size_bytes"] == EXPECTED_CHECKPOINT_SIZE_BYTES,
        "TAPNext++ checkpoint size differs",
    )

    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    require(len(rgb_paths) == EXPECTED_FRAMES, "RGB frame count differs")
    with np.load(args.initials) as archive:
        require(
            set(archive.files) == {"witness_id", "initial_frame_index", "initial_coordinate_px"},
            "initials expose unexpected arrays",
        )
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        initial_frame = int(archive["initial_frame_index"])
        initial_coordinate = np.asarray(archive["initial_coordinate_px"], dtype=np.float32)
    require(witness_id.shape == (EXPECTED_WITNESSES,), "witness IDs differ")
    require(initial_frame == 0, "initial frame differs")
    require(initial_coordinate.shape == (EXPECTED_WITNESSES, 2), "initial coordinates differ")
    frames_bgr = [_load_bgr(path) for path in rgb_paths]

    sys.path.insert(0, str(args.tapnet_root.resolve()))
    import einops  # pylint: disable=import-outside-toplevel
    import torchvision  # pylint: disable=import-outside-toplevel
    from tapnet.tapnextpp.votsp2026.model import (  # pylint: disable=import-outside-toplevel
        TAPNextPP,
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = TAPNextPP.from_checkpoint(
        args.checkpoint,
        device=args.device,
        half_precision=False,
        compile_model=False,
        input_resolution=MODEL_RESOLUTION,
    )
    prefix_order = list(range(DETERMINISM_PREFIX_FRAMES))
    prefix_positions_a, prefix_visibility_a = _run_traversal(
        model, frames_bgr, initial_coordinate, prefix_order
    )
    prefix_positions_b, prefix_visibility_b = _run_traversal(
        model, frames_bgr, initial_coordinate, prefix_order
    )
    prefix_positions_exact = bool(np.array_equal(prefix_positions_a, prefix_positions_b))
    prefix_visibility_exact = bool(np.array_equal(prefix_visibility_a, prefix_visibility_b))
    require(prefix_positions_exact, "repeated TAPNext++ prefix positions differ")
    require(prefix_visibility_exact, "repeated TAPNext++ prefix visibility differs")
    _validate_prediction_arrays(prefix_positions_a, prefix_visibility_a)

    forward_order = list(range(EXPECTED_FRAMES))
    reverse_order = [0, *range(EXPECTED_FRAMES - 1, 0, -1)]
    forward_positions, forward_visibility = _run_traversal(
        model, frames_bgr, initial_coordinate, forward_order
    )
    reverse_positions_ordered, reverse_visibility_ordered = _run_traversal(
        model, frames_bgr, initial_coordinate, reverse_order
    )
    reverse_positions, reverse_visibility = _canonicalize(
        reverse_order, reverse_positions_ordered, reverse_visibility_ordered
    )
    _validate_prediction_arrays(forward_positions, forward_visibility)
    _validate_prediction_arrays(reverse_positions, reverse_visibility)
    require(forward_positions.shape[0] == EXPECTED_FRAMES, "forward frame count differs")
    require(reverse_positions.shape[0] == EXPECTED_FRAMES, "reverse frame count differs")

    direction_difference = np.linalg.norm(
        forward_positions.astype(np.float64) - reverse_positions.astype(np.float64),
        axis=-1,
    )
    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_TAPNEXTPP_256_BIDIRECTIONAL_TEACHER.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(initial_frame, dtype=np.int64),
        initial_coordinate_px=initial_coordinate,
        forward_prediction_xy=forward_positions,
        forward_visible=forward_visibility,
        reverse_prediction_xy=reverse_positions,
        reverse_visible=reverse_visibility,
        forward_reverse_difference_px=direction_difference,
        forward_traversal=np.asarray(forward_order, dtype=np.int64),
        reverse_traversal=np.asarray(reverse_order, dtype=np.int64),
        repeated_prefix_prediction_xy=prefix_positions_a,
        repeated_prefix_visible=prefix_visibility_a,
    )

    source_records = {
        relpath: file_record(args.tapnet_root / relpath) for relpath in SOURCE_RELPATHS
    }
    receipt = {
        "schema_version": "raw_tapnextpp_256_bidirectional_teacher.v1",
        "artifact_type": "prediction_only_pretrained_bidirectional_material_tracks",
        "decision": {
            "raw_tracker_semantic_pass": True,
            "branch": "authorize_privileged_posthash_evaluation",
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "frame_zero_initials": initials_record,
            "tapnextpp_checkpoint": checkpoint_record,
            "tapnet": {
                "origin": tapnet_remote,
                "commit": tapnet_commit,
                "source_files": source_records,
            },
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "einops": einops.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_device_count_visible": torch.cuda.device_count(),
            "model_input_resolution": MODEL_RESOLUTION,
            "half_precision": False,
            "torch_compile": False,
            "allow_tf32": False,
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_rgb_hashes_rechecked_before_open": True,
            "only_frame_zero_initial_coordinates_opened": True,
            "supplied_masks_opened": False,
            "non_frame_zero_truth_opened": False,
            "renderer_angle_or_pivot_opened": False,
            "learned_keypoint_checkpoint_or_features_opened": False,
            "local_laptop_gpu_used": False,
            "cluster_cuda_only": True,
            "repeated_prefix_frame_count": DETERMINISM_PREFIX_FRAMES,
            "repeated_prefix_positions_exact": prefix_positions_exact,
            "repeated_prefix_visibility_exact": prefix_visibility_exact,
            "all_forward_coordinates_finite_and_in_image": True,
            "all_reverse_coordinates_finite_and_in_image": True,
            "forward_visible_count_diagnostic": int(np.sum(forward_visibility)),
            "reverse_visible_count_diagnostic": int(np.sum(reverse_visibility)),
        },
        "direction_disagreement_diagnostic_px": {
            "n": int(direction_difference.size),
            "mean": float(np.mean(direction_difference)),
            "median": float(np.median(direction_difference)),
            "q90": float(np.quantile(direction_difference, 0.90)),
            "maximum": float(np.max(direction_difference)),
        },
        "frame_count": EXPECTED_FRAMES,
        "witness_count": EXPECTED_WITNESSES,
        "command_argv": list(sys.argv),
        "runtime_seconds": float(time.time() - started),
        "privileged_evaluation_authorized": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over one correlated 180-frame orbit; descriptive only",
    }
    receipt_path = args.output_dir / "RAW_TAPNEXTPP_256_BIDIRECTIONAL_TEACHER_RECEIPT.json"
    write_json(receipt_path, receipt)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--initials", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--tapnet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
