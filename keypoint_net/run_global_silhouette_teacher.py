"""Run the RGB-only global silhouette material-point teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from .global_silhouette_teacher import (
        IMAGE_SIZE,
        MINIMUM_SELECTED_IOU,
        decode_sequence,
    )
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from global_silhouette_teacher import IMAGE_SIZE, MINIMUM_SELECTED_IOU, decode_sequence
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_LOCK_SHA256 = "bcb24fc39510b829acdf64022a3ab7e7cfa4ffc6a3c89fd2510a9ea788b27362"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_INITIALS_SHA256 = "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        value = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    require(value.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"RGB shape differs: {path}")
    return np.ascontiguousarray(value)


def _require_exact(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    require(set(first) == set(second), "frame-order control keys differ")
    for key in first:
        require(np.array_equal(first[key], second[key]), f"frame-order control changed {key}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")

    lock_record = file_record(args.semantic_lock)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic-lock SHA-256 differs")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    initials_record = file_record(args.initials)
    require(initials_record["sha256"] == EXPECTED_INITIALS_SHA256, "initials SHA-256 differs")

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
        initial_coordinate = np.asarray(archive["initial_coordinate_px"], dtype=np.float64)
    require(witness_id.shape == (EXPECTED_WITNESSES,), "witness IDs differ")
    require(initial_frame == 0, "initial frame differs")
    require(initial_coordinate.shape == (EXPECTED_WITNESSES, 2), "initial coordinates differ")

    rgbs = [_load_rgb(path) for path in rgb_paths]
    canonical = decode_sequence(rgbs, initial_coordinate, range(EXPECTED_FRAMES))
    reverse_control = decode_sequence(
        rgbs, initial_coordinate, reversed(range(EXPECTED_FRAMES))
    )
    _require_exact(canonical, reverse_control)
    pose_semantic_pass = bool(
        (canonical["selected_iou"] >= MINIMUM_SELECTED_IOU).all()
    )
    require(bool(np.isfinite(canonical["prediction_xy"]).all()), "predictions are non-finite")

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_GLOBAL_SILHOUETTE_TEACHER.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(initial_frame, dtype=np.int64),
        initial_coordinate_px=initial_coordinate,
        **canonical,
    )
    receipt = {
        "schema_version": "raw_global_rgb_silhouette_teacher.v1",
        "artifact_type": "prediction_only_global_rigid_silhouette_tracks",
        "decision": {
            "raw_pose_semantic_pass": pose_semantic_pass,
            "branch": (
                "authorize_privileged_posthash_evaluation"
                if pose_semantic_pass
                else "stop_before_truth_due_pose_semantic_failure"
            ),
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "frame_zero_initials": initials_record,
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "silhouette_primitives": file_record(
                Path(__file__).with_name("global_silhouette_teacher.py")
            ),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
        },
        "environment": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_rgb_hashes_rechecked_before_open": True,
            "only_frame_zero_initial_coordinates_opened": True,
            "supplied_masks_opened": False,
            "non_frame_zero_truth_opened": False,
            "renderer_angle_or_pivot_opened": False,
            "model_optimizer_feature_or_operator_opened": False,
            "frame_order_reversal_exact_every_array": True,
            "minimum_selected_silhouette_iou": MINIMUM_SELECTED_IOU,
            "observed_minimum_selected_silhouette_iou": float(
                np.min(canonical["selected_iou"])
            ),
            "observed_minimum_orientation_ambiguity_gap_iou": float(
                np.min(canonical["ambiguity_gap_iou"])
            ),
            "frames_below_minimum_selected_silhouette_iou": int(
                np.sum(canonical["selected_iou"] < MINIMUM_SELECTED_IOU)
            ),
        },
        "frame_count": EXPECTED_FRAMES,
        "witness_count": EXPECTED_WITNESSES,
        "command_argv": list(sys.argv),
        "runtime_seconds": float(time.time() - started),
        "training_or_weight_update_performed": False,
        "privileged_evaluation_authorized": pose_semantic_pass,
        "statistical_scope": "one rendered 180-frame orbit; pose diagnostics are descriptive only",
    }
    receipt_path = args.output_dir / "RAW_GLOBAL_SILHOUETTE_TEACHER_RECEIPT.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--initials", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
