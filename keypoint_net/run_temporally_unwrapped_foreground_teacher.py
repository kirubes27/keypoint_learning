"""Run the RGB-only temporally unwrapped foreground-moment teacher."""

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
        decode_temporally_unwrapped_sequence,
    )
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from global_silhouette_teacher import (
        IMAGE_SIZE,
        decode_temporally_unwrapped_sequence,
    )
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_LOCK_SHA256 = "3d246e3a1babd7dbeea6f919340781e8d1901dc3d668a851f049b3cf6ea63e5f"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_INITIALS_SHA256 = "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
FORWARD_REVERSE_ATOL_PX = 1.0e-10


COMMON_EXACT_FIELDS = (
    "foreground_mask",
    "background_rgb",
    "otsu_threshold",
    "component_label",
    "component_area",
    "centroid_xy",
    "second_moment_real_imag",
    "anisotropy",
    "candidate_angle_rad",
    "candidate_iou",
    "candidate_matrix",
)


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        value = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    require(value.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"RGB shape differs: {path}")
    return np.ascontiguousarray(value)


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
    forward = decode_temporally_unwrapped_sequence(
        rgbs, initial_coordinate, range(EXPECTED_FRAMES)
    )
    reverse = decode_temporally_unwrapped_sequence(
        rgbs,
        initial_coordinate,
        [0, *range(EXPECTED_FRAMES - 1, 0, -1)],
    )
    common_exact = all(np.array_equal(forward[key], reverse[key]) for key in COMMON_EXACT_FIELDS)
    selected_branch_exact = bool(
        np.array_equal(forward["selected_index"], reverse["selected_index"])
    )
    maximum_matrix_difference = float(
        np.max(np.abs(forward["matrix"] - reverse["matrix"]))
    )
    maximum_prediction_difference = float(
        np.max(np.abs(forward["prediction_xy"] - reverse["prediction_xy"]))
    )
    semantic_pass = bool(
        common_exact
        and selected_branch_exact
        and maximum_matrix_difference <= FORWARD_REVERSE_ATOL_PX
        and maximum_prediction_difference <= FORWARD_REVERSE_ATOL_PX
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_TEMPORALLY_UNWRAPPED_FOREGROUND_TEACHER.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(initial_frame, dtype=np.int64),
        initial_coordinate_px=initial_coordinate,
        **{key: forward[key] for key in COMMON_EXACT_FIELDS},
        forward_selected_index=forward["selected_index"],
        forward_selected_angle_unwrapped_rad=forward["selected_angle_unwrapped_rad"],
        forward_selected_iou_diagnostic=forward["selected_iou_diagnostic"],
        forward_traversal_step_delta_rad=forward["traversal_step_delta_rad"],
        forward_closure_delta_rad=forward["closure_delta_rad"],
        forward_matrix=forward["matrix"],
        forward_prediction_xy=forward["prediction_xy"],
        reverse_selected_index=reverse["selected_index"],
        reverse_selected_angle_unwrapped_rad=reverse["selected_angle_unwrapped_rad"],
        reverse_selected_iou_diagnostic=reverse["selected_iou_diagnostic"],
        reverse_traversal_step_delta_rad=reverse["traversal_step_delta_rad"],
        reverse_closure_delta_rad=reverse["closure_delta_rad"],
        reverse_matrix=reverse["matrix"],
        reverse_prediction_xy=reverse["prediction_xy"],
    )
    maximum_forward_step = float(
        max(
            np.max(np.abs(forward["traversal_step_delta_rad"])),
            abs(float(forward["closure_delta_rad"])),
        )
    )
    maximum_reverse_step = float(
        max(
            np.max(np.abs(reverse["traversal_step_delta_rad"])),
            abs(float(reverse["closure_delta_rad"])),
        )
    )
    receipt = {
        "schema_version": "raw_temporally_unwrapped_foreground_moment_teacher.v1",
        "artifact_type": "prediction_only_temporally_unwrapped_global_rigid_tracks",
        "decision": {
            "raw_temporal_semantic_pass": semantic_pass,
            "branch": (
                "authorize_privileged_posthash_evaluation"
                if semantic_pass
                else "stop_before_truth_due_temporal_control_failure"
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
            "foreground_moment_primitives": file_record(
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
            "common_observation_arrays_forward_reverse_exact": common_exact,
            "selected_pi_branch_forward_reverse_exact": selected_branch_exact,
            "maximum_forward_reverse_matrix_difference": maximum_matrix_difference,
            "maximum_forward_reverse_prediction_difference_px": maximum_prediction_difference,
            "forward_reverse_atol_px": FORWARD_REVERSE_ATOL_PX,
            "maximum_absolute_forward_step_deg_including_closure": float(
                np.degrees(maximum_forward_step)
            ),
            "maximum_absolute_reverse_step_deg_including_closure": float(
                np.degrees(maximum_reverse_step)
            ),
            "all_steps_strictly_below_90_degrees": bool(
                maximum_forward_step < 0.5 * np.pi
                and maximum_reverse_step < 0.5 * np.pi
            ),
        },
        "diagnostics": {
            "forward_selected_iou_minimum": float(
                np.min(forward["selected_iou_diagnostic"])
            ),
            "forward_selected_iou_median": float(
                np.median(forward["selected_iou_diagnostic"])
            ),
        },
        "frame_count": EXPECTED_FRAMES,
        "witness_count": EXPECTED_WITNESSES,
        "command_argv": list(sys.argv),
        "runtime_seconds": float(time.time() - started),
        "privileged_evaluation_authorized": semantic_pass,
        "training_or_weight_update_performed": False,
        "statistical_scope": "one rendered 180-frame orbit; temporal controls are descriptive only",
    }
    receipt_path = args.output_dir / "RAW_TEMPORALLY_UNWRAPPED_FOREGROUND_TEACHER_RECEIPT.json"
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
