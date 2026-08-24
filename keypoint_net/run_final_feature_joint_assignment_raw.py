"""Run Gate 4c truth-blind joint assignment without opening evaluation truth."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from final_feature_joint_assignment import (
    EXPECTED_SEMANTIC_LOCK_SHA256,
    EXPECTED_SOURCE_RAW_RECEIPT_SHA256,
    EXPECTED_SOURCE_SCORE_SHA256,
    FINAL_FEATURE_NAME,
    compact_information_boundary,
    decode_joint_assignments,
    extract_final_feature_scores,
    file_record,
    require,
)


SCHEMA_VERSION = "final_feature_joint_assignment_raw_receipt.v1"
MAXIMUM_PROJECTED_SECONDS = 3600.0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root differs: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_clean_head(repo_root: Path, expected_head: str) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == expected_head, "repository HEAD differs")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "repository is not clean")
    return head


def _maximum_replay_difference(
    forward: dict[str, np.ndarray], reversed_restored: dict[str, np.ndarray]
) -> tuple[bool, dict[str, float]]:
    require(set(forward) == set(reversed_restored), "replay field set differs")
    exact = True
    differences: dict[str, float] = {}
    for name in sorted(forward):
        left = np.asarray(forward[name])
        right = np.asarray(reversed_restored[name])
        require(left.shape == right.shape, f"replay shape differs: {name}")
        same = bool(np.array_equal(left, right))
        exact = exact and same
        differences[name] = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
    return exact, differences


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "raw output directory already exists")
    command_text = " ".join(sys.argv).lower()
    for forbidden in (
        "sealed_validation_truth",
        "mask_manifest",
        "evaluation_result",
        "baseline_arrays",
        "rgb_manifest",
    ):
        require(forbidden not in command_text, f"privileged input entered raw command: {forbidden}")

    repository_head = _verify_clean_head(args.repo_root, args.expected_repo_head)
    semantic_lock_record = file_record(args.semantic_lock)
    require(
        semantic_lock_record["sha256"] == EXPECTED_SEMANTIC_LOCK_SHA256,
        "semantic-lock SHA-256 differs",
    )
    source_receipt_record = file_record(args.source_raw_receipt)
    require(
        source_receipt_record["sha256"] == EXPECTED_SOURCE_RAW_RECEIPT_SHA256,
        "source raw receipt SHA-256 differs",
    )
    source_receipt = _load_json(args.source_raw_receipt)
    require(
        source_receipt.get("schema_version")
        == "augmented_encoder_head_localization_raw_receipt.v1",
        "source raw receipt schema differs",
    )
    require(source_receipt.get("privileged_evaluation_authorized") is True, "source scores are not authorized")
    controls = source_receipt.get("controls", {})
    require(controls.get("validation_target_received_or_opened") is False, "source process opened validation truth")
    require(controls.get("mask_received_or_opened") is False, "source process opened masks")
    require(controls.get("training_or_weight_update_performed") is False, "source score run changed weights")
    require(
        source_receipt.get("repository_head")
        == "c6802e2aa13da075c7ad969315461000393d860e",
        "source score lineage differs",
    )
    score_archive_path = Path(str(source_receipt["raw_arrays"]["absolute_path"]))
    require(file_record(score_archive_path) == source_receipt["raw_arrays"], "source score record differs")
    score_archive_record = file_record(score_archive_path)
    require(score_archive_record["sha256"] == EXPECTED_SOURCE_SCORE_SHA256, "source score SHA-256 differs")
    with np.load(score_archive_path, allow_pickle=False) as loaded:
        source_arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    frame_index, witness_id, final_scores = extract_final_feature_scores(source_arrays)

    smoke_count = 2
    smoke_start = time.perf_counter()
    smoke = decode_joint_assignments(final_scores[:smoke_count])
    smoke_seconds = time.perf_counter() - smoke_start
    require(
        smoke["assigned_local_3x3_prediction_px"].shape == (smoke_count, 10, 2),
        "smoke prediction shape differs",
    )
    projected_seconds = smoke_seconds / smoke_count * len(frame_index) * 2.0
    require(projected_seconds <= MAXIMUM_PROJECTED_SECONDS, "projected CPU runtime exceeds gate")

    start = time.perf_counter()
    forward = decode_joint_assignments(final_scores)
    reverse_native = decode_joint_assignments(final_scores[::-1])
    reverse_restored = {name: np.asarray(value)[::-1] for name, value in reverse_native.items()}
    runtime_seconds = time.perf_counter() - start
    replay_exact, replay_differences = _maximum_replay_difference(forward, reverse_restored)
    require(replay_exact, "forward/reverse frame-order replay differs")
    require("torch" not in sys.modules, "Torch entered the truth-blind process")

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_FINAL_FEATURE_JOINT_ASSIGNMENT_PREDICTIONS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        representation_name=np.asarray(FINAL_FEATURE_NAME),
        final_feature_query_score_map=final_scores,
        **forward,
    )
    arrays_record = file_record(arrays_path)
    receipt_path = args.output_dir / "RAW_FINAL_FEATURE_JOINT_ASSIGNMENT_RECEIPT.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "truth_blind_frozen_final_feature_joint_mode_assignment",
        "command_argv": list(sys.argv),
        "repository_head": repository_head,
        "semantic_lock": semantic_lock_record,
        "source_raw_receipt": source_receipt_record,
        "source_score_archive": score_archive_record,
        "raw_arrays": arrays_record,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "joint_assignment_contract": file_record(
                args.repo_root / "keypoint_net" / "final_feature_joint_assignment.py"
            ),
        },
        "information_boundary": compact_information_boundary(),
        "runtime": {
            "smoke_frame_count": smoke_count,
            "smoke_seconds": smoke_seconds,
            "projected_forward_reverse_seconds": projected_seconds,
            "maximum_authorized_seconds": MAXIMUM_PROJECTED_SECONDS,
            "actual_forward_reverse_seconds": runtime_seconds,
        },
        "candidate_mode_counts": {
            "minimum": int(forward["candidate_mode_count"].min()),
            "median": float(np.median(forward["candidate_mode_count"])),
            "maximum": int(forward["candidate_mode_count"].max()),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "execution_backend": "numpy_scipy_cpu",
            "torch_imported": False,
            "cuda_api_used": False,
            "mps_api_used": False,
            "laptop_gpu_used": False,
        },
        "controls": {
            "source_archive_loaded_without_privileged_inputs": True,
            "validation_target_received_or_opened": False,
            "mask_path_hash_or_pixels_received_or_opened": False,
            "rgb_received_or_opened": False,
            "prior_evaluation_classification_received_or_opened": False,
            "operator_or_tracker_input_received_or_opened": False,
            "previous_or_future_frame_received": False,
            "training_or_weight_update_performed": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "forward_reverse_frame_order_exact": replay_exact,
            "forward_reverse_maximum_absolute_difference": replay_differences,
        },
        "privileged_evaluation_authorized": replay_exact,
    }
    _write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--source-raw-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        receipt = run(parse_args())
    except Exception as exc:
        print(f"FINAL_FEATURE_JOINT_ASSIGNMENT_RAW_FAILED: {exc}", file=sys.stderr)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
