"""Score prediction-only recursive RGB teacher tracks against fixed witnesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

try:
    from .evaluate_material_transport_continuous_query import MINIMUM_QUERY_RMS
    from .evaluate_material_transport_witness_distribution_replay import HALF_CELL_DIAGONAL_PX
    from .material_transport_gate_io import file_record, load_json, require, write_json
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_material_transport_continuous_query import MINIMUM_QUERY_RMS
    from evaluate_material_transport_witness_distribution_replay import HALF_CELL_DIAGONAL_PX
    from material_transport_gate_io import file_record, load_json, require, write_json


EXPECTED_AMENDMENT_SHA256 = "913398162f3795c062793f1423d548edcd7109b62d7d95417d1097ec02d41854"
EXPECTED_CAPABILITY_MANIFEST_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    capability_record = file_record(args.capability_manifest)
    require(capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256, "capability SHA-256 differs")
    capability = load_json(args.capability_manifest)
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == capability["portable_tracks"]["sha256"], "track SHA-256 differs")
    raw_receipt_record = file_record(args.raw_receipt)
    raw_receipt = load_json(args.raw_receipt)
    require(raw_receipt["schema_version"] == "raw_recursive_continuous_teacher_predictions.v1", "raw receipt schema differs")
    require(raw_receipt["sources"]["amendment"] == amendment_record, "raw amendment binding differs")
    raw_predictions_record = file_record(args.raw_predictions)
    require(raw_receipt["raw_predictions"] == raw_predictions_record, "raw prediction binding differs")
    require(raw_receipt["controls"]["non_frame_zero_truth_opened"] is False, "raw runner opened later truth")
    require(raw_receipt["controls"]["witness_order_reversal_exact_every_array"] is True, "witness-order control failed")

    with np.load(args.raw_predictions) as raw:
        witness_id = np.asarray(raw["witness_id"], dtype=np.int64)
        initial_coordinate = np.asarray(raw["initial_coordinate_px"], dtype=np.float64)
        forward_target_frame = np.asarray(raw["forward_target_frame_index"], dtype=np.int64)
        reverse_target_frame = np.asarray(raw["reverse_target_frame_index"], dtype=np.int64)
        forward_prediction = np.asarray(raw["forward_prediction_coordinate_px"], dtype=np.float64)
        reverse_prediction = np.asarray(raw["reverse_prediction_coordinate_px"], dtype=np.float64)
        forward_hard = np.asarray(raw["forward_hard_coordinate_px"], dtype=np.float64)
        reverse_hard = np.asarray(raw["reverse_hard_coordinate_px"], dtype=np.float64)
        forward_query_rms = np.asarray(raw["forward_query_rms"], dtype=np.float64)
        reverse_query_rms = np.asarray(raw["reverse_query_rms"], dtype=np.float64)
    require(forward_prediction.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "forward shape differs")
    require(reverse_prediction.shape == forward_prediction.shape, "reverse shape differs")
    require(np.array_equal(forward_target_frame, (np.arange(EXPECTED_FRAMES) + 1) % EXPECTED_FRAMES), "forward target order differs")
    require(np.array_equal(reverse_target_frame, (-np.arange(EXPECTED_FRAMES) - 1) % EXPECTED_FRAMES), "reverse target order differs")
    require(bool((forward_query_rms >= MINIMUM_QUERY_RMS).all()), "forward query became invalid")
    require(bool((reverse_query_rms >= MINIMUM_QUERY_RMS).all()), "reverse query became invalid")

    with np.load(args.tracks) as archive:
        truth_witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        truth = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(witness_id, truth_witness_id), "witness order differs")
    require(truth.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "truth shape differs")
    require(np.array_equal(initial_coordinate, truth[0]), "initial coordinate differs from frame-zero truth")
    forward_truth = truth[forward_target_frame]
    reverse_truth = truth[reverse_target_frame]
    forward_error = np.linalg.norm(forward_prediction - forward_truth, axis=-1)
    reverse_error = np.linalg.norm(reverse_prediction - reverse_truth, axis=-1)
    forward_hard_error = np.linalg.norm(forward_hard - forward_truth, axis=-1)
    reverse_hard_error = np.linalg.norm(reverse_hard - reverse_truth, axis=-1)
    forward_pass = (forward_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12) & (
        forward_hard_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12
    )
    reverse_pass = (reverse_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12) & (
        reverse_hard_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12
    )
    witness_pass = np.all(forward_pass, axis=0) & np.all(reverse_pass, axis=0)
    all_ten = bool(np.all(witness_pass))

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RECURSIVE_TEACHER_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        forward_target_frame_index=forward_target_frame,
        reverse_target_frame_index=reverse_target_frame,
        forward_truth_coordinate_px=forward_truth,
        reverse_truth_coordinate_px=reverse_truth,
        forward_prediction_error_px=forward_error,
        reverse_prediction_error_px=reverse_error,
        forward_hard_error_px=forward_hard_error,
        reverse_hard_error_px=reverse_hard_error,
        forward_pass=forward_pass,
        reverse_pass=reverse_pass,
        witness_pass=witness_pass,
    )
    witness_reports = []
    for witness, identifier in enumerate(witness_id):
        witness_reports.append(
            {
                "witness_id": int(identifier),
                "strict_forward_and_reverse_all_edges": bool(witness_pass[witness]),
                "forward_pass_edges": int(np.sum(forward_pass[:, witness])),
                "reverse_pass_edges": int(np.sum(reverse_pass[:, witness])),
                "forward_maximum_prediction_error_px": float(np.max(forward_error[:, witness])),
                "reverse_maximum_prediction_error_px": float(np.max(reverse_error[:, witness])),
                "forward_maximum_hard_error_px": float(np.max(forward_hard_error[:, witness])),
                "reverse_maximum_hard_error_px": float(np.max(reverse_hard_error[:, witness])),
                "forward_cyclic_return_error_px": float(forward_error[-1, witness]),
                "reverse_cyclic_return_error_px": float(reverse_error[-1, witness]),
            }
        )
    result = {
        "schema_version": "recursive_continuous_teacher_evaluation.v1",
        "artifact_type": "privileged_posthash_recursive_teacher_check",
        "decision": {
            "all_ten_forward_and_reverse_all_edges_pass": all_ten,
            "branch": "authorize_raw_only_seed_mining" if all_ten else "reject_greedy_recursive_continuous_zncc",
        },
        "thresholds": {
            "prediction_and_hard_mode_maximum_error_px": HALF_CELL_DIAGONAL_PX,
            "every_query_rms_at_least": MINIMUM_QUERY_RMS,
        },
        "aggregate": {
            "strict_witness_count": int(np.sum(witness_pass)),
            "witness_count": EXPECTED_WITNESSES,
            "forward_pass_cases": int(np.sum(forward_pass)),
            "reverse_pass_cases": int(np.sum(reverse_pass)),
            "case_count_per_direction": EXPECTED_FRAMES * EXPECTED_WITNESSES,
            "forward_maximum_prediction_error_px": float(np.max(forward_error)),
            "reverse_maximum_prediction_error_px": float(np.max(reverse_error)),
            "forward_maximum_hard_error_px": float(np.max(forward_hard_error)),
            "reverse_maximum_hard_error_px": float(np.max(reverse_hard_error)),
        },
        "witness_reports": witness_reports,
        "sources": {
            "amendment": amendment_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
            "raw_predictions": raw_predictions_record,
            "raw_receipt": raw_receipt_record,
        },
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "arrays": file_record(arrays_path),
        "raw_predictions_hashed_before_truth_open": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over 180 correlated cyclic edges; descriptive only",
    }
    result_path = args.output_dir / "RECURSIVE_TEACHER_EVALUATION_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
