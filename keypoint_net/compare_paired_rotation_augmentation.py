"""Verify matched-arm lineage and compare the frozen detector outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from certified_witness_capability import (
    CapabilityContractError,
    file_record,
    require,
)
from leakage_safe_distillation_contract import (
    EXPECTED_INITIAL_MODEL_STATE_SHA256,
    SEMANTIC_LOCK_SHA256,
    load_json,
    verify_clean_repository,
    verify_exact_file,
    verify_record,
    verify_semantic_lock,
    write_json,
)


EXPECTED_HISTORICAL_CONTROL_STATE_SHA256 = (
    "cec041c285020006eb2c4890f75f824fea331a8d5375a1d72a72d00c96bbdebf"
)
EXPECTED_HISTORICAL_CONTROL_SELECTED_UPDATE = 3700
EXPECTED_PREVIOUS_CONTROL_EVALUATION_SHA256 = (
    "0ee5d54090a410b6a15998d8d629ba36756c58f4a150e5c44a804d13410c76e9"
)


def _load_training(
    receipt_path: Path, arm: str, repository_head: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path)
    require(
        receipt.get("schema_version")
        == "leakage_safe_witness_distillation_training_receipt.v1",
        f"{arm} training receipt schema differs",
    )
    require(receipt.get("paired_arm") == arm, f"{arm} training receipt arm differs")
    require(receipt.get("run_kind") == "full", f"{arm} training is not full")
    require(
        receipt.get("repository_head") == repository_head,
        f"{arm} training receipt repository differs",
    )
    require(
        receipt.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        f"{arm} training receipt lock differs",
    )
    result_path = verify_record(receipt["result"], f"{arm} training result")
    for key in (
        "config",
        "semantic_controls",
        "selected_model",
        "selected_checkpoint",
        "history",
        "training_predictions",
        "training_worst_visual",
    ):
        verify_record(receipt[key], f"{arm} training {key}")
    result = load_json(result_path)
    require(
        result.get("schema_version")
        == "leakage_safe_witness_distillation_training_result.v1",
        f"{arm} training result schema differs",
    )
    require(result.get("paired_arm") == arm, f"{arm} training result arm differs")
    require(
        result.get("repository_head") == repository_head,
        f"{arm} training result repository differs",
    )
    require(
        result.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        f"{arm} training result lock differs",
    )
    require(result.get("completed_updates") == 5000, f"{arm} update count differs")
    require(
        result.get("initial_model_state_sha256")
        == EXPECTED_INITIAL_MODEL_STATE_SHA256,
        f"{arm} initial model differs",
    )
    require(
        result.get("validation_truth_received_or_opened") is False,
        f"{arm} training opened validation truth",
    )
    return receipt, result


def _load_evaluation(
    receipt_path: Path, arm: str, repository_head: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path)
    require(
        receipt.get("schema_version")
        == "leakage_safe_distilled_detector_evaluation_receipt.v1",
        f"{arm} evaluation receipt schema differs",
    )
    require(receipt.get("paired_arm") == arm, f"{arm} evaluation receipt arm differs")
    require(
        receipt.get("repository_head") == repository_head,
        f"{arm} evaluation receipt repository differs",
    )
    require(
        receipt.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        f"{arm} evaluation receipt lock differs",
    )
    result_path = verify_record(receipt["result"], f"{arm} evaluation result")
    for key in (
        "arrays",
        "raw_receipt",
        "raw_predictions",
        "evaluation_input_receipt",
        "validation_global_visual",
        "validation_local_visual",
    ):
        verify_record(receipt[key], f"{arm} evaluation {key}")
    result = load_json(result_path)
    require(
        result.get("schema_version")
        == "leakage_safe_distilled_detector_evaluation.v1",
        f"{arm} evaluation result schema differs",
    )
    require(result.get("paired_arm") == arm, f"{arm} evaluation result arm differs")
    require(
        result.get("repository_head") == repository_head,
        f"{arm} evaluation result repository differs",
    )
    require(
        result.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        f"{arm} evaluation result lock differs",
    )
    require(
        result.get("raw_predictions_hashed_and_loaded_before_truth_or_masks") is True,
        f"{arm} raw-before-truth boundary differs",
    )
    return receipt, result


def _compact_validation(result: dict[str, Any]) -> dict[str, Any]:
    report = result["validation_local_candidate"]
    return {
        "strict_capability_pass": report["strict_capability_pass"],
        "violations": report["violations"],
        "median_error_px": report["material_error_px"]["median"],
        "q90_error_px": report["material_error_px"]["q90"],
        "maximum_error_px": report["material_error_px"]["maximum"],
        "on_object_rate": report["on_object_rate"],
        "identity_assignment_rate": report["identity_assignment_rate"],
        "coarse_mode_categories": result[
            "validation_localization_category_counts"
        ],
        "within_two_cells": result["validation_local_within_two_cells"],
        "numeric_strict_detector_pass": result["decision"][
            "strict_detector_pass"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)
    verify_semantic_lock(args.semantic_lock)
    control_receipt, control_training = _load_training(
        args.control_training_receipt, "control", repository_head
    )
    candidate_receipt, candidate_training = _load_training(
        args.candidate_training_receipt, "candidate", repository_head
    )
    control_eval_receipt, control_evaluation = _load_evaluation(
        args.control_evaluation_receipt, "control", repository_head
    )
    candidate_eval_receipt, candidate_evaluation = _load_evaluation(
        args.candidate_evaluation_receipt, "candidate", repository_head
    )
    verify_exact_file(
        args.previous_control_evaluation_result,
        EXPECTED_PREVIOUS_CONTROL_EVALUATION_SHA256,
        "historical control evaluation",
    )
    previous_control = load_json(args.previous_control_evaluation_result)

    control_schedule = control_training["paired_schedule"]
    candidate_schedule = candidate_training["paired_schedule"]
    matched_schedule_keys = (
        "sample_exposure_count",
        "selected_exposure_count",
        "sample_order_sha256",
        "proposed_angle_schedule_sha256",
        "selector_schedule_sha256",
    )
    matched_schedule = all(
        control_schedule[key] == candidate_schedule[key]
        for key in matched_schedule_keys
    )
    require(matched_schedule, "paired schedule or sample order differs")
    require(
        control_schedule["applied_rotation_count"] == 0,
        "control applied a rotation",
    )
    require(
        candidate_schedule["applied_rotation_count"] == 37500,
        "candidate rotation count differs",
    )

    historical_control_state_replay = bool(
        control_training["selected_model_state_sha256"]
        == EXPECTED_HISTORICAL_CONTROL_STATE_SHA256
        and int(control_training["selected_update"])
        == EXPECTED_HISTORICAL_CONTROL_SELECTED_UPDATE
    )
    require(historical_control_state_replay, "historical control checkpoint did not replay")
    replay_keys = (
        "training_global_control",
        "training_local_candidate",
        "validation_global_control",
        "validation_local_candidate",
        "validation_local_within_two_cells",
        "validation_localization_category_counts",
    )
    historical_control_metrics_replay = all(
        control_evaluation[key] == previous_control[key] for key in replay_keys
    )
    require(historical_control_metrics_replay, "historical control metrics did not replay")

    control_compact = _compact_validation(control_evaluation)
    candidate_compact = _compact_validation(candidate_evaluation)
    candidate_numeric_pass = bool(
        candidate_evaluation["decision"][
            "numeric_operator_eligible_pending_visual_inspection"
        ]
    )
    branch = (
        "candidate_numeric_strict_pass_pending_visual_inspection"
        if candidate_numeric_pass
        else "candidate_failed_stop_detector_branch_before_operator"
    )
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "PAIRED_ROTATION_COMPARISON_RESULT.json"
    result = {
        "schema_version": "paired_rotation_augmentation_comparison.v1",
        "artifact_type": "matched_seed42_control_candidate_comparison",
        "repository_head": repository_head,
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "implementation_source": file_record(Path(__file__)),
        "matched_schedule_keys": list(matched_schedule_keys),
        "matched_schedule_and_sample_order": matched_schedule,
        "historical_control_checkpoint_replay_exact": historical_control_state_replay,
        "historical_control_metrics_replay_exact": historical_control_metrics_replay,
        "control": control_compact,
        "candidate": candidate_compact,
        "descriptive_candidate_minus_control": {
            "median_error_px": float(candidate_compact["median_error_px"])
            - float(control_compact["median_error_px"]),
            "q90_error_px": float(candidate_compact["q90_error_px"])
            - float(control_compact["q90_error_px"]),
            "maximum_error_px": float(candidate_compact["maximum_error_px"])
            - float(control_compact["maximum_error_px"]),
            "wrong_identity_count": int(
                candidate_compact["violations"]["wrong_identity_count"]
            )
            - int(control_compact["violations"]["wrong_identity_count"]),
            "collapsed_pair_count": int(
                candidate_compact["violations"]["collapsed_pair_count"]
            )
            - int(control_compact["violations"]["collapsed_pair_count"]),
            "off_object_count": int(
                candidate_compact["violations"]["off_object_count"]
            )
            - int(control_compact["violations"]["off_object_count"]),
            "outside_half_cell_count": int(
                candidate_compact["violations"]["outside_half_cell_count"]
            )
            - int(control_compact["violations"]["outside_half_cell_count"]),
        },
        "decision": {
            "candidate_numeric_strict_pass": candidate_numeric_pass,
            "visual_inspection_required_before_operator": candidate_numeric_pass,
            "operator_authorized": False,
            "branch": branch,
        },
        "statistical_scope": {
            "inference": "descriptive_paired_same_seed_only",
            "optimization_seed_count_per_arm": 1,
            "object_count": 1,
            "validation_frame_count": 24,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
            "population_or_cross_object_generalization_authorized": False,
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "PAIRED_ROTATION_COMPARISON_RECEIPT.json"
    receipt = {
        "schema_version": "paired_rotation_augmentation_comparison_receipt.v1",
        "result": file_record(result_path),
        "semantic_lock": file_record(args.semantic_lock),
        "control_training_receipt": file_record(args.control_training_receipt),
        "candidate_training_receipt": file_record(args.candidate_training_receipt),
        "control_evaluation_receipt": file_record(args.control_evaluation_receipt),
        "candidate_evaluation_receipt": file_record(args.candidate_evaluation_receipt),
        "branch": branch,
        "operator_authorized": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--control-training-receipt", type=Path, required=True)
    parser.add_argument("--candidate-training-receipt", type=Path, required=True)
    parser.add_argument("--control-evaluation-receipt", type=Path, required=True)
    parser.add_argument("--candidate-evaluation-receipt", type=Path, required=True)
    parser.add_argument(
        "--previous-control-evaluation-result", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"PAIRED ROTATION COMPARISON FAILURE: {error}") from error


if __name__ == "__main__":
    main()
