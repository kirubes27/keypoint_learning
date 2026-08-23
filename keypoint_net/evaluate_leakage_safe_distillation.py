"""Evaluate frozen raw detector coordinates only after hashing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    require,
)
from certified_witness_local_readout import (
    classify_localization_failures,
    readout_arrays,
)
from leakage_safe_distillation_contract import (
    HALF_CELL_DIAGONAL_PX,
    SEMANTIC_LOCK_SHA256,
    TRAIN_FRAMES,
    TWO_CELL_SPACING_PX,
    VALIDATION_FRAMES,
    compact_report,
    load_json,
    operational_near_pass,
    verify_clean_repository,
    verify_exact_file,
    verify_record,
    verify_semantic_lock,
    write_json,
)
from run_certified_witness_capability import _save_worst_montage


def _load_target_artifact(
    path: Path, expected_frames: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as arrays:
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        witness_id = np.asarray(arrays["witness_id"], dtype=np.int64)
        target_px = np.asarray(arrays["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(arrays["physical_valid"], dtype=bool)
        target_on_object = np.asarray(arrays["target_on_object"], dtype=bool)
    require(np.array_equal(frames, expected_frames), f"{label} frames differ")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, f"{label} witness order differs")
    require(target_px.shape == (len(expected_frames), EXPECTED_WITNESSES, 2), f"{label} target shape differs")
    require(bool(physical_valid.all()), f"{label} contains invalid target")
    require(bool(target_on_object.all()), f"{label} records off-object target")
    return frames, target_px


def _load_masks_and_validation_images(
    mask_manifest_path: Path,
    rgb_manifest_path: Path,
    object_root: Path,
    raw_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask_manifest = load_json(mask_manifest_path)
    require(
        mask_manifest.get("schema_version")
        == "leakage_safe_distillation_evaluation_mask_manifest.v1",
        "evaluation mask manifest schema differs",
    )
    mask_records = {
        int(record["frame_index"]): record for record in mask_manifest["frames"]
    }
    require(set(mask_records) == set(raw_frames.tolist()), "mask manifest frames differ")
    masks = np.empty((len(raw_frames), 512, 512), dtype=bool)
    for local_index, frame in enumerate(raw_frames.tolist()):
        record = mask_records[frame]
        path = object_root / str(record["mask_relpath"])
        verify_exact_file(path, str(record["mask_sha256"]), f"evaluation mask frame {frame}")
        mask = np.asarray(Image.open(path).convert("L")) > 0
        require(mask.shape == (512, 512), "evaluation mask shape differs")
        masks[local_index] = mask

    rgb_manifest = load_json(rgb_manifest_path)
    require(
        rgb_manifest.get("schema_version")
        == "leakage_safe_distillation_raw_rgb_manifest.v1",
        "evaluation RGB manifest schema differs",
    )
    rgb_records = {
        int(record["frame_index"]): record for record in rgb_manifest["frames"]
    }
    validation_images = np.empty((len(VALIDATION_FRAMES), 512, 512, 3), dtype=np.uint8)
    for local_index, frame in enumerate(VALIDATION_FRAMES.tolist()):
        record = rgb_records[frame]
        path = object_root / str(record["image_relpath"])
        verify_exact_file(path, str(record["image_sha256"]), f"evaluation RGB frame {frame}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        require(image.shape == (512, 512, 3), "evaluation RGB shape differs")
        validation_images[local_index] = image
    return masks, validation_images, {
        "mask_hashes_verified": len(raw_frames),
        "validation_rgb_hashes_verified_for_visuals": len(VALIDATION_FRAMES),
    }


def _partition_indices(raw_frames: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    lookup = {int(frame): index for index, frame in enumerate(raw_frames.tolist())}
    require(all(int(frame) in lookup for frame in wanted), "raw frames omit partition")
    return np.asarray([lookup[int(frame)] for frame in wanted], dtype=np.int64)


def _within_two_cell_report(
    prediction_px: np.ndarray, target_px: np.ndarray
) -> dict[str, Any]:
    error = np.linalg.norm(prediction_px - target_px, axis=-1)
    within = error <= TWO_CELL_SPACING_PX + 1e-12
    return {
        "threshold_px": TWO_CELL_SPACING_PX,
        "pass_count": int(within.sum()),
        "total_count": int(within.size),
        "rate": float(within.mean()),
        "all_pass": bool(within.all()),
        "per_witness_rate": [float(within[:, channel].mean()) for channel in range(EXPECTED_WITNESSES)],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")

    # This entire block completes before the privileged evaluation receipt is read.
    raw_receipt = load_json(args.raw_receipt)
    require(
        raw_receipt.get("schema_version")
        == "leakage_safe_distillation_raw_receipt.v1",
        "raw receipt schema differs",
    )
    require(raw_receipt.get("privileged_evaluation_authorized") is True, "raw receipt does not authorize evaluation")
    paired_arm = str(raw_receipt.get("paired_arm", ""))
    require(paired_arm in ("control", "candidate"), "raw paired arm differs")
    raw_arrays_path = verify_record(raw_receipt["raw_arrays"], "raw arrays")
    with np.load(raw_arrays_path) as loaded:
        raw = {name: np.asarray(loaded[name]) for name in loaded.files}
    raw_hashed_and_loaded_before_truth = True

    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)
    verify_semantic_lock(args.semantic_lock)
    require(
        raw_receipt.get("repository_head") == repository_head,
        "raw receipt repository differs",
    )
    require(
        raw_receipt.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        "raw receipt lock differs",
    )
    evaluation_input = load_json(args.evaluation_input_receipt)
    require(
        evaluation_input.get("schema_version")
        == "leakage_safe_distillation_evaluation_input_receipt.v1",
        "evaluation input receipt schema differs",
    )
    require(evaluation_input.get("frame_partitions_disjoint") is True, "evaluation split is not disjoint")
    require(
        evaluation_input.get("repository_head") == repository_head,
        "evaluation input repository differs",
    )
    semantic_lock_record = evaluation_input["semantic_lock"]
    require(semantic_lock_record["sha256"] == SEMANTIC_LOCK_SHA256, "evaluation lock binding differs")
    verify_record(semantic_lock_record, "evaluation semantic lock")
    train_targets_path = verify_record(evaluation_input["train_targets"], "train targets")
    validation_truth_path = verify_record(evaluation_input["validation_truth"], "validation truth")
    mask_manifest_path = verify_record(evaluation_input["evaluation_mask_manifest"], "evaluation masks")
    rgb_manifest_path = verify_record(evaluation_input["raw_rgb_manifest"], "evaluation RGB manifest")

    raw_frames = np.asarray(raw["frame_index"], dtype=np.int64)
    expected_raw_frames = np.concatenate((VALIDATION_FRAMES, TRAIN_FRAMES))
    require(np.array_equal(raw_frames, expected_raw_frames), "raw prediction frames differ")
    _, train_target_px = _load_target_artifact(
        train_targets_path, TRAIN_FRAMES, "train targets"
    )
    _, validation_target_px = _load_target_artifact(
        validation_truth_path, VALIDATION_FRAMES, "validation truth"
    )
    masks, validation_images, visual_controls = _load_masks_and_validation_images(
        mask_manifest_path, rgb_manifest_path, args.object_root, raw_frames
    )
    train_indices = _partition_indices(raw_frames, TRAIN_FRAMES)
    validation_indices = _partition_indices(raw_frames, VALIDATION_FRAMES)
    train_masks = masks[train_indices]
    validation_masks = masks[validation_indices]

    train_global_px = np.asarray(raw["global_soft_prediction_px"], dtype=np.float64)[train_indices]
    train_local_px = np.asarray(raw["local_3x3_prediction_px"], dtype=np.float64)[train_indices]
    validation_global_px = np.asarray(raw["global_soft_prediction_px"], dtype=np.float64)[validation_indices]
    validation_local_px = np.asarray(raw["local_3x3_prediction_px"], dtype=np.float64)[validation_indices]
    train_global_report, train_global_derived = evaluate_predictions(
        train_global_px, train_target_px, train_masks
    )
    train_local_report, train_local_derived = evaluate_predictions(
        train_local_px, train_target_px, train_masks
    )
    validation_global_report, validation_global_derived = evaluate_predictions(
        validation_global_px, validation_target_px, validation_masks
    )
    validation_local_report, validation_local_derived = evaluate_predictions(
        validation_local_px, validation_target_px, validation_masks
    )

    train_logits = np.asarray(raw["native_heatmap_logits"])[train_indices]
    train_diagnostic_readout = readout_arrays(train_logits, train_target_px)
    train_category, train_category_counts = classify_localization_failures(
        train_diagnostic_readout, train_local_derived["within_half_cell"]
    )
    train_coarse_mode_pass = bool(
        np.asarray(
            train_diagnostic_readout["target_cell_inside_local_window"], dtype=bool
        ).all()
    )
    validation_logits = np.asarray(raw["native_heatmap_logits"])[validation_indices]
    diagnostic_readout = readout_arrays(validation_logits, validation_target_px)
    category, category_counts = classify_localization_failures(
        diagnostic_readout, validation_local_derived["within_half_cell"]
    )
    validation_coarse_mode_pass = bool(
        np.asarray(
            diagnostic_readout["target_cell_inside_local_window"], dtype=bool
        ).all()
    )
    training_strict_pass = bool(train_local_report["strict_capability_pass"])
    validation_strict_pass = bool(validation_local_report["strict_capability_pass"])
    strict_pass = bool(
        training_strict_pass
        and validation_strict_pass
        and train_coarse_mode_pass
        and validation_coarse_mode_pass
    )
    near_pass = bool(
        operational_near_pass(train_local_report)
        and operational_near_pass(validation_local_report)
    )
    # The semantic lock also requires a human inspection of both worst-event
    # montages. Numeric eligibility alone therefore never authorizes an operator.
    operator_authorized = False
    branch = (
        "strict_numeric_pass_pending_visual_inspection"
        if strict_pass
        else "operational_near_pass_report_only_stop_before_operator"
        if near_pass
        else "reject_paired_detector_stop_before_operator"
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "DISTILLED_DETECTOR_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        raw_frame_index=raw_frames,
        validation_frame_index=VALIDATION_FRAMES,
        validation_target_coordinate_px=validation_target_px,
        validation_global_prediction_px=validation_global_px,
        validation_local_prediction_px=validation_local_px,
        validation_global_material_error_px=validation_global_derived["material_error_px"],
        validation_local_material_error_px=validation_local_derived["material_error_px"],
        validation_local_within_half_cell=validation_local_derived["within_half_cell"],
        validation_local_on_object=validation_local_derived["on_object"],
        validation_local_identity_correct=validation_local_derived["identity_correct"],
        validation_local_distinct_pair=validation_local_derived["distinct_pair"],
        validation_localization_category_code=category,
        train_localization_category_code=train_category,
        train_frame_index=TRAIN_FRAMES,
        train_target_coordinate_px=train_target_px,
        train_local_prediction_px=train_local_px,
        train_local_material_error_px=train_local_derived["material_error_px"],
    )
    global_visual = args.output_dir / "VALIDATION_WORST_GLOBAL_EVENTS.png"
    local_visual = args.output_dir / "VALIDATION_WORST_LOCAL_EVENTS.png"
    _save_worst_montage(
        validation_images,
        validation_global_px,
        validation_target_px,
        validation_global_derived["material_error_px"],
        global_visual,
    )
    _save_worst_montage(
        validation_images,
        validation_local_px,
        validation_target_px,
        validation_local_derived["material_error_px"],
        local_visual,
    )

    result_path = args.output_dir / "DISTILLED_DETECTOR_EVALUATION_RESULT.json"
    result = {
        "schema_version": "leakage_safe_distilled_detector_evaluation.v1",
        "artifact_type": "privileged_posthash_distilled_detector_evaluation",
        "paired_arm": paired_arm,
        "repository_head": repository_head,
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "raw_predictions_hashed_and_loaded_before_truth_or_masks": raw_hashed_and_loaded_before_truth,
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "contract": file_record(
                args.repo_root
                / "keypoint_net"
                / "leakage_safe_distillation_contract.py"
            ),
            "capability_contract": file_record(
                args.repo_root / "keypoint_net" / "certified_witness_capability.py"
            ),
        },
        "training_global_control": compact_report(train_global_report),
        "training_local_candidate": compact_report(train_local_report),
        "validation_global_control": compact_report(validation_global_report),
        "validation_local_candidate": compact_report(validation_local_report),
        "validation_local_within_two_cells": _within_two_cell_report(
            validation_local_px, validation_target_px
        ),
        "validation_localization_category_counts": category_counts,
        "training_localization_category_counts": train_category_counts,
        "coarse_mode_gate": {
            "training_all_target_cells_inside_hard_centered_3x3": train_coarse_mode_pass,
            "validation_all_target_cells_inside_hard_centered_3x3": validation_coarse_mode_pass,
        },
        "thresholds": {
            "strict_half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "operational_maximum_two_cell_spacing_px": TWO_CELL_SPACING_PX,
            "strict_requires_both_training_and_validation": True,
            "strict_requires_zero_localization_identity_collapse_and_off_object_violations": True,
            "strict_requires_all_target_cells_inside_hard_centered_3x3": True,
            "operational_requires_zero_identity_collapse_and_off_object_violations": True,
            "operational_requires_median_within_half_cell": True,
            "operational_requires_all_errors_within_two_cells": True,
        },
        "decision": {
            "strict_detector_pass": strict_pass,
            "training_strict_capability_pass": training_strict_pass,
            "validation_strict_capability_pass": validation_strict_pass,
            "operational_near_pass": near_pass,
            "numeric_operator_eligible_pending_visual_inspection": strict_pass,
            "visual_inspection_completed": False,
            "operator_authorized": operator_authorized,
            "operator_role": None,
            "branch": branch,
        },
        "controls": {
            **visual_controls,
            "train_and_validation_frame_sets_disjoint": True,
            "guard_frames_excluded": True,
            "training_or_weight_update_performed": False,
            "validation_truth_opened_only_after_raw_hash": True,
        },
        "statistical_scope": {
            "inference": "descriptive_only",
            "sample_unit": "fixed_witness_over_one_contiguous_24_frame_heldout_wedge",
            "validation_frame_count": len(VALIDATION_FRAMES),
            "witness_count": EXPECTED_WITNESSES,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
            "population_or_cross_object_generalization_authorized": False,
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "DISTILLED_DETECTOR_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "leakage_safe_distilled_detector_evaluation_receipt.v1",
        "paired_arm": paired_arm,
        "repository_head": repository_head,
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "raw_receipt": file_record(args.raw_receipt),
        "raw_predictions": file_record(raw_arrays_path),
        "evaluation_input_receipt": file_record(args.evaluation_input_receipt),
        "validation_global_visual": file_record(global_visual),
        "validation_local_visual": file_record(local_visual),
        "operator_authorized": operator_authorized,
        "numeric_operator_eligible_pending_visual_inspection": strict_pass,
        "branch": branch,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-input-receipt", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"DISTILLATION EVALUATION CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
