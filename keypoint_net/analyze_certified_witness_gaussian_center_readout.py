"""Evaluate the frozen known-sigma Gaussian-center readout on seed 42."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_certified_witness_readouts import _infer
from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    require,
    sha256_file,
)
from certified_witness_gaussian_center_readout import (
    compact_information_boundary,
    gaussian_center_readout_arrays,
)
from certified_witness_local_readout import (
    category_name,
    classify_localization_failures,
    readout_arrays,
)
from run_certified_witness_capability import _load_bound_inputs, _save_worst_montage


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_capability_pass": report["strict_capability_pass"],
        "violations": report["violations"],
        "material_error_px": report["material_error_px"],
        "within_half_cell_rate": report["within_half_cell_rate"],
        "on_object_rate": report["on_object_rate"],
        "identity_assignment_rate": report["identity_assignment_rate"],
        "minimum_predicted_pair_distance_px": report[
            "minimum_predicted_pair_distance_px"
        ],
        "minimum_predicted_to_physical_pair_ratio": report[
            "minimum_predicted_to_physical_pair_ratio"
        ],
    }


def _verify_repository(args: argparse.Namespace) -> str:
    head = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == args.expected_repo_head, "repository HEAD differs from command lock")
    status = subprocess.run(
        ["git", "-C", str(args.repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "repository is not clean")
    return head


def _verify_record(record: dict[str, Any], expected_path: Path, label: str) -> None:
    require(
        Path(str(record.get("absolute_path", ""))).resolve() == expected_path.resolve(),
        f"{label} path differs from receipt",
    )
    require(expected_path.is_file(), f"{label} missing")
    require(
        sha256_file(expected_path) == record.get("sha256"),
        f"{label} SHA-256 differs from receipt",
    )


def _event_list(
    baseline: dict[str, np.ndarray],
    gaussian: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
    target_px: np.ndarray,
    category: np.ndarray,
) -> list[dict[str, Any]]:
    residual = (
        np.logical_not(derived["within_half_cell"])
        | np.logical_not(derived["on_object"])
        | np.logical_not(derived["identity_correct"])
    )
    events: list[dict[str, Any]] = []
    for frame_value, channel_value in np.argwhere(residual):
        frame = int(frame_value)
        channel = int(channel_value)
        code = int(category[frame, channel])
        events.append(
            {
                "frame": frame,
                "channel": channel,
                "witness_id": int(EXPECTED_WITNESS_IDS[channel]),
                "localization_failed": bool(not derived["within_half_cell"][frame, channel]),
                "localization_category": category_name(code) if code else None,
                "off_object": bool(not derived["on_object"][frame, channel]),
                "wrong_identity": bool(not derived["identity_correct"][frame, channel]),
                "material_error_px": float(derived["material_error_px"][frame, channel]),
                "target_coordinate_px": target_px[frame, channel].tolist(),
                "gaussian_center_coordinate_px": gaussian["prediction_px"][frame, channel].tolist(),
                "local_expectation_coordinate_px": baseline["local_3x3_prediction_px"][frame, channel].tolist(),
                "hard_cell_xy": [
                    int(baseline["hard_cell_x"][frame, channel]),
                    int(baseline["hard_cell_y"][frame, channel]),
                ],
                "target_nearest_cell_xy": [
                    int(baseline["target_cell_x"][frame, channel]),
                    int(baseline["target_cell_y"][frame, channel]),
                ],
                "target_nearest_cell_rank": int(
                    baseline["target_nearest_cell_rank"][frame, channel]
                ),
                "target_cell_inside_local_window": bool(
                    baseline["target_cell_inside_local_window"][frame, channel]
                ),
                "raw_offset_grid": gaussian["raw_offset_grid"][frame, channel].tolist(),
                "clamped_offset_grid": gaussian["clamped_offset_grid"][frame, channel].tolist(),
                "clamp_applied": bool(gaussian["clamp_applied"][frame, channel]),
                "fit_residual_sum_squares": float(
                    gaussian["fit_residual_sum_squares"][frame, channel]
                ),
            }
        )
    return events


def _checkpoint_row(
    label: str,
    update: int,
    logits: np.ndarray,
    baseline: dict[str, np.ndarray],
    target_px: np.ndarray,
    masks: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    gaussian = gaussian_center_readout_arrays(logits)
    require(
        np.array_equal(gaussian["hard_cell_x"], baseline["hard_cell_x"])
        and np.array_equal(gaussian["hard_cell_y"], baseline["hard_cell_y"]),
        f"{label} hard center differs between readouts",
    )
    report, derived = evaluate_predictions(gaussian["prediction_px"], target_px, masks)
    category, category_counts = classify_localization_failures(
        baseline, derived["within_half_cell"]
    )
    localization_failure = np.logical_not(derived["within_half_cell"])
    per_witness = [
        {
            "channel": channel,
            "witness_id": int(EXPECTED_WITNESS_IDS[channel]),
            "localization_failure_count": int(localization_failure[:, channel].sum()),
            "off_object_count": int(np.logical_not(derived["on_object"][:, channel]).sum()),
            "wrong_identity_count": int(
                np.logical_not(derived["identity_correct"][:, channel]).sum()
            ),
        }
        for channel in range(EXPECTED_WITNESSES)
    ]
    row = {
        "label": label,
        "update": update,
        "report": _compact_report(report),
        "localization_category_counts": category_counts,
        "clamp_applied_count": int(gaussian["clamp_applied"].sum()),
        "per_witness": per_witness,
        "residual_events": _event_list(
            baseline, gaussian, derived, target_px, category
        ),
    }
    arrays = {
        "prediction_px": gaussian["prediction_px"],
        "raw_offset_grid": gaussian["raw_offset_grid"],
        "clamped_offset_grid": gaussian["clamped_offset_grid"],
        "clamp_applied": gaussian["clamp_applied"],
        "fit_residual_sum_squares": gaussian["fit_residual_sum_squares"],
        "within_half_cell": derived["within_half_cell"],
        "on_object": derived["on_object"],
        "identity_correct": derived["identity_correct"],
        "distinct_pair": derived["distinct_pair"],
        "material_error_px": derived["material_error_px"],
        "localization_category_code": category,
    }
    return row, arrays


def _has_new_escape(
    baseline_report: dict[str, Any], candidate_report: dict[str, Any]
) -> bool:
    baseline = baseline_report["violations"]
    candidate = candidate_report["violations"]
    return any(
        int(candidate[name]) > int(baseline[name])
        for name in (
            "wrong_identity_count",
            "collapsed_pair_count",
            "off_object_count",
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh path")
    repository_head = _verify_repository(args)
    require(
        sha256_file(args.analysis_lock) == args.expected_analysis_lock_sha256,
        "analysis-lock SHA-256 differs",
    )
    require(
        sha256_file(args.confirmation_receipt)
        == args.expected_confirmation_receipt_sha256,
        "confirmation receipt SHA-256 differs",
    )
    receipt = _load_json(args.confirmation_receipt)
    require(receipt.get("scientific_full_confirmation") is True, "confirmation was not full")
    require(receipt.get("strict_local_capability_pass") is False, "confirmation unexpectedly passed")
    _verify_record(receipt["selected_checkpoint"], args.primary_checkpoint, "primary checkpoint")
    _verify_record(receipt["predictions"], args.primary_predictions, "primary predictions")
    primary_payload = torch.load(
        args.primary_checkpoint, map_location="cpu", weights_only=False
    )
    require(
        primary_payload.get("schema_version")
        == "certified_witness_local_confirmation_checkpoint.v1",
        "primary checkpoint schema differs",
    )
    require(int(primary_payload.get("update", -1)) == 4200, "primary update differs")
    require(
        sha256_file(args.secondary_checkpoint) == args.expected_secondary_checkpoint_sha256,
        "secondary checkpoint SHA-256 differs",
    )
    require(
        sha256_file(args.secondary_predictions) == args.expected_secondary_predictions_sha256,
        "secondary saved predictions SHA-256 differs",
    )
    require(
        sha256_file(args.previous_readout_arrays)
        == args.expected_previous_readout_arrays_sha256,
        "previous readout arrays SHA-256 differs",
    )

    _, dataset, target_px, masks, _ = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        EXPECTED_FRAMES,
    )
    device = torch.device(args.device)
    start = time.perf_counter()

    with np.load(args.primary_predictions) as primary_saved:
        primary_logits = np.asarray(primary_saved["native_heatmap_logits"])
        primary_local_saved = np.asarray(
            primary_saved["local_3x3_prediction_px"], dtype=np.float64
        )
        primary_global_saved = np.asarray(
            primary_saved["global_soft_prediction_px"], dtype=np.float64
        )
    primary_baseline = readout_arrays(primary_logits, target_px)
    require(
        np.array_equal(primary_baseline["local_3x3_prediction_px"], primary_local_saved),
        "primary local-expectation replay differs",
    )
    primary_baseline_report, _ = evaluate_predictions(
        primary_local_saved, target_px, masks
    )

    secondary_payload, secondary_global, secondary_logits = _infer(
        args.secondary_checkpoint,
        args.expected_secondary_checkpoint_sha256,
        dataset,
        device,
        args.batch_size,
    )
    require(int(secondary_payload["update"]) == 4700, "secondary update differs")
    with np.load(args.secondary_predictions) as secondary_saved:
        require(
            np.array_equal(
                secondary_global,
                np.asarray(secondary_saved["prediction_coordinate_px"], dtype=np.float64),
            ),
            "secondary saved global replay differs",
        )
    secondary_baseline = readout_arrays(secondary_logits, target_px)
    with np.load(args.previous_readout_arrays) as previous:
        require(
            np.array_equal(
                np.asarray(previous["global_soft_prediction_px"])[0],
                secondary_global,
            ),
            "secondary previous global replay differs",
        )
        require(
            np.array_equal(
                np.asarray(previous["local_3x3_prediction_px"])[0],
                secondary_baseline["local_3x3_prediction_px"],
            ),
            "secondary previous local replay differs",
        )
    secondary_baseline_report, _ = evaluate_predictions(
        secondary_baseline["local_3x3_prediction_px"], target_px, masks
    )

    primary_row, primary_arrays = _checkpoint_row(
        "prospective_local_score_winner",
        4200,
        primary_logits,
        primary_baseline,
        target_px,
        masks,
    )
    secondary_row, secondary_arrays = _checkpoint_row(
        "best_balanced_observed_state",
        4700,
        secondary_logits,
        secondary_baseline,
        target_px,
        masks,
    )
    primary_candidate_report = primary_row["report"]
    secondary_candidate_report = secondary_row["report"]
    strict_any = bool(
        primary_candidate_report["strict_capability_pass"]
        or secondary_candidate_report["strict_capability_pass"]
    )
    wrong_coarse_count = sum(
        int(row["localization_category_counts"][name])
        for row in (primary_row, secondary_row)
        for name in (
            "wrong_coarse_mode_target_top10",
            "wrong_coarse_mode_target_below_top10",
        )
    )
    new_escape = _has_new_escape(primary_baseline_report, primary_candidate_report) or _has_new_escape(
        secondary_baseline_report, secondary_candidate_report
    )
    decision_branch = (
        "posthoc_strict_mechanism_pass_requires_fresh_confirmation"
        if strict_any
        else "wrong_coarse_residual_stop_subcell_work"
        if wrong_coarse_count
        else "reject_analytic_readout_new_escape"
        if new_escape
        else "analytic_readout_near_fail_design_one_learned_offset_head"
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "GAUSSIAN_CENTER_READOUT_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        target_coordinate_px=target_px,
        primary_native_heatmap_logits=primary_logits,
        primary_local_expectation_px=primary_local_saved,
        primary_global_soft_px=primary_global_saved,
        secondary_native_heatmap_logits=secondary_logits,
        secondary_local_expectation_px=secondary_baseline[
            "local_3x3_prediction_px"
        ],
        secondary_global_soft_px=secondary_global,
        **{f"primary_{name}": value for name, value in primary_arrays.items()},
        **{f"secondary_{name}": value for name, value in secondary_arrays.items()},
    )
    primary_visual = args.output_dir / "PRIMARY_GAUSSIAN_CENTER_WORST.png"
    secondary_visual = args.output_dir / "SECONDARY_GAUSSIAN_CENTER_WORST.png"
    _save_worst_montage(
        dataset.images,
        primary_arrays["prediction_px"],
        target_px,
        primary_arrays["material_error_px"],
        primary_visual,
    )
    _save_worst_montage(
        dataset.images,
        secondary_arrays["prediction_px"],
        target_px,
        secondary_arrays["material_error_px"],
        secondary_visual,
    )

    result = {
        "schema_version": "certified_witness_gaussian_center_readout_diagnostic.v1",
        "artifact_type": "source_bound_frozen_known_sigma_readout_diagnostic",
        "weights_optimized": False,
        "batchnorm_state_changed": False,
        "information_boundary": compact_information_boundary(),
        "primary_local_expectation_replay_exact": True,
        "secondary_global_and_local_replay_exact": True,
        "primary_baseline_local_expectation": _compact_report(primary_baseline_report),
        "secondary_baseline_local_expectation": _compact_report(
            secondary_baseline_report
        ),
        "checkpoint_rows": [primary_row, secondary_row],
        "posthoc_strict_pass_any_checkpoint": strict_any,
        "wrong_coarse_residual_count_both_checkpoints": wrong_coarse_count,
        "new_identity_collapse_or_offobject_escape": new_escape,
        "decision_branch": decision_branch,
        "fresh_confirmation_required_for_capability_claim": strict_any,
        "unsupervised_discovery_established": False,
        "runtime_seconds": time.perf_counter() - start,
        "statistical_scope": {
            "inference": "descriptive_only",
            "optimization_seed_count": 1,
            "checkpoint_count": 2,
            "object_count": 1,
            "orbit_count": 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    result_path = args.output_dir / "GAUSSIAN_CENTER_READOUT_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_gaussian_center_readout_config.v1",
        "repository_head": repository_head,
        "analysis_source": file_record(Path(__file__)),
        "readout_source": file_record(
            args.repo_root
            / "keypoint_net"
            / "certified_witness_gaussian_center_readout.py"
        ),
        "analysis_lock": file_record(args.analysis_lock),
        "confirmation_receipt": file_record(args.confirmation_receipt),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "primary_checkpoint": file_record(args.primary_checkpoint),
        "primary_predictions": file_record(args.primary_predictions),
        "secondary_checkpoint": file_record(args.secondary_checkpoint),
        "secondary_predictions": file_record(args.secondary_predictions),
        "previous_readout_arrays": file_record(args.previous_readout_arrays),
        "device": str(device),
        "batch_size": args.batch_size,
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt_result = {
        "schema_version": "certified_witness_gaussian_center_readout_receipt.v1",
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "config": file_record(config_path),
        "primary_visual": file_record(primary_visual),
        "secondary_visual": file_record(secondary_visual),
        "posthoc_strict_pass_any_checkpoint": strict_any,
        "decision_branch": decision_branch,
        "weights_optimized": False,
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt_result, indent=2, sort_keys=True) + "\n")
    return receipt_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--expected-analysis-lock-sha256", required=True)
    parser.add_argument("--confirmation-receipt", type=Path, required=True)
    parser.add_argument("--expected-confirmation-receipt-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--secondary-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-secondary-checkpoint-sha256", required=True)
    parser.add_argument("--secondary-predictions", type=Path, required=True)
    parser.add_argument("--expected-secondary-predictions-sha256", required=True)
    parser.add_argument("--previous-readout-arrays", type=Path, required=True)
    parser.add_argument("--expected-previous-readout-arrays-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"GAUSSIAN CENTER READOUT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
