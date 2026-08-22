"""Summarize the frozen three-seed certified-witness capability matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    evaluate_predictions,
    evaluation_score,
    file_record,
    require,
    sha256_file,
)
from run_certified_witness_capability import _load_bound_inputs, _summary


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _verify_record(record: dict[str, Any]) -> None:
    path = Path(record["absolute_path"])
    require(path.is_file(), f"receipt file missing: {path}")
    require(path.stat().st_size == int(record["size_bytes"]), f"receipt size differs: {path}")
    require(sha256_file(path) == record["sha256"], f"receipt SHA-256 differs: {path}")


def _load_run(
    seed: int,
    run_dir: Path,
    expected_result_sha256: str,
    expected_predictions_sha256: str,
    expected_receipt_sha256: str,
    target_px: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    result_path = run_dir / "CAPABILITY_RESULT.json"
    predictions_path = run_dir / "predictions.npz"
    receipt_path = run_dir / "RUN_RECEIPT.json"
    require(sha256_file(result_path) == expected_result_sha256, f"seed {seed} result SHA-256 differs")
    require(
        sha256_file(predictions_path) == expected_predictions_sha256,
        f"seed {seed} predictions SHA-256 differs",
    )
    require(sha256_file(receipt_path) == expected_receipt_sha256, f"seed {seed} receipt SHA-256 differs")
    result = _load_json(result_path)
    receipt = _load_json(receipt_path)
    require(int(result["seed"]) == seed, f"seed {seed} result seed differs")
    require(result["scientific_full_orbit_run"] is True, f"seed {seed} is not a full-orbit run")
    require(receipt["scientific_full_orbit_run"] is True, f"seed {seed} receipt is not full-orbit")
    require(receipt["result"]["sha256"] == expected_result_sha256, f"seed {seed} result receipt differs")
    require(
        receipt["predictions"]["sha256"] == expected_predictions_sha256,
        f"seed {seed} prediction receipt differs",
    )
    for key in (
        "result",
        "predictions",
        "config",
        "semantic_controls",
        "best_model",
        "selected_checkpoint",
        "history",
        "worst_events_visual",
    ):
        _verify_record(receipt[key])
    with np.load(predictions_path) as arrays:
        frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
        prediction_px = np.asarray(arrays["prediction_coordinate_px"], dtype=np.float64)
        saved_target_px = np.asarray(arrays["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), f"seed {seed} frame order differs")
    require(np.array_equal(saved_target_px, target_px), f"seed {seed} target coordinates differ")
    require(
        prediction_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2),
        f"seed {seed} prediction shape differs",
    )
    replay_report, replay_derived = evaluate_predictions(prediction_px, target_px, masks)
    require(
        evaluation_score(replay_report) == evaluation_score(result["evaluation"]),
        f"seed {seed} evaluation replay differs",
    )
    require(
        replay_report["strict_capability_pass"] == result["strict_capability_pass"],
        f"seed {seed} strict result replay differs",
    )
    history_path = Path(receipt["history"]["absolute_path"])
    with history_path.open(newline="") as handle:
        history = list(csv.DictReader(handle))
    require(bool(history), f"seed {seed} history is empty")
    outside_history = np.asarray([int(row["outside_half_cell_count"]) for row in history], dtype=np.int64)
    on_object_history = np.asarray([float(row["on_object_rate"]) for row in history], dtype=np.float64)
    return {
        "seed": seed,
        "run_dir": run_dir,
        "result": result,
        "receipt": receipt,
        "prediction_px": prediction_px,
        "report": replay_report,
        "derived": replay_derived,
        "history_diagnostic": {
            "evaluation_count": len(history),
            "outside_half_cell_count_minimum": int(outside_history.min()),
            "outside_half_cell_count_maximum": int(outside_history.max()),
            "maximum_adjacent_outside_count_change": int(np.abs(np.diff(outside_history)).max()),
            "on_object_rate_minimum": float(on_object_history.min()),
            "on_object_rate_maximum": float(on_object_history.max()),
            "final_outside_half_cell_count": int(outside_history[-1]),
        },
    }


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    require(int(union) > 0, "empty failure-set union")
    return float(np.logical_and(left, right).sum() / union)


def _save_failure_heatmap(path: Path, failure_count: np.ndarray) -> None:
    require(failure_count.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES), "failure heatmap shape differs")
    cell_width = 4
    cell_height = 20
    left = 72
    top = 44
    right = 20
    bottom = 44
    width = left + EXPECTED_FRAMES * cell_width + right
    height = top + EXPECTED_WITNESSES * cell_height + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {
        0: "#1b9e77",
        1: "#f1c40f",
        2: "#e67e22",
        3: "#c0392b",
    }
    draw.text((left, 8), "Number of seeds failing the half-cell localization gate", fill="black", font=font)
    for witness_index, witness_id in enumerate(EXPECTED_WITNESS_IDS):
        y0 = top + witness_index * cell_height
        draw.text((4, y0 + 5), str(witness_id), fill="black", font=font)
        for frame in range(EXPECTED_FRAMES):
            x0 = left + frame * cell_width
            value = int(failure_count[frame, witness_index])
            draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 2), fill=colors[value])
    for frame in (0, 30, 60, 90, 120, 150, 179):
        x = left + frame * cell_width
        draw.line((x, top - 4, x, top + EXPECTED_WITNESSES * cell_height), fill="#333333", width=1)
        draw.text((x - 6, top + EXPECTED_WITNESSES * cell_height + 4), str(frame), fill="black", font=font)
    legend_x = left
    legend_y = height - 18
    for value in range(4):
        x = legend_x + value * 92
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=colors[value])
        draw.text((x + 16, legend_y + 1), f"{value} seed{'s' if value != 1 else ''}", fill="black", font=font)
    canvas.save(path)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_capability_pass": report["strict_capability_pass"],
        "violations": report["violations"],
        "material_error_px": report["material_error_px"],
        "within_half_cell_rate": report["within_half_cell_rate"],
        "on_object_rate": report["on_object_rate"],
        "identity_assignment_rate": report["identity_assignment_rate"],
        "minimum_predicted_pair_distance_px": report["minimum_predicted_pair_distance_px"],
        "minimum_predicted_to_physical_pair_ratio": report["minimum_predicted_to_physical_pair_ratio"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh summary path")
    repository_head = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(repository_head == args.expected_repo_head, "repository HEAD differs from command lock")
    require(sha256_file(args.analysis_lock) == args.expected_analysis_lock_sha256, "analysis-lock SHA-256 differs")
    _, _, target_px, masks, controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        EXPECTED_FRAMES,
    )
    run_specs = (
        (42, args.seed42_dir, args.seed42_result_sha256, args.seed42_predictions_sha256, args.seed42_receipt_sha256),
        (43, args.seed43_dir, args.seed43_result_sha256, args.seed43_predictions_sha256, args.seed43_receipt_sha256),
        (44, args.seed44_dir, args.seed44_result_sha256, args.seed44_predictions_sha256, args.seed44_receipt_sha256),
    )
    start = time.perf_counter()
    runs = [
        _load_run(seed, directory, result_sha, predictions_sha, receipt_sha, target_px, masks)
        for seed, directory, result_sha, predictions_sha, receipt_sha in run_specs
    ]
    require([run["seed"] for run in runs] == [42, 43, 44], "seed matrix order differs")
    predictions = np.stack([run["prediction_px"] for run in runs], axis=0)
    within = np.stack([run["derived"]["within_half_cell"] for run in runs], axis=0)
    failure = np.logical_not(within)
    failure_count = failure.sum(axis=0).astype(np.int8)
    all_three_fail = failure_count == 3
    any_seed_passes = failure_count < 3
    best_single_failure_count = min(int(run["report"]["violations"]["outside_half_cell_count"]) for run in runs)
    all_three_fail_count = int(all_three_fail.sum())
    intersection_over_best = float(all_three_fail_count / best_single_failure_count)

    pairwise_jaccard: dict[str, float] = {}
    for left_index in range(len(runs)):
        for right_index in range(left_index + 1, len(runs)):
            key = f"seed{runs[left_index]['seed']}_seed{runs[right_index]['seed']}"
            pairwise_jaccard[key] = _jaccard(failure[left_index], failure[right_index])

    mean_prediction = predictions.mean(axis=0)
    median_prediction = np.median(predictions, axis=0)
    mean_report, mean_derived = evaluate_predictions(mean_prediction, target_px, masks)
    median_report, median_derived = evaluate_predictions(median_prediction, target_px, masks)
    seed_error = np.linalg.norm(predictions - target_px[None, ...], axis=-1)
    oracle_seed_index = np.argmin(seed_error, axis=0)
    oracle_prediction = np.take_along_axis(
        predictions,
        oracle_seed_index[None, ..., None],
        axis=0,
    )[0]
    oracle_report, oracle_derived = evaluate_predictions(oracle_prediction, target_px, masks)
    ensemble_reports = {
        "coordinate_mean": mean_report,
        "coordinate_median": median_report,
    }
    best_ensemble_name = min(ensemble_reports, key=lambda name: evaluation_score(ensemble_reports[name]))
    best_ensemble_outside = int(ensemble_reports[best_ensemble_name]["violations"]["outside_half_cell_count"])
    ensemble_improvement_fraction = float((best_single_failure_count - best_ensemble_outside) / best_single_failure_count)
    if intersection_over_best <= 0.25 and ensemble_improvement_fraction >= 0.25:
        mechanism_triage = "complementary_stability_dominant"
        next_branch = "test_one_batch_independent_normalization_change"
    elif intersection_over_best >= 0.50 and best_ensemble_outside >= best_single_failure_count:
        mechanism_triage = "systematic_spatial_representation_dominant"
        next_branch = "inspect_prior_spatial_head_evidence_then_test_one_representation_change"
    else:
        mechanism_triage = "mixed"
        next_branch = "inspect_persistent_regions_and_prior_head_evidence_then_choose_one_change"

    per_witness_overlap = []
    for witness_index, witness_id in enumerate(EXPECTED_WITNESS_IDS):
        per_witness_overlap.append(
            {
                "witness_id": int(witness_id),
                "seed_failure_counts": {
                    str(run["seed"]): int(failure[index, :, witness_index].sum())
                    for index, run in enumerate(runs)
                },
                "all_three_fail_count": int(all_three_fail[:, witness_index].sum()),
                "at_least_one_seed_pass_count": int(any_seed_passes[:, witness_index].sum()),
            }
        )

    seed_matrix = []
    for run in runs:
        result = run["result"]
        seed_matrix.append(
            {
                "seed": run["seed"],
                "strict_capability_pass": result["strict_capability_pass"],
                "best_update": result["best_update"],
                "completed_updates": result["completed_updates"],
                "runtime_seconds": result["runtime_seconds"],
                "evaluation": _compact_report(run["report"]),
                "history_diagnostic": run["history_diagnostic"],
                "result": run["receipt"]["result"],
                "predictions": run["receipt"]["predictions"],
                "best_model": run["receipt"]["best_model"],
                "worst_events_visual": run["receipt"]["worst_events_visual"],
            }
        )

    result = {
        "schema_version": "certified_witness_capability_three_seed_matrix.v1",
        "artifact_type": "source_bound_three_seed_descriptive_matrix",
        "strict_seed_pass_count": int(sum(bool(run["result"]["strict_capability_pass"]) for run in runs)),
        "seed_count": len(runs),
        "predeclared_capability_branch": "zero_of_three_strict_seed_passes",
        "seed_matrix": seed_matrix,
        "failure_overlap": {
            "best_single_seed_localization_failure_count": best_single_failure_count,
            "all_three_seed_localization_failure_count": all_three_fail_count,
            "all_three_over_best_single_ratio": intersection_over_best,
            "at_least_one_seed_localization_pass_count": int(any_seed_passes.sum()),
            "failure_count_histogram": {
                str(value): int((failure_count == value).sum()) for value in range(4)
            },
            "pairwise_failure_set_jaccard": pairwise_jaccard,
            "per_witness": per_witness_overlap,
        },
        "label_free_coordinate_ensembles": {
            name: _compact_report(report) for name, report in ensemble_reports.items()
        },
        "best_label_free_ensemble": {
            "name": best_ensemble_name,
            "outside_half_cell_count": best_ensemble_outside,
            "improvement_fraction_vs_best_single_seed": ensemble_improvement_fraction,
        },
        "target_leaking_best_of_three_oracle": {
            "deployable": False,
            "interpretation": "upper-bound diagnostic selected with ground-truth target error",
            "report": _compact_report(oracle_report),
            "seed_selection_histogram": {
                str(run["seed"]): int((oracle_seed_index == index).sum())
                for index, run in enumerate(runs)
            },
        },
        "mechanism_triage": mechanism_triage,
        "next_experiment_branch": next_branch,
        "preservation_phase_authorized": False,
        "semantic_controls": {
            "source_bound_inputs_replayed": True,
            "identical_target_coordinates_across_seeds": True,
            "identical_frame_order_across_seeds": True,
            "three_seed_evaluations_replayed": True,
            "planted_evaluator_control": controls["planted_bilinear_logit_softargmax_evaluator_control"],
        },
        "statistical_scope": {
            "inference": "descriptive_only",
            "object_count": 1,
            "orbit_count": 1,
            "optimization_seed_count": 3,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
        "runtime_seconds": time.perf_counter() - start,
    }

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "THREE_SEED_ANALYSIS_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        seed=np.asarray([42, 43, 44], dtype=np.int64),
        prediction_coordinate_px=predictions,
        target_coordinate_px=target_px,
        within_half_cell=within,
        localization_failure_count=failure_count,
        all_three_localization_fail=all_three_fail,
        coordinate_mean_prediction_px=mean_prediction,
        coordinate_median_prediction_px=median_prediction,
        target_leaking_oracle_seed_index=oracle_seed_index,
        target_leaking_oracle_prediction_px=oracle_prediction,
        coordinate_mean_material_error_px=mean_derived["material_error_px"],
        coordinate_median_material_error_px=median_derived["material_error_px"],
        target_leaking_oracle_material_error_px=oracle_derived["material_error_px"],
    )
    heatmap_path = args.output_dir / "THREE_SEED_LOCALIZATION_FAILURE_MAP.png"
    _save_failure_heatmap(heatmap_path, failure_count)
    result_path = args.output_dir / "THREE_SEED_CAPABILITY_MATRIX_SUMMARY.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_capability_three_seed_matrix_config.v1",
        "diagnostic_implementation_head": repository_head,
        "diagnostic_source": file_record(Path(__file__).resolve()),
        "analysis_lock": file_record(args.analysis_lock),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "run_receipts": {str(run["seed"]): file_record(run["run_dir"] / "RUN_RECEIPT.json") for run in runs},
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_capability_three_seed_matrix_receipt.v1",
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "visual": file_record(heatmap_path),
        "config": file_record(config_path),
        "strict_seed_pass_count": result["strict_seed_pass_count"],
        "mechanism_triage": mechanism_triage,
        "next_experiment_branch": next_branch,
        "preservation_phase_authorized": False,
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--expected-analysis-lock-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    for seed in (42, 43, 44):
        parser.add_argument(f"--seed{seed}-dir", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-result-sha256", required=True)
        parser.add_argument(f"--seed{seed}-predictions-sha256", required=True)
        parser.add_argument(f"--seed{seed}-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"THREE-SEED MATRIX CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
