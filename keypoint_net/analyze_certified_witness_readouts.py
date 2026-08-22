"""Compare frozen global, hard, and local readouts on certified checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    FEATURE_SIZE,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    model_state_sha256,
    normalized_to_pixel,
    require,
    sha256_file,
)
from model import KeypointExtractor
from run_certified_witness_capability import _load_bound_inputs, _summary


def _new_model(state_dict: dict[str, torch.Tensor], device: torch.device) -> KeypointExtractor:
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def _grid_to_pixel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    normalized = np.stack(
        [
            -1.0 + 2.0 * np.asarray(x, dtype=np.float64) / (FEATURE_SIZE - 1),
            -1.0 + 2.0 * np.asarray(y, dtype=np.float64) / (FEATURE_SIZE - 1),
        ],
        axis=-1,
    )
    return normalized_to_pixel(normalized)


def _readout_arrays(logits: np.ndarray, target_px: np.ndarray) -> dict[str, np.ndarray]:
    require(
        logits.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "logit array shape differs",
    )
    flat = logits.reshape(EXPECTED_FRAMES, EXPECTED_WITNESSES, -1).astype(np.float64)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    order = np.argpartition(probability, kth=-2, axis=-1)[..., -2:]
    top_two = np.take_along_axis(probability, order, axis=-1)
    top_two.sort(axis=-1)
    hard_index = np.argmax(flat, axis=-1)
    hard_y, hard_x = np.divmod(hard_index, FEATURE_SIZE)
    hard_prediction_px = _grid_to_pixel(hard_x, hard_y)

    local_prediction_px = np.empty_like(hard_prediction_px)
    grid_x = np.arange(FEATURE_SIZE, dtype=np.float64)
    grid_y = np.arange(FEATURE_SIZE, dtype=np.float64)
    for frame in range(EXPECTED_FRAMES):
        for witness in range(EXPECTED_WITNESSES):
            center_x = int(hard_x[frame, witness])
            center_y = int(hard_y[frame, witness])
            x0 = max(0, center_x - 1)
            x1 = min(FEATURE_SIZE, center_x + 2)
            y0 = max(0, center_y - 1)
            y1 = min(FEATURE_SIZE, center_y + 2)
            window = logits[frame, witness, y0:y1, x0:x1].astype(np.float64)
            window = np.exp(window - window.max())
            window /= window.sum()
            local_x = float((window * grid_x[x0:x1][None, :]).sum())
            local_y = float((window * grid_y[y0:y1][:, None]).sum())
            local_prediction_px[frame, witness] = _grid_to_pixel(local_x, local_y)

    target_x = np.rint(target_px[..., 0] / 511.0 * (FEATURE_SIZE - 1)).astype(np.int64)
    target_y = np.rint(target_px[..., 1] / 511.0 * (FEATURE_SIZE - 1)).astype(np.int64)
    target_cell_index = target_y * FEATURE_SIZE + target_x
    target_cell_logit = np.take_along_axis(flat, target_cell_index[..., None], axis=-1)[..., 0]
    target_cell_rank = 1 + (flat > target_cell_logit[..., None]).sum(axis=-1)
    entropy = -(probability * np.log(np.clip(probability, 1e-300, None))).sum(axis=-1)
    return {
        "hard_prediction_px": hard_prediction_px,
        "local_3x3_prediction_px": local_prediction_px,
        "target_nearest_cell_rank": target_cell_rank.astype(np.int64),
        "top1_probability": top_two[..., 1],
        "top2_probability": top_two[..., 0],
        "top1_top2_probability_margin": top_two[..., 1] - top_two[..., 0],
        "heatmap_entropy": entropy,
    }


@torch.no_grad()
def _infer(
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    dataset: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    require(sha256_file(checkpoint_path) == expected_checkpoint_sha256, "checkpoint SHA-256 differs")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require(
        checkpoint["schema_version"] == "certified_witness_supervised_capability_checkpoint.v1",
        "checkpoint schema differs",
    )
    model = _new_model(checkpoint["extractor_state_dict"], device)
    require(model_state_sha256(model) == checkpoint["model_state_sha256"], "model state hash differs")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    global_coordinates: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for batch in loader:
        flat_coordinates, batch_logits = model(batch["image"].to(device))
        global_coordinates.append(flat_coordinates.view(-1, EXPECTED_WITNESSES, 2).cpu().numpy())
        logits.append(batch_logits.cpu().numpy())
        frames.append(batch["frame"].numpy())
    frame_index = np.concatenate(frames)
    order = np.argsort(frame_index)
    require(np.array_equal(frame_index[order], np.arange(EXPECTED_FRAMES)), "inference frame order differs")
    global_prediction_px = normalized_to_pixel(np.concatenate(global_coordinates)[order])
    logit_array = np.concatenate(logits)[order]
    return checkpoint, global_prediction_px, logit_array


def _rank_summary(rank: np.ndarray) -> dict[str, Any]:
    values = np.asarray(rank, dtype=np.int64).reshape(-1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "maximum": int(values.max()),
        "top1_rate": float((values == 1).mean()),
        "top10_rate": float((values <= 10).mean()),
    }


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


def _save_comparison_map(path: Path, failure_counts: dict[str, np.ndarray]) -> None:
    cell_width = 4
    cell_height = 14
    panel_gap = 34
    left = 72
    top = 38
    bottom = 38
    panel_height = EXPECTED_WITNESSES * cell_height
    width = left + EXPECTED_FRAMES * cell_width + 20
    height = top + len(failure_counts) * panel_height + (len(failure_counts) - 1) * panel_gap + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {0: "#1b9e77", 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b"}
    draw.text((left, 7), "Seeds failing localization by readout (0 green, 3 red)", fill="black", font=font)
    for panel_index, (name, counts) in enumerate(failure_counts.items()):
        panel_top = top + panel_index * (panel_height + panel_gap)
        draw.text((left, panel_top - 15), name, fill="black", font=font)
        for witness_index, witness_id in enumerate(EXPECTED_WITNESS_IDS):
            y0 = panel_top + witness_index * cell_height
            draw.text((4, y0 + 2), str(witness_id), fill="black", font=font)
            for frame in range(EXPECTED_FRAMES):
                x0 = left + frame * cell_width
                draw.rectangle(
                    (x0, y0, x0 + cell_width - 1, y0 + cell_height - 2),
                    fill=colors[int(counts[frame, witness_index])],
                )
        for frame in (0, 30, 60, 90, 120, 150, 179):
            x = left + frame * cell_width
            draw.line((x, panel_top - 3, x, panel_top + panel_height), fill="#333333", width=1)
            draw.text((x - 5, panel_top + panel_height + 3), str(frame), fill="black", font=font)
    canvas.save(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh diagnostic path")
    repository_head = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(repository_head == args.expected_repo_head, "repository HEAD differs from command lock")
    require(sha256_file(args.analysis_lock) == args.expected_analysis_lock_sha256, "analysis-lock SHA-256 differs")
    require(sha256_file(args.matrix_summary) == args.expected_matrix_summary_sha256, "matrix summary SHA-256 differs")
    require(sha256_file(args.matrix_arrays) == args.expected_matrix_arrays_sha256, "matrix arrays SHA-256 differs")
    _, dataset, target_px, masks, controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        EXPECTED_FRAMES,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    with np.load(args.matrix_arrays) as matrix_arrays:
        matrix_predictions = np.asarray(matrix_arrays["prediction_coordinate_px"], dtype=np.float64)
        persistent_global_failure = np.asarray(matrix_arrays["all_three_localization_fail"], dtype=bool)
    require(matrix_predictions.shape == (3, EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "matrix prediction shape differs")
    require(int(persistent_global_failure.sum()) == 123, "persistent global-failure count differs")

    run_specs = (
        (42, args.seed42_checkpoint, args.seed42_checkpoint_sha256, args.seed42_predictions, args.seed42_predictions_sha256),
        (43, args.seed43_checkpoint, args.seed43_checkpoint_sha256, args.seed43_predictions, args.seed43_predictions_sha256),
        (44, args.seed44_checkpoint, args.seed44_checkpoint_sha256, args.seed44_predictions, args.seed44_predictions_sha256),
    )
    start = time.perf_counter()
    seed_rows: list[dict[str, Any]] = []
    global_predictions: list[np.ndarray] = []
    hard_predictions: list[np.ndarray] = []
    local_predictions: list[np.ndarray] = []
    target_ranks: list[np.ndarray] = []
    top1_probability: list[np.ndarray] = []
    top2_probability: list[np.ndarray] = []
    top_margin: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    reports_by_mode: dict[str, list[dict[str, Any]]] = {
        "global_soft": [],
        "hard_argmax": [],
        "local_3x3": [],
    }
    failures_by_mode: dict[str, list[np.ndarray]] = {name: [] for name in reports_by_mode}
    for seed_index, (seed, checkpoint_path, checkpoint_sha, predictions_path, predictions_sha) in enumerate(run_specs):
        require(sha256_file(predictions_path) == predictions_sha, f"seed {seed} prediction SHA-256 differs")
        with np.load(predictions_path) as saved_arrays:
            saved_global_prediction = np.asarray(saved_arrays["prediction_coordinate_px"], dtype=np.float64)
        checkpoint, global_prediction, logits = _infer(
            checkpoint_path, checkpoint_sha, dataset, device, args.batch_size
        )
        require(
            np.array_equal(global_prediction, saved_global_prediction),
            f"seed {seed} global prediction does not replay saved array",
        )
        require(
            np.array_equal(global_prediction, matrix_predictions[seed_index]),
            f"seed {seed} global prediction does not replay matrix array",
        )
        readouts = _readout_arrays(logits, target_px)
        predictions_for_mode = {
            "global_soft": global_prediction,
            "hard_argmax": readouts["hard_prediction_px"],
            "local_3x3": readouts["local_3x3_prediction_px"],
        }
        seed_reports: dict[str, Any] = {}
        for mode, prediction in predictions_for_mode.items():
            report, derived = evaluate_predictions(prediction, target_px, masks)
            reports_by_mode[mode].append(report)
            failures_by_mode[mode].append(np.logical_not(derived["within_half_cell"]))
            seed_reports[mode] = _compact_report(report)
        soft_hard_distance = np.linalg.norm(global_prediction - readouts["hard_prediction_px"], axis=-1)
        persistent_rank = readouts["target_nearest_cell_rank"][persistent_global_failure]
        seed_rows.append(
            {
                "seed": seed,
                "selected_update": int(checkpoint["update"]),
                "reports": seed_reports,
                "global_soft_to_hard_distance_px": _summary(soft_hard_distance),
                "target_nearest_cell_rank_all": _rank_summary(readouts["target_nearest_cell_rank"]),
                "target_nearest_cell_rank_on_123_persistent_cases": _rank_summary(persistent_rank),
                "top1_probability": _summary(readouts["top1_probability"]),
                "top1_top2_probability_margin": _summary(readouts["top1_top2_probability_margin"]),
                "heatmap_entropy": _summary(readouts["heatmap_entropy"]),
                "checkpoint": file_record(checkpoint_path),
                "saved_predictions": file_record(predictions_path),
            }
        )
        global_predictions.append(global_prediction)
        hard_predictions.append(readouts["hard_prediction_px"])
        local_predictions.append(readouts["local_3x3_prediction_px"])
        target_ranks.append(readouts["target_nearest_cell_rank"])
        top1_probability.append(readouts["top1_probability"])
        top2_probability.append(readouts["top2_probability"])
        top_margin.append(readouts["top1_top2_probability_margin"])
        entropy.append(readouts["heatmap_entropy"])

    stacked_failures = {
        mode: np.stack(failures, axis=0) for mode, failures in failures_by_mode.items()
    }
    failure_counts = {mode: failures.sum(axis=0) for mode, failures in stacked_failures.items()}
    intersections = {
        mode: int((counts == 3).sum()) for mode, counts in failure_counts.items()
    }
    global_outside = [int(report["violations"]["outside_half_cell_count"]) for report in reports_by_mode["global_soft"]]
    per_seed_improvement: dict[str, list[float]] = {}
    readout_meets_seed_rule: dict[str, bool] = {}
    readout_meets_intersection_rule: dict[str, bool] = {}
    for mode in ("hard_argmax", "local_3x3"):
        improvements = [
            float((global_outside[index] - reports_by_mode[mode][index]["violations"]["outside_half_cell_count"]) / global_outside[index])
            for index in range(3)
        ]
        per_seed_improvement[mode] = improvements
        readout_meets_seed_rule[mode] = sum(value >= 0.25 for value in improvements) >= 2
        readout_meets_intersection_rule[mode] = intersections[mode] <= int(np.floor(0.75 * 123))
    passing_readouts = [
        mode
        for mode in ("hard_argmax", "local_3x3")
        if readout_meets_seed_rule[mode] and readout_meets_intersection_rule[mode]
    ]
    decision_branch = (
        "readout_dominant_test_one_differentiable_local_or_offset_readout"
        if passing_readouts
        else "response_mode_or_representation_dominant_choose_one_feature_optimization_intervention"
    )

    result = {
        "schema_version": "certified_witness_frozen_readout_diagnostic.v1",
        "artifact_type": "source_bound_frozen_checkpoint_readout_diagnostic",
        "weights_optimized": False,
        "batchnorm_state_changed": False,
        "global_prediction_replay_exact_all_seeds": True,
        "seed_rows": seed_rows,
        "all_three_seed_localization_failure_intersection": intersections,
        "per_seed_localization_failure_improvement_fraction_vs_global": per_seed_improvement,
        "readout_meets_two_of_three_25pct_rule": readout_meets_seed_rule,
        "readout_meets_persistent_intersection_25pct_rule": readout_meets_intersection_rule,
        "passing_readouts": passing_readouts,
        "decision_branch": decision_branch,
        "preservation_phase_authorized": False,
        "semantic_controls": {
            "source_bound_inputs_replayed": True,
            "global_soft_saved_predictions_replayed_exactly": True,
            "matrix_predictions_replayed_exactly": True,
            "persistent_global_failure_count_replayed": 123,
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
    arrays_path = args.output_dir / "FROZEN_READOUT_DIAGNOSTIC_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        seed=np.asarray([42, 43, 44], dtype=np.int64),
        target_coordinate_px=target_px,
        global_soft_prediction_px=np.stack(global_predictions),
        hard_argmax_prediction_px=np.stack(hard_predictions),
        local_3x3_prediction_px=np.stack(local_predictions),
        target_nearest_cell_rank=np.stack(target_ranks),
        top1_probability=np.stack(top1_probability),
        top2_probability=np.stack(top2_probability),
        top1_top2_probability_margin=np.stack(top_margin),
        heatmap_entropy=np.stack(entropy),
        global_soft_localization_failure_count=failure_counts["global_soft"],
        hard_argmax_localization_failure_count=failure_counts["hard_argmax"],
        local_3x3_localization_failure_count=failure_counts["local_3x3"],
        persistent_global_failure=persistent_global_failure,
    )
    visual_path = args.output_dir / "FROZEN_READOUT_FAILURE_COMPARISON.png"
    _save_comparison_map(visual_path, failure_counts)
    result_path = args.output_dir / "FROZEN_READOUT_DIAGNOSTIC_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_frozen_readout_diagnostic_config.v1",
        "diagnostic_implementation_head": repository_head,
        "diagnostic_source": file_record(Path(__file__).resolve()),
        "analysis_lock": file_record(args.analysis_lock),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "matrix_summary": file_record(args.matrix_summary),
        "matrix_arrays": file_record(args.matrix_arrays),
        "device": str(device),
        "batch_size": args.batch_size,
        "readouts": {
            "global_soft": "native global soft-argmax, temperature 1.0",
            "hard_argmax": "native r64 hard argmax",
            "local_3x3": "temperature-1.0 softmax restricted to 3x3 window around hard argmax",
        },
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_frozen_readout_diagnostic_receipt.v1",
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "visual": file_record(visual_path),
        "config": file_record(config_path),
        "decision_branch": decision_branch,
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
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--expected-matrix-summary-sha256", required=True)
    parser.add_argument("--matrix-arrays", type=Path, required=True)
    parser.add_argument("--expected-matrix-arrays-sha256", required=True)
    for seed in (42, 43, 44):
        parser.add_argument(f"--seed{seed}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-checkpoint-sha256", required=True)
        parser.add_argument(f"--seed{seed}-predictions", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-predictions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"FROZEN READOUT CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
