"""Evaluate frozen query-decoder predictions only after hashing raw output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from certified_witness_capability import (
    EXPECTED_WITNESS_IDS,
    HALF_CELL_DIAGONAL_PX,
    evaluate_predictions,
    file_record,
    require,
)
from certified_witness_local_readout import (
    LOCALIZATION_CATEGORY_NAMES,
    classify_localization_failures,
    readout_arrays,
)
from final_feature_query_decoder import (
    BASELINE_COLLAPSED_PAIR,
    BASELINE_MAXIMUM_ERROR_PX,
    BASELINE_OFF_OBJECT,
    BASELINE_WRONG_COARSE,
    BASELINE_WRONG_IDENTITY,
    EXPECTED_FRAMES,
    EXPECTED_SEMANTIC_LOCK_SHA256,
    FINAL_FEATURE_NAME,
    TWO_CELL_SPACING_PX,
    material_head_rescue_support,
    practical_complete_solution,
    select_decision_branch,
)


SCHEMA_VERSION = "final_feature_query_decoder_evaluation.v1"
EXPECTED_RAW_SCHEMA = "final_feature_query_decoder_raw_receipt.v1"
EXPECTED_BINDINGS = {
    "validation_truth": "3e188cd699cbddfe10ce51a3c1de97f7be95f958012219ed2699f4c0d565819b",
    "mask_manifest": "d4a02868b6bd645f106703b88af59513d0aa2eb8ba2ebef6eac64683af667b00",
    "rgb_manifest": "5fdad8b65a438cdc52d7f1b4080d772d5312bb3c01076a66715226fe2754c596",
    "baseline_arrays": "79a6a7f1b862df16dcd42442731e152988371fb598eac11a6114d3e4d0158e82",
    "baseline_result": "6904cd54b5d92d35080e6ac9b3ba6f13dd1bf49806eba778f7d5e9d294269968",
}


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


def _bound(path: Path, label: str) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == EXPECTED_BINDINGS[label], f"{label} SHA-256 differs")
    return record


def require_frame_coverage(
    record_frames: set[int], requested_frames: np.ndarray, label: str
) -> None:
    """Allow bound manifests to be supersets but never omit a requested frame."""

    requested = {int(frame) for frame in np.asarray(requested_frames).tolist()}
    missing = sorted(requested - {int(frame) for frame in record_frames})
    require(not missing, f"{label} omits requested frames: {missing}")


def _load_targets(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        frame_index = np.asarray(loaded["frame_index"], dtype=np.int64)
        witness_id = np.asarray(loaded["witness_id"], dtype=np.int64)
        target_px = np.asarray(loaded["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(loaded["physical_valid"], dtype=bool)
        target_on_object = np.asarray(loaded["target_on_object"], dtype=bool)
    require(np.array_equal(frame_index, EXPECTED_FRAMES), "validation frame order differs")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "validation witness order differs")
    require(target_px.shape == (24, 10, 2), "validation target shape differs")
    require(bool(physical_valid.all() and target_on_object.all()), "validation target validity differs")
    return frame_index, witness_id, target_px


def _load_masks_and_images(
    mask_manifest_path: Path,
    rgb_manifest_path: Path,
    object_root: Path,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    mask_manifest = _load_json(mask_manifest_path)
    require(
        mask_manifest.get("schema_version")
        == "leakage_safe_distillation_evaluation_mask_manifest.v1",
        "mask manifest schema differs",
    )
    mask_records = {int(row["frame_index"]): row for row in mask_manifest["frames"]}
    require_frame_coverage(set(mask_records), frames, "mask manifest")
    masks = np.empty((len(frames), 512, 512), dtype=bool)
    for local_index, frame in enumerate(frames.tolist()):
        row = mask_records[frame]
        path = object_root / str(row["mask_relpath"])
        require(file_record(path)["sha256"] == row["mask_sha256"], f"mask hash differs: {frame}")
        mask = np.asarray(Image.open(path).convert("L")) > 0
        require(mask.shape == (512, 512), "mask shape differs")
        masks[local_index] = mask

    rgb_manifest = _load_json(rgb_manifest_path)
    require(
        rgb_manifest.get("schema_version") == "leakage_safe_distillation_raw_rgb_manifest.v1",
        "RGB manifest schema differs",
    )
    rgb_records = {int(row["frame_index"]): row for row in rgb_manifest["frames"]}
    require_frame_coverage(set(rgb_records), frames, "RGB manifest")
    images = np.empty((len(frames), 512, 512, 3), dtype=np.uint8)
    for local_index, frame in enumerate(frames.tolist()):
        row = rgb_records[frame]
        path = object_root / str(row["image_relpath"])
        require(file_record(path)["sha256"] == row["image_sha256"], f"RGB hash differs: {frame}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        require(image.shape == (512, 512, 3), "RGB shape differs")
        images[local_index] = image
    return masks, images, {
        "mask_hashes_verified": len(frames),
        "rgb_hashes_verified_for_visuals": len(frames),
    }


def _change_counts(baseline_good: np.ndarray, candidate_good: np.ndarray) -> dict[str, int]:
    baseline = np.asarray(baseline_good, dtype=bool)
    candidate = np.asarray(candidate_good, dtype=bool)
    require(baseline.shape == candidate.shape, "comparison mask shape differs")
    return {
        "baseline_failure_count": int((~baseline).sum()),
        "candidate_failure_count": int((~candidate).sum()),
        "rescued_failure_count": int(((~baseline) & candidate).sum()),
        "new_regression_count": int((baseline & (~candidate)).sum()),
        "unchanged_failure_count": int(((~baseline) & (~candidate)).sum()),
    }


def _category_counts(category: np.ndarray) -> dict[str, int]:
    return {
        LOCALIZATION_CATEGORY_NAMES[code]: int((category == code).sum())
        for code in range(1, 5)
    }


def _summary_figure(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    baseline_wrong_coarse: int,
    candidate_wrong_coarse: int,
    output: Path,
) -> None:
    labels = ["outside\nhalf-cell", "wrong\nidentity", "collapsed\npairs", "off\nobject", "wrong\ncoarse"]
    baseline = [
        baseline_report["violations"]["outside_half_cell_count"],
        baseline_report["violations"]["wrong_identity_count"],
        baseline_report["violations"]["collapsed_pair_count"],
        baseline_report["violations"]["off_object_count"],
        baseline_wrong_coarse,
    ]
    candidate = [
        candidate_report["violations"]["outside_half_cell_count"],
        candidate_report["violations"]["wrong_identity_count"],
        candidate_report["violations"]["collapsed_pair_count"],
        candidate_report["violations"]["off_object_count"],
        candidate_wrong_coarse,
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    old = ax.bar(x - width / 2, baseline, width, label="static heatmap head")
    new = ax.bar(x + width / 2, candidate, width, label="final-feature query decoder")
    ax.bar_label(old, padding=3)
    ax.bar_label(new, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("correlated event count (lower is better)")
    ax.set_title("Gate 4b: same checkpoint and held-out wedge, head replaced only")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _worst_figure(
    images: np.ndarray,
    target_px: np.ndarray,
    baseline_px: np.ndarray,
    candidate_px: np.ndarray,
    score_maps: np.ndarray,
    candidate_error: np.ndarray,
    output: Path,
) -> list[dict[str, Any]]:
    flat_order = np.argsort(candidate_error.reshape(-1))[::-1][:8]
    events = [np.unravel_index(int(index), candidate_error.shape) for index in flat_order]
    fig, axes = plt.subplots(len(events), 2, figsize=(11, 3.1 * len(events)), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for row_index, (frame, witness) in enumerate(events):
        rgb_ax, score_ax = axes[row_index]
        rgb_ax.imshow(images[frame])
        rgb_ax.scatter(
            [target_px[frame, witness, 0]], [target_px[frame, witness, 1]],
            c="#00dd66", marker="+", s=120, linewidths=2.6, label="certified target",
        )
        rgb_ax.scatter(
            [baseline_px[frame, witness, 0]], [baseline_px[frame, witness, 1]],
            c="#ffaa22", marker="x", s=80, linewidths=2.0, label="static head",
        )
        rgb_ax.scatter(
            [candidate_px[frame, witness, 0]], [candidate_px[frame, witness, 1]],
            facecolors="none", edgecolors="#ff2255", marker="o", s=90, linewidths=2.0,
            label="query decoder",
        )
        rgb_ax.set_title(
            f"frame {frame}, KP{witness}: query error {candidate_error[frame, witness]:.1f}px"
        )
        rgb_ax.axis("off")
        if row_index == 0:
            rgb_ax.legend(loc="lower right", fontsize=7)

        score_ax.imshow(score_maps[frame, witness], cmap="magma")
        target_cell = target_px[frame, witness] / 511.0 * 63.0
        peak = np.unravel_index(np.argmax(score_maps[frame, witness]), (64, 64))
        score_ax.scatter([target_cell[0]], [target_cell[1]], c="#00dd66", marker="+", s=100, linewidths=2.4)
        score_ax.scatter([peak[1]], [peak[0]], c="#33bbff", marker="x", s=80, linewidths=2.0)
        score_ax.set_title("final-feature query score: target green, selected peak blue")
        score_ax.set_xticks([])
        score_ax.set_yticks([])
        records.append(
            {
                "frame_index": int(frame),
                "witness_index": int(witness),
                "candidate_material_error_px": float(candidate_error[frame, witness]),
            }
        )
    fig.suptitle("Worst final-feature query-decoder events (fixed 24-frame held-out wedge)", fontsize=14)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "evaluation output directory already exists")

    # This block finishes before any validation truth, mask, baseline result, or RGB is opened.
    raw_receipt = _load_json(args.raw_receipt)
    require(raw_receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw receipt schema differs")
    require(raw_receipt.get("privileged_evaluation_authorized") is True, "raw receipt blocks evaluation")
    raw_arrays_path = Path(str(raw_receipt["raw_arrays"]["absolute_path"]))
    require(file_record(raw_arrays_path) == raw_receipt["raw_arrays"], "raw-array binding differs")
    with np.load(raw_arrays_path, allow_pickle=False) as loaded:
        raw = {name: np.asarray(loaded[name]) for name in loaded.files}
    raw_loaded_before_privileged_inputs = True

    repository_head = _verify_clean_head(args.repo_root, args.expected_repo_head)
    raw_repository_head = str(raw_receipt.get("repository_head", ""))
    require(bool(raw_repository_head), "raw repository HEAD missing")
    ancestry = subprocess.run(
        ["git", "-C", str(args.repo_root), "merge-base", "--is-ancestor", raw_repository_head, repository_head],
        check=False,
    )
    require(ancestry.returncode == 0, "raw repository HEAD is not an evaluator ancestor")
    semantic_lock_record = file_record(args.semantic_lock)
    require(semantic_lock_record["sha256"] == EXPECTED_SEMANTIC_LOCK_SHA256, "semantic lock differs")
    require(raw_receipt.get("semantic_lock") == semantic_lock_record, "raw semantic-lock binding differs")
    for label, record in raw_receipt["implementation_sources"].items():
        require(file_record(Path(str(record["absolute_path"]))) == record, f"raw implementation differs: {label}")

    validation_truth_record = _bound(args.validation_truth, "validation_truth")
    mask_manifest_record = _bound(args.mask_manifest, "mask_manifest")
    rgb_manifest_record = _bound(args.rgb_manifest, "rgb_manifest")
    baseline_arrays_record = _bound(args.baseline_arrays, "baseline_arrays")
    baseline_result_record = _bound(args.baseline_result, "baseline_result")

    frame_index, witness_id, target_px = _load_targets(args.validation_truth)
    object_root = args.object_root.resolve(strict=True)
    masks, images, visual_controls = _load_masks_and_images(
        args.mask_manifest, args.rgb_manifest, object_root, frame_index
    )
    with np.load(args.baseline_arrays, allow_pickle=False) as loaded:
        baseline = {name: np.asarray(loaded[name]) for name in loaded.files}
    baseline_result = _load_json(args.baseline_result)

    require(np.array_equal(raw["frame_index"], frame_index), "raw frame order differs")
    require(tuple(raw["witness_id"].tolist()) == EXPECTED_WITNESS_IDS, "raw witness order differs")
    require(str(np.asarray(raw["representation_name"]).item()) == FINAL_FEATURE_NAME, "raw representation differs")
    score_maps = np.asarray(raw["final_feature_query_score_map"], dtype=np.float32)
    require(score_maps.shape == (24, 10, 64, 64), "raw score-map shape differs")
    prediction_px = np.asarray(raw["local_3x3_prediction_px"], dtype=np.float64)
    hard_prediction_px = np.asarray(raw["hard_prediction_px"], dtype=np.float64)
    replay = readout_arrays(score_maps)
    require(np.array_equal(prediction_px, replay["local_3x3_prediction_px"]), "raw local prediction replay differs")
    require(np.array_equal(hard_prediction_px, replay["hard_prediction_px"]), "raw hard prediction replay differs")

    require(np.array_equal(baseline["validation_frame_index"], frame_index), "baseline frames differ")
    require(np.array_equal(baseline["validation_target_coordinate_px"], target_px), "baseline targets differ")
    baseline_px = np.asarray(baseline["validation_local_prediction_px"], dtype=np.float64)
    baseline_report, baseline_derived = evaluate_predictions(baseline_px, target_px, masks)
    require(
        np.array_equal(baseline_derived["within_half_cell"], baseline["validation_local_within_half_cell"]),
        "baseline half-cell replay differs",
    )
    require(
        np.array_equal(baseline_derived["identity_correct"], baseline["validation_local_identity_correct"]),
        "baseline identity replay differs",
    )
    require(
        np.array_equal(baseline_derived["on_object"], baseline["validation_local_on_object"]),
        "baseline grounding replay differs",
    )
    require(baseline_report["violations"]["wrong_identity_count"] == BASELINE_WRONG_IDENTITY, "baseline wrong identity differs")
    require(baseline_report["violations"]["collapsed_pair_count"] == BASELINE_COLLAPSED_PAIR, "baseline collapsed pairs differ")
    require(baseline_report["violations"]["off_object_count"] == BASELINE_OFF_OBJECT, "baseline off-object differs")
    require(abs(baseline_report["material_error_px"]["maximum"] - BASELINE_MAXIMUM_ERROR_PX) <= 1e-12, "baseline maximum differs")
    require(
        baseline_result["validation_local_candidate"]["violations"] == baseline_report["violations"],
        "baseline result violation receipt differs",
    )
    baseline_category = np.asarray(baseline["validation_localization_category_code"], dtype=np.int8)
    baseline_wrong_coarse_mask = np.isin(baseline_category, (2, 3))
    require(int(baseline_wrong_coarse_mask.sum()) == BASELINE_WRONG_COARSE, "baseline wrong-coarse differs")

    candidate_report, candidate_derived = evaluate_predictions(prediction_px, target_px, masks)
    hard_report, hard_derived = evaluate_predictions(hard_prediction_px, target_px, masks)
    diagnostic = readout_arrays(score_maps, target_px)
    candidate_category, candidate_category_counts = classify_localization_failures(
        diagnostic, candidate_derived["within_half_cell"]
    )
    candidate_wrong_coarse_mask = np.isin(candidate_category, (2, 3))
    candidate_wrong_coarse_count = int(candidate_wrong_coarse_mask.sum())
    all_target_cells_inside = bool(np.asarray(diagnostic["target_cell_inside_local_window"]).all())
    strict_complete = bool(candidate_report["strict_capability_pass"] and all_target_cells_inside)
    practical_complete, practical_components = practical_complete_solution(
        candidate_report, prediction_px, target_px
    )
    head_rescue_supported, head_rescue_components = material_head_rescue_support(
        candidate_report, candidate_wrong_coarse_count
    )
    branch = select_decision_branch(
        strict_complete=strict_complete,
        practical_complete=practical_complete,
        head_rescue_supported=head_rescue_supported,
    )

    target_rank = np.asarray(diagnostic["target_nearest_cell_rank"], dtype=np.int64)
    residual_rank = target_rank[candidate_wrong_coarse_mask]
    residual_rank_strata = {
        "rank_1_to_3": int(np.sum(residual_rank <= 3)),
        "rank_4_to_10": int(np.sum((residual_rank >= 4) & (residual_rank <= 10))),
        "rank_above_10": int(np.sum(residual_rank > 10)),
    }
    within_two_baseline = baseline_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    within_two_candidate = candidate_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    comparisons = {
        "within_half_cell": _change_counts(
            baseline_derived["within_half_cell"], candidate_derived["within_half_cell"]
        ),
        "within_two_cells": _change_counts(within_two_baseline, within_two_candidate),
        "identity_correct": _change_counts(
            baseline_derived["identity_correct"], candidate_derived["identity_correct"]
        ),
        "on_object": _change_counts(baseline_derived["on_object"], candidate_derived["on_object"]),
        "distinct_pair": _change_counts(
            baseline_derived["distinct_pair"], candidate_derived["distinct_pair"]
        ),
        "coarse_mode_correct": _change_counts(
            ~baseline_wrong_coarse_mask, ~candidate_wrong_coarse_mask
        ),
    }

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "FINAL_FEATURE_QUERY_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        target_coordinate_px=target_px,
        baseline_prediction_px=baseline_px,
        query_prediction_px=prediction_px,
        query_hard_prediction_px=hard_prediction_px,
        baseline_material_error_px=baseline_derived["material_error_px"],
        query_material_error_px=candidate_derived["material_error_px"],
        baseline_within_half_cell=baseline_derived["within_half_cell"],
        query_within_half_cell=candidate_derived["within_half_cell"],
        baseline_on_object=baseline_derived["on_object"],
        query_on_object=candidate_derived["on_object"],
        baseline_identity_correct=baseline_derived["identity_correct"],
        query_identity_correct=candidate_derived["identity_correct"],
        baseline_distinct_pair=baseline_derived["distinct_pair"],
        query_distinct_pair=candidate_derived["distinct_pair"],
        baseline_wrong_coarse_event=baseline_wrong_coarse_mask,
        query_wrong_coarse_event=candidate_wrong_coarse_mask,
        query_localization_category_code=candidate_category,
        query_target_nearest_cell_rank=target_rank,
        query_target_cell_inside_local_window=diagnostic["target_cell_inside_local_window"],
    )
    summary_figure = args.output_dir / "01_BASELINE_VS_QUERY_COUNTS.png"
    worst_figure = args.output_dir / "02_QUERY_DECODER_WORST_EVENTS.png"
    _summary_figure(
        baseline_report,
        candidate_report,
        BASELINE_WRONG_COARSE,
        candidate_wrong_coarse_count,
        summary_figure,
    )
    worst_events = _worst_figure(
        images,
        target_px,
        baseline_px,
        prediction_px,
        score_maps,
        candidate_derived["material_error_px"],
        worst_figure,
    )

    result_path = args.output_dir / "FINAL_FEATURE_QUERY_RESULT.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "privileged_posthash_final_feature_query_decoder_evaluation",
        "repository_head": repository_head,
        "raw_repository_head": raw_repository_head,
        "raw_predictions_hashed_and_loaded_before_truth_masks_baseline_or_rgb": raw_loaded_before_privileged_inputs,
        "sample_scope": {
            "object_count": 1,
            "checkpoint_count": 1,
            "heldout_frame_count": 24,
            "witness_count": 10,
            "event_count": 240,
            "sample_unit": "one fixed witness channel on one held-out frame",
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
            "cross_object_generalization_authorized": False,
        },
        "baseline_static_heatmap_head": baseline_report,
        "candidate_final_feature_query_decoder": candidate_report,
        "concurrent_hard_grid_control": hard_report,
        "localization_category_counts": {
            "baseline_static_heatmap_head": _category_counts(baseline_category),
            "candidate_final_feature_query_decoder": candidate_category_counts,
        },
        "event_changes": comparisons,
        "residual_wrong_coarse_target_rank": residual_rank_strata,
        "decision": {
            "strict_complete_solution": strict_complete,
            "strict_all_target_cells_inside_hard_centered_3x3": all_target_cells_inside,
            "practical_complete_solution": practical_complete,
            "practical_components": practical_components,
            "material_head_rescue_supported": head_rescue_supported,
            "material_head_rescue_components": head_rescue_components,
            "branch": branch,
            "numeric_operator_eligible_pending_human_visual_inspection": strict_complete,
            "human_visual_inspection_completed_in_evaluator": False,
            "operator_authorized": False,
            "training_authorized": False,
        },
        "thresholds": {
            "strict_half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "practical_two_cell_spacing_px": TWO_CELL_SPACING_PX,
            "head_rescue_wrong_identity_maximum": 15,
            "head_rescue_wrong_coarse_maximum": 24,
            "head_rescue_collapsed_pair_maximum": BASELINE_COLLAPSED_PAIR,
            "head_rescue_off_object_maximum": BASELINE_OFF_OBJECT,
            "head_rescue_maximum_error_px": BASELINE_MAXIMUM_ERROR_PX,
        },
        "worst_visual_events": worst_events,
        "controls": {
            **visual_controls,
            "baseline_recomputed_under_same_targets_and_masks": True,
            "training_or_weight_update_performed": False,
            "laptop_gpu_used": False,
            "validation_truth_opened_only_after_raw_prediction_hash": True,
        },
        "bindings": {
            "semantic_lock": semantic_lock_record,
            "raw_receipt": file_record(args.raw_receipt),
            "raw_arrays": file_record(raw_arrays_path),
            "validation_truth": validation_truth_record,
            "mask_manifest": mask_manifest_record,
            "rgb_manifest": rgb_manifest_record,
            "baseline_arrays": baseline_arrays_record,
            "baseline_result": baseline_result_record,
            "derived_arrays": file_record(arrays_path),
            "summary_figure": file_record(summary_figure),
            "worst_figure": file_record(worst_figure),
        },
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "decoder_contract": file_record(args.repo_root / "keypoint_net" / "final_feature_query_decoder.py"),
            "local_readout": file_record(args.repo_root / "keypoint_net" / "certified_witness_local_readout.py"),
            "capability_contract": file_record(args.repo_root / "keypoint_net" / "certified_witness_capability.py"),
        },
    }
    _write_json(result_path, result)
    receipt_path = args.output_dir / "FINAL_FEATURE_QUERY_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "final_feature_query_decoder_evaluation_receipt.v1",
        "repository_head": repository_head,
        "raw_repository_head": raw_repository_head,
        "result": file_record(result_path),
        "derived_arrays": file_record(arrays_path),
        "summary_figure": file_record(summary_figure),
        "worst_figure": file_record(worst_figure),
        "raw_loaded_before_privileged_inputs": raw_loaded_before_privileged_inputs,
        "decision_branch": branch,
        "strict_complete_solution": strict_complete,
        "practical_complete_solution": practical_complete,
        "material_head_rescue_supported": head_rescue_supported,
        "operator_authorized": False,
        "training_authorized": False,
        "command_argv": list(sys.argv),
    }
    _write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--baseline-arrays", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        receipt = run(parse_args())
    except Exception as exc:
        print(f"FINAL_FEATURE_QUERY_EVALUATION_FAILED: {exc}", file=sys.stderr)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
