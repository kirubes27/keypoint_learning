"""Evaluate Gate 4c only after hashing its truth-blind raw predictions."""

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

from final_feature_joint_assignment import (
    BASELINE_COLLAPSED_PAIR,
    BASELINE_MAXIMUM_ERROR_PX,
    BASELINE_OFF_OBJECT,
    BASELINE_WRONG_COARSE,
    BASELINE_WRONG_IDENTITY,
    EXPECTED_FRAMES,
    EXPECTED_GATE4B_RAW_SHA256,
    EXPECTED_SEMANTIC_LOCK_SHA256,
    EXPECTED_WITNESS_IDS,
    EXPECTED_WITNESSES,
    FINAL_FEATURE_NAME,
    HALF_CELL_DIAGONAL_PX,
    TWO_CELL_SPACING_PX,
    bilinear_sample_all_sites,
    decode_joint_assignments,
    evaluate_predictions,
    file_record,
    require,
    shared_true_site_cells,
    square_assignment_diagnostics,
    target_rank_and_category,
)


SCHEMA_VERSION = "final_feature_joint_assignment_evaluation.v1"
EXPECTED_RAW_SCHEMA = "final_feature_joint_assignment_raw_receipt.v1"
EXPECTED_BINDINGS = {
    "validation_truth": "3e188cd699cbddfe10ce51a3c1de97f7be95f958012219ed2699f4c0d565819b",
    "mask_manifest": "d4a02868b6bd645f106703b88af59513d0aa2eb8ba2ebef6eac64683af667b00",
    "rgb_manifest": "5fdad8b65a438cdc52d7f1b4080d772d5312bb3c01076a66715226fe2754c596",
    "baseline_arrays": "79a6a7f1b862df16dcd42442731e152988371fb598eac11a6114d3e4d0158e82",
    "baseline_result": "6904cd54b5d92d35080e6ac9b3ba6f13dd1bf49806eba778f7d5e9d294269968",
}
LOCALIZATION_CATEGORY_NAMES = {
    1: "border_window_truncation",
    2: "wrong_coarse_mode_target_top10",
    3: "wrong_coarse_mode_target_below_top10",
    4: "local_offset_failure",
}
GATE4B_EXPECTED_VIOLATIONS = {
    "outside_half_cell_count": 112,
    "off_object_count": 37,
    "wrong_identity_count": 32,
    "collapsed_pair_count": 26,
}
GATE4B_EXPECTED_MAXIMUM_ERROR_PX = 170.23608321604635


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


def _require_frame_coverage(record_frames: set[int], frames: np.ndarray, label: str) -> None:
    requested = {int(frame) for frame in np.asarray(frames).tolist()}
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
    require(target_px.shape == (24, EXPECTED_WITNESSES, 2), "validation target shape differs")
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
    _require_frame_coverage(set(mask_records), frames, "mask manifest")
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
    _require_frame_coverage(set(rgb_records), frames, "RGB manifest")
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


def _change_counts(baseline_good: Any, candidate_good: Any) -> dict[str, int]:
    baseline = np.asarray(baseline_good, dtype=bool)
    candidate = np.asarray(candidate_good, dtype=bool)
    require(baseline.shape == candidate.shape, "comparison mask shape differs")
    return {
        "baseline_failure_count": int((~baseline).sum()),
        "candidate_failure_count": int((~candidate).sum()),
        "rescued_failure_count": int(((~baseline) & candidate).sum()),
        "new_regression_count": int((baseline & ~candidate).sum()),
        "unchanged_failure_count": int(((~baseline) & ~candidate).sum()),
    }


def _category_counts(category: Any) -> dict[str, int]:
    values = np.asarray(category, dtype=np.int8)
    return {
        name: int((values == code).sum())
        for code, name in LOCALIZATION_CATEGORY_NAMES.items()
    }


def _numeric_summary(values: Any) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "numeric summary invalid")
    return {
        "n": int(vector.size),
        "minimum": float(vector.min()),
        "median": float(np.median(vector)),
        "mean": float(vector.mean()),
        "maximum": float(vector.max()),
    }


def _summary_figure(
    static_report: dict[str, Any],
    independent_report: dict[str, Any],
    joint_report: dict[str, Any],
    wrong_coarse: tuple[int, int, int],
    output: Path,
) -> None:
    labels = ["outside\nhalf-cell", "wrong\nidentity", "collapsed\npairs", "off\nobject", "wrong\ncoarse"]

    def counts(report: dict[str, Any], coarse: int) -> list[int]:
        violations = report["violations"]
        return [
            violations["outside_half_cell_count"],
            violations["wrong_identity_count"],
            violations["collapsed_pair_count"],
            violations["off_object_count"],
            coarse,
        ]

    values = [
        counts(static_report, wrong_coarse[0]),
        counts(independent_report, wrong_coarse[1]),
        counts(joint_report, wrong_coarse[2]),
    ]
    names = ["static heatmap head", "independent query", "joint mode assignment"]
    colors = ["#777777", "#dd9922", "#3366cc"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for index, (row, name, color) in enumerate(zip(values, names, colors, strict=True)):
        bars = ax.bar(x + (index - 1) * width, row, width, label=name, color=color)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("correlated event count (lower is better)")
    ax.set_title("Gate 4c: frozen one-to-one mode assignment on the same 24-frame wedge")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _worst_figure(
    images: np.ndarray,
    target_px: np.ndarray,
    static_px: np.ndarray,
    independent_px: np.ndarray,
    joint_px: np.ndarray,
    score_maps: np.ndarray,
    assigned_y: np.ndarray,
    assigned_x: np.ndarray,
    joint_error: np.ndarray,
    output: Path,
) -> list[dict[str, Any]]:
    flat_order = np.argsort(joint_error.reshape(-1))[::-1][:8]
    events = [np.unravel_index(int(index), joint_error.shape) for index in flat_order]
    fig, axes = plt.subplots(len(events), 2, figsize=(12, 3.2 * len(events)), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for row_index, (frame, witness) in enumerate(events):
        rgb_ax, score_ax = axes[row_index]
        rgb_ax.imshow(images[frame])
        rgb_ax.scatter(
            [target_px[frame, witness, 0]], [target_px[frame, witness, 1]],
            c="#00dd66", marker="+", s=120, linewidths=2.6, label="certified target",
        )
        rgb_ax.scatter(
            [static_px[frame, witness, 0]], [static_px[frame, witness, 1]],
            c="#999999", marker="x", s=65, linewidths=1.8, label="static head",
        )
        rgb_ax.scatter(
            [independent_px[frame, witness, 0]], [independent_px[frame, witness, 1]],
            facecolors="none", edgecolors="#ffaa22", marker="s", s=70, linewidths=1.8,
            label="independent query",
        )
        rgb_ax.scatter(
            [joint_px[frame, witness, 0]], [joint_px[frame, witness, 1]],
            facecolors="none", edgecolors="#2266ff", marker="o", s=85, linewidths=2.0,
            label="joint assignment",
        )
        rgb_ax.set_title(
            f"frame {frame}, KP{witness}: joint error {joint_error[frame, witness]:.1f}px"
        )
        rgb_ax.axis("off")
        if row_index == 0:
            rgb_ax.legend(loc="lower right", fontsize=7)

        score_ax.imshow(score_maps[frame, witness], cmap="magma")
        target_cell = target_px[frame, witness] / 511.0 * 63.0
        score_ax.scatter(
            [target_cell[0]], [target_cell[1]], c="#00dd66", marker="+", s=100, linewidths=2.4
        )
        score_ax.scatter(
            [assigned_x[frame, witness]], [assigned_y[frame, witness]],
            facecolors="none", edgecolors="#33bbff", marker="o", s=90, linewidths=2.0,
        )
        score_ax.set_title("final-feature score: true site green, assigned mode blue")
        score_ax.set_xticks([])
        score_ax.set_yticks([])
        records.append(
            {
                "frame_index": int(frame),
                "witness_index": int(witness),
                "joint_material_error_px": float(joint_error[frame, witness]),
            }
        )
    fig.suptitle("Worst Gate 4c truth-blind joint-assignment events", fontsize=14)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return records


def _margin_figure(
    frame_index: np.ndarray,
    signed_correct_margin: np.ndarray,
    optimizer_margin: np.ndarray,
    all_identity_correct: np.ndarray,
    shared_frame: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(frame_index, signed_correct_margin, marker="o", label="correct minus best wrong permutation")
    ax.plot(frame_index, optimizer_margin, marker="s", label="optimizer best minus second best")
    wrong = ~all_identity_correct
    if wrong.any():
        ax.scatter(frame_index[wrong], signed_correct_margin[wrong], c="red", marker="x", s=90, label="wrong winning assignment")
    if shared_frame.any():
        ax.scatter(frame_index[shared_frame], signed_correct_margin[shared_frame], facecolors="none", edgecolors="purple", s=100, label="at least one shared r64 true-site cell")
    ax.set_xlabel("held-out frame index")
    ax.set_ylabel("total cosine-score margin")
    ax.set_title("Privileged true-site assignment ceiling (diagnostic only)")
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "evaluation output directory already exists")

    # Finish this raw binding/replay block before any privileged input is opened.
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

    require(file_record(args.gate4b_raw_arrays)["sha256"] == EXPECTED_GATE4B_RAW_SHA256, "Gate 4b raw binding differs")
    with np.load(args.gate4b_raw_arrays, allow_pickle=False) as loaded:
        gate4b = {name: np.asarray(loaded[name]) for name in loaded.files}

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
    require(score_maps.shape == (24, EXPECTED_WITNESSES, 64, 64), "raw score-map shape differs")
    replay = decode_joint_assignments(score_maps)
    for name, value in replay.items():
        require(name in raw, f"raw replay field missing: {name}")
        require(np.array_equal(np.asarray(raw[name]), value), f"raw replay differs: {name}")
    joint_px = np.asarray(raw["assigned_local_3x3_prediction_px"], dtype=np.float64)
    joint_hard_px = np.asarray(raw["assigned_hard_prediction_px"], dtype=np.float64)

    require(np.array_equal(gate4b["frame_index"], frame_index), "Gate 4b frames differ")
    require(tuple(gate4b["witness_id"].tolist()) == EXPECTED_WITNESS_IDS, "Gate 4b witnesses differ")
    require(str(np.asarray(gate4b["representation_name"]).item()) == FINAL_FEATURE_NAME, "Gate 4b representation differs")
    require(np.array_equal(gate4b["final_feature_query_score_map"], score_maps), "Gate 4b score maps differ")
    independent_px = np.asarray(gate4b["local_3x3_prediction_px"], dtype=np.float64)

    require(np.array_equal(baseline["validation_frame_index"], frame_index), "baseline frames differ")
    require(np.array_equal(baseline["validation_target_coordinate_px"], target_px), "baseline targets differ")
    static_px = np.asarray(baseline["validation_local_prediction_px"], dtype=np.float64)
    static_report, static_derived = evaluate_predictions(static_px, target_px, masks)
    require(static_report["violations"]["wrong_identity_count"] == BASELINE_WRONG_IDENTITY, "baseline wrong identity differs")
    require(static_report["violations"]["collapsed_pair_count"] == BASELINE_COLLAPSED_PAIR, "baseline collapsed pairs differ")
    require(static_report["violations"]["off_object_count"] == BASELINE_OFF_OBJECT, "baseline off-object differs")
    require(abs(static_report["material_error_px"]["maximum"] - BASELINE_MAXIMUM_ERROR_PX) <= 1e-12, "baseline maximum differs")
    require(baseline_result["validation_local_candidate"]["violations"] == static_report["violations"], "baseline result replay differs")
    static_category = np.asarray(baseline["validation_localization_category_code"], dtype=np.int8)
    static_wrong_coarse = int(np.isin(static_category, (2, 3)).sum())
    require(static_wrong_coarse == BASELINE_WRONG_COARSE, "baseline wrong-coarse differs")

    independent_report, independent_derived = evaluate_predictions(independent_px, target_px, masks)
    require(independent_report["violations"] == GATE4B_EXPECTED_VIOLATIONS, "Gate 4b violation replay differs")
    require(abs(independent_report["material_error_px"]["maximum"] - GATE4B_EXPECTED_MAXIMUM_ERROR_PX) <= 1e-12, "Gate 4b maximum replay differs")
    independent_diagnostic = target_rank_and_category(
        score_maps,
        gate4b["hard_cell_y"],
        gate4b["hard_cell_x"],
        target_px,
        independent_derived["within_half_cell"],
    )
    independent_category = independent_diagnostic["localization_category_code"]
    independent_wrong_coarse = int(np.isin(independent_category, (2, 3)).sum())
    require(independent_wrong_coarse == 38, "Gate 4b wrong-coarse replay differs")

    joint_report, joint_derived = evaluate_predictions(joint_px, target_px, masks)
    joint_hard_report, joint_hard_derived = evaluate_predictions(joint_hard_px, target_px, masks)
    joint_diagnostic = target_rank_and_category(
        score_maps,
        raw["assigned_cell_y"],
        raw["assigned_cell_x"],
        target_px,
        joint_derived["within_half_cell"],
    )
    joint_category = joint_diagnostic["localization_category_code"]
    joint_wrong_coarse_mask = np.isin(joint_category, (2, 3))
    joint_wrong_coarse = int(joint_wrong_coarse_mask.sum())

    within_two_static = static_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    within_two_independent = independent_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    within_two_joint = joint_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    truth_blind_components = {
        "zero_wrong_identity": joint_report["violations"]["wrong_identity_count"] == 0,
        "zero_collapsed_pairs": joint_report["violations"]["collapsed_pair_count"] == 0,
        "zero_off_object": joint_report["violations"]["off_object_count"] == 0,
        "all_within_two_cells": bool(within_two_joint.all()),
        "within_two_cells_count": int(within_two_joint.sum()),
        "event_count": int(within_two_joint.size),
    }
    truth_blind_pass = bool(all(truth_blind_components[key] for key in (
        "zero_wrong_identity", "zero_collapsed_pairs", "zero_off_object", "all_within_two_cells"
    )))

    true_site_scores = bilinear_sample_all_sites(score_maps, target_px)
    ceiling_assignment = np.empty((24, EXPECTED_WITNESSES), dtype=np.int64)
    best_score = np.empty(24, dtype=np.float64)
    second_score = np.empty(24, dtype=np.float64)
    optimizer_margin = np.empty(24, dtype=np.float64)
    correct_score = np.empty(24, dtype=np.float64)
    competing_score = np.empty(24, dtype=np.float64)
    signed_correct_margin = np.empty(24, dtype=np.float64)
    for frame in range(24):
        diagnostic = square_assignment_diagnostics(true_site_scores[frame])
        ceiling_assignment[frame] = diagnostic["best_assignment"]
        best_score[frame] = diagnostic["best_assignment_score"]
        second_score[frame] = diagnostic["optimizer_second_best_score"]
        optimizer_margin[frame] = diagnostic["optimizer_best_minus_second_margin"]
        correct_score[frame] = diagnostic["correct_assignment_score"]
        competing_score[frame] = diagnostic["best_competing_assignment_score"]
        signed_correct_margin[frame] = diagnostic["signed_correct_assignment_margin"]
    ceiling_identity_correct = ceiling_assignment == np.arange(EXPECTED_WITNESSES)[None]
    ceiling_frame_all_correct = ceiling_identity_correct.all(axis=1)
    shared_cell_pair, shared_cell_site = shared_true_site_cells(target_px)
    shared_cell_frame = shared_cell_pair.any(axis=(1, 2))
    ceiling_perfect_positive = bool(
        ceiling_identity_correct.all() and (signed_correct_margin > 0.0).all()
    )

    if truth_blind_pass:
        branch = "cheap_joint_decoder_candidate_full_orbit_and_second_object_required"
    elif ceiling_perfect_positive:
        branch = "true_site_metric_sufficient_truth_blind_failure_candidate_exclusion_or_temporal_ambiguous"
    else:
        branch = "fixed_frame27_cosine_metric_lacks_unique_correct_margin_test_frozen_encoder_learned_metric_next"

    comparisons = {
        "static_to_joint": {
            "within_half_cell": _change_counts(static_derived["within_half_cell"], joint_derived["within_half_cell"]),
            "within_two_cells": _change_counts(within_two_static, within_two_joint),
            "identity_correct": _change_counts(static_derived["identity_correct"], joint_derived["identity_correct"]),
            "on_object": _change_counts(static_derived["on_object"], joint_derived["on_object"]),
            "distinct_pair": _change_counts(static_derived["distinct_pair"], joint_derived["distinct_pair"]),
            "coarse_mode_correct": _change_counts(~np.isin(static_category, (2, 3)), ~joint_wrong_coarse_mask),
        },
        "independent_query_to_joint": {
            "within_half_cell": _change_counts(independent_derived["within_half_cell"], joint_derived["within_half_cell"]),
            "within_two_cells": _change_counts(within_two_independent, within_two_joint),
            "identity_correct": _change_counts(independent_derived["identity_correct"], joint_derived["identity_correct"]),
            "on_object": _change_counts(independent_derived["on_object"], joint_derived["on_object"]),
            "distinct_pair": _change_counts(independent_derived["distinct_pair"], joint_derived["distinct_pair"]),
            "coarse_mode_correct": _change_counts(~np.isin(independent_category, (2, 3)), ~joint_wrong_coarse_mask),
        },
    }

    wrong_ceiling_events = [
        {"frame_index": int(frame), "query_index": int(query), "assigned_site_index": int(ceiling_assignment[frame, query])}
        for frame, query in np.argwhere(~ceiling_identity_correct)
    ]
    shared_pairs = [
        {"frame_index": int(frame), "left_site": int(left), "right_site": int(right)}
        for frame, left, right in np.argwhere(shared_cell_pair)
    ]

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "FINAL_FEATURE_JOINT_ASSIGNMENT_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        target_coordinate_px=target_px,
        static_prediction_px=static_px,
        independent_query_prediction_px=independent_px,
        joint_prediction_px=joint_px,
        joint_hard_prediction_px=joint_hard_px,
        static_material_error_px=static_derived["material_error_px"],
        independent_query_material_error_px=independent_derived["material_error_px"],
        joint_material_error_px=joint_derived["material_error_px"],
        static_within_half_cell=static_derived["within_half_cell"],
        independent_query_within_half_cell=independent_derived["within_half_cell"],
        joint_within_half_cell=joint_derived["within_half_cell"],
        joint_within_two_cells=within_two_joint,
        static_identity_correct=static_derived["identity_correct"],
        independent_query_identity_correct=independent_derived["identity_correct"],
        joint_identity_correct=joint_derived["identity_correct"],
        joint_on_object=joint_derived["on_object"],
        joint_distinct_pair=joint_derived["distinct_pair"],
        joint_localization_category_code=joint_category,
        joint_target_nearest_cell_rank=joint_diagnostic["target_nearest_cell_rank"],
        joint_target_cell_inside_local_window=joint_diagnostic["target_cell_inside_local_window"],
        true_site_score_matrix=true_site_scores,
        true_site_best_assignment=ceiling_assignment,
        true_site_identity_correct=ceiling_identity_correct,
        true_site_best_assignment_score=best_score,
        true_site_optimizer_second_best_score=second_score,
        true_site_optimizer_best_minus_second_margin=optimizer_margin,
        true_site_correct_assignment_score=correct_score,
        true_site_best_competing_assignment_score=competing_score,
        true_site_signed_correct_assignment_margin=signed_correct_margin,
        true_site_shared_r64_cell_pair=shared_cell_pair,
        true_site_shares_r64_cell=shared_cell_site,
    )
    summary_figure = args.output_dir / "01_STATIC_INDEPENDENT_JOINT_COUNTS.png"
    worst_figure = args.output_dir / "02_JOINT_ASSIGNMENT_WORST_EVENTS.png"
    margin_figure = args.output_dir / "03_TRUE_SITE_ASSIGNMENT_MARGINS.png"
    _summary_figure(
        static_report,
        independent_report,
        joint_report,
        (static_wrong_coarse, independent_wrong_coarse, joint_wrong_coarse),
        summary_figure,
    )
    worst_events = _worst_figure(
        images,
        target_px,
        static_px,
        independent_px,
        joint_px,
        score_maps,
        raw["assigned_cell_y"],
        raw["assigned_cell_x"],
        joint_derived["material_error_px"],
        worst_figure,
    )
    _margin_figure(
        frame_index,
        signed_correct_margin,
        optimizer_margin,
        ceiling_frame_all_correct,
        shared_cell_frame,
        margin_figure,
    )

    result_path = args.output_dir / "FINAL_FEATURE_JOINT_ASSIGNMENT_RESULT.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "privileged_posthash_final_feature_joint_assignment_evaluation",
        "repository_head": repository_head,
        "raw_repository_head": raw_repository_head,
        "raw_predictions_hashed_and_loaded_before_truth_masks_baseline_or_rgb": raw_loaded_before_privileged_inputs,
        "sample_scope": {
            "object_count": 1,
            "checkpoint_count": 1,
            "heldout_frame_count": 24,
            "witness_count": 10,
            "event_count": 240,
            "sample_unit": "one fixed witness query on one held-out frame",
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
            "cross_object_generalization_authorized": False,
        },
        "static_heatmap_head": static_report,
        "independent_final_feature_query_gate4b": independent_report,
        "truth_blind_joint_mode_assignment": joint_report,
        "concurrent_joint_hard_grid_control": joint_hard_report,
        "localization_category_counts": {
            "static_heatmap_head": _category_counts(static_category),
            "independent_final_feature_query_gate4b": _category_counts(independent_category),
            "truth_blind_joint_mode_assignment": _category_counts(joint_category),
        },
        "event_changes": comparisons,
        "truth_blind_gate": {
            "passed": truth_blind_pass,
            "components": truth_blind_components,
            "strict_5_735px_complete_solution": joint_report["strict_capability_pass"],
            "strict_half_cell_threshold_px": HALF_CELL_DIAGONAL_PX,
            "practical_two_cell_threshold_px": TWO_CELL_SPACING_PX,
            "wrong_coarse_count": joint_wrong_coarse,
            "candidate_mode_count": _numeric_summary(raw["candidate_mode_count"]),
        },
        "privileged_true_site_score_ceiling": {
            "diagnostic_only_not_deployable": True,
            "identity_correct_count": int(ceiling_identity_correct.sum()),
            "wrong_identity_count": int(ceiling_identity_correct.size - ceiling_identity_correct.sum()),
            "all_240_identities_correct": bool(ceiling_identity_correct.all()),
            "frames_all_identities_correct": int(ceiling_frame_all_correct.sum()),
            "all_frames_positive_signed_correct_margin": bool((signed_correct_margin > 0.0).all()),
            "perfect_with_positive_signed_margin": ceiling_perfect_positive,
            "signed_correct_assignment_margin": _numeric_summary(signed_correct_margin),
            "optimizer_best_minus_second_margin": _numeric_summary(optimizer_margin),
            "wrong_assignment_events": wrong_ceiling_events,
            "shared_r64_true_site_pair_count": len(shared_pairs),
            "shared_r64_true_site_pairs": shared_pairs,
            "margin_definition": {
                "decision_critical": "score(correct identity permutation) minus score(best competing permutation)",
                "optimizer_uniqueness_only": "score(optimizer-best assignment) minus score(optimizer-second-best assignment)",
            },
        },
        "decision": {
            "branch": branch,
            "operator_authorized": False,
            "training_authorized": False,
            "gpu_run_authorized": False,
            "full_orbit_or_second_object_authorized": truth_blind_pass,
            "next_action_if_metric_branch": "design, do not yet run, the smallest leakage-safe learned projection or metric with the encoder frozen",
        },
        "interpretation_limits": [
            "A failed true-site ceiling rejects the fixed frame-27 cosine query metric; it does not prove frozen feature vectors contain no decodable identity.",
            "Distinct r64 candidate cells need not be distinct material regions, so truth-blind failure is ambiguous among candidate generation, spatial exclusion or joint constraints, and score ambiguity.",
            "A positive optimizer-best-minus-second margin does not establish correct identity; only the signed correct-assignment margin addresses that claim.",
            "One hammer, one checkpoint, and 24 correlated frames cannot establish cross-object generalization.",
        ],
        "worst_visual_events": worst_events,
        "controls": {
            **visual_controls,
            "raw_joint_prediction_recomputed_exactly": True,
            "baseline_and_gate4b_recomputed_under_same_targets_and_masks": True,
            "truth_opened_only_after_raw_prediction_hash": True,
            "training_or_weight_update_performed": False,
            "torch_imported": "torch" in sys.modules,
            "laptop_gpu_used": False,
        },
        "bindings": {
            "semantic_lock": semantic_lock_record,
            "raw_receipt": file_record(args.raw_receipt),
            "raw_arrays": file_record(raw_arrays_path),
            "gate4b_raw_arrays": file_record(args.gate4b_raw_arrays),
            "validation_truth": validation_truth_record,
            "mask_manifest": mask_manifest_record,
            "rgb_manifest": rgb_manifest_record,
            "baseline_arrays": baseline_arrays_record,
            "baseline_result": baseline_result_record,
            "derived_arrays": file_record(arrays_path),
            "summary_figure": file_record(summary_figure),
            "worst_figure": file_record(worst_figure),
            "margin_figure": file_record(margin_figure),
        },
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "joint_assignment_contract": file_record(
                args.repo_root / "keypoint_net" / "final_feature_joint_assignment.py"
            ),
        },
    }
    require(result["controls"]["torch_imported"] is False, "Torch entered the evaluator")
    _write_json(result_path, result)
    receipt_path = args.output_dir / "FINAL_FEATURE_JOINT_ASSIGNMENT_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "final_feature_joint_assignment_evaluation_receipt.v1",
        "repository_head": repository_head,
        "raw_repository_head": raw_repository_head,
        "result": file_record(result_path),
        "derived_arrays": file_record(arrays_path),
        "summary_figure": file_record(summary_figure),
        "worst_figure": file_record(worst_figure),
        "margin_figure": file_record(margin_figure),
        "raw_loaded_before_privileged_inputs": raw_loaded_before_privileged_inputs,
        "decision_branch": branch,
        "truth_blind_gate_passed": truth_blind_pass,
        "true_site_ceiling_perfect_positive_margin": ceiling_perfect_positive,
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
    parser.add_argument("--gate4b-raw-arrays", type=Path, required=True)
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
        print(f"FINAL_FEATURE_JOINT_ASSIGNMENT_EVALUATION_FAILED: {exc}", file=sys.stderr)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
