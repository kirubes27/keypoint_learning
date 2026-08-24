"""Privileged rank/margin evaluation of frozen encoder/head score maps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from certified_witness_capability import EXPECTED_WITNESS_IDS, file_record, require
from encoder_head_localization import (
    REPRESENTATION_NAMES,
    explicit_competitor_margins,
    fixed_transition_labels,
    stage_class,
    summarize_vector,
)


SCHEMA_VERSION = "augmented_encoder_head_localization_evaluation.v1"
EXPECTED_RAW_SCHEMA = "augmented_encoder_head_localization_raw_receipt.v1"
EXPECTED_BINDINGS = {
    "validation_truth": "3e188cd699cbddfe10ce51a3c1de97f7be95f958012219ed2699f4c0d565819b",
    "evaluation_arrays": "79a6a7f1b862df16dcd42442731e152988371fb598eac11a6114d3e4d0158e82",
    "evaluation_result": "6904cd54b5d92d35080e6ac9b3ba6f13dd1bf49806eba778f7d5e9d294269968",
}
LEVEL_LABELS = ("penultimate", "final pre-head", "heatmap logits")
TRANSITION_ORDER = (
    "badly_ranked_by_penultimate",
    "lost_in_final_encoder_block",
    "lost_in_heatmap_head",
    "selection_or_local_readout",
    "ambiguous_or_nonmonotonic",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root differs: {path}")
    return value


def _verify_record(record: dict[str, Any], label: str) -> Path:
    require(isinstance(record, dict), f"{label} record missing")
    path = Path(str(record.get("absolute_path", "")))
    require(file_record(path) == record, f"{label} binding differs")
    return path


def _bound(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == expected_hash, f"{label} SHA-256 differs")
    return record


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


def _rate(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    require(values.size > 0, "empty rate")
    return float(values.mean())


def _level_summary(rank: np.ndarray, margin: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected_rank = np.asarray(rank)[mask]
    selected_margin = np.asarray(margin)[mask]
    require(selected_rank.size > 0, "summary stratum is empty")
    return {
        "rank": summarize_vector(selected_rank),
        "top1_rate": _rate(selected_rank <= 1),
        "top3_rate": _rate(selected_rank <= 3),
        "top10_rate": _rate(selected_rank <= 10),
        "badly_ranked_gt10_rate": _rate(selected_rank > 10),
        "margin": summarize_vector(selected_margin),
        "positive_margin_rate": _rate(selected_margin > 0.0),
    }


def _stratified_summary(
    rank: np.ndarray,
    margin: np.ndarray,
    wrong_coarse: np.ndarray,
    wrong_identity: np.ndarray,
) -> dict[str, Any]:
    masks = {
        "all_heldout_events": np.ones(wrong_coarse.shape, dtype=bool),
        "preexisting_wrong_coarse": wrong_coarse,
        "preexisting_wrong_identity": wrong_identity,
    }
    result: dict[str, Any] = {}
    for level_index, name in enumerate(REPRESENTATION_NAMES):
        result[name] = {
            stratum: _level_summary(rank[level_index], margin[level_index], mask)
            for stratum, mask in masks.items()
        }
        result[name]["per_witness_all_events"] = [
            _level_summary(
                rank[level_index, :, witness],
                margin[level_index, :, witness],
                np.ones(rank.shape[1], dtype=bool),
            )
            for witness in range(rank.shape[2])
        ]
    return result


def _transition_counts(labels: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    counter = Counter(np.asarray(labels)[mask].tolist())
    return {name: int(counter.get(name, 0)) for name in TRANSITION_ORDER}


def _rank_figure(rank: np.ndarray, output: Path) -> None:
    median = np.median(rank, axis=1)
    top3 = np.mean(rank <= 3, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    image = axes[0].imshow(np.log10(median + 1.0), cmap="magma", aspect="auto")
    axes[0].set_title("Median target-cell rank by frozen representation\n(log10(rank + 1); 24 correlated held-out frames)")
    axes[0].set_yticks(range(3), LEVEL_LABELS)
    axes[0].set_xticks(range(10), [f"KP{k}" for k in range(10)])
    for row in range(3):
        for column in range(10):
            axes[0].text(column, row, f"{median[row, column]:.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046)

    image = axes[1].imshow(top3, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title("Target cell top-3 rate\n(descriptive event rate; no independence claim)")
    axes[1].set_yticks(range(3), LEVEL_LABELS)
    axes[1].set_xticks(range(10), [f"KP{k}" for k in range(10)])
    for row in range(3):
        for column in range(10):
            axes[1].text(column, row, f"{100 * top3[row, column]:.0f}%", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _transition_figure(all_counts: dict[str, int], wrong_counts: dict[str, int], output: Path) -> None:
    labels = [
        "bad by\npenultimate",
        "lost in final\nencoder block",
        "lost in\nheatmap head",
        "selection /\nreadout",
        "ambiguous /\nnonmonotonic",
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    left = ax.bar(x - width / 2, [all_counts[name] for name in TRANSITION_ORDER], width, label="all 240 events")
    right = ax.bar(x + width / 2, [wrong_counts[name] for name in TRANSITION_ORDER], width, label="49 pre-existing wrong-coarse events")
    ax.bar_label(left, padding=3)
    ax.bar_label(right, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("correlated channel-frame event count")
    ax.set_title("Pre-registered representation-stage localization\n(descriptive counts, one object and one checkpoint)")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _worst_event_figure(
    score_maps: np.ndarray,
    target_px: np.ndarray,
    local_prediction_px: np.ndarray,
    rank: np.ndarray,
    wrong_coarse: np.ndarray,
    manifest: dict[str, Any],
    output: Path,
) -> list[dict[str, int]]:
    candidates = np.argwhere(wrong_coarse)
    require(len(candidates) == 49, "wrong-coarse visual candidate count differs")
    ordering = sorted(
        candidates.tolist(),
        key=lambda pair: (int(rank[1, pair[0], pair[1]]), int(rank[0, pair[0], pair[1]])),
        reverse=True,
    )[:6]
    records = {int(row["frame_index"]): row for row in manifest["evaluation"]["frame_records"]}
    root = Path(str(manifest["object_root"]))
    fig, axes = plt.subplots(len(ordering), 4, figsize=(14, 3.2 * len(ordering)), constrained_layout=True)
    for row_index, (frame, witness) in enumerate(ordering):
        record = records[frame]
        image_path = root / str(record["image_relpath"])
        require(file_record(image_path)["sha256"] == record["image_sha256"], "visual RGB binding differs")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].scatter(
            [target_px[frame, witness, 0]],
            [target_px[frame, witness, 1]],
            c="#00ff66",
            marker="+",
            s=120,
            linewidths=2.5,
            label="certified target",
        )
        axes[row_index, 0].scatter(
            [local_prediction_px[frame, witness, 0]],
            [local_prediction_px[frame, witness, 1]],
            c="#ff3355",
            marker="x",
            s=90,
            linewidths=2.0,
            label="detector",
        )
        axes[row_index, 0].set_title(f"frame {frame}, KP{witness}")
        axes[row_index, 0].axis("off")
        if row_index == 0:
            axes[row_index, 0].legend(fontsize=7, loc="lower right")
        target_cell = target_px[frame, witness] / 511.0 * 63.0
        logit_peak = np.unravel_index(np.argmax(score_maps[2, frame, witness]), (64, 64))
        for level in range(3):
            ax = axes[row_index, level + 1]
            ax.imshow(score_maps[level, frame, witness], cmap="magma")
            ax.scatter([target_cell[0]], [target_cell[1]], c="#00ff66", marker="+", s=90, linewidths=2.0)
            ax.scatter([logit_peak[1]], [logit_peak[0]], c="#33bbff", marker="x", s=65, linewidths=1.8)
            ax.set_title(f"{LEVEL_LABELS[level]}\nrank={rank[level, frame, witness]}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Worst final-prehead ranks among the fixed wrong-coarse events\ngreen = certified target; blue = frozen heatmap-logit peak", fontsize=14)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return [{"frame_index": int(frame), "witness_index": int(witness)} for frame, witness in ordering]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "evaluation output directory already exists")

    # Load and hash frozen raw scores before opening any privileged target or classification.
    raw_receipt = _load_json(args.raw_receipt)
    require(raw_receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw receipt schema differs")
    require(raw_receipt.get("privileged_evaluation_authorized") is True, "raw receipt blocks evaluation")
    require(raw_receipt.get("controls", {}).get("validation_target_received_or_opened") is False, "raw process opened target")
    raw_arrays_path = _verify_record(raw_receipt["raw_arrays"], "raw score arrays")
    with np.load(raw_arrays_path) as loaded:
        raw = {name: np.asarray(loaded[name]) for name in loaded.files}
    raw_loaded_before_truth = True

    repository_head = _verify_clean_head(args.repo_root, args.expected_repo_head)
    require(raw_receipt.get("repository_head") == repository_head, "raw repository differs")
    manifest_path = _verify_record(raw_receipt["manifest"], "manifest")
    manifest = _load_json(manifest_path)
    _verify_record(manifest["semantic_lock"], "semantic lock")
    for relative, record in manifest["implementation_sources"].items():
        require(file_record(args.repo_root / relative) == record, f"implementation differs: {relative}")

    validation_truth_record = _bound(
        args.validation_truth, EXPECTED_BINDINGS["validation_truth"], "validation truth"
    )
    evaluation_arrays_record = _bound(
        args.prior_evaluation_arrays, EXPECTED_BINDINGS["evaluation_arrays"], "prior evaluation arrays"
    )
    evaluation_result_record = _bound(
        args.prior_evaluation_result, EXPECTED_BINDINGS["evaluation_result"], "prior evaluation result"
    )

    with np.load(args.validation_truth) as loaded:
        frame_index = np.asarray(loaded["frame_index"], dtype=np.int64)
        witness_id = np.asarray(loaded["witness_id"], dtype=np.int64)
        target_px = np.asarray(loaded["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(loaded["physical_valid"], dtype=bool)
        target_on_object = np.asarray(loaded["target_on_object"], dtype=bool)
    require(np.array_equal(frame_index, np.arange(24)), "validation frame order differs")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "validation witness order differs")
    require(target_px.shape == (24, 10, 2), "validation target shape differs")
    require(bool(physical_valid.all() and target_on_object.all()), "validation target validity differs")

    with np.load(args.prior_evaluation_arrays) as loaded:
        prior = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(np.array_equal(prior["validation_frame_index"], frame_index), "prior validation frames differ")
    require(np.array_equal(prior["validation_target_coordinate_px"], target_px), "prior validation targets differ")
    category = np.asarray(prior["validation_localization_category_code"], dtype=np.int8)
    wrong_coarse = np.isin(category, (2, 3))
    wrong_identity = ~np.asarray(prior["validation_local_identity_correct"], dtype=bool)
    require(int(wrong_coarse.sum()) == 49, "pre-existing wrong-coarse count differs")
    require(int(wrong_identity.sum()) == 31, "pre-existing wrong-identity count differs")

    require(tuple(raw["representation_name"].tolist()) == REPRESENTATION_NAMES, "representation order differs")
    require(np.array_equal(raw["frame_index"], frame_index), "raw frame order differs")
    require(tuple(raw["witness_id"].tolist()) == EXPECTED_WITNESS_IDS, "raw witness order differs")
    score_maps = np.asarray(raw["score_maps"], dtype=np.float64)
    require(score_maps.shape == (3, 24, 10, 64, 64), "raw score map shape differs")
    hard_flat = np.argmax(raw["native_heatmap_logits"].reshape(24, 10, -1), axis=-1)
    hard_y, hard_x = np.divmod(hard_flat, 64)
    measurements = explicit_competitor_margins(
        score_maps,
        target_px,
        hard_x.astype(np.int64),
        hard_y.astype(np.int64),
    )
    require(np.array_equal(measurements["head_wrong_coarse_event"], wrong_coarse), "recomputed wrong-coarse mask differs")
    rank = measurements["target_cell_rank"]
    margin = measurements["target_minus_competitor_margin"]
    classes = stage_class(rank)
    transitions = fixed_transition_labels(rank, wrong_coarse)
    all_transition_counts = _transition_counts(transitions, np.ones(wrong_coarse.shape, dtype=bool))
    wrong_transition_counts = _transition_counts(transitions, wrong_coarse)
    wrong_identity_transition_counts = _transition_counts(transitions, wrong_identity)

    head_only_complete = bool(np.all(rank[1][wrong_coarse] <= 3))
    penultimate_bad_wrong = int(np.sum((rank[0] > 10) & wrong_coarse))
    final_block_loss_wrong = int(np.sum((transitions == "lost_in_final_encoder_block") & wrong_coarse))
    head_loss_wrong = int(np.sum((transitions == "lost_in_heatmap_head") & wrong_coarse))
    if head_only_complete:
        recommended_branch = "retain_encoder_build_query_conditioned_head"
    elif penultimate_bad_wrong > 0:
        recommended_branch = "head_only_incomplete_replace_or_domain_adapt_earlier_representation"
    elif final_block_loss_wrong > 0:
        recommended_branch = "fine_tune_from_final_encoder_block_upward"
    elif head_loss_wrong > 0:
        recommended_branch = "retain_encoder_build_query_conditioned_head_with_ambiguous_residuals"
    else:
        recommended_branch = "mixed_or_ambiguous_requires_bounded_decoder_falsifier"

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "ENCODER_HEAD_LOCALIZATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        representation_name=np.asarray(REPRESENTATION_NAMES),
        frame_index=frame_index,
        witness_id=witness_id,
        target_coordinate_px=target_px,
        wrong_coarse_event=wrong_coarse,
        wrong_identity_event=wrong_identity,
        target_cell_x=measurements["target_cell_x"],
        target_cell_y=measurements["target_cell_y"],
        target_cell_rank=rank,
        stage_class_code=classes,
        continuous_target_score=measurements["continuous_target_score"],
        maximum_competitor_score=measurements["maximum_competitor_score"],
        maximum_competitor_source_code=measurements["maximum_competitor_source_code"],
        target_minus_competitor_margin=margin,
        identity_competitor_witness=measurements["identity_competitor_witness"],
        separated_spatial_competitor_x=measurements["separated_spatial_competitor_x"],
        separated_spatial_competitor_y=measurements["separated_spatial_competitor_y"],
        transition_label=transitions,
    )
    rank_figure = args.output_dir / "01_TARGET_RANK_LOCALIZATION.png"
    transition_figure = args.output_dir / "02_STAGE_TRANSITION_COUNTS.png"
    worst_figure = args.output_dir / "03_WORST_WRONG_COARSE_EVENTS.png"
    _rank_figure(rank, rank_figure)
    _transition_figure(all_transition_counts, wrong_transition_counts, transition_figure)
    worst_events = _worst_event_figure(
        score_maps,
        target_px,
        np.asarray(prior["validation_local_prediction_px"], dtype=np.float64),
        rank,
        wrong_coarse,
        manifest,
        worst_figure,
    )

    result_path = args.output_dir / "ENCODER_HEAD_LOCALIZATION_RESULT.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "privileged_augmented_encoder_head_rank_margin_localization",
        "repository_head": repository_head,
        "raw_scores_hashed_and_loaded_before_validation_truth": raw_loaded_before_truth,
        "sample_scope": {
            "object_count": 1,
            "checkpoint_count": 1,
            "heldout_frame_count": 24,
            "witness_count": 10,
            "event_count": 240,
            "sample_unit": "one fixed witness channel on one held-out frame",
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
        },
        "frozen_strata": {
            "wrong_coarse_event_count": int(wrong_coarse.sum()),
            "wrong_identity_event_count": int(wrong_identity.sum()),
        },
        "representation_summary": _stratified_summary(rank, margin, wrong_coarse, wrong_identity),
        "transition_counts": {
            "all_heldout_events": all_transition_counts,
            "preexisting_wrong_coarse": wrong_transition_counts,
            "preexisting_wrong_identity": wrong_identity_transition_counts,
        },
        "decision": {
            "head_only_complete_candidate_eligible": head_only_complete,
            "wrong_coarse_badly_ranked_by_penultimate_count": penultimate_bad_wrong,
            "wrong_coarse_lost_in_final_encoder_block_count": final_block_loss_wrong,
            "wrong_coarse_lost_in_heatmap_head_count": head_loss_wrong,
            "recommended_branch": recommended_branch,
            "training_authorized": False,
            "operator_authorized": False,
        },
        "score_contract": {
            "target_cell_rank": "1 plus number of strictly greater cells among all 4096 cells",
            "strongly_retained": "rank 1 through 3",
            "weak_or_ambiguous": "rank 4 through 10",
            "badly_ranked": "rank greater than 10",
            "margin": "continuous target score minus maximum of other material sites, wrong coarse peak when present, and best cell more than 4 cells away",
            "maximum_competitor_source_codes": {"0": "other_material_identity", "1": "wrong_coarse_logit_peak", "2": "separated_spatial_cell"},
        },
        "worst_visual_events": worst_events,
        "bindings": {
            "raw_receipt": file_record(args.raw_receipt),
            "raw_arrays": file_record(raw_arrays_path),
            "validation_truth": validation_truth_record,
            "prior_evaluation_arrays": evaluation_arrays_record,
            "prior_evaluation_result": evaluation_result_record,
            "derived_arrays": file_record(arrays_path),
            "rank_figure": file_record(rank_figure),
            "transition_figure": file_record(transition_figure),
            "worst_event_figure": file_record(worst_figure),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = args.output_dir / "ENCODER_HEAD_LOCALIZATION_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "augmented_encoder_head_localization_evaluation_receipt.v1",
        "command_argv": list(sys.argv),
        "repository_head": repository_head,
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "measurements": file_record(args.repo_root / "keypoint_net" / "encoder_head_localization.py"),
        },
        "result": file_record(result_path),
        "derived_arrays": file_record(arrays_path),
        "raw_loaded_before_truth": True,
        "training_or_weight_update_performed": False,
        "laptop_gpu_used": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--prior-evaluation-arrays", type=Path, required=True)
    parser.add_argument("--prior-evaluation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
