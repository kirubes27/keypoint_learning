"""Privileged post-hash audit of Gate 4c candidate coverage versus selection."""

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
from scipy.optimize import linear_sum_assignment

from final_feature_joint_assignment import (
    EXPECTED_FRAMES,
    EXPECTED_GATE4B_RAW_SHA256,
    EXPECTED_WITNESS_IDS,
    EXPECTED_WITNESSES,
    HALF_CELL_DIAGONAL_PX,
    TWO_CELL_SPACING_PX,
    evaluate_predictions,
    file_record,
    local_readout_at_cells,
    nearest_r64_cells,
    require,
    spatial_local_maxima,
)


SCHEMA_VERSION = "joint_assignment_candidate_failure_audit.v1"
EXPECTED_AUDIT_LOCK_SHA256 = (
    "1c462ba6cfd154eba4dbcb39c16c213356da4b4843d64825d710d489f9d232de"
)
EXPECTED_RAW_RECEIPT_SHA256 = (
    "f8be926a837c3649c8257bedc3a5ab1dbdb4f619b7cef4c4f92f4f9ef14ea321"
)
EXPECTED_VALIDATION_TRUTH_SHA256 = (
    "3e188cd699cbddfe10ce51a3c1de97f7be95f958012219ed2699f4c0d565819b"
)
EXPECTED_MASK_MANIFEST_SHA256 = (
    "d4a02868b6bd645f106703b88af59513d0aa2eb8ba2ebef6eac64683af667b00"
)


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


def pair_distance_counts(cell_y: Any, cell_x: Any) -> dict[str, int]:
    y = np.asarray(cell_y, dtype=np.int64)
    x = np.asarray(cell_x, dtype=np.int64)
    require(y.shape == x.shape and y.ndim == 2, "pair-cell shape differs")
    pair_distance = np.maximum(
        np.abs(y[:, :, None] - y[:, None, :]),
        np.abs(x[:, :, None] - x[:, None, :]),
    )
    pair_mask = np.triu(np.ones((y.shape[1], y.shape[1]), dtype=bool), k=1)
    values = pair_distance[:, pair_mask]
    return {
        "exact_same_cell": int((values == 0).sum()),
        "within_1_cell": int((values <= 1).sum()),
        "within_2_cells": int((values <= 2).sum()),
        "within_4_cells": int((values <= 4).sum()),
        "pair_event_count": int(values.size),
    }


def score_rank_strictly_greater(candidate_scores: Any, selected_index: int) -> int:
    scores = np.asarray(candidate_scores, dtype=np.float64)
    require(scores.ndim == 1 and 0 <= selected_index < len(scores), "rank input differs")
    require(bool(np.isfinite(scores).all()), "rank score is non-finite")
    return 1 + int((scores > scores[selected_index]).sum())


def spatial_candidate_assignment(
    candidate_y: Any, candidate_x: Any, target_px: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(candidate_y, dtype=np.int64)
    x = np.asarray(candidate_x, dtype=np.int64)
    targets = np.asarray(target_px, dtype=np.float64)
    require(y.shape == x.shape and y.ndim == 1, "spatial candidates differ")
    require(len(y) >= EXPECTED_WITNESSES, "spatial oracle has too few candidates")
    require(targets.shape == (EXPECTED_WITNESSES, 2), "spatial target shape differs")
    target_grid_x = targets[:, 0] / 511.0 * 63.0
    target_grid_y = targets[:, 1] / 511.0 * 63.0
    squared_distance = (
        (target_grid_y[:, None] - y[None]) ** 2
        + (target_grid_x[:, None] - x[None]) ** 2
    )
    row, column = linear_sum_assignment(squared_distance)
    require(np.array_equal(row, np.arange(EXPECTED_WITNESSES)), "spatial assignment row differs")
    return y[column], x[column], np.sqrt(squared_distance[row, column])


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


def _load_masks(manifest_path: Path, object_root: Path, frames: np.ndarray) -> np.ndarray:
    manifest = _load_json(manifest_path)
    require(
        manifest.get("schema_version")
        == "leakage_safe_distillation_evaluation_mask_manifest.v1",
        "mask manifest schema differs",
    )
    records = {int(row["frame_index"]): row for row in manifest["frames"]}
    missing = sorted(set(map(int, frames.tolist())) - set(records))
    require(not missing, f"mask manifest omits frames: {missing}")
    masks = np.empty((len(frames), 512, 512), dtype=bool)
    for local_index, frame in enumerate(frames.tolist()):
        row = records[frame]
        path = object_root / str(row["mask_relpath"])
        require(file_record(path)["sha256"] == row["mask_sha256"], f"mask hash differs: {frame}")
        mask = np.asarray(Image.open(path).convert("L")) > 0
        require(mask.shape == (512, 512), "mask shape differs")
        masks[local_index] = mask
    return masks


def _summary(values: Any) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "summary input invalid")
    return {
        "n": int(vector.size),
        "minimum": float(vector.min()),
        "median": float(np.median(vector)),
        "mean": float(vector.mean()),
        "maximum": float(vector.max()),
    }


def _figure(
    pair_counts: dict[str, int],
    union_distance: np.ndarray,
    own_distance: np.ndarray,
    candidate_rank: np.ndarray,
    oracle_report: dict[str, Any],
    oracle_within_two_count: int,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    pair_names = ["exact", "<=1", "<=2", "<=4"]
    pair_values = [
        pair_counts["exact_same_cell"],
        pair_counts["within_1_cell"],
        pair_counts["within_2_cells"],
        pair_counts["within_4_cells"],
    ]
    bars = axes[0].bar(pair_names, pair_values, color="#777777")
    axes[0].bar_label(bars)
    axes[0].set_title("Gate 4b pair proximity\n(Chebyshev r64 cells)")
    axes[0].set_ylabel("pair-frame count")

    coverage_names = ["union <=1", "own-query <=1", "nearest rank <=1", "rank <=3", "rank <=10"]
    coverage_values = [
        int((union_distance <= 1).sum()),
        int((own_distance <= 1).sum()),
        int((candidate_rank <= 1).sum()),
        int((candidate_rank <= 3).sum()),
        int((candidate_rank <= 10).sum()),
    ]
    bars = axes[1].bar(coverage_names, coverage_values, color="#3366cc")
    axes[1].bar_label(bars, fontsize=8)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, 250)
    axes[1].set_title("Candidate coverage and score rank\n(out of 240 query-events)")

    violations = oracle_report["violations"]
    oracle_names = ["outside\nhalf-cell", "outside\ntwo-cell", "wrong\nidentity", "collapsed\npairs", "off\nobject"]
    oracle_values = [
        violations["outside_half_cell_count"],
        240 - oracle_within_two_count,
        violations["wrong_identity_count"],
        violations["collapsed_pair_count"],
        violations["off_object_count"],
    ]
    bars = axes[2].bar(oracle_names, oracle_values, color="#22aa66")
    axes[2].bar_label(bars)
    axes[2].set_title("Privileged spatial candidate oracle\n(lower is better)")
    fig.suptitle("Gate 4c post-hash localization of candidate failure", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "audit output directory already exists")
    repository_head = _verify_clean_head(args.repo_root, args.expected_repo_head)
    audit_lock_record = file_record(args.audit_lock)
    require(audit_lock_record["sha256"] == EXPECTED_AUDIT_LOCK_SHA256, "audit lock differs")
    raw_receipt_record = file_record(args.raw_receipt)
    require(raw_receipt_record["sha256"] == EXPECTED_RAW_RECEIPT_SHA256, "raw receipt differs")
    raw_receipt = _load_json(args.raw_receipt)
    raw_arrays_path = Path(str(raw_receipt["raw_arrays"]["absolute_path"]))
    require(file_record(raw_arrays_path) == raw_receipt["raw_arrays"], "raw arrays differ")
    with np.load(raw_arrays_path, allow_pickle=False) as loaded:
        raw = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(file_record(args.gate4b_raw_arrays)["sha256"] == EXPECTED_GATE4B_RAW_SHA256, "Gate 4b arrays differ")
    with np.load(args.gate4b_raw_arrays, allow_pickle=False) as loaded:
        gate4b = {name: np.asarray(loaded[name]) for name in loaded.files}

    truth_record = file_record(args.validation_truth)
    mask_record = file_record(args.mask_manifest)
    require(truth_record["sha256"] == EXPECTED_VALIDATION_TRUTH_SHA256, "validation truth differs")
    require(mask_record["sha256"] == EXPECTED_MASK_MANIFEST_SHA256, "mask manifest differs")
    frame_index, witness_id, target_px = _load_targets(args.validation_truth)
    masks = _load_masks(args.mask_manifest, args.object_root.resolve(strict=True), frame_index)

    require(np.array_equal(raw["frame_index"], frame_index), "raw frame order differs")
    require(np.array_equal(gate4b["frame_index"], frame_index), "Gate 4b frame order differs")
    score_maps = np.asarray(raw["final_feature_query_score_map"], dtype=np.float32)
    changed = (
        (np.asarray(raw["assigned_cell_y"]) != np.asarray(gate4b["hard_cell_y"]))
        | (np.asarray(raw["assigned_cell_x"]) != np.asarray(gate4b["hard_cell_x"]))
    )
    pair_counts = pair_distance_counts(gate4b["hard_cell_y"], gate4b["hard_cell_x"])
    target_x, target_y = nearest_r64_cells(target_px)
    union_distance = np.empty((24, EXPECTED_WITNESSES), dtype=np.int64)
    own_distance = np.empty_like(union_distance)
    nearest_candidate_rank = np.empty_like(union_distance)
    spatial_y = np.empty_like(union_distance)
    spatial_x = np.empty_like(union_distance)
    spatial_distance = np.empty((24, EXPECTED_WITNESSES), dtype=np.float64)
    for frame in range(24):
        count = int(raw["candidate_mode_count"][frame])
        candidates_y = np.asarray(raw["candidate_mode_y"][frame, :count], dtype=np.int64)
        candidates_x = np.asarray(raw["candidate_mode_x"][frame, :count], dtype=np.int64)
        for witness in range(EXPECTED_WITNESSES):
            distance = np.maximum(
                np.abs(target_y[frame, witness] - candidates_y),
                np.abs(target_x[frame, witness] - candidates_x),
            )
            nearest = int(np.argmin(distance))
            union_distance[frame, witness] = int(distance[nearest])
            candidate_scores = score_maps[frame, witness, candidates_y, candidates_x]
            nearest_candidate_rank[frame, witness] = score_rank_strictly_greater(
                candidate_scores, nearest
            )
            own_modes = spatial_local_maxima(score_maps[frame, witness])
            own_distance[frame, witness] = int(
                np.max(
                    np.abs(
                        own_modes
                        - np.asarray([target_y[frame, witness], target_x[frame, witness]])
                    ),
                    axis=1,
                ).min()
            )
        spatial_y[frame], spatial_x[frame], spatial_distance[frame] = spatial_candidate_assignment(
            candidates_y, candidates_x, target_px[frame]
        )

    spatial_readout = local_readout_at_cells(score_maps, spatial_y, spatial_x)
    oracle_px = spatial_readout["assigned_local_3x3_prediction_px"]
    oracle_report, oracle_derived = evaluate_predictions(oracle_px, target_px, masks)
    oracle_within_two = oracle_derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    oracle_pass = bool(
        oracle_report["violations"]["wrong_identity_count"] == 0
        and oracle_report["violations"]["collapsed_pair_count"] == 0
        and oracle_report["violations"]["off_object_count"] == 0
        and oracle_within_two.all()
    )
    branch = (
        "candidate_pool_spatially_adequate_failure_is_truth_blind_selection"
        if oracle_pass
        else "candidate_pool_spatial_coverage_insufficient_replace_proposal_rule"
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "CANDIDATE_FAILURE_AUDIT_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        gate4c_changed_from_gate4b=changed,
        union_nearest_candidate_chebyshev_cells=union_distance,
        own_query_nearest_local_maximum_chebyshev_cells=own_distance,
        union_nearest_candidate_score_rank=nearest_candidate_rank,
        spatial_oracle_cell_y=spatial_y,
        spatial_oracle_cell_x=spatial_x,
        spatial_oracle_continuous_grid_distance=spatial_distance,
        spatial_oracle_prediction_px=oracle_px,
        spatial_oracle_material_error_px=oracle_derived["material_error_px"],
        spatial_oracle_within_half_cell=oracle_derived["within_half_cell"],
        spatial_oracle_within_two_cells=oracle_within_two,
        spatial_oracle_identity_correct=oracle_derived["identity_correct"],
        spatial_oracle_on_object=oracle_derived["on_object"],
        spatial_oracle_distinct_pair=oracle_derived["distinct_pair"],
    )
    figure_path = args.output_dir / "01_CANDIDATE_COVERAGE_AND_SPATIAL_ORACLE.png"
    _figure(
        pair_counts,
        union_distance,
        own_distance,
        nearest_candidate_rank,
        oracle_report,
        int(oracle_within_two.sum()),
        figure_path,
    )
    result_path = args.output_dir / "CANDIDATE_FAILURE_AUDIT_RESULT.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "privileged_posthash_candidate_failure_localization",
        "repository_head": repository_head,
        "scratch_result_known_before_formal_audit": True,
        "gate4b_pair_proximity": pair_counts,
        "gate4c_assignment_changes": {
            "changed_event_count": int(changed.sum()),
            "event_count": int(changed.size),
            "changed_by_witness": [int(value) for value in changed.sum(axis=0)],
        },
        "candidate_coverage": {
            "union_nearest_mode_chebyshev_cells": _summary(union_distance),
            "union_within_one_cell_count": int((union_distance <= 1).sum()),
            "own_query_nearest_local_maximum_chebyshev_cells": _summary(own_distance),
            "own_query_within_one_cell_count": int((own_distance <= 1).sum()),
            "nearest_union_candidate_score_rank": _summary(nearest_candidate_rank),
            "nearest_union_candidate_score_rank_top1_count": int((nearest_candidate_rank <= 1).sum()),
            "nearest_union_candidate_score_rank_top3_count": int((nearest_candidate_rank <= 3).sum()),
            "nearest_union_candidate_score_rank_top10_count": int((nearest_candidate_rank <= 10).sum()),
        },
        "privileged_spatial_candidate_oracle": {
            "diagnostic_only_not_deployable": True,
            "candidate_distance_continuous_r64_cells": _summary(spatial_distance),
            "report": oracle_report,
            "within_two_cells_count": int(oracle_within_two.sum()),
            "event_count": int(oracle_within_two.size),
            "passed_spatial_adequacy_branch": oracle_pass,
        },
        "decision": {
            "branch": branch,
            "next_experiment_class": (
                "truth_blind temporal or region-level selection over existing candidate modes"
                if oracle_pass
                else "replacement candidate proposal generator"
            ),
            "encoder_adaptation_authorized": False,
            "gpu_run_authorized": False,
            "operator_authorized": False,
        },
        "interpretation_limits": [
            "The spatial oracle uses certified targets and is not deployable.",
            "Spatial coverage on one hammer wedge does not prove a particular temporal or region-level selector will work.",
            "Candidate score ranks are descriptive over correlated frame-query events.",
        ],
        "sample_scope": {
            "event_count": 240,
            "pair_event_count": 1080,
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
        },
        "bindings": {
            "audit_lock": audit_lock_record,
            "raw_receipt": raw_receipt_record,
            "raw_arrays": file_record(raw_arrays_path),
            "gate4b_raw_arrays": file_record(args.gate4b_raw_arrays),
            "validation_truth": truth_record,
            "mask_manifest": mask_record,
            "derived_arrays": file_record(arrays_path),
            "figure": file_record(figure_path),
        },
        "implementation_sources": {
            "audit": file_record(Path(__file__)),
            "joint_assignment_contract": file_record(
                args.repo_root / "keypoint_net" / "final_feature_joint_assignment.py"
            ),
        },
        "controls": {
            "training_or_weight_update_performed": False,
            "torch_imported": "torch" in sys.modules,
            "laptop_gpu_used": False,
        },
    }
    require(result["controls"]["torch_imported"] is False, "Torch entered the audit")
    _write_json(result_path, result)
    receipt_path = args.output_dir / "CANDIDATE_FAILURE_AUDIT_RECEIPT.json"
    receipt = {
        "schema_version": "joint_assignment_candidate_failure_audit_receipt.v1",
        "repository_head": repository_head,
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "figure": file_record(figure_path),
        "decision_branch": branch,
        "spatial_adequacy_passed": oracle_pass,
        "training_authorized": False,
        "command_argv": list(sys.argv),
    }
    _write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--audit-lock", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--gate4b-raw-arrays", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        receipt = run(parse_args())
    except Exception as exc:
        print(f"CANDIDATE_FAILURE_AUDIT_FAILED: {exc}", file=sys.stderr)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
