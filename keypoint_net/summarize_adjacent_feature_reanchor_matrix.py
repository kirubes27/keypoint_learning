"""Summarize all 24 adjacent learned-feature evaluations without hiding channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_ROLES = 24
EXPECTED_CHANNELS = 10
EXPECTED_EDGES = 180
IMPLEMENTATION_SOURCES = (
    "keypoint_net/summarize_adjacent_feature_reanchor_matrix.py",
    "keypoint_net/evaluate_adjacent_feature_reanchor.py",
)


class AdjacentSummaryError(ValueError):
    """Raised when the adjacent matrix is incomplete or non-comparable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjacentSummaryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {"absolute_path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _distribution(values: list[float], *, unit: str, sample_unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "distribution is empty")
    return {
        "unit": unit,
        "sample_unit": sample_unit,
        "n": int(array.size),
        "descriptive_not_inferential": True,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _plot_heatmaps(records: list[dict[str, Any]], output: Path) -> None:
    feature = np.asarray([[row["channels"][channel]["feature_success"] for channel in range(10)] for row in records])
    detector = np.asarray([[row["channels"][channel]["detector_success"] for channel in range(10)] for row in records])
    labels = [row["role_key"] for row in records]
    fig, axes = plt.subplots(1, 3, figsize=(18, 11), sharey=True)
    images = [
        axes[0].imshow(feature, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis"),
        axes[1].imshow(detector, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis"),
        axes[2].imshow(feature - detector, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm"),
    ]
    axes[0].set_title("adjacent feature success fraction")
    axes[1].set_title("hard detector local success fraction")
    axes[2].set_title("feature minus detector")
    for axis in axes:
        axis.set_xticks(range(10), [f"KP{i}" for i in range(10)], rotation=45)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(images[0], ax=axes[:2], fraction=0.025, pad=0.02, label="fraction of 180 cyclic edges")
    fig.colorbar(images[2], ax=axes[2], fraction=0.05, pad=0.02, label="fraction difference")
    fig.suptitle("Per-role, per-keypoint adjacent material recoverability; descriptive only")
    fig.subplots_adjust(left=0.28, right=0.96, top=0.93, bottom=0.08, wspace=0.12)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_scatter(records: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 8))
    markers = {"task55_clean": "o", "task80_assisted": "s"}
    colors = {"control": "tab:blue", "ocr_zncc": "tab:orange"}
    seen: set[tuple[str, str]] = set()
    for row in records:
        for channel in row["channels"]:
            key = (row["task"], row["arm"])
            label = f"{row['task']} / {row['arm']}" if key not in seen else None
            seen.add(key)
            axis.scatter(
                channel["detector_success"],
                channel["feature_success"],
                s=22,
                alpha=0.65,
                marker=markers[row["task"]],
                color=colors[row["arm"]],
                label=label,
            )
    axis.plot([0, 1], [0, 1], "k--", lw=1.0, label="equal success")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("hard detector local success fraction")
    axis.set_ylabel("adjacent feature success fraction")
    axis.set_title("Each point is one role-keypoint (180 correlated cyclic edges)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    matrix_root = args.matrix_root.resolve(strict=True)
    manifest_record = _file_record(args.manifest.resolve(strict=True))
    manifest = json.loads(Path(manifest_record["absolute_path"]).read_text(encoding="utf-8"))
    roles = manifest.get("roles")
    _require(isinstance(roles, list) and len(roles) == EXPECTED_ROLES, "manifest role inventory differs")
    run_receipt_path = matrix_root / "MATRIX_RUN_RECEIPT.json"
    run_receipt_record = _file_record(run_receipt_path)
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    _require(run_receipt.get("role_count") == EXPECTED_ROLES, "matrix run is incomplete")
    records: list[dict[str, Any]] = []
    implementation_sets: set[tuple[tuple[str, str], ...]] = set()
    for role in roles:
        role_key = str(role["role_key"])
        evaluation_path = matrix_root / role_key / "evaluation_v1/adjacent_feature_evaluation.json"
        evaluation_record = _file_record(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _require(evaluation.get("role", {}).get("role_key") == role_key, f"role differs: {role_key}")
        _require(evaluation.get("training_or_weight_update_performed") is False, f"training flag differs: {role_key}")
        report = evaluation.get("report")
        _require(isinstance(report, Mapping) and len(report.get("channels", [])) == EXPECTED_CHANNELS, f"report differs: {role_key}")
        _require(report.get("self_retrieval_total") == EXPECTED_EDGES * EXPECTED_CHANNELS, f"self total differs: {role_key}")
        implementation_sets.add(tuple(sorted((name, rec["sha256"]) for name, rec in evaluation["implementation_sources"].items())))
        channels = []
        for channel in report["channels"]:
            channels.append(
                {
                    "channel": int(channel["channel"]),
                    "strict": bool(channel["strict_adjacent_local_pass"]),
                    "feature_success": float(channel["adjacent_material_success_fraction_all_edges"]),
                    "detector_success": float(channel["detector_local_success_fraction_all_edges"]),
                    "feature_better_fraction": float(channel["feature_better_than_detector_fraction"]),
                    "repair_count": int(channel["detector_bad_feature_good_count"]),
                    "both_bad_count": int(channel["detector_bad_feature_bad_count"]),
                    "detector_good_feature_bad_count": int(channel["detector_good_feature_bad_count"]),
                    "eligible_edges": int(channel["eligible_on_object_edge_count"]),
                    "feature_error_median_px": float(channel["feature_material_error"]["median"]),
                    "feature_error_max_px": float(channel["feature_material_error"]["maximum"]),
                    "detector_error_median_px": float(channel["detector_local_material_error"]["median"]),
                    "detector_error_max_px": float(channel["detector_local_material_error"]["maximum"]),
                    "adjacent_margin_median": float(channel["adjacent_separated_cosine_margin"]["median"]),
                    "failed_checks": [name for name, passed in channel["checks"].items() if not passed],
                }
            )
        records.append(
            {
                "role_key": role_key,
                "task": role["task"],
                "arm": role["arm"],
                "seed": role["seed"],
                "checkpoint_role": role["checkpoint_role"],
                "evaluation": evaluation_record,
                "strict_pass_count": int(report["strict_pass_count"]),
                "self_retrieval_same_cell_count": int(report["self_retrieval_same_cell_count"]),
                "feature_success_count": int(report["adjacent_material_success_count"]),
                "detector_success_count": int(report["detector_local_success_count"]),
                "channels": channels,
            }
        )
    _require(len(records) == EXPECTED_ROLES, "complete role records are missing")
    _require(len(implementation_sets) == 1, "evaluation implementation source hashes differ across roles")

    flat = [(row, channel) for row in records for channel in row["channels"]]
    failure_counts: Counter[str] = Counter()
    for _, channel in flat:
        failure_counts.update(channel["failed_checks"])
    per_channel = []
    for channel_index in range(EXPECTED_CHANNELS):
        selected = [(row, channel) for row, channel in flat if channel["channel"] == channel_index]
        per_channel.append(
            {
                "channel": channel_index,
                "role_count": len(selected),
                "strict_role_count": sum(channel["strict"] for _, channel in selected),
                "feature_success": _distribution(
                    [channel["feature_success"] for _, channel in selected],
                    unit="fraction of 180 cyclic edges",
                    sample_unit="frozen role",
                ),
                "detector_success": _distribution(
                    [channel["detector_success"] for _, channel in selected],
                    unit="fraction of 180 cyclic edges",
                    sample_unit="frozen role",
                ),
                "feature_minus_detector_success": _distribution(
                    [channel["feature_success"] - channel["detector_success"] for _, channel in selected],
                    unit="fraction difference",
                    sample_unit="frozen role",
                ),
                "repair_count_total": sum(channel["repair_count"] for _, channel in selected),
                "both_bad_count_total": sum(channel["both_bad_count"] for _, channel in selected),
                "eligible_edge_count_total": sum(channel["eligible_edges"] for _, channel in selected),
            }
        )

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, channel in flat:
        groups[f"{row['task']}__{row['arm']}__{row['checkpoint_role']}"] .append((row, channel))
    group_summary = {}
    for key, selected in sorted(groups.items()):
        group_summary[key] = {
            "n_role_channels": len(selected),
            "feature_success": _distribution(
                [channel["feature_success"] for _, channel in selected],
                unit="fraction of 180 cyclic edges",
                sample_unit="role-keypoint",
            ),
            "detector_success": _distribution(
                [channel["detector_success"] for _, channel in selected],
                unit="fraction of 180 cyclic edges",
                sample_unit="role-keypoint",
            ),
            "strict_count": sum(channel["strict"] for _, channel in selected),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    heatmap = args.output_dir / "per_role_channel_success_heatmap.png"
    scatter = args.output_dir / "feature_vs_detector_success.png"
    _plot_heatmaps(records, heatmap)
    _plot_scatter(records, scatter)
    repo_root = args.repo_root.resolve(strict=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "schema_version": "adjacent_feature_reanchor_matrix_summary.v1",
        "artifact_type": "complete_adjacent_feature_reanchor_matrix_summary",
        "inventory": {
            "role_count": len(records),
            "role_channel_count": len(flat),
            "edge_count_per_role_channel": EXPECTED_EDGES,
            "matrix_complete": len(records) == EXPECTED_ROLES,
        },
        "overall": {
            "strict_pass_count": sum(channel["strict"] for _, channel in flat),
            "strict_all_240_pass": all(channel["strict"] for _, channel in flat),
            "self_retrieval_same_cell_count": sum(row["self_retrieval_same_cell_count"] for row in records),
            "self_retrieval_total": EXPECTED_ROLES * EXPECTED_CHANNELS * EXPECTED_EDGES,
            "feature_success_count": sum(row["feature_success_count"] for row in records),
            "detector_success_count": sum(row["detector_success_count"] for row in records),
            "edge_total": EXPECTED_ROLES * EXPECTED_CHANNELS * EXPECTED_EDGES,
            "failure_counts_by_role_channel": dict(sorted(failure_counts.items())),
            "feature_success_fraction_per_role_channel": _distribution(
                [channel["feature_success"] for _, channel in flat],
                unit="fraction of 180 cyclic edges",
                sample_unit="role-keypoint",
            ),
            "detector_success_fraction_per_role_channel": _distribution(
                [channel["detector_success"] for _, channel in flat],
                unit="fraction of 180 cyclic edges",
                sample_unit="role-keypoint",
            ),
            "feature_better_fraction_per_role_channel": _distribution(
                [channel["feature_better_fraction"] for _, channel in flat],
                unit="fraction of 180 cyclic edges",
                sample_unit="role-keypoint",
            ),
            "detector_bad_feature_good_total": sum(channel["repair_count"] for _, channel in flat),
            "detector_bad_feature_bad_total": sum(channel["both_bad_count"] for _, channel in flat),
            "detector_good_feature_bad_total": sum(channel["detector_good_feature_bad_count"] for _, channel in flat),
        },
        "per_channel": per_channel,
        "groups": group_summary,
        "per_role": records,
        "manifest": manifest_record,
        "matrix_run_receipt": run_receipt_record,
        "evaluation_implementation_source_hashes_homogeneous": True,
        "implementation_head": head,
        "implementation_sources": {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES},
        "visuals": {
            "per_role_channel_success_heatmap": _file_record(heatmap),
            "feature_vs_detector_success": _file_record(scatter),
        },
        "statistical_scope": {
            "descriptive_not_inferential": True,
            "sample_units": ["correlated cyclic edge", "role-keypoint", "frozen role"],
            "correlation_caveat": "edges overlap in frames; all roles share one hammer orbit; seeds are descriptive, not population inference",
            "no_sem_or_population_ci": True,
        },
        "training_or_weight_update_performed": False,
    }
    output = args.output_dir / "ADJACENT_FEATURE_REANCHOR_MATRIX_SUMMARY.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "strict_pass_count": summary["overall"]["strict_pass_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
