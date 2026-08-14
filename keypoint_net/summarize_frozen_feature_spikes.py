"""Summarize one exact complete frozen feature-spike forensic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .summarize_frozen_wobble_matrix import _content_hash
except ImportError:
    from summarize_frozen_wobble_matrix import _content_hash  # type: ignore


SCHEMA_VERSION = "frozen_feature_spike_summary.v1"
EXPECTED_SOURCE_SCHEMA = "frozen_feature_spike_forensics.v1_1"
EXPECTED_ROLES = 24
EXPECTED_VISUALS = 48


class FeatureSummaryError(ValueError):
    """Fail-closed source or summary semantic error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureSummaryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "summary values must be finite and non-empty")
    return {
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def summarize_grounded_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grounded = [row for row in rows if bool(row["grounded_physical_edge"])]
    result: dict[str, Any] = {
        "selected_edge_count": len(rows),
        "grounded_physical_edge_count": len(grounded),
        "grounded_physical_edge_fraction": len(grounded) / len(rows) if rows else None,
    }
    if not grounded:
        result["grounded_metrics"] = None
        return result
    physical_outscores = np.asarray(
        [float(row["detector_minus_physical_similarity"]) < 0.0 for row in grounded]
    )
    feature_closer = np.asarray(
        [
            float(row["object_argmax_distance_to_physical_cells"])
            < float(row["detector_distance_to_physical_cells"])
            for row in grounded
        ]
    )
    result["grounded_metrics"] = {
        "physical_target_mask_percentile": numeric_summary(
            float(row["physical_target_mask_percentile"]) for row in grounded
        ),
        "detector_minus_physical_similarity": numeric_summary(
            float(row["detector_minus_physical_similarity"]) for row in grounded
        ),
        "feature_argmax_distance_to_physical_cells": numeric_summary(
            float(row["object_argmax_distance_to_physical_cells"]) for row in grounded
        ),
        "detector_distance_to_physical_cells": numeric_summary(
            float(row["detector_distance_to_physical_cells"]) for row in grounded
        ),
        "object_best_minus_physical_similarity": numeric_summary(
            float(row["object_best_minus_physical_similarity"]) for row in grounded
        ),
        "object_top1_top2_margin": numeric_summary(
            float(row["object_top1_top2_margin"]) for row in grounded
        ),
        "physical_similarity_exceeds_detector_fraction": float(np.mean(physical_outscores)),
        "feature_argmax_closer_than_detector_fraction": float(np.mean(feature_closer)),
    }
    return result


def _group_rows(edges: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        for basis in ("hard_peak", "soft_coordinate"):
            grouped[
                (
                    str(edge["task"]),
                    str(edge["arm"]),
                    str(edge["checkpoint_role"]),
                    str(edge["condition"]),
                    basis,
                )
            ].append(edge[basis])
    output = []
    for key, rows in sorted(grouped.items()):
        task, arm, role, condition, basis = key
        output.append(
            {
                "task": task,
                "arm": arm,
                "checkpoint_role": role,
                "condition": condition,
                "basis": basis,
                **summarize_grounded_rows(rows),
                "descriptive_not_inferential": True,
            }
        )
    return output


def _verify_visuals(source: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for role_key, role_visuals in sorted(source["visuals"].items()):
        for reason, expected in sorted(role_visuals.items()):
            live = _file_record(Path(expected["absolute_path"]))
            _require(live == expected, f"visual file differs for {role_key} {reason}")
            records.append({"role_key": role_key, "reason": reason, "file": live})
    _require(len(records) == EXPECTED_VISUALS, "visual inventory is not exactly 48 files")
    return {"count": len(records), "records": records}


def _plot_distance_scatter(edges: list[Mapping[str, Any]], output_path: Path) -> None:
    groups = (
        ("task55_clean", "control"),
        ("task55_clean", "ocr_zncc"),
        ("task80_assisted", "control"),
        ("task80_assisted", "ocr_zncc"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), sharex=True, sharey=True)
    maximum = 0.0
    arrays = {}
    for task, arm in groups:
        rows = [
            edge["hard_peak"]
            for edge in edges
            if edge["task"] == task
            and edge["arm"] == arm
            and edge["condition"] == "spike"
            and edge["hard_peak"]["grounded_physical_edge"]
        ]
        detector = np.asarray([row["detector_distance_to_physical_cells"] for row in rows])
        feature = np.asarray([row["object_argmax_distance_to_physical_cells"] for row in rows])
        arrays[(task, arm)] = (detector, feature)
        maximum = max(maximum, float(np.max(detector)), float(np.max(feature)))
    limit = np.ceil(maximum / 5.0) * 5.0
    for ax, (task, arm) in zip(axes.flat, groups):
        detector, feature = arrays[(task, arm)]
        better = feature < detector
        ax.scatter(detector[~better], feature[~better], s=24, alpha=0.7, color="#D95F02", label="feature no closer")
        ax.scatter(detector[better], feature[better], s=24, alpha=0.7, color="#1B9E77", label="feature closer")
        ax.plot((0, limit), (0, limit), color="black", lw=1, ls="--")
        ax.set_title(f"{task.replace('_', ' ')} | {arm.replace('_', ' ')}\nn={len(detector)} grounded spike edges")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("detector error from physical point (64-grid cells)")
    for ax in axes[:, 0]:
        ax.set_ylabel("feature-map best-match error (64-grid cells)")
    fig.suptitle(
        "Frozen encoder versus keypoint detector at selected wobble spikes\n"
        "Below diagonal: feature map contains a closer candidate. Descriptive correlated edges; no uncertainty band.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_rank_boxplot(edges: list[Mapping[str, Any]], output_path: Path) -> None:
    groups = (
        ("task55_clean", "control"),
        ("task55_clean", "ocr_zncc"),
        ("task80_assisted", "control"),
        ("task80_assisted", "ocr_zncc"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharey=True)
    for ax, (task, arm) in zip(axes.flat, groups):
        arrays = []
        labels = []
        ns = []
        for condition in ("spike", "low_wobble_control"):
            rows = [
                edge["hard_peak"]
                for edge in edges
                if edge["task"] == task
                and edge["arm"] == arm
                and edge["condition"] == condition
                and edge["hard_peak"]["grounded_physical_edge"]
            ]
            arrays.append(np.asarray([row["physical_target_mask_percentile"] for row in rows]))
            labels.append("spike" if condition == "spike" else "calm control")
            ns.append(len(rows))
        ax.boxplot(arrays, tick_labels=labels, showfliers=True, whis=(0, 100))
        ax.set_title(
            f"{task.replace('_', ' ')} | {arm.replace('_', ' ')}\n"
            f"n={ns[0]} spike, {ns[1]} calm grounded edges"
        )
        ax.set_ylim(0, 1.01)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("physical target percentile among hammer feature cells")
    fig.suptitle(
        "How highly the frozen encoder ranks the physical next location\n"
        "Box=quartiles, line=median, whiskers=observed min/max; descriptive correlated edges.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(source_path: Path, expected_sha256: str, output_dir: Path) -> dict[str, Any]:
    source_record = _file_record(source_path)
    _require(source_record["sha256"] == expected_sha256, "source SHA-256 differs")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _require(source["schema_version"] == EXPECTED_SOURCE_SCHEMA, "source schema differs")
    _require(source["content_hash_sha256"] == _content_hash(source), "source content hash differs")
    _require(source["training_or_weight_update_performed"] is False, "source performed training")
    _require(len(source["role_records"]) == EXPECTED_ROLES, "source does not contain 24 roles")
    edges = source["edges"]
    _require(len(edges) > 0, "source contains no edges")
    visual_inventory = _verify_visuals(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    distance_path = output_dir / "01_grounded_spike_detector_vs_feature_distance.png"
    rank_path = output_dir / "02_grounded_physical_feature_rank.png"
    _plot_distance_scatter(edges, distance_path)
    _plot_rank_boxplot(edges, rank_path)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "bound_descriptive_summary_of_complete_frozen_feature_spike_forensics",
        "source": source_record,
        "expected_source_sha256": expected_sha256,
        "implementation_head": source["implementation"]["implementation_head"],
        "training_or_weight_update_performed": False,
        "inventory": {
            "checkpoint_roles": len(source["role_records"]),
            "selected_edges": len(edges),
            "visuals": visual_inventory,
        },
        "grouped_grounded_descriptives": _group_rows(edges),
        "plots": {
            "detector_vs_feature_distance": _file_record(distance_path),
            "physical_feature_rank": _file_record(rank_path),
        },
        "statistical_scope": {
            "sample_unit": "selected adjacent edge within one correlated orbit/checkpoint role",
            "descriptive_not_inferential": True,
            "correlation_caveat": "incoming/outgoing edges overlap, spike/control selections share orbits, and checkpoint roles share training runs",
            "error_bars_or_uncertainty_bands": "none",
            "boxplot_definition": "box is q25-q75, centre line is median, whiskers are observed minimum and maximum, points are not independent",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    result_path = output_dir / "FROZEN_FEATURE_SPIKE_SUMMARY.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "summary": _file_record(result_path),
        "plots": result["plots"],
        "checkpoint_roles": len(source["role_records"]),
        "selected_edges": len(edges),
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.expected_sha256, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
