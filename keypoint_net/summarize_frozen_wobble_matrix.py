"""Aggregate the fixed local frozen-checkpoint forensic matrix.

This command performs no model inference and no training.  It verifies every
input report and raw-array binding, applies the planted-oracle contract to each
fixed channel, cross-checks its classification against the established paired
summary implementation, and writes descriptive tables and matrix visuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    from .plot_frozen_wobble_pair import _load_run, _paired_numeric_summary
except ImportError:
    from plot_frozen_wobble_pair import _load_run, _paired_numeric_summary  # type: ignore


SCHEMA_VERSION = "frozen_wobble_complete_matrix_summary.v1"
EXPECTED_RUN_IMPLEMENTATION_ANCESTOR = "49d1203d18fd4cb66a78f308a2c33944a69c03ca"
EXPECTED_CHANNELS = 10
EXPECTED_FRAMES = 180
TASKS = ("task55_clean", "task80_assisted")
ARMS = ("control", "ocr_zncc")
SEEDS = (42, 43, 44)
ROLES = ("best_model", "final_model")
IMPLEMENTATION_SOURCES = (
    "keypoint_net/summarize_frozen_wobble_matrix.py",
    "keypoint_net/plot_frozen_wobble_pair.py",
)


class MatrixSummaryError(ValueError):
    """Fail-closed matrix binding or semantic error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixSummaryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_RUN_IMPLEMENTATION_ANCESTOR, head],
        cwd=repo_root,
        check=False,
    )
    _require(ancestry.returncode == 0, "matrix summarizer does not descend from the frozen-run implementation")
    cleanliness = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCES],
        cwd=repo_root,
        check=False,
    )
    _require(cleanliness.returncode == 0, "matrix summarizer source differs from its recorded HEAD")
    return {
        "implementation_head": head,
        "required_frozen_run_ancestor_commit": EXPECTED_RUN_IMPLEMENTATION_ANCESTOR,
        "implementation_sources": {
            relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES
        },
    }


def _role_key(task: str, arm: str, seed: int, role: str) -> str:
    return f"{task}__{arm}__seed{seed}__{role}"


def _all_expected_roles() -> set[str]:
    return {
        _role_key(task, arm, seed, role)
        for task in TASKS
        for arm in ARMS
        for seed in SEEDS
        for role in ROLES
    }


def _commit_descends(repo_root: Path, commit: str, ancestor: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, commit],
        cwd=repo_root,
        check=False,
    ).returncode == 0


def _runtime_source_bundle(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = report["bindings"]["implementation_sources"]
    return {
        relative: {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
        for relative, record in sorted(sources.items())
    }


def _single_run_numeric_summary(
    report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    thresholds = report["oracle_envelope_exceedance"]
    hard_thresholds = thresholds["hard_thresholds"]
    topology_thresholds = thresholds["single_gaussian_heatmap_topology_thresholds"]
    activity_thresholds = thresholds["activity_thresholds"]
    grounding_thresholds = thresholds["grounding_and_distinctness_thresholds"]
    phase_thresholds = thresholds["phase_dominance_thresholds"]
    soft = report["soft_coordinate_metrics"]["full_nonseam"]
    hard = report["hard_coordinate_metrics"]["full_nonseam"]
    heat = report["heatmap_topology"]
    raw = arrays["soft_coordinate"]
    _require(raw.shape == (EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), "raw coordinate shape differs")

    category_counts: dict[str, int] = defaultdict(int)
    flag_counts = {
        "sliding": 0,
        "wobbling": 0,
        "both": 0,
        "clean": 0,
        "dead_or_off_object": 0,
        "persistent_duplicate": 0,
        "diffuse_or_multimodal_majority": 0,
        "three_phase_energy_dominant": 0,
        "hard_peak_moves": 0,
    }
    rows: list[dict[str, Any]] = []
    for channel in range(EXPECTED_CHANNELS):
        phase = report["phase_metrics"]["full"]["channels"][channel]
        raw_activity = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(raw[:, channel] - np.mean(raw[:, channel], axis=0, keepdims=True)),
                        axis=-1,
                    )
                )
            )
        )
        pair_rows = arrays["pair_channel_indices"]
        pair_distance = arrays["pairwise_distance_normalized"]
        close_fractions = [
            float(
                np.mean(
                    pair_distance[:, pair_index]
                    < grounding_thresholds["minimum_fixed_channel_pair_distance_normalized"]
                )
            )
            for pair_index, pair in enumerate(pair_rows)
            if channel in (int(pair[0]), int(pair[1]))
        ]
        maximum_close_fraction = max(close_fractions)
        persistent_duplicate = (
            maximum_close_fraction >= grounding_thresholds["persistent_duplicate_close_fraction"]
        )
        active = raw_activity >= activity_thresholds["minimum_raw_orbit_rms_normalized"]
        on_object = report["localization"]["on_object_rate_per_channel"][channel]
        grounded = on_object >= grounding_thresholds["required_on_object_rate"]
        border_distance = report["localization"]["minimum_image_border_distance_px_per_channel"][channel]
        border_safe = border_distance >= grounding_thresholds["minimum_image_border_distance_px"]
        sliding = soft["canonical_rms_per_channel"][channel] > hard_thresholds["canonical_rms_max"]
        hard_jump_rate = heat["hard_jump_rate_per_channel"][channel]
        wobbling = (
            soft["adjacent_plus2_step"]["per_channel"][channel]["maximum"]
            > hard_thresholds["adjacent_step_max"]
            or soft["cyclic_second_difference"]["per_channel"][channel]["maximum"]
            > hard_thresholds["cyclic_second_difference_max"]
            or hard_jump_rate > 0.0
        )
        width_exceedance_rate = float(
            np.mean(arrays["rms_width_cells"][:, channel] > topology_thresholds["rms_width_cells_max"])
        )
        soft_hard_exceedance_rate = float(
            np.mean(
                arrays["soft_hard_distance_cells"][:, channel]
                > topology_thresholds["soft_hard_distance_cells_max"]
            )
        )
        low_margin_rate = float(
            np.mean(
                arrays["separated_peak_r4_logit_margin"][:, channel]
                < topology_thresholds["separated_peak_radius4_logit_margin_min"]
            )
        )
        diffuse_majority = max(width_exceedance_rate, soft_hard_exceedance_rate, low_margin_rate) >= 0.5
        dead_or_off_object = (not active) or (not grounded) or (not border_safe)
        eligible = active and grounded and border_safe and not persistent_duplicate
        if dead_or_off_object:
            primary_category = "dead_or_off_object"
        elif persistent_duplicate:
            primary_category = "persistent_duplicate"
        elif sliding and wobbling:
            primary_category = "both_sliding_and_wobbling"
        elif sliding:
            primary_category = "sliding_only"
        elif wobbling:
            primary_category = "wobbling_only"
        else:
            primary_category = "clean"
        phase_fraction = phase["adjacent_step_energy_removed_by_constant_residue_correction"]
        phase_dominant = phase_fraction >= phase_thresholds["minimum_adjacent_step_energy_fraction"]
        metrics = {
            "soft_canonical_rms_normalized": soft["canonical_rms_per_channel"][channel],
            "soft_canonical_rms_input_pixels": soft["canonical_rms_per_channel"][channel] * 255.5,
            "hard_canonical_rms_normalized": hard["canonical_rms_per_channel"][channel],
            "adjacent_plus2_q99_heatmap_cells": soft["adjacent_plus2_step"]["per_channel"][channel]["q99"] * 31.5,
            "second_difference_q99_heatmap_cells": soft["noncyclic_second_difference"]["per_channel"][channel]["q99"] * 31.5,
            "within_residue_plus6_q99_heatmap_cells": soft["within_residue_plus6_step"]["per_channel"][channel]["q99"] * 31.5,
            "adjacent_energy_removed_by_constant_three_residue_offsets": phase_fraction,
            "position_energy_removed_by_constant_three_residue_offsets": phase["position_energy_removed_by_constant_residue_correction"],
            "maximum_spike_excess_over_fitted_phase_normalized": phase["maximum_spike_excess_over_phase_bound"],
            "hard_jump_rate": hard_jump_rate,
            "on_object_rate": on_object,
            "soft_hard_distance_median_cells": heat["soft_hard_distance_cells"]["per_channel"][channel]["median"],
            "heatmap_width_median_cells": heat["rms_width_cells"]["per_channel"][channel]["median"],
            "separated_peak_logit_margin_median": heat["separated_peak_logit_margin_radius4"]["per_channel"][channel]["median"],
            "raw_orbit_rms_normalized": raw_activity,
            "active": active,
            "grounded_all_frames": grounded,
            "border_safe_all_frames": border_safe,
            "maximum_fixed_pair_close_fraction": maximum_close_fraction,
            "persistent_duplicate": persistent_duplicate,
            "sliding": sliding,
            "wobbling": wobbling,
            "eligible_active_grounded_distinct": eligible,
            "heatmap_width_exceedance_rate": width_exceedance_rate,
            "soft_hard_distance_exceedance_rate": soft_hard_exceedance_rate,
            "low_separated_peak_margin_rate": low_margin_rate,
            "diffuse_or_multimodal_majority": diffuse_majority,
            "primary_category": primary_category,
            "three_phase_energy_dominant": phase_dominant,
        }
        rows.append({"channel": channel, **metrics})
        category_counts[primary_category] += 1
        flag_counts["sliding"] += int(sliding)
        flag_counts["wobbling"] += int(wobbling)
        flag_counts["both"] += int(sliding and wobbling)
        flag_counts["clean"] += int(primary_category == "clean")
        flag_counts["dead_or_off_object"] += int(dead_or_off_object)
        flag_counts["persistent_duplicate"] += int(persistent_duplicate)
        flag_counts["diffuse_or_multimodal_majority"] += int(diffuse_majority)
        flag_counts["three_phase_energy_dominant"] += int(phase_dominant)
        flag_counts["hard_peak_moves"] += int(hard_jump_rate > 0.0)
    return {
        "per_channel": rows,
        "mutually_exclusive_primary_category_counts": dict(category_counts),
        "overlapping_failure_flag_counts": flag_counts,
        "strict_success_contract_pass": category_counts.get("clean", 0) == EXPECTED_CHANNELS,
        "frozen_thresholds": {
            "hard_quality": hard_thresholds,
            "single_gaussian_heatmap_topology": topology_thresholds,
            "activity": activity_thresholds,
            "grounding_and_distinctness": grounding_thresholds,
            "phase_dominance": phase_thresholds,
        },
    }


def _assert_classifier_equivalence(
    paired: Mapping[str, Any],
    singles: Mapping[str, Mapping[str, Any]],
    names: tuple[str, str],
) -> None:
    for channel in range(EXPECTED_CHANNELS):
        for name in names:
            paired_metrics = paired["per_channel"][channel]["arms"][name]
            single_metrics = singles[name]["per_channel"][channel]
            comparable = {key: single_metrics[key] for key in paired_metrics}
            _require(comparable == paired_metrics, "standalone classifier differs from paired classifier")


def _worst(rows: list[dict[str, Any]], key: str, *, maximum: bool) -> dict[str, Any]:
    selected = (max if maximum else min)(rows, key=lambda row: row[key])
    return {"role_key": selected["role_key"], "channel": selected["channel"], "value": selected[key]}


def _group_summaries(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in run_rows:
        for channel in run["summary"]["per_channel"]:
            grouped[(run["task"], run["arm"], run["checkpoint_role"])].append(
                {"role_key": run["role_key"], **channel}
            )
    output = []
    for (task, arm, role), rows in sorted(grouped.items()):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["primary_category"]] += 1
        output.append(
            {
                "task": task,
                "arm": arm,
                "checkpoint_role": role,
                "available_checkpoint_roles": len({row["role_key"] for row in rows}),
                "channel_observations": len(rows),
                "primary_category_counts": dict(counts),
                "worst_soft_canonical_rms_normalized": _worst(rows, "soft_canonical_rms_normalized", maximum=True),
                "worst_adjacent_plus2_q99_heatmap_cells": _worst(rows, "adjacent_plus2_q99_heatmap_cells", maximum=True),
                "worst_second_difference_q99_heatmap_cells": _worst(rows, "second_difference_q99_heatmap_cells", maximum=True),
                "worst_hard_jump_rate": _worst(rows, "hard_jump_rate", maximum=True),
                "minimum_on_object_rate": _worst(rows, "on_object_rate", maximum=False),
                "three_phase_dominant_channels": sum(row["three_phase_energy_dominant"] for row in rows),
                "diffuse_or_multimodal_majority_channels": sum(row["diffuse_or_multimodal_majority"] for row in rows),
                "hard_peak_moves_channels": sum(row["hard_jump_rate"] > 0.0 for row in rows),
                "descriptive_not_inferential": True,
            }
        )
    return output


def _paired_record(
    task: str,
    seed: int,
    role: str,
    reports: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    singles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = ("control", "ocr_zncc")
    paired = _paired_numeric_summary(
        {name: reports[name] for name in names},
        {name: arrays[name] for name in names},
        list(names),
    )
    _assert_classifier_equivalence(paired, singles, names)
    per_channel = []
    for channel in range(EXPECTED_CHANNELS):
        control = singles["control"]["per_channel"][channel]
        ocr = singles["ocr_zncc"]["per_channel"][channel]
        per_channel.append(
            {
                "channel": channel,
                "ocr_minus_control": {
                    key: ocr[key] - control[key]
                    for key in (
                        "soft_canonical_rms_normalized",
                        "adjacent_plus2_q99_heatmap_cells",
                        "second_difference_q99_heatmap_cells",
                        "hard_jump_rate",
                        "on_object_rate",
                    )
                },
                "control_primary_category": control["primary_category"],
                "ocr_primary_category": ocr["primary_category"],
            }
        )
    return {
        "task": task,
        "seed": seed,
        "checkpoint_role": role,
        "control_epoch": reports["control"]["checkpoint_epoch"],
        "ocr_epoch": reports["ocr_zncc"]["checkpoint_epoch"],
        "same_epoch": reports["control"]["checkpoint_epoch"] == reports["ocr_zncc"]["checkpoint_epoch"],
        "ocr_improved_channel_counts": {
            "lower_canonical_rms": sum(
                row["ocr_minus_control"]["soft_canonical_rms_normalized"] < 0 for row in per_channel
            ),
            "lower_adjacent_q99": sum(
                row["ocr_minus_control"]["adjacent_plus2_q99_heatmap_cells"] < 0 for row in per_channel
            ),
            "lower_second_difference_q99": sum(
                row["ocr_minus_control"]["second_difference_q99_heatmap_cells"] < 0 for row in per_channel
            ),
            "lower_hard_jump_rate": sum(
                row["ocr_minus_control"]["hard_jump_rate"] < 0 for row in per_channel
            ),
            "higher_on_object_rate": sum(
                row["ocr_minus_control"]["on_object_rate"] > 0 for row in per_channel
            ),
        },
        "per_channel": per_channel,
        "descriptive_not_inferential": True,
    }


def _matrix_visual(run_rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    labels = [run["role_key"] for run in run_rows]
    rms_threshold = run_rows[0]["summary"]["frozen_thresholds"]["hard_quality"]["canonical_rms_max"]
    step_threshold = run_rows[0]["summary"]["frozen_thresholds"]["hard_quality"]["adjacent_step_max"] * 31.5
    second_threshold = run_rows[0]["summary"]["frozen_thresholds"]["hard_quality"]["cyclic_second_difference_max"] * 31.5
    rms = np.asarray(
        [[row["soft_canonical_rms_normalized"] / rms_threshold for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    adjacent = np.asarray(
        [[row["adjacent_plus2_q99_heatmap_cells"] / step_threshold for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    second = np.asarray(
        [[row["second_difference_q99_heatmap_cells"] / second_threshold for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    hard_jump = np.asarray(
        [[row["hard_jump_rate"] for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    phase = np.asarray(
        [[row["adjacent_energy_removed_by_constant_three_residue_offsets"] for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    fig, axes = plt.subplots(1, 5, figsize=(27, 10), sharey=True)
    panels = (
        (np.log10(np.maximum(rms, 1e-12)), "log10(drift / clean limit)", "coolwarm", -1.0, 2.0),
        (np.log10(np.maximum(adjacent, 1e-12)), "log10(adjacent q99 / clean max)", "coolwarm", -1.0, 2.0),
        (np.log10(np.maximum(second, 1e-12)), "log10(wobble q99 / clean max)", "coolwarm", -1.0, 2.0),
        (hard_jump, "hard-peak jump rate", "magma", 0.0, 1.0),
        (phase, "energy explained by frame mod 3", "viridis", 0.0, 1.0),
    )
    for index, (values, title, cmap, vmin, vmax) in enumerate(panels):
        image = axes[index].imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[index].set_title(title)
        axes[index].set_xticks(range(EXPECTED_CHANNELS), [f"KP{x}" for x in range(EXPECTED_CHANNELS)], rotation=45)
        fig.colorbar(image, ax=axes[index], fraction=0.046, pad=0.03)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    fig.suptitle(
        "Frozen 180-frame matrix: every square is one fixed keypoint channel\n"
        "Ratios above 1 fail the planted clean envelope; q99 panels are descriptive tails and the strict contract uses maxima.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return _file_record(output_path)


def _category_visual(run_rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    categories = (
        "clean",
        "sliding_only",
        "wobbling_only",
        "both_sliding_and_wobbling",
        "dead_or_off_object",
        "persistent_duplicate",
    )
    codes = {name: index for index, name in enumerate(categories)}
    values = np.asarray(
        [[codes[row["primary_category"]] for row in run["summary"]["per_channel"]] for run in run_rows]
    )
    colors = ("#2CA02C", "#1F77B4", "#FFBF00", "#D62728", "#7F7F7F", "#9467BD")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(13, 9))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(EXPECTED_CHANNELS), [f"KP{x}" for x in range(EXPECTED_CHANNELS)])
    ax.set_yticks(range(len(run_rows)), [run["role_key"] for run in run_rows], fontsize=7)
    ax.set_title("Strict per-keypoint result; grey cannot count as clean")
    colorbar = fig.colorbar(image, ax=ax, ticks=range(len(categories)), fraction=0.035, pad=0.02)
    colorbar.ax.set_yticklabels(categories)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return _file_record(output_path)


def run(matrix_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    implementation = _implementation_binding(repo_root)
    report_paths = sorted(matrix_root.resolve(strict=True).glob("*/forensic_report.json"))
    all_expected = _all_expected_roles()
    _require(len(report_paths) == len(all_expected), "available report count differs from the full protocol inventory")

    loaded: dict[str, dict[str, Any]] = {}
    observed_roles: set[str] = set()
    common_thresholds: Mapping[str, Any] | None = None
    common_runtime_sources: Mapping[str, Any] | None = None
    run_implementation_heads: set[str] = set()
    for report_path in report_paths:
        run_dir = report_path.parent
        report, arrays = _load_run(run_dir)
        _require(report["content_hash_sha256"] == _content_hash(report), "forensic report content hash differs")
        run_head = report["bindings"]["implementation_head"]
        _require(
            _commit_descends(repo_root, run_head, EXPECTED_RUN_IMPLEMENTATION_ANCESTOR),
            "frozen-run implementation does not descend from the bound forensic implementation",
        )
        run_implementation_heads.add(run_head)
        runtime_sources = _runtime_source_bundle(report)
        if common_runtime_sources is None:
            common_runtime_sources = runtime_sources
        _require(
            runtime_sources == common_runtime_sources,
            "frozen roles used different runtime evaluator source bytes",
        )
        role_key = _role_key(report["recipe"], report["arm"], int(report["seed"]), report["checkpoint_role"])
        _require(run_dir.name == role_key, "run directory and bound role key differ")
        _require(role_key not in observed_roles, "duplicate role in matrix")
        observed_roles.add(role_key)
        summary = _single_run_numeric_summary(report, arrays)
        if common_thresholds is None:
            common_thresholds = summary["frozen_thresholds"]
        _require(summary["frozen_thresholds"] == common_thresholds, "matrix roles use different frozen thresholds")
        loaded[role_key] = {
            "role_key": role_key,
            "task": report["recipe"],
            "arm": report["arm"],
            "seed": int(report["seed"]),
            "checkpoint_role": report["checkpoint_role"],
            "checkpoint_epoch": int(report["checkpoint_epoch"]),
            "report": report,
            "arrays": arrays,
            "summary": summary,
            "bindings": {
                "forensic_report": _file_record(report_path),
                "raw_arrays": _file_record(run_dir / "raw_forensic_arrays.npz"),
            },
        }
    _require(observed_roles == all_expected, "observed roles differ from the full protocol inventory")

    paired_rows = []
    for task in TASKS:
        for seed in SEEDS:
            for role in ROLES:
                control_key = _role_key(task, "control", seed, role)
                ocr_key = _role_key(task, "ocr_zncc", seed, role)
                if control_key not in loaded or ocr_key not in loaded:
                    continue
                by_arm = {"control": loaded[control_key], "ocr_zncc": loaded[ocr_key]}
                paired_rows.append(
                    _paired_record(
                        task,
                        seed,
                        role,
                        {arm: row["report"] for arm, row in by_arm.items()},
                        {arm: row["arrays"] for arm, row in by_arm.items()},
                        {arm: row["summary"] for arm, row in by_arm.items()},
                    )
                )

    run_rows = [loaded[key] for key in sorted(loaded)]
    output_dir.mkdir(parents=True, exist_ok=False)
    visuals = {
        "metric_matrix": _matrix_visual(run_rows, output_dir / "01_metric_matrix.png"),
        "strict_category_matrix": _category_visual(run_rows, output_dir / "02_strict_category_matrix.png"),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "source_bound_frozen_checkpoint_complete_matrix_summary",
        "training_or_weight_update_performed": False,
        "implementation": implementation,
        "matrix_root": str(matrix_root.resolve(strict=True)),
        "frozen_run_implementation": {
            "observed_heads": sorted(run_implementation_heads),
            "required_ancestor_commit": EXPECTED_RUN_IMPLEMENTATION_ANCESTOR,
            "identical_runtime_source_bytes_across_all_roles": True,
            "runtime_sources": common_runtime_sources,
        },
        "inventory": {
            "full_protocol_roles": len(all_expected),
            "available_local_roles": len(observed_roles),
            "missing_roles": sorted(all_expected - observed_roles),
            "matrix_complete": observed_roles == all_expected,
        },
        "frozen_thresholds": common_thresholds,
        "per_run": [
            {
                key: row[key]
                for key in (
                    "role_key",
                    "task",
                    "arm",
                    "seed",
                    "checkpoint_role",
                    "checkpoint_epoch",
                    "bindings",
                    "summary",
                )
            }
            for row in run_rows
        ],
        "task_arm_role_groups": _group_summaries(run_rows),
        "paired_control_ocr_comparisons": paired_rows,
        "visuals": visuals,
        "strict_matrix_success": all(row["summary"]["strict_success_contract_pass"] for row in run_rows),
        "classifier_cross_check": {
            "paired_roles_checked": len(paired_rows) * 2,
            "channels_per_role": EXPECTED_CHANNELS,
            "standalone_equals_established_paired_classifier": True,
        },
        "statistical_scope": {
            "statistics": "per-channel RMS, empirical q99, deterministic maxima, rates, and phase energy fractions",
            "sample_unit": "fixed keypoint channel over one 180-frame rendered orbit",
            "n_frames_per_role": EXPECTED_FRAMES,
            "n_channels_per_role": EXPECTED_CHANNELS,
            "available_checkpoint_roles": len(observed_roles),
            "descriptive_not_inferential": True,
            "correlation_caveat": "frames and overlapping differences are correlated; no naive SEM, CI, or population claim is made",
            "error_bars_or_uncertainty_bands": "none",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    result_path = output_dir / "FROZEN_LOCAL_MATRIX_SUMMARY.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "summary": _file_record(result_path),
        "visuals": visuals,
        "available_roles": len(observed_roles),
        "missing_roles": len(all_expected - observed_roles),
        "paired_comparisons": len(paired_rows),
        "strict_matrix_success": result["strict_matrix_success"],
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.matrix_root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
