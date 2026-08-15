"""Create visual and numeric paired evidence from two frozen forensic runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .frozen_wobble_classification import soft_coordinate_failure_flags
    from .frozen_wobble_forensics import rotate_points
except ImportError:
    from frozen_wobble_classification import soft_coordinate_failure_flags  # type: ignore
    from frozen_wobble_forensics import rotate_points  # type: ignore


PALETTE = [
    "#FF0000", "#00AA00", "#0000FF", "#C8A900", "#FF00FF",
    "#00AAAA", "#FFA500", "#800080", "#008000", "#000080",
]
RESIDUE_COLORS = ("#E41A1C", "#377EB8", "#4DAF4A")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = run_dir / "forensic_report.json"
    arrays_path = run_dir / "raw_forensic_arrays.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report["bindings"]["raw_arrays"]["sha256"]
    if _sha256(arrays_path) != expected:
        raise ValueError(f"raw-array hash differs for {run_dir}")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return report, arrays


def _save_figure(fig: plt.Figure, path: Path) -> dict[str, Any]:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"absolute_path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _equal_limits(values: list[np.ndarray], *, minimum_half_width: float = 0.12) -> tuple[float, float, float, float]:
    stacked = np.concatenate([value.reshape(-1, 2) for value in values], axis=0)
    x_mid = 0.5 * (float(np.min(stacked[:, 0])) + float(np.max(stacked[:, 0])))
    y_mid = 0.5 * (float(np.min(stacked[:, 1])) + float(np.max(stacked[:, 1])))
    half = max(
        minimum_half_width,
        0.55 * float(np.max(stacked[:, 0]) - np.min(stacked[:, 0])),
        0.55 * float(np.max(stacked[:, 1]) - np.min(stacked[:, 1])),
    )
    return x_mid - half, x_mid + half, y_mid - half, y_mid + half


def _trajectory_overview(
    names: list[str], arrays_by_name: dict[str, dict[str, np.ndarray]], output: Path
) -> dict[str, Any]:
    canonical_limits = _equal_limits(
        [arrays_by_name[name]["soft_canonical_coordinate"] for name in names],
        minimum_half_width=0.45,
    )
    kp2_limits = _equal_limits(
        [arrays_by_name[name]["soft_canonical_coordinate"][:, 2] for name in names],
        minimum_half_width=0.18,
    )
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 5.4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        arrays = arrays_by_name[name]
        raw = arrays["soft_coordinate"]
        canonical = arrays["soft_canonical_coordinate"]
        frames = arrays["frame_index"].astype(int)
        for channel in range(raw.shape[1]):
            axes[row, 0].plot(raw[:, channel, 0], raw[:, channel, 1], color=PALETTE[channel], lw=0.8, alpha=0.72)
            axes[row, 0].scatter(raw[::6, channel, 0], raw[::6, channel, 1], color=PALETTE[channel], s=5, alpha=0.65)
            axes[row, 1].plot(canonical[:, channel, 0], canonical[:, channel, 1], color=PALETTE[channel], lw=0.8, alpha=0.72)
        for residue in range(3):
            keep = frames % 3 == residue
            axes[row, 2].plot(
                canonical[keep, 2, 0], canonical[keep, 2, 1],
                color=RESIDUE_COLORS[residue], lw=1.2, alpha=0.75,
                label=f"frame mod 3 = {residue}",
            )
            axes[row, 2].scatter(
                canonical[keep, 2, 0], canonical[keep, 2, 1],
                color=RESIDUE_COLORS[residue], s=10, alpha=0.72,
            )
        axes[row, 0].set_title(f"{name}: raw image trajectories (KP2 is blue)")
        axes[row, 1].set_title(f"{name}: physically de-rotated trajectories")
        axes[row, 2].set_title(f"{name}: KP2 de-rotated, coloured by frame group")
        axes[row, 0].set_xlim(-1.03, 1.03)
        axes[row, 0].set_ylim(1.03, -1.03)
        axes[row, 1].set_xlim(canonical_limits[0], canonical_limits[1])
        axes[row, 1].set_ylim(canonical_limits[3], canonical_limits[2])
        axes[row, 2].set_xlim(kp2_limits[0], kp2_limits[1])
        axes[row, 2].set_ylim(kp2_limits[3], kp2_limits[2])
        axes[row, 2].legend(loc="best", fontsize=8)
        for ax in axes[row]:
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            ax.set_xlabel("normalized x")
            ax.set_ylabel("normalized y (image-down)")
    fig.suptitle(
        "Full 180-frame orbit; lines connect displayed +2° frames\n"
        "Visual evidence is descriptive. No uncertainty band: all frames belong to one correlated orbit.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output)


def _kp2_timeseries(
    names: list[str], arrays_by_name: dict[str, dict[str, np.ndarray]], reports: dict[str, dict[str, Any]], output: Path
) -> dict[str, Any]:
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    arm_colors = ("#333333", "#D55E00")
    for arm_index, name in enumerate(names):
        arrays = arrays_by_name[name]
        frame = arrays["frame_index"]
        canonical = arrays["soft_canonical_coordinate"][:, 2]
        hard_canonical = arrays["hard_canonical_coordinate"][:, 2]
        adjacent = np.linalg.norm(np.diff(canonical, axis=0), axis=-1)
        within = np.linalg.norm(canonical[3:] - canonical[:-3], axis=-1)
        second = np.linalg.norm(canonical[2:] - 2.0 * canonical[1:-1] + canonical[:-2], axis=-1)
        hard_adjacent = np.linalg.norm(np.diff(hard_canonical, axis=0), axis=-1)
        axes[0].plot(frame, canonical[:, 0], color=arm_colors[arm_index], lw=1.2, label=f"{name} x")
        axes[0].plot(frame, canonical[:, 1], color=arm_colors[arm_index], lw=1.0, ls="--", label=f"{name} y")
        axes[1].plot(frame[1:], adjacent * 31.5, color=arm_colors[arm_index], lw=1.0, label=name)
        axes[2].plot(frame[3:], within * 31.5, color=arm_colors[arm_index], lw=1.0, label=name)
        axes[3].plot(frame[1:-1], second * 31.5, color=arm_colors[arm_index], lw=1.0, label=name)
        axes[4].plot(frame[1:], hard_adjacent * 31.5, color=arm_colors[arm_index], lw=1.0, label=name)
    threshold = float(reports[names[0]]["oracle_envelope_exceedance"]["hard_thresholds"]["adjacent_step_max"]) * 31.5
    axes[4].axhline(threshold, color="#0072B2", ls=":", lw=1.5, label="clean r64 hard-grid maximum")
    titles = (
        "KP2 de-rotated coordinates",
        "KP2 adjacent displayed-frame step (+2°)",
        "KP2 within-training-chain step (+6°)",
        "KP2 local wobble: second difference",
        "KP2 hard-peak adjacent step (+2°)",
    )
    ylabels = ("normalized coordinate", "heatmap cells", "heatmap cells", "heatmap cells", "heatmap cells")
    for index, ax in enumerate(axes):
        ax.set_title(titles[index])
        ax.set_ylabel(ylabels[index])
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        for boundary in (24, 27, 177):
            ax.axvline(boundary, color="#888888", lw=0.8, alpha=0.7)
    axes[-1].set_xlabel("rendered frame index (2° per frame)")
    fig.suptitle(
        "Task80 seed42 best checkpoint, blue KP2\n"
        "Vertical lines mark holdout/guard/train boundaries. Traces are descriptive over one correlated orbit.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output)


def _heatmap_traces(
    names: list[str], arrays_by_name: dict[str, dict[str, np.ndarray]], output: Path
) -> dict[str, Any]:
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    colors = ("#333333", "#D55E00")
    keys = (
        "soft_hard_distance_cells",
        "rms_width_cells",
        "separated_peak_r4_logit_margin",
        "entropy_nats",
    )
    labels = ("soft-to-hard distance (cells)", "heatmap RMS width (cells)", "top1 - separated top2 logit", "entropy (nats)")
    for arm_index, name in enumerate(names):
        arrays = arrays_by_name[name]
        frame = arrays["frame_index"]
        for ax, key in zip(axes, keys):
            ax.plot(frame, arrays[key][:, 2], color=colors[arm_index], lw=1.0, label=name)
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        for boundary in (24, 27, 177):
            ax.axvline(boundary, color="#888888", lw=0.8, alpha=0.7)
    axes[0].set_title("KP2 heatmap/readout evidence")
    axes[-1].set_xlabel("rendered frame index")
    fig.suptitle("Per-frame descriptive heatmap statistics; no independence or inferential error band assumed", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output)


def _frame_image(data_root: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    object_root = data_root / "train" / "engineers_hammer_vray"
    with Image.open(object_root / "frames" / "a" / f"img_{frame:04d}.png") as handle:
        image = np.asarray(handle.convert("RGB"))
    with Image.open(object_root / "masks" / "a" / f"mask_{frame:04d}.png") as handle:
        mask = np.asarray(handle.convert("L")) > 0
    return image, mask


def _overlay_points(ax: plt.Axes, arrays: dict[str, np.ndarray], frame: int, *, heatmap: bool) -> None:
    soft = arrays["soft_coordinate"][frame, 2]
    hard = arrays["hard_coordinate"][frame, 2]
    if frame > 0:
        expected = rotate_points(arrays["soft_coordinate"][frame - 1, 2][None], 2.0)[0]
    else:
        expected = rotate_points(arrays["soft_coordinate"][-1, 2][None], 2.0)[0]
    size = 70 if not heatmap else 55
    ax.scatter(soft[0], soft[1], s=size, marker="o", facecolors="none", edgecolors="#00FFFF", linewidths=1.8, label="soft")
    ax.scatter(hard[0], hard[1], s=size, marker="+", color="#FFFF00", linewidths=1.8, label="hard")
    ax.scatter(expected[0], expected[1], s=size, marker="x", color="#FFFFFF", linewidths=1.8, label="previous +2°")


def _spike_montage(
    name: str,
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    data_root: Path,
    output: Path,
) -> dict[str, Any]:
    spikes = [int(row["centre_frame_index"]) for row in report["spike_frames"]["soft"][2]]
    fig, axes = plt.subplots(len(spikes), 6, figsize=(19, 3.35 * len(spikes)), squeeze=False)
    for row_index, centre in enumerate(spikes):
        frames = (centre - 1, centre, centre + 1)
        for column, frame in enumerate(frames):
            image, mask = _frame_image(data_root, frame)
            ax = axes[row_index, column]
            ax.imshow(image, extent=(-1, 1, 1, -1))
            # Explicit axes keep row 0 at normalized y=-1.  Passing ``extent``
            # directly to contour flipped the audit outline vertically even
            # though the underlying localization arrays were correct.
            mask_axis = np.linspace(-1.0, 1.0, mask.shape[0])
            ax.contour(
                mask_axis,
                mask_axis,
                mask.astype(float),
                levels=[0.5],
                colors=["white"],
                linewidths=0.4,
            )
            _overlay_points(ax, arrays, frame, heatmap=False)
            ax.set_title(f"RGB frame {frame}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(1, -1)
            ax.axis("off")
        for offset, frame in enumerate(frames):
            ax = axes[row_index, 3 + offset]
            ax.imshow(arrays["logits"][frame, 2], cmap="magma", extent=(-1, 1, 1, -1), interpolation="nearest")
            _overlay_points(ax, arrays, frame, heatmap=True)
            ax.set_title(f"KP2 heatmap frame {frame}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(1, -1)
            ax.axis("off")
        magnitude = next(row["canonical_second_difference"] for row in report["spike_frames"]["soft"][2] if int(row["centre_frame_index"]) == centre)
        axes[row_index, 0].text(
            -0.98, 0.92, f"spike={magnitude * 31.5:.2f} cells",
            color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.930))
    fig.suptitle(
        f"{name}: five largest non-seam KP2 wobble frames\n"
        "Cyan circle = soft point; yellow + = hard peak; white x = previous point physically rotated by +2°",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    return _save_figure(fig, output)


def _paired_numeric_summary(
    reports: dict[str, dict[str, Any]],
    arrays_by_name: dict[str, dict[str, np.ndarray]],
    names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact_type": "paired_task80_seed42_frozen_checkpoint_forensic_summary",
        "arms": names,
        "checkpoint_epoch": {name: reports[name]["checkpoint_epoch"] for name in names},
        "training_or_weight_update_performed": False,
        "comparison_status": "descriptive frozen-checkpoint evidence, not a population inference",
        "per_channel": [],
        "success_contract": {
            "quality_envelope": "worst-case deterministic r64 quantized clean-rotation oracle",
            "grounding": "all 180 nearest pixels must lie inside the bound object mask",
            "activity": "raw orbit RMS must reach the planted half-grid-cell rotating oracle",
            "distinctness": "no fixed-channel pair may remain within one r64 cell for at least half the orbit",
            "fixed_channel_identity_primary": True,
            "averages_cannot_hide_a_failed_channel": True,
        },
    }
    thresholds = reports[names[0]]["oracle_envelope_exceedance"]
    for name in names[1:]:
        if reports[name]["oracle_envelope_exceedance"]["soft_single_gaussian_thresholds"] != thresholds["soft_single_gaussian_thresholds"]:
            raise ValueError("paired runs use different soft quality thresholds")
        if reports[name]["oracle_envelope_exceedance"]["hard_thresholds"] != thresholds["hard_thresholds"]:
            raise ValueError("paired runs use different hard quality thresholds")
        if reports[name]["oracle_envelope_exceedance"]["grounding_and_distinctness_thresholds"] != thresholds["grounding_and_distinctness_thresholds"]:
            raise ValueError("paired runs use different grounding/distinctness thresholds")
    hard_thresholds = thresholds["hard_thresholds"]
    soft_thresholds = thresholds["soft_single_gaussian_thresholds"]
    topology_thresholds = thresholds["single_gaussian_heatmap_topology_thresholds"]
    activity_thresholds = thresholds["activity_thresholds"]
    grounding_thresholds = thresholds["grounding_and_distinctness_thresholds"]
    result["frozen_thresholds"] = {
        "soft_quality": soft_thresholds,
        "hard_quality": hard_thresholds,
        "single_gaussian_heatmap_topology": topology_thresholds,
        "activity": activity_thresholds,
        "grounding_and_distinctness": grounding_thresholds,
        "phase_dominance": thresholds["phase_dominance_thresholds"],
    }
    category_counts = {name: {} for name in names}
    overlapping_counts = {name: {
        "sliding": 0, "wobbling": 0, "both": 0, "clean": 0,
        "dead_or_off_object": 0, "persistent_duplicate": 0,
        "diffuse_or_multimodal_majority": 0,
    } for name in names}
    for channel in range(10):
        row: dict[str, Any] = {"channel": channel, "arms": {}}
        for name in names:
            report = reports[name]
            soft = report["soft_coordinate_metrics"]["full_nonseam"]
            hard = report["hard_coordinate_metrics"]["full_nonseam"]
            phase = report["phase_metrics"]["full"]["channels"][channel]
            heat = report["heatmap_topology"]
            arrays = arrays_by_name[name]
            raw = arrays["soft_coordinate"]
            raw_activity = float(np.sqrt(np.mean(np.sum(
                np.square(raw[:, channel] - np.mean(raw[:, channel], axis=0, keepdims=True)),
                axis=-1,
            ))))
            pair_rows = arrays["pair_channel_indices"]
            pair_distance = arrays["pairwise_distance_normalized"]
            close_fractions = [
                float(np.mean(pair_distance[:, pair_index] < grounding_thresholds["minimum_fixed_channel_pair_distance_normalized"]))
                for pair_index, pair in enumerate(pair_rows)
                if channel in (int(pair[0]), int(pair[1]))
            ]
            maximum_close_fraction = max(close_fractions)
            persistent_duplicate = maximum_close_fraction >= grounding_thresholds["persistent_duplicate_close_fraction"]
            active = raw_activity >= activity_thresholds["minimum_raw_orbit_rms_normalized"]
            on_object = report["localization"]["on_object_rate_per_channel"][channel]
            grounded = on_object >= grounding_thresholds["required_on_object_rate"]
            border_distance = report["localization"]["minimum_image_border_distance_px_per_channel"][channel]
            border_safe = border_distance >= grounding_thresholds["minimum_image_border_distance_px"]
            soft_metrics = soft
            sliding, wobbling = soft_coordinate_failure_flags(
                soft_metrics,
                channel=channel,
                soft_thresholds=soft_thresholds,
                hard_jump_rate=heat["hard_jump_rate_per_channel"][channel],
            )
            width_exceedance_rate = float(np.mean(
                arrays["rms_width_cells"][:, channel] > topology_thresholds["rms_width_cells_max"]
            ))
            soft_hard_exceedance_rate = float(np.mean(
                arrays["soft_hard_distance_cells"][:, channel] > topology_thresholds["soft_hard_distance_cells_max"]
            ))
            low_margin_rate = float(np.mean(
                arrays["separated_peak_r4_logit_margin"][:, channel]
                < topology_thresholds["separated_peak_radius4_logit_margin_min"]
            ))
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
            row["arms"][name] = {
                "soft_canonical_rms_normalized": soft["canonical_rms_per_channel"][channel],
                "soft_canonical_rms_input_pixels": soft["canonical_rms_per_channel"][channel] * 255.5,
                "hard_canonical_rms_normalized": hard["canonical_rms_per_channel"][channel],
                "adjacent_plus2_q99_heatmap_cells": soft["adjacent_plus2_step"]["per_channel"][channel]["q99"] * 31.5,
                "second_difference_q99_heatmap_cells": soft["noncyclic_second_difference"]["per_channel"][channel]["q99"] * 31.5,
                "within_residue_plus6_q99_heatmap_cells": soft["within_residue_plus6_step"]["per_channel"][channel]["q99"] * 31.5,
                "adjacent_energy_removed_by_constant_three_residue_offsets": phase["adjacent_step_energy_removed_by_constant_residue_correction"],
                "position_energy_removed_by_constant_three_residue_offsets": phase["position_energy_removed_by_constant_residue_correction"],
                "maximum_spike_excess_over_fitted_phase_normalized": phase["maximum_spike_excess_over_phase_bound"],
                "hard_jump_rate": heat["hard_jump_rate_per_channel"][channel],
                "on_object_rate": report["localization"]["on_object_rate_per_channel"][channel],
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
            }
            category_counts[name][primary_category] = category_counts[name].get(primary_category, 0) + 1
            overlapping_counts[name]["sliding"] += int(sliding)
            overlapping_counts[name]["wobbling"] += int(wobbling)
            overlapping_counts[name]["both"] += int(sliding and wobbling)
            overlapping_counts[name]["clean"] += int(primary_category == "clean")
            overlapping_counts[name]["dead_or_off_object"] += int(dead_or_off_object)
            overlapping_counts[name]["persistent_duplicate"] += int(persistent_duplicate)
            overlapping_counts[name]["diffuse_or_multimodal_majority"] += int(diffuse_majority)
        result["per_channel"].append(row)
    result["mutually_exclusive_primary_category_counts"] = category_counts
    result["overlapping_failure_flag_counts"] = overlapping_counts
    result["strict_success_contract_pass"] = {
        name: bool(category_counts[name].get("clean", 0) == 10)
        for name in names
    }
    kp2 = result["per_channel"][2]["arms"]
    first, second = names
    result["kp2_direct_comparison"] = {
        "soft_canonical_rms_ratio_second_over_first": kp2[second]["soft_canonical_rms_normalized"] / kp2[first]["soft_canonical_rms_normalized"],
        "adjacent_plus2_q99_ratio_second_over_first": kp2[second]["adjacent_plus2_q99_heatmap_cells"] / kp2[first]["adjacent_plus2_q99_heatmap_cells"],
        "second_difference_q99_ratio_second_over_first": kp2[second]["second_difference_q99_heatmap_cells"] / kp2[first]["second_difference_q99_heatmap_cells"],
        "hard_jump_rate_ratio_second_over_first": kp2[second]["hard_jump_rate"] / kp2[first]["hard_jump_rate"],
        "on_object_rate_first": kp2[first]["on_object_rate"],
        "on_object_rate_second": kp2[second]["on_object_rate"],
        "plain_language": (
            "The second arm has lower long-term KP2 drift and better object grounding, "
            "but worse adjacent-tail wobble, worse acceleration-tail wobble, and many more hard-peak jumps."
        ),
    }
    result["statistical_scope"] = {
        "statistics": "RMS, empirical q99, rates, and deterministic energy fractions",
        "sample_unit": "fixed keypoint channel over 180 correlated frames from one orbit",
        "n_frames_per_arm": 180,
        "n_channels_per_arm": 10,
        "descriptive_not_inferential": True,
        "correlation_caveat": "frames and overlapping differences are correlated; no SEM or population CI is reported",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-name", default="control")
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-name", default="ocr")
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = [args.first_name, args.second_name]
    reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, run_dir in ((args.first_name, args.first_run), (args.second_name, args.second_run)):
        reports[name], arrays[name] = _load_run(run_dir.resolve(strict=True))
    epochs = {int(report["checkpoint_epoch"]) for report in reports.values()}
    if len(epochs) != 1:
        raise ValueError("paired visualization requires the same checkpoint epoch in both arms")
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "trajectory_overview": _trajectory_overview(names, arrays, output_dir / "01_trajectory_overview.png"),
        "kp2_timeseries": _kp2_timeseries(names, arrays, reports, output_dir / "02_kp2_timeseries.png"),
        "kp2_heatmap_traces": _heatmap_traces(names, arrays, output_dir / "03_kp2_heatmap_traces.png"),
        "first_kp2_spike_montage": _spike_montage(names[0], reports[names[0]], arrays[names[0]], args.data_root, output_dir / "04_control_kp2_spike_montage.png"),
        "second_kp2_spike_montage": _spike_montage(names[1], reports[names[1]], arrays[names[1]], args.data_root, output_dir / "05_ocr_kp2_spike_montage.png"),
    }
    summary = _paired_numeric_summary(reports, arrays, names)
    summary["visual_files"] = files
    summary_path = output_dir / "PAIRED_FORENSIC_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": {"absolute_path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
        "visual_files": files,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
