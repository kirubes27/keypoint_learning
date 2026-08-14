"""Render a complete visual audit of the 24-role frozen forensic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .frozen_wobble_forensics import rotate_points
    from .plot_frozen_wobble_pair import PALETTE, RESIDUE_COLORS, _load_run
    from .summarize_frozen_wobble_matrix import (
        _all_expected_roles,
        _content_hash,
        _role_key,
    )
except ImportError:
    from frozen_wobble_forensics import rotate_points  # type: ignore
    from plot_frozen_wobble_pair import PALETTE, RESIDUE_COLORS, _load_run  # type: ignore
    from summarize_frozen_wobble_matrix import _all_expected_roles, _content_hash, _role_key  # type: ignore


SCHEMA_VERSION = "frozen_wobble_complete_visual_audit.v1"
EXPECTED_MATRIX_SUMMARY_SCHEMA = "frozen_wobble_complete_matrix_summary.v1"
EXPECTED_CHANNELS = 10
EXPECTED_FRAMES = 180
IMPLEMENTATION_SOURCES = (
    "keypoint_net/render_frozen_wobble_matrix_audit.py",
    "keypoint_net/plot_frozen_wobble_pair.py",
    "keypoint_net/summarize_frozen_wobble_matrix.py",
)


class VisualAuditError(ValueError):
    """Fail-closed visual-audit error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualAuditError(message)


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


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCES],
        cwd=repo_root,
        check=False,
    )
    _require(clean.returncode == 0, "visual-audit implementation differs from its recorded HEAD")
    return {
        "implementation_head": head,
        "implementation_sources": {
            relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES
        },
    }


def _save_figure(fig: plt.Figure, path: Path) -> dict[str, Any]:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return _file_record(path)


def _square_limits(points: np.ndarray, minimum_half_width: float) -> tuple[float, float, float, float]:
    flat = points.reshape(-1, 2)
    x_mid = 0.5 * (float(np.min(flat[:, 0])) + float(np.max(flat[:, 0])))
    y_mid = 0.5 * (float(np.min(flat[:, 1])) + float(np.max(flat[:, 1])))
    half = max(
        minimum_half_width,
        0.55 * float(np.ptp(flat[:, 0])),
        0.55 * float(np.ptp(flat[:, 1])),
    )
    return x_mid - half, x_mid + half, y_mid - half, y_mid + half


def _worst_channel(report: Mapping[str, Any]) -> int:
    per_channel = report["soft_coordinate_metrics"]["full_nonseam"]["noncyclic_second_difference"]["per_channel"]
    return int(max(range(EXPECTED_CHANNELS), key=lambda channel: per_channel[channel]["q99"]))


def _trajectory_matrix(
    task: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    all_canonical = np.concatenate([row["arrays"]["soft_canonical_coordinate"] for row in rows], axis=0)
    all_limits = _square_limits(all_canonical, minimum_half_width=0.48)
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 4.2 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        canonical = row["arrays"]["soft_canonical_coordinate"]
        frames = row["arrays"]["frame_index"].astype(int)
        worst = row["worst_channel"]
        for channel in range(EXPECTED_CHANNELS):
            axes[row_index, 0].plot(
                canonical[:, channel, 0],
                canonical[:, channel, 1],
                color=PALETTE[channel],
                lw=0.75,
                alpha=0.72,
            )
        axes[row_index, 0].set_title(f"{row['role_key']}: all fixed channels (KP2 blue)", fontsize=9)
        axes[row_index, 0].set_xlim(all_limits[0], all_limits[1])
        axes[row_index, 0].set_ylim(all_limits[3], all_limits[2])
        local_limits = _square_limits(canonical[:, worst], minimum_half_width=0.15)
        axes[row_index, 1].plot(
            canonical[:, worst, 0], canonical[:, worst, 1], color="#777777", lw=0.55, alpha=0.65
        )
        for residue in range(3):
            keep = frames % 3 == residue
            axes[row_index, 1].scatter(
                canonical[keep, worst, 0],
                canonical[keep, worst, 1],
                color=RESIDUE_COLORS[residue],
                s=7,
                alpha=0.65,
                label=f"frame mod 3={residue}",
            )
        axes[row_index, 1].set_title(
            f"worst wobble: KP{worst}, locally zoomed; line joins every +2° frame", fontsize=9
        )
        axes[row_index, 1].set_xlim(local_limits[0], local_limits[1])
        axes[row_index, 1].set_ylim(local_limits[3], local_limits[2])
        if row_index == 0:
            axes[row_index, 1].legend(loc="best", fontsize=7)
        for ax in axes[row_index]:
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            ax.set_xlabel("de-rotated normalized x")
            ax.set_ylabel("de-rotated normalized y (image-down)")
    fig.suptitle(
        f"{task}: actual de-rotated 180-frame trajectories\n"
        "Visual evidence is descriptive over one correlated orbit; no uncertainty band is shown.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, output_path)


def _kp2_timeseries_matrix(
    task: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    threshold_cells = (
        rows[0]["report"]["oracle_envelope_exceedance"]["hard_thresholds"]["cyclic_second_difference_max"]
        * 31.5
    )
    fig, axes = plt.subplots(len(rows), 2, figsize=(17, 2.9 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        arrays = row["arrays"]
        frame = arrays["frame_index"]
        canonical = arrays["soft_canonical_coordinate"][:, 2]
        second = np.linalg.norm(canonical[2:] - 2.0 * canonical[1:-1] + canonical[:-2], axis=-1) * 31.5
        axes[row_index, 0].plot(frame, canonical[:, 0], color="#0072B2", lw=0.85, label="x")
        axes[row_index, 0].plot(frame, canonical[:, 1], color="#D55E00", lw=0.85, label="y")
        axes[row_index, 0].set_ylabel(row["role_key"], fontsize=7)
        axes[row_index, 1].plot(frame[1:-1], second, color="#333333", lw=0.8)
        axes[row_index, 1].axhline(threshold_cells, color="#D62728", ls=":", lw=1.0)
        for ax in axes[row_index]:
            ax.grid(alpha=0.18)
            for boundary in (24, 27, 177):
                ax.axvline(boundary, color="#888888", lw=0.6, alpha=0.6)
        if row_index == 0:
            axes[row_index, 0].legend(loc="upper right", fontsize=7)
            axes[row_index, 0].set_title("blue KP2 de-rotated coordinates")
            axes[row_index, 1].set_title("blue KP2 local wobble in r64 heatmap cells")
    axes[-1, 0].set_xlabel("frame index (2° per frame)")
    axes[-1, 1].set_xlabel("frame index (2° per frame)")
    fig.suptitle(
        f"{task}: blue KP2 coordinate-over-time visual audit\n"
        "Red dotted line is the worst-case clean r64 quantization maximum; traces are descriptive, not inferential.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, output_path)


def _frame_image(data_root: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    object_root = data_root / "train" / "engineers_hammer_vray"
    with Image.open(object_root / "frames" / "a" / f"img_{frame:04d}.png") as handle:
        image = np.asarray(handle.convert("RGB"))
    with Image.open(object_root / "masks" / "a" / f"mask_{frame:04d}.png") as handle:
        mask = np.asarray(handle.convert("L")) > 0
    return image, mask


def _cell_to_normalized(cell: np.ndarray) -> np.ndarray:
    return -1.0 + (2.0 * np.asarray(cell, dtype=np.float64) / 63.0)


def _overlay(ax: plt.Axes, arrays: Mapping[str, np.ndarray], frame: int, channel: int) -> None:
    soft = arrays["soft_coordinate"][frame, channel]
    hard = arrays["hard_coordinate"][frame, channel]
    previous = arrays["soft_coordinate"][frame - 1 if frame > 0 else -1, channel]
    expected = rotate_points(previous[None], 2.0)[0]
    second = _cell_to_normalized(
        np.asarray(
            (
                arrays["separated_peak_r4_second_peak_x_cell"][frame, channel],
                arrays["separated_peak_r4_second_peak_y_cell"][frame, channel],
            )
        )
    )
    ax.scatter(soft[0], soft[1], s=52, marker="o", facecolors="none", edgecolors="#00FFFF", lw=1.5, label="soft")
    ax.scatter(hard[0], hard[1], s=52, marker="+", color="#FFFF00", lw=1.6, label="hard top peak")
    ax.scatter(second[0], second[1], s=35, marker="D", facecolors="none", edgecolors="#FF00FF", lw=1.2, label="separated second peak")
    ax.scatter(expected[0], expected[1], s=48, marker="x", color="#FFFFFF", lw=1.5, label="previous +2°")


def _spike_montage(
    row: Mapping[str, Any],
    data_root: Path,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = row["report"]
    arrays = row["arrays"]
    channel = int(row["worst_channel"])
    spikes = report["spike_frames"]["soft"][channel]
    centres = [int(spike["centre_frame_index"]) for spike in spikes]
    fig, axes = plt.subplots(len(centres), 6, figsize=(19, 3.2 * len(centres)), squeeze=False)
    for row_index, (centre, spike) in enumerate(zip(centres, spikes)):
        frames = (centre - 1, centre, centre + 1)
        for column, frame in enumerate(frames):
            image, mask = _frame_image(data_root, frame)
            ax = axes[row_index, column]
            ax.imshow(image, extent=(-1, 1, 1, -1))
            mask_axis = np.linspace(-1.0, 1.0, mask.shape[0])
            ax.contour(mask_axis, mask_axis, mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.35)
            _overlay(ax, arrays, frame, channel)
            ax.set_title(f"RGB frame {frame}", fontsize=8)
            ax.set_xlim(-1, 1)
            ax.set_ylim(1, -1)
            ax.axis("off")
        for offset, frame in enumerate(frames):
            ax = axes[row_index, 3 + offset]
            ax.imshow(arrays["logits"][frame, channel], cmap="magma", extent=(-1, 1, 1, -1), interpolation="nearest")
            _overlay(ax, arrays, frame, channel)
            ax.set_title(f"KP{channel} heatmap frame {frame}", fontsize=8)
            ax.set_xlim(-1, 1)
            ax.set_ylim(1, -1)
            ax.axis("off")
        axes[row_index, 0].text(
            -0.98,
            0.92,
            f"local wobble={float(spike['canonical_second_difference']) * 31.5:.2f} cells",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.935), fontsize=8)
    fig.suptitle(
        f"{row['role_key']}: five worst visible spike moments for KP{channel}\n"
        "RGB and heatmap evidence are shown together; the white mask outline is the bound hammer silhouette.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    record = _save_figure(fig, output_path)
    selection = {
        "role_key": row["role_key"],
        "selected_worst_channel": channel,
        "selection_metric": "largest non-seam canonical second-difference q99",
        "spike_frames": spikes,
    }
    return record, selection


def run(
    matrix_root: Path,
    matrix_summary_path: Path,
    expected_matrix_summary_sha256: str,
    data_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    implementation = _implementation_binding(repo_root)
    summary_record = _file_record(matrix_summary_path)
    _require(summary_record["sha256"] == expected_matrix_summary_sha256, "matrix summary SHA-256 differs")
    matrix_summary = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
    _require(matrix_summary["schema_version"] == EXPECTED_MATRIX_SUMMARY_SCHEMA, "matrix summary schema differs")
    _require(matrix_summary["content_hash_sha256"] == _content_hash(matrix_summary), "matrix summary content hash differs")
    _require(matrix_summary["inventory"]["matrix_complete"] is True, "matrix summary is incomplete")
    _require(matrix_summary["training_or_weight_update_performed"] is False, "matrix summary reports weight updates")

    rows = []
    observed = set()
    for report_path in sorted(matrix_root.resolve(strict=True).glob("*/forensic_report.json")):
        report, arrays = _load_run(report_path.parent)
        role_key = _role_key(report["recipe"], report["arm"], int(report["seed"]), report["checkpoint_role"])
        _require(role_key == report_path.parent.name, "visual input directory differs from bound role")
        observed.add(role_key)
        rows.append(
            {
                "role_key": role_key,
                "task": report["recipe"],
                "report": report,
                "arrays": arrays,
                "worst_channel": _worst_channel(report),
            }
        )
    _require(observed == _all_expected_roles(), "visual audit does not contain all 24 roles")

    output_dir.mkdir(parents=True, exist_ok=False)
    spike_root = output_dir / "worst_channel_spike_montages"
    spike_root.mkdir()
    task_files = {}
    for index, task in enumerate(("task55_clean", "task80_assisted"), start=1):
        task_rows = [row for row in rows if row["task"] == task]
        task_files[task] = {
            "canonical_trajectories": _trajectory_matrix(
                task, task_rows, output_dir / f"0{2 * index - 1}_{task}_canonical_trajectories.png"
            ),
            "kp2_timeseries": _kp2_timeseries_matrix(
                task, task_rows, output_dir / f"0{2 * index}_{task}_kp2_timeseries.png"
            ),
        }
    spike_files = {}
    selections = []
    for row in rows:
        path = spike_root / f"{row['role_key']}__worst_channel_spikes.png"
        spike_files[row["role_key"]], selection = _spike_montage(row, data_root, path)
        selections.append(selection)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "source_bound_complete_frozen_matrix_visual_audit",
        "training_or_weight_update_performed": False,
        "implementation": implementation,
        "bindings": {
            "matrix_summary": summary_record,
            "expected_matrix_summary_sha256": expected_matrix_summary_sha256,
            "matrix_root": str(matrix_root.resolve(strict=True)),
            "data_root": str(data_root.resolve(strict=True)),
        },
        "inventory": {
            "checkpoint_roles": len(rows),
            "trajectory_matrices": 2,
            "kp2_timeseries_matrices": 2,
            "worst_channel_spike_montages": len(spike_files),
            "spike_moments_per_role": 5,
        },
        "task_files": task_files,
        "spike_files": spike_files,
        "spike_selections": selections,
        "statistical_scope": {
            "sample_unit": "fixed keypoint channel over one correlated 180-frame orbit",
            "descriptive_not_inferential": True,
            "error_bars_or_uncertainty_bands": "none",
            "reason": "frames and overlapping differences are correlated; the figures show the complete deterministic orbit",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    manifest_path = output_dir / "VISUAL_AUDIT_MANIFEST.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "manifest": _file_record(manifest_path),
        "checkpoint_roles": len(rows),
        "spike_montages": len(spike_files),
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--expected-matrix-summary-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.matrix_root,
                args.matrix_summary,
                args.expected_matrix_summary_sha256,
                args.data_root,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
