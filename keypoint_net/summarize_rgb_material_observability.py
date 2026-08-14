"""Summarize the complete 24-role adjacent RGB material-observability matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = "rgb_material_observability_matrix_summary.v2"
EXPECTED_ROLES = 24
EXPECTED_CHANNELS = 10
PATCH_SIZES = (35, 105)
SCOPES = ("global", "local")


class RGBSummaryError(ValueError):
    """Raised when a complete source-bound summary cannot be made."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBSummaryError(message)


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


def _distribution(values: list[float], *, unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    _require(array.size > 0, "distribution is empty")
    return {
        "unit": unit,
        "n_role_channels": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "descriptive_not_inferential": True,
    }


def _group_key(role: Mapping[str, Any]) -> str:
    return f"{role['task']}__{role['arm']}__{role['checkpoint_role']}"


def _plot_heatmap(output: Path, rows: list[dict[str, Any]], patch_size: int, scope: str) -> None:
    matrix = np.asarray([
        [
            channel["material_error_px"]["maximum"]
            if channel["material_error_px"]["maximum"] is not None
            else np.nan
            for channel in row["reports"][str(patch_size)][scope]["channels"]
        ]
        for row in rows
    ], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_xticks(np.arange(EXPECTED_CHANNELS), [f"KP{channel}" for channel in range(EXPECTED_CHANNELS)])
    axis.set_yticks(np.arange(len(rows)), [row["role"]["role_key"] for row in rows], fontsize=7)
    axis.set_title(f"Worst adjacent material error — {patch_size}px / {scope}")
    colourbar = fig.colorbar(image, ax=axis)
    colourbar.set_label("maximum error (512-input pixels)")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_global_local(output: Path, rows: list[dict[str, Any]], patch_size: int) -> None:
    global_error, local_error, task, arm = [], [], [], []
    for row in rows:
        for global_channel, local_channel in zip(
            row["reports"][str(patch_size)]["global"]["channels"],
            row["reports"][str(patch_size)]["local"]["channels"],
        ):
            global_error.append(
                global_channel["material_error_px"]["maximum"]
                if global_channel["material_error_px"]["maximum"] is not None
                else np.nan
            )
            local_error.append(
                local_channel["material_error_px"]["maximum"]
                if local_channel["material_error_px"]["maximum"] is not None
                else np.nan
            )
            task.append(row["role"]["task"])
            arm.append(row["role"]["arm"])
    g = np.asarray(global_error, dtype=np.float64)
    l = np.asarray(local_error, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    styles = {
        ("task55_clean", "control"): ("tab:blue", "o"),
        ("task55_clean", "ocr_zncc"): ("tab:cyan", "s"),
        ("task80_assisted", "control"): ("tab:orange", "o"),
        ("task80_assisted", "ocr_zncc"): ("tab:red", "s"),
    }
    for key, (colour, marker) in styles.items():
        selected = np.asarray([(t, a) == key for t, a in zip(task, arm)], dtype=bool)
        axis.scatter(g[selected], l[selected], c=colour, marker=marker, s=25, alpha=0.7, label=" / ".join(key))
    maximum = float(max(np.nanmax(g), np.nanmax(l)))
    axis.plot([0, maximum], [0, maximum], color="black", ls="--", lw=1.0, label="equal worst error")
    axis.set_xlabel("global-match max material error (pixels)")
    axis.set_ylabel("local-match max material error (pixels)")
    axis.set_title(f"Local vs global adjacent RGB evidence — {patch_size}px")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def summarize(manifest_path: Path, matrix_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest_record = _file_record(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = manifest.get("roles")
    _require(isinstance(roles, list) and len(roles) == EXPECTED_ROLES, "manifest role count differs")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    bindings = []
    for role in roles:
        role_key = str(role["role_key"])
        role_root = matrix_root / role_key
        raw_path = role_root / "raw" / "raw_rgb_observability_receipt.json"
        evaluation_path = role_root / "evaluation" / "rgb_material_observability_evaluation.json"
        raw_record = _file_record(raw_path)
        evaluation_record = _file_record(evaluation_path)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _require(raw.get("role", {}).get("role_key") == role_key, f"{role_key}: raw role differs")
        _require(raw.get("frame_order_reversal_exact") is True, f"{role_key}: reverse-order proof failed")
        _require(evaluation.get("role", {}).get("role_key") == role_key, f"{role_key}: evaluation role differs")
        _require(evaluation.get("raw_prediction_hash_fixed_before_masks_or_theta_opened") is True, f"{role_key}: stage order differs")
        rows.append(evaluation)
        bindings.append({"role_key": role_key, "raw_receipt": raw_record, "evaluation": evaluation_record})

    configurations: dict[str, Any] = {}
    for patch_size in PATCH_SIZES:
        configurations[str(patch_size)] = {}
        for scope in SCOPES:
            role_channels = [
                (row, channel)
                for row in rows
                for channel in row["reports"][str(patch_size)][scope]["channels"]
            ]
            _require(len(role_channels) == EXPECTED_ROLES * EXPECTED_CHANNELS, "role-channel count differs")
            finite_max = [
                float(channel["material_error_px"]["maximum"])
                for _, channel in role_channels
                if channel["material_error_px"]["maximum"] is not None
            ]
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row, channel in role_channels:
                groups[_group_key(row["role"])].append(channel)
            configurations[str(patch_size)][scope] = {
                "n_roles": EXPECTED_ROLES,
                "n_role_channels": len(role_channels),
                "strict_pass_count": int(sum(channel["strict_pass"] for _, channel in role_channels)),
                "source_eligible_count": int(sum(channel["source_eligible"] for _, channel in role_channels)),
                "grounded_matcher_assessable_count": int(
                    sum(channel["grounded_matcher_assessable"] for _, channel in role_channels)
                ),
                "grounded_matcher_strict_pass_count": int(
                    sum(channel["grounded_matcher_strict_pass"] for _, channel in role_channels)
                ),
                "grounded_edge_count": int(sum(channel["grounded_edge_count"] for _, channel in role_channels)),
                "grounded_failure_count": int(sum(channel["grounded_failure_count"] for _, channel in role_channels)),
                "valid_edge_count": int(sum(channel["valid_edge_count"] for _, channel in role_channels)),
                "maximum_material_error_px": _distribution(finite_max, unit="512-input pixels"),
                "physical_candidate_top1_count": int(sum(channel["physical_candidate_top1_count"] for _, channel in role_channels)),
                "groups": {
                    key: {
                        "n_role_channels": len(channels),
                        "strict_pass_count": int(sum(channel["strict_pass"] for channel in channels)),
                        "source_eligible_count": int(sum(channel["source_eligible"] for channel in channels)),
                        "grounded_matcher_assessable_count": int(
                            sum(channel["grounded_matcher_assessable"] for channel in channels)
                        ),
                        "grounded_matcher_strict_pass_count": int(
                            sum(channel["grounded_matcher_strict_pass"] for channel in channels)
                        ),
                        "grounded_failure_count": int(sum(channel["grounded_failure_count"] for channel in channels)),
                    }
                    for key, channels in sorted(groups.items())
                },
            }

    visual_records = {}
    for patch_size in PATCH_SIZES:
        visual_records[str(patch_size)] = {}
        for scope in SCOPES:
            heatmap = output_dir / f"material_error_heatmap__patch{patch_size}__{scope}.png"
            _plot_heatmap(heatmap, rows, patch_size, scope)
            visual_records[str(patch_size)][scope] = {"material_error_heatmap": _file_record(heatmap)}
        scatter = output_dir / f"local_vs_global__patch{patch_size}.png"
        _plot_global_local(scatter, rows, patch_size)
        visual_records[str(patch_size)]["local_vs_global"] = _file_record(scatter)

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "complete_adjacent_rgb_material_observability_matrix_summary",
        "manifest": manifest_record,
        "matrix_root": str(matrix_root.resolve(strict=True)),
        "inventory": {"expected_roles": EXPECTED_ROLES, "complete_roles": len(rows), "role_channels_per_configuration": EXPECTED_ROLES * EXPECTED_CHANNELS},
        "raw_information_lock": {"all_24_reverse_order_exact": True, "all_24_geometry_absent_before_raw_hash": True},
        "configurations": configurations,
        "role_bindings": bindings,
        "visuals": visual_records,
        "statistical_scope": {
            "statistics": "deterministic descriptive counts, minimum, median, mean, q90, q99, maximum",
            "sample_unit": "fixed channel within one checkpoint role; adjacent edges are correlated",
            "n_role_channels_per_configuration": EXPECTED_ROLES * EXPECTED_CHANNELS,
            "n_seeds_per_task_arm": 3,
            "descriptive_not_inferential": True,
            "correlation_caveat": "best/final and channels share runs and all roles share one rendered orbit; no SEM, CI, or population inference",
        },
        "training_or_weight_update_performed": False,
    }
    report_path = output_dir / "RGB_MATERIAL_OBSERVABILITY_MATRIX_SUMMARY.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"report": _file_record(report_path), "configurations": configurations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory exists; use a fresh attempt")
    print(json.dumps(summarize(args.manifest.resolve(strict=True), args.matrix_root.resolve(strict=True), args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
