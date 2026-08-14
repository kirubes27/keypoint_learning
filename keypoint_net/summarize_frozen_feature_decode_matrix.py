"""Summarize the complete 24-role frozen feature-decode matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = "frozen_feature_decode_matrix_summary.v1"
EXPECTED_ROLES = 24
EXPECTED_CHANNELS = 10
BASIS_NAMES = ("hard_peak", "soft_coordinate")


class DecodeSummaryError(ValueError):
    """Raised when the full matrix is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeSummaryError(message)


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


def _one_evaluation(role_dir: Path, role_key: str) -> Path:
    candidates = sorted(role_dir.glob("evaluation_v*/feature_decode_evaluation.json"))
    valid = []
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("role", {}).get("role_key") == role_key:
            valid.append(path)
    _require(len(valid) == 1, f"{role_key}: expected one complete evaluation, got {len(valid)}")
    return valid[0]


def _distribution(values: list[float], *, unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "cannot summarize empty/non-finite values")
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


def _plot_material_heatmap(output: Path, role_rows: list[dict[str, Any]], basis: str) -> None:
    matrix = np.asarray([[channel["material_error_max_input_pixels"] for channel in row["basis_reports"][basis]["channels"]] for row in role_rows])
    fig, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_xticks(np.arange(EXPECTED_CHANNELS), [f"KP{channel}" for channel in range(EXPECTED_CHANNELS)])
    axis.set_yticks(np.arange(len(role_rows)), [row["role"]["role_key"] for row in role_rows], fontsize=7)
    axis.set_title(f"Worst full-orbit material error per fixed channel — {basis}")
    colourbar = fig.colorbar(image, ax=axis)
    colourbar.set_label("maximum material error (512-input pixels)")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_decode_vs_detector(output: Path, role_rows: list[dict[str, Any]], basis: str) -> None:
    decoded, detector, task, arm = [], [], [], []
    for row in role_rows:
        for channel in row["basis_reports"][basis]["channels"]:
            decoded.append(channel["material_error_max_input_pixels"])
            detector.append(channel["detector_material_error_max_input_pixels"])
            task.append(row["role"]["task"])
            arm.append(row["role"]["arm"])
    decoded_array = np.asarray(decoded)
    detector_array = np.asarray(detector)
    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    styles = {
        ("task55_clean", "control"): ("tab:blue", "o"),
        ("task55_clean", "ocr_zncc"): ("tab:cyan", "s"),
        ("task80_assisted", "control"): ("tab:orange", "o"),
        ("task80_assisted", "ocr_zncc"): ("tab:red", "s"),
    }
    for key, (colour, marker) in styles.items():
        selected = np.asarray([(t, a) == key for t, a in zip(task, arm)], dtype=bool)
        axis.scatter(detector_array[selected], decoded_array[selected], s=25, alpha=0.7, c=colour, marker=marker, label=" / ".join(key))
    maximum = float(max(np.max(decoded_array), np.max(detector_array)))
    axis.plot([0, maximum], [0, maximum], color="black", ls="--", lw=1.0, label="equal worst error")
    axis.set_xlabel("original detector max material error (input pixels)")
    axis.set_ylabel("feature decode max material error (input pixels)")
    axis.set_title(f"Below line = frozen feature decode improves worst error — {basis}")
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
    role_rows = []
    bindings = []
    all_raw_exact = True
    for role in roles:
        role_key = str(role["role_key"])
        role_dir = matrix_root / role_key
        receipt_path = role_dir / "raw_feature_decode_receipt.json"
        receipt_record = _file_record(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(receipt.get("role", {}).get("role_key") == role_key, f"{role_key}: raw role differs")
        _require(receipt.get("masks_or_geometry_opened") is False, f"{role_key}: raw information lock failed")
        all_raw_exact = all_raw_exact and receipt.get("frame_order_reversal_exact") is True
        evaluation_path = _one_evaluation(role_dir, role_key)
        evaluation_record = _file_record(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _require(evaluation.get("raw_prediction_hash_fixed_before_masks_or_theta_opened") is True, f"{role_key}: stage order differs")
        role_rows.append(evaluation)
        bindings.append({"role_key": role_key, "raw_receipt": receipt_record, "evaluation": evaluation_record})
    _require(all_raw_exact, "at least one raw role is not exactly order invariant")

    basis_summary: dict[str, Any] = {}
    for basis in BASIS_NAMES:
        channels = [(row, channel) for row in role_rows for channel in row["basis_reports"][basis]["channels"]]
        _require(len(channels) == EXPECTED_ROLES * EXPECTED_CHANNELS, "role-channel count differs")
        failure_counts: Counter[str] = Counter()
        for _, channel in channels:
            for check, passed in channel["checks"].items():
                if not passed:
                    failure_counts[check] += 1
        group_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for row, channel in channels:
            group_rows[_group_key(row["role"])].append((row, channel))
        groups = {}
        for key, rows in sorted(group_rows.items()):
            groups[key] = {
                "n_role_channels": len(rows),
                "strict_pass_count": sum(channel["strict_r64_diagnostic_pass"] for _, channel in rows),
                "on_object_all_frames_count": sum(channel["checks"]["on_object_all_frames"] for _, channel in rows),
                "distinct_all_frames_count": sum(channel["checks"]["distinct_all_frames"] for _, channel in rows),
                "material_error_max_pixels": _distribution([channel["material_error_max_input_pixels"] for _, channel in rows], unit="512-input pixels"),
            }
        kp2 = [(row, channel) for row, channel in channels if channel["channel"] == 2]
        worst_row, worst_channel = max(channels, key=lambda item: item[1]["material_error_max_input_pixels"])
        basis_summary[basis] = {
            "n_roles": EXPECTED_ROLES,
            "n_role_channels": len(channels),
            "strict_pass_count": sum(channel["strict_r64_diagnostic_pass"] for _, channel in channels),
            "strict_all_240_pass": all(channel["strict_r64_diagnostic_pass"] for _, channel in channels),
            "failure_counts": dict(sorted(failure_counts.items())),
            "on_object_all_frames_count": sum(channel["checks"]["on_object_all_frames"] for _, channel in channels),
            "distinct_all_frames_count": sum(channel["checks"]["distinct_all_frames"] for _, channel in channels),
            "decode_worst_error_better_than_detector_count": sum(channel["material_error_max_input_pixels"] < channel["detector_material_error_max_input_pixels"] for _, channel in channels),
            "decoded_closer_than_detector_fraction": _distribution([channel["decoded_closer_than_detector_fraction"] for _, channel in channels], unit="fraction of 180 correlated frames"),
            "material_error_max_pixels": _distribution([channel["material_error_max_input_pixels"] for _, channel in channels], unit="512-input pixels"),
            "canonical_rms_pixels": _distribution([channel["canonical_rms_input_pixels"] for _, channel in channels], unit="512-input pixels"),
            "adjacent_step_max_cells": _distribution([channel["adjacent_step_max_cells"] for _, channel in channels], unit="r64 cells"),
            "cyclic_second_difference_max_cells": _distribution([channel["cyclic_second_difference_max_cells"] for _, channel in channels], unit="r64 cells"),
            "kp2": {
                "n_roles": len(kp2),
                "strict_pass_count": sum(channel["strict_r64_diagnostic_pass"] for _, channel in kp2),
                "on_object_all_frames_count": sum(channel["checks"]["on_object_all_frames"] for _, channel in kp2),
                "material_error_max_pixels": _distribution([channel["material_error_max_input_pixels"] for _, channel in kp2], unit="512-input pixels"),
                "decoded_closer_than_detector_fraction": _distribution([channel["decoded_closer_than_detector_fraction"] for _, channel in kp2], unit="fraction of 180 correlated frames"),
            },
            "worst_role_channel": {
                "role_key": worst_row["role"]["role_key"],
                "channel": worst_channel["channel"],
                "material_error_max_input_pixels": worst_channel["material_error_max_input_pixels"],
            },
            "groups": groups,
        }

    visual_records = {}
    for basis in BASIS_NAMES:
        heatmap = output_dir / f"material_error_heatmap__{basis}.png"
        scatter = output_dir / f"decode_vs_detector__{basis}.png"
        _plot_material_heatmap(heatmap, role_rows, basis)
        _plot_decode_vs_detector(scatter, role_rows, basis)
        visual_records[basis] = {"material_error_heatmap": _file_record(heatmap), "decode_vs_detector": _file_record(scatter)}

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "complete_frozen_feature_decode_matrix_summary",
        "manifest": manifest_record,
        "matrix_root": str(matrix_root.resolve(strict=True)),
        "inventory": {"expected_roles": EXPECTED_ROLES, "complete_roles": len(role_rows), "role_channels_per_basis": EXPECTED_ROLES * EXPECTED_CHANNELS},
        "raw_information_lock": {"all_24_frame_order_reversal_exact": all_raw_exact, "all_24_masks_or_geometry_absent_before_raw_hash": True},
        "basis_summary": basis_summary,
        "role_bindings": bindings,
        "visuals": visual_records,
        "statistical_scope": {
            "statistics": "descriptive empirical minimum, median, mean, q90, q99, maximum, and counts",
            "sample_unit": "fixed channel within a checkpoint role; 240 role-channels per anchor basis",
            "n_seeds_per_task_arm": 3,
            "descriptive_not_inferential": True,
            "correlation_caveat": "best/final checkpoints and channels share training runs, roles share one rendered hammer orbit, and frame-level values are correlated; no population CI or SEM is reported",
        },
        "training_or_weight_update_performed": False,
    }
    report_path = output_dir / "FROZEN_FEATURE_DECODE_MATRIX_SUMMARY.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"report": _file_record(report_path), "hard_pass_count": basis_summary["hard_peak"]["strict_pass_count"], "soft_pass_count": basis_summary["soft_coordinate"]["strict_pass_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "summary output directory exists; use a fresh attempt")
    print(json.dumps(summarize(args.manifest.resolve(strict=True), args.matrix_root.resolve(strict=True), args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
