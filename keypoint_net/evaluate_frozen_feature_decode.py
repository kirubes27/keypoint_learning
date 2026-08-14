"""Evaluate hashed geometry-blind feature decodes with masks and physical roll metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .frozen_feature_forensics import coordinate_on_mask
    from .frozen_wobble_forensics import canonicalize, coordinate_dynamics, rotate_points
    from .run_frozen_wobble_forensics import _mask_localization
except ImportError:
    from frozen_feature_forensics import coordinate_on_mask  # type: ignore
    from frozen_wobble_forensics import canonicalize, coordinate_dynamics, rotate_points  # type: ignore
    from run_frozen_wobble_forensics import _mask_localization  # type: ignore


SCHEMA_VERSION = "frozen_feature_decode_evaluation.v1"
EXPECTED_RAW_SCHEMA = "frozen_feature_decode_raw_receipt.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
PIXEL_SCALE = 255.5
CELL_SCALE = 31.5
BASIS_NAMES = ("hard_peak", "soft_coordinate")
IMPLEMENTATION_SOURCES = (
    "keypoint_net/evaluate_frozen_feature_decode.py",
    "keypoint_net/frozen_feature_decode.py",
    "keypoint_net/frozen_wobble_forensics.py",
    "keypoint_net/run_frozen_wobble_forensics.py",
)
HARD_R64_THRESHOLDS = {
    "canonical_rms_max": 0.013402778437084897,
    "canonical_radius_max": 0.031370361965226204,
    "adjacent_step_max": 0.04411225264635446,
    "cyclic_second_difference_max": 0.07060099491763448,
    "within_residue_step_max": 0.04254940872104899,
    "material_error_max": float(np.nextafter(math.sqrt(2.0) / 63.0, math.inf)),
    "seam_step_max": 0.04411225264635446,
    "required_on_object_rate": 1.0,
    "minimum_image_border_distance_px": 4.055555555555555,
    "minimum_pair_distance_normalized": 0.031746031746031744,
    "minimum_raw_orbit_rms_normalized": 0.01587301587301587,
}


class DecodeEvaluationError(ValueError):
    """Raised when post-hash evaluation lineage or semantics differ."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeEvaluationError(message)


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


def _load_bound_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    observed = _file_record(str(record["absolute_path"]))
    _require(observed == dict(record), f"{name} binding differs")
    value = json.loads(Path(observed["absolute_path"]).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _summary(values: np.ndarray, *, unit: str, sample_unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 2 and array.shape[1] == EXPECTED_CHANNELS, "summary array shape differs")
    return {
        "unit": unit,
        "n_per_channel": int(array.shape[0]),
        "sample_unit": sample_unit,
        "descriptive_not_inferential": True,
        "per_channel": [
            {
                "channel": channel,
                "mean": float(np.mean(array[:, channel])),
                "median": float(np.median(array[:, channel])),
                "q90": float(np.quantile(array[:, channel], 0.90)),
                "q99": float(np.quantile(array[:, channel], 0.99)),
                "maximum": float(np.max(array[:, channel])),
            }
            for channel in range(EXPECTED_CHANNELS)
        ],
    }


def _load_geometry_and_masks(report: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    corpus = report.get("bindings", {}).get("corpus")
    _require(isinstance(corpus, Mapping), "source corpus binding missing")
    object_root = Path(str(corpus["object_root"])).resolve(strict=True)
    frames = corpus.get("frame_records")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "source frame count differs")
    theta = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    masks = np.empty((EXPECTED_FRAMES, 512, 512), dtype=bool)
    image_paths: list[Path] = []
    for index, row in enumerate(frames):
        _require(int(row.get("frame_index", -1)) == index, "source frame index differs")
        theta[index] = float(row["theta_deg"])
        mask_path = (object_root / str(row["mask_relpath"])).resolve(strict=True)
        image_path = (object_root / str(row["image_relpath"])).resolve(strict=True)
        _require(_sha256(mask_path) == row["mask_sha256"], f"mask hash differs at frame {index}")
        _require(_sha256(image_path) == row["image_sha256"], f"image hash differs at frame {index}")
        with Image.open(mask_path) as image:
            masks[index] = np.asarray(image.convert("L")) > 0
        image_paths.append(image_path)
    _require(np.array_equal(theta, np.arange(EXPECTED_FRAMES, dtype=np.float64) * 2.0), "theta is not exact +2-degree roll")
    return theta, masks, image_paths


def _pair_diagnostics(points: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    pairs = [(left, right) for left in range(EXPECTED_CHANNELS) for right in range(left + 1, EXPECTED_CHANNELS)]
    distance = np.stack([np.linalg.norm(points[:, left] - points[:, right], axis=-1) for left, right in pairs], axis=1)
    rows = []
    for index, (left, right) in enumerate(pairs):
        rows.append(
            {
                "left_channel": left,
                "right_channel": right,
                "minimum_normalized": float(np.min(distance[:, index])),
                "median_normalized": float(np.median(distance[:, index])),
                "close_fraction": float(np.mean(distance[:, index] < HARD_R64_THRESHOLDS["minimum_pair_distance_normalized"])),
            }
        )
    return rows, distance


def _basis_metrics(
    decoded: np.ndarray,
    detector: np.ndarray,
    anchor: np.ndarray,
    theta: np.ndarray,
    masks: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    anchor_frame = 27
    physical = rotate_points(
        np.broadcast_to(anchor, decoded.shape),
        theta[:, None] - theta[anchor_frame],
    )
    material = np.linalg.norm(decoded - physical, axis=-1)
    detector_material = np.linalg.norm(detector - physical, axis=-1)
    canonical = canonicalize(decoded, theta)
    dynamics = coordinate_dynamics(canonical)
    cyclic_second = np.linalg.norm(
        np.roll(canonical, -1, axis=0) - 2.0 * canonical + np.roll(canonical, 1, axis=0), axis=-1
    )
    seam = np.linalg.norm(canonical[0] - canonical[-1], axis=-1)
    centred = canonical - np.mean(canonical, axis=0, keepdims=True)
    canonical_rms = np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=0))
    raw_centred = decoded - np.mean(decoded, axis=0, keepdims=True)
    raw_rms = np.sqrt(np.mean(np.sum(np.square(raw_centred), axis=-1), axis=0))
    localization, localization_arrays = _mask_localization(decoded, masks)
    pair_rows, pair_distance = _pair_diagnostics(decoded)
    minimum_other = np.full(EXPECTED_CHANNELS, np.inf, dtype=np.float64)
    for pair_index, row in enumerate(pair_rows):
        left, right = row["left_channel"], row["right_channel"]
        minimum = float(np.min(pair_distance[:, pair_index]))
        minimum_other[left] = min(minimum_other[left], minimum)
        minimum_other[right] = min(minimum_other[right], minimum)

    channels = []
    for channel in range(EXPECTED_CHANNELS):
        checks = {
            "anchor_on_object": coordinate_on_mask(masks[anchor_frame], anchor[channel]),
            "on_object_all_frames": localization["on_object_rate_per_channel"][channel] == 1.0,
            "border_safe_all_frames": localization["minimum_image_border_distance_px_per_channel"][channel] >= HARD_R64_THRESHOLDS["minimum_image_border_distance_px"],
            "active": raw_rms[channel] >= HARD_R64_THRESHOLDS["minimum_raw_orbit_rms_normalized"],
            "distinct_all_frames": minimum_other[channel] >= HARD_R64_THRESHOLDS["minimum_pair_distance_normalized"],
            "material_error": float(np.max(material[:, channel])) <= HARD_R64_THRESHOLDS["material_error_max"],
            "canonical_rms": canonical_rms[channel] <= HARD_R64_THRESHOLDS["canonical_rms_max"],
            "canonical_radius": float(np.max(dynamics["radius_about_mean"][:, channel])) <= HARD_R64_THRESHOLDS["canonical_radius_max"],
            "adjacent_step": float(np.max(dynamics["adjacent_step"][:, channel])) <= HARD_R64_THRESHOLDS["adjacent_step_max"],
            "within_residue_step": float(np.max(dynamics["within_residue_step"][:, channel])) <= HARD_R64_THRESHOLDS["within_residue_step_max"],
            "cyclic_second_difference": float(np.max(cyclic_second[:, channel])) <= HARD_R64_THRESHOLDS["cyclic_second_difference_max"],
            "seam_step": seam[channel] <= HARD_R64_THRESHOLDS["seam_step_max"],
        }
        channels.append(
            {
                "channel": channel,
                "strict_r64_diagnostic_pass": all(checks.values()),
                "checks": checks,
                "material_error_max_input_pixels": float(np.max(material[:, channel]) * PIXEL_SCALE),
                "material_error_median_input_pixels": float(np.median(material[:, channel]) * PIXEL_SCALE),
                "detector_material_error_max_input_pixels": float(np.max(detector_material[:, channel]) * PIXEL_SCALE),
                "decoded_closer_than_detector_fraction": float(np.mean(material[:, channel] < detector_material[:, channel])),
                "canonical_rms_input_pixels": float(canonical_rms[channel] * PIXEL_SCALE),
                "adjacent_step_max_cells": float(np.max(dynamics["adjacent_step"][:, channel]) * CELL_SCALE),
                "cyclic_second_difference_max_cells": float(np.max(cyclic_second[:, channel]) * CELL_SCALE),
                "on_object_rate": float(localization["on_object_rate_per_channel"][channel]),
                "minimum_other_channel_distance_input_pixels": float(minimum_other[channel] * PIXEL_SCALE),
            }
        )
    return {
        "strict_pass_count": int(sum(row["strict_r64_diagnostic_pass"] for row in channels)),
        "strict_all_ten_pass": all(row["strict_r64_diagnostic_pass"] for row in channels),
        "channels": channels,
        "material_error": _summary(material * PIXEL_SCALE, unit="512-input pixels", sample_unit="correlated frames"),
        "detector_material_error": _summary(detector_material * PIXEL_SCALE, unit="512-input pixels", sample_unit="correlated frames"),
        "canonical_rms_per_channel_normalized": [float(value) for value in canonical_rms],
        "adjacent_step": _summary(dynamics["adjacent_step"] * CELL_SCALE, unit="r64 cells", sample_unit="overlapping adjacent frame pairs"),
        "cyclic_second_difference": _summary(cyclic_second * CELL_SCALE, unit="r64 cells", sample_unit="overlapping cyclic triplets"),
        "within_residue_step": _summary(dynamics["within_residue_step"] * CELL_SCALE, unit="r64 cells", sample_unit="overlapping +6-degree pairs"),
        "cyclic_seam_step_normalized_per_channel": [float(value) for value in seam],
        "localization": localization,
        "pairwise_distinctness": {"pairs": pair_rows, "minimum_other_channel_normalized": minimum_other.tolist()},
    }, {
        "physical_coordinate": physical,
        "canonical_coordinate": canonical,
        "material_error_pixels": material * PIXEL_SCALE,
        "detector_material_error_pixels": detector_material * PIXEL_SCALE,
        "cyclic_second_difference_cells": cyclic_second * CELL_SCALE,
        "on_object": localization_arrays["on_object"],
        "pairwise_distance_normalized": pair_distance,
    }


def _plot_trajectories(
    output: Path,
    decoded: np.ndarray,
    physical: np.ndarray,
    canonical: np.ndarray,
    basis: str,
) -> None:
    colours = plt.cm.tab10(np.arange(EXPECTED_CHANNELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for channel in range(EXPECTED_CHANNELS):
        axes[0].plot(decoded[:, channel, 0], decoded[:, channel, 1], color=colours[channel], lw=1.0, label=f"KP{channel}")
        axes[0].plot(physical[:, channel, 0], physical[:, channel, 1], color=colours[channel], lw=0.6, ls="--", alpha=0.6)
        axes[1].plot(canonical[:, channel, 0], canonical[:, channel, 1], color=colours[channel], lw=1.0)
    axes[0].set_title("Image-space orbit: feature decode (solid), material track (dashed)")
    axes[1].set_title("After exact physical de-rotation (a clean point stays compact)")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("normalized x")
        axis.set_ylabel("normalized y")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Full-orbit frozen feature decode — {basis}")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_time_series(output: Path, material: np.ndarray, second: np.ndarray, basis: str) -> None:
    colours = plt.cm.tab10(np.arange(EXPECTED_CHANNELS))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for channel in range(EXPECTED_CHANNELS):
        axes[0].plot(np.arange(EXPECTED_FRAMES), material[:, channel], color=colours[channel], lw=0.9, label=f"KP{channel}")
        axes[1].plot(np.arange(EXPECTED_FRAMES), second[:, channel], color=colours[channel], lw=0.9)
    axes[0].axhline(HARD_R64_THRESHOLDS["material_error_max"] * PIXEL_SCALE, color="black", ls="--", lw=1.0, label="r64 clean limit")
    axes[1].axhline(HARD_R64_THRESHOLDS["cyclic_second_difference_max"] * CELL_SCALE, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("material error (input pixels)")
    axes[1].set_ylabel("cyclic 2nd difference (r64 cells)")
    axes[1].set_xlabel("frame index (2 degrees per frame)")
    axes[0].legend(ncol=6, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"Where sliding and wobble happen — {basis}; descriptive correlated frames")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_worst_events(
    output: Path,
    image_paths: list[Path],
    decoded: np.ndarray,
    detector: np.ndarray,
    physical: np.ndarray,
    material: np.ndarray,
    second: np.ndarray,
    basis: str,
) -> dict[str, Any]:
    worst_channel = int(np.argmax(np.max(material, axis=0)))
    selections = []
    for channel in sorted({2, worst_channel}):
        selections.append((channel, int(np.argmax(material[:, channel])), "largest material error"))
        selections.append((channel, int(np.argmax(second[:, channel])), "largest cyclic second difference"))
    fig, axes = plt.subplots(len(selections), 2, figsize=(12, 4 * len(selections)), constrained_layout=True)
    if len(selections) == 1:
        axes = np.asarray([axes])
    for row, (channel, frame, label) in enumerate(selections):
        with Image.open(image_paths[27]) as image:
            source = np.asarray(image.convert("RGB"))
        with Image.open(image_paths[frame]) as image:
            target = np.asarray(image.convert("RGB"))
        axes[row, 0].imshow(source)
        anchor = physical[27, channel]
        axes[row, 0].scatter((anchor[0] + 1) * PIXEL_SCALE, (anchor[1] + 1) * PIXEL_SCALE, s=80, facecolors="none", edgecolors="lime", lw=2)
        axes[row, 0].set_title(f"Frame 27 anchor, KP{channel}")
        axes[row, 1].imshow(target)
        for point, colour, marker, name in (
            (physical[frame, channel], "lime", "o", "material"),
            (decoded[frame, channel], "red", "x", "feature decode"),
            (detector[frame, channel], "cyan", "+", "original detector"),
        ):
            axes[row, 1].scatter((point[0] + 1) * PIXEL_SCALE, (point[1] + 1) * PIXEL_SCALE, s=90, c=colour, marker=marker, lw=2, label=name)
        axes[row, 1].set_title(f"Frame {frame}, KP{channel}: {label}")
        axes[row, 1].legend(fontsize=8, loc="upper right")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle(f"Visual spike audit — {basis}")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"worst_material_channel": worst_channel, "selections": [{"channel": c, "frame": f, "reason": label} for c, f, label in selections]}


def evaluate(raw_receipt_path: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    raw_receipt_record = _file_record(raw_receipt_path)
    receipt = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw receipt schema differs")
    _require(receipt.get("frame_order_reversal_exact") is True, "raw frame-order proof failed")
    _require(receipt.get("masks_or_geometry_opened") is False, "raw stage opened masks or geometry")
    _require(receipt.get("forbidden_inputs_opened") == [], "raw stage opened a forbidden input")
    raw_record = receipt.get("raw_arrays")
    _require(isinstance(raw_record, Mapping), "raw array binding missing")
    observed_raw = _file_record(str(raw_record["absolute_path"]))
    _require(observed_raw == dict(raw_record), "raw arrays changed before evaluation")
    with np.load(observed_raw["absolute_path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _require(arrays["decoded_coordinate"].shape == (2, EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), "decoded shape differs")

    report_record = receipt.get("role", {}).get("source_forensic_report")
    _require(isinstance(report_record, Mapping), "source forensic report binding missing")
    source_report = _load_bound_json(report_record, name="source forensic report")
    theta, masks, image_paths = _load_geometry_and_masks(source_report)

    output_dir.mkdir(parents=True, exist_ok=False)
    basis_reports: dict[str, Any] = {}
    visuals: dict[str, Any] = {}
    evaluation_arrays: dict[str, np.ndarray] = {}
    for basis_index, basis in enumerate(BASIS_NAMES):
        metrics, derived = _basis_metrics(
            arrays["decoded_coordinate"][basis_index],
            arrays["detector_coordinate"][basis_index],
            arrays["anchor_coordinate"][basis_index],
            theta,
            masks,
        )
        margin = arrays["top1_top2_margin"][basis_index]
        metrics["similarity_margin"] = _summary(margin, unit="cosine similarity", sample_unit="correlated frames")
        basis_reports[basis] = metrics
        trajectory_path = output_dir / f"trajectories__{basis}.png"
        time_path = output_dir / f"coordinate_time__{basis}.png"
        montage_path = output_dir / f"worst_events__{basis}.png"
        _plot_trajectories(trajectory_path, arrays["decoded_coordinate"][basis_index], derived["physical_coordinate"], derived["canonical_coordinate"], basis)
        _plot_time_series(time_path, derived["material_error_pixels"], derived["cyclic_second_difference_cells"], basis)
        montage = _plot_worst_events(
            montage_path,
            image_paths,
            arrays["decoded_coordinate"][basis_index],
            arrays["detector_coordinate"][basis_index],
            derived["physical_coordinate"],
            derived["material_error_pixels"],
            derived["cyclic_second_difference_cells"],
            basis,
        )
        visuals[basis] = {
            "trajectories": _file_record(trajectory_path),
            "coordinate_time": _file_record(time_path),
            "worst_events": _file_record(montage_path),
            "montage_selection": montage,
        }
        for name, value in derived.items():
            evaluation_arrays[f"{basis}__{name}"] = value

    arrays_path = output_dir / "evaluation_arrays.npz"
    np.savez_compressed(arrays_path, theta_deg=theta, **evaluation_arrays)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    implementation = {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES}
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "post_hash_physical_evaluation_of_frozen_feature_decode",
        "role": receipt["role"],
        "raw_receipt": raw_receipt_record,
        "raw_arrays": observed_raw,
        "source_forensic_report": dict(report_record),
        "implementation_head": head,
        "implementation_sources": implementation,
        "raw_prediction_hash_fixed_before_masks_or_theta_opened": True,
        "geometry": {
            "physical_transform": "world-Z roll, +2 degrees per frame",
            "pivot_normalized": [0.0, 0.0],
            "material_track": "R(theta_frame - theta_anchor) applied to frame-27 anchor",
            "canonical_operation": "R(-theta_frame) applied only after raw hash",
        },
        "frozen_r64_diagnostic_thresholds": HARD_R64_THRESHOLDS,
        "basis_reports": basis_reports,
        "visuals": visuals,
        "evaluation_arrays": _file_record(arrays_path),
        "statistical_scope": {
            "statistics": "median, empirical q90/q99, maximum, mean, and RMS",
            "unit": "fixed keypoint channel over one 180-frame rendered orbit",
            "n_frames_per_channel": EXPECTED_FRAMES,
            "n_channels": EXPECTED_CHANNELS,
            "descriptive_not_inferential": True,
            "correlation_caveat": "frames share one orbit and adjacent/second-difference samples overlap; no naive SEM or population CI is reported",
        },
        "training_or_weight_update_performed": False,
    }
    result_path = output_dir / "feature_decode_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"report": _file_record(result_path), "hard_pass_count": basis_reports["hard_peak"]["strict_pass_count"], "soft_pass_count": basis_reports["soft_coordinate"]["strict_pass_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory already exists; use a fresh evaluation attempt")
    print(json.dumps(evaluate(args.raw_receipt.resolve(strict=True), args.output_dir, args.repo_root.resolve(strict=True)), sort_keys=True))


if __name__ == "__main__":
    main()
