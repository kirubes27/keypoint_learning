"""Evaluate hashed adjacent feature re-anchoring against renderer geometry."""

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
    from .adjacent_feature_reanchor import NORMALIZATION_EPSILON, endpoint_coordinates_to_cells
    from .frozen_feature_forensics import coordinate_on_mask
    from .frozen_wobble_forensics import canonicalize, rotate_points
except ImportError:
    from adjacent_feature_reanchor import NORMALIZATION_EPSILON, endpoint_coordinates_to_cells  # type: ignore
    from frozen_feature_forensics import coordinate_on_mask  # type: ignore
    from frozen_wobble_forensics import canonicalize, rotate_points  # type: ignore


SCHEMA_VERSION = "adjacent_feature_reanchor_evaluation.v1"
EXPECTED_RAW_SCHEMA = "adjacent_feature_reanchor_raw_receipt.v1"
EXPECTED_MANIFEST_SCHEMA = "adjacent_feature_reanchor_manifest.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
PIXEL_SCALE = 255.5
CELL_SCALE = 31.5
MATERIAL_ERROR_MAX = float(np.nextafter(math.sqrt(2.0) / 63.0, math.inf))
MINIMUM_BORDER_PX = 4.055555555555555
MINIMUM_PAIR_DISTANCE = 2.0 / 63.0
MINIMUM_RAW_ORBIT_RMS = 1.0 / 63.0
IMPLEMENTATION_SOURCES = (
    "keypoint_net/evaluate_adjacent_feature_reanchor.py",
    "keypoint_net/adjacent_feature_reanchor.py",
    "keypoint_net/frozen_feature_forensics.py",
    "keypoint_net/frozen_wobble_forensics.py",
)


class AdjacentFeatureEvaluationError(ValueError):
    """Raised when an adjacent evaluation is not source-bound or meaningful."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjacentFeatureEvaluationError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _load_geometry_and_masks(report: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    corpus = report.get("bindings", {}).get("corpus")
    _require(isinstance(corpus, Mapping), "source corpus binding missing")
    object_root = Path(str(corpus["object_root"])).resolve(strict=True)
    frames = corpus.get("frame_records")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "source frame count differs")
    theta = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    masks = np.empty((EXPECTED_FRAMES, 512, 512), dtype=bool)
    images: list[Path] = []
    for index, row in enumerate(frames):
        _require(int(row.get("frame_index", -1)) == index, "source frame index differs")
        theta[index] = float(row["theta_deg"])
        mask_path = (object_root / str(row["mask_relpath"])).resolve(strict=True)
        image_path = (object_root / str(row["image_relpath"])).resolve(strict=True)
        _require(_sha256(mask_path) == row["mask_sha256"], f"mask hash differs at frame {index}")
        _require(_sha256(image_path) == row["image_sha256"], f"image hash differs at frame {index}")
        with Image.open(mask_path) as image:
            masks[index] = np.asarray(image.convert("L")) > 0
        images.append(image_path)
    _require(np.array_equal(theta, np.arange(EXPECTED_FRAMES, dtype=np.float64) * 2.0), "theta is not exact +2-degree roll")
    return theta, masks, images


def _coordinate_mask_membership(points: np.ndarray, masks: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    result = np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=bool)
    for source in range(EXPECTED_FRAMES):
        for channel in range(EXPECTED_CHANNELS):
            result[source, channel] = coordinate_on_mask(masks[int(frame_indices[source])], points[source, channel])
    return result


def _border_distance_px(points: np.ndarray) -> np.ndarray:
    x = points[..., 0]
    y = points[..., 1]
    return np.minimum.reduce((x + 1.0, 1.0 - x, y + 1.0, 1.0 - y)) * PIXEL_SCALE


def _pairwise_minimum(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    pairs = [(left, right) for left in range(EXPECTED_CHANNELS) for right in range(left + 1, EXPECTED_CHANNELS)]
    distance = np.stack([np.linalg.norm(points[:, left] - points[:, right], axis=-1) for left, right in pairs], axis=1)
    minimum = np.full((EXPECTED_FRAMES, EXPECTED_CHANNELS), np.inf, dtype=np.float64)
    for index, (left, right) in enumerate(pairs):
        minimum[:, left] = np.minimum(minimum[:, left], distance[:, index])
        minimum[:, right] = np.minimum(minimum[:, right], distance[:, index])
    return minimum, distance, pairs


def _distribution(values: np.ndarray, *, unit: str, sample_unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "distribution input must be nonempty 1D")
    return {
        "unit": unit,
        "sample_unit": sample_unit,
        "n": int(array.size),
        "descriptive_not_inferential": True,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def adjacent_metrics(
    *,
    source: np.ndarray,
    target_detector: np.ndarray,
    self_decoded: np.ndarray,
    adjacent_decoded: np.ndarray,
    raw_norm: np.ndarray,
    self_margin: np.ndarray,
    adjacent_margin: np.ndarray,
    masks: np.ndarray,
    target_index: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_coordinate_shape = (EXPECTED_FRAMES, EXPECTED_CHANNELS, 2)
    expected_scalar_shape = (EXPECTED_FRAMES, EXPECTED_CHANNELS)
    for name, value in {
        "source": source,
        "target_detector": target_detector,
        "self_decoded": self_decoded,
        "adjacent_decoded": adjacent_decoded,
    }.items():
        _require(value.shape == expected_coordinate_shape and np.isfinite(value).all(), f"{name} shape/finite check failed")
    for name, value in {"raw_norm": raw_norm, "self_margin": self_margin, "adjacent_margin": adjacent_margin}.items():
        _require(value.shape == expected_scalar_shape and np.isfinite(value).all(), f"{name} shape/finite check failed")
    _require(np.array_equal(target_index, np.remainder(np.arange(EXPECTED_FRAMES) + 1, EXPECTED_FRAMES)), "target index differs")

    physical = rotate_points(source, 2.0)
    feature_error = np.linalg.norm(adjacent_decoded - physical, axis=-1)
    detector_error = np.linalg.norm(target_detector - physical, axis=-1)
    feature_good = feature_error <= MATERIAL_ERROR_MAX
    detector_good = detector_error <= MATERIAL_ERROR_MAX
    source_cells = endpoint_coordinates_to_cells(source)
    self_cells = endpoint_coordinates_to_cells(self_decoded)
    self_same_cell = np.all(source_cells == self_cells, axis=-1)
    source_on_object = _coordinate_mask_membership(source, masks, np.arange(EXPECTED_FRAMES, dtype=np.int64))
    physical_on_object = _coordinate_mask_membership(physical, masks, target_index)
    adjacent_on_object = _coordinate_mask_membership(adjacent_decoded, masks, target_index)
    detector_on_object = _coordinate_mask_membership(target_detector, masks, target_index)
    eligible = source_on_object & physical_on_object
    minimum_other, pairwise_distance, pairs = _pairwise_minimum(adjacent_decoded)
    raw_centred = source - np.mean(source, axis=0, keepdims=True)
    raw_rms = np.sqrt(np.mean(np.sum(np.square(raw_centred), axis=-1), axis=0))
    combined_border = np.minimum.reduce(
        (_border_distance_px(source), _border_distance_px(physical), _border_distance_px(adjacent_decoded))
    )

    channels: list[dict[str, Any]] = []
    for channel in range(EXPECTED_CHANNELS):
        eligible_channel = eligible[:, channel]
        eligible_count = int(np.sum(eligible_channel))
        checks = {
            "descriptor_nonzero_all_frames": bool(np.all(raw_norm[:, channel] > NORMALIZATION_EPSILON)),
            "self_retrieval_same_cell_all_frames": bool(np.all(self_same_cell[:, channel])),
            "source_on_object_all_frames": bool(np.all(source_on_object[:, channel])),
            "physical_target_on_object_all_frames": bool(np.all(physical_on_object[:, channel])),
            "adjacent_decode_on_object_all_frames": bool(np.all(adjacent_on_object[:, channel])),
            "border_safe_all_frames": bool(np.min(combined_border[:, channel]) >= MINIMUM_BORDER_PX),
            "source_active": bool(raw_rms[channel] >= MINIMUM_RAW_ORBIT_RMS),
            "adjacent_decode_distinct_all_frames": bool(np.min(minimum_other[:, channel]) >= MINIMUM_PAIR_DISTANCE),
            "adjacent_material_all_edges": bool(np.all(feature_good[:, channel])),
        }
        feature_px = feature_error[:, channel] * PIXEL_SCALE
        detector_px = detector_error[:, channel] * PIXEL_SCALE
        channel_report = {
            "channel": channel,
            "strict_adjacent_local_pass": all(checks.values()),
            "checks": checks,
            "eligible_on_object_edge_count": eligible_count,
            "eligible_on_object_edge_fraction": float(np.mean(eligible_channel)),
            "adjacent_material_success_fraction_all_edges": float(np.mean(feature_good[:, channel])),
            "adjacent_material_success_fraction_eligible_edges": (
                float(np.mean(feature_good[eligible_channel, channel])) if eligible_count else None
            ),
            "detector_local_success_fraction_all_edges": float(np.mean(detector_good[:, channel])),
            "feature_better_than_detector_fraction": float(np.mean(feature_error[:, channel] < detector_error[:, channel])),
            "detector_bad_feature_good_count": int(np.sum((~detector_good[:, channel]) & feature_good[:, channel])),
            "detector_bad_feature_bad_count": int(np.sum((~detector_good[:, channel]) & (~feature_good[:, channel]))),
            "detector_good_feature_bad_count": int(np.sum(detector_good[:, channel] & (~feature_good[:, channel]))),
            "both_good_count": int(np.sum(detector_good[:, channel] & feature_good[:, channel])),
            "feature_material_error": _distribution(feature_px, unit="512-input pixels", sample_unit="cyclic adjacent edge"),
            "detector_local_material_error": _distribution(detector_px, unit="512-input pixels", sample_unit="cyclic adjacent edge"),
            "source_descriptor_raw_norm": _distribution(raw_norm[:, channel], unit="L2 norm", sample_unit="source frame"),
            "self_separated_cosine_margin": _distribution(self_margin[:, channel], unit="cosine similarity", sample_unit="source frame"),
            "adjacent_separated_cosine_margin": _distribution(adjacent_margin[:, channel], unit="cosine similarity", sample_unit="cyclic adjacent edge"),
            "source_on_object_rate": float(np.mean(source_on_object[:, channel])),
            "physical_target_on_object_rate": float(np.mean(physical_on_object[:, channel])),
            "adjacent_decode_on_object_rate": float(np.mean(adjacent_on_object[:, channel])),
            "target_detector_on_object_rate": float(np.mean(detector_on_object[:, channel])),
            "source_orbit_rms_normalized": float(raw_rms[channel]),
            "minimum_other_adjacent_decode_distance_normalized": float(np.min(minimum_other[:, channel])),
        }
        channels.append(channel_report)

    return {
        "strict_pass_count": int(sum(row["strict_adjacent_local_pass"] for row in channels)),
        "strict_all_ten_pass": all(row["strict_adjacent_local_pass"] for row in channels),
        "self_retrieval_same_cell_count": int(np.sum(self_same_cell)),
        "self_retrieval_total": EXPECTED_FRAMES * EXPECTED_CHANNELS,
        "adjacent_material_success_count": int(np.sum(feature_good)),
        "adjacent_material_total": EXPECTED_FRAMES * EXPECTED_CHANNELS,
        "detector_local_success_count": int(np.sum(detector_good)),
        "detector_local_total": EXPECTED_FRAMES * EXPECTED_CHANNELS,
        "channels": channels,
        "fixed_channel_identity_only": True,
        "pair_count": len(pairs),
    }, {
        "physical_target_coordinate": physical,
        "feature_material_error_pixels": feature_error * PIXEL_SCALE,
        "detector_local_material_error_pixels": detector_error * PIXEL_SCALE,
        "feature_material_success": feature_good,
        "detector_local_success": detector_good,
        "self_same_cell": self_same_cell,
        "source_on_object": source_on_object,
        "physical_target_on_object": physical_on_object,
        "adjacent_decode_on_object": adjacent_on_object,
        "target_detector_on_object": detector_on_object,
        "eligible_on_object_edge": eligible,
        "minimum_other_adjacent_decode_distance_normalized": minimum_other,
        "pairwise_adjacent_decode_distance_normalized": pairwise_distance,
        "pair_channel_indices": np.asarray(pairs, dtype=np.int64),
    }


def _plot_errors(arrays: Mapping[str, np.ndarray], output: Path) -> None:
    threshold = MATERIAL_ERROR_MAX * PIXEL_SCALE
    frames = np.arange(EXPECTED_FRAMES)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for channel in range(EXPECTED_CHANNELS):
        axes[0].plot(frames, arrays["feature_material_error_pixels"][:, channel], lw=0.9, label=f"KP{channel}")
        axes[1].plot(frames, arrays["detector_local_material_error_pixels"][:, channel], lw=0.9, label=f"KP{channel}")
    axes[0].axhline(threshold, color="black", ls="--", lw=1.0, label="r64 clean envelope")
    axes[1].axhline(threshold, color="black", ls="--", lw=1.0, label="r64 clean envelope")
    axes[0].set_ylabel("adjacent feature error (input px)")
    axes[1].set_ylabel("original detector local error (input px)")
    axes[1].set_xlabel("source frame; target is cyclic t+1")
    axes[0].set_title("Adjacent learned-feature re-anchoring versus detector wobble")
    axes[0].legend(ncol=6, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_canonical(
    source: np.ndarray,
    target_detector: np.ndarray,
    adjacent: np.ndarray,
    physical: np.ndarray,
    theta: np.ndarray,
    target_index: np.ndarray,
    output: Path,
) -> None:
    source_can = canonicalize(source, theta)
    target_theta = theta[target_index].copy()
    target_theta[-1] = 360.0
    detector_can = canonicalize(target_detector, target_theta)
    adjacent_can = canonicalize(adjacent, target_theta)
    physical_can = canonicalize(physical, target_theta)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for channel, axis in enumerate(axes.flat):
        axis.plot(source_can[:, channel, 0], source_can[:, channel, 1], color="0.65", lw=0.8, label="source detector")
        axis.scatter(detector_can[:, channel, 0], detector_can[:, channel, 1], s=5, c="cyan", alpha=0.55, label="target detector")
        axis.scatter(adjacent_can[:, channel, 0], adjacent_can[:, channel, 1], s=5, c="red", alpha=0.55, label="adjacent feature")
        axis.scatter(physical_can[:, channel, 0], physical_can[:, channel, 1], s=5, c="lime", alpha=0.45, label="transported source")
        axis.set_title(f"KP{channel}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Canonical adjacent re-anchoring trajectories; 180 correlated cyclic edges")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _to_pixel(point: np.ndarray) -> tuple[float, float]:
    return float((point[0] + 1.0) * PIXEL_SCALE), float((point[1] + 1.0) * PIXEL_SCALE)


def _plot_spikes(
    image_paths: list[Path],
    source: np.ndarray,
    target_detector: np.ndarray,
    adjacent: np.ndarray,
    physical: np.ndarray,
    feature_error_px: np.ndarray,
    detector_error_px: np.ndarray,
    target_index: np.ndarray,
    output: Path,
) -> None:
    kp2 = 2
    worst_channel = int(np.unravel_index(np.argmax(feature_error_px), feature_error_px.shape)[1])
    candidates: list[tuple[int, int, str]] = [
        (int(np.argmax(feature_error_px[:, kp2])), kp2, "KP2 largest feature error"),
        (int(np.argmax(detector_error_px[:, kp2])), kp2, "KP2 largest detector wobble"),
        (int(np.argmax(feature_error_px[:, worst_channel])), worst_channel, "role largest feature error"),
    ]
    correction = np.where((detector_error_px > MATERIAL_ERROR_MAX * PIXEL_SCALE) & (feature_error_px <= MATERIAL_ERROR_MAX * PIXEL_SCALE))
    if correction[0].size:
        index = int(np.argmax(detector_error_px[correction]))
        candidates.append((int(correction[0][index]), int(correction[1][index]), "detector bad / feature good"))
    else:
        candidates.append((int(np.argmax(detector_error_px)), int(np.unravel_index(np.argmax(detector_error_px), detector_error_px.shape)[1]), "largest detector wobble"))
    fig, axes = plt.subplots(len(candidates), 2, figsize=(10, 5 * len(candidates)))
    for row, (frame, channel, label) in enumerate(candidates):
        target = int(target_index[frame])
        with Image.open(image_paths[frame]) as image:
            axes[row, 0].imshow(image.convert("RGB"))
        with Image.open(image_paths[target]) as image:
            axes[row, 1].imshow(image.convert("RGB"))
        sx, sy = _to_pixel(source[frame, channel])
        axes[row, 0].scatter([sx], [sy], s=90, facecolors="none", edgecolors="yellow", linewidths=2.0)
        px, py = _to_pixel(physical[frame, channel])
        fx, fy = _to_pixel(adjacent[frame, channel])
        dx, dy = _to_pixel(target_detector[frame, channel])
        axes[row, 1].scatter([px], [py], s=90, c="lime", label="transported source")
        axes[row, 1].scatter([fx], [fy], s=90, c="red", marker="x", linewidths=2.0, label="adjacent feature")
        axes[row, 1].scatter([dx], [dy], s=90, c="cyan", marker="+", linewidths=2.0, label="target detector")
        axes[row, 0].set_title(f"source frame {frame}, KP{channel}")
        axes[row, 1].set_title(
            f"target {target}: {label}\nfeature={feature_error_px[frame, channel]:.2f}px, detector={detector_error_px[frame, channel]:.2f}px"
        )
        axes[row, 1].legend(fontsize=8)
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    receipt_record = _file_record(args.raw_receipt.resolve(strict=True))
    receipt = json.loads(args.raw_receipt.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw receipt schema differs")
    _require(receipt.get("artifact_type") == "geometry_blind_adjacent_feature_reanchor_raw_receipt", "raw receipt type differs")
    for key, expected in {
        "frame_order_reversal_exact": True,
        "model_state_unchanged": True,
        "masks_or_geometry_opened": False,
        "previous_decode_propagated": False,
        "target_detector_used_to_choose_or_centre_search": False,
        "optimizer_constructed": False,
        "gradients_enabled": False,
        "training_or_weight_update_performed": False,
    }.items():
        _require(receipt.get(key) is expected, f"raw receipt {key} differs")
    _require(receipt.get("forbidden_inputs_opened") == [], "raw receipt lists forbidden inputs")
    manifest = _load_bound_json(receipt["manifest"], name="adjacent manifest")
    _require(manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "adjacent manifest schema differs")
    raw_record = _file_record(str(receipt["raw_arrays"]["absolute_path"]))
    _require(raw_record == receipt["raw_arrays"], "raw array binding differs")

    with np.load(raw_record["absolute_path"], allow_pickle=False) as loaded:
        raw = {name: loaded[name].copy() for name in loaded.files}
    required = {
        "source_detector_coordinate",
        "target_detector_coordinate",
        "source_descriptor_raw_norm",
        "self_decoded_coordinate",
        "self_top1_top2_margin",
        "adjacent_decoded_coordinate",
        "adjacent_top1_top2_margin",
        "source_frame_index",
        "target_frame_index",
    }
    _require(required.issubset(raw), "raw arrays are incomplete")
    _require(np.array_equal(raw["source_frame_index"], np.arange(EXPECTED_FRAMES)), "source frame index differs")
    _require(
        np.array_equal(raw["target_frame_index"], np.remainder(np.arange(EXPECTED_FRAMES) + 1, EXPECTED_FRAMES)),
        "target frame index differs",
    )
    # The exact raw hash and array semantics are fixed before geometry is opened below.
    raw_prediction_hash_fixed_before_geometry = True

    role = receipt.get("role")
    _require(isinstance(role, Mapping), "role binding missing")
    source_report = _load_bound_json(role["source_forensic_report"], name="source forensic report")
    theta, masks, image_paths = _load_geometry_and_masks(source_report)
    report, derived = adjacent_metrics(
        source=raw["source_detector_coordinate"],
        target_detector=raw["target_detector_coordinate"],
        self_decoded=raw["self_decoded_coordinate"],
        adjacent_decoded=raw["adjacent_decoded_coordinate"],
        raw_norm=raw["source_descriptor_raw_norm"],
        self_margin=raw["self_top1_top2_margin"],
        adjacent_margin=raw["adjacent_top1_top2_margin"],
        masks=masks,
        target_index=raw["target_frame_index"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "adjacent_feature_evaluation_arrays.npz"
    np.savez_compressed(arrays_path, theta_deg=theta, **derived)
    error_plot = args.output_dir / "adjacent_error_over_time.png"
    canonical_plot = args.output_dir / "canonical_adjacent_trajectories.png"
    spike_plot = args.output_dir / "worst_events.png"
    _plot_errors(derived, error_plot)
    _plot_canonical(
        raw["source_detector_coordinate"],
        raw["target_detector_coordinate"],
        raw["adjacent_decoded_coordinate"],
        derived["physical_target_coordinate"],
        theta,
        raw["target_frame_index"],
        canonical_plot,
    )
    _plot_spikes(
        image_paths,
        raw["source_detector_coordinate"],
        raw["target_detector_coordinate"],
        raw["adjacent_decoded_coordinate"],
        derived["physical_target_coordinate"],
        derived["feature_material_error_pixels"],
        derived["detector_local_material_error_pixels"],
        raw["target_frame_index"],
        spike_plot,
    )
    repo_root = Path(__file__).resolve().parents[1]
    implementation_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    implementation = {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "adjacent_feature_reanchor_evaluation",
        "role": dict(role),
        "raw_receipt": receipt_record,
        "raw_arrays": raw_record,
        "raw_prediction_hash_fixed_before_masks_or_theta_opened": raw_prediction_hash_fixed_before_geometry,
        "source_forensic_report": dict(role["source_forensic_report"]),
        "geometry": {
            "physical_target": "R(+2 degrees) around normalized pivot (0,0)",
            "target_rule": "(source + 1) mod 180, with frame 179 -> 0 treated as +2 degrees",
            "renderer_control": "mask_geometry_control_v1 independently locks sign and pivot",
        },
        "frozen_thresholds": {
            "material_error_max_normalized": MATERIAL_ERROR_MAX,
            "material_error_max_input_pixels": MATERIAL_ERROR_MAX * PIXEL_SCALE,
            "descriptor_raw_norm_minimum_exclusive": NORMALIZATION_EPSILON,
            "self_retrieval": "same endpoint-aligned r64 cell",
            "minimum_image_border_distance_px": MINIMUM_BORDER_PX,
            "minimum_pair_distance_normalized": MINIMUM_PAIR_DISTANCE,
            "minimum_raw_orbit_rms_normalized": MINIMUM_RAW_ORBIT_RMS,
        },
        "report": report,
        "statistical_scope": {
            "descriptive_not_inferential": True,
            "sample_unit": "correlated cyclic adjacent edge, n=180 per role-channel",
            "correlation_caveat": "one closed hammer orbit; adjacent edges overlap in frames; no SEM or population CI",
            "statistics": "empirical mean, median, q90, q99, maximum, and success fraction",
        },
        "implementation_head": implementation_head,
        "implementation_sources": implementation,
        "evaluation_arrays": _file_record(arrays_path),
        "visuals": {
            "adjacent_error_over_time": _file_record(error_plot),
            "canonical_adjacent_trajectories": _file_record(canonical_plot),
            "worst_events": _file_record(spike_plot),
        },
        "training_or_weight_update_performed": False,
    }
    output = args.output_dir / "adjacent_feature_evaluation.json"
    output.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "strict_pass_count": report["strict_pass_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
