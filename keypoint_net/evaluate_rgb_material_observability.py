"""Evaluate hashed adjacent RGB matches against masks and physical material motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

try:
    from .frozen_wobble_forensics import rotate_points
    from .rgb_material_observability import (
        PATCH_SIZES,
        RGBObservabilityConfig,
        candidate_coordinate_grids,
        candidate_rank,
        decode_score_map,
        flatten_decode,
        local_candidate_mask,
        normalized_to_pixel,
        patch_inside,
        rgb_correlation_map,
    )
except ImportError:
    from frozen_wobble_forensics import rotate_points  # type: ignore
    from rgb_material_observability import (  # type: ignore
        PATCH_SIZES,
        RGBObservabilityConfig,
        candidate_coordinate_grids,
        candidate_rank,
        decode_score_map,
        flatten_decode,
        local_candidate_mask,
        normalized_to_pixel,
        patch_inside,
        rgb_correlation_map,
    )


SCHEMA_VERSION = "rgb_material_observability_evaluation.v2"
EXPECTED_RAW_SCHEMA = "rgb_material_observability_raw_receipt.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
SCOPES = ("global", "local")
PIXEL_SCALE = 255.5
MATERIAL_ERROR_LIMIT_PX = float(np.nextafter(math.sqrt(2.0), math.inf))
MINIMUM_PAIR_DISTANCE_PX = 8.11111111111111
MINIMUM_ACTIVITY_RMS_PX = 4.055555555555555
IMPLEMENTATION_SOURCES = (
    "keypoint_net/rgb_material_observability.py",
    "keypoint_net/evaluate_rgb_material_observability.py",
    "keypoint_net/frozen_wobble_forensics.py",
)


class RGBEvaluationError(ValueError):
    """Raised when post-hash evidence or evaluation semantics differ."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBEvaluationError(message)


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


def _load_raw(receipt_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    receipt_record = _file_record(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw receipt schema differs")
    _require(receipt.get("frame_order_reversal_exact") is True, "raw reverse-order proof failed")
    _require(receipt.get("masks_or_geometry_opened") is False, "raw stage opened geometry")
    _require(receipt.get("forbidden_inputs_opened") == [], "raw stage opened a forbidden input")
    _require(receipt.get("optimizer_gradient_or_weight_update_used") is False, "raw stage updated weights")
    raw_record = receipt.get("raw_arrays")
    _require(isinstance(raw_record, Mapping), "raw arrays binding missing")
    observed = _file_record(str(raw_record["absolute_path"]))
    _require(observed == dict(raw_record), "raw arrays changed before evaluation")
    with np.load(observed["absolute_path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected_shape = (len(PATCH_SIZES), EXPECTED_FRAMES, EXPECTED_CHANNELS)
    _require(arrays["source_valid"].shape == expected_shape, "source-valid shape differs")
    _require(arrays["source_coordinate_normalized"].shape == (EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), "source coordinate shape differs")
    _require(np.array_equal(arrays["patch_size"], np.asarray(PATCH_SIZES)), "patch sizes differ")
    _require(np.array_equal(arrays["target_frame_index"], np.roll(np.arange(EXPECTED_FRAMES), -1)), "target pairing differs")
    return receipt, arrays, receipt_record


def _load_images_and_geometry(receipt: Mapping[str, Any]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[Path]]:
    report_record = receipt.get("role", {}).get("source_forensic_report")
    _require(isinstance(report_record, Mapping), "source forensic report binding missing")
    report = _load_bound_json(report_record, name="source forensic report")
    corpus = report.get("bindings", {}).get("corpus")
    _require(isinstance(corpus, Mapping), "source corpus binding missing")
    object_root = Path(str(corpus["object_root"])).resolve(strict=True)
    rows = corpus.get("frame_records")
    _require(isinstance(rows, list) and len(rows) == EXPECTED_FRAMES, "source frame count differs")
    images: list[np.ndarray] = []
    masks = np.empty((EXPECTED_FRAMES, 512, 512), dtype=bool)
    theta = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    image_paths: list[Path] = []
    for frame, row in enumerate(rows):
        _require(int(row.get("frame_index", -1)) == frame, "source frame index differs")
        theta[frame] = float(row["theta_deg"])
        image_path = (object_root / str(row["image_relpath"])).resolve(strict=True)
        mask_path = (object_root / str(row["mask_relpath"])).resolve(strict=True)
        _require(_sha256(image_path) == row["image_sha256"], f"image hash differs at frame {frame}")
        _require(_sha256(mask_path) == row["mask_sha256"], f"mask hash differs at frame {frame}")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        with Image.open(mask_path) as image:
            masks[frame] = np.asarray(image.convert("L")) > 0
        images.append(np.ascontiguousarray(rgb))
        image_paths.append(image_path)
    _require(np.array_equal(theta, np.arange(EXPECTED_FRAMES, dtype=np.float64) * 2.0), "theta differs from exact +2-degree orbit")
    return images, masks, theta, image_paths


def _points_on_masks(points_px: np.ndarray, masks: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    _require(points_px.shape == (EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), "mask point shape differs")
    result = np.zeros((EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=bool)
    for row in range(EXPECTED_FRAMES):
        mask = masks[int(frame_indices[row])]
        for channel in range(EXPECTED_CHANNELS):
            x, y = points_px[row, channel]
            if 0.0 <= x <= 511.0 and 0.0 <= y <= 511.0:
                result[row, channel] = bool(mask[int(np.rint(y)), int(np.rint(x))])
    return result


def _source_state(source_px: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    source_on_object = _points_on_masks(source_px, masks, np.arange(EXPECTED_FRAMES))
    centred = source_px - np.mean(source_px, axis=0, keepdims=True)
    activity_rms = np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=0))
    minimum_other = np.full((EXPECTED_FRAMES, EXPECTED_CHANNELS), np.inf, dtype=np.float64)
    for left in range(EXPECTED_CHANNELS):
        for right in range(left + 1, EXPECTED_CHANNELS):
            distance = np.linalg.norm(source_px[:, left] - source_px[:, right], axis=-1)
            minimum_other[:, left] = np.minimum(minimum_other[:, left], distance)
            minimum_other[:, right] = np.minimum(minimum_other[:, right], distance)
    return {
        "source_on_object": source_on_object,
        "activity_rms_px": activity_rms,
        "active": activity_rms >= MINIMUM_ACTIVITY_RMS_PX,
        "on_object_all_frames": np.all(source_on_object, axis=0),
        "minimum_other_channel_px": minimum_other,
        "distinct_all_frames": np.all(minimum_other >= MINIMUM_PAIR_DISTANCE_PX, axis=0),
    }


def _verify_scalar(observed: Any, expected: Any, *, context: str) -> None:
    if isinstance(expected, (bool, np.bool_)):
        _require(bool(observed) == bool(expected), f"{context} boolean differs")
    elif isinstance(expected, (int, np.integer)):
        _require(int(observed) == int(expected), f"{context} integer differs")
    else:
        _require(float(observed) == float(expected), f"{context} scalar differs")


def _recompute_and_rank(
    images: Sequence[np.ndarray],
    arrays: Mapping[str, np.ndarray],
    physical_target_px: np.ndarray,
    *,
    config: RGBObservabilityConfig,
) -> dict[str, np.ndarray]:
    shape = (len(PATCH_SIZES), EXPECTED_FRAMES, EXPECTED_CHANNELS)
    result: dict[str, np.ndarray] = {}
    for scope in SCOPES:
        result[f"{scope}_physical_candidate_valid"] = np.zeros(shape, dtype=bool)
        result[f"{scope}_physical_candidate_score"] = np.full(shape, -2.0, dtype=np.float64)
        result[f"{scope}_physical_candidate_rank"] = np.zeros(shape, dtype=np.int64)
        result[f"{scope}_physical_candidate_rank_percentile"] = np.zeros(shape, dtype=np.float64)
        result[f"{scope}_physical_candidate_distance_px"] = np.full(shape, np.inf, dtype=np.float64)
    for scale_index, patch_size in enumerate(PATCH_SIZES):
        for frame in range(EXPECTED_FRAMES):
            target_frame = (frame + 1) % EXPECTED_FRAMES
            for channel in range(EXPECTED_CHANNELS):
                centre = arrays["source_coordinate_px"][frame, channel]
                scores, evidence = rgb_correlation_map(
                    images[frame], images[target_frame], centre, patch_size, config=config
                )
                _require(
                    bool(evidence["source_patch_inside_and_informative"])
                    == bool(arrays["source_valid"][scale_index, frame, channel]),
                    "recomputed source validity differs",
                )
                _verify_scalar(
                    arrays["source_patch_rms"][scale_index, frame, channel],
                    evidence["source_patch_rms"],
                    context="source patch RMS",
                )
                if scores is None:
                    continue
                decoded = flatten_decode(
                    decode_score_map(
                        scores,
                        centre,
                        patch_size,
                        evidence=evidence,
                        config=config,
                    )
                )
                for name, expected in decoded.items():
                    observed = arrays[name][scale_index, frame, channel]
                    if np.ndim(expected) > 0:
                        _require(np.array_equal(observed, expected), f"recomputed {name} differs")
                    else:
                        _verify_scalar(observed, expected, context=f"recomputed {name}")
                if not patch_inside(physical_target_px[frame, channel], patch_size):
                    continue
                local_mask = local_candidate_mask(scores.shape, patch_size, centre, radius_px=config.local_radius_px)
                for scope in SCOPES:
                    ranking = candidate_rank(
                        scores,
                        patch_size,
                        physical_target_px[frame, channel],
                        allowed=local_mask if scope == "local" else None,
                    )
                    result[f"{scope}_physical_candidate_valid"][scale_index, frame, channel] = ranking["valid"]
                    result[f"{scope}_physical_candidate_score"][scale_index, frame, channel] = ranking["score"]
                    result[f"{scope}_physical_candidate_rank"][scale_index, frame, channel] = ranking["rank"]
                    result[f"{scope}_physical_candidate_rank_percentile"][scale_index, frame, channel] = ranking["rank_percentile"]
                    result[f"{scope}_physical_candidate_distance_px"][scale_index, frame, channel] = ranking["distance_px"]
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "minimum": None, "median": None, "mean": None, "q90": None, "q99": None, "maximum": None}
    return {
        "n": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "q90": float(np.quantile(finite, 0.90)),
        "q99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def _scope_metrics(
    arrays: Mapping[str, np.ndarray],
    ranks: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    physical_target_px: np.ndarray,
    physical_target_on_object: np.ndarray,
    masks: np.ndarray,
    target_frames: np.ndarray,
    scale_index: int,
    scope: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction = arrays[f"{scope}_top1_coordinate_px"][scale_index]
    valid = arrays[f"{scope}_valid"][scale_index]
    error = np.full((EXPECTED_FRAMES, EXPECTED_CHANNELS), np.nan, dtype=np.float64)
    error[valid] = np.linalg.norm(prediction[valid] - physical_target_px[valid], axis=-1)
    predicted_on_object = _points_on_masks(prediction, masks, target_frames) & valid
    grounded = source["source_on_object"] & physical_target_on_object
    pair_minimum = np.full((EXPECTED_FRAMES, EXPECTED_CHANNELS), np.inf, dtype=np.float64)
    for left in range(EXPECTED_CHANNELS):
        for right in range(left + 1, EXPECTED_CHANNELS):
            pair_valid = valid[:, left] & valid[:, right]
            distance = np.full(EXPECTED_FRAMES, np.nan, dtype=np.float64)
            distance[pair_valid] = np.linalg.norm(prediction[pair_valid, left] - prediction[pair_valid, right], axis=-1)
            pair_minimum[pair_valid, left] = np.minimum(pair_minimum[pair_valid, left], distance[pair_valid])
            pair_minimum[pair_valid, right] = np.minimum(pair_minimum[pair_valid, right], distance[pair_valid])
    source_valid_all = np.all(arrays["source_valid"][scale_index], axis=0)
    eligible = source["active"] & source["on_object_all_frames"] & source["distinct_all_frames"] & source_valid_all
    rank = ranks[f"{scope}_physical_candidate_rank"][scale_index]
    rank_valid = ranks[f"{scope}_physical_candidate_valid"][scale_index]
    channels = []
    for channel in range(EXPECTED_CHANNELS):
        strict_checks = {
            "source_active": bool(source["active"][channel]),
            "source_on_object_all_frames": bool(source["on_object_all_frames"][channel]),
            "source_distinct_all_frames": bool(source["distinct_all_frames"][channel]),
            "source_patch_valid_all_frames": bool(source_valid_all[channel]),
            "match_valid_all_frames": bool(np.all(valid[:, channel])),
            "material_error_all_edges": bool(np.all(valid[:, channel]) and np.nanmax(error[:, channel]) <= MATERIAL_ERROR_LIMIT_PX),
            "predicted_target_on_object_all_frames": bool(np.all(predicted_on_object[:, channel])),
            "no_cross_channel_collision_all_frames": bool(np.all(pair_minimum[:, channel] >= MINIMUM_PAIR_DISTANCE_PX)),
        }
        grounded_rows = grounded[:, channel]
        grounded_failures = grounded_rows & (
            ~valid[:, channel]
            | (np.nan_to_num(error[:, channel], nan=np.inf) > MATERIAL_ERROR_LIMIT_PX)
            | ~predicted_on_object[:, channel]
        )
        grounded_matcher_assessable = bool(np.any(grounded_rows))
        grounded_matcher_strict_pass = bool(
            grounded_matcher_assessable and not np.any(grounded_failures)
        )
        channels.append(
            {
                "channel": channel,
                "source_eligible": bool(eligible[channel]),
                "strict_pass": bool(all(strict_checks.values())),
                "checks": strict_checks,
                "valid_edge_count": int(np.sum(valid[:, channel])),
                "grounded_edge_count": int(np.sum(grounded_rows)),
                "grounded_failure_count": int(np.sum(grounded_failures)),
                "grounded_matcher_assessable": grounded_matcher_assessable,
                "grounded_matcher_strict_pass": grounded_matcher_strict_pass,
                "material_error_px": _distribution(error[:, channel]),
                "predicted_target_on_object_rate": float(np.mean(predicted_on_object[:, channel])),
                "minimum_other_prediction_distance_px": (
                    float(np.min(pair_minimum[:, channel]))
                    if np.isfinite(pair_minimum[:, channel]).any()
                    else None
                ),
                "physical_candidate_top1_count": int(np.sum(rank_valid[:, channel] & (rank[:, channel] == 1))),
                "physical_candidate_rank": _distribution(np.where(rank_valid[:, channel], rank[:, channel], np.nan)),
                "top1_top2_margin": _distribution(arrays[f"{scope}_margin"][scale_index, :, channel]),
            }
        )
    return {
        "strict_pass_count": int(sum(row["strict_pass"] for row in channels)),
        "strict_all_ten_pass": bool(all(row["strict_pass"] for row in channels)),
        "source_eligible_count": int(np.sum(eligible)),
        "grounded_matcher_assessable_count": int(
            sum(row["grounded_matcher_assessable"] for row in channels)
        ),
        "grounded_matcher_strict_pass_count": int(
            sum(row["grounded_matcher_strict_pass"] for row in channels)
        ),
        "channels": channels,
        "material_error_px_all_valid_edges": _distribution(error),
        "physical_candidate_rank_all_valid": _distribution(
            np.where(rank_valid, rank, np.nan)
        ),
        "valid_edge_count": int(np.sum(valid)),
        "expected_edge_count": EXPECTED_FRAMES * EXPECTED_CHANNELS,
        "grounded_edge_count": int(np.sum(grounded)),
        "descriptive_not_inferential": True,
    }, {
        "prediction_px": prediction,
        "valid": valid,
        "material_error_px": error,
        "predicted_on_object": predicted_on_object,
        "minimum_other_prediction_distance_px": pair_minimum,
        "grounded": grounded,
    }


def _plot_time(
    output: Path,
    errors: np.ndarray,
    margins: np.ndarray,
    patch_size: int,
    scope: str,
) -> None:
    colours = plt.cm.tab10(np.arange(EXPECTED_CHANNELS))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for channel in range(EXPECTED_CHANNELS):
        axes[0].plot(np.arange(EXPECTED_FRAMES), errors[:, channel], color=colours[channel], lw=0.9, label=f"KP{channel}")
        axes[1].plot(np.arange(EXPECTED_FRAMES), margins[:, channel], color=colours[channel], lw=0.9)
    axes[0].axhline(MATERIAL_ERROR_LIMIT_PX, color="black", ls="--", lw=1.0, label="strict sqrt(2)-pixel limit")
    axes[0].set_ylabel("material error (input pixels)")
    axes[1].set_ylabel("top1 - separated top2 ZNCC")
    axes[1].set_xlabel("source frame (target is next +2-degree frame)")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=6, fontsize=8)
    fig.suptitle(f"Adjacent raw-RGB observability — {patch_size}px / {scope}; descriptive correlated edges")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_worst(
    output: Path,
    images: Sequence[np.ndarray],
    arrays: Mapping[str, np.ndarray],
    physical_target_px: np.ndarray,
    errors: np.ndarray,
    patch_size: int,
    scale_index: int,
    scope: str,
    config: RGBObservabilityConfig,
) -> dict[str, Any]:
    finite_max = np.asarray([
        np.nanmax(errors[:, channel]) if np.isfinite(errors[:, channel]).any() else -np.inf
        for channel in range(EXPECTED_CHANNELS)
    ])
    worst_channel = int(np.argmax(finite_max))
    selected_channels = sorted({2, worst_channel})
    selections = []
    for channel in selected_channels:
        valid_frames = np.where(np.isfinite(errors[:, channel]))[0]
        if valid_frames.size == 0:
            continue
        frame = int(valid_frames[np.argmax(errors[valid_frames, channel])])
        selections.append((channel, frame))
    _require(selections, "no valid match exists for worst-event visual")
    fig, axes = plt.subplots(len(selections), 3, figsize=(16, 5 * len(selections)), constrained_layout=True)
    if len(selections) == 1:
        axes = np.asarray([axes])
    records = []
    for row, (channel, frame) in enumerate(selections):
        target_frame = (frame + 1) % EXPECTED_FRAMES
        source_center = arrays["source_coordinate_px"][frame, channel]
        top1 = arrays[f"{scope}_top1_coordinate_px"][scale_index, frame, channel]
        top2 = arrays[f"{scope}_top2_coordinate_px"][scale_index, frame, channel]
        scores, evidence = rgb_correlation_map(images[frame], images[target_frame], source_center, patch_size, config=config)
        _require(scores is not None and evidence["source_patch_inside_and_informative"], "selected visual has invalid correlation")
        axes[row, 0].imshow(images[frame])
        axes[row, 0].scatter(source_center[0], source_center[1], c="cyan", marker="+", s=110, lw=2, label="source detector")
        axes[row, 0].set_title(f"Frame {frame}, KP{channel}: source")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].imshow(images[target_frame])
        for point, colour, marker, label in (
            (physical_target_px[frame, channel], "lime", "o", "same material point"),
            (top1, "red", "x", f"{scope} RGB top1"),
            (top2, "yellow", "+", "separated top2"),
        ):
            axes[row, 1].scatter(point[0], point[1], c=colour, marker=marker, s=100, lw=2, label=label)
        axes[row, 1].set_title(f"Frame {target_frame}: error {errors[frame, channel]:.2f}px")
        axes[row, 1].legend(fontsize=8)
        shown = scores.astype(np.float64)
        if scope == "local":
            shown = np.where(local_candidate_mask(scores.shape, patch_size, source_center), shown, np.nan)
        radius = (patch_size - 1) / 2.0
        axes[row, 2].imshow(shown, origin="upper", extent=(radius, 511 - radius, 511 - radius, radius), cmap="viridis")
        axes[row, 2].scatter(physical_target_px[frame, channel, 0], physical_target_px[frame, channel, 1], c="lime", marker="o", s=80)
        axes[row, 2].scatter(top1[0], top1[1], c="red", marker="x", s=80)
        axes[row, 2].scatter(top2[0], top2[1], c="yellow", marker="+", s=80)
        axes[row, 2].set_title("Raw ZNCC field (geometry opened only after raw hash)")
        for axis in axes[row]:
            axis.set_xlim(0, 511)
            axis.set_ylim(511, 0)
        records.append({"channel": channel, "source_frame": frame, "target_frame": target_frame, "material_error_px": float(errors[frame, channel])})
    fig.suptitle(f"Worst visual correspondence audit — {patch_size}px / {scope}")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"worst_channel": worst_channel, "selections": records}


def evaluate(raw_receipt_path: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    # Freeze OpenCV's internal threading so parallel role workers do not change
    # scheduling or oversubscribe the CPU during the exact score recomputation.
    cv2.setNumThreads(1)
    receipt, arrays, receipt_record = _load_raw(raw_receipt_path)
    images, masks, theta, image_paths = _load_images_and_geometry(receipt)
    source_normalized = np.asarray(arrays["source_coordinate_normalized"], dtype=np.float64)
    source_px = np.asarray(arrays["source_coordinate_px"], dtype=np.float64)
    physical_target_normalized = rotate_points(source_normalized, 2.0)
    physical_target_px = normalized_to_pixel(physical_target_normalized)
    target_frames = np.roll(np.arange(EXPECTED_FRAMES, dtype=np.int64), -1)
    physical_target_on_object = _points_on_masks(physical_target_px, masks, target_frames)
    source = _source_state(source_px, masks)
    config = RGBObservabilityConfig()
    _require(config.as_dict() == receipt.get("config"), "evaluation config differs from raw receipt")

    output_dir.mkdir(parents=True, exist_ok=False)
    ranks = _recompute_and_rank(images, arrays, physical_target_px, config=config)
    reports: dict[str, Any] = {}
    derived_arrays: dict[str, np.ndarray] = {
        "physical_target_normalized": physical_target_normalized,
        "physical_target_px": physical_target_px,
        "physical_target_on_object": physical_target_on_object,
        **{f"source__{name}": value for name, value in source.items()},
        **ranks,
    }
    visuals: dict[str, Any] = {}
    for scale_index, patch_size in enumerate(PATCH_SIZES):
        reports[str(patch_size)] = {}
        visuals[str(patch_size)] = {}
        for scope in SCOPES:
            metrics, derived = _scope_metrics(
                arrays,
                ranks,
                source,
                physical_target_px,
                physical_target_on_object,
                masks,
                target_frames,
                scale_index,
                scope,
            )
            reports[str(patch_size)][scope] = metrics
            for name, value in derived.items():
                derived_arrays[f"patch{patch_size}__{scope}__{name}"] = value
            time_path = output_dir / f"error_margin_time__patch{patch_size}__{scope}.png"
            worst_path = output_dir / f"worst_events__patch{patch_size}__{scope}.png"
            _plot_time(time_path, derived["material_error_px"], arrays[f"{scope}_margin"][scale_index], patch_size, scope)
            montage = _plot_worst(
                worst_path,
                images,
                arrays,
                physical_target_px,
                derived["material_error_px"],
                patch_size,
                scale_index,
                scope,
                config,
            )
            visuals[str(patch_size)][scope] = {
                "error_margin_time": _file_record(time_path),
                "worst_events": _file_record(worst_path),
                "selection": montage,
            }
        global_point = arrays["global_top1_coordinate_px"][scale_index]
        local_point = arrays["local_top1_coordinate_px"][scale_index]
        both = arrays["global_valid"][scale_index] & arrays["local_valid"][scale_index]
        disagreement = np.full((EXPECTED_FRAMES, EXPECTED_CHANNELS), np.nan, dtype=np.float64)
        disagreement[both] = np.linalg.norm(global_point[both] - local_point[both], axis=-1)
        reports[str(patch_size)]["global_local_disagreement_px"] = _distribution(disagreement)
        derived_arrays[f"patch{patch_size}__global_local_disagreement_px"] = disagreement

    arrays_path = output_dir / "rgb_observability_evaluation_arrays.npz"
    np.savez_compressed(arrays_path, theta_deg=theta, target_frame_index=target_frames, **derived_arrays)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "post_hash_physical_evaluation_of_adjacent_rgb_observability",
        "role": receipt["role"],
        "raw_receipt": receipt_record,
        "raw_arrays": receipt["raw_arrays"],
        "raw_prediction_hash_fixed_before_masks_or_theta_opened": True,
        "raw_predictions_recomputed_exact_before_physical_ranking": True,
        "implementation_head": head,
        "implementation_sources": {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES},
        "geometry": {
            "verified_theta_deg": "2 * frame_index",
            "pair": "t -> (t+1) mod 180",
            "physical_target": "R(+2 degrees) around normalized pivot (0,0) applied after raw hash",
        },
        "strict_thresholds": {
            "material_error_max_px": MATERIAL_ERROR_LIMIT_PX,
            "minimum_pair_distance_px": MINIMUM_PAIR_DISTANCE_PX,
            "minimum_source_activity_rms_px": MINIMUM_ACTIVITY_RMS_PX,
            "derivation": "unchanged full-resolution planted calibration; not fitted to matcher outcomes",
        },
        "reporting_contract": {
            "strict_pass": "all upstream source-state, match, material-error, on-object, and cross-channel checks pass for all 180 edges",
            "grounded_matcher_strict_pass": "at least one physically grounded edge exists and every grounded edge is valid, within the frozen material-error limit, and predicts on-object; upstream activity/distinctness is reported separately",
        },
        "source_state": {
            "active_count": int(np.sum(source["active"])),
            "on_object_all_frames_count": int(np.sum(source["on_object_all_frames"])),
            "distinct_all_frames_count": int(np.sum(source["distinct_all_frames"])),
            "per_channel_activity_rms_px": source["activity_rms_px"].tolist(),
        },
        "reports": reports,
        "visuals": visuals,
        "evaluation_arrays": _file_record(arrays_path),
        "statistical_scope": {
            "statistics": "deterministic descriptive counts, minimum, median, mean, q90, q99, maximum",
            "sample_unit": "one source channel on one adjacent cyclic edge",
            "n_edges_per_channel": EXPECTED_FRAMES,
            "correlation_caveat": "edges overlap in one orbit; roles and channels share images/training runs; no SEM, CI, or population inference",
            "descriptive_not_inferential": True,
        },
        "training_or_weight_update_performed": False,
    }
    report_path = output_dir / "rgb_material_observability_evaluation.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"report": _file_record(report_path), "evaluation_arrays": _file_record(arrays_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory exists; use a fresh attempt")
    print(json.dumps(evaluate(args.raw_receipt.resolve(strict=True), args.output_dir, args.repo_root.resolve(strict=True)), sort_keys=True))


if __name__ == "__main__":
    main()
