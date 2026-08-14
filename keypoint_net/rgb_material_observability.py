"""Pure RGB correspondence primitives for the adjacent material-observability gate.

This module intentionally has no model, mask, transform, optimizer, or temporal
state dependency.  A source-centred RGB patch is compared independently with
all valid target patches using OpenCV's multi-channel normalized correlation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


IMAGE_SIZE = 512
PATCH_SIZES = (35, 105)
LOCAL_RADIUS_PX = 32.0
GLOBAL_COMPETITOR_RADIUS_PX = 32.0
LOCAL_COMPETITOR_RADIUS_PX = 8.0
INVALID_SCORE = -2.0
INVALID_COORDINATE = -1.0


class RGBObservabilityError(ValueError):
    """Raised when a correspondence input violates the frozen gate."""


@dataclass(frozen=True)
class RGBObservabilityConfig:
    image_size: int = IMAGE_SIZE
    patch_sizes: tuple[int, int] = PATCH_SIZES
    local_radius_px: float = LOCAL_RADIUS_PX
    global_competitor_radius_px: float = GLOBAL_COMPETITOR_RADIUS_PX
    local_competitor_radius_px: float = LOCAL_COMPETITOR_RADIUS_PX
    method: int = cv2.TM_CCOEFF_NORMED
    minimum_query_rms: float = 1e-8

    def validate(self) -> None:
        if self.image_size != IMAGE_SIZE:
            raise RGBObservabilityError("the frozen gate requires 512 x 512 RGB")
        if tuple(self.patch_sizes) != PATCH_SIZES:
            raise RGBObservabilityError("the frozen gate requires exactly 35 and 105 pixel patches")
        if any(size <= 1 or size % 2 != 1 for size in self.patch_sizes):
            raise RGBObservabilityError("patch sizes must be odd integers greater than one")
        if self.local_radius_px != LOCAL_RADIUS_PX:
            raise RGBObservabilityError("local radius differs from the frozen 32-pixel value")
        if self.global_competitor_radius_px != GLOBAL_COMPETITOR_RADIUS_PX:
            raise RGBObservabilityError("global competitor radius differs")
        if self.local_competitor_radius_px != LOCAL_COMPETITOR_RADIUS_PX:
            raise RGBObservabilityError("local competitor radius differs")
        if self.method != cv2.TM_CCOEFF_NORMED:
            raise RGBObservabilityError("only multi-channel TM_CCOEFF_NORMED is allowed")
        if not np.isfinite(self.minimum_query_rms) or self.minimum_query_rms <= 0.0:
            raise RGBObservabilityError("minimum query RMS must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["patch_sizes"] = list(self.patch_sizes)
        value["method_name"] = "cv2.TM_CCOEFF_NORMED"
        return value


def _require_rgb(image: Any, *, name: str, image_size: int) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (image_size, image_size, 3):
        raise RGBObservabilityError(
            f"{name} must have shape {(image_size, image_size, 3)}, got {value.shape}"
        )
    if value.dtype == np.uint8:
        value = value.astype(np.float32) / np.float32(255.0)
    elif value.dtype == np.float32:
        value = np.ascontiguousarray(value)
    else:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise RGBObservabilityError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def normalized_to_pixel(coordinate: Any, *, image_size: int = IMAGE_SIZE) -> np.ndarray:
    value = np.asarray(coordinate, dtype=np.float64)
    if value.shape[-1:] != (2,) or not np.isfinite(value).all():
        raise RGBObservabilityError("normalized coordinates must be finite (...,2) values")
    return (value + 1.0) * (float(image_size - 1) / 2.0)


def pixel_to_normalized(coordinate: Any, *, image_size: int = IMAGE_SIZE) -> np.ndarray:
    value = np.asarray(coordinate, dtype=np.float64)
    if value.shape[-1:] != (2,) or not np.isfinite(value).all():
        raise RGBObservabilityError("pixel coordinates must be finite (...,2) values")
    return value * (2.0 / float(image_size - 1)) - 1.0


def patch_inside(center_xy: Any, patch_size: int, *, image_size: int = IMAGE_SIZE) -> bool:
    centre = np.asarray(center_xy, dtype=np.float64)
    if centre.shape != (2,) or not np.isfinite(centre).all():
        return False
    radius = (int(patch_size) - 1) / 2.0
    return bool(
        radius <= centre[0] <= image_size - 1 - radius
        and radius <= centre[1] <= image_size - 1 - radius
    )


def query_patch(
    source_rgb: Any,
    source_center_xy: Any,
    patch_size: int,
    *,
    config: RGBObservabilityConfig = RGBObservabilityConfig(),
) -> tuple[np.ndarray | None, float]:
    """Extract one subpixel-centred RGB template and its centred RMS."""

    config.validate()
    image = _require_rgb(source_rgb, name="source_rgb", image_size=config.image_size)
    centre = np.asarray(source_center_xy, dtype=np.float64)
    if patch_size not in config.patch_sizes or not patch_inside(
        centre, patch_size, image_size=config.image_size
    ):
        return None, 0.0
    patch = cv2.getRectSubPix(
        image,
        (int(patch_size), int(patch_size)),
        (float(centre[0]), float(centre[1])),
    )
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    centred = patch - np.mean(patch, axis=(0, 1), keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(centred, dtype=np.float64))))
    if not np.isfinite(rms) or rms < config.minimum_query_rms:
        return None, rms if np.isfinite(rms) else 0.0
    return patch, rms


def rgb_correlation_map(
    source_rgb: Any,
    target_rgb: Any,
    source_center_xy: Any,
    patch_size: int,
    *,
    config: RGBObservabilityConfig = RGBObservabilityConfig(),
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return the complete valid-centre target ZNCC field for one source patch."""

    config.validate()
    source = _require_rgb(source_rgb, name="source_rgb", image_size=config.image_size)
    target = _require_rgb(target_rgb, name="target_rgb", image_size=config.image_size)
    patch, rms = query_patch(source, source_center_xy, patch_size, config=config)
    if patch is None:
        return None, {"source_patch_inside_and_informative": False, "source_patch_rms": rms}
    scores = cv2.matchTemplate(target, patch, config.method)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    if scores.shape != (
        config.image_size - patch_size + 1,
        config.image_size - patch_size + 1,
    ):
        raise RGBObservabilityError("OpenCV correlation-map shape differs")
    if not np.isfinite(scores).all():
        raise RGBObservabilityError("OpenCV correlation map contains non-finite values")
    return scores, {"source_patch_inside_and_informative": True, "source_patch_rms": rms}


def candidate_coordinate_grids(score_shape: tuple[int, int], patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(score_shape[0]), int(score_shape[1]))
    expected = IMAGE_SIZE - int(patch_size) + 1
    if (height, width) != (expected, expected):
        raise RGBObservabilityError("score shape does not match patch size and image size")
    radius = (int(patch_size) - 1) / 2.0
    x = np.arange(width, dtype=np.float64) + radius
    y = np.arange(height, dtype=np.float64) + radius
    return np.meshgrid(x, y)


def local_candidate_mask(
    score_shape: tuple[int, int],
    patch_size: int,
    source_center_xy: Any,
    *,
    radius_px: float = LOCAL_RADIUS_PX,
) -> np.ndarray:
    centre = np.asarray(source_center_xy, dtype=np.float64)
    if centre.shape != (2,) or not np.isfinite(centre).all():
        raise RGBObservabilityError("source centre must be a finite xy pair")
    xx, yy = candidate_coordinate_grids(score_shape, patch_size)
    return (np.abs(xx - centre[0]) <= radius_px) & (np.abs(yy - centre[1]) <= radius_px)


def stable_top_two(
    scores: Any,
    patch_size: int,
    *,
    allowed: Any | None = None,
    exclusion_radius_px: float,
) -> dict[str, Any]:
    """Select stable row-major top one and a spatially separated competitor."""

    value = np.asarray(scores, dtype=np.float32)
    xx, yy = candidate_coordinate_grids(value.shape, patch_size)
    valid = np.isfinite(value)
    if allowed is not None:
        allow = np.asarray(allowed, dtype=bool)
        if allow.shape != value.shape:
            raise RGBObservabilityError("allowed mask shape differs from scores")
        valid &= allow
    if not np.any(valid):
        return {
            "valid": False,
            "top1_coordinate_px": [INVALID_COORDINATE, INVALID_COORDINATE],
            "top1_score": INVALID_SCORE,
            "top2_coordinate_px": [INVALID_COORDINATE, INVALID_COORDINATE],
            "top2_score": INVALID_SCORE,
            "margin": 0.0,
            "candidate_count": 0,
        }
    masked = np.where(valid, value, -np.inf)
    first_flat = int(np.argmax(masked.reshape(-1)))
    first_y, first_x = np.unravel_index(first_flat, value.shape)
    first_coordinate = np.asarray((xx[first_y, first_x], yy[first_y, first_x]))
    separated = valid & (
        np.maximum(np.abs(xx - first_coordinate[0]), np.abs(yy - first_coordinate[1]))
        > float(exclusion_radius_px)
    )
    if np.any(separated):
        second_masked = np.where(separated, value, -np.inf)
        second_flat = int(np.argmax(second_masked.reshape(-1)))
        second_y, second_x = np.unravel_index(second_flat, value.shape)
        second_coordinate = [float(xx[second_y, second_x]), float(yy[second_y, second_x])]
        second_score = float(value[second_y, second_x])
        second_valid = True
    else:
        second_coordinate = [INVALID_COORDINATE, INVALID_COORDINATE]
        second_score = INVALID_SCORE
        second_valid = False
    first_score = float(value[first_y, first_x])
    return {
        "valid": bool(second_valid),
        "top1_coordinate_px": [float(first_coordinate[0]), float(first_coordinate[1])],
        "top1_score": first_score,
        "top2_coordinate_px": second_coordinate,
        "top2_score": second_score,
        "margin": float(first_score - second_score) if second_valid else 0.0,
        "candidate_count": int(np.sum(valid)),
    }


def decode_rgb_edge(
    source_rgb: Any,
    target_rgb: Any,
    source_center_xy: Any,
    patch_size: int,
    *,
    config: RGBObservabilityConfig = RGBObservabilityConfig(),
) -> dict[str, Any]:
    scores, evidence = rgb_correlation_map(
        source_rgb,
        target_rgb,
        source_center_xy,
        patch_size,
        config=config,
    )
    if scores is None:
        invalid = {
            "valid": False,
            "top1_coordinate_px": [INVALID_COORDINATE, INVALID_COORDINATE],
            "top1_score": INVALID_SCORE,
            "top2_coordinate_px": [INVALID_COORDINATE, INVALID_COORDINATE],
            "top2_score": INVALID_SCORE,
            "margin": 0.0,
            "candidate_count": 0,
        }
        return {"evidence": evidence, "global": dict(invalid), "local": dict(invalid)}
    return decode_score_map(
        scores,
        source_center_xy,
        patch_size,
        evidence=evidence,
        config=config,
    )


def decode_score_map(
    scores: Any,
    source_center_xy: Any,
    patch_size: int,
    *,
    evidence: Mapping[str, Any],
    config: RGBObservabilityConfig = RGBObservabilityConfig(),
) -> dict[str, Any]:
    """Decode global/local modes from one already-computed correlation map."""

    config.validate()
    value = np.asarray(scores, dtype=np.float32)
    local = local_candidate_mask(
        value.shape,
        patch_size,
        source_center_xy,
        radius_px=config.local_radius_px,
    )
    return {
        "evidence": dict(evidence),
        "global": stable_top_two(
            value,
            patch_size,
            exclusion_radius_px=config.global_competitor_radius_px,
        ),
        "local": stable_top_two(
            value,
            patch_size,
            allowed=local,
            exclusion_radius_px=config.local_competitor_radius_px,
        ),
    }


def candidate_rank(
    scores: Any,
    patch_size: int,
    target_coordinate_px: Any,
    *,
    allowed: Any | None = None,
) -> dict[str, Any]:
    """Rank the nearest valid candidate with stable row-major tie semantics."""

    value = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target_coordinate_px, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise RGBObservabilityError("target coordinate must be a finite xy pair")
    xx, yy = candidate_coordinate_grids(value.shape, patch_size)
    valid = np.isfinite(value)
    if allowed is not None:
        allow = np.asarray(allowed, dtype=bool)
        if allow.shape != value.shape:
            raise RGBObservabilityError("allowed mask shape differs from scores")
        valid &= allow
    distance2 = np.square(xx - target[0]) + np.square(yy - target[1])
    nearest_masked = np.where(valid, distance2, np.inf)
    nearest_flat = int(np.argmin(nearest_masked.reshape(-1)))
    if not np.isfinite(nearest_masked.reshape(-1)[nearest_flat]):
        return {
            "valid": False,
            "coordinate_px": [INVALID_COORDINATE, INVALID_COORDINATE],
            "distance_px": float("inf"),
            "score": INVALID_SCORE,
            "rank": 0,
            "candidate_count": 0,
            "rank_percentile": 0.0,
        }
    nearest_y, nearest_x = np.unravel_index(nearest_flat, value.shape)
    target_score = float(value[nearest_y, nearest_x])
    flat_scores = value.reshape(-1)
    flat_valid = valid.reshape(-1)
    earlier = np.arange(flat_scores.size) < nearest_flat
    outrank = flat_valid & (
        (flat_scores > target_score) | ((flat_scores == target_score) & earlier)
    )
    rank = int(np.sum(outrank)) + 1
    count = int(np.sum(flat_valid))
    percentile = 1.0 if count <= 1 else 1.0 - float(rank - 1) / float(count - 1)
    return {
        "valid": True,
        "coordinate_px": [float(xx[nearest_y, nearest_x]), float(yy[nearest_y, nearest_x])],
        "distance_px": float(np.sqrt(distance2[nearest_y, nearest_x])),
        "score": target_score,
        "rank": rank,
        "candidate_count": count,
        "rank_percentile": percentile,
    }


def flatten_decode(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return scalar/array fields used by the raw matrix writer."""

    evidence = result["evidence"]
    output: dict[str, Any] = {
        "source_valid": bool(evidence["source_patch_inside_and_informative"]),
        "source_patch_rms": float(evidence["source_patch_rms"]),
    }
    for scope in ("global", "local"):
        selected = result[scope]
        output[f"{scope}_valid"] = bool(selected["valid"])
        output[f"{scope}_top1_coordinate_px"] = np.asarray(
            selected["top1_coordinate_px"], dtype=np.float64
        )
        output[f"{scope}_top1_score"] = float(selected["top1_score"])
        output[f"{scope}_top2_coordinate_px"] = np.asarray(
            selected["top2_coordinate_px"], dtype=np.float64
        )
        output[f"{scope}_top2_score"] = float(selected["top2_score"])
        output[f"{scope}_margin"] = float(selected["margin"])
        output[f"{scope}_candidate_count"] = int(selected["candidate_count"])
    return output


__all__ = [
    "GLOBAL_COMPETITOR_RADIUS_PX",
    "IMAGE_SIZE",
    "INVALID_COORDINATE",
    "INVALID_SCORE",
    "LOCAL_COMPETITOR_RADIUS_PX",
    "LOCAL_RADIUS_PX",
    "PATCH_SIZES",
    "RGBObservabilityConfig",
    "RGBObservabilityError",
    "candidate_coordinate_grids",
    "candidate_rank",
    "decode_score_map",
    "decode_rgb_edge",
    "flatten_decode",
    "local_candidate_mask",
    "normalized_to_pixel",
    "patch_inside",
    "pixel_to_normalized",
    "query_patch",
    "rgb_correlation_map",
    "stable_top_two",
]
