"""Pure helpers for frozen spike-aligned encoder-feature forensics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class FeatureForensicError(ValueError):
    """Raised when feature-match semantics would be ambiguous."""


def coordinate_on_mask(image_mask: Any, coordinate: Any) -> bool:
    """Return nearest-pixel mask membership for endpoint-aligned ``(x,y)``."""
    mask = np.asarray(image_mask, dtype=bool)
    point = _finite(coordinate, name="coordinate", ndim=1).astype(np.float64, copy=False)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise FeatureForensicError("image_mask must be a square 2D array")
    if point.shape != (2,):
        raise FeatureForensicError("coordinate must contain x and y")
    if np.any(point < -1.0) or np.any(point > 1.0):
        return False
    pixel = (point + 1.0) * 0.5 * (mask.shape[0] - 1)
    x = int(np.clip(np.rint(pixel[0]), 0, mask.shape[1] - 1))
    y = int(np.clip(np.rint(pixel[1]), 0, mask.shape[0] - 1))
    return bool(mask[y, x])


def _finite(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise FeatureForensicError(f"{name} must have ndim={ndim}")
    if not np.isfinite(array).all():
        raise FeatureForensicError(f"{name} contains non-finite values")
    return array


def sample_normalized_feature_vectors(
    feature_field: Any,
    coordinates: Any,
) -> np.ndarray:
    """Bilinearly sample endpoint-aligned `(x,y)` coordinates and L2-normalize."""
    features = _finite(feature_field, name="feature_field", ndim=3).astype(np.float32, copy=False)
    coords = _finite(coordinates, name="coordinates").astype(np.float32, copy=False)
    if coords.ndim == 1:
        coords = coords[None]
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise FeatureForensicError("coordinates must have shape (samples,2)")
    if np.any(coords < -1.0) or np.any(coords > 1.0):
        raise FeatureForensicError("coordinates must remain inside [-1,1]")
    tensor = torch.from_numpy(features[None])
    grid = torch.from_numpy(coords.reshape(1, coords.shape[0], 1, 2))
    with torch.inference_mode():
        sampled = F.grid_sample(
            tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(-1).transpose(0, 1)
        normalized = F.normalize(sampled, p=2.0, dim=-1, eps=1e-6)
    return normalized.numpy().astype(np.float64, copy=False)


def normalized_correlation_map(anchor_descriptor: Any, target_feature_field: Any) -> np.ndarray:
    """Return cosine similarity from one normalized descriptor to every target cell."""
    anchor = _finite(anchor_descriptor, name="anchor_descriptor", ndim=1).astype(np.float64, copy=False)
    target = _finite(target_feature_field, name="target_feature_field", ndim=3).astype(np.float64, copy=False)
    if target.shape[0] != anchor.shape[0]:
        raise FeatureForensicError("anchor/target channel counts differ")
    anchor_norm = np.linalg.norm(anchor)
    if anchor_norm <= 1e-12:
        raise FeatureForensicError("anchor descriptor has zero norm")
    normalized_anchor = anchor / anchor_norm
    target_norm = np.linalg.norm(target, axis=0, keepdims=True)
    normalized_target = target / np.maximum(target_norm, 1e-6)
    return np.einsum("c,chw->hw", normalized_anchor, normalized_target)


def endpoint_feature_mask(image_mask: Any, feature_size: int) -> np.ndarray:
    """Sample an image mask at the exact endpoint-aligned feature-cell pixels."""
    mask = np.asarray(image_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise FeatureForensicError("image_mask must be a square 2D array")
    if feature_size < 2:
        raise FeatureForensicError("feature_size must be at least two")
    pixel = np.rint(np.linspace(0, mask.shape[0] - 1, feature_size)).astype(np.int64)
    sampled = mask[np.ix_(pixel, pixel)]
    if not np.any(sampled):
        raise FeatureForensicError("endpoint feature mask contains no object cell")
    return sampled


def feature_match_metrics(
    source_feature_field: Any,
    target_feature_field: Any,
    target_image_mask: Any,
    *,
    source_coordinate: Any,
    physical_target_coordinate: Any,
    detector_target_coordinate: Any,
    separated_second_peak_coordinate: Any,
) -> tuple[dict[str, float | int | list[float]], np.ndarray]:
    """Compare the physical material target with detector and feature choices."""
    source = _finite(source_feature_field, name="source_feature_field", ndim=3)
    target = _finite(target_feature_field, name="target_feature_field", ndim=3)
    if source.shape != target.shape or source.shape[1] != source.shape[2]:
        raise FeatureForensicError("source/target feature shapes differ or are not square")
    coordinates = np.stack(
        (
            _finite(source_coordinate, name="source_coordinate", ndim=1),
            _finite(physical_target_coordinate, name="physical_target_coordinate", ndim=1),
            _finite(detector_target_coordinate, name="detector_target_coordinate", ndim=1),
            _finite(separated_second_peak_coordinate, name="separated_second_peak_coordinate", ndim=1),
        )
    )
    if coordinates.shape != (4, 2):
        raise FeatureForensicError("each coordinate must contain x and y")
    source_descriptor = sample_normalized_feature_vectors(source, coordinates[0])[0]
    target_descriptors = sample_normalized_feature_vectors(target, coordinates[1:])
    similarities = target_descriptors @ source_descriptor
    correlation = normalized_correlation_map(source_descriptor, target)
    object_mask = endpoint_feature_mask(target_image_mask, target.shape[-1])
    object_similarities = correlation[object_mask]
    order = np.sort(object_similarities)
    best = float(order[-1])
    second_best = float(order[-2]) if order.size > 1 else best
    physical_similarity = float(similarities[0])
    best_flat = int(np.argmax(np.where(object_mask, correlation, -np.inf)))
    best_y, best_x = np.unravel_index(best_flat, correlation.shape)
    best_coordinate = np.asarray(
        (-1.0 + 2.0 * best_x / (target.shape[-1] - 1), -1.0 + 2.0 * best_y / (target.shape[-2] - 1)),
        dtype=np.float64,
    )
    scale = 0.5 * (target.shape[-1] - 1)
    physical = coordinates[1]
    detector = coordinates[2]
    return {
        "physical_target_similarity": physical_similarity,
        "detector_target_similarity": float(similarities[1]),
        "separated_second_peak_similarity": float(similarities[2]),
        "detector_minus_physical_similarity": float(similarities[1] - similarities[0]),
        "second_peak_minus_physical_similarity": float(similarities[2] - similarities[0]),
        "physical_target_mask_percentile": float(np.mean(object_similarities <= physical_similarity)),
        "mask_cells_higher_than_physical_fraction": float(np.mean(object_similarities > physical_similarity)),
        "object_best_similarity": best,
        "object_second_best_similarity": second_best,
        "object_top1_top2_margin": best - second_best,
        "object_best_minus_physical_similarity": best - physical_similarity,
        "object_argmax_coordinate": [float(value) for value in best_coordinate],
        "object_argmax_distance_to_physical_cells": float(np.linalg.norm(best_coordinate - physical) * scale),
        "detector_distance_to_physical_cells": float(np.linalg.norm(detector - physical) * scale),
        "object_feature_cells": int(object_similarities.size),
    }, correlation


def select_low_wobble_centres(
    canonical_channel: Any,
    spike_centres: Iterable[int],
    *,
    count: int = 5,
    exclusion_radius: int = 2,
) -> list[int]:
    """Select the lowest non-seam second differences with stable tie-breaking."""
    points = _finite(canonical_channel, name="canonical_channel", ndim=2).astype(np.float64, copy=False)
    if points.shape[1] != 2 or points.shape[0] < 3:
        raise FeatureForensicError("canonical_channel must have shape (frames>=3,2)")
    if count <= 0 or exclusion_radius < 0:
        raise FeatureForensicError("count must be positive and exclusion_radius non-negative")
    spikes = tuple(int(value) for value in spike_centres)
    centres = np.arange(1, points.shape[0] - 1, dtype=np.int64)
    second = np.linalg.norm(points[2:] - 2.0 * points[1:-1] + points[:-2], axis=-1)
    allowed = np.asarray(
        [all(abs(int(centre) - spike) > exclusion_radius for spike in spikes) for centre in centres],
        dtype=bool,
    )
    candidates = centres[allowed]
    values = second[allowed]
    if candidates.size < count:
        raise FeatureForensicError("too few low-wobble control centres remain")
    order = np.lexsort((candidates, values))[:count]
    return [int(value) for value in candidates[order]]


__all__ = [
    "FeatureForensicError",
    "coordinate_on_mask",
    "endpoint_feature_mask",
    "feature_match_metrics",
    "normalized_correlation_map",
    "sample_normalized_feature_vectors",
    "select_low_wobble_centres",
]
