"""Pure helpers for independently re-anchored adjacent feature decoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


NORMALIZATION_EPSILON = 1e-6


class AdjacentFeatureReanchorError(ValueError):
    """Raised when adjacent feature-decode semantics would be ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjacentFeatureReanchorError(message)


def cyclic_target_indices(frame_count: int) -> np.ndarray:
    """Return target index ``(source + 1) mod frame_count`` for every source."""
    _require(isinstance(frame_count, int) and frame_count > 1, "frame_count must be an integer above one")
    return np.remainder(np.arange(frame_count, dtype=np.int64) + 1, frame_count)


def endpoint_coordinates_to_cells(
    coordinates: Any,
    *,
    size: int = 64,
    tolerance_cells: float = 1e-5,
) -> np.ndarray:
    """Map endpoint-aligned ``(x,y)`` coordinates to nearest integer cells.

    The returned final axis is also ``(x,y)``. ``tolerance_cells`` is an
    implementation-consistency guard, not a scientific quality threshold.
    """
    points = np.asarray(coordinates, dtype=np.float64)
    _require(points.ndim >= 1 and points.shape[-1] == 2, "coordinates must end in (x,y)")
    _require(np.isfinite(points).all(), "coordinates contain non-finite values")
    _require(isinstance(size, int) and size > 1, "size must be an integer above one")
    _require(tolerance_cells >= 0.0, "tolerance_cells must be nonnegative")
    scaled = (points + 1.0) * 0.5 * float(size - 1)
    nearest = np.rint(scaled)
    _require(float(np.max(np.abs(scaled - nearest))) <= tolerance_cells, "coordinate is not on endpoint grid")
    _require(bool(np.all(nearest >= 0.0)) and bool(np.all(nearest <= size - 1)), "cell leaves grid")
    return nearest.astype(np.int64)


def sample_paired_descriptors(
    feature_fields: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one raw and normalized descriptor per batch/keypoint.

    ``feature_fields`` is ``(B,C,H,W)`` and ``coordinates`` is ``(B,K,2)``.
    Returns ``(raw, raw_norm, normalized)`` with shapes ``(B,K,C)``,
    ``(B,K)``, and ``(B,K,C)``.
    """
    _require(feature_fields.ndim == 4, "feature_fields must have shape (B,C,H,W)")
    _require(coordinates.ndim == 3 and coordinates.shape[-1] == 2, "coordinates must have shape (B,K,2)")
    _require(feature_fields.shape[0] == coordinates.shape[0], "feature/coordinate batch counts differ")
    _require(feature_fields.is_floating_point() and coordinates.is_floating_point(), "inputs must be floating point")
    _require(feature_fields.device == coordinates.device, "inputs must share a device")
    _require(bool(torch.isfinite(feature_fields).all()) and bool(torch.isfinite(coordinates).all()), "inputs are non-finite")
    _require(bool(torch.all(coordinates >= -1.0)) and bool(torch.all(coordinates <= 1.0)), "coordinates leave [-1,1]")
    sampled = F.grid_sample(
        feature_fields,
        coordinates.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(-1).transpose(1, 2)
    raw_norm = torch.linalg.vector_norm(sampled, ord=2, dim=-1)
    normalized = F.normalize(sampled, p=2.0, dim=-1, eps=NORMALIZATION_EPSILON)
    return sampled, raw_norm, normalized


def paired_cosine_correlation_maps(
    source_descriptors: torch.Tensor,
    target_feature_fields: torch.Tensor,
) -> torch.Tensor:
    """Return paired cosine maps with shape ``(B,K,H,W)``."""
    _require(source_descriptors.ndim == 3, "source_descriptors must have shape (B,K,C)")
    _require(target_feature_fields.ndim == 4, "target_feature_fields must have shape (B,C,H,W)")
    _require(source_descriptors.shape[0] == target_feature_fields.shape[0], "batch counts differ")
    _require(source_descriptors.shape[-1] == target_feature_fields.shape[1], "feature channel counts differ")
    _require(source_descriptors.device == target_feature_fields.device, "inputs must share a device")
    _require(bool(torch.isfinite(source_descriptors).all()), "source descriptors are non-finite")
    _require(bool(torch.isfinite(target_feature_fields).all()), "target fields are non-finite")
    normalized_target = F.normalize(target_feature_fields, p=2.0, dim=1, eps=NORMALIZATION_EPSILON)
    return torch.einsum("bkc,bchw->bkhw", source_descriptors, normalized_target)


def sample_paired_target_similarities(
    source_descriptors: torch.Tensor,
    target_feature_fields: torch.Tensor,
    target_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Return paired cosine similarity at supplied target coordinates."""
    _, _, target_descriptors = sample_paired_descriptors(target_feature_fields, target_coordinates)
    _require(source_descriptors.shape == target_descriptors.shape, "descriptor shapes differ")
    return torch.einsum("bkc,bkc->bk", source_descriptors, target_descriptors)


__all__ = [
    "AdjacentFeatureReanchorError",
    "NORMALIZATION_EPSILON",
    "cyclic_target_indices",
    "endpoint_coordinates_to_cells",
    "paired_cosine_correlation_maps",
    "sample_paired_descriptors",
    "sample_paired_target_similarities",
]
