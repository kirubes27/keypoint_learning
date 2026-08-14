"""Pure helpers for frozen full-orbit encoder-feature decoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class FrozenFeatureDecodeError(ValueError):
    """Raised when feature-decode semantics or arrays are invalid."""


def endpoint_cells_to_coordinates(x_cell: Any, y_cell: Any, *, size: int = 64) -> np.ndarray:
    """Convert feature-cell indices to endpoint-aligned normalized ``(x,y)``."""
    if size < 2:
        raise FrozenFeatureDecodeError("size must be at least two")
    x = np.asarray(x_cell, dtype=np.float64)
    y = np.asarray(y_cell, dtype=np.float64)
    if x.shape != y.shape:
        raise FrozenFeatureDecodeError("x_cell and y_cell shapes differ")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FrozenFeatureDecodeError("cell coordinates contain non-finite values")
    if np.any(x < 0) or np.any(x >= size) or np.any(y < 0) or np.any(y >= size):
        raise FrozenFeatureDecodeError("cell coordinate lies outside the feature field")
    return np.stack((-1.0 + 2.0 * x / (size - 1), -1.0 + 2.0 * y / (size - 1)), axis=-1)


def stable_spatial_top_two(
    correlation: Any,
    *,
    exclusion_radius_cells: float = 4.0,
) -> dict[str, np.ndarray]:
    """Return stable top-one and spatially separated top-two cells.

    ``correlation`` has shape ``(..., H, W)``. NumPy's first flat argmax is
    the frozen tie rule. The second cell must lie strictly outside the stated
    Euclidean radius around the first.
    """
    values = np.asarray(correlation, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise FrozenFeatureDecodeError("correlation must end in a square spatial field")
    if not np.isfinite(values).all():
        raise FrozenFeatureDecodeError("correlation contains non-finite values")
    if exclusion_radius_cells < 0:
        raise FrozenFeatureDecodeError("exclusion_radius_cells must be non-negative")
    size = values.shape[-1]
    flat = values.reshape(-1, size * size)
    top1_flat = np.argmax(flat, axis=1)
    top1_y, top1_x = np.divmod(top1_flat, size)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    separated = np.empty_like(flat)
    for row in range(flat.shape[0]):
        distance = np.hypot(xx - top1_x[row], yy - top1_y[row])
        separated[row] = np.where(distance > exclusion_radius_cells, flat[row].reshape(size, size), -np.inf).reshape(-1)
    if np.any(~np.isfinite(np.max(separated, axis=1))):
        raise FrozenFeatureDecodeError("no spatially separated second peak exists")
    top2_flat = np.argmax(separated, axis=1)
    top2_y, top2_x = np.divmod(top2_flat, size)
    prefix = values.shape[:-2]
    top1 = flat[np.arange(flat.shape[0]), top1_flat].reshape(prefix)
    top2 = flat[np.arange(flat.shape[0]), top2_flat].reshape(prefix)
    return {
        "top1_similarity": top1,
        "top2_similarity": top2,
        "margin": top1 - top2,
        "top1_x_cell": top1_x.reshape(prefix),
        "top1_y_cell": top1_y.reshape(prefix),
        "top2_x_cell": top2_x.reshape(prefix),
        "top2_y_cell": top2_y.reshape(prefix),
    }


def sample_anchor_descriptors(
    feature_field: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Sample and normalize one descriptor per basis and keypoint.

    ``feature_field`` must be ``(1,C,H,W)`` and coordinates ``(A,K,2)``.
    The result has shape ``(A,K,C)``.
    """
    if feature_field.ndim != 4 or feature_field.shape[0] != 1:
        raise FrozenFeatureDecodeError("feature_field must have shape (1,C,H,W)")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise FrozenFeatureDecodeError("coordinates must have shape (anchors,keypoints,2)")
    if not bool(torch.isfinite(feature_field).all()) or not bool(torch.isfinite(coordinates).all()):
        raise FrozenFeatureDecodeError("feature inputs contain non-finite values")
    if bool(torch.any(coordinates < -1.0)) or bool(torch.any(coordinates > 1.0)):
        raise FrozenFeatureDecodeError("anchor coordinates leave [-1,1]")
    anchors, keypoints, _ = coordinates.shape
    grid = coordinates.reshape(1, anchors * keypoints, 1, 2)
    sampled = F.grid_sample(
        feature_field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled.reshape(anchors, keypoints, -1), p=2.0, dim=-1, eps=1e-6)


def cosine_correlation_maps(
    anchor_descriptors: torch.Tensor,
    target_feature_field: torch.Tensor,
) -> torch.Tensor:
    """Return ``(A,B,K,H,W)`` cosine fields for fixed anchors."""
    if anchor_descriptors.ndim != 3:
        raise FrozenFeatureDecodeError("anchor_descriptors must have shape (A,K,C)")
    if target_feature_field.ndim != 4:
        raise FrozenFeatureDecodeError("target_feature_field must have shape (B,C,H,W)")
    if anchor_descriptors.shape[-1] != target_feature_field.shape[1]:
        raise FrozenFeatureDecodeError("descriptor channel counts differ")
    target = F.normalize(target_feature_field, p=2.0, dim=1, eps=1e-6)
    return torch.einsum("akc,bchw->abkhw", anchor_descriptors, target)


def sample_target_similarities(
    anchor_descriptors: torch.Tensor,
    target_feature_field: torch.Tensor,
    detector_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Return anchor cosine at each basis-matched detector coordinate.

    Coordinates have shape ``(A,B,K,2)`` and output ``(A,B,K)``.
    """
    if detector_coordinates.ndim != 4 or detector_coordinates.shape[-1] != 2:
        raise FrozenFeatureDecodeError("detector_coordinates must have shape (A,B,K,2)")
    anchors, batch, keypoints, _ = detector_coordinates.shape
    if anchor_descriptors.shape[:2] != (anchors, keypoints):
        raise FrozenFeatureDecodeError("anchor and detector basis/keypoint counts differ")
    if target_feature_field.shape[0] != batch:
        raise FrozenFeatureDecodeError("target batch count differs")
    normalized = F.normalize(target_feature_field, p=2.0, dim=1, eps=1e-6)
    fields = normalized[:, None].expand(batch, anchors, -1, -1, -1).reshape(
        batch * anchors, normalized.shape[1], normalized.shape[2], normalized.shape[3]
    )
    grid = detector_coordinates.permute(1, 0, 2, 3).reshape(batch * anchors, keypoints, 1, 2)
    sampled = F.grid_sample(fields, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = F.normalize(sampled.squeeze(-1).permute(0, 2, 1), p=2.0, dim=-1, eps=1e-6)
    sampled = sampled.reshape(batch, anchors, keypoints, -1).permute(1, 0, 2, 3)
    return torch.einsum("akc,abkc->abk", anchor_descriptors, sampled)


__all__ = [
    "FrozenFeatureDecodeError",
    "cosine_correlation_maps",
    "endpoint_cells_to_coordinates",
    "sample_anchor_descriptors",
    "sample_target_similarities",
    "stable_spatial_top_two",
]
