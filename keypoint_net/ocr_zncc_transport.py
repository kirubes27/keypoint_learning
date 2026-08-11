"""Operator-Centred Reciprocal ZNCC transport for retained 64x64 RGB grids.

The shared operator supplies only a detached local search centre.  A fixed RGB
patch at the detached source keypoint is matched against fixed spatial target
locations with zero-normalized cross-correlation (ZNCC).  The hard match,
confidence, patch evidence, reciprocal check, source coordinate, and search
centre are all detached.  Consequently the auxiliary loss pulls only the
target-frame keypoint coordinate toward an accepted correspondence.

Other keypoint channels are never positives or negatives.  This module uses no
masks, transform metadata, renderer material identifiers, reconstruction,
optical flow, tracker, or learned feature field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OCRZNCCConfig:
    """Frozen Stage-0 engineering configuration.

    The numerical acceptance values are preregistered engineering thresholds,
    not established scientific constants.  They must not be tuned after
    looking at the Stage-0 direction result.
    """

    grid_size: int = 64
    patch_size: int = 7
    search_radius_cells: int = 8
    minimum_patch_rms: float = 0.02
    minimum_peak_zncc: float = 0.80
    minimum_peak_margin: float = 0.03
    peak_margin_exclusion_radius_cells: int = 0
    reciprocal_tolerance_cells: float = 1.0
    zncc_epsilon: float = 1e-6
    reject_search_boundary: bool = True

    def validate(self) -> None:
        if self.grid_size != 64:
            raise ValueError("OCR-ZNCC Stage 0 requires the retained 64x64 grid")
        if self.patch_size <= 0 or self.patch_size % 2 != 1:
            raise ValueError("patch_size must be a positive odd integer")
        if self.search_radius_cells <= 0:
            raise ValueError("search_radius_cells must be positive")
        if not 0.0 < self.minimum_patch_rms < 1.0:
            raise ValueError("minimum_patch_rms must be in (0,1)")
        if not -1.0 < self.minimum_peak_zncc < 1.0:
            raise ValueError("minimum_peak_zncc must be in (-1,1)")
        if not 0.0 < self.minimum_peak_margin < 2.0:
            raise ValueError("minimum_peak_margin must be in (0,2)")
        if (
            isinstance(self.peak_margin_exclusion_radius_cells, bool)
            or not isinstance(self.peak_margin_exclusion_radius_cells, int)
            or not 0 <= self.peak_margin_exclusion_radius_cells < self.search_radius_cells
        ):
            raise ValueError(
                "peak_margin_exclusion_radius_cells must be an integer in "
                "[0, search_radius_cells)"
            )
        if self.reciprocal_tolerance_cells < 0.0:
            raise ValueError("reciprocal_tolerance_cells must be non-negative")
        if self.zncc_epsilon <= 0.0:
            raise ValueError("zncc_epsilon must be positive")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class OCRZNCCResult:
    """Loss plus detached evidence needed by Stage-0 auditing."""

    loss: torch.Tensor
    match_coordinates: torch.Tensor
    accepted: torch.Tensor
    confidence: torch.Tensor
    source_patch_rms: torch.Tensor
    matched_target_patch_rms: torch.Tensor
    target_peak_zncc: torch.Tensor
    target_peak_margin: torch.Tensor
    reciprocal_peak_zncc: torch.Tensor
    reciprocal_peak_margin: torch.Tensor
    reciprocal_error_cells: torch.Tensor
    target_competitor_distance_cells: torch.Tensor
    reciprocal_competitor_distance_cells: torch.Tensor
    target_search_centres: torch.Tensor
    target_match_offsets_cells: torch.Tensor
    source_patch_inside: torch.Tensor

    @property
    def accepted_count(self) -> int:
        return int(self.accepted.sum().item())


def grid_step(grid_size: int = 64) -> float:
    if grid_size <= 1:
        raise ValueError("grid_size must exceed one")
    return 2.0 / float(grid_size - 1)


def normalized_images_to_retained_rgb(
    images: torch.Tensor,
    *,
    grid_size: int = 64,
) -> torch.Tensor:
    """Invert ImageNet normalization and resample to the retained RGB grid."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (images.detach() * std + mean).clamp(0.0, 1.0)
    return F.interpolate(
        rgb,
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=True,
    )


def _validate_rgb_and_centres(images: torch.Tensor, centres: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("retained RGB images must have shape (B,3,H,W)")
    if images.shape[-1] != images.shape[-2]:
        raise ValueError("retained RGB grid must be square")
    if centres.ndim < 3 or centres.shape[0] != images.shape[0] or centres.shape[-1] != 2:
        raise ValueError("centres must have shape (B,...,2) with at least one point axis")
    if images.device != centres.device or images.dtype != centres.dtype:
        raise ValueError("images and centres must share device and dtype")
    if not images.is_floating_point() or not centres.is_floating_point():
        raise TypeError("images and centres must be floating point")


def patch_centres_inside(
    centres: torch.Tensor,
    *,
    patch_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Return whether every sample of a patch remains in the closed image."""

    radius = patch_size // 2
    maximum = 1.0 - radius * grid_step(grid_size)
    return (centres.abs() <= maximum + 1e-7).all(dim=-1)


def sample_rgb_patches(
    images: torch.Tensor,
    centres: torch.Tensor,
    *,
    patch_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Bilinearly sample patches at normalized xy centres.

    The result shape is ``(B,...,3,patch_size,patch_size)`` and preserves all
    point axes from ``centres``.
    """

    _validate_rgb_and_centres(images, centres)
    if tuple(images.shape[-2:]) != (grid_size, grid_size):
        raise ValueError("image shape differs from declared grid_size")
    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    point_shape = centres.shape[1:-1]
    point_count = 1
    for size in point_shape:
        point_count *= int(size)
    radius = patch_size // 2
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=centres.device,
        dtype=centres.dtype,
    ) * grid_step(grid_size)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    patch_offsets = torch.stack((xx, yy), dim=-1)
    grid = centres.reshape(images.shape[0], point_count, 1, 1, 2)
    grid = grid + patch_offsets.view(1, 1, patch_size, patch_size, 2)
    sampled = F.grid_sample(
        images,
        grid.reshape(images.shape[0], point_count * patch_size, patch_size, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.reshape(
        images.shape[0], 3, *point_shape, patch_size, patch_size
    )
    channel_axis = 1 + len(point_shape)
    permutation = [0, *range(2, channel_axis + 1), 1, channel_axis + 1, channel_axis + 2]
    return sampled.permute(permutation).contiguous()


def _center_and_measure(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if patches.ndim < 5 or patches.shape[-3] != 3:
        raise ValueError("patch tensor must end in (3,P,P)")
    centred = patches - patches.mean(dim=(-2, -1), keepdim=True)
    rms = torch.sqrt(torch.mean(centred.square(), dim=(-3, -2, -1)))
    return centred.flatten(start_dim=-3), rms


def _zncc(query: torch.Tensor, candidates: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Whole-RGB ZNCC after per-channel spatial centring."""

    if candidates.shape[:-2] != query.shape[:-1] or candidates.shape[-1] != query.shape[-1]:
        raise ValueError("query/candidate ZNCC shapes differ")
    numerator = (candidates * query.unsqueeze(-2)).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(candidates, dim=-1)
        * torch.linalg.vector_norm(query, dim=-1).unsqueeze(-1)
    )
    return numerator / denominator.clamp_min(epsilon)


def _search_offsets(config: OCRZNCCConfig, like: torch.Tensor) -> torch.Tensor:
    values = torch.arange(
        -config.search_radius_cells,
        config.search_radius_cells + 1,
        device=like.device,
        dtype=like.dtype,
    )
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _best_match(
    *,
    query_patch: torch.Tensor,
    query_rms: torch.Tensor,
    search_images: torch.Tensor,
    search_centres: torch.Tensor,
    config: OCRZNCCConfig,
) -> dict[str, torch.Tensor]:
    offsets_cells = _search_offsets(config, search_centres)
    candidate_centres = (
        search_centres.unsqueeze(-2)
        + offsets_cells.view(*([1] * (search_centres.ndim - 1)), -1, 2)
        * grid_step(config.grid_size)
    )
    candidate_patches = sample_rgb_patches(
        search_images,
        candidate_centres,
        patch_size=config.patch_size,
        grid_size=config.grid_size,
    )
    candidate_vectors, candidate_rms = _center_and_measure(candidate_patches)
    scores = _zncc(query_patch, candidate_vectors, config.zncc_epsilon)
    valid_geometry = patch_centres_inside(
        candidate_centres,
        patch_size=config.patch_size,
        grid_size=config.grid_size,
    )
    valid_evidence = candidate_rms >= config.minimum_patch_rms
    valid = valid_geometry & valid_evidence
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    best = torch.max(masked_scores, dim=-1)
    raw_peak = best.values
    index = best.indices
    best_offset = offsets_cells[index]
    competitor_separation = (
        offsets_cells.view(*([1] * (best_offset.ndim - 1)), -1, 2)
        - best_offset.unsqueeze(-2)
    ).abs().amax(dim=-1)
    competitor_valid = valid & (
        competitor_separation
        > float(config.peak_margin_exclusion_radius_cells)
    )
    competitor = torch.max(
        scores.masked_fill(~competitor_valid, -torch.inf),
        dim=-1,
    )
    raw_margin = raw_peak - competitor.values
    competitor_offset = offsets_cells[competitor.indices]
    competitor_distance = torch.linalg.vector_norm(
        competitor_offset - best_offset,
        dim=-1,
    )
    gather_index = index.unsqueeze(-1).expand(*index.shape, 2)
    match = torch.gather(candidate_centres, -2, gather_index.unsqueeze(-2)).squeeze(-2)
    match_offsets = offsets_cells[index]
    patch_gather = index.view(*index.shape, 1, 1, 1).expand(
        *index.shape, 1, 3, config.patch_size * config.patch_size
    )
    flattened_patches = candidate_patches.flatten(start_dim=-2)
    matched_patch = torch.gather(flattened_patches, -3, patch_gather).squeeze(-3)
    matched_patch = matched_patch.reshape(*index.shape, 3, config.patch_size, config.patch_size)
    rms_gather = index.unsqueeze(-1)
    matched_rms = torch.gather(candidate_rms, -1, rms_gather).squeeze(-1)
    boundary = (
        match_offsets.abs().amax(dim=-1) == float(config.search_radius_cells)
    )
    enough_valid = valid.sum(dim=-1) >= 2
    enough_competitors = competitor_valid.any(dim=-1)
    accepted = (
        enough_valid
        & enough_competitors
        & torch.isfinite(raw_peak)
        & torch.isfinite(raw_margin)
        & (query_rms >= config.minimum_patch_rms)
        & (raw_peak >= config.minimum_peak_zncc)
        & (raw_margin >= config.minimum_peak_margin)
    )
    if config.reject_search_boundary:
        accepted = accepted & ~boundary
    peak = torch.where(torch.isfinite(raw_peak), raw_peak, torch.full_like(raw_peak, -1.0))
    margin = torch.where(torch.isfinite(raw_margin), raw_margin, torch.zeros_like(raw_margin))
    return {
        "match": match,
        "match_offsets_cells": match_offsets,
        "matched_patch": matched_patch,
        "matched_rms": matched_rms,
        "peak": peak,
        "margin": margin,
        "competitor_distance_cells": competitor_distance,
        "accepted": accepted,
    }


def ocr_zncc_transport_loss(
    source_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    source_coordinates: torch.Tensor,
    operator_search_centres: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    config: OCRZNCCConfig = OCRZNCCConfig(),
) -> OCRZNCCResult:
    """Construct detached correspondences and pull only target coordinates."""

    config.validate()
    _validate_rgb_and_centres(source_rgb, source_coordinates)
    _validate_rgb_and_centres(target_rgb, target_coordinates)
    if source_rgb.shape != target_rgb.shape:
        raise ValueError("source and target RGB shapes differ")
    if source_coordinates.shape != target_coordinates.shape:
        raise ValueError("source and target coordinate shapes differ")
    if operator_search_centres.shape != target_coordinates.shape:
        raise ValueError("operator search-centre shape differs")
    if tuple(source_rgb.shape[-2:]) != (config.grid_size, config.grid_size):
        raise ValueError("OCR-ZNCC requires the retained grid size")

    source_values = source_rgb.detach()
    target_values = target_rgb.detach()
    source = source_coordinates.detach()
    search_centres = operator_search_centres.detach()

    source_patches = sample_rgb_patches(
        source_values,
        source,
        patch_size=config.patch_size,
        grid_size=config.grid_size,
    )
    source_vectors, source_rms = _center_and_measure(source_patches)
    source_inside = patch_centres_inside(
        source,
        patch_size=config.patch_size,
        grid_size=config.grid_size,
    ).detach()
    forward = _best_match(
        query_patch=source_vectors,
        query_rms=source_rms,
        search_images=target_values,
        search_centres=search_centres,
        config=config,
    )

    matched_target_vectors, matched_target_rms = _center_and_measure(
        forward["matched_patch"]
    )
    reciprocal = _best_match(
        query_patch=matched_target_vectors,
        query_rms=matched_target_rms,
        search_images=source_values,
        search_centres=source,
        config=config,
    )
    reciprocal_error_cells = torch.linalg.vector_norm(
        reciprocal["match"] - source,
        dim=-1,
    ) / grid_step(config.grid_size)
    accepted = (
        source_inside
        &
        forward["accepted"]
        & reciprocal["accepted"]
        & torch.isfinite(reciprocal_error_cells)
        & (reciprocal_error_cells <= config.reciprocal_tolerance_cells)
    ).detach()

    forward_confidence = ((forward["peak"] + 1.0) * 0.5).clamp(0.0, 1.0)
    reciprocal_confidence = ((reciprocal["peak"] + 1.0) * 0.5).clamp(0.0, 1.0)
    confidence = torch.minimum(forward_confidence, reciprocal_confidence).detach()
    matches = forward["match"].detach()
    radius_normalized = config.search_radius_cells * grid_step(config.grid_size)
    residual = (target_coordinates - matches) / radius_normalized
    element_loss = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=1.0 / float(config.search_radius_cells),
        reduction="none",
    ).mean(dim=-1)
    accepted_float = accepted.to(dtype=element_loss.dtype)
    accepted_count = accepted_float.sum()
    weighted_sum = (element_loss * confidence * accepted_float).sum()
    loss = torch.where(
        accepted_count > 0,
        weighted_sum / accepted_count.clamp_min(1.0),
        target_coordinates.sum() * 0.0,
    )

    return OCRZNCCResult(
        loss=loss,
        match_coordinates=matches,
        accepted=accepted,
        confidence=confidence,
        source_patch_rms=source_rms.detach(),
        matched_target_patch_rms=forward["matched_rms"].detach(),
        target_peak_zncc=forward["peak"].detach(),
        target_peak_margin=forward["margin"].detach(),
        reciprocal_peak_zncc=reciprocal["peak"].detach(),
        reciprocal_peak_margin=reciprocal["margin"].detach(),
        reciprocal_error_cells=reciprocal_error_cells.detach(),
        target_competitor_distance_cells=forward[
            "competitor_distance_cells"
        ].detach(),
        reciprocal_competitor_distance_cells=reciprocal[
            "competitor_distance_cells"
        ].detach(),
        target_search_centres=search_centres,
        target_match_offsets_cells=forward["match_offsets_cells"].detach(),
        source_patch_inside=source_inside,
    )


def combine_with_base_loss(
    base_loss: torch.Tensor,
    transport_loss: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Combine losses while preserving exact zero-weight control semantics."""

    if not isinstance(weight, (int, float)) or not torch.isfinite(
        base_loss.new_tensor(float(weight))
    ) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")
    if weight == 0.0:
        return base_loss
    return base_loss + float(weight) * transport_loss


__all__ = [
    "OCRZNCCConfig",
    "OCRZNCCResult",
    "combine_with_base_loss",
    "grid_step",
    "normalized_images_to_retained_rgb",
    "ocr_zncc_transport_loss",
    "patch_centres_inside",
    "sample_rgb_patches",
]
