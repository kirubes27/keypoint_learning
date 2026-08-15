"""Distribution-level same-frame equivariance candidate for detector wobble.

This module never consumes physical roll metadata, masks, material coordinates,
or an operator prediction.  It compares a heatmap distribution with the same
image passed through one of two locked tiny synthetic geometric transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .frozen_nuisance_sensitivity import (
        IMAGE_SIZE,
        LOCKED_SPECS,
        NuisanceSpec,
        apply_nuisance,
        nuisance_output_to_input_affine,
    )
except ImportError:  # Support direct historical script imports.
    from frozen_nuisance_sensitivity import (  # type: ignore
        IMAGE_SIZE,
        LOCKED_SPECS,
        NuisanceSpec,
        apply_nuisance,
        nuisance_output_to_input_affine,
    )


class SameFrameEquivarianceError(ValueError):
    """Raised when the candidate's locked semantics are violated."""


LOCKED_EQUIVARIANCE_SPECS = tuple(
    spec
    for spec in LOCKED_SPECS
    if spec.name
    in {
        "translation_plus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
    }
)


@dataclass(frozen=True)
class SameFrameEquivarianceResult:
    """Differentiable loss plus auditable per-transform components."""

    loss: torch.Tensor
    per_transform_loss: dict[str, torch.Tensor]
    common_batch_base_heatmaps: torch.Tensor
    augmented_heatmaps: torch.Tensor
    untransformed_image_count: int
    transformed_image_count: int
    auxiliary_forward_image_count: int


@dataclass(frozen=True)
class PairedEquivarianceLossResult:
    """One paired training path after the always-executed extra forwards."""

    optimization_loss: torch.Tensor
    equivariance: SameFrameEquivarianceResult


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameFrameEquivarianceError(message)


def _is_exact_identity(spec: NuisanceSpec) -> bool:
    """Return whether a geometric probe is an exact no-op control."""
    return (
        spec.kind in {"translation", "rotation"}
        and float(spec.dx_pixels) == 0.0
        and float(spec.dy_pixels) == 0.0
        and float(spec.clockwise_degrees) == 0.0
    )


def spatial_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert BxNxHxW logits into a probability distribution per channel."""
    _require(logits.ndim == 4, "logits must be BxNxHxW")
    _require(logits.shape[-2] >= 2 and logits.shape[-1] >= 2, "heatmap is too small")
    _require(bool(torch.isfinite(logits).all()), "logits contain non-finite values")
    return torch.softmax(logits.flatten(-2), dim=-1).reshape_as(logits)


def warp_spatial_distribution(
    probabilities: torch.Tensor,
    spec: NuisanceSpec,
    *,
    input_coordinate_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Warp heatmap mass by the same normalized transform as the input image.

    The spec's translation remains measured in input pixels even when the
    probability map is 64x64.  Zero padding avoids reflecting a probability
    mode across a heatmap boundary; the retained mass is renormalized.
    """
    _require(probabilities.ndim == 4, "probabilities must be BxNxHxW")
    _require(spec.kind in {"translation", "rotation"}, "spec must be geometric")
    _require(input_coordinate_size >= 2, "input coordinate size must be at least two")
    _require(bool(torch.isfinite(probabilities).all()), "probabilities contain non-finite values")
    _require(bool((probabilities >= 0).all()), "probabilities must be non-negative")
    mass = probabilities.sum(dim=(-2, -1), keepdim=True)
    _require(bool((mass > 0).all()), "every heatmap must have positive mass")
    normalized = probabilities / mass
    if _is_exact_identity(spec):
        return normalized
    theta = nuisance_output_to_input_affine(
        spec,
        batch_size=normalized.shape[0],
        coordinate_size=input_coordinate_size,
        reference=normalized,
    )
    grid = F.affine_grid(theta, normalized.shape, align_corners=True)
    warped = F.grid_sample(
        normalized,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).clamp_min(0.0)
    retained_mass = warped.sum(dim=(-2, -1), keepdim=True)
    _require(bool((retained_mass > 0).all()), "transform removed all probability mass")
    return warped / retained_mass


def jensen_shannon_per_heatmap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence for every batch/channel heatmap."""
    _require(first.shape == second.shape and first.ndim == 4, "distribution shapes differ")
    _require(epsilon > 0.0, "epsilon must be positive")
    _require(bool(torch.isfinite(first).all() and torch.isfinite(second).all()), "non-finite distribution")
    _require(bool((first >= 0).all() and (second >= 0).all()), "negative distribution mass")
    first = first / first.sum(dim=(-2, -1), keepdim=True)
    second = second / second.sum(dim=(-2, -1), keepdim=True)
    midpoint = 0.5 * (first + second)
    first_term = torch.where(
        first > 0,
        first * (torch.log(first.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))),
        torch.zeros_like(first),
    )
    second_term = torch.where(
        second > 0,
        second * (torch.log(second.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))),
        torch.zeros_like(second),
    )
    return 0.5 * (first_term + second_term).sum(dim=(-2, -1))


def distribution_equivariance_loss(
    base_heatmaps: torch.Tensor,
    augmented_heatmaps: torch.Tensor,
    *,
    specs: Sequence[NuisanceSpec] = LOCKED_EQUIVARIANCE_SPECS,
    input_coordinate_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare actual transformed heatmaps with warped original distributions.

    ``augmented_heatmaps`` has shape SxBxNxHxW in the same order as ``specs``.
    Both the original and transformed branches remain attached to autograd.
    """
    _require(base_heatmaps.ndim == 4, "base heatmaps must be BxNxHxW")
    _require(len(specs) >= 1, "at least one transform is required")
    expected_shape = (len(specs), *base_heatmaps.shape)
    _require(tuple(augmented_heatmaps.shape) == expected_shape, "augmented heatmap shape differs")
    base_probabilities = spatial_probabilities(base_heatmaps)
    components: dict[str, torch.Tensor] = {}
    for index, spec in enumerate(specs):
        _require(spec.kind in {"translation", "rotation"}, "only geometric transforms are allowed")
        expected = warp_spatial_distribution(
            base_probabilities,
            spec,
            input_coordinate_size=input_coordinate_size,
        )
        actual = spatial_probabilities(augmented_heatmaps[index])
        components[spec.name] = jensen_shannon_per_heatmap(expected, actual).mean()
    return torch.stack(tuple(components.values())).mean(), components


def run_locked_equivariance_path(
    extractor: nn.Module,
    normalized_images: torch.Tensor,
    *,
    specs: Sequence[NuisanceSpec] = LOCKED_EQUIVARIANCE_SPECS,
) -> SameFrameEquivarianceResult:
    """Execute one common-BatchNorm auxiliary forward in both paired arms.

    The untransformed and transformed heatmaps compared by Jensen-Shannon are
    sliced from one extractor call.  This prevents train-mode BatchNorm batch
    composition from becoming part of the equivariance target.
    """
    _require(normalized_images.ndim == 4, "images must be BxCxHxW")
    _require(tuple(normalized_images.shape[-2:]) == (IMAGE_SIZE, IMAGE_SIZE), "locked path requires 512x512 images")
    variants = [
        normalized_images if _is_exact_identity(spec) else apply_nuisance(normalized_images, spec)
        for spec in specs
    ]
    auxiliary_images = torch.cat((normalized_images, *variants), dim=0)
    extractor_output = extractor(auxiliary_images)
    _require(isinstance(extractor_output, tuple) and len(extractor_output) >= 2, "extractor output differs")
    heatmaps_flat = extractor_output[1]
    base_count = normalized_images.shape[0]
    common_batch_base_heatmaps = heatmaps_flat[:base_count]
    augmented_flat = heatmaps_flat[base_count:]
    augmented_heatmaps = augmented_flat.reshape(
        len(specs),
        base_count,
        *augmented_flat.shape[1:],
    )
    loss, components = distribution_equivariance_loss(
        common_batch_base_heatmaps,
        augmented_heatmaps,
        specs=specs,
        input_coordinate_size=IMAGE_SIZE,
    )
    return SameFrameEquivarianceResult(
        loss=loss,
        per_transform_loss=components,
        common_batch_base_heatmaps=common_batch_base_heatmaps,
        augmented_heatmaps=augmented_heatmaps,
        untransformed_image_count=int(base_count),
        transformed_image_count=int(augmented_flat.shape[0]),
        auxiliary_forward_image_count=int(auxiliary_images.shape[0]),
    )


def combine_with_base_loss(
    base_loss: torch.Tensor,
    equivariance_loss: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Add the candidate while retaining exact zero-weight loss aliasing."""
    _require(isinstance(weight, (int, float)), "weight must be numeric")
    weight_tensor = base_loss.new_tensor(float(weight))
    _require(bool(torch.isfinite(weight_tensor)), "weight must be finite")
    _require(weight >= 0.0, "weight must be non-negative")
    if weight == 0.0:
        return base_loss
    return base_loss + float(weight) * equivariance_loss


def add_locked_equivariance_to_pair_loss(
    extractor: nn.Module,
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    heatmaps_t: torch.Tensor,
    heatmaps_t1: torch.Tensor,
    base_optimization_loss: torch.Tensor,
    *,
    weight: float,
) -> PairedEquivarianceLossResult:
    """Execute one fair pair path and combine it with an existing objective.

    The caller must invoke this function in both the zero-weight control and
    positive candidate.  Thus the untransformed/transformed common auxiliary
    extractor forward and BatchNorm exposure are identical before the weight
    is applied.  ``heatmaps_t`` and ``heatmaps_t1`` are the historical base
    forward and are retained only to verify the common auxiliary output shape;
    they are not one side of the Jensen-Shannon comparison.
    """
    _require(x_t.shape == x_t1.shape, "training-pair image shapes differ")
    _require(heatmaps_t.shape == heatmaps_t1.shape, "training-pair heatmap shapes differ")
    _require(x_t.shape[0] == heatmaps_t.shape[0], "image/heatmap batch differs")
    equivariance = run_locked_equivariance_path(
        extractor,
        torch.cat((x_t, x_t1), dim=0),
    )
    historical_shape = tuple(torch.cat((heatmaps_t, heatmaps_t1), dim=0).shape)
    _require(
        tuple(equivariance.common_batch_base_heatmaps.shape) == historical_shape,
        "common auxiliary heatmap shape differs from the historical base forward",
    )
    return PairedEquivarianceLossResult(
        optimization_loss=combine_with_base_loss(
            base_optimization_loss,
            equivariance.loss,
            weight=weight,
        ),
        equivariance=equivariance,
    )


__all__ = [
    "LOCKED_EQUIVARIANCE_SPECS",
    "PairedEquivarianceLossResult",
    "SameFrameEquivarianceError",
    "SameFrameEquivarianceResult",
    "add_locked_equivariance_to_pair_loss",
    "combine_with_base_loss",
    "distribution_equivariance_loss",
    "jensen_shannon_per_heatmap",
    "run_locked_equivariance_path",
    "spatial_probabilities",
    "warp_spatial_distribution",
]
