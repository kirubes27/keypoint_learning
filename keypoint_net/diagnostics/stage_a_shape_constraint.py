"""Prediction-centred heatmap-shape constraint for Stage R0.

The constraint is location-free: it compares each spatial probability map with
a fixed-width Gaussian centred on a detached copy of that map's own expected
coordinate. No target coordinate, mask, transform, or semantic label enters the
shape loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShapeConstraintOutput:
    loss: torch.Tensor
    per_channel_loss: torch.Tensor
    probability: torch.Tensor
    target_gaussian: torch.Tensor
    detached_center_cells: torch.Tensor


@dataclass(frozen=True)
class ConditionalDeadzoneOutput:
    loss: torch.Tensor
    per_channel_loss: torch.Tensor
    probability: torch.Tensor
    max_probability: torch.Tensor
    effective_support_cells: torch.Tensor
    dominant_mass_r2: torch.Tensor
    detached_dominant_index: torch.Tensor


DEADZONE_MAX_PROBABILITY_RANGE = (0.08, 0.30)
DEADZONE_EFFECTIVE_SUPPORT_RANGE = (8.0, 32.0)
DEADZONE_DOMINANT_MASS_R2_MIN = 0.70
DEADZONE_DOMINANT_RADIUS_CELLS = 2.0


def normalized_squared_hinge_band(
    value: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    return (
        torch.relu((low - value) / low) ** 2
        + torch.relu((value - high) / high) ** 2
    )


def normalized_squared_hinge_minimum(
    value: torch.Tensor, minimum: float
) -> torch.Tensor:
    return torch.relu((minimum - value) / minimum) ** 2


def _cell_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def probability_and_detached_gaussian(
    logits: torch.Tensor,
    *,
    sigma_cells: float = 1.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return p, q and detached p-mean, all flattened over space."""
    if logits.ndim != 4:
        raise ValueError(f"expected BxKxHxW logits, got {tuple(logits.shape)}")
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    probability = torch.softmax(flat / temperature, dim=-1)
    grid = _cell_grid(height, width, device=logits.device, dtype=logits.dtype)
    center = torch.einsum("bks,sd->bkd", probability, grid).detach()
    squared_distance = torch.sum(
        (grid[None, None, :, :] - center[:, :, None, :]) ** 2,
        dim=-1,
    )
    gaussian_logits = -0.5 * squared_distance / (sigma_cells**2)
    gaussian = torch.softmax(gaussian_logits, dim=-1)
    return probability, gaussian, center


def prediction_centered_js(
    logits: torch.Tensor,
    *,
    sigma_cells: float = 1.0,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> ShapeConstraintOutput:
    """Jensen-Shannon divergence to a detached-centre Gaussian."""
    probability, gaussian, center = probability_and_detached_gaussian(
        logits,
        sigma_cells=sigma_cells,
        temperature=temperature,
    )
    mixture = 0.5 * (probability + gaussian)
    log_probability = torch.log(probability.clamp_min(eps))
    log_gaussian = torch.log(gaussian.clamp_min(eps))
    log_mixture = torch.log(mixture.clamp_min(eps))
    per_channel = 0.5 * torch.sum(
        probability * (log_probability - log_mixture)
        + gaussian * (log_gaussian - log_mixture),
        dim=-1,
    )
    return ShapeConstraintOutput(
        loss=per_channel.mean(),
        per_channel_loss=per_channel,
        probability=probability,
        target_gaussian=gaussian,
        detached_center_cells=center,
    )


def conditional_deadzone_shape(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-30,
) -> ConditionalDeadzoneOutput:
    """Location-free squared-hinge constraint with an exact healthy dead zone.

    The dominant-neighbourhood mask follows a detached argmax. This is
    piecewise differentiable and deliberately breaks exact mode ties in the
    deterministic direction chosen by ``torch.max``.
    """
    if logits.ndim != 4:
        raise ValueError(f"expected BxKxHxW logits, got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, channels, height, width = logits.shape
    probability = torch.softmax(logits.reshape(batch, channels, -1) / temperature, dim=-1)
    max_probability, dominant_index = probability.max(dim=-1)
    dominant_index = dominant_index.detach()
    entropy = -torch.sum(
        probability * torch.log(probability.clamp_min(eps)), dim=-1
    )
    effective_support = torch.exp(entropy)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=logits.device),
        torch.arange(width, device=logits.device),
        indexing="ij",
    )
    dominant_y = torch.div(dominant_index, width, rounding_mode="floor")
    dominant_x = dominant_index % width
    dominant_mask = (
        (xx[None, None] - dominant_x[:, :, None, None]) ** 2
        + (yy[None, None] - dominant_y[:, :, None, None]) ** 2
        <= DEADZONE_DOMINANT_RADIUS_CELLS**2
    ).flatten(-2)
    dominant_mass = torch.sum(probability * dominant_mask, dim=-1)

    max_low, max_high = DEADZONE_MAX_PROBABILITY_RANGE
    support_low, support_high = DEADZONE_EFFECTIVE_SUPPORT_RANGE
    dominant_min = DEADZONE_DOMINANT_MASS_R2_MIN
    per_channel = (
        normalized_squared_hinge_band(max_probability, max_low, max_high)
        + normalized_squared_hinge_band(
            effective_support, support_low, support_high
        )
        + normalized_squared_hinge_minimum(dominant_mass, dominant_min)
    )
    return ConditionalDeadzoneOutput(
        loss=per_channel.mean(),
        per_channel_loss=per_channel,
        probability=probability,
        max_probability=max_probability,
        effective_support_cells=effective_support,
        dominant_mass_r2=dominant_mass,
        detached_dominant_index=dominant_index,
    )


def heatmap_shape_metrics(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-frame/channel shape metrics without statistical aggregation."""
    probability = torch.softmax(logits.flatten(-2), dim=-1)
    entropy = -torch.sum(
        probability * torch.log(probability.clamp_min(1e-30)), dim=-1
    )
    spatial_size = probability.shape[-1]
    return {
        "max_probability": probability.max(dim=-1).values,
        "normalized_entropy": entropy / math.log(spatial_size),
        "effective_support_cells": torch.exp(entropy),
    }


def coordinate_logit_gradients_per_unit(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Analytic gradient of per-unit x/y-mean MSE with respect to logits.

    Returns BxKx(HW), without the global BxK reduction factor. This makes
    frame/channel sensitivity ratios independent of batch size.
    """
    if target.shape != (*logits.shape[:2], 2):
        raise ValueError(
            f"target shape {tuple(target.shape)} incompatible with "
            f"logits {tuple(logits.shape)}"
        )
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    probability = torch.softmax(flat / temperature, dim=-1)
    grid = _cell_grid(height, width, device=logits.device, dtype=logits.dtype)
    grid_normalized = grid.clone()
    grid_normalized[:, 0] = grid_normalized[:, 0] / (width - 1) * 2.0 - 1.0
    grid_normalized[:, 1] = grid_normalized[:, 1] / (height - 1) * 2.0 - 1.0
    coordinate = torch.einsum("bks,sd->bkd", probability, grid_normalized)
    difference = coordinate - target
    centered = grid_normalized[None, None, :, :] - coordinate[:, :, None, :]
    return (
        probability
        * torch.sum(centered * difference[:, :, None, :], dim=-1)
        / temperature
    )
