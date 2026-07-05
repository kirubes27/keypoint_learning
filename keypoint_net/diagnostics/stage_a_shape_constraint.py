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

