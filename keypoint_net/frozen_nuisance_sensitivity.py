"""Pure helpers for frozen detector nuisance-sensitivity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


IMAGE_SIZE = 512


class NuisanceSensitivityError(ValueError):
    """Raised when a planted nuisance violates the frozen semantics."""


@dataclass(frozen=True)
class NuisanceSpec:
    name: str
    kind: str
    brightness_delta: float = 0.0
    dx_pixels: float = 0.0
    dy_pixels: float = 0.0
    clockwise_degrees: float = 0.0


LOCKED_SPECS = (
    NuisanceSpec(name="identity_repeat", kind="identity"),
    NuisanceSpec(
        name="brightness_plus_1pct",
        kind="brightness",
        brightness_delta=0.01,
    ),
    NuisanceSpec(
        name="translation_plus_quarter_pixel_xy",
        kind="translation",
        dx_pixels=0.25,
        dy_pixels=0.25,
    ),
    NuisanceSpec(
        name="rotation_plus_quarter_degree_clockwise_image",
        kind="rotation",
        clockwise_degrees=0.25,
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NuisanceSensitivityError(message)


def _imagenet_constants(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return mean, std


def normalized_to_rgb(normalized: torch.Tensor) -> torch.Tensor:
    _require(normalized.ndim == 4 and normalized.shape[1] == 3, "input must be Bx3xHxW")
    mean, std = _imagenet_constants(normalized)
    return normalized * std + mean


def rgb_to_normalized(rgb: torch.Tensor) -> torch.Tensor:
    _require(rgb.ndim == 4 and rgb.shape[1] == 3, "RGB must be Bx3xHxW")
    mean, std = _imagenet_constants(rgb)
    return (rgb - mean) / std


def _normalized_translation(dx_pixels: float, dy_pixels: float, *, size: int) -> tuple[float, float]:
    _require(size >= 2, "image size must be at least two")
    return 2.0 * dx_pixels / (size - 1), 2.0 * dy_pixels / (size - 1)


def apply_nuisance(normalized: torch.Tensor, spec: NuisanceSpec) -> torch.Tensor:
    """Apply one locked nuisance directly to an unmodified normalized batch.

    Positive translation moves image content right/down. Positive rotation is
    clockwise in image coordinates (x right, y down). Geometric transforms use
    endpoint-aligned bilinear sampling with reflection padding.
    """
    _require(normalized.ndim == 4 and normalized.shape[1] == 3, "input must be Bx3xHxW")
    height, width = normalized.shape[-2:]
    _require(height == width == IMAGE_SIZE, "locked nuisance requires 512x512 input")
    _require(bool(torch.isfinite(normalized).all()), "input contains non-finite values")
    if spec.kind == "identity":
        return normalized.clone()

    rgb = normalized_to_rgb(normalized)
    if spec.kind == "brightness":
        _require(spec.brightness_delta > 0.0, "brightness delta must be positive")
        return rgb_to_normalized((rgb + spec.brightness_delta).clamp(0.0, 1.0))

    batch = normalized.shape[0]
    theta = torch.zeros((batch, 2, 3), device=rgb.device, dtype=rgb.dtype)
    if spec.kind == "translation":
        tx, ty = _normalized_translation(spec.dx_pixels, spec.dy_pixels, size=width)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        # affine_grid maps output positions to input sample positions, so use
        # the inverse of the desired content translation.
        theta[:, 0, 2] = -tx
        theta[:, 1, 2] = -ty
    elif spec.kind == "rotation":
        radians = math.radians(spec.clockwise_degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        # Inverse of clockwise content rotation in image x/y coordinates.
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = sine
        theta[:, 1, 0] = -sine
        theta[:, 1, 1] = cosine
    else:
        raise NuisanceSensitivityError(f"unknown nuisance kind: {spec.kind}")

    grid = F.affine_grid(theta, rgb.shape, align_corners=True)
    transformed = F.grid_sample(
        rgb,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return rgb_to_normalized(transformed.clamp(0.0, 1.0))


def undo_nuisance_coordinates(coordinates: torch.Tensor, spec: NuisanceSpec) -> torch.Tensor:
    """Map perturbed-image normalized xy coordinates back to source image xy."""
    _require(coordinates.shape[-1] == 2, "coordinates must end in xy")
    _require(bool(torch.isfinite(coordinates).all()), "coordinates contain non-finite values")
    if spec.kind in {"identity", "brightness"}:
        return coordinates.clone()
    if spec.kind == "translation":
        tx, ty = _normalized_translation(spec.dx_pixels, spec.dy_pixels, size=IMAGE_SIZE)
        delta = coordinates.new_tensor([tx, ty])
        return coordinates - delta
    if spec.kind == "rotation":
        radians = math.radians(spec.clockwise_degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        # Inverse of clockwise content rotation.
        return torch.stack((cosine * x + sine * y, -sine * x + cosine * y), dim=-1)
    raise NuisanceSensitivityError(f"unknown nuisance kind: {spec.kind}")


def apply_nuisance_coordinates(coordinates: torch.Tensor, spec: NuisanceSpec) -> torch.Tensor:
    """Map source normalized xy coordinates into the planted nuisance image."""
    _require(coordinates.shape[-1] == 2, "coordinates must end in xy")
    _require(bool(torch.isfinite(coordinates).all()), "coordinates contain non-finite values")
    if spec.kind in {"identity", "brightness"}:
        return coordinates.clone()
    if spec.kind == "translation":
        tx, ty = _normalized_translation(spec.dx_pixels, spec.dy_pixels, size=IMAGE_SIZE)
        delta = coordinates.new_tensor([tx, ty])
        return coordinates + delta
    if spec.kind == "rotation":
        radians = math.radians(spec.clockwise_degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        return torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)
    raise NuisanceSensitivityError(f"unknown nuisance kind: {spec.kind}")


def normalized_residual_pixels(reference: torch.Tensor, recovered: torch.Tensor) -> torch.Tensor:
    _require(reference.shape == recovered.shape and reference.shape[-1] == 2, "coordinate shapes differ")
    return torch.linalg.vector_norm(recovered - reference, dim=-1) * ((IMAGE_SIZE - 1) / 2.0)


def normalized_residual_cells(
    reference: torch.Tensor,
    recovered: torch.Tensor,
    *,
    heatmap_size: int = 64,
) -> torch.Tensor:
    _require(reference.shape == recovered.shape and reference.shape[-1] == 2, "coordinate shapes differ")
    _require(heatmap_size >= 2, "heatmap size must be at least two")
    return torch.linalg.vector_norm(recovered - reference, dim=-1) * ((heatmap_size - 1) / 2.0)


__all__ = [
    "IMAGE_SIZE",
    "LOCKED_SPECS",
    "NuisanceSensitivityError",
    "NuisanceSpec",
    "apply_nuisance_coordinates",
    "apply_nuisance",
    "normalized_residual_cells",
    "normalized_residual_pixels",
    "normalized_to_rgb",
    "rgb_to_normalized",
    "undo_nuisance_coordinates",
]
