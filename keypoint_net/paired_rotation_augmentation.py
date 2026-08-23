"""Frozen known-warp rotation used by the paired detector experiment."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from certified_witness_capability import require


IMAGE_SIZE = 512
ROTATION_CENTER_PX = (255.5, 255.5)
BACKGROUND_RGB_UINT8 = (166, 166, 166)
AUGMENTATION_SEED = 20260823
ANGLE_MIN_DEG = -180.0
ANGLE_MAX_DEG = 180.0
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

Arm = Literal["control", "candidate"]

_PIXEL_GRID_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _pixel_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is None:
        rows, cols = torch.meshgrid(
            torch.arange(IMAGE_SIZE, dtype=dtype, device=device),
            torch.arange(IMAGE_SIZE, dtype=dtype, device=device),
            indexing="ij",
        )
        cached = torch.stack((cols, rows), dim=-1)
        _PIXEL_GRID_CACHE[key] = cached
    return cached


def _validate_angles(angles_deg: torch.Tensor, batch_size: int) -> torch.Tensor:
    angles = torch.as_tensor(angles_deg, dtype=torch.float32, device="cpu")
    require(tuple(angles.shape) == (batch_size,), "angle vector shape differs")
    require(bool(torch.isfinite(angles).all()), "angle vector is non-finite")
    require(
        bool(((angles >= ANGLE_MIN_DEG) & (angles < ANGLE_MAX_DEG)).all()),
        "angle lies outside the frozen half-open range",
    )
    return angles


def draw_proposed_angles(
    batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw the frozen float32 uniform angle schedule on CPU."""

    require(batch_size > 0, "batch size must be positive")
    unit = torch.rand((batch_size,), generator=generator, dtype=torch.float32)
    return unit * (ANGLE_MAX_DEG - ANGLE_MIN_DEG) + ANGLE_MIN_DEG


def exposure_selector(global_exposure_start: int, batch_size: int) -> torch.Tensor:
    """Select exactly the even zero-based sample exposures."""

    require(global_exposure_start >= 0, "global exposure start is negative")
    require(batch_size > 0, "batch size must be positive")
    indices = torch.arange(
        global_exposure_start,
        global_exposure_start + batch_size,
        dtype=torch.int64,
    )
    return indices.remainder(2).eq(0)


def transform_points_px(
    points_px: torch.Tensor, angles_deg: torch.Tensor
) -> torch.Tensor:
    """Forward image-down rotation of BxKx2 pixel coordinates."""

    points = torch.as_tensor(points_px)
    require(points.ndim == 3 and points.shape[-1] == 2, "point tensor shape differs")
    angles = _validate_angles(angles_deg, int(points.shape[0])).to(
        device=points.device, dtype=points.dtype
    )
    radians = angles * (math.pi / 180.0)
    cosine = torch.cos(radians)[:, None]
    sine = torch.sin(radians)[:, None]
    center = torch.tensor(
        ROTATION_CENTER_PX, dtype=points.dtype, device=points.device
    ).view(1, 1, 2)
    shifted = points - center
    x = cosine * shifted[..., 0] - sine * shifted[..., 1]
    y = sine * shifted[..., 0] + cosine * shifted[..., 1]
    return torch.stack((x, y), dim=-1) + center


def normalized_to_pixel_torch(points_normalized: torch.Tensor) -> torch.Tensor:
    return (points_normalized + 1.0) * (IMAGE_SIZE - 1) * 0.5


def pixel_to_normalized_torch(points_px: torch.Tensor) -> torch.Tensor:
    return points_px / (IMAGE_SIZE - 1) * 2.0 - 1.0


def _inverse_sampling_grid(
    angles_deg: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = int(angles_deg.numel())
    angles = _validate_angles(angles_deg, batch_size).to(device=device, dtype=dtype)
    output = _pixel_grid(device, dtype)
    center = torch.tensor(
        ROTATION_CENTER_PX, dtype=dtype, device=device
    ).view(1, 1, 1, 2)
    shifted = output.view(1, IMAGE_SIZE, IMAGE_SIZE, 2) - center
    radians = angles * (math.pi / 180.0)
    cosine = torch.cos(radians).view(-1, 1, 1)
    sine = torch.sin(radians).view(-1, 1, 1)
    # Inverse map for forward x'=c*x-s*y, y'=s*x+c*y.
    source_x = cosine * shifted[..., 0] + sine * shifted[..., 1]
    source_y = -sine * shifted[..., 0] + cosine * shifted[..., 1]
    source = torch.stack((source_x, source_y), dim=-1) + center
    return source / (IMAGE_SIZE - 1) * 2.0 - 1.0


def warp_normalized_images_and_targets(
    images_normalized: torch.Tensor,
    targets_normalized: torch.Tensor,
    angles_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exactly one shared bilinear RGB / analytic target rotation."""

    images = torch.as_tensor(images_normalized)
    targets = torch.as_tensor(targets_normalized)
    require(
        images.ndim == 4
        and tuple(images.shape[1:]) == (3, IMAGE_SIZE, IMAGE_SIZE),
        "image tensor shape differs",
    )
    require(
        targets.ndim == 3
        and targets.shape[0] == images.shape[0]
        and targets.shape[-1] == 2,
        "target tensor shape differs",
    )
    require(images.dtype == torch.float32, "image dtype differs")
    require(targets.dtype == torch.float32, "target dtype differs")
    angles = _validate_angles(angles_deg, int(images.shape[0]))
    grid = _inverse_sampling_grid(
        angles, device=images.device, dtype=images.dtype
    )
    mean = MEAN.to(device=images.device, dtype=images.dtype)
    std = STD.to(device=images.device, dtype=images.dtype)
    raw = images * std + mean
    background = torch.tensor(
        BACKGROUND_RGB_UINT8, device=images.device, dtype=images.dtype
    ).view(1, 3, 1, 1) / 255.0
    warped_raw = F.grid_sample(
        raw - background,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ) + background
    points_px = normalized_to_pixel_torch(targets)
    warped_points_px = transform_points_px(points_px, angles)
    require(
        bool(
            (
                (warped_points_px >= 0.0)
                & (warped_points_px <= IMAGE_SIZE - 1)
            ).all()
        ),
        "a transformed target left the canvas",
    )
    warped_targets = pixel_to_normalized_torch(warped_points_px)
    return (warped_raw.clamp(0.0, 1.0) - mean) / std, warped_targets


def warp_masks(masks: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour audit warp for Bx512x512 binary masks."""

    values = torch.as_tensor(masks)
    require(
        values.ndim == 3
        and tuple(values.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE),
        "mask tensor shape differs",
    )
    angles = _validate_angles(angles_deg, int(values.shape[0]))
    grid = _inverse_sampling_grid(
        angles, device=values.device, dtype=torch.float32
    )
    warped = F.grid_sample(
        values.float()[:, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    return warped > 0.5


def apply_arm_transform(
    images: torch.Tensor,
    targets: torch.Tensor,
    proposed_angles: torch.Tensor,
    *,
    global_exposure_start: int,
    arm: Arm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the locked candidate transform while preserving exact controls."""

    require(arm in ("control", "candidate"), "unknown paired arm")
    batch_size = int(images.shape[0])
    proposed = _validate_angles(proposed_angles, batch_size)
    selector = exposure_selector(global_exposure_start, batch_size)
    effective = torch.zeros_like(proposed)
    if arm == "control":
        # Exact bypass preserves the historical control tensor path.
        return images, targets, selector, effective

    effective[selector] = proposed[selector]
    output_images = images.clone()
    output_targets = targets.clone()
    selected = torch.nonzero(selector, as_tuple=False).flatten()
    warped_images, warped_targets = warp_normalized_images_and_targets(
        images[selected], targets[selected], proposed[selected]
    )
    output_images[selected] = warped_images
    output_targets[selected] = warped_targets
    return output_images, output_targets, selector, effective


__all__ = [
    "ANGLE_MAX_DEG",
    "ANGLE_MIN_DEG",
    "AUGMENTATION_SEED",
    "BACKGROUND_RGB_UINT8",
    "IMAGE_SIZE",
    "ROTATION_CENTER_PX",
    "apply_arm_transform",
    "draw_proposed_angles",
    "exposure_selector",
    "normalized_to_pixel_torch",
    "pixel_to_normalized_torch",
    "transform_points_px",
    "warp_masks",
    "warp_normalized_images_and_targets",
]
