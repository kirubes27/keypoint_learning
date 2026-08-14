from __future__ import annotations

import math

import torch

from keypoint_net.frozen_nuisance_sensitivity import (
    IMAGE_SIZE,
    LOCKED_SPECS,
    apply_nuisance_coordinates,
    apply_nuisance,
    normalized_residual_cells,
    normalized_residual_pixels,
    normalized_to_rgb,
    rgb_to_normalized,
    undo_nuisance_coordinates,
)
from keypoint_net.run_frozen_nuisance_sensitivity import _spatial_peak_summary


def _spec(name: str):
    return next(item for item in LOCKED_SPECS if item.name == name)


def test_identity_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    value = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    result = apply_nuisance(value, _spec("identity_repeat"))
    assert torch.equal(result, value)
    assert result.data_ptr() != value.data_ptr()


def test_imagenet_round_trip_is_numerically_tight() -> None:
    value = torch.linspace(-2.0, 2.0, 3 * 17 * 19).reshape(1, 3, 17, 19)
    recovered = rgb_to_normalized(normalized_to_rgb(value))
    assert torch.max(torch.abs(recovered - value)).item() < 5e-7


def test_translation_coordinate_undo_recovers_source() -> None:
    spec = _spec("translation_plus_quarter_pixel_xy")
    source = torch.tensor([[[0.2, -0.4], [-0.7, 0.5]]], dtype=torch.float64)
    delta = torch.tensor(
        [2.0 * spec.dx_pixels / (IMAGE_SIZE - 1), 2.0 * spec.dy_pixels / (IMAGE_SIZE - 1)],
        dtype=source.dtype,
    )
    perturbed = apply_nuisance_coordinates(source, spec)
    assert torch.equal(perturbed, source + delta)
    recovered = undo_nuisance_coordinates(perturbed, spec)
    assert torch.equal(recovered, source)


def test_rotation_coordinate_undo_recovers_source() -> None:
    spec = _spec("rotation_plus_quarter_degree_clockwise_image")
    source = torch.tensor([[[0.2, -0.4], [-0.7, 0.5]]], dtype=torch.float64)
    angle = math.radians(spec.clockwise_degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y = source[..., 0], source[..., 1]
    perturbed = apply_nuisance_coordinates(source, spec)
    assert torch.equal(perturbed, torch.stack((c * x - s * y, s * x + c * y), dim=-1))
    recovered = undo_nuisance_coordinates(perturbed, spec)
    assert torch.max(torch.abs(recovered - source)).item() < 1e-15


def test_residual_unit_conversions() -> None:
    source = torch.zeros((1, 1, 2), dtype=torch.float64)
    target = torch.tensor([[[2.0 / (IMAGE_SIZE - 1), 0.0]]], dtype=torch.float64)
    assert normalized_residual_pixels(source, target).item() == 1.0
    assert normalized_residual_cells(source, target).item() == 63.0 / 511.0


def test_geometric_perturbations_preserve_shape_and_finiteness() -> None:
    value = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    for name in (
        "translation_plus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
    ):
        result = apply_nuisance(value, _spec(name))
        assert result.shape == value.shape
        assert torch.isfinite(result).all()


def test_spatial_peak_summary_uses_xy_and_separated_peak() -> None:
    logits = torch.zeros((1, 10, 64, 64))
    logits[:, :, 7, 11] = 5.0
    logits[:, :, 7, 12] = 4.5  # excluded as spatially adjacent
    logits[:, :, 40, 50] = 3.0
    cells, coordinates, margin = _spatial_peak_summary(logits)
    assert torch.equal(cells, torch.full((1, 10), 7 * 64 + 11, dtype=torch.int64))
    assert torch.allclose(coordinates[..., 0], torch.full((1, 10), -1.0 + 22.0 / 63.0))
    assert torch.allclose(coordinates[..., 1], torch.full((1, 10), -1.0 + 14.0 / 63.0))
    assert torch.equal(margin, torch.full((1, 10), 2.0))
