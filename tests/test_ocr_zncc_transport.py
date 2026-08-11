from __future__ import annotations

from dataclasses import replace

import torch

from keypoint_net.model import spatial_softmax
from keypoint_net.ocr_zncc_transport import (
    OCRZNCCConfig,
    combine_with_base_loss,
    grid_step,
    ocr_zncc_transport_loss,
    sample_rgb_patches,
)


CONFIG = OCRZNCCConfig()


def _coordinate(x: int, y: int) -> torch.Tensor:
    return torch.tensor(
        [-1.0 + 2.0 * x / 63.0, -1.0 + 2.0 * y / 63.0],
        dtype=torch.float32,
    )


def _translated_pair(dx: int = 3, dy: int = -2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    source = torch.rand((1, 3, 64, 64), generator=generator)
    target = torch.roll(source, shifts=(dy, dx), dims=(-2, -1))
    return source, target


def _run_translation(*, target_coordinate_shift_cells: tuple[int, int] = (0, 0)):
    source_rgb, target_rgb = _translated_pair()
    source = _coordinate(31, 31).view(1, 1, 2)
    match = _coordinate(34, 29).view(1, 1, 2)
    target = match + torch.tensor(target_coordinate_shift_cells).view(1, 1, 2) * grid_step()
    target.requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        match,
        target,
        config=CONFIG,
    )
    return result, source, match, target


def test_exact_synthetic_translation_is_recovered() -> None:
    result, _, match, _ = _run_translation(target_coordinate_shift_cells=(1, 0))
    assert result.accepted.tolist() == [[True]]
    assert torch.allclose(result.match_coordinates, match, atol=1e-6, rtol=0.0)
    assert result.target_peak_zncc.item() > 0.999
    assert result.reciprocal_error_cells.item() < 1e-5
    assert result.loss.item() > 0.0


def test_deliberately_ambiguous_constant_patch_abstains() -> None:
    source = torch.full((1, 3, 64, 64), 0.5)
    target = source.clone()
    coordinate = _coordinate(31, 31).view(1, 1, 2)
    candidate = coordinate.clone().requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source,
        target,
        coordinate,
        coordinate,
        candidate,
        config=CONFIG,
    )
    assert result.accepted.tolist() == [[False]]
    assert result.accepted_count == 0
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.target_peak_zncc).all()
    assert torch.isfinite(result.target_peak_margin).all()
    assert torch.isfinite(result.reciprocal_peak_zncc).all()
    assert torch.isfinite(result.reciprocal_peak_margin).all()
    assert result.loss.item() == 0.0


def test_distinct_competitor_margin_excludes_only_adjacent_peak_shoulders() -> None:
    axis = torch.linspace(-1.0, 1.0, 64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.sin(3.0 * xx) + torch.cos(4.0 * yy) + 0.3 * torch.sin(2.0 * xx + yy)
    source_rgb = torch.stack((base, 0.8 * base + 0.2 * xx, 0.6 * base + 0.4 * yy)).unsqueeze(0)
    target_rgb = torch.roll(source_rgb, shifts=(-2, 3), dims=(-2, -1))
    source = _coordinate(31, 31).view(1, 1, 2)
    match = _coordinate(34, 29).view(1, 1, 2)
    target = (match + torch.tensor([[[1.0, 0.0]]]) * grid_step()).requires_grad_(True)
    legacy = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        match,
        target,
        config=CONFIG,
    )
    distinct = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        match,
        target,
        config=replace(CONFIG, peak_margin_exclusion_radius_cells=1),
    )
    assert torch.equal(legacy.match_coordinates, distinct.match_coordinates)
    assert legacy.target_competitor_distance_cells.item() == 1.0
    assert distinct.target_competitor_distance_cells.item() >= 2.0
    assert distinct.target_peak_margin.item() >= legacy.target_peak_margin.item()
    assert distinct.reciprocal_peak_margin.item() >= legacy.reciprocal_peak_margin.item()


def test_distinct_competitor_still_abstains_on_spatially_repeated_evidence() -> None:
    generator = torch.Generator().manual_seed(701)
    repeated_patch = torch.rand((3, 7, 7), generator=generator) * 0.4 + 0.3
    source_image = torch.full((1, 3, 64, 64), 0.5)
    target_image = source_image.clone()
    source_image[0, :, 28:35, 28:35] = repeated_patch
    target_image[0, :, 28:35, 24:31] = repeated_patch
    target_image[0, :, 28:35, 32:39] = repeated_patch
    source = _coordinate(31, 31).view(1, 1, 2)
    search = source.clone()
    target = _coordinate(32, 31).view(1, 1, 2).requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source_image,
        target_image,
        source,
        search,
        target,
        config=replace(CONFIG, peak_margin_exclusion_radius_cells=1),
    )
    assert result.target_peak_zncc.item() > 0.999
    assert result.target_competitor_distance_cells.item() == 8.0
    assert abs(result.target_peak_margin.item()) < 1e-6
    assert not bool(result.accepted.item())
    assert torch.isfinite(result.loss)


def test_reciprocal_failure_abstains() -> None:
    generator = torch.Generator().manual_seed(91)
    source = torch.rand((1, 3, 64, 64), generator=generator) * 0.02 + 0.49
    target = torch.rand((1, 3, 64, 64), generator=generator) * 0.02 + 0.49
    raw_a = torch.randn((3, 7, 7), generator=generator)
    raw_c = torch.randn((3, 7, 7), generator=generator)
    a = 0.5 + 0.18 * raw_a / raw_a.abs().amax()
    b_raw = 0.85 * raw_a + 0.5268 * raw_c
    b = 0.5 + 0.18 * b_raw / b_raw.abs().amax()
    source[..., 21:28, 21:28] = a
    source[..., 21:28, 28:35] = b
    target[..., 19:26, 24:31] = b
    anchor = _coordinate(24, 24).view(1, 1, 2)
    expected_forward = _coordinate(27, 22).view(1, 1, 2)
    candidate = expected_forward.clone().requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source,
        target,
        anchor,
        expected_forward,
        candidate,
        config=CONFIG,
    )
    assert result.target_peak_zncc.item() >= CONFIG.minimum_peak_zncc
    assert result.reciprocal_error_cells.item() > CONFIG.reciprocal_tolerance_cells
    assert result.accepted.tolist() == [[False]]


def test_all_invalid_batch_returns_finite_differentiable_zero() -> None:
    source = torch.zeros((2, 3, 64, 64))
    target = torch.zeros_like(source)
    coordinates = torch.zeros((2, 3, 2))
    candidate = coordinates.clone().requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source,
        target,
        coordinates,
        coordinates,
        candidate,
        config=CONFIG,
    )
    result.loss.backward()
    assert result.accepted_count == 0
    assert torch.isfinite(result.loss)
    assert result.loss.item() == 0.0
    assert candidate.grad is not None
    assert torch.equal(candidate.grad, torch.zeros_like(candidate))


def test_xy_endpoint_coordinate_convention_and_corners() -> None:
    image = torch.zeros((1, 3, 64, 64))
    image[0, 0, 0, 0] = 0.1
    image[0, 0, 0, -1] = 0.2
    image[0, 0, -1, 0] = 0.3
    image[0, 0, -1, -1] = 0.4
    centres = torch.tensor(
        [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]
    )
    patches = sample_rgb_patches(
        image,
        centres,
        patch_size=1,
        grid_size=64,
    )
    assert torch.allclose(
        patches[0, :, 0, 0, 0],
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        atol=1e-7,
        rtol=0.0,
    )


def test_zero_weight_returns_the_exact_base_tensor() -> None:
    base = torch.tensor(2.5, requires_grad=True)
    transport = torch.tensor(9.0, requires_grad=True)
    combined = combine_with_base_loss(base, transport, weight=0.0)
    assert combined is base
    combined.backward()
    assert base.grad.item() == 1.0
    assert transport.grad is None


def test_auxiliary_gradient_skips_source_and_operator_and_reaches_target() -> None:
    source_rgb, target_rgb = _translated_pair()
    source = _coordinate(31, 31).view(1, 1, 2).requires_grad_(True)
    operator = torch.eye(2, requires_grad=True)
    bias = (torch.tensor([3.0, -2.0]) * grid_step()).requires_grad_(True)
    search = torch.matmul(source, operator.t()) + bias.view(1, 1, 2)
    target = (search.detach() + torch.tensor([[[1.0, 0.0]]]) * grid_step()).requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        search,
        target,
        config=CONFIG,
    )
    assert result.accepted_count == 1
    result.loss.backward()
    assert source.grad is None
    assert operator.grad is None
    assert bias.grad is None
    assert target.grad is not None and torch.linalg.vector_norm(target.grad).item() > 0.0


def test_auxiliary_reaches_only_target_heatmap_path() -> None:
    source_rgb, target_rgb = _translated_pair()
    source_logits = torch.full((1, 1, 64, 64), -20.0, requires_grad=True)
    target_logits = torch.full((1, 1, 64, 64), -20.0, requires_grad=True)
    with torch.no_grad():
        source_logits[0, 0, 31, 31] = 20.0
        target_logits[0, 0, 29, 35] = 20.0
    source_coordinate = spatial_softmax(source_logits)
    target_coordinate = spatial_softmax(target_logits)
    operator = torch.eye(2, requires_grad=True)
    bias = (torch.tensor([3.0, -2.0]) * grid_step()).requires_grad_(True)
    search = torch.matmul(source_coordinate, operator.t()) + bias.view(1, 1, 2)
    result = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source_coordinate,
        search,
        target_coordinate,
        config=CONFIG,
    )
    assert result.accepted_count == 1
    result.loss.backward()
    assert source_logits.grad is None
    assert operator.grad is None
    assert bias.grad is None
    assert target_logits.grad is not None
    assert torch.linalg.vector_norm(target_logits.grad).item() > 0.0


def test_loss_is_normalized_by_accepted_match_count() -> None:
    single, source, match, target = _run_translation(target_coordinate_shift_cells=(1, 0))
    source_rgb, target_rgb = _translated_pair()
    doubled = ocr_zncc_transport_loss(
        source_rgb.repeat(2, 1, 1, 1),
        target_rgb.repeat(2, 1, 1, 1),
        source.repeat(2, 1, 1),
        match.repeat(2, 1, 1),
        target.detach().repeat(2, 1, 1).requires_grad_(True),
        config=CONFIG,
    )
    assert single.accepted_count == 1
    assert doubled.accepted_count == 2
    assert torch.allclose(single.loss, doubled.loss, atol=1e-8, rtol=0.0)


def test_unrelated_keypoint_channel_is_not_a_negative() -> None:
    source_rgb, target_rgb = _translated_pair()
    source = torch.stack((_coordinate(31, 31), _coordinate(20, 20))).view(1, 2, 2)
    search = torch.stack((_coordinate(34, 29), _coordinate(23, 18))).view(1, 2, 2)
    target = search.clone().requires_grad_(True)
    first = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        search,
        target,
        config=CONFIG,
    )
    changed_source = source.clone()
    changed_search = search.clone()
    changed_target = target.detach().clone()
    changed_source[:, 1] = _coordinate(45, 45)
    changed_search[:, 1] = _coordinate(48, 43)
    changed_target[:, 1] = _coordinate(47, 43)
    changed_target.requires_grad_(True)
    second = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        changed_source,
        changed_search,
        changed_target,
        config=CONFIG,
    )
    assert torch.equal(first.accepted[:, 0], second.accepted[:, 0])
    assert torch.allclose(
        first.match_coordinates[:, 0],
        second.match_coordinates[:, 0],
        atol=0.0,
        rtol=0.0,
    )


def test_source_patch_crossing_image_boundary_is_rejected() -> None:
    source_rgb, target_rgb = _translated_pair()
    source = _coordinate(1, 31).view(1, 1, 2)
    search = _coordinate(4, 29).view(1, 1, 2)
    target = search.clone().requires_grad_(True)
    result = ocr_zncc_transport_loss(
        source_rgb,
        target_rgb,
        source,
        search,
        target,
        config=CONFIG,
    )
    assert result.source_patch_inside.tolist() == [[False]]
    assert result.accepted.tolist() == [[False]]
