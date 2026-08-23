from __future__ import annotations

import torch

from paired_rotation_augmentation import (
    AUGMENTATION_SEED,
    MEAN,
    STD,
    apply_arm_transform,
    draw_proposed_angles,
    exposure_selector,
    pixel_to_normalized_torch,
    transform_points_px,
    warp_masks,
    warp_normalized_images_and_targets,
)


def _constant_batch(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.full((batch_size, 3, 512, 512), 166.0 / 255.0)
    images = (raw - MEAN) / STD
    points_px = torch.tensor([[[300.0, 255.5]]] * batch_size)
    targets = pixel_to_normalized_torch(points_px)
    return images, targets


def test_selector_is_exact_global_alternation() -> None:
    assert torch.equal(
        exposure_selector(0, 6),
        torch.tensor([True, False, True, False, True, False]),
    )
    assert torch.equal(
        exposure_selector(5, 5),
        torch.tensor([False, True, False, True, False]),
    )


def test_angle_schedule_is_deterministic_and_half_open() -> None:
    first = draw_proposed_angles(
        128, torch.Generator(device="cpu").manual_seed(AUGMENTATION_SEED)
    )
    second = draw_proposed_angles(
        128, torch.Generator(device="cpu").manual_seed(AUGMENTATION_SEED)
    )
    assert torch.equal(first, second)
    assert bool(((first >= -180.0) & (first < 180.0)).all())


def test_positive_90_moves_right_point_down_in_image_coordinates() -> None:
    point = torch.tensor([[[300.0, 255.5]]], dtype=torch.float64)
    moved = transform_points_px(point, torch.tensor([90.0]))
    assert torch.allclose(
        moved, torch.tensor([[[255.5, 300.0]]], dtype=torch.float64), atol=1e-10
    )


def test_control_bypass_is_bitwise_exact() -> None:
    images, targets = _constant_batch()
    proposed = torch.tensor((-90.0, -10.0, 10.0, 90.0))
    result_images, result_targets, selector, effective = apply_arm_transform(
        images,
        targets,
        proposed,
        global_exposure_start=0,
        arm="control",
    )
    assert torch.equal(result_images, images)
    assert torch.equal(result_targets, targets)
    assert torch.equal(selector, torch.tensor([True, False, True, False]))
    assert torch.count_nonzero(effective).item() == 0


def test_candidate_preserves_unselected_and_rotates_selected() -> None:
    images, targets = _constant_batch()
    proposed = torch.tensor((90.0, 90.0, -90.0, -90.0))
    result_images, result_targets, selector, effective = apply_arm_transform(
        images,
        targets,
        proposed,
        global_exposure_start=0,
        arm="candidate",
    )
    assert torch.equal(selector, torch.tensor([True, False, True, False]))
    assert torch.equal(result_images[~selector], images[~selector])
    assert torch.equal(result_targets[~selector], targets[~selector])
    expected_selected_px = transform_points_px(
        torch.tensor([[[300.0, 255.5]], [[300.0, 255.5]]]),
        torch.tensor((90.0, -90.0)),
    )
    actual_selected_px = (result_targets[selector] + 1.0) * 255.5
    assert torch.allclose(actual_selected_px, expected_selected_px, atol=1e-4)
    assert torch.equal(effective, torch.tensor((90.0, 0.0, -90.0, 0.0)))


def test_zero_warp_and_constant_background_are_semantically_exact() -> None:
    images, targets = _constant_batch()
    warped, warped_targets = warp_normalized_images_and_targets(
        images, targets, torch.zeros(4)
    )
    raw = warped * STD + MEAN
    assert float(torch.max(torch.abs(raw - 166.0 / 255.0))) <= 1e-7
    assert torch.allclose(warped_targets, targets, atol=1e-7)
    masks = torch.zeros((4, 512, 512), dtype=torch.bool)
    masks[:, 200:300, 200:300] = True
    assert torch.equal(warp_masks(masks, torch.zeros(4)), masks)


def test_zero_warp_on_nonconstant_float32_is_within_locked_tolerance() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    raw = torch.rand((1, 3, 512, 512), generator=generator, dtype=torch.float32)
    images = (raw - MEAN) / STD
    targets = pixel_to_normalized_torch(
        torch.tensor([[[123.25, 376.75]]], dtype=torch.float32)
    )
    warped, warped_targets = warp_normalized_images_and_targets(
        images, targets, torch.zeros(1)
    )
    warped_raw = warped * STD + MEAN
    assert float(torch.max(torch.abs(warped_raw - raw))) <= 1e-4
    target_error_px = torch.max(torch.abs(warped_targets - targets)) * 255.5
    assert float(target_error_px) <= 1e-4
