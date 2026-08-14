from __future__ import annotations

import copy

import torch

from keypoint_net.frozen_nuisance_sensitivity import (
    IMAGE_SIZE,
    NuisanceSpec,
    apply_nuisance,
    normalized_to_rgb,
    rgb_to_normalized,
)
from keypoint_net.model import PhaseAModel
from keypoint_net.same_frame_equivariance import (
    LOCKED_EQUIVARIANCE_SPECS,
    add_locked_equivariance_to_pair_loss,
    combine_with_base_loss,
    distribution_equivariance_loss,
    run_locked_equivariance_path,
    spatial_probabilities,
    warp_spatial_distribution,
)


def _soft_xy(probabilities: torch.Tensor) -> torch.Tensor:
    height, width = probabilities.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=probabilities.dtype),
        torch.linspace(-1.0, 1.0, width, dtype=probabilities.dtype),
        indexing="ij",
    )
    return torch.stack(
        (
            (probabilities * xx).sum(dim=(-2, -1)),
            (probabilities * yy).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )


def _mass_xy_pixels(image: torch.Tensor) -> torch.Tensor:
    mass = image[:, :1].clamp_min(0.0)
    mass = mass / mass.sum(dim=(-2, -1), keepdim=True)
    yy, xx = torch.meshgrid(
        torch.arange(image.shape[-2], dtype=image.dtype),
        torch.arange(image.shape[-1], dtype=image.dtype),
        indexing="ij",
    )
    return torch.stack(
        (
            (mass * xx).sum(dim=(-2, -1)),
            (mass * yy).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )


def test_locked_specs_are_exact_diagnostic_geometric_probes() -> None:
    assert [spec.name for spec in LOCKED_EQUIVARIANCE_SPECS] == [
        "translation_plus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
    ]
    assert LOCKED_EQUIVARIANCE_SPECS[0].dx_pixels == 0.25
    assert LOCKED_EQUIVARIANCE_SPECS[0].dy_pixels == 0.25
    assert LOCKED_EQUIVARIANCE_SPECS[1].clockwise_degrees == 0.25


def test_probability_translation_uses_input_pixel_units_and_correct_sign() -> None:
    size = 17
    one_heatmap_cell_in_input_pixels = (IMAGE_SIZE - 1) / (size - 1)
    spec = NuisanceSpec(
        name="one_heatmap_cell_right_down",
        kind="translation",
        dx_pixels=one_heatmap_cell_in_input_pixels,
        dy_pixels=one_heatmap_cell_in_input_pixels,
    )
    probability = torch.zeros((1, 1, size, size), dtype=torch.float64)
    probability[0, 0, 8, 8] = 1.0
    moved = warp_spatial_distribution(probability, spec)
    expected_delta = torch.tensor([2.0 / (size - 1), 2.0 / (size - 1)], dtype=torch.float64)
    assert torch.allclose(_soft_xy(moved) - _soft_xy(probability), expected_delta.view(1, 1, 2), atol=1e-12, rtol=0.0)


def test_image_translation_and_clockwise_rotation_have_locked_visual_sign() -> None:
    rgb = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float64)
    rgb[:, :, 255, 300] = 1.0
    normalized = rgb_to_normalized(rgb)
    translated = normalized_to_rgb(
        apply_nuisance(
            normalized,
            NuisanceSpec(name="one_right", kind="translation", dx_pixels=1.0),
        )
    )
    assert torch.allclose(
        _mass_xy_pixels(translated),
        torch.tensor([[[301.0, 255.0]]], dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )
    rotated = normalized_to_rgb(
        apply_nuisance(
            normalized,
            NuisanceSpec(name="quarter_turn", kind="rotation", clockwise_degrees=90.0),
        )
    )
    # About the half-pixel centre (255.5, 255.5), (300,255) maps to (256,300).
    assert torch.allclose(
        _mass_xy_pixels(rotated),
        torch.tensor([[[256.0, 300.0]]], dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )


def test_perfectly_warped_distribution_has_numerical_zero_loss() -> None:
    generator = torch.Generator().manual_seed(12)
    base_logits = torch.randn((2, 3, 19, 19), generator=generator, dtype=torch.float64) * 0.2
    base_probability = spatial_probabilities(base_logits)
    augmented_logits = []
    for spec in LOCKED_EQUIVARIANCE_SPECS:
        expected = warp_spatial_distribution(base_probability, spec)
        augmented_logits.append(torch.log(expected.clamp_min(1e-100)))
    augmented = torch.stack(augmented_logits)
    loss, components = distribution_equivariance_loss(base_logits, augmented)
    assert loss.item() < 1e-12
    assert max(value.item() for value in components.values()) < 1e-12


def test_remote_competing_mode_is_penalized() -> None:
    base = torch.full((1, 1, 64, 64), -12.0)
    base[0, 0, 12, 14] = 12.0
    correct = []
    remote = []
    base_probability = spatial_probabilities(base)
    for spec in LOCKED_EQUIVARIANCE_SPECS:
        expected = warp_spatial_distribution(base_probability, spec)
        correct.append(torch.log(expected.clamp_min(1e-20)))
        wrong = torch.full_like(base, -12.0)
        wrong[0, 0, 50, 48] = 12.0
        remote.append(wrong)
    correct_loss, _ = distribution_equivariance_loss(base, torch.stack(correct))
    remote_loss, _ = distribution_equivariance_loss(base, torch.stack(remote))
    assert correct_loss.item() < 1e-6
    assert remote_loss.item() > 0.5


def test_gradients_reach_both_heatmap_branches() -> None:
    generator = torch.Generator().manual_seed(19)
    base = torch.randn((1, 2, 16, 16), generator=generator, requires_grad=True)
    augmented = torch.randn((2, 1, 2, 16, 16), generator=generator, requires_grad=True)
    loss, _ = distribution_equivariance_loss(base, augmented)
    base_gradient, augmented_gradient = torch.autograd.grad(loss, (base, augmented))
    assert torch.isfinite(base_gradient).all() and torch.count_nonzero(base_gradient) > 0
    assert torch.isfinite(augmented_gradient).all() and torch.count_nonzero(augmented_gradient) > 0


def test_full_path_updates_extractor_but_not_operator() -> None:
    generator = torch.Generator().manual_seed(23)
    rgb_t = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    rgb_t1 = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    x_t = rgb_to_normalized(rgb_t)
    x_t1 = rgb_to_normalized(rgb_t1)
    model = PhaseAModel(
        num_keypoints=2,
        base_channels=2,
        operator_type="shared_affine",
        heatmap_res=64,
    )
    outputs = model(x_t, x_t1)
    result = run_locked_equivariance_path(
        model.extractor,
        torch.cat((x_t, x_t1), dim=0),
        torch.cat((outputs["heatmaps_t"], outputs["heatmaps_t1"]), dim=0),
    )
    assert result.transformed_image_count == 4
    result.loss.backward()
    extractor_gradients = [
        parameter.grad
        for parameter in model.extractor.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in extractor_gradients)
    assert all(parameter.grad is None for parameter in model.operator.parameters())


def test_uniform_and_duplicated_heatmaps_expose_trivial_zero_loss_escape() -> None:
    # This is a required negative control: equivariance alone cannot prevent
    # uniform maps or ten channels from duplicating the same stable map.
    base = torch.zeros((1, 10, 32, 32))
    augmented = torch.zeros((2, 1, 10, 32, 32))
    loss, _ = distribution_equivariance_loss(base, augmented)
    # Zero padding introduces only a tiny boundary term; the loss is still
    # roughly five orders of magnitude below the remote-mode counterexample.
    assert loss.item() < 1e-5


def test_zero_weight_returns_exact_base_loss_object() -> None:
    base = torch.tensor(3.0, requires_grad=True)
    auxiliary = torch.tensor(5.0, requires_grad=True)
    combined = combine_with_base_loss(base, auxiliary, weight=0.0)
    assert combined is base
    combined.backward()
    assert base.grad.item() == 1.0
    assert auxiliary.grad is None


def test_paired_control_and_candidate_match_batchnorm_and_operator_gradient() -> None:
    generator = torch.Generator().manual_seed(29)
    x_t = rgb_to_normalized(
        torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    )
    x_t1 = rgb_to_normalized(
        torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    )
    control = PhaseAModel(
        num_keypoints=2,
        base_channels=2,
        operator_type="shared_affine",
        heatmap_res=64,
    )
    candidate = copy.deepcopy(control)
    control.train()
    candidate.train()

    def execute(model: PhaseAModel, weight: float):
        outputs = model(x_t, x_t1)
        base = torch.nn.functional.mse_loss(outputs["p_hat_t1"], outputs["p_t1"])
        paired = add_locked_equivariance_to_pair_loss(
            model.extractor,
            x_t,
            x_t1,
            outputs["heatmaps_t"],
            outputs["heatmaps_t1"],
            base,
            weight=weight,
        )
        return outputs, base, paired

    _, control_base, control_result = execute(control, 0.0)
    _, _, candidate_result = execute(candidate, 0.5)
    assert control_result.optimization_loss is control_base
    for name, control_value in control.state_dict().items():
        assert torch.equal(control_value, candidate.state_dict()[name]), name

    control_result.optimization_loss.backward()
    candidate_result.optimization_loss.backward()
    for control_parameter, candidate_parameter in zip(
        control.operator.parameters(), candidate.operator.parameters(), strict=True
    ):
        assert control_parameter.grad is not None
        assert candidate_parameter.grad is not None
        assert torch.equal(control_parameter.grad, candidate_parameter.grad)
    assert any(
        control_parameter.grad is not None
        and candidate_parameter.grad is not None
        and not torch.equal(control_parameter.grad, candidate_parameter.grad)
        for control_parameter, candidate_parameter in zip(
            control.extractor.parameters(), candidate.extractor.parameters(), strict=True
        )
    )
