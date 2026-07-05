import sys
from pathlib import Path

import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_shape_constraint import (  # noqa: E402
    conditional_deadzone_shape,
    coordinate_logit_gradients_per_unit,
    heatmap_shape_metrics,
    normalized_squared_hinge_band,
    normalized_squared_hinge_minimum,
    prediction_centered_js,
    probability_and_detached_gaussian,
)
from model import spatial_softmax  # noqa: E402


def _take_normalized_step(logits: torch.Tensor, step: float = 0.1) -> torch.Tensor:
    loss = prediction_centered_js(logits).loss
    gradient = torch.autograd.grad(loss, logits)[0]
    norm = torch.linalg.vector_norm(gradient)
    assert torch.isfinite(norm) and float(norm) > 0.0
    return (logits - step * gradient / norm).detach()


def _take_deadzone_step(logits: torch.Tensor, step: float = 0.1) -> torch.Tensor:
    loss = conditional_deadzone_shape(logits).loss
    gradient = torch.autograd.grad(loss, logits)[0]
    norm = torch.linalg.vector_norm(gradient)
    assert torch.isfinite(norm) and float(norm) > 0.0
    return (logits - step * gradient / norm).detach()


def _gaussian_logits(
    *, size: int = 33, center_x: int = 16, center_y: int = 16, sigma: float = 1.0
) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (
        -0.5
        * ((xx - center_x) ** 2 + (yy - center_y) ** 2)
        / sigma**2
    ).reshape(1, 1, size, size).float()


def test_matching_gaussian_has_near_zero_loss() -> None:
    seed = torch.zeros(1, 1, 33, 33)
    _, gaussian, center = probability_and_detached_gaussian(seed)
    logits = torch.log(gaussian.reshape_as(seed).clamp_min(1e-30))
    output = prediction_centered_js(logits)
    assert torch.allclose(center, output.detached_center_cells, atol=1e-5)
    assert float(output.loss) < 1e-7


def test_delta_direction_broadens_probability() -> None:
    logits = torch.zeros(1, 1, 33, 33, requires_grad=True)
    logits.data[0, 0, 16, 16] = 12.0
    before = heatmap_shape_metrics(logits)
    after_logits = _take_normalized_step(logits)
    after = heatmap_shape_metrics(after_logits)
    assert float(after["max_probability"]) < float(before["max_probability"])
    assert float(after["effective_support_cells"]) > float(
        before["effective_support_cells"]
    )


def test_uniform_direction_concentrates_probability() -> None:
    logits = torch.zeros(1, 1, 33, 33, requires_grad=True)
    before_loss = prediction_centered_js(logits).loss
    before = heatmap_shape_metrics(logits)
    after_logits = _take_normalized_step(logits)
    after_loss = prediction_centered_js(after_logits).loss
    after = heatmap_shape_metrics(after_logits)
    assert float(after_loss) < float(before_loss)
    assert float(after["max_probability"]) > float(before["max_probability"])
    assert float(after["effective_support_cells"]) < float(
        before["effective_support_cells"]
    )


def test_symmetric_two_peak_direction_builds_central_unimodal_mass() -> None:
    logits = torch.zeros(1, 1, 33, 33, requires_grad=True)
    logits.data[0, 0, 16, 10] = 10.0
    logits.data[0, 0, 16, 22] = 10.0
    before_probability = torch.softmax(logits.flatten(-2), dim=-1).reshape_as(logits)
    before_loss = prediction_centered_js(logits).loss
    after_logits = _take_normalized_step(logits)
    after_probability = torch.softmax(after_logits.flatten(-2), dim=-1).reshape_as(logits)
    after_loss = prediction_centered_js(after_logits).loss
    assert float(after_loss) < float(before_loss)
    assert float(after_probability[0, 0, 16, 16]) > float(
        before_probability[0, 0, 16, 16]
    )


def test_interior_translation_preserves_shape_loss() -> None:
    first = torch.full((1, 1, 41, 41), -20.0)
    first[0, 0, 20, 14] = 0.0
    first[0, 0, 20, 18] = 0.0
    second = torch.roll(first, shifts=(5, 6), dims=(-2, -1))
    loss_first = prediction_centered_js(first).loss
    loss_second = prediction_centered_js(second).loss
    assert torch.allclose(loss_first, loss_second, atol=1e-6, rtol=1e-6)


def test_rendered_gaussian_center_is_stop_gradient() -> None:
    logits = torch.randn(2, 3, 17, 17, requires_grad=True)
    probability, gaussian, center = probability_and_detached_gaussian(logits)
    assert probability.requires_grad
    assert not gaussian.requires_grad
    assert not center.requires_grad


def test_exact_symmetry_and_extreme_collapse_are_finite() -> None:
    for logits in (
        torch.zeros(2, 3, 17, 17, requires_grad=True),
        torch.nn.functional.one_hot(torch.tensor([0]), 17 * 17)
        .float()
        .reshape(1, 1, 17, 17)
        .mul(80.0)
        .requires_grad_(),
    ):
        output = prediction_centered_js(logits)
        gradient = torch.autograd.grad(output.loss, logits)[0]
        assert torch.isfinite(output.loss)
        assert torch.isfinite(output.per_channel_loss).all()
        assert torch.isfinite(gradient).all()


def test_coordinate_logit_gradient_matches_autograd() -> None:
    logits = torch.randn(2, 3, 9, 9, requires_grad=True)
    target = torch.empty(2, 3, 2).uniform_(-0.5, 0.5)
    coordinate = spatial_softmax(logits)
    loss = torch.nn.functional.mse_loss(coordinate, target)
    autograd = torch.autograd.grad(loss, logits)[0].flatten(-2)
    analytic = coordinate_logit_gradients_per_unit(logits.detach(), target)
    analytic = analytic / (logits.shape[0] * logits.shape[1])
    assert torch.allclose(autograd, analytic, atol=1e-7, rtol=1e-5)


def test_conditional_deadzone_is_exactly_silent_on_healthy_gaussian() -> None:
    logits = _gaussian_logits().requires_grad_()
    output = conditional_deadzone_shape(logits)
    gradient = torch.autograd.grad(output.loss, logits)[0]
    assert float(output.loss) == 0.0
    assert torch.count_nonzero(gradient) == 0
    assert 0.08 < float(output.max_probability) < 0.30
    assert 8.0 < float(output.effective_support_cells) < 32.0
    assert float(output.dominant_mass_r2) > 0.70


def test_conditional_deadzone_healthy_translation_preserves_loss() -> None:
    first = _gaussian_logits(center_x=11, center_y=13)
    second = _gaussian_logits(center_x=20, center_y=19)
    assert float(conditional_deadzone_shape(first).loss) == 0.0
    assert float(conditional_deadzone_shape(second).loss) == 0.0


def test_conditional_deadzone_spike_direction_broadens() -> None:
    logits = _gaussian_logits(sigma=0.25).requires_grad_()
    before = conditional_deadzone_shape(logits)
    after = conditional_deadzone_shape(_take_deadzone_step(logits))
    assert float(before.loss) > 0.0
    assert float(after.max_probability) < float(before.max_probability)
    assert float(after.effective_support_cells) > float(
        before.effective_support_cells
    )


def test_conditional_deadzone_diffuse_direction_concentrates() -> None:
    logits = _gaussian_logits(sigma=5.0).requires_grad_()
    before = conditional_deadzone_shape(logits)
    after = conditional_deadzone_shape(_take_deadzone_step(logits))
    assert float(before.loss) > 0.0
    assert float(after.max_probability) > float(before.max_probability)
    assert float(after.effective_support_cells) < float(
        before.effective_support_cells
    )


def test_conditional_deadzone_equal_modes_are_finite_and_break_tie() -> None:
    first = _gaussian_logits(center_x=10)
    second = _gaussian_logits(center_x=22)
    logits = torch.logaddexp(first, second).requires_grad_()
    before = conditional_deadzone_shape(logits)
    gradient = torch.autograd.grad(before.loss, logits)[0]
    assert float(before.loss) > 0.0
    assert torch.isfinite(gradient).all()
    after = conditional_deadzone_shape(
        (logits - 0.1 * gradient / torch.linalg.vector_norm(gradient)).detach()
    )
    assert float(after.loss) < float(before.loss)


def test_conditional_deadzone_exact_uniform_is_finite_and_active() -> None:
    logits = torch.zeros(1, 1, 17, 17, requires_grad=True)
    output = conditional_deadzone_shape(logits)
    gradient = torch.autograd.grad(output.loss, logits)[0]
    assert float(output.loss) > 0.0
    assert torch.isfinite(gradient).all()
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_conditional_deadzone_scalar_boundaries_have_zero_gradient() -> None:
    for value, loss_function in (
        (0.08, lambda tensor: normalized_squared_hinge_band(tensor, 0.08, 0.30)),
        (0.30, lambda tensor: normalized_squared_hinge_band(tensor, 0.08, 0.30)),
        (8.0, lambda tensor: normalized_squared_hinge_band(tensor, 8.0, 32.0)),
        (32.0, lambda tensor: normalized_squared_hinge_band(tensor, 8.0, 32.0)),
        (0.70, lambda tensor: normalized_squared_hinge_minimum(tensor, 0.70)),
    ):
        tensor = torch.tensor(value, requires_grad=True)
        loss = loss_function(tensor)
        gradient = torch.autograd.grad(loss, tensor)[0]
        assert float(loss) == 0.0
        assert float(gradient) == 0.0
