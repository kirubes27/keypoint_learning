import sys
from pathlib import Path

import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_shape_constraint import (  # noqa: E402
    heatmap_shape_metrics,
    prediction_centered_js,
    probability_and_detached_gaussian,
)


def _take_normalized_step(logits: torch.Tensor, step: float = 0.1) -> torch.Tensor:
    loss = prediction_centered_js(logits).loss
    gradient = torch.autograd.grad(loss, logits)[0]
    norm = torch.linalg.vector_norm(gradient)
    assert torch.isfinite(norm) and float(norm) > 0.0
    return (logits - step * gradient / norm).detach()


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
