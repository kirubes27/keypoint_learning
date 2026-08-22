from __future__ import annotations

import math

import numpy as np
import torch

from keypoint_net.certified_witness_capability import (
    CELL_SPACING_PX,
    EXPECTED_WITNESSES,
    HALF_CELL_DIAGONAL_PX,
    dense_heatmap_cross_entropy,
    evaluate_predictions,
    evaluation_score,
    gaussian_target_distribution,
    model_state_sha256,
    nearest_r64_grid_prediction,
    normalized_to_pixel,
    pixel_to_normalized,
)


def _targets(frames: int = 3) -> np.ndarray:
    cells = np.asarray(
        [[8, 8], [20, 8], [32, 8], [44, 8], [56, 8], [8, 48], [20, 48], [32, 48], [44, 48], [56, 48]],
        dtype=np.float64,
    )
    normalized = -1.0 + 2.0 * cells / 63.0
    return np.broadcast_to(normalized_to_pixel(normalized), (frames, EXPECTED_WITNESSES, 2)).copy()


def test_pixel_normalized_round_trip_and_cell_contract() -> None:
    points = np.asarray([[0.0, 0.0], [255.5, 255.5], [511.0, 511.0], [419.5, 141.25]])
    np.testing.assert_allclose(normalized_to_pixel(pixel_to_normalized(points)), points, rtol=0.0, atol=1e-12)
    assert CELL_SPACING_PX == 511.0 / 63.0
    assert HALF_CELL_DIAGONAL_PX == CELL_SPACING_PX / math.sqrt(2.0)


def test_gaussian_target_distribution_is_normalized_and_centred() -> None:
    target = torch.tensor(pixel_to_normalized(np.asarray([[[200.0, 300.0]]])), dtype=torch.float32)
    distribution = gaussian_target_distribution(target, sigma_input_px=8.0)
    torch.testing.assert_close(distribution.sum(dim=-1), torch.ones((1, 1)))
    peak = int(torch.argmax(distribution[0, 0]))
    peak_y, peak_x = divmod(peak, 64)
    expected_cell = np.rint((pixel_to_normalized(np.asarray([200.0, 300.0])) + 1.0) * 0.5 * 63.0).astype(int)
    assert (peak_x, peak_y) == (int(expected_cell[0]), int(expected_cell[1]))


def test_dense_heatmap_cross_entropy_has_finite_gradient() -> None:
    logits = torch.zeros((2, EXPECTED_WITNESSES, 64, 64), requires_grad=True)
    target = torch.tensor(pixel_to_normalized(_targets(2)), dtype=torch.float32)
    loss = dense_heatmap_cross_entropy(logits, target)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0.0


def test_planted_grid_prediction_passes_strict_contract() -> None:
    target = _targets(3)
    masks = np.ones((3, 512, 512), dtype=bool)
    prediction = nearest_r64_grid_prediction(target)
    report, _ = evaluate_predictions(prediction, target, masks)
    assert report["strict_capability_pass"] is True
    assert report["violations"] == {
        "outside_half_cell_count": 0,
        "off_object_count": 0,
        "wrong_identity_count": 0,
        "collapsed_pair_count": 0,
    }


def test_identity_swap_and_collision_fail_closed() -> None:
    target = _targets(2)
    masks = np.ones((2, 512, 512), dtype=bool)
    swapped = target.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    swap_report, _ = evaluate_predictions(swapped, target, masks)
    assert swap_report["strict_capability_pass"] is False
    assert swap_report["violations"]["wrong_identity_count"] == 4

    collapsed = target.copy()
    collapsed[:, 1] = collapsed[:, 0]
    collapse_report, _ = evaluate_predictions(collapsed, target, masks)
    assert collapse_report["strict_capability_pass"] is False
    assert collapse_report["violations"]["collapsed_pair_count"] >= 2


def test_checkpoint_score_prioritizes_semantic_violations() -> None:
    target = _targets(1)
    masks = np.ones((1, 512, 512), dtype=bool)
    clean, _ = evaluate_predictions(target, target, masks)
    bad_prediction = target.copy()
    bad_prediction[:, 0, 0] += 20.0
    bad, _ = evaluate_predictions(bad_prediction, target, masks)
    assert evaluation_score(clean) < evaluation_score(bad)


def test_model_state_hash_changes_after_optimizer_step() -> None:
    model = torch.nn.Linear(2, 1)
    before = model_state_sha256(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones((1, 2))).square().sum()
    loss.backward()
    optimizer.step()
    assert model_state_sha256(model) != before

