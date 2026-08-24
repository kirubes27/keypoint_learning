"""Semantic-lock tests for the frozen encoder/head localization audit."""

from __future__ import annotations

import numpy as np
import torch

from keypoint_net.encoder_head_localization import (
    explicit_competitor_margins,
    fixed_transition_labels,
    nearest_target_cells,
    pixel_to_normalized,
    target_cell_ranks,
)
from keypoint_net.model import KeypointExtractor


def _cell_to_pixel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack((x, y), axis=-1).astype(np.float64) / 63.0 * 511.0


def test_endpoint_geometry_and_roundtrip() -> None:
    points = np.asarray([[0.0, 0.0], [511.0, 511.0], [255.5, 255.5]])
    normalized = pixel_to_normalized(points)
    assert np.array_equal(normalized[:2], np.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    x, y = nearest_target_cells(points)
    assert np.array_equal(x, np.asarray([0, 63, 32]))
    assert np.array_equal(y, np.asarray([0, 63, 32]))


def test_planted_spatial_rank_is_exact() -> None:
    scores = np.zeros((3, 1, 10, 64, 64), dtype=np.float64)
    x = np.arange(10, dtype=np.int64) * 6 + 3
    y = np.full(10, 31, dtype=np.int64)
    targets = _cell_to_pixel(x[None], y[None])
    for level in range(3):
        for witness in range(10):
            scores[level, 0, witness, y[witness], x[witness]] = 2.0
    planted = [(0, 0), (0, 1), (1, 0)]
    for yy, xx in planted:
        scores[2, 0, 0, yy, xx] = 3.0
    result = target_cell_ranks(scores, targets)
    assert np.all(result["target_cell_rank"][:2] == 1)
    assert result["target_cell_rank"][2, 0, 0] == 4


def test_explicit_competitor_margin_and_wrong_coarse_semantics() -> None:
    scores = np.full((3, 1, 10, 64, 64), -10.0, dtype=np.float64)
    x = np.arange(10, dtype=np.int64) * 6 + 3
    y = np.full(10, 31, dtype=np.int64)
    targets = _cell_to_pixel(x[None], y[None])
    for level in range(3):
        for witness in range(10):
            scores[level, 0, witness, y[witness], x[witness]] = 5.0
            scores[level, 0, witness, 0, 0] = 3.0
            other = (witness + 1) % 10
            scores[level, 0, witness, y[other], x[other]] = 4.0
    hard_x = x[None].copy()
    hard_y = y[None].copy()
    hard_x[0, 0] = 60
    hard_y[0, 0] = 60
    scores[:, 0, 0, 60, 60] = 4.5
    result = explicit_competitor_margins(scores, targets, hard_x, hard_y)
    assert result["head_wrong_coarse_event"].sum() == 1
    assert bool(result["head_wrong_coarse_event"][0, 0])
    assert np.allclose(result["target_minus_competitor_margin"][:, 0, 0], 0.5)
    assert np.all(result["maximum_competitor_source_code"][:, 0, 0] == 1)
    assert np.all(result["target_minus_competitor_margin"][:, 0, 1:] == 1.0)
    assert np.all(result["maximum_competitor_source_code"][:, 0, 1:] == 0)


def test_fixed_transition_labels_do_not_overclaim_nonmonotonic_events() -> None:
    rank = np.asarray(
        [
            [[20, 1, 1, 1, 5]],
            [[20, 20, 1, 1, 1]],
            [[20, 20, 20, 1, 20]],
        ],
        dtype=np.int64,
    )
    wrong = np.asarray([[True, True, True, True, True]])
    labels = fixed_transition_labels(rank, wrong)
    assert labels.tolist() == [[
        "badly_ranked_by_penultimate",
        "lost_in_final_encoder_block",
        "lost_in_heatmap_head",
        "selection_or_local_readout",
        "ambiguous_or_nonmonotonic",
    ]]


def test_manual_stage_partition_replays_model_forward_exactly() -> None:
    torch.manual_seed(7)
    model = KeypointExtractor(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).eval()
    batch = torch.randn(2, 3, 512, 512)
    with torch.inference_mode():
        penultimate = model.encoder[:9](batch)
        final = model.encoder[9:](penultimate)
        logits = model.heatmap_head(final)
        _, expected_logits, expected_final = model(batch, return_descriptor_features=True)
    assert tuple(penultimate.shape) == (2, 128, 64, 64)
    assert torch.equal(final, expected_final)
    assert torch.equal(logits, expected_logits)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"passed {len(tests)} encoder/head localization semantic tests")
