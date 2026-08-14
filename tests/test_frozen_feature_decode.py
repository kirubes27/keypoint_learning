from __future__ import annotations

import numpy as np
import torch

from keypoint_net.frozen_feature_decode import (
    cosine_correlation_maps,
    endpoint_cells_to_coordinates,
    sample_anchor_descriptors,
    sample_target_similarities,
    stable_spatial_top_two,
)


def test_endpoint_cells_reach_exact_corners() -> None:
    result = endpoint_cells_to_coordinates(np.asarray([0, 63]), np.asarray([63, 0]))
    np.testing.assert_allclose(result, [[-1.0, 1.0], [1.0, -1.0]], rtol=0.0, atol=0.0)


def test_spatial_top_two_uses_stable_ties_and_radius() -> None:
    field = np.zeros((1, 8, 8), dtype=np.float64)
    field[0, 2, 3] = 5.0
    field[0, 2, 4] = 6.0
    field[0, 7, 0] = 5.0
    result = stable_spatial_top_two(field, exclusion_radius_cells=4.0)
    assert int(result["top1_x_cell"][0]) == 4
    assert int(result["top1_y_cell"][0]) == 2
    assert int(result["top2_x_cell"][0]) == 0
    assert int(result["top2_y_cell"][0]) == 7
    assert float(result["margin"][0]) == 1.0


def test_anchor_sampling_and_cosine_decode_are_exact_at_endpoints() -> None:
    features = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    features[0, :, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    features[0, :, 3, 3] = torch.tensor([0.0, 1.0, 0.0])
    coordinates = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]], dtype=torch.float32)
    anchors = sample_anchor_descriptors(features, coordinates)
    correlation = cosine_correlation_maps(anchors, features)
    assert tuple(correlation.shape) == (1, 1, 2, 4, 4)
    assert int(torch.argmax(correlation[0, 0, 0]).item()) == 0
    assert int(torch.argmax(correlation[0, 0, 1]).item()) == 15


def test_detector_similarity_matches_correlation_at_same_cell() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((2, 5, 4, 4), generator=generator)
    anchors = torch.nn.functional.normalize(torch.randn((2, 3, 5), generator=generator), dim=-1)
    coords = torch.tensor(
        [
            [[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]],
            [[[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]],
        ],
        dtype=torch.float32,
    )
    correlation = cosine_correlation_maps(anchors, features)
    sampled = sample_target_similarities(anchors, features, coords)
    for basis in range(2):
        for batch in range(2):
            for keypoint in range(3):
                x = 0 if float(coords[basis, batch, keypoint, 0]) == -1.0 else 3
                y = 0 if float(coords[basis, batch, keypoint, 1]) == -1.0 else 3
                torch.testing.assert_close(sampled[basis, batch, keypoint], correlation[basis, batch, keypoint, y, x])
