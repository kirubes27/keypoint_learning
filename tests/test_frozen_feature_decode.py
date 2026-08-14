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
from keypoint_net.evaluate_frozen_feature_decode import _basis_metrics, _json_safe
from keypoint_net.frozen_wobble_forensics import rotate_points


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


def test_physical_full_orbit_track_passes_every_channel() -> None:
    theta = np.arange(180, dtype=np.float64) * 2.0
    angle = np.arange(10, dtype=np.float64) * 36.0
    canonical = np.stack((0.4 * np.cos(np.deg2rad(angle)), 0.4 * np.sin(np.deg2rad(angle))), axis=-1)
    anchor = rotate_points(canonical, theta[27])
    decoded = rotate_points(np.broadcast_to(anchor, (180, 10, 2)), theta[:, None] - theta[27])
    masks = np.ones((180, 512, 512), dtype=bool)
    report, derived = _basis_metrics(decoded, decoded.copy(), anchor, theta, masks)
    assert report["strict_pass_count"] == 10
    assert report["strict_all_ten_pass"] is True
    np.testing.assert_allclose(derived["material_error_pixels"], 0.0, atol=1e-12)


def test_one_large_jump_fails_only_affected_identity_quality() -> None:
    theta = np.arange(180, dtype=np.float64) * 2.0
    angle = np.arange(10, dtype=np.float64) * 36.0
    canonical = np.stack((0.4 * np.cos(np.deg2rad(angle)), 0.4 * np.sin(np.deg2rad(angle))), axis=-1)
    anchor = rotate_points(canonical, theta[27])
    decoded = rotate_points(np.broadcast_to(anchor, (180, 10, 2)), theta[:, None] - theta[27])
    decoded[90, 2] += np.asarray([0.2, -0.1])
    masks = np.ones((180, 512, 512), dtype=bool)
    report, _ = _basis_metrics(decoded, decoded.copy(), anchor, theta, masks)
    assert report["channels"][2]["strict_r64_diagnostic_pass"] is False
    assert report["channels"][2]["checks"]["material_error"] is False


def test_json_safe_converts_nested_numpy_scalars() -> None:
    result = _json_safe({"pass": np.bool_(True), "count": [np.int64(3)], "value": np.float32(1.25)})
    assert result == {"pass": True, "count": [3], "value": 1.25}
    assert type(result["pass"]) is bool
