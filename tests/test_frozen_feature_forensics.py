from __future__ import annotations

import numpy as np

from keypoint_net.frozen_feature_forensics import (
    coordinate_on_mask,
    endpoint_feature_mask,
    feature_match_metrics,
    sample_normalized_feature_vectors,
    select_low_wobble_centres,
)


def test_coordinate_on_mask_uses_endpoint_aligned_nearest_pixel() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[2, 3] = True
    assert coordinate_on_mask(mask, (-1.0, -1.0))
    assert coordinate_on_mask(mask, (0.5, 0.0))
    assert not coordinate_on_mask(mask, (0.0, 0.0))
    assert not coordinate_on_mask(mask, (1.01, 0.0))


def test_endpoint_sampling_reaches_exact_corner_vectors() -> None:
    field = np.zeros((2, 4, 4), dtype=np.float32)
    field[:, 0, 0] = (3.0, 4.0)
    field[:, -1, -1] = (0.0, 7.0)
    sampled = sample_normalized_feature_vectors(field, np.asarray(((-1.0, -1.0), (1.0, 1.0))))
    assert np.allclose(sampled[0], (0.6, 0.8), atol=1e-7)
    assert np.allclose(sampled[1], (0.0, 1.0), atol=1e-7)


def test_feature_match_ranks_unique_physical_target_first() -> None:
    source = np.zeros((3, 4, 4), dtype=np.float32)
    target = np.zeros_like(source)
    source[:, 1, 1] = (1.0, 0.0, 0.0)
    target[:, 2, 2] = (1.0, 0.0, 0.0)
    target[:, 0, 3] = (0.0, 1.0, 0.0)
    mask = np.ones((7, 7), dtype=bool)
    metrics, correlation = feature_match_metrics(
        source,
        target,
        mask,
        source_coordinate=(-1.0 / 3.0, -1.0 / 3.0),
        physical_target_coordinate=(1.0 / 3.0, 1.0 / 3.0),
        detector_target_coordinate=(1.0, -1.0),
        separated_second_peak_coordinate=(1.0, -1.0),
    )
    assert correlation.shape == (4, 4)
    assert metrics["physical_target_similarity"] > 0.999
    assert metrics["detector_target_similarity"] < 0.001
    assert metrics["physical_target_mask_percentile"] == 1.0
    assert metrics["object_argmax_distance_to_physical_cells"] < 1e-12


def test_endpoint_feature_mask_uses_image_endpoints() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[0, 0] = True
    mask[-1, -1] = True
    sampled = endpoint_feature_mask(mask, 4)
    assert sampled[0, 0]
    assert sampled[-1, -1]
    assert np.count_nonzero(sampled) == 2


def test_low_wobble_controls_exclude_spike_neighbourhood_and_tie_break_by_frame() -> None:
    points = np.zeros((20, 2), dtype=np.float64)
    points[9, 0] = 5.0
    selected = select_low_wobble_centres(points, (9,), count=5, exclusion_radius=2)
    assert selected == [1, 2, 3, 4, 5]
    assert all(abs(value - 9) > 2 for value in selected)
