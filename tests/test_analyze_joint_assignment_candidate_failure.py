from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from analyze_joint_assignment_candidate_failure import (  # noqa: E402
    pair_distance_counts,
    score_rank_strictly_greater,
    spatial_candidate_assignment,
)


def test_pair_distance_counts_separates_exact_from_nearby_cells() -> None:
    y = np.asarray([[0, 0, 1, 4]], dtype=np.int64)
    x = np.asarray([[0, 0, 1, 4]], dtype=np.int64)
    result = pair_distance_counts(y, x)
    assert result == {
        "exact_same_cell": 1,
        "within_1_cell": 3,
        "within_2_cells": 3,
        "within_4_cells": 6,
        "pair_event_count": 6,
    }


def test_score_rank_counts_only_strictly_greater_candidates() -> None:
    scores = np.asarray([4.0, 7.0, 7.0, 3.0])
    assert score_rank_strictly_greater(scores, 1) == 1
    assert score_rank_strictly_greater(scores, 0) == 3


def test_spatial_candidate_assignment_is_one_to_one_and_near_targets() -> None:
    candidate_y = np.arange(12, dtype=np.int64) * 4
    candidate_x = np.arange(12, dtype=np.int64) * 3
    targets = np.empty((10, 2), dtype=np.float64)
    targets[:, 0] = candidate_x[:10] / 63.0 * 511.0
    targets[:, 1] = candidate_y[:10] / 63.0 * 511.0
    assigned_y, assigned_x, distance = spatial_candidate_assignment(
        candidate_y, candidate_x, targets
    )
    assert np.array_equal(assigned_y, candidate_y[:10])
    assert np.array_equal(assigned_x, candidate_x[:10])
    assert np.allclose(distance, 0.0)
