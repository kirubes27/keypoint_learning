from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from evaluate_frozen_operator_on_tracker_tracks import (  # noqa: E402
    aggregate_arm,
    operator_metrics,
    pixel_xy_to_normalized,
    rotation_matrix,
    select_decision,
    validate_pair_rows,
)


CRITERIA = {
    "absolute_angle_error_degrees_max": 0.25,
    "matrix_frobenius_error_max": 0.01,
    "bias_l2_max": 0.005,
    "validation_pair_mse_max": 1e-4,
    "validation_identity_normalized_mse_max": 0.1,
}


def test_pixel_normalization_is_endpoint_aligned_xy_without_y_flip() -> None:
    normalized = pixel_xy_to_normalized(np.asarray([[0.0, 0.0], [511.0, 511.0], [255.5, 255.5]]))
    assert np.array_equal(normalized[0], [-1.0, -1.0])
    assert np.array_equal(normalized[1], [1.0, 1.0])
    assert np.array_equal(normalized[2], [0.0, 0.0])


def test_exact_six_degree_operator_passes_historical_metrics() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(-0.8, 0.8, size=(21, 10, 2))
    matrix = rotation_matrix(6.0)
    target = source @ matrix.T
    metrics, per_pair = operator_metrics(source, target, matrix, np.zeros(2), CRITERIA)
    assert metrics["passes_all_criteria"] is True
    assert metrics["proper_rotation_angle_degrees"] == pytest.approx(6.0)
    assert metrics["validation_pair_mse"] == pytest.approx(0.0, abs=1e-30)
    assert np.allclose(per_pair, 0.0, atol=1e-30)


def test_validation_rows_lock_stride_sign_and_endpoint_sets() -> None:
    pairs = []
    for source in range(21):
        pairs.append(
            {
                "model_name": "engineers_hammer_vray",
                "object_role": "development",
                "direction": "forward",
                "physical_axis": "world_z",
                "stride": 3,
                "signed_generator": 6.0,
                "src_frame_index": source,
                "dst_frame_index": source + 3,
                "pair_id": f"pair-{source}",
            }
        )
    source, target, pair_ids = validate_pair_rows(
        {
            "schema_version": "representation_pair_index.v1",
            "split": "validation",
            "pair_count": 21,
            "pairs": pairs,
        }
    )
    assert np.array_equal(source, np.arange(21))
    assert np.array_equal(target, np.arange(3, 24))
    assert pair_ids[0] == "pair-0"


def _cells(pass_count_per_recipe: int):
    cells = []
    for recipe in ("task55_clean", "task80_assisted"):
        for offset, seed in enumerate((42, 43, 44)):
            cells.append(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "metrics": {
                        "passes_all_criteria": offset < pass_count_per_recipe,
                        "validation_pair_mse": 1e-6,
                        "validation_identity_normalized_mse": 1e-3,
                    },
                }
            )
    return cells


def test_arm_aggregation_and_decision_require_two_of_three_per_recipe() -> None:
    passing = aggregate_arm(_cells(2))
    failing = aggregate_arm(_cells(1))
    assert passing["arm_succeeds"] is True
    assert failing["arm_succeeds"] is False
    assert (
        select_decision(
            {
                "reference_material_targets": passing,
                "certified_anchor_tracker": passing,
                "detector_initialized_tracker": passing,
            }
        )
        == "both_tracker_arms_operator_compatible_short_horizon"
    )
    assert (
        select_decision(
            {
                "reference_material_targets": passing,
                "certified_anchor_tracker": passing,
                "detector_initialized_tracker": failing,
            }
        )
        == "certified_tracker_compatible_detector_initialization_blocks_bridge"
    )
