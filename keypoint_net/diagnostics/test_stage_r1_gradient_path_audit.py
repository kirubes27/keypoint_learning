import sys
from pathlib import Path

import pytest
import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_r1_gradient_path_audit import (  # noqa: E402
    classify_from_seed_values,
    reconstruction_relative_error,
    vector_metrics,
)


def test_vector_metrics_detect_half_cancellation() -> None:
    coordinate = torch.tensor([1.0, 0.0])
    shape = torch.tensor([-0.5, 0.0])
    result = vector_metrics(coordinate, shape)
    assert result["coordinate_shape_cosine"] == pytest.approx(-1.0)
    assert result["coordinate_descent_multiplier"] == pytest.approx(0.5)
    assert result["harmful_cancellation_fraction"] == pytest.approx(0.5)
    assert result["combined_to_coordinate_norm_ratio"] == pytest.approx(0.5)


def test_reconstruction_relative_error_is_zero_for_exact_sum() -> None:
    coordinate = torch.tensor([1.0, 2.0])
    shape = torch.tensor([-0.25, 0.5])
    direct = coordinate + shape
    assert reconstruction_relative_error(direct, coordinate, shape) == pytest.approx(0.0)


def test_seed_classifier_respects_two_of_three_support_rule() -> None:
    result = classify_from_seed_values(
        {"head": [0.6, 0.7, 0.1], "backbone": [0.0, 0.0, 0.0]},
        support_at_or_beyond=lambda value: value >= 0.5,
        reject_strictly_beyond=lambda value: value < 0.25,
    )
    assert result["status"] == "supported"
    assert result["supported_levels"] == ["head"]


def test_seed_classifier_can_be_mixed() -> None:
    result = classify_from_seed_values(
        {"head": [0.3, 0.3, 0.3], "backbone": [0.1, 0.1, 0.1]},
        support_at_or_beyond=lambda value: value >= 0.5,
        reject_strictly_beyond=lambda value: value < 0.25,
    )
    assert result["status"] == "mixed"
