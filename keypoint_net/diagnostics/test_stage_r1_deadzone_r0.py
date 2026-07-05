import sys
from pathlib import Path

import pytest
import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_r1_deadzone_r0 import (  # noqa: E402
    _gaussian_logits,
    _is_healthy,
    compatibility_prototypes,
    directional_compatibility,
    gradient_compatibility,
    prototypes,
)
from diagnostics.stage_a_shape_constraint import conditional_deadzone_shape  # noqa: E402
from model import spatial_softmax  # noqa: E402


def test_prototype_names_and_shapes_are_frozen() -> None:
    values = prototypes()
    assert set(values) == {
        "spike",
        "diffuse",
        "uniform",
        "equal_separated_modes",
    }
    assert all(value.shape == (1, 1, 33, 33) for value in values.values())


def test_compatibility_uses_movable_invalid_spike() -> None:
    values = compatibility_prototypes()
    assert set(values) == {
        "spike_sigma0.35",
        "diffuse",
        "equal_separated_modes",
    }
    spike = conditional_deadzone_shape(values["spike_sigma0.35"])
    assert not _is_healthy(spike)
    assert float(spike.loss) > 0.0


def test_healthy_gaussian_is_inside_dead_zone() -> None:
    output = conditional_deadzone_shape(_gaussian_logits(16.0, 16.0, 1.0))
    assert _is_healthy(output)
    assert float(output.loss) == 0.0


def test_zero_shape_gradient_preserves_coordinate_descent() -> None:
    logits = _gaussian_logits(16.0, 16.0, 1.0)
    coordinate = spatial_softmax(logits).reshape(1, 1, 2)
    target = coordinate + torch.tensor([[[0.5 * 2.0 / 32.0, 0.0]]])
    result = gradient_compatibility(logits, target, weight=100.0)
    assert result["coordinate_descent_multiplier"] == pytest.approx(1.0)
    assert result["weighted_shape_gradient_l2"] == pytest.approx(0.0)
    assert result["gradient_reconstruction_relative_error"] <= 1e-5


def test_directional_compatibility_has_four_named_directions() -> None:
    result = directional_compatibility(
        _gaussian_logits(16.0, 16.0, 1.0), weight=1.0
    )
    assert set(result) == {"+x", "-x", "+y", "-y"}
    assert all(value["finite"] for value in result.values())
