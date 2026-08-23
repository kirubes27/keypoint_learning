from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.material_transport_gate_io import MaterialTransportIOError
from keypoint_net.run_tapnextpp_bidirectional_teacher import (
    EXPECTED_WITNESSES,
    _canonicalize,
    _coordinate_mapping_smoke,
    _validate_prediction_arrays,
)


def test_reverse_traversal_canonicalizes_without_changing_queries() -> None:
    order = [0, 4, 3, 2, 1]
    positions = np.empty((5, EXPECTED_WITNESSES, 2), dtype=np.float32)
    visibility = np.empty((5, EXPECTED_WITNESSES), dtype=bool)
    for step, frame in enumerate(order):
        positions[step, :, 0] = frame
        positions[step, :, 1] = np.arange(EXPECTED_WITNESSES)
        visibility[step] = (frame + np.arange(EXPECTED_WITNESSES)) % 2 == 0
    canonical_positions, canonical_visibility = _canonicalize(
        order, positions, visibility
    )
    np.testing.assert_array_equal(
        canonical_positions[:, :, 0],
        np.broadcast_to(np.arange(5)[:, None], (5, EXPECTED_WITNESSES)),
    )
    np.testing.assert_array_equal(
        canonical_positions[:, :, 1],
        np.broadcast_to(np.arange(EXPECTED_WITNESSES), (5, EXPECTED_WITNESSES)),
    )
    for frame in range(5):
        np.testing.assert_array_equal(
            canonical_visibility[frame],
            (frame + np.arange(EXPECTED_WITNESSES)) % 2 == 0,
        )


def test_prediction_validator_fails_closed_outside_image() -> None:
    positions = np.zeros((3, EXPECTED_WITNESSES, 2), dtype=np.float32)
    visibility = np.ones((3, EXPECTED_WITNESSES), dtype=bool)
    _validate_prediction_arrays(positions, visibility)
    positions[2, 4, 0] = 512.0
    with pytest.raises(MaterialTransportIOError, match="out-of-image"):
        _validate_prediction_arrays(positions, visibility)


def test_prediction_validator_fails_closed_on_query_shape_change() -> None:
    positions = np.zeros((3, EXPECTED_WITNESSES - 1, 2), dtype=np.float32)
    visibility = np.ones((3, EXPECTED_WITNESSES - 1), dtype=bool)
    with pytest.raises(MaterialTransportIOError, match="query shape differs"):
        _validate_prediction_arrays(positions, visibility)


def test_coordinate_mapping_smoke_records_exact_official_scale_convention() -> None:
    initial = np.asarray(
        [[0.0, 0.0], [0.5, 1.5], [255.5, 255.5], [511.0, 511.0]],
        dtype=np.float32,
    )

    def display_to_model(points, height, width, model_size):
        assert (height, width, model_size) == (512, 512, 256)
        return points * np.asarray([model_size / width, model_size / height])

    def model_to_display(points, height, width, model_size):
        assert (height, width, model_size) == (512, 512, 256)
        return points * np.asarray([width / model_size, height / model_size])

    report = _coordinate_mapping_smoke(initial, display_to_model, model_to_display)
    assert report["maximum_absolute_roundtrip_error_px"] == 0.0
    assert report["display_input_order"] == "x_y"
    assert report["inner_query_order"] == "t_y_x"
    assert report["custom_half_pixel_correction_applied"] is False


def test_coordinate_mapping_smoke_fails_closed_on_mapping_offset() -> None:
    initial = np.zeros((EXPECTED_WITNESSES, 2), dtype=np.float32)

    def display_to_model(points, _height, _width, _model_size):
        return points * 0.5

    def model_to_display(points, _height, _width, _model_size):
        return points * 2.0 + 0.25

    with pytest.raises(MaterialTransportIOError, match="round-trip exceeds"):
        _coordinate_mapping_smoke(initial, display_to_model, model_to_display)
