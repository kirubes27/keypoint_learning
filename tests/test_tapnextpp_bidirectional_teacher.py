from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.material_transport_gate_io import MaterialTransportIOError
from keypoint_net.run_tapnextpp_bidirectional_teacher import (
    EXPECTED_WITNESSES,
    _canonicalize,
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
