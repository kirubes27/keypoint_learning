from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.material_transport_gate_io import MaterialTransportIOError
from keypoint_net.run_tapnextpp_512_bidirectional_teacher import (
    CPU_MAX_PROJECTED_SECONDS,
    EXPECTED_INTERNAL_QUERIES,
    EXPECTED_SUPPORT_POINTS,
    EXPECTED_WITNESSES,
    _build_internal_queries,
    _canonicalize,
    _coordinate_mapping_smoke,
    _grid_support_points,
    _project_cpu_full_seconds,
    _run_traversal,
)


def test_official_support_grid_is_exact_eight_by_eight_cell_centres() -> None:
    points = _grid_support_points(64, 64.0, 64.0)
    assert points.shape == (64, 2)
    np.testing.assert_array_equal(
        np.unique(points[:, 0]),
        np.asarray([4, 12, 20, 28, 36, 44, 52, 60], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.unique(points[:, 1]),
        np.asarray([4, 12, 20, 28, 36, 44, 52, 60], dtype=np.float32),
    )


def test_internal_queries_keep_real_queries_first_and_add_64_each() -> None:
    real = np.stack(
        [
            np.linspace(100.0, 190.0, EXPECTED_WITNESSES),
            np.linspace(200.0, 290.0, EXPECTED_WITNESSES),
        ],
        axis=-1,
    ).astype(np.float32)
    internal, support = _build_internal_queries(real)
    assert support.shape == (EXPECTED_SUPPORT_POINTS, 2)
    assert internal.shape == (EXPECTED_INTERNAL_QUERIES, 2)
    np.testing.assert_array_equal(internal[:EXPECTED_WITNESSES], real)
    np.testing.assert_array_equal(internal[EXPECTED_WITNESSES:], support)
    np.testing.assert_array_equal(support[0], real[0] + np.asarray([-28.0, -28.0]))
    np.testing.assert_array_equal(support[63], real[0] + np.asarray([28.0, 28.0]))
    np.testing.assert_array_equal(support[64], real[1] + np.asarray([-28.0, -28.0]))


def test_internal_support_clamps_to_image_without_reordering_real_queries() -> None:
    real = np.zeros((EXPECTED_WITNESSES, 2), dtype=np.float32)
    real[-1] = 511.0
    internal, support = _build_internal_queries(real)
    np.testing.assert_array_equal(internal[:EXPECTED_WITNESSES], real)
    assert np.min(support) == 0.0
    assert np.max(support) == 511.0


class _FakeModel:
    def __init__(self, wrong_count: bool = False) -> None:
        self.step = 0
        self.wrong_count = wrong_count

    def track_frame(
        self,
        _frame,
        query_points_xy=None,
        state=None,
        autocast=False,
    ):
        assert autocast is False
        if self.step == 0:
            assert state is None
            assert query_points_xy.shape == (EXPECTED_INTERNAL_QUERIES, 2)
        else:
            assert query_points_xy is None
            assert state == self.step
        count = EXPECTED_INTERNAL_QUERIES - int(self.wrong_count)
        values = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
        visible = np.ones(count, dtype=bool)
        self.step += 1
        return values, visible, self.step


def test_traversal_tracks_650_but_returns_only_ten_real_queries() -> None:
    model = _FakeModel()
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    queries = np.zeros((EXPECTED_INTERNAL_QUERIES, 2), dtype=np.float32)
    positions, visible = _run_traversal(
        model,
        frames,
        queries,
        [0, 1, 2],
        autocast=False,
    )
    assert positions.shape == (3, EXPECTED_WITNESSES, 2)
    assert visible.shape == (3, EXPECTED_WITNESSES)
    np.testing.assert_array_equal(
        positions[0],
        np.arange(EXPECTED_WITNESSES * 2, dtype=np.float32).reshape(
            EXPECTED_WITNESSES, 2
        ),
    )


def test_traversal_fails_closed_if_internal_query_count_changes() -> None:
    model = _FakeModel(wrong_count=True)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    queries = np.zeros((EXPECTED_INTERNAL_QUERIES, 2), dtype=np.float32)
    with pytest.raises(MaterialTransportIOError, match="internal query output shape"):
        _run_traversal(model, frames, queries, [0], autocast=False)


def test_reverse_traversal_canonicalization_is_identity_preserving() -> None:
    order = [0, 4, 3, 2, 1]
    positions = np.empty((5, EXPECTED_WITNESSES, 2), dtype=np.float32)
    visibility = np.ones((5, EXPECTED_WITNESSES), dtype=bool)
    for step, frame in enumerate(order):
        positions[step, :, 0] = frame
        positions[step, :, 1] = np.arange(EXPECTED_WITNESSES)
    canonical_positions, canonical_visibility = _canonicalize(
        order, positions, visibility
    )
    np.testing.assert_array_equal(
        canonical_positions[:, :, 0],
        np.broadcast_to(np.arange(5)[:, None], (5, EXPECTED_WITNESSES)),
    )
    assert bool(canonical_visibility.all())


def test_512_input_uses_official_256_coordinate_roundtrip() -> None:
    points = np.asarray([[0.0, 0.0], [255.5, 255.5], [511.0, 511.0]])

    def display_to_model(values, height, width, model_size):
        assert (height, width, model_size) == (512, 512, 256)
        return values * 0.5

    def model_to_display(values, height, width, model_size):
        assert (height, width, model_size) == (512, 512, 256)
        return values * 2.0

    report = _coordinate_mapping_smoke(points, display_to_model, model_to_display)
    assert report["model_input_resolution"] == [512, 512]
    assert report["model_coordinate_resolution"] == [256, 256]
    assert report["maximum_absolute_roundtrip_error_px"] == 0.0


def test_cpu_projection_remains_bounded_by_two_hour_gate() -> None:
    projection = _project_cpu_full_seconds(3.0, 35.0, 40.0)
    assert projection == pytest.approx(1.25 * (3.0 + 370 * 8.0))
    assert projection < CPU_MAX_PROJECTED_SECONDS
