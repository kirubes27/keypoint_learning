from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.run_frozen_equivariance_alignment import (
    FrozenEquivarianceAlignmentError,
    _average_ranks,
    _spearman,
    _summarize_alignment,
)


def test_average_ranks_handle_ties() -> None:
    values = np.asarray([4.0, 1.0, 1.0, 3.0])
    np.testing.assert_array_equal(_average_ranks(values), [4.0, 1.5, 1.5, 3.0])


def test_spearman_recovers_monotone_and_reverse_order() -> None:
    values = np.arange(10, dtype=np.float64)
    assert _spearman(values, values**3) == 0.9999999999999999
    assert _spearman(values, -values) == -0.9999999999999999


def test_summary_binds_worst_event_and_rank_percentile() -> None:
    js = np.arange(1800, dtype=np.float64).reshape(180, 10)
    residual = np.zeros((180, 10), dtype=np.float64)
    residual[179, 9] = 50.0
    changed = np.zeros((180, 10), dtype=np.bool_)
    changed[179, 9] = True
    summary = _summarize_alignment(js, residual, changed)
    assert summary["observation_count"] == 1800
    assert summary["worst_soft_residual_event"] == {
        "frame_index": 179,
        "channel_index": 9,
        "soft_residual_pixels": 50.0,
        "jensen_shannon": 1799.0,
        "jensen_shannon_rank_percentile": 1.0,
        "hard_peak_changed": True,
    }


def test_average_ranks_reject_nonfinite_values() -> None:
    with pytest.raises(FrozenEquivarianceAlignmentError):
        _average_ranks(np.asarray([0.0, np.nan]))
