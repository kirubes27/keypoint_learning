from __future__ import annotations

from keypoint_net.frozen_wobble_classification import soft_coordinate_failure_flags


def _soft_metrics(*, rms: float, step: float, second: float) -> dict:
    return {
        "canonical_rms_per_channel": [rms],
        "adjacent_plus2_step": {"per_channel": [{"maximum": step}]},
        "cyclic_second_difference": {"per_channel": [{"maximum": second}]},
    }


def test_soft_failure_uses_soft_envelope_even_when_value_passes_hard_envelope() -> None:
    # Every value below would pass the historical hard-r64 limits (~0.01-0.07)
    # but must fail the much tighter soft single-Gaussian limits.
    soft_thresholds = {
        "canonical_rms_max": 0.0000035,
        "adjacent_step_max": 0.0000091,
        "cyclic_second_difference_max": 0.0000182,
    }
    sliding, wobbling = soft_coordinate_failure_flags(
        _soft_metrics(rms=0.001, step=0.002, second=0.003),
        channel=0,
        soft_thresholds=soft_thresholds,
        hard_jump_rate=0.0,
    )
    assert sliding
    assert wobbling


def test_hard_jump_remains_an_independent_wobble_trigger() -> None:
    soft_thresholds = {
        "canonical_rms_max": 0.1,
        "adjacent_step_max": 0.1,
        "cyclic_second_difference_max": 0.1,
    }
    sliding, wobbling = soft_coordinate_failure_flags(
        _soft_metrics(rms=0.0, step=0.0, second=0.0),
        channel=0,
        soft_thresholds=soft_thresholds,
        hard_jump_rate=1.0 / 179.0,
    )
    assert not sliding
    assert wobbling
