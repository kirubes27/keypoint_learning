"""Shared soft-coordinate failure routing for frozen wobble reports."""

from __future__ import annotations

from typing import Any, Mapping


def soft_coordinate_failure_flags(
    soft_metrics: Mapping[str, Any],
    *,
    channel: int,
    soft_thresholds: Mapping[str, float],
    hard_jump_rate: float,
) -> tuple[bool, bool]:
    """Classify soft sliding/wobble with soft-oracle thresholds.

    A hard jump remains an independent wobble trigger.  Hard-coordinate quality
    values are never routed through this helper.
    """
    sliding = float(soft_metrics["canonical_rms_per_channel"][channel]) > float(
        soft_thresholds["canonical_rms_max"]
    )
    wobbling = (
        float(
            soft_metrics["adjacent_plus2_step"]["per_channel"][channel]["maximum"]
        )
        > float(soft_thresholds["adjacent_step_max"])
        or float(
            soft_metrics["cyclic_second_difference"]["per_channel"][channel]["maximum"]
        )
        > float(soft_thresholds["cyclic_second_difference_max"])
        or float(hard_jump_rate) > 0.0
    )
    return sliding, wobbling


__all__ = ["soft_coordinate_failure_flags"]
