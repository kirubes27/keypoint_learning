from __future__ import annotations

import copy

import numpy as np

from leakage_safe_distillation_contract import (
    HALF_CELL_DIAGONAL_PX,
    TRAIN_FRAMES,
    TRAIN_PAIR_SOURCE_FRAMES,
    TRAIN_PAIR_TARGET_FRAMES,
    TWO_CELL_SPACING_PX,
    VALIDATION_FRAMES,
    VALIDATION_PAIR_SOURCE_FRAMES,
    VALIDATION_PAIR_TARGET_FRAMES,
    operational_near_pass,
)


def _report() -> dict:
    return {
        "violations": {
            "outside_half_cell_count": 3,
            "wrong_identity_count": 0,
            "collapsed_pair_count": 0,
            "off_object_count": 0,
        },
        "material_error_px": {
            "median": HALF_CELL_DIAGONAL_PX,
            "maximum": TWO_CELL_SPACING_PX,
        },
    }


def test_frozen_frames_and_pairs_are_exact_and_disjoint() -> None:
    assert np.array_equal(TRAIN_FRAMES, np.arange(27, 177))
    assert np.array_equal(VALIDATION_FRAMES, np.arange(24))
    assert set(TRAIN_FRAMES).isdisjoint(set(VALIDATION_FRAMES))
    assert np.array_equal(TRAIN_PAIR_SOURCE_FRAMES, np.arange(27, 174))
    assert np.array_equal(TRAIN_PAIR_TARGET_FRAMES, np.arange(30, 177))
    assert np.array_equal(VALIDATION_PAIR_SOURCE_FRAMES, np.arange(21))
    assert np.array_equal(VALIDATION_PAIR_TARGET_FRAMES, np.arange(3, 24))


def test_operational_near_pass_accepts_only_the_frozen_boundary() -> None:
    assert operational_near_pass(_report()) is True


def test_operational_near_pass_rejects_identity_collapse_or_offobject() -> None:
    for key in ("wrong_identity_count", "collapsed_pair_count", "off_object_count"):
        report = copy.deepcopy(_report())
        report["violations"][key] = 1
        assert operational_near_pass(report) is False


def test_operational_near_pass_rejects_error_threshold_relaxation() -> None:
    report = copy.deepcopy(_report())
    report["material_error_px"]["median"] = HALF_CELL_DIAGONAL_PX + 1e-9
    assert operational_near_pass(report) is False
    report = copy.deepcopy(_report())
    report["material_error_px"]["maximum"] = TWO_CELL_SPACING_PX + 1e-9
    assert operational_near_pass(report) is False
