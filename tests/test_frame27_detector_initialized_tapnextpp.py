from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from evaluate_frame27_detector_initialized_tapnextpp import (  # noqa: E402
    distance_to_object_mask_px,
    select_bridge_branch,
)
from run_frame27_detector_initialized_tapnextpp import (  # noqa: E402
    EXPECTED_WITNESSES,
    select_detector_anchor,
)


def _report(
    *, strict: bool, outside: int = 0, wrong: int = 0, collapsed: int = 0, off: int = 0
):
    return {
        "strict_capability_pass": strict,
        "violations": {
            "outside_half_cell_count": outside,
            "wrong_identity_count": wrong,
            "collapsed_pair_count": collapsed,
            "off_object_count": off,
        },
    }


def test_detector_anchor_selects_unique_frame_27_without_targets() -> None:
    frames = np.asarray([26, 27, 28], dtype=np.int64)
    local = np.zeros((3, EXPECTED_WITNESSES, 2), dtype=np.float64)
    local[1] = 27.5
    selected = select_detector_anchor(frames, local)
    assert selected.dtype == np.float32
    assert np.array_equal(selected, np.full((EXPECTED_WITNESSES, 2), 27.5, np.float32))


def test_detector_anchor_rejects_duplicate_frame_27() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        select_detector_anchor(
            np.asarray([27, 27]),
            np.zeros((2, EXPECTED_WITNESSES, 2), dtype=np.float64),
        )


def test_mask_distance_uses_rounded_prediction_and_nearest_true_pixel() -> None:
    prediction = np.zeros((1, EXPECTED_WITNESSES, 2), dtype=np.float64)
    prediction[:] = (10.2, 10.2)
    prediction[0, 0] = (12.1, 10.0)
    masks = np.zeros((1, 512, 512), dtype=bool)
    masks[0, 10, 10] = True
    distance = distance_to_object_mask_px(prediction, masks)
    assert distance.shape == (1, EXPECTED_WITNESSES)
    assert distance[0, 0] == 2.0
    assert np.all(distance[0, 1:] == 0.0)


def test_bridge_branch_preserves_strict_and_boundary_qualified_results() -> None:
    zero = np.zeros((24, EXPECTED_WITNESSES), dtype=np.float64)
    assert (
        select_bridge_branch(_report(strict=True), zero)
        == "strict_practical_hammer_bridge"
    )
    one = zero.copy()
    one[0, 0] = 1.0
    assert (
        select_bridge_branch(_report(strict=False, off=1), one)
        == "raster_boundary_qualified_material_bridge"
    )
    one[0, 0] = 1.001
    assert (
        select_bridge_branch(_report(strict=False, off=1), one)
        == "detector_initialization_failure_distil_selector"
    )
    assert (
        select_bridge_branch(_report(strict=False, wrong=1), zero)
        == "detector_initialization_failure_distil_selector"
    )
