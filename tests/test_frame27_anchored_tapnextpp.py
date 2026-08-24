from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from run_frame27_anchored_tapnextpp import (  # noqa: E402
    EXPECTED_INTERNAL_QUERIES,
    EXPECTED_WITNESSES,
    MODEL_FRAME_CALLS,
    OUTPUT_FRAMES,
    PREFIX_FRAMES,
    TRAVERSAL,
    canonicalize_anchor_traversal,
    projected_full_seconds,
    run_anchor_traversal,
)
from evaluate_frame27_anchored_tapnextpp import (  # noqa: E402
    select_branch,
    visibility_diagnostic,
)


class _FakeModel:
    def __init__(self) -> None:
        self.query_call_count = 0
        self.state_call_count = 0

    def track_frame(
        self,
        frame: np.ndarray,
        *,
        query_points_xy: np.ndarray | None = None,
        state: int | None = None,
        autocast: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        assert autocast is False
        frame_index = int(frame[0, 0, 0])
        if query_points_xy is not None:
            self.query_call_count += 1
            assert state is None
        else:
            self.state_call_count += 1
            assert state is not None
        position = np.full((EXPECTED_INTERNAL_QUERIES, 2), frame_index, dtype=np.float32)
        visible = np.ones(EXPECTED_INTERNAL_QUERIES, dtype=bool)
        return position, visible, frame_index


def test_anchor_traversal_queries_once_then_carries_state() -> None:
    model = _FakeModel()
    frames = {
        frame: np.full((2, 2, 3), frame, dtype=np.uint8)
        for frame in range(OUTPUT_FRAMES)
    }
    query = np.zeros((EXPECTED_INTERNAL_QUERIES, 2), dtype=np.float32)
    positions, visible = run_anchor_traversal(
        model, frames, query, TRAVERSAL, autocast=False
    )
    assert positions.shape == (OUTPUT_FRAMES, EXPECTED_WITNESSES, 2)
    assert visible.shape == (OUTPUT_FRAMES, EXPECTED_WITNESSES)
    assert model.query_call_count == 1
    assert model.state_call_count == OUTPUT_FRAMES - 1
    assert np.array_equal(positions[:, 0, 0], np.asarray(TRAVERSAL))


def test_canonicalization_restores_frame_zero_to_27_order() -> None:
    ordered = np.empty((OUTPUT_FRAMES, EXPECTED_WITNESSES, 2), dtype=np.float32)
    for step, frame in enumerate(TRAVERSAL):
        ordered[step] = frame
    visible = np.ones((OUTPUT_FRAMES, EXPECTED_WITNESSES), dtype=bool)
    canonical, canonical_visible = canonicalize_anchor_traversal(
        TRAVERSAL, ordered, visible
    )
    assert np.array_equal(canonical[:, 0, 0], np.arange(OUTPUT_FRAMES))
    assert canonical_visible.all()


def test_projection_uses_repeated_prefix_and_all_model_calls() -> None:
    projected = projected_full_seconds(10.0, 25.0, 30.0)
    expected = 1.25 * (10.0 + MODEL_FRAME_CALLS * (30.0 / PREFIX_FRAMES))
    assert projected == expected


def _report(*, strict: bool, wrong: int = 0, collapsed: int = 0, off: int = 0):
    return {
        "strict_capability_pass": strict,
        "violations": {
            "wrong_identity_count": wrong,
            "collapsed_pair_count": collapsed,
            "off_object_count": off,
        },
    }


def test_decision_branch_requires_identity_distinctness_and_object_support() -> None:
    assert (
        select_branch(_report(strict=True, wrong=2, collapsed=3, off=4), False)
        == "heldout_wedge_fix_detect_once_track_thereafter"
    )
    assert (
        select_branch(_report(strict=False), True)
        == "bounded_temporal_support_residual_continuous_refinement_needed"
    )
    assert (
        select_branch(_report(strict=False, wrong=1), True)
        == "off_the_shelf_continuation_insufficient_design_domain_adapted_query_tracker"
    )
    assert (
        select_branch(_report(strict=False), False)
        == "off_the_shelf_continuation_insufficient_design_domain_adapted_query_tracker"
    )


def test_visibility_is_diagnostic_and_never_suppresses_predictions() -> None:
    diagnostic = visibility_diagnostic(
        visible=np.asarray([[True, False], [True, False]]),
        within_half=np.asarray([[True, True], [False, False]]),
        material_error=np.asarray([[1.0, 2.0], [9.0, 10.0]]),
    )
    assert diagnostic == {
        "visible_count": 2,
        "invisible_count": 2,
        "invisible_but_within_half_cell_count": 1,
        "visible_but_outside_half_cell_count": 1,
        "visible_material_error_mean_px": 5.0,
        "invisible_material_error_mean_px": 6.0,
        "visibility_used_to_suppress_predictions": False,
    }
