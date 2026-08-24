from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from final_feature_joint_assignment import (  # noqa: E402
    EXPECTED_FRAMES,
    EXPECTED_WITNESS_IDS,
    JointAssignmentContractError,
    bilinear_sample_all_sites,
    decode_joint_assignments,
    evaluate_predictions,
    extract_final_feature_scores,
    local_readout_at_cells,
    maximize_assignment,
    shared_true_site_cells,
    spatial_local_maxima,
    square_assignment_diagnostics,
    union_candidate_modes,
)


def _archive(scores: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "representation_name": np.asarray(
            (
                "penultimate_encoder_block",
                "final_prehead_feature_map",
                "heatmap_head_logits",
            )
        ),
        "frame_index": EXPECTED_FRAMES.copy(),
        "witness_id": np.asarray(EXPECTED_WITNESS_IDS, dtype=np.int64),
        "score_maps": np.stack((scores - 1.0, scores, scores + 1.0), axis=0),
        "anchor_target_coordinate_px": np.zeros((10, 2), dtype=np.float64),
    }


def test_extracts_only_fixed_final_feature_and_rejects_truth() -> None:
    scores = np.zeros((24, 10, 64, 64), dtype=np.float32)
    frames, witnesses, selected = extract_final_feature_scores(_archive(scores))
    assert np.array_equal(frames, EXPECTED_FRAMES)
    assert tuple(witnesses.tolist()) == EXPECTED_WITNESS_IDS
    assert np.array_equal(selected, scores)
    privileged = _archive(scores)
    privileged["validation_target_coordinate_px"] = np.zeros((24, 10, 2))
    with pytest.raises(JointAssignmentContractError, match="forbidden"):
        extract_final_feature_scores(privileged)


def test_local_maximum_plateau_collapses_to_row_major_first_cell() -> None:
    field = np.zeros((64, 64), dtype=np.float32)
    field[10:12, 20:22] = 5.0
    field[40, 41] = 4.0
    maxima = spatial_local_maxima(field)
    assert (10, 20) in map(tuple, maxima.tolist())
    assert not any(tuple(cell) in {(10, 21), (11, 20), (11, 21)} for cell in maxima)
    assert (40, 41) in map(tuple, maxima.tolist())


def test_candidate_union_is_unique_and_row_major() -> None:
    scores = np.zeros((10, 64, 64), dtype=np.float32)
    for witness in range(10):
        scores[witness, 2 + witness, 20 + witness] = 10.0
    candidates = union_candidate_modes(scores)
    assert len(candidates) >= 10
    assert len({tuple(cell) for cell in candidates.tolist()}) == len(candidates)
    assert candidates.tolist() == sorted(map(list, candidates.tolist()))


def test_rectangular_hungarian_maximizes_total_and_uses_distinct_columns() -> None:
    scores = np.asarray(
        [[9.0, 8.0, 0.0, 0.0], [9.0, 0.0, 7.0, 0.0], [9.0, 0.0, 0.0, 6.0]]
    )
    assignment, total = maximize_assignment(scores)
    assert len(set(assignment.tolist())) == 3
    assert total == pytest.approx(24.0)


def test_local_readout_is_centred_on_assignment_not_global_peak() -> None:
    scores = np.full((1, 10, 64, 64), -20.0, dtype=np.float32)
    scores[:, :, 50, 50] = 20.0
    scores[:, :, 5, 6] = 10.0
    center_y = np.full((1, 10), 5, dtype=np.int64)
    center_x = np.full((1, 10), 6, dtype=np.int64)
    decoded = local_readout_at_cells(scores, center_y, center_x)
    assert np.array_equal(decoded["assigned_cell_y"], center_y)
    assert np.array_equal(decoded["assigned_cell_x"], center_x)
    assert np.all(decoded["assigned_local_3x3_prediction_px"][..., 0] < 60.0)
    assert np.all(decoded["assigned_local_3x3_prediction_px"][..., 1] < 60.0)


def test_joint_decode_replays_exactly_under_reversed_frame_order() -> None:
    rng = np.random.default_rng(20260824)
    scores = rng.normal(size=(4, 10, 64, 64)).astype(np.float32)
    forward = decode_joint_assignments(scores)
    reverse = decode_joint_assignments(scores[::-1])
    assert np.all(forward["candidate_mode_count"] >= 10)
    for name in forward:
        assert np.array_equal(forward[name], reverse[name][::-1]), name


def test_joint_decode_fails_when_union_has_fewer_than_ten_modes() -> None:
    scores = np.zeros((1, 10, 64, 64), dtype=np.float32)
    with pytest.raises(JointAssignmentContractError, match="fewer than ten"):
        decode_joint_assignments(scores)


def test_signed_correct_margin_distinguishes_unique_wrong_optimum() -> None:
    scores = np.eye(10, dtype=np.float64) * 5.0
    correct = square_assignment_diagnostics(scores)
    assert bool(np.asarray(correct["identity_correct"]).all())
    assert correct["signed_correct_assignment_margin"] > 0.0
    assert correct["optimizer_best_minus_second_margin"] > 0.0

    wrong = np.eye(10, dtype=np.float64) * 5.0
    wrong[0, 0] = 0.0
    wrong[1, 1] = 0.0
    wrong[0, 1] = 9.0
    wrong[1, 0] = 9.0
    diagnostic = square_assignment_diagnostics(wrong)
    assert not bool(np.asarray(diagnostic["identity_correct"]).all())
    assert diagnostic["optimizer_best_minus_second_margin"] > 0.0
    assert diagnostic["signed_correct_assignment_margin"] < 0.0


def test_exact_assignment_tie_has_zero_signed_correct_margin() -> None:
    scores = np.zeros((10, 10), dtype=np.float64)
    diagnostic = square_assignment_diagnostics(scores)
    assert diagnostic["signed_correct_assignment_margin"] == pytest.approx(0.0)
    assert diagnostic["optimizer_best_minus_second_margin"] == pytest.approx(0.0)


def test_bilinear_true_site_sampling_and_shared_cell_flags() -> None:
    x_grid = np.arange(64, dtype=np.float64)[None, :]
    y_grid = np.arange(64, dtype=np.float64)[:, None]
    field = x_grid + 2.0 * y_grid
    scores = np.broadcast_to(field, (1, 10, 64, 64)).copy()
    targets = np.zeros((1, 10, 2), dtype=np.float64)
    targets[0, :, 0] = np.linspace(0.0, 511.0, 10)
    targets[0, :, 1] = np.linspace(511.0, 0.0, 10)
    sampled = bilinear_sample_all_sites(scores, targets)
    expected = targets[0, :, 0] / 511.0 * 63.0 + 2.0 * targets[0, :, 1] / 511.0 * 63.0
    assert np.allclose(sampled[0, 0], expected)

    targets[0, 1] = targets[0, 0] + 0.1
    pair, per_site = shared_true_site_cells(targets)
    assert pair[0, 0, 1]
    assert per_site[0, 0] and per_site[0, 1]


def test_pure_numpy_evaluator_reports_material_identity_and_grounding() -> None:
    targets = np.empty((1, 10, 2), dtype=np.float64)
    targets[0, :, 0] = np.linspace(20.0, 470.0, 10)
    targets[0, :, 1] = np.linspace(30.0, 390.0, 10)
    masks = np.ones((1, 512, 512), dtype=bool)
    report, derived = evaluate_predictions(targets.copy(), targets, masks)
    assert report["strict_capability_pass"]
    assert report["violations"] == {
        "outside_half_cell_count": 0,
        "off_object_count": 0,
        "wrong_identity_count": 0,
        "collapsed_pair_count": 0,
    }
    assert bool(derived["identity_correct"].all())
