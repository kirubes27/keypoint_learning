from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from certified_witness_capability import (  # noqa: E402
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
)
from final_feature_query_decoder import (  # noqa: E402
    BASELINE_MAXIMUM_ERROR_PX,
    EXPECTED_FRAMES,
    TWO_CELL_SPACING_PX,
    decode_final_feature_scores,
    extract_final_feature_scores,
    material_head_rescue_support,
    practical_complete_solution,
    select_decision_branch,
)


def _archive(scores: np.ndarray) -> dict[str, np.ndarray]:
    stacked = np.stack((scores - 1.0, scores, scores + 1.0), axis=0)
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
        "score_maps": stacked,
        "anchor_target_coordinate_px": np.zeros((10, 2), dtype=np.float64),
    }


def _scores(frames: int = 24) -> np.ndarray:
    result = np.zeros((frames, 10, 64, 64), dtype=np.float32)
    result[:, :, 20, 30] = 5.0
    return result


def _report(
    *, wrong_identity: int = 0, collapsed: int = 0, off_object: int = 0, maximum: float = 1.0
) -> dict[str, object]:
    return {
        "violations": {
            "wrong_identity_count": wrong_identity,
            "collapsed_pair_count": collapsed,
            "off_object_count": off_object,
        },
        "material_error_px": {"maximum": maximum},
    }


def test_extracts_only_bound_final_feature_representation() -> None:
    scores = _scores()
    frames, witnesses, selected = extract_final_feature_scores(_archive(scores))
    assert np.array_equal(frames, EXPECTED_FRAMES)
    assert tuple(witnesses.tolist()) == EXPECTED_WITNESS_IDS
    assert np.array_equal(selected, scores)


def test_rejects_privileged_truth_or_mask_key() -> None:
    archive = _archive(_scores())
    archive["validation_target_coordinate_px"] = np.zeros((24, 10, 2))
    with pytest.raises(CapabilityContractError, match="forbidden"):
        extract_final_feature_scores(archive)


def test_row_major_tie_and_endpoint_mapping() -> None:
    scores = np.zeros((1, 10, 64, 64), dtype=np.float32)
    scores[:, :, 0, 0] = 7.0
    scores[:, :, 63, 63] = 7.0
    decoded = decode_final_feature_scores(scores)
    assert np.array_equal(decoded["hard_cell_x"], np.zeros((1, 10), dtype=np.int64))
    assert np.array_equal(decoded["hard_cell_y"], np.zeros((1, 10), dtype=np.int64))
    assert np.array_equal(decoded["hard_prediction_px"], np.zeros((1, 10, 2)))


def test_planted_cell_and_clipped_border_local_readout() -> None:
    scores = np.full((1, 10, 64, 64), -20.0, dtype=np.float32)
    scores[:, :, 0, 0] = 10.0
    scores[:, :, 0, 1] = 9.0
    scores[:, :, 1, 0] = 8.0
    scores[:, :, 1, 1] = 7.0
    decoded = decode_final_feature_scores(scores)
    local = decoded["local_3x3_prediction_px"]
    assert np.all(local >= 0.0)
    assert np.all(local < 511.0 / 63.0)
    assert np.array_equal(decoded["hard_cell_x"], np.zeros((1, 10), dtype=np.int64))
    assert np.array_equal(decoded["hard_cell_y"], np.zeros((1, 10), dtype=np.int64))


def test_reverse_frame_replay_is_exact() -> None:
    rng = np.random.default_rng(20260824)
    scores = rng.normal(size=(5, 10, 64, 64)).astype(np.float32)
    forward = decode_final_feature_scores(scores)
    reverse = decode_final_feature_scores(scores[::-1])
    for name in forward:
        assert np.array_equal(forward[name], reverse[name][::-1])


def test_practical_rule_requires_every_precommitted_component() -> None:
    target = np.zeros((2, 10, 2), dtype=np.float64)
    prediction = np.zeros_like(target)
    passed, components = practical_complete_solution(_report(), prediction, target)
    assert passed
    assert components["all_within_two_cells"]
    prediction[0, 0, 0] = TWO_CELL_SPACING_PX + 1e-6
    passed, components = practical_complete_solution(_report(), prediction, target)
    assert not passed
    assert not components["all_within_two_cells"]


@pytest.mark.parametrize(
    ("wrong_identity", "wrong_coarse", "collapsed", "off_object", "maximum", "expected"),
    [
        (15, 24, 12, 33, BASELINE_MAXIMUM_ERROR_PX, True),
        (16, 24, 12, 33, BASELINE_MAXIMUM_ERROR_PX, False),
        (15, 25, 12, 33, BASELINE_MAXIMUM_ERROR_PX, False),
        (15, 24, 13, 33, BASELINE_MAXIMUM_ERROR_PX, False),
        (15, 24, 12, 34, BASELINE_MAXIMUM_ERROR_PX, False),
        (15, 24, 12, 33, BASELINE_MAXIMUM_ERROR_PX + 1e-6, False),
    ],
)
def test_material_head_rescue_thresholds(
    wrong_identity: int,
    wrong_coarse: int,
    collapsed: int,
    off_object: int,
    maximum: float,
    expected: bool,
) -> None:
    passed, _ = material_head_rescue_support(
        _report(
            wrong_identity=wrong_identity,
            collapsed=collapsed,
            off_object=off_object,
            maximum=maximum,
        ),
        wrong_coarse,
    )
    assert passed is expected


def test_branch_precedence_is_frozen() -> None:
    assert select_decision_branch(
        strict_complete=True, practical_complete=True, head_rescue_supported=True
    ).startswith("strict_numeric")
    assert select_decision_branch(
        strict_complete=False, practical_complete=True, head_rescue_supported=True
    ).startswith("practical_complete")
    assert select_decision_branch(
        strict_complete=False, practical_complete=False, head_rescue_supported=True
    ).startswith("retain_query")
    assert select_decision_branch(
        strict_complete=False, practical_complete=False, head_rescue_supported=False
    ).startswith("reject_raw")
