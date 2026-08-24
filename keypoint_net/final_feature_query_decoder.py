"""Pure contract for the frozen final-feature query-decoder gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    FEATURE_SIZE,
    HALF_CELL_DIAGONAL_PX,
    require,
)
from certified_witness_local_readout import readout_arrays


EXPECTED_FRAMES = np.arange(24, dtype=np.int64)
EXPECTED_SOURCE_RAW_RECEIPT_SHA256 = (
    "ad8e46e23fc75e1459f7f78047436167c7009ae998ed17c99d901d9b0d2d0527"
)
EXPECTED_SOURCE_SCORE_SHA256 = (
    "d5ea9d47cb3c475c7c237a5ee7cd6347af3dedded9685985f83aa504bca92fc2"
)
EXPECTED_SEMANTIC_LOCK_SHA256 = (
    "4dc1fd6a12b08a607c4e127c0d5c070ac4a826ff2bcd5d964894459851c0330a"
)
FINAL_FEATURE_INDEX = 1
FINAL_FEATURE_NAME = "final_prehead_feature_map"
TWO_CELL_SPACING_PX = 2.0 * 511.0 / 63.0

BASELINE_WRONG_COARSE = 49
BASELINE_WRONG_IDENTITY = 31
BASELINE_COLLAPSED_PAIR = 12
BASELINE_OFF_OBJECT = 33
BASELINE_MAXIMUM_ERROR_PX = 159.18726219197546


def extract_final_feature_scores(
    archive: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the frozen truth-free archive and select representation index 1."""

    names = set(archive)
    forbidden_tokens = (
        "validation_target",
        "validation_truth",
        "mask",
        "operator",
        "tracker",
        "prior_evaluation",
    )
    exposed = sorted(
        name for name in names if any(token in name.lower() for token in forbidden_tokens)
    )
    require(not exposed, f"score archive exposes forbidden prediction inputs: {exposed}")
    required = {"representation_name", "frame_index", "witness_id", "score_maps"}
    require(required.issubset(names), "score archive omits required arrays")

    representation_name = tuple(np.asarray(archive["representation_name"]).tolist())
    require(len(representation_name) == 3, "representation count differs")
    require(
        representation_name[FINAL_FEATURE_INDEX] == FINAL_FEATURE_NAME,
        "final-feature representation identity differs",
    )
    frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
    witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
    require(np.array_equal(frame_index, EXPECTED_FRAMES), "held-out frame order differs")
    require(
        tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS,
        "witness identity/order differs",
    )
    score_maps = np.asarray(archive["score_maps"])
    require(
        score_maps.shape
        == (3, len(EXPECTED_FRAMES), EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "score-map shape differs",
    )
    final_scores = np.asarray(score_maps[FINAL_FEATURE_INDEX], dtype=np.float32)
    require(bool(np.isfinite(final_scores).all()), "final-feature score is non-finite")
    return frame_index, witness_id, final_scores


def decode_final_feature_scores(score_maps: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the precommitted hard-cell plus clipped-local-3x3 readout."""

    scores = np.asarray(score_maps)
    require(
        scores.ndim == 4
        and scores.shape[1:] == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "final-feature query score shape differs",
    )
    require(bool(np.isfinite(scores).all()), "final-feature query score is non-finite")
    return readout_arrays(scores)


def practical_complete_solution(
    report: Mapping[str, Any], prediction_px: np.ndarray, target_px: np.ndarray
) -> tuple[bool, dict[str, Any]]:
    """Return the frozen practical-envelope decision and its raw components."""

    prediction = np.asarray(prediction_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    require(prediction.shape == target.shape, "practical prediction/target shape differs")
    error = np.linalg.norm(prediction - target, axis=-1)
    violations = report["violations"]
    components = {
        "median_within_half_cell": bool(np.median(error) <= HALF_CELL_DIAGONAL_PX + 1e-12),
        "all_within_two_cells": bool(np.all(error <= TWO_CELL_SPACING_PX + 1e-12)),
        "zero_wrong_identity": int(violations["wrong_identity_count"]) == 0,
        "zero_collapsed_pairs": int(violations["collapsed_pair_count"]) == 0,
        "zero_off_object": int(violations["off_object_count"]) == 0,
        "two_cell_threshold_px": TWO_CELL_SPACING_PX,
        "within_two_cells_count": int(np.sum(error <= TWO_CELL_SPACING_PX + 1e-12)),
        "event_count": int(error.size),
    }
    passed = bool(
        components["median_within_half_cell"]
        and components["all_within_two_cells"]
        and components["zero_wrong_identity"]
        and components["zero_collapsed_pairs"]
        and components["zero_off_object"]
    )
    return passed, components


def material_head_rescue_support(
    report: Mapping[str, Any], wrong_coarse_count: int
) -> tuple[bool, dict[str, Any]]:
    """Evaluate the precommitted component-localization rescue rule."""

    violations = report["violations"]
    components = {
        "wrong_identity_at_most_15": int(violations["wrong_identity_count"]) <= 15,
        "wrong_coarse_at_most_24": int(wrong_coarse_count) <= 24,
        "collapsed_pairs_no_worse_than_12": int(violations["collapsed_pair_count"]) <= 12,
        "off_object_no_worse_than_33": int(violations["off_object_count"]) <= 33,
        "maximum_error_no_worse_than_baseline": float(report["material_error_px"]["maximum"])
        <= BASELINE_MAXIMUM_ERROR_PX + 1e-12,
    }
    return bool(all(components.values())), components


def select_decision_branch(
    *, strict_complete: bool, practical_complete: bool, head_rescue_supported: bool
) -> str:
    """Select exactly one branch in the order frozen by the semantic lock."""

    if strict_complete:
        return "strict_numeric_pass_pending_human_visual_inspection"
    if practical_complete:
        return "practical_complete_strict_miss_preserve_both_labels"
    if head_rescue_supported:
        return "retain_query_conditioned_head_scope_smallest_residual_fix"
    return "reject_raw_cosine_head_only_patch_scope_domain_query_tracker_encoder"


def compact_information_boundary() -> dict[str, Any]:
    return {
        "selected_representation_index": FINAL_FEATURE_INDEX,
        "selected_representation_name": FINAL_FEATURE_NAME,
        "query": "frozen_frame_27_bilinear_final_feature_descriptor_per_witness",
        "score": "cosine_similarity",
        "coarse_cell": "numpy_first_row_major_hard_argmax",
        "local_readout": "temperature_1_clipped_3x3_softmax_endpoint_aligned_to_512",
        "prediction_inputs": ["frozen_truth_free_final_feature_query_score_maps"],
        "forbidden_prediction_inputs": [
            "validation_target",
            "mask",
            "previous_or_future_frame",
            "operator",
            "tracker",
            "physical_geometry",
            "manual_or_posthoc_peak_selection",
        ],
    }
