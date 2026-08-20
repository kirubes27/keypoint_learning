from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.aliked_lightglue_bridge import (
    AlikedBridgeError,
    AlikedLightGlueConfig,
    DirectMatches,
    predict_selected_identities,
    select_train_identities,
)


def seed_features(count: int = 14) -> tuple[np.ndarray, np.ndarray]:
    xy = np.stack(
        (np.arange(count, dtype=np.float64) * 12.0 + 10.0, np.full(count, 40.0)),
        axis=1,
    )
    score = np.linspace(0.9, 0.2, count, dtype=np.float64)
    return xy, score


def direct(seed_count: int, target_count: int | None = None) -> DirectMatches:
    if target_count is None:
        target_count = seed_count
    target = np.full(seed_count, -1, dtype=np.int64)
    accepted = min(seed_count, target_count)
    target[:accepted] = np.arange(accepted, dtype=np.int64)
    score = np.zeros(seed_count, dtype=np.float64)
    score[:accepted] = np.linspace(0.95, 0.5, accepted)
    return DirectMatches(target_index=target, score=score)


def test_r1_config_rejects_adaptive_or_mixed_precision_paths() -> None:
    AlikedLightGlueConfig().validate()
    with pytest.raises(AlikedBridgeError, match="adaptive"):
        AlikedLightGlueConfig(depth_confidence=0.95).validate()
    with pytest.raises(AlikedBridgeError, match="mixed precision"):
        AlikedLightGlueConfig(mixed_precision=True).validate()


def test_direct_matches_require_one_to_one_targets_and_finite_scores() -> None:
    duplicated = DirectMatches(
        target_index=np.asarray([0, 0, -1], dtype=np.int64),
        score=np.asarray([0.9, 0.8, 0.0], dtype=np.float64),
    )
    with pytest.raises(AlikedBridgeError, match="one-to-one"):
        duplicated.validate(seed_count=3, target_count=2)

    # LightGlue retains the best candidate score even if the frozen threshold
    # rejects it; target index -1, not a forced score value, defines missing.
    nonzero_missing = DirectMatches(
        target_index=np.asarray([0, -1], dtype=np.int64),
        score=np.asarray([0.9, 0.1], dtype=np.float64),
    )
    nonzero_missing.validate(seed_count=2, target_count=1)

    negative = DirectMatches(
        target_index=np.asarray([0, -1], dtype=np.int64),
        score=np.asarray([0.9, -0.1], dtype=np.float64),
    )
    with pytest.raises(AlikedBridgeError, match="non-negative"):
        negative.validate(seed_count=2, target_count=1)


def test_selection_uses_exact_train_membership_and_frozen_ranking() -> None:
    xy, detector_score = seed_features()
    train = (27, 28, 29)
    matches = {frame: direct(xy.shape[0]) for frame in train}
    counts = {frame: xy.shape[0] for frame in train}

    # Candidate 0 misses one train frame; candidate 1 keeps full coverage and
    # must rank before it even though candidate 0 has the higher detector score.
    target = matches[29].target_index.copy()
    score = matches[29].score.copy()
    target[0] = -1
    score[0] = 0.0
    matches[29] = DirectMatches(target_index=target, score=score)

    model = select_train_identities(
        xy,
        detector_score,
        matches,
        counts,
        train,
        AlikedLightGlueConfig(),
    )
    assert model.complete_seed_ranking[0] == 1
    assert 0 not in model.selected_seed_indices[:10]
    assert np.all(model.train_coverage == 1.0)

    with pytest.raises(AlikedBridgeError, match="exactly the frozen train frames"):
        select_train_identities(
            xy,
            detector_score,
            {27: matches[27], 28: matches[28]},
            {27: counts[27], 28: counts[28]},
            train,
            AlikedLightGlueConfig(),
        )


def test_calibrated_seed_separation_is_enforced_greedily() -> None:
    xy, detector_score = seed_features(15)
    xy[1] = xy[0] + np.asarray([2.0, 0.0])
    train = (27, 28)
    matches = {frame: direct(xy.shape[0]) for frame in train}
    counts = {frame: xy.shape[0] for frame in train}
    model = select_train_identities(
        xy,
        detector_score,
        matches,
        counts,
        train,
        AlikedLightGlueConfig(),
    )
    assert 0 in model.selected_seed_indices
    assert 1 not in model.selected_seed_indices
    distances = np.linalg.norm(
        model.seed_xy_px[:, None, :] - model.seed_xy_px[None, :, :], axis=-1
    )
    distances[np.diag_indices_from(distances)] = np.inf
    assert np.min(distances) >= model.config.seed_separation_px


def test_missing_match_stays_nan_and_is_never_filled() -> None:
    xy, detector_score = seed_features()
    train = (27, 28)
    matches = {frame: direct(xy.shape[0]) for frame in train}
    counts = {frame: xy.shape[0] for frame in train}
    model = select_train_identities(
        xy,
        detector_score,
        matches,
        counts,
        train,
        AlikedLightGlueConfig(),
    )

    target_xy = xy + np.asarray([5.0, 7.0])
    target = np.arange(xy.shape[0], dtype=np.int64)
    score = np.full(xy.shape[0], 0.9, dtype=np.float64)
    missing_seed = int(model.selected_seed_indices[3])
    target[missing_seed] = -1
    score[missing_seed] = 0.0
    coordinates, accepted, target_index, selected_score = predict_selected_identities(
        model,
        DirectMatches(target_index=target, score=score),
        target_xy,
    )
    assert not accepted[3]
    assert target_index[3] == -1
    assert selected_score[3] == 0.0
    assert np.isnan(coordinates[3]).all()


def test_prediction_has_no_temporal_state() -> None:
    xy, detector_score = seed_features()
    train = (27, 28)
    matches = {frame: direct(xy.shape[0]) for frame in train}
    counts = {frame: xy.shape[0] for frame in train}
    model = select_train_identities(
        xy,
        detector_score,
        matches,
        counts,
        train,
        AlikedLightGlueConfig(),
    )
    direct_matches = direct(xy.shape[0])
    first = predict_selected_identities(model, direct_matches, xy)
    _ = predict_selected_identities(model, direct_matches, xy + 100.0)
    second = predict_selected_identities(model, direct_matches, xy)
    for left, right in zip(first, second):
        np.testing.assert_equal(left, right)
