"""Pure contracts for a frozen, stateless ALIKED + LightGlue bridge.

This module deliberately contains no model import.  It validates direct
seed-to-frame matches, selects ten identities using train evidence only, and
maps those identities to actual target detections without interpolation or
temporal state.  The external model runtime lives in the runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np


class AlikedBridgeError(ValueError):
    """Raised when the frozen bridge contract is violated."""


@dataclass(frozen=True)
class AlikedLightGlueConfig:
    """R1 configuration frozen before pretrained hammer inference."""

    n_identities: int = 10
    seed_frame_index: int = 27
    seed_separation_px: float = 8.11111111111111
    model_name: str = "aliked-n16"
    max_num_keypoints: int = 2048
    detection_threshold: float = 0.2
    nms_radius: int = 2
    n_layers: int = 9
    depth_confidence: float = -1.0
    width_confidence: float = -1.0
    filter_threshold: float = 0.1
    flash: bool = False
    mixed_precision: bool = False

    def validate(self) -> None:
        if self.n_identities <= 0:
            raise AlikedBridgeError("n_identities must be positive")
        if self.seed_frame_index < 0:
            raise AlikedBridgeError("seed_frame_index must be non-negative")
        if not np.isfinite(self.seed_separation_px) or self.seed_separation_px <= 0:
            raise AlikedBridgeError("seed_separation_px must be finite and positive")
        if self.model_name != "aliked-n16":
            raise AlikedBridgeError("R1 requires aliked-n16")
        if self.max_num_keypoints != 2048:
            raise AlikedBridgeError("R1 requires max_num_keypoints=2048")
        if self.detection_threshold != 0.2 or self.nms_radius != 2:
            raise AlikedBridgeError("R1 detector settings differ from the lock")
        if self.n_layers != 9:
            raise AlikedBridgeError("R1 requires all nine LightGlue layers")
        if self.depth_confidence != -1 or self.width_confidence != -1:
            raise AlikedBridgeError("adaptive depth and width must be disabled")
        if self.filter_threshold != 0.1:
            raise AlikedBridgeError("R1 requires the pretrained default match threshold")
        if self.flash or self.mixed_precision:
            raise AlikedBridgeError("R1 CPU path forbids flash and mixed precision")


@dataclass(frozen=True)
class DirectMatches:
    """One direct LightGlue result from every seed keypoint to one target frame."""

    target_index: np.ndarray
    score: np.ndarray

    def validate(self, *, seed_count: int, target_count: int) -> None:
        target = np.asarray(self.target_index)
        score = np.asarray(self.score)
        if target.shape != (seed_count,) or score.shape != (seed_count,):
            raise AlikedBridgeError("direct match fields must have one row per seed keypoint")
        if not np.issubdtype(target.dtype, np.integer):
            raise AlikedBridgeError("target_index must be integer")
        if np.any(target < -1) or np.any(target >= target_count):
            raise AlikedBridgeError("target_index is outside the target detection range")
        if not np.isfinite(score).all() or np.any(score < 0):
            raise AlikedBridgeError("match scores must be finite and non-negative")
        accepted = target >= 0
        used = target[accepted]
        if np.unique(used).size != used.size:
            raise AlikedBridgeError("direct LightGlue matches are not one-to-one")


@dataclass(frozen=True)
class FrozenAlikedIdentities:
    """Ten seed identities selected using the frozen train partition only."""

    config: AlikedLightGlueConfig
    train_frame_indices: tuple[int, ...]
    selected_seed_indices: np.ndarray
    seed_xy_px: np.ndarray
    seed_detector_score: np.ndarray
    train_coverage: np.ndarray
    train_median_match_score: np.ndarray
    complete_seed_ranking: np.ndarray

    def validate(self, *, total_seed_count: int) -> None:
        self.config.validate()
        n = self.config.n_identities
        if not self.train_frame_indices:
            raise AlikedBridgeError("train_frame_indices is empty")
        if tuple(sorted(set(self.train_frame_indices))) != self.train_frame_indices:
            raise AlikedBridgeError("train_frame_indices must be sorted and unique")
        if self.config.seed_frame_index not in self.train_frame_indices:
            raise AlikedBridgeError("seed frame is outside the train partition")
        for name, value, shape in (
            ("selected_seed_indices", self.selected_seed_indices, (n,)),
            ("seed_xy_px", self.seed_xy_px, (n, 2)),
            ("seed_detector_score", self.seed_detector_score, (n,)),
            ("train_coverage", self.train_coverage, (n,)),
            ("train_median_match_score", self.train_median_match_score, (n,)),
            ("complete_seed_ranking", self.complete_seed_ranking, (total_seed_count,)),
        ):
            array = np.asarray(value)
            if array.shape != shape:
                raise AlikedBridgeError(f"{name} has the wrong shape")
            if name not in {"selected_seed_indices", "complete_seed_ranking"} and not np.isfinite(array).all():
                raise AlikedBridgeError(f"{name} contains non-finite values")
        selected = np.asarray(self.selected_seed_indices, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= total_seed_count):
            raise AlikedBridgeError("selected seed index is out of range")
        if np.unique(selected).size != n:
            raise AlikedBridgeError("selected seed identities are not unique")
        ranking = np.asarray(self.complete_seed_ranking, dtype=np.int64)
        if not np.array_equal(np.sort(ranking), np.arange(total_seed_count)):
            raise AlikedBridgeError("complete_seed_ranking is not a permutation")
        distances = np.linalg.norm(
            self.seed_xy_px[:, None, :] - self.seed_xy_px[None, :, :], axis=-1
        )
        distances[np.diag_indices_from(distances)] = np.inf
        if float(np.min(distances)) < self.config.seed_separation_px:
            raise AlikedBridgeError("selected seed identities violate calibrated separation")


def config_as_dict(config: AlikedLightGlueConfig) -> dict[str, object]:
    """Return a stable JSON-compatible configuration mapping."""

    config.validate()
    return asdict(config)


def _validate_seed_features(seed_xy_px: np.ndarray, seed_score: np.ndarray) -> None:
    xy = np.asarray(seed_xy_px)
    score = np.asarray(seed_score)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AlikedBridgeError("seed_xy_px must have shape (seed keypoints, 2)")
    if score.shape != (xy.shape[0],):
        raise AlikedBridgeError("seed_score must have one value per seed keypoint")
    if xy.shape[0] == 0:
        raise AlikedBridgeError("seed frame has no ALIKED detections")
    if not np.isfinite(xy).all() or not np.isfinite(score).all():
        raise AlikedBridgeError("seed features contain non-finite values")
    if np.any(score < 0):
        raise AlikedBridgeError("seed detector scores must be non-negative")


def select_train_identities(
    seed_xy_px: np.ndarray,
    seed_score: np.ndarray,
    matches_by_frame: Mapping[int, DirectMatches],
    target_counts: Mapping[int, int],
    train_frame_indices: Iterable[int],
    config: AlikedLightGlueConfig,
) -> FrozenAlikedIdentities:
    """Select ten separated seed keypoints using direct train matches only."""

    config.validate()
    _validate_seed_features(seed_xy_px, seed_score)
    xy = np.asarray(seed_xy_px, dtype=np.float64)
    detector_score = np.asarray(seed_score, dtype=np.float64)
    seed_count = xy.shape[0]
    train = tuple(sorted(set(int(frame) for frame in train_frame_indices)))
    if not train or any(frame < 0 for frame in train):
        raise AlikedBridgeError("train_frame_indices must be non-empty and non-negative")
    if config.seed_frame_index not in train:
        raise AlikedBridgeError("seed frame must belong to train")
    if set(matches_by_frame) != set(train) or set(target_counts) != set(train):
        raise AlikedBridgeError("selection must receive exactly the frozen train frames")

    accepted = np.zeros((len(train), seed_count), dtype=bool)
    scores = np.zeros((len(train), seed_count), dtype=np.float64)
    for row, frame in enumerate(train):
        count = int(target_counts[frame])
        if count < 0:
            raise AlikedBridgeError("target detection count cannot be negative")
        direct = matches_by_frame[frame]
        direct.validate(seed_count=seed_count, target_count=count)
        target = np.asarray(direct.target_index, dtype=np.int64)
        accepted[row] = target >= 0
        scores[row] = np.asarray(direct.score, dtype=np.float64)

    coverage = np.mean(accepted, axis=0)
    median_score = np.zeros(seed_count, dtype=np.float64)
    for index in range(seed_count):
        values = scores[:, index][accepted[:, index]]
        median_score[index] = float(np.median(values)) if values.size else 0.0

    ranking = np.asarray(
        sorted(
            range(seed_count),
            key=lambda index: (
                -float(coverage[index]),
                -float(median_score[index]),
                -float(detector_score[index]),
                int(index),
            ),
        ),
        dtype=np.int64,
    )
    selected: list[int] = []
    for index in ranking.tolist():
        if all(
            float(np.linalg.norm(xy[index] - xy[prior])) >= config.seed_separation_px
            for prior in selected
        ):
            selected.append(index)
        if len(selected) == config.n_identities:
            break
    if len(selected) != config.n_identities:
        raise AlikedBridgeError(
            f"only {len(selected)} separated ALIKED seed candidates; need {config.n_identities}"
        )

    chosen = np.asarray(selected, dtype=np.int64)
    result = FrozenAlikedIdentities(
        config=config,
        train_frame_indices=train,
        selected_seed_indices=chosen,
        seed_xy_px=xy[chosen].copy(),
        seed_detector_score=detector_score[chosen].copy(),
        train_coverage=coverage[chosen].copy(),
        train_median_match_score=median_score[chosen].copy(),
        complete_seed_ranking=ranking,
    )
    result.validate(total_seed_count=seed_count)
    return result


def predict_selected_identities(
    model: FrozenAlikedIdentities,
    direct_matches: DirectMatches,
    target_xy_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map frozen seed identities to actual target detections in one frame."""

    target_xy = np.asarray(target_xy_px, dtype=np.float64)
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise AlikedBridgeError("target_xy_px must have shape (target detections, 2)")
    if not np.isfinite(target_xy).all():
        raise AlikedBridgeError("target coordinates contain non-finite values")
    seed_count = model.complete_seed_ranking.size
    model.validate(total_seed_count=seed_count)
    direct_matches.validate(seed_count=seed_count, target_count=target_xy.shape[0])

    selected = np.asarray(model.selected_seed_indices, dtype=np.int64)
    target_index = np.asarray(direct_matches.target_index, dtype=np.int64)[selected].copy()
    score = np.asarray(direct_matches.score, dtype=np.float64)[selected].copy()
    accepted = target_index >= 0
    coordinates = np.full((model.config.n_identities, 2), np.nan, dtype=np.float64)
    coordinates[accepted] = target_xy[target_index[accepted]]
    if np.unique(target_index[accepted]).size != int(np.count_nonzero(accepted)):
        raise AlikedBridgeError("selected identity prediction is not one-to-one")
    if not np.array_equal(np.isfinite(coordinates).all(axis=1), accepted):
        raise AlikedBridgeError("missing predictions were not represented as NaN")
    return coordinates, accepted, target_index, score
