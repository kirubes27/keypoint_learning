"""Pure measurements for frozen encoder-versus-head localization."""

from __future__ import annotations

from typing import Any

import numpy as np


IMAGE_SIZE = 512
FIELD_SIZE = 64
EXPECTED_WITNESSES = 10
REPRESENTATION_NAMES = (
    "penultimate_encoder_block",
    "final_prehead_feature_map",
    "heatmap_head_logits",
)
SEPARATED_SPATIAL_RADIUS_CELLS = 4.0
STAGE_STRONG_MAX_RANK = 3
STAGE_AMBIGUOUS_MAX_RANK = 10


class EncoderHeadLocalizationError(ValueError):
    """Raised when localization inputs or frozen semantics are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EncoderHeadLocalizationError(message)


def pixel_to_normalized(points_px: Any) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    _require(points.shape[-1:] == (2,), "points must end in (x,y)")
    _require(bool(np.isfinite(points).all()), "points contain non-finite values")
    _require(bool((points >= 0.0).all() and (points <= IMAGE_SIZE - 1).all()), "points leave image")
    return points / (IMAGE_SIZE - 1) * 2.0 - 1.0


def nearest_target_cells(points_px: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_px, dtype=np.float64)
    _require(points.shape[-1:] == (2,), "target points must end in (x,y)")
    _require(bool(np.isfinite(points).all()), "target points contain non-finite values")
    cell = np.rint(points / (IMAGE_SIZE - 1) * (FIELD_SIZE - 1)).astype(np.int64)
    cell = np.clip(cell, 0, FIELD_SIZE - 1)
    return cell[..., 0], cell[..., 1]


def target_cell_ranks(score_maps: Any, target_px: Any) -> dict[str, np.ndarray]:
    """Rank each corresponding material cell among all 4,096 spatial cells."""

    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    _require(
        scores.ndim == 5
        and scores.shape[0] == len(REPRESENTATION_NAMES)
        and scores.shape[2:] == (EXPECTED_WITNESSES, FIELD_SIZE, FIELD_SIZE),
        "score maps must have shape (3,F,10,64,64)",
    )
    _require(targets.shape == (scores.shape[1], EXPECTED_WITNESSES, 2), "target shape differs")
    _require(bool(np.isfinite(scores).all()), "score maps contain non-finite values")
    x, y = nearest_target_cells(targets)
    frame = np.arange(scores.shape[1])[None, :, None]
    witness = np.arange(EXPECTED_WITNESSES)[None, None, :]
    level = np.arange(scores.shape[0])[:, None, None]
    target_score = scores[level, frame, witness, y[None], x[None]]
    flat = scores.reshape(scores.shape[0], scores.shape[1], EXPECTED_WITNESSES, -1)
    rank = 1 + (flat > target_score[..., None]).sum(axis=-1)
    return {
        "target_cell_x": x,
        "target_cell_y": y,
        "target_cell_score": target_score,
        "target_cell_rank": rank.astype(np.int64),
    }


def _bilinear_sample_one_map(field: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Sample maps ``(...,H,W)`` at matching-prefix points ``(...,2)``."""

    values = np.asarray(field, dtype=np.float64)
    points = np.asarray(points_px, dtype=np.float64)
    _require(values.shape[:-2] == points.shape[:-1], "map and point prefixes differ")
    _require(values.shape[-2:] == (FIELD_SIZE, FIELD_SIZE), "map size differs")
    _require(bool(np.isfinite(values).all() and np.isfinite(points).all()), "bilinear inputs invalid")
    x = points[..., 0] / (IMAGE_SIZE - 1) * (FIELD_SIZE - 1)
    y = points[..., 1] / (IMAGE_SIZE - 1) * (FIELD_SIZE - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, FIELD_SIZE - 1)
    y1 = np.clip(y0 + 1, 0, FIELD_SIZE - 1)
    wx = x - x0
    wy = y - y0
    prefix = np.indices(values.shape[:-2], sparse=True)
    v00 = values[(*prefix, y0, x0)]
    v01 = values[(*prefix, y0, x1)]
    v10 = values[(*prefix, y1, x0)]
    v11 = values[(*prefix, y1, x1)]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v01 * wx * (1.0 - wy)
        + v10 * (1.0 - wx) * wy
        + v11 * wx * wy
    )


def sample_corresponding_targets(score_maps: Any, target_px: Any) -> np.ndarray:
    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    _require(scores.ndim == 5, "score maps must have five dimensions")
    expanded = np.broadcast_to(targets[None], scores.shape[:3] + (2,))
    return _bilinear_sample_one_map(scores, expanded)


def sample_all_material_sites(score_maps: Any, target_px: Any) -> np.ndarray:
    """Return score of every query channel at every physical witness site."""

    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    _require(scores.ndim == 5, "score maps must have five dimensions")
    _require(targets.shape == (scores.shape[1], EXPECTED_WITNESSES, 2), "target shape differs")
    sampled = np.empty(scores.shape[:3] + (EXPECTED_WITNESSES,), dtype=np.float64)
    for site in range(EXPECTED_WITNESSES):
        points = np.broadcast_to(targets[None, :, None, site], scores.shape[:3] + (2,))
        sampled[..., site] = _bilinear_sample_one_map(scores, points)
    return sampled


def sample_at_cells(score_maps: Any, x_cell: Any, y_cell: Any) -> np.ndarray:
    scores = np.asarray(score_maps, dtype=np.float64)
    x = np.asarray(x_cell, dtype=np.int64)
    y = np.asarray(y_cell, dtype=np.int64)
    _require(scores.ndim == 5, "score maps must have five dimensions")
    _require(x.shape == y.shape == scores.shape[1:3], "cell shape differs")
    _require(bool((x >= 0).all() and (x < FIELD_SIZE).all()), "x cell leaves field")
    _require(bool((y >= 0).all() and (y < FIELD_SIZE).all()), "y cell leaves field")
    level = np.arange(scores.shape[0])[:, None, None]
    frame = np.arange(scores.shape[1])[None, :, None]
    witness = np.arange(scores.shape[2])[None, None, :]
    return scores[level, frame, witness, y[None], x[None]]


def separated_spatial_competitor(
    score_maps: Any,
    target_x: Any,
    target_y: Any,
    *,
    radius_cells: float = SEPARATED_SPATIAL_RADIUS_CELLS,
) -> dict[str, np.ndarray]:
    scores = np.asarray(score_maps, dtype=np.float64)
    x = np.asarray(target_x, dtype=np.int64)
    y = np.asarray(target_y, dtype=np.int64)
    _require(scores.ndim == 5, "score maps must have five dimensions")
    _require(x.shape == y.shape == scores.shape[1:3], "target cell shape differs")
    _require(radius_cells >= 0.0, "spatial radius must be non-negative")
    flat = scores.reshape(scores.shape[0], scores.shape[1], scores.shape[2], -1)
    yy, xx = np.meshgrid(np.arange(FIELD_SIZE), np.arange(FIELD_SIZE), indexing="ij")
    best_score = np.empty(scores.shape[:3], dtype=np.float64)
    best_x = np.empty(scores.shape[:3], dtype=np.int64)
    best_y = np.empty(scores.shape[:3], dtype=np.int64)
    for level in range(scores.shape[0]):
        for frame in range(scores.shape[1]):
            for witness in range(scores.shape[2]):
                keep = np.hypot(xx - x[frame, witness], yy - y[frame, witness]) > radius_cells
                candidate = np.where(keep.reshape(-1), flat[level, frame, witness], -np.inf)
                index = int(np.argmax(candidate))
                _require(bool(np.isfinite(candidate[index])), "no separated spatial competitor")
                best_score[level, frame, witness] = candidate[index]
                best_y[level, frame, witness], best_x[level, frame, witness] = divmod(index, FIELD_SIZE)
    return {"score": best_score, "x_cell": best_x, "y_cell": best_y}


def explicit_competitor_margins(
    score_maps: Any,
    target_px: Any,
    head_hard_x: Any,
    head_hard_y: Any,
) -> dict[str, np.ndarray]:
    """Compare the true site with identity, wrong-mode, and spatial distractors."""

    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    hard_x = np.asarray(head_hard_x, dtype=np.int64)
    hard_y = np.asarray(head_hard_y, dtype=np.int64)
    ranks = target_cell_ranks(scores, targets)
    target_x = ranks["target_cell_x"]
    target_y = ranks["target_cell_y"]
    correct = sample_corresponding_targets(scores, targets)

    material = sample_all_material_sites(scores, targets)
    diagonal = np.arange(EXPECTED_WITNESSES)
    material[..., diagonal, diagonal] = -np.inf
    identity_index = np.argmax(material, axis=-1)
    identity_score = np.max(material, axis=-1)

    wrong_coarse = np.maximum(np.abs(hard_x - target_x), np.abs(hard_y - target_y)) > 1
    hard_score = sample_at_cells(scores, hard_x, hard_y)
    hard_score = np.where(wrong_coarse[None], hard_score, -np.inf)

    spatial = separated_spatial_competitor(scores, target_x, target_y)
    candidates = np.stack((identity_score, hard_score, spatial["score"]), axis=-1)
    source_index = np.argmax(candidates, axis=-1)
    competitor = np.max(candidates, axis=-1)
    _require(bool(np.isfinite(competitor).all()), "competitor score is non-finite")
    return {
        **ranks,
        "continuous_target_score": correct,
        "identity_competitor_score": identity_score,
        "identity_competitor_witness": identity_index.astype(np.int64),
        "wrong_coarse_competitor_score": hard_score,
        "separated_spatial_competitor_score": spatial["score"],
        "separated_spatial_competitor_x": spatial["x_cell"],
        "separated_spatial_competitor_y": spatial["y_cell"],
        "maximum_competitor_score": competitor,
        "maximum_competitor_source_code": source_index.astype(np.int8),
        "target_minus_competitor_margin": correct - competitor,
        "head_wrong_coarse_event": wrong_coarse,
    }


def stage_class(rank: Any) -> np.ndarray:
    values = np.asarray(rank, dtype=np.int64)
    _require(bool((values >= 1).all() and (values <= FIELD_SIZE * FIELD_SIZE).all()), "rank invalid")
    return np.where(
        values <= STAGE_STRONG_MAX_RANK,
        0,
        np.where(values <= STAGE_AMBIGUOUS_MAX_RANK, 1, 2),
    ).astype(np.int8)


def fixed_transition_labels(rank: Any, wrong_coarse: Any) -> np.ndarray:
    """Apply only the preregistered unambiguous transition interpretations."""

    values = np.asarray(rank, dtype=np.int64)
    wrong = np.asarray(wrong_coarse, dtype=bool)
    _require(values.shape[0] == 3 and values.shape[1:] == wrong.shape, "transition shapes differ")
    classes = stage_class(values)
    penultimate, final, logits = classes
    labels = np.full(wrong.shape, "ambiguous_or_nonmonotonic", dtype="<U40")
    labels[penultimate == 2] = "badly_ranked_by_penultimate"
    labels[(penultimate == 0) & (final == 2)] = "lost_in_final_encoder_block"
    labels[(penultimate == 0) & (final == 0) & (logits == 2)] = "lost_in_heatmap_head"
    labels[wrong & (penultimate == 0) & (final == 0) & (logits == 0)] = "selection_or_local_readout"
    return labels


def summarize_vector(values: Any) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(vector.size > 0 and bool(np.isfinite(vector).all()), "summary vector invalid")
    return {
        "n": int(vector.size),
        "median": float(np.median(vector)),
        "mean": float(vector.mean()),
        "q90": float(np.quantile(vector, 0.9)),
        "maximum": float(vector.max()),
        "minimum": float(vector.min()),
    }


__all__ = [
    "EncoderHeadLocalizationError",
    "EXPECTED_WITNESSES",
    "FIELD_SIZE",
    "IMAGE_SIZE",
    "REPRESENTATION_NAMES",
    "SEPARATED_SPATIAL_RADIUS_CELLS",
    "explicit_competitor_margins",
    "fixed_transition_labels",
    "nearest_target_cells",
    "pixel_to_normalized",
    "sample_all_material_sites",
    "sample_corresponding_targets",
    "separated_spatial_competitor",
    "stage_class",
    "summarize_vector",
    "target_cell_ranks",
]
