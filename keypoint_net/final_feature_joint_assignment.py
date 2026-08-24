"""Pure NumPy/SciPy contracts for Gate 4c frozen joint assignment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


IMAGE_SIZE = 512
FEATURE_SIZE = 64
EXPECTED_WITNESSES = 10
EXPECTED_WITNESS_IDS = (
    1857,
    2237,
    2241,
    12601,
    12606,
    12980,
    12993,
    13100,
    13868,
    14394,
)
EXPECTED_FRAMES = np.arange(24, dtype=np.int64)
CELL_SPACING_PX = (IMAGE_SIZE - 1) / (FEATURE_SIZE - 1)
HALF_CELL_DIAGONAL_PX = CELL_SPACING_PX / math.sqrt(2.0)
TWO_CELL_SPACING_PX = 2.0 * CELL_SPACING_PX

EXPECTED_SOURCE_RAW_RECEIPT_SHA256 = (
    "ad8e46e23fc75e1459f7f78047436167c7009ae998ed17c99d901d9b0d2d0527"
)
EXPECTED_SOURCE_SCORE_SHA256 = (
    "d5ea9d47cb3c475c7c237a5ee7cd6347af3dedded9685985f83aa504bca92fc2"
)
EXPECTED_GATE4B_RAW_SHA256 = (
    "4baccb73464ac683b25996fa53d556e33cfaecbe1edb0ad6fdb994ad8618b71f"
)
EXPECTED_SEMANTIC_LOCK_SHA256 = (
    "1c9bad238d16dd798c1c8eac813bae215f63c9acdb2ee30474d4d3d43c2ccda6"
)
FINAL_FEATURE_INDEX = 1
FINAL_FEATURE_NAME = "final_prehead_feature_map"

BASELINE_WRONG_COARSE = 49
BASELINE_WRONG_IDENTITY = 31
BASELINE_COLLAPSED_PAIR = 12
BASELINE_OFF_OBJECT = 33
BASELINE_MAXIMUM_ERROR_PX = 159.18726219197546


class JointAssignmentContractError(RuntimeError):
    """Raised when a Gate 4c semantic or provenance contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JointAssignmentContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"missing file: {resolved}")
    return {
        "absolute_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


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
    names_array = tuple(np.asarray(archive["representation_name"]).tolist())
    require(len(names_array) == 3, "representation count differs")
    require(names_array[FINAL_FEATURE_INDEX] == FINAL_FEATURE_NAME, "representation identity differs")
    frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
    witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
    require(np.array_equal(frame_index, EXPECTED_FRAMES), "held-out frame order differs")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "witness identity/order differs")
    score_maps = np.asarray(archive["score_maps"])
    require(
        score_maps.shape
        == (3, len(EXPECTED_FRAMES), EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "score-map shape differs",
    )
    final_scores = np.asarray(score_maps[FINAL_FEATURE_INDEX], dtype=np.float32)
    require(bool(np.isfinite(final_scores).all()), "final-feature score is non-finite")
    return frame_index, witness_id, final_scores


def grid_to_pixel(x: Any, y: Any) -> np.ndarray:
    return np.stack(
        [
            np.asarray(x, dtype=np.float64) * CELL_SPACING_PX,
            np.asarray(y, dtype=np.float64) * CELL_SPACING_PX,
        ],
        axis=-1,
    )


def spatial_local_maxima(field: Any) -> np.ndarray:
    """Return row-major representatives of 8-connected 3x3-max plateaus."""

    values = np.asarray(field)
    require(values.shape == (FEATURE_SIZE, FEATURE_SIZE), "local-max field shape differs")
    require(bool(np.isfinite(values).all()), "local-max field is non-finite")
    padded = np.pad(values, 1, mode="constant", constant_values=-np.inf)
    neighbors = [
        padded[dy : dy + FEATURE_SIZE, dx : dx + FEATURE_SIZE]
        for dy in range(3)
        for dx in range(3)
    ]
    local_mask = values == np.maximum.reduce(neighbors)
    visited = np.zeros_like(local_mask, dtype=bool)
    representatives: list[tuple[int, int]] = []
    for flat_index in np.flatnonzero(local_mask):
        y, x = divmod(int(flat_index), FEATURE_SIZE)
        if visited[y, x]:
            continue
        representatives.append((y, x))
        plateau_value = values[y, x]
        visited[y, x] = True
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny in range(max(0, cy - 1), min(FEATURE_SIZE, cy + 2)):
                for nx in range(max(0, cx - 1), min(FEATURE_SIZE, cx + 2)):
                    if (
                        not visited[ny, nx]
                        and local_mask[ny, nx]
                        and values[ny, nx] == plateau_value
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    require(bool(representatives), "field has no local maximum")
    return np.asarray(representatives, dtype=np.int64)


def union_candidate_modes(frame_scores: Any) -> np.ndarray:
    scores = np.asarray(frame_scores)
    require(
        scores.shape == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "frame score shape differs",
    )
    candidates: set[tuple[int, int]] = set()
    for witness in range(EXPECTED_WITNESSES):
        candidates.update(map(tuple, spatial_local_maxima(scores[witness]).tolist()))
    result = np.asarray(sorted(candidates), dtype=np.int64)
    require(result.ndim == 2 and result.shape[1] == 2, "candidate union shape differs")
    return result


def maximize_assignment(score_matrix: Any) -> tuple[np.ndarray, float]:
    """Maximize a finite rectangular score matrix with one column per row."""

    scores = np.asarray(score_matrix, dtype=np.float64)
    require(scores.ndim == 2, "assignment score must be a matrix")
    require(scores.shape[1] >= scores.shape[0] > 0, "assignment has too few candidates")
    require(not bool(np.isnan(scores).any()), "assignment score contains NaN")
    row, column = linear_sum_assignment(-scores)
    require(np.array_equal(row, np.arange(scores.shape[0])), "assignment row order differs")
    selected = scores[row, column]
    require(bool(np.isfinite(selected).all()), "assignment selected a forbidden edge")
    return column.astype(np.int64), float(selected.sum())


def local_readout_at_cells(score_maps: Any, center_y: Any, center_x: Any) -> dict[str, np.ndarray]:
    """Apply the fixed clipped temperature-1 3x3 readout around supplied cells."""

    scores = np.asarray(score_maps)
    y = np.asarray(center_y, dtype=np.int64)
    x = np.asarray(center_x, dtype=np.int64)
    require(
        scores.ndim == 4
        and scores.shape[1:] == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "local-readout score shape differs",
    )
    require(y.shape == x.shape == scores.shape[:2], "local-readout centre shape differs")
    require(bool(np.isfinite(scores).all()), "local-readout score is non-finite")
    require(bool(((x >= 0) & (x < FEATURE_SIZE) & (y >= 0) & (y < FEATURE_SIZE)).all()), "centre leaves field")
    local_px = np.empty(scores.shape[:2] + (2,), dtype=np.float64)
    assigned_score = np.empty(scores.shape[:2], dtype=np.float64)
    for frame in range(scores.shape[0]):
        for witness in range(EXPECTED_WITNESSES):
            cx = int(x[frame, witness])
            cy = int(y[frame, witness])
            x0, x1 = max(0, cx - 1), min(FEATURE_SIZE, cx + 2)
            y0, y1 = max(0, cy - 1), min(FEATURE_SIZE, cy + 2)
            window = scores[frame, witness, y0:y1, x0:x1].astype(np.float64)
            probability = np.exp(window - window.max())
            probability /= probability.sum()
            local_x = float((probability * np.arange(x0, x1)[None, :]).sum())
            local_y = float((probability * np.arange(y0, y1)[:, None]).sum())
            local_px[frame, witness] = grid_to_pixel(local_x, local_y)
            assigned_score[frame, witness] = float(scores[frame, witness, cy, cx])
    return {
        "assigned_cell_y": y,
        "assigned_cell_x": x,
        "assigned_cell_score": assigned_score,
        "assigned_hard_prediction_px": grid_to_pixel(x, y),
        "assigned_local_3x3_prediction_px": local_px,
    }


def decode_joint_assignments(score_maps: Any) -> dict[str, np.ndarray]:
    """Decode every frame independently using the frozen candidate/assignment rule."""

    scores = np.asarray(score_maps)
    require(
        scores.ndim == 4
        and scores.shape[1:] == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "joint score shape differs",
    )
    require(bool(np.isfinite(scores).all()), "joint score is non-finite")
    frame_candidates = [union_candidate_modes(scores[frame]) for frame in range(scores.shape[0])]
    counts = np.asarray([len(candidates) for candidates in frame_candidates], dtype=np.int64)
    require(bool((counts >= EXPECTED_WITNESSES).all()), "a frame has fewer than ten modes")
    width = int(counts.max())
    candidate_y = np.full((scores.shape[0], width), -1, dtype=np.int64)
    candidate_x = np.full_like(candidate_y, -1)
    assignment_index = np.empty((scores.shape[0], EXPECTED_WITNESSES), dtype=np.int64)
    assigned_y = np.empty_like(assignment_index)
    assigned_x = np.empty_like(assignment_index)
    assignment_total = np.empty(scores.shape[0], dtype=np.float64)
    for frame, candidates in enumerate(frame_candidates):
        count = len(candidates)
        candidate_y[frame, :count] = candidates[:, 0]
        candidate_x[frame, :count] = candidates[:, 1]
        matrix = scores[frame][:, candidates[:, 0], candidates[:, 1]]
        assignment, total = maximize_assignment(matrix)
        assignment_index[frame] = assignment
        assigned_y[frame] = candidates[assignment, 0]
        assigned_x[frame] = candidates[assignment, 1]
        assignment_total[frame] = total
    readout = local_readout_at_cells(scores, assigned_y, assigned_x)
    return {
        "candidate_mode_count": counts,
        "candidate_mode_y": candidate_y,
        "candidate_mode_x": candidate_x,
        "candidate_mode_valid": np.arange(width)[None, :] < counts[:, None],
        "assignment_candidate_index": assignment_index,
        "assignment_total_score": assignment_total,
        **readout,
    }


def _best_score_with_forbidden_edges(scores: np.ndarray, edges: list[tuple[int, int]]) -> float:
    alternatives: list[float] = []
    for row, column in edges:
        constrained = scores.copy()
        constrained[row, column] = -np.inf
        _, total = maximize_assignment(constrained)
        alternatives.append(total)
    return max(alternatives)


def square_assignment_diagnostics(score_matrix: Any) -> dict[str, Any]:
    """Return optimizer uniqueness and the signed correct-permutation margin."""

    scores = np.asarray(score_matrix, dtype=np.float64)
    require(
        scores.shape == (EXPECTED_WITNESSES, EXPECTED_WITNESSES),
        "true-site score matrix shape differs",
    )
    best_assignment, best_score = maximize_assignment(scores)
    best_edges = list(enumerate(best_assignment.tolist()))
    optimizer_second = _best_score_with_forbidden_edges(scores, best_edges)
    identity = np.arange(EXPECTED_WITNESSES, dtype=np.int64)
    correct_score = float(scores[identity, identity].sum())
    best_competing = _best_score_with_forbidden_edges(
        scores, list(zip(identity.tolist(), identity.tolist(), strict=True))
    )
    return {
        "best_assignment": best_assignment,
        "best_assignment_score": best_score,
        "optimizer_second_best_score": optimizer_second,
        "optimizer_best_minus_second_margin": best_score - optimizer_second,
        "correct_assignment_score": correct_score,
        "best_competing_assignment_score": best_competing,
        "signed_correct_assignment_margin": correct_score - best_competing,
        "identity_correct": best_assignment == identity,
    }


def bilinear_sample_all_sites(score_maps: Any, target_px: Any) -> np.ndarray:
    """Sample each query field at every certified continuous material site."""

    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    require(
        scores.ndim == 4
        and scores.shape[1:] == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "true-site score shape differs",
    )
    require(targets.shape == scores.shape[:2] + (2,), "true-site target shape differs")
    require(bool(np.isfinite(scores).all() and np.isfinite(targets).all()), "true-site inputs invalid")
    require(bool(((targets >= 0.0) & (targets <= IMAGE_SIZE - 1)).all()), "true site leaves image")
    result = np.empty(scores.shape[:2] + (EXPECTED_WITNESSES,), dtype=np.float64)
    for frame in range(scores.shape[0]):
        for query in range(EXPECTED_WITNESSES):
            field = scores[frame, query]
            for site in range(EXPECTED_WITNESSES):
                gx = targets[frame, site, 0] / (IMAGE_SIZE - 1) * (FEATURE_SIZE - 1)
                gy = targets[frame, site, 1] / (IMAGE_SIZE - 1) * (FEATURE_SIZE - 1)
                x0, y0 = int(np.floor(gx)), int(np.floor(gy))
                x1, y1 = min(x0 + 1, FEATURE_SIZE - 1), min(y0 + 1, FEATURE_SIZE - 1)
                wx, wy = gx - x0, gy - y0
                result[frame, query, site] = (
                    field[y0, x0] * (1.0 - wx) * (1.0 - wy)
                    + field[y0, x1] * wx * (1.0 - wy)
                    + field[y1, x0] * (1.0 - wx) * wy
                    + field[y1, x1] * wx * wy
                )
    return result


def nearest_r64_cells(target_px: Any) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(target_px, dtype=np.float64)
    require(targets.shape[-1:] == (2,), "target cells require xy points")
    cells = np.rint(targets / (IMAGE_SIZE - 1) * (FEATURE_SIZE - 1)).astype(np.int64)
    cells = np.clip(cells, 0, FEATURE_SIZE - 1)
    return cells[..., 0], cells[..., 1]


def shared_true_site_cells(target_px: Any) -> tuple[np.ndarray, np.ndarray]:
    x, y = nearest_r64_cells(target_px)
    require(x.ndim == 2 and x.shape[1] == EXPECTED_WITNESSES, "shared-cell target shape differs")
    pair = np.zeros((x.shape[0], EXPECTED_WITNESSES, EXPECTED_WITNESSES), dtype=bool)
    for left in range(EXPECTED_WITNESSES):
        for right in range(left + 1, EXPECTED_WITNESSES):
            pair[:, left, right] = (x[:, left] == x[:, right]) & (y[:, left] == y[:, right])
    per_site = pair.any(axis=1) | pair.any(axis=2)
    return pair, per_site


def _summary(values: Any) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "summary values invalid")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "q90": float(np.quantile(vector, 0.90)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def evaluate_predictions(
    prediction_px: Any, target_px: Any, masks: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pure-NumPy replay of the frozen material/identity/grounding contract."""

    prediction = np.asarray(prediction_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=bool)
    require(prediction.shape == target.shape, "prediction/target shape differs")
    require(prediction.ndim == 3 and prediction.shape[1:] == (EXPECTED_WITNESSES, 2), "point shape differs")
    require(mask_array.shape == (prediction.shape[0], IMAGE_SIZE, IMAGE_SIZE), "mask shape differs")
    require(bool(np.isfinite(prediction).all() and np.isfinite(target).all()), "point input is non-finite")
    material_error = np.linalg.norm(prediction - target, axis=-1)
    rounded = np.rint(prediction).astype(np.int64)
    in_image = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < IMAGE_SIZE)
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < IMAGE_SIZE)
    )
    clipped = np.clip(rounded, 0, IMAGE_SIZE - 1)
    frames = np.arange(prediction.shape[0])[:, None]
    on_object = in_image & mask_array[frames, clipped[..., 1], clipped[..., 0]]
    target_distance = np.linalg.norm(prediction[:, :, None] - target[:, None], axis=-1)
    assigned_identity = np.argmin(target_distance, axis=-1)
    identity_correct = assigned_identity == np.arange(EXPECTED_WITNESSES)[None]
    prediction_pair = np.linalg.norm(prediction[:, :, None] - prediction[:, None], axis=-1)
    target_pair = np.linalg.norm(target[:, :, None] - target[:, None], axis=-1)
    pair_mask = np.triu(np.ones((EXPECTED_WITNESSES, EXPECTED_WITNESSES), dtype=bool), k=1)
    pair_ratio = prediction_pair[:, pair_mask] / target_pair[:, pair_mask]
    distinct_pair = pair_ratio >= 0.5
    within_half = material_error <= HALF_CELL_DIAGONAL_PX + 1e-12
    violations = {
        "outside_half_cell_count": int(within_half.size - within_half.sum()),
        "off_object_count": int(on_object.size - on_object.sum()),
        "wrong_identity_count": int(identity_correct.size - identity_correct.sum()),
        "collapsed_pair_count": int(distinct_pair.size - distinct_pair.sum()),
    }
    per_witness = []
    for channel, witness_id in enumerate(EXPECTED_WITNESS_IDS):
        per_witness.append(
            {
                "channel": channel,
                "witness_id": witness_id,
                "material_error_px": _summary(material_error[:, channel]),
                "within_half_cell_rate": float(within_half[:, channel].mean()),
                "on_object_rate": float(on_object[:, channel].mean()),
                "identity_assignment_rate": float(identity_correct[:, channel].mean()),
            }
        )
    report = {
        "strict_capability_pass": bool(
            within_half.all() and on_object.all() and identity_correct.all() and distinct_pair.all()
        ),
        "frame_count": int(prediction.shape[0]),
        "witness_count": EXPECTED_WITNESSES,
        "cell_spacing_px": CELL_SPACING_PX,
        "half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
        "violations": violations,
        "material_error_px": _summary(material_error),
        "within_half_cell_rate": float(within_half.mean()),
        "on_object_rate": float(on_object.mean()),
        "identity_assignment_rate": float(identity_correct.mean()),
        "minimum_predicted_pair_distance_px": float(prediction_pair[:, pair_mask].min()),
        "minimum_predicted_to_physical_pair_ratio": float(pair_ratio.min()),
        "per_witness": per_witness,
        "statistical_scope": {
            "inference": "descriptive_only",
            "sample_unit": "fixed_witness_event_over_one_24_frame_correlated_heldout_wedge",
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    derived = {
        "material_error_px": material_error,
        "within_half_cell": within_half,
        "on_object": on_object,
        "identity_correct": identity_correct,
        "assigned_identity": assigned_identity,
        "prediction_pair_distance_px": prediction_pair,
        "target_pair_distance_px": target_pair,
        "distinct_pair": distinct_pair,
    }
    return report, derived


def target_rank_and_category(
    score_maps: Any,
    center_y: Any,
    center_x: Any,
    target_px: Any,
    within_half_cell: Any,
) -> dict[str, np.ndarray]:
    scores = np.asarray(score_maps, dtype=np.float64)
    targets = np.asarray(target_px, dtype=np.float64)
    cy = np.asarray(center_y, dtype=np.int64)
    cx = np.asarray(center_x, dtype=np.int64)
    within = np.asarray(within_half_cell, dtype=bool)
    tx, ty = nearest_r64_cells(targets)
    flat = scores.reshape(scores.shape[0], EXPECTED_WITNESSES, -1)
    target_index = ty * FEATURE_SIZE + tx
    target_score = np.take_along_axis(flat, target_index[..., None], axis=-1)[..., 0]
    rank = 1 + (flat > target_score[..., None]).sum(axis=-1)
    distance = np.maximum(np.abs(tx - cx), np.abs(ty - cy))
    inside = distance <= 1
    failed = ~within
    border = (cx == 0) | (cx == FEATURE_SIZE - 1) | (cy == 0) | (cy == FEATURE_SIZE - 1)
    category = np.zeros(within.shape, dtype=np.int8)
    category[failed & border] = 1
    unresolved = failed & (category == 0)
    category[unresolved & ~inside & (rank <= 10)] = 2
    category[unresolved & ~inside & (rank > 10)] = 3
    category[unresolved & inside] = 4
    require(bool((category[failed] > 0).all() and (category[within] == 0).all()), "category partition differs")
    return {
        "target_cell_x": tx,
        "target_cell_y": ty,
        "target_nearest_cell_rank": rank.astype(np.int64),
        "target_to_assigned_chebyshev_cells": distance.astype(np.int64),
        "target_cell_inside_local_window": inside,
        "localization_category_code": category,
    }


def compact_information_boundary() -> dict[str, Any]:
    return {
        "selected_representation_index": FINAL_FEATURE_INDEX,
        "selected_representation_name": FINAL_FEATURE_NAME,
        "query": "frozen_frame_27_bilinear_final_feature_descriptor_per_witness",
        "score": "cosine_similarity",
        "candidate_modes": "all_3x3_local_maxima_with_8_connected_equal_plateau_row_major_collapse",
        "candidate_union": "unique_cells_over_all_ten_query_fields_sorted_row_major",
        "assignment": "scipy_linear_sum_assignment_maximize_total_score_one_cell_per_query",
        "local_readout": "temperature_1_clipped_3x3_softmax_endpoint_aligned_to_512",
        "prediction_inputs": ["frozen_truth_free_final_feature_query_score_maps"],
        "forbidden_prediction_inputs": [
            "validation_target",
            "mask",
            "rgb",
            "previous_or_future_frame",
            "operator",
            "tracker",
            "physical_geometry",
            "top_k_or_posthoc_peak_selection",
        ],
    }
