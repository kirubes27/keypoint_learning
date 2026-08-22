"""Frozen image-only local readout used by the certified-witness gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
    normalized_to_pixel,
    require,
)


LOCALIZATION_CATEGORY_NAMES = {
    0: "not_a_localization_failure",
    1: "border_window_truncation",
    2: "wrong_coarse_mode_target_top10",
    3: "wrong_coarse_mode_target_below_top10",
    4: "local_offset_failure",
}


def grid_to_pixel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map endpoint-aligned r64 grid coordinates to 512-pixel coordinates."""

    normalized = np.stack(
        [
            -1.0 + 2.0 * np.asarray(x, dtype=np.float64) / (FEATURE_SIZE - 1),
            -1.0 + 2.0 * np.asarray(y, dtype=np.float64) / (FEATURE_SIZE - 1),
        ],
        axis=-1,
    )
    return normalized_to_pixel(normalized)


def readout_arrays(logits: np.ndarray, target_px: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute global diagnostics and the fixed hard-centred local-3x3 readout.

    The centre is the current-frame row-major hard argmax. The local softmax is
    temperature 1, clipped at image-grid borders, and renormalized only inside
    the selected window. No target or temporal information enters prediction.
    """

    logit_array = np.asarray(logits)
    require(
        logit_array.ndim == 4
        and logit_array.shape[1:] == (
            EXPECTED_WITNESSES,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ),
        "logit array shape differs",
    )
    frame_count = int(logit_array.shape[0])
    flat = logit_array.reshape(frame_count, EXPECTED_WITNESSES, -1).astype(np.float64)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    order = np.argpartition(probability, kth=-2, axis=-1)[..., -2:]
    top_two = np.take_along_axis(probability, order, axis=-1)
    top_two.sort(axis=-1)

    hard_index = np.argmax(flat, axis=-1)
    hard_y, hard_x = np.divmod(hard_index, FEATURE_SIZE)
    hard_prediction_px = grid_to_pixel(hard_x, hard_y)

    local_prediction_px = np.empty_like(hard_prediction_px)
    inside_window_probability_mass = np.empty(
        (frame_count, EXPECTED_WITNESSES), dtype=np.float64
    )
    grid_x = np.arange(FEATURE_SIZE, dtype=np.float64)
    grid_y = np.arange(FEATURE_SIZE, dtype=np.float64)
    probability_grid = probability.reshape(
        frame_count, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE
    )
    for frame in range(frame_count):
        for witness in range(EXPECTED_WITNESSES):
            center_x = int(hard_x[frame, witness])
            center_y = int(hard_y[frame, witness])
            x0 = max(0, center_x - 1)
            x1 = min(FEATURE_SIZE, center_x + 2)
            y0 = max(0, center_y - 1)
            y1 = min(FEATURE_SIZE, center_y + 2)
            window = logit_array[frame, witness, y0:y1, x0:x1].astype(np.float64)
            window = np.exp(window - window.max())
            window /= window.sum()
            local_x = float((window * grid_x[x0:x1][None, :]).sum())
            local_y = float((window * grid_y[y0:y1][:, None]).sum())
            local_prediction_px[frame, witness] = grid_to_pixel(local_x, local_y)
            inside_window_probability_mass[frame, witness] = float(
                probability_grid[frame, witness, y0:y1, x0:x1].sum()
            )

    entropy = -(probability * np.log(np.clip(probability, 1e-300, None))).sum(axis=-1)
    result = {
        "hard_cell_x": hard_x.astype(np.int64),
        "hard_cell_y": hard_y.astype(np.int64),
        "hard_prediction_px": hard_prediction_px,
        "local_3x3_prediction_px": local_prediction_px,
        "top1_probability": top_two[..., 1],
        "top2_probability": top_two[..., 0],
        "top1_top2_probability_margin": top_two[..., 1] - top_two[..., 0],
        "heatmap_entropy": entropy,
        "inside_window_probability_mass": inside_window_probability_mass,
        "outside_window_probability_mass": 1.0 - inside_window_probability_mass,
    }

    if target_px is not None:
        target = np.asarray(target_px, dtype=np.float64)
        require(
            target.shape == (frame_count, EXPECTED_WITNESSES, 2),
            "target array shape differs",
        )
        target_x = np.rint(target[..., 0] / 511.0 * (FEATURE_SIZE - 1)).astype(np.int64)
        target_y = np.rint(target[..., 1] / 511.0 * (FEATURE_SIZE - 1)).astype(np.int64)
        target_x = np.clip(target_x, 0, FEATURE_SIZE - 1)
        target_y = np.clip(target_y, 0, FEATURE_SIZE - 1)
        target_cell_index = target_y * FEATURE_SIZE + target_x
        target_cell_logit = np.take_along_axis(
            flat, target_cell_index[..., None], axis=-1
        )[..., 0]
        target_rank = 1 + (flat > target_cell_logit[..., None]).sum(axis=-1)
        target_chebyshev_distance = np.maximum(
            np.abs(target_x - hard_x), np.abs(target_y - hard_y)
        )
        result.update(
            {
                "target_cell_x": target_x,
                "target_cell_y": target_y,
                "target_nearest_cell_rank": target_rank.astype(np.int64),
                "target_to_hard_chebyshev_cells": target_chebyshev_distance.astype(
                    np.int64
                ),
                "target_cell_inside_local_window": target_chebyshev_distance <= 1,
            }
        )
    return result


def classify_localization_failures(
    readouts: dict[str, np.ndarray], within_half_cell: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign the predeclared mutually exclusive category to each failure."""

    within = np.asarray(within_half_cell, dtype=bool)
    hard_x = np.asarray(readouts["hard_cell_x"], dtype=np.int64)
    hard_y = np.asarray(readouts["hard_cell_y"], dtype=np.int64)
    target_inside = np.asarray(readouts["target_cell_inside_local_window"], dtype=bool)
    target_rank = np.asarray(readouts["target_nearest_cell_rank"], dtype=np.int64)
    require(within.shape == hard_x.shape, "localization mask shape differs")

    failed = np.logical_not(within)
    border = (hard_x == 0) | (hard_x == FEATURE_SIZE - 1) | (hard_y == 0) | (
        hard_y == FEATURE_SIZE - 1
    )
    category = np.zeros(within.shape, dtype=np.int8)
    category[failed & border] = 1
    unresolved = failed & (category == 0)
    category[unresolved & ~target_inside & (target_rank <= 10)] = 2
    category[unresolved & ~target_inside & (target_rank > 10)] = 3
    category[unresolved & target_inside] = 4
    require(bool(np.all(category[failed] > 0)), "a localization failure was not classified")
    require(bool(np.all(category[within] == 0)), "a passing localization case was classified")
    counts = {
        LOCALIZATION_CATEGORY_NAMES[code]: int((category == code).sum())
        for code in range(1, 5)
    }
    return category, counts


def category_name(code: int) -> str:
    require(int(code) in LOCALIZATION_CATEGORY_NAMES, "unknown localization category")
    return LOCALIZATION_CATEGORY_NAMES[int(code)]


def compact_information_boundary() -> dict[str, Any]:
    return {
        "window_center": "current_frame_row_major_hard_argmax",
        "prediction_inputs": ["current_frame_native_r64_heatmap_logits"],
        "forbidden_prediction_inputs": [
            "target_track",
            "mask",
            "physical_geometry",
            "learned_operator_prediction",
            "previous_frame",
            "tracker",
            "manual_peak_selection",
        ],
        "temperature": 1.0,
        "window": "clipped_3x3",
        "inside_window_renormalization": True,
        "hard_tie_rule": "numpy_row_major_first_argmax",
    }
