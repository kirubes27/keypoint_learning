"""Known-sigma local log-Gaussian center readout for r64 heatmaps."""

from __future__ import annotations

import numpy as np

from certified_witness_capability import (
    CELL_SPACING_PX,
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
    require,
)
from certified_witness_local_readout import grid_to_pixel


SIGMA_INPUT_PX = 8.0
SIGMA_GRID = SIGMA_INPUT_PX / CELL_SPACING_PX


def gaussian_center_readout_arrays(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Decode local continuous centers under the frozen known-sigma model.

    The current-frame row-major hard argmax chooses a clipped 3x3 patch. For
    ideal Gaussian logits with the training sigma, removing the known
    quadratic term leaves a plane whose slopes identify the continuous center.
    """

    logit_array = np.asarray(logits)
    require(
        logit_array.ndim == 4
        and logit_array.shape[1:]
        == (EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "logit array shape differs",
    )
    require(bool(np.isfinite(logit_array).all()), "logit array contains non-finite values")
    frame_count = int(logit_array.shape[0])
    flat = logit_array.reshape(frame_count, EXPECTED_WITNESSES, -1)
    hard_index = np.argmax(flat, axis=-1)
    hard_y, hard_x = np.divmod(hard_index, FEATURE_SIZE)
    predicted_grid = np.empty((frame_count, EXPECTED_WITNESSES, 2), dtype=np.float64)
    raw_offset = np.empty_like(predicted_grid)
    clamped_offset = np.empty_like(predicted_grid)
    clamp_applied = np.zeros((frame_count, EXPECTED_WITNESSES), dtype=bool)
    design_rank = np.empty((frame_count, EXPECTED_WITNESSES), dtype=np.int64)
    fit_residual_sum_squares = np.empty(
        (frame_count, EXPECTED_WITNESSES), dtype=np.float64
    )
    sigma_squared = SIGMA_GRID**2

    for frame in range(frame_count):
        for witness in range(EXPECTED_WITNESSES):
            center_x = int(hard_x[frame, witness])
            center_y = int(hard_y[frame, witness])
            x0 = max(0, center_x - 1)
            x1 = min(FEATURE_SIZE, center_x + 2)
            y0 = max(0, center_y - 1)
            y1 = min(FEATURE_SIZE, center_y + 2)
            dx_values = np.arange(x0, x1, dtype=np.float64) - center_x
            dy_values = np.arange(y0, y1, dtype=np.float64) - center_y
            dy, dx = np.meshgrid(dy_values, dx_values, indexing="ij")
            window = logit_array[frame, witness, y0:y1, x0:x1].astype(np.float64)
            transformed = window + (dx**2 + dy**2) / (2.0 * sigma_squared)
            design = np.stack(
                [np.ones(dx.size, dtype=np.float64), dx.reshape(-1), dy.reshape(-1)],
                axis=-1,
            )
            coefficient, residuals, rank, _ = np.linalg.lstsq(
                design, transformed.reshape(-1), rcond=None
            )
            require(int(rank) == 3, "known-sigma local plane fit is rank deficient")
            offset = sigma_squared * coefficient[1:3]
            require(bool(np.isfinite(offset).all()), "known-sigma local offset is non-finite")
            lower = np.asarray([dx_values.min(), dy_values.min()], dtype=np.float64)
            upper = np.asarray([dx_values.max(), dy_values.max()], dtype=np.float64)
            bounded = np.clip(offset, lower, upper)
            raw_offset[frame, witness] = offset
            clamped_offset[frame, witness] = bounded
            clamp_applied[frame, witness] = bool(np.any(bounded != offset))
            design_rank[frame, witness] = int(rank)
            fit_residual_sum_squares[frame, witness] = (
                float(residuals[0]) if residuals.size else 0.0
            )
            predicted_grid[frame, witness] = [
                center_x + bounded[0],
                center_y + bounded[1],
            ]

    return {
        "hard_cell_x": hard_x.astype(np.int64),
        "hard_cell_y": hard_y.astype(np.int64),
        "raw_offset_grid": raw_offset,
        "clamped_offset_grid": clamped_offset,
        "clamp_applied": clamp_applied,
        "design_rank": design_rank,
        "fit_residual_sum_squares": fit_residual_sum_squares,
        "prediction_grid_xy": predicted_grid,
        "prediction_px": grid_to_pixel(predicted_grid[..., 0], predicted_grid[..., 1]),
    }


def compact_information_boundary() -> dict[str, object]:
    return {
        "coarse_center": "current_frame_row_major_hard_argmax",
        "prediction_inputs": [
            "current_frame_native_r64_heatmap_logits",
            "frozen_training_sigma_8_input_px",
        ],
        "forbidden_prediction_inputs": [
            "target_track",
            "mask",
            "physical_geometry",
            "learned_operator_prediction",
            "previous_frame",
            "tracker",
            "manual_peak_selection",
        ],
        "window": "clipped_3x3",
        "fit": "unweighted_plane_after_known_quadratic_removal",
        "sigma_input_px": SIGMA_INPUT_PX,
        "sigma_grid": SIGMA_GRID,
        "offset_clamp": "actual_clipped_patch_extent",
        "fallback": None,
        "hard_tie_rule": "numpy_row_major_first_argmax",
    }
