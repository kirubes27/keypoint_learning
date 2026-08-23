"""Frozen-feature affine per-cell offset head for the capability gate."""

from __future__ import annotations

from typing import Any

import numpy as np

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
    require,
)
from certified_witness_local_readout import grid_to_pixel


FEATURE_CHANNELS = 128
OFFSET_LIMIT_GRID = 1.5


def target_pixel_to_grid(target_px: np.ndarray) -> np.ndarray:
    return np.asarray(target_px, dtype=np.float64) / 511.0 * (FEATURE_SIZE - 1)


def solve_affine_offset(
    design: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == FEATURE_CHANNELS + 1, "offset design shape differs")
    require(y.shape == (x.shape[0], 2), "offset label shape differs")
    require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), "offset fit contains non-finite values")
    coefficient, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficient
    residual = prediction - y
    require(bool(np.isfinite(coefficient).all()), "offset coefficient is non-finite")
    report = {
        "example_count": int(x.shape[0]),
        "parameter_count_per_axis": int(x.shape[1]),
        "design_rank": int(rank),
        "minimum_singular_value": float(singular_values.min()),
        "maximum_singular_value": float(singular_values.max()),
        "condition_number": float(singular_values.max() / singular_values.min()),
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficient)),
        "residual_sum_squares_solver": residuals.tolist(),
        "training_residual_mse": float(np.mean(residual**2)),
        "training_residual_max_abs": float(np.max(np.abs(residual))),
    }
    return coefficient, report


def build_offset_examples(
    features: np.ndarray, target_px: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    feature_array = np.asarray(features)
    target_grid = target_pixel_to_grid(target_px)
    require(
        feature_array.ndim == 4
        and feature_array.shape[1:]
        == (FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        "encoder feature shape differs",
    )
    frame_count = int(feature_array.shape[0])
    require(
        target_grid.shape == (frame_count, EXPECTED_WITNESSES, 2),
        "target shape differs",
    )
    nearest = np.rint(target_grid).astype(np.int64)
    nearest = np.clip(nearest, 0, FEATURE_SIZE - 1)
    design_values: list[np.ndarray] = []
    label_values: list[np.ndarray] = []
    metadata_values: list[dict[str, np.ndarray]] = []
    for witness in range(EXPECTED_WITNESSES):
        design_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        frame_rows: list[int] = []
        x_rows: list[int] = []
        y_rows: list[int] = []
        for frame in range(frame_count):
            target_x, target_y = target_grid[frame, witness]
            center_x, center_y = nearest[frame, witness]
            for anchor_y in range(max(0, center_y - 1), min(FEATURE_SIZE, center_y + 2)):
                for anchor_x in range(max(0, center_x - 1), min(FEATURE_SIZE, center_x + 2)):
                    row = np.empty(FEATURE_CHANNELS + 1, dtype=np.float64)
                    row[0] = 1.0
                    row[1:] = feature_array[frame, :, anchor_y, anchor_x]
                    label = np.asarray(
                        [target_x - anchor_x, target_y - anchor_y], dtype=np.float64
                    )
                    require(
                        bool(np.all(np.abs(label) <= OFFSET_LIMIT_GRID + 1e-12)),
                        "supervised anchor offset exceeds frozen support",
                    )
                    design_rows.append(row)
                    label_rows.append(label)
                    frame_rows.append(frame)
                    x_rows.append(anchor_x)
                    y_rows.append(anchor_y)
        design_values.append(np.stack(design_rows))
        label_values.append(np.stack(label_rows))
        metadata_values.append(
            {
                "frame": np.asarray(frame_rows, dtype=np.int64),
                "anchor_x": np.asarray(x_rows, dtype=np.int64),
                "anchor_y": np.asarray(y_rows, dtype=np.int64),
            }
        )
    return design_values, label_values, metadata_values


def fit_linear_offset_heads(
    features: np.ndarray, target_px: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    designs, labels, metadata = build_offset_examples(features, target_px)
    coefficients: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for witness, (design, label) in enumerate(zip(designs, labels, strict=True)):
        coefficient, report = solve_affine_offset(design, label)
        report["channel"] = witness
        coefficients.append(coefficient)
        reports.append(report)
    return np.stack(coefficients), reports, metadata


def predict_linear_offset_head(
    features: np.ndarray, logits: np.ndarray, coefficients: np.ndarray
) -> dict[str, np.ndarray]:
    feature_array = np.asarray(features)
    logit_array = np.asarray(logits)
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    require(
        feature_array.ndim == 4
        and feature_array.shape[1:]
        == (FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        "encoder feature shape differs",
    )
    frame_count = int(feature_array.shape[0])
    require(
        logit_array.shape
        == (frame_count, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        "heatmap logit shape differs",
    )
    require(
        coefficient_array.shape
        == (EXPECTED_WITNESSES, FEATURE_CHANNELS + 1, 2),
        "offset coefficient shape differs",
    )
    require(
        bool(np.isfinite(feature_array).all())
        and bool(np.isfinite(logit_array).all())
        and bool(np.isfinite(coefficient_array).all()),
        "linear offset prediction contains a non-finite input",
    )
    hard_index = np.argmax(
        logit_array.reshape(frame_count, EXPECTED_WITNESSES, -1), axis=-1
    )
    hard_y, hard_x = np.divmod(hard_index, FEATURE_SIZE)
    raw_offset = np.empty((frame_count, EXPECTED_WITNESSES, 2), dtype=np.float64)
    for frame in range(frame_count):
        for witness in range(EXPECTED_WITNESSES):
            row = np.empty(FEATURE_CHANNELS + 1, dtype=np.float64)
            row[0] = 1.0
            row[1:] = feature_array[
                frame, :, hard_y[frame, witness], hard_x[frame, witness]
            ]
            raw_offset[frame, witness] = row @ coefficient_array[witness]
    bounded_offset = np.clip(raw_offset, -OFFSET_LIMIT_GRID, OFFSET_LIMIT_GRID)
    prediction_grid = np.stack([hard_x, hard_y], axis=-1).astype(np.float64)
    prediction_grid += bounded_offset
    image_clamp_applied = np.any(
        (prediction_grid < 0.0) | (prediction_grid > FEATURE_SIZE - 1), axis=-1
    )
    prediction_grid = np.clip(prediction_grid, 0.0, FEATURE_SIZE - 1)
    return {
        "hard_cell_x": hard_x.astype(np.int64),
        "hard_cell_y": hard_y.astype(np.int64),
        "raw_offset_grid": raw_offset,
        "bounded_offset_grid": bounded_offset,
        "offset_clamp_applied": np.any(raw_offset != bounded_offset, axis=-1),
        "image_clamp_applied": image_clamp_applied,
        "prediction_grid_xy": prediction_grid,
        "prediction_px": grid_to_pixel(prediction_grid[..., 0], prediction_grid[..., 1]),
    }


def compact_information_boundary() -> dict[str, object]:
    return {
        "coarse_center": "current_frame_native_r64_row_major_hard_argmax",
        "prediction_inputs": [
            "current_frame_native_r64_heatmap_logits_for_anchor_only",
            "current_frame_frozen_r64_encoder_feature_at_anchor",
            "fixed_fitted_affine_offset_coefficients",
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
        "head": "independent_affine_128_feature_to_xy_offset_per_witness",
        "fit": "unweighted_numpy_lstsq_rcond_none",
        "hidden_layers": 0,
        "offset_clip_grid_cells": [-OFFSET_LIMIT_GRID, OFFSET_LIMIT_GRID],
        "final_grid_clip": [0.0, float(FEATURE_SIZE - 1)],
        "hard_tie_rule": "numpy_row_major_first_argmax",
    }
