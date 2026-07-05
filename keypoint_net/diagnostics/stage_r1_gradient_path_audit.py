"""Read-only R1 gradient-path audit.

Compare the authoritative prediction-centred-JS checkpoints with matched
coordinate-only controls. No optimizer step or checkpoint mutation occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_gradient_audit import (
    SEEDS,
    _checkpoint_result,
    load_checkpoint_model,
    load_fixed_batch,
)
from diagnostics.stage_a_shape_constraint import prediction_centered_js


LEVELS = ("logits", "head", "backbone")
PARAMETER_LEVELS = ("head", "backbone")
RECONSTRUCTION_RELATIVE_ERROR_MAX = 1e-5
CONFLICT_SUPPORT_CANCELLATION_MIN = 0.50
CONFLICT_REJECT_CANCELLATION_MAX = 0.25
ATTENUATION_SUPPORT_RATIO_MAX = 0.25
ATTENUATION_REJECT_RATIO_MIN = 0.50
REPLICATIONS_REQUIRED = 2
EPS = 1e-12


def _flat_gradients(
    gradients: Iterable[torch.Tensor | None],
    tensors: Iterable[torch.Tensor],
) -> torch.Tensor:
    pieces = []
    for gradient, tensor in zip(gradients, tensors, strict=True):
        pieces.append(
            torch.zeros_like(tensor).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    if not pieces:
        raise ValueError("gradient group cannot be empty")
    return torch.cat(pieces)


def reconstruction_relative_error(
    direct: torch.Tensor,
    coordinate: torch.Tensor,
    shape: torch.Tensor,
) -> float:
    expected = coordinate + shape
    denominator = max(
        float(torch.linalg.vector_norm(direct)),
        float(torch.linalg.vector_norm(expected)),
        EPS,
    )
    return float(torch.linalg.vector_norm(direct - expected)) / denominator


def vector_metrics(
    coordinate: torch.Tensor,
    shape: torch.Tensor,
) -> dict[str, float]:
    coordinate_norm = float(torch.linalg.vector_norm(coordinate))
    shape_norm = float(torch.linalg.vector_norm(shape))
    if coordinate_norm <= EPS:
        raise ValueError("coordinate gradient is zero")
    dot = float(torch.dot(coordinate, shape))
    cosine = dot / max(coordinate_norm * shape_norm, EPS)
    total = coordinate + shape
    descent_multiplier = 1.0 + dot / (coordinate_norm**2)
    return {
        "coordinate_gradient_l2": coordinate_norm,
        "weighted_shape_gradient_l2": shape_norm,
        "shape_to_coordinate_norm_ratio": shape_norm / coordinate_norm,
        "coordinate_shape_cosine": cosine,
        "combined_to_coordinate_norm_ratio": (
            float(torch.linalg.vector_norm(total)) / coordinate_norm
        ),
        "coordinate_descent_multiplier": descent_multiplier,
        "harmful_cancellation_fraction": max(0.0, 1.0 - descent_multiplier),
    }


def classify_from_seed_values(
    values_by_level: dict[str, list[float]],
    *,
    support_at_or_beyond: Callable[[float], bool],
    reject_strictly_beyond: Callable[[float], bool],
) -> dict[str, Any]:
    support_counts = {
        level: sum(bool(support_at_or_beyond(value)) for value in values)
        for level, values in values_by_level.items()
    }
    supported_levels = [
        level
        for level, count in support_counts.items()
        if count >= REPLICATIONS_REQUIRED
    ]
    rejected = all(
        bool(reject_strictly_beyond(value))
        for values in values_by_level.values()
        for value in values
    )
    status = "supported" if supported_levels else "rejected" if rejected else "mixed"
    return {
        "status": status,
        "supported_levels": supported_levels,
        "support_count_out_of_3_by_level": support_counts,
        "values_by_level_and_seed": values_by_level,
    }


def _group_parameters(model: torch.nn.Module) -> dict[str, tuple[torch.Tensor, ...]]:
    head = tuple(model.heatmap_head.parameters())
    if getattr(model, "head_upsample", None) is not None:
        head = tuple(model.head_upsample.parameters()) + head
    return {
        "head": head,
        "backbone": tuple(model.encoder.parameters()),
    }


def measure_model_gradients(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    shape_weight: float,
    shape_sigma_cells: float,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    model.train()
    parameter = next(model.parameters())
    image = image.to(device=parameter.device, dtype=parameter.dtype)
    target = target.to(device=parameter.device, dtype=parameter.dtype)
    flat_coordinates, logits = model(image)
    coordinates = flat_coordinates.view(image.shape[0], -1, 2)
    coordinate_loss = F.mse_loss(coordinates, target)
    unweighted_shape_loss = prediction_centered_js(
        logits, sigma_cells=shape_sigma_cells
    ).loss
    weighted_shape_loss = shape_weight * unweighted_shape_loss

    parameter_groups = _group_parameters(model)
    tensors: list[torch.Tensor] = [logits]
    slices: dict[str, slice] = {"logits": slice(0, 1)}
    for level in PARAMETER_LEVELS:
        start = len(tensors)
        tensors.extend(parameter_groups[level])
        slices[level] = slice(start, len(tensors))

    coordinate_raw = torch.autograd.grad(
        coordinate_loss, tensors, retain_graph=True, allow_unused=True
    )
    shape_raw = torch.autograd.grad(
        weighted_shape_loss, tensors, retain_graph=True, allow_unused=True
    )
    direct_raw = torch.autograd.grad(
        coordinate_loss + weighted_shape_loss,
        tensors,
        retain_graph=False,
        allow_unused=True,
    )

    rows: list[dict[str, float | str]] = []
    coordinate_logit_norm: float | None = None
    shape_logit_norm: float | None = None
    for level in LEVELS:
        subset = tensors[slices[level]]
        coordinate_vector = _flat_gradients(coordinate_raw[slices[level]], subset)
        shape_vector = _flat_gradients(shape_raw[slices[level]], subset)
        direct_vector = _flat_gradients(direct_raw[slices[level]], subset)
        row: dict[str, float | str] = {"level": level}
        row.update(vector_metrics(coordinate_vector, shape_vector))
        row["reconstruction_relative_error"] = reconstruction_relative_error(
            direct_vector, coordinate_vector, shape_vector
        )
        if level == "logits":
            coordinate_logit_norm = float(row["coordinate_gradient_l2"])
            shape_logit_norm = float(row["weighted_shape_gradient_l2"])
            row["coordinate_transmission_gain"] = 1.0
            row["shape_transmission_gain"] = 1.0
        else:
            assert coordinate_logit_norm is not None and shape_logit_norm is not None
            row["coordinate_transmission_gain"] = (
                float(row["coordinate_gradient_l2"]) / coordinate_logit_norm
            )
            row["shape_transmission_gain"] = (
                float(row["weighted_shape_gradient_l2"]) / max(shape_logit_norm, EPS)
            )
        rows.append(row)

    losses = {
        "coordinate_loss": float(coordinate_loss.detach()),
        "unweighted_shape_loss": float(unweighted_shape_loss.detach()),
        "weighted_shape_loss": float(weighted_shape_loss.detach()),
    }
    return rows, losses


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repaired = [row for row in rows if row["condition"] == "repaired_r1"]
    control = {
        (int(row["seed"]), row["level"]): row
        for row in rows
        if row["condition"] == "coordinate_only_control"
    }
    for row in repaired:
        key = (int(row["seed"]), row["level"])
        row["coordinate_transmission_vs_control_ratio"] = (
            float(row["coordinate_transmission_gain"])
            / max(float(control[key]["coordinate_transmission_gain"]), EPS)
        )
    for row in rows:
        row.setdefault("coordinate_transmission_vs_control_ratio", None)

    conflict_values = {
        level: [
            float(row["harmful_cancellation_fraction"])
            for row in repaired
            if row["level"] == level
        ]
        for level in PARAMETER_LEVELS
    }
    attenuation_values = {
        level: [
            float(row["coordinate_transmission_vs_control_ratio"])
            for row in repaired
            if row["level"] == level
        ]
        for level in PARAMETER_LEVELS
    }
    conflict = classify_from_seed_values(
        conflict_values,
        support_at_or_beyond=lambda value: value >= CONFLICT_SUPPORT_CANCELLATION_MIN,
        reject_strictly_beyond=lambda value: value < CONFLICT_REJECT_CANCELLATION_MAX,
    )
    attenuation = classify_from_seed_values(
        attenuation_values,
        support_at_or_beyond=lambda value: value <= ATTENUATION_SUPPORT_RATIO_MAX,
        reject_strictly_beyond=lambda value: value > ATTENUATION_REJECT_RATIO_MIN,
    )
    reconstruction_errors = [
        float(row["reconstruction_relative_error"]) for row in rows
    ]
    numeric_values = [
        float(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float)) and value is not None
    ]
    valid = bool(
        all(math.isfinite(value) for value in numeric_values)
        and max(reconstruction_errors) <= RECONSTRUCTION_RELATIVE_ERROR_MAX
    )
    if not valid:
        decision = "invalid_audit"
    elif conflict["status"] == "supported" and attenuation["status"] == "supported":
        decision = "both_loss_conflict_and_network_attenuation_supported"
    elif conflict["status"] == "supported":
        decision = "loss_conflict_supported"
    elif attenuation["status"] == "supported":
        decision = "network_attenuation_supported"
    else:
        decision = "inconclusive"
    return {
        "audit_valid": valid,
        "decision": decision,
        "loss_conflict": conflict,
        "network_attenuation": attenuation,
        "max_gradient_reconstruction_relative_error": max(reconstruction_errors),
        "thresholds": {
            "gradient_reconstruction_relative_error_max": (
                RECONSTRUCTION_RELATIVE_ERROR_MAX
            ),
            "conflict_support_cancellation_fraction_min": (
                CONFLICT_SUPPORT_CANCELLATION_MIN
            ),
            "conflict_reject_cancellation_fraction_below": (
                CONFLICT_REJECT_CANCELLATION_MAX
            ),
            "attenuation_support_repaired_to_control_ratio_max": (
                ATTENUATION_SUPPORT_RATIO_MAX
            ),
            "attenuation_reject_repaired_to_control_ratio_above": (
                ATTENUATION_REJECT_RATIO_MIN
            ),
            "replications_required_out_of_3": REPLICATIONS_REQUIRED,
        },
        "formulas": {
            "coordinate_descent_multiplier": (
                "dot(g_coord, g_coord + g_shape) / ||g_coord||^2"
            ),
            "harmful_cancellation_fraction": (
                "max(0, 1 - coordinate_descent_multiplier)"
            ),
            "gradient_transmission_gain": "||g_parameter|| / ||g_logits||",
            "attenuation_ratio": "repaired_gain / matched_control_gain",
        },
        "statistical_scope": (
            "descriptive one-object/four-correlated-frame checkpoint audit; "
            "n=3 optimization seeds per condition; seed is the replication unit; "
            "no error bars, hypothesis test, or population inference"
        ),
    }


def run(args: argparse.Namespace) -> Path:
    image, target, frames = load_fixed_batch(args.data_root, 0)
    rows: list[dict[str, Any]] = []
    expected_weight: float | None = None
    for seed in SEEDS:
        repaired_checkpoint = (
            args.repaired_root
            / f"coordinate_standard64_k10_shapejs_seed{seed}"
            / "model.pt"
        )
        control_checkpoint = (
            args.control_root
            / f"coordinate_standard64_k10_seed{seed}"
            / "model.pt"
        )
        for checkpoint in (repaired_checkpoint, control_checkpoint):
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        repaired_result = _checkpoint_result(repaired_checkpoint)
        shape_weight = float(repaired_result["shape_weight"])
        shape_sigma = float(repaired_result["shape_sigma_cells"])
        if expected_weight is None:
            expected_weight = shape_weight
        elif not math.isclose(shape_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("R1 shape weights differ across seeds")
        if not math.isclose(shape_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("R1 shape sigma is not the frozen one-cell value")

        sources = (
            ("repaired_r1", repaired_checkpoint),
            ("coordinate_only_control", control_checkpoint),
        )
        for condition, checkpoint in sources:
            # The direct-vs-summed reconstruction gate is intentionally strict.
            # Evaluate the frozen float32 weights in float64 so convolution
            # reduction-order noise does not dominate that correctness check.
            model = load_checkpoint_model(checkpoint).double()
            measurements, losses = measure_model_gradients(
                model,
                image,
                target,
                shape_weight=shape_weight,
                shape_sigma_cells=shape_sigma,
            )
            for measurement in measurements:
                rows.append({
                    "condition": condition,
                    "seed": seed,
                    "frames": ",".join(str(frame) for frame in frames),
                    "checkpoint": str(checkpoint),
                    "shape_weight": shape_weight,
                    "shape_sigma_cells": shape_sigma,
                    "analysis_dtype": "float64",
                    **losses,
                    **measurement,
                })

    result = summarize(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "R1_GRADIENT_PATH_ROWS.csv"
    json_path = args.output_root / "R1_GRADIENT_PATH_SUMMARY.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
