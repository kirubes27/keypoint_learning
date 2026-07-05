"""Frozen synthetic gate for the single conditional-dead-zone fallback."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_gradient_audit import (  # noqa: E402
    SEEDS,
    initial_model,
    load_checkpoint_model,
    load_fixed_batch,
)
from diagnostics.stage_a_shape_constraint import (  # noqa: E402
    DEADZONE_DOMINANT_MASS_R2_MIN,
    DEADZONE_EFFECTIVE_SUPPORT_RANGE,
    DEADZONE_MAX_PROBABILITY_RANGE,
    conditional_deadzone_shape,
)
from diagnostics.stage_r1_gradient_path_audit import (  # noqa: E402
    reconstruction_relative_error,
)
from model import spatial_softmax  # noqa: E402


CALIBRATION_SEED = 42
NORM_FLOOR = 1e-12
ZERO_LOSS_MAX = 1e-10
ZERO_GRADIENT_MAX = 1e-8
POSITIVE_CONTROL_GRADIENT_MAX = 1e-10
TRANSLATION_LOSS_TOLERANCE = 1e-8
HEALTHY_DESCENT_MULTIPLIER_MIN = 0.90
ACTIVE_DESCENT_MULTIPLIER_MIN = 0.0
RECONSTRUCTION_ERROR_MAX = 1e-5
REPAIR_MAX_STEPS = 2000
REPAIR_LR = 0.10
PROTOTYPE_SIZE = 33
HEALTH_COMPARISON_TOLERANCE = 1e-8


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _gaussian_logits(
    center_x: float,
    center_y: float,
    sigma: float,
    *,
    size: int = PROTOTYPE_SIZE,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=dtype),
        torch.arange(size, dtype=dtype),
        indexing="ij",
    )
    return (
        -0.5 * ((xx - center_x) ** 2 + (yy - center_y) ** 2) / sigma**2
    ).reshape(1, 1, size, size)


def prototypes() -> dict[str, torch.Tensor]:
    spike = _gaussian_logits(16.0, 16.0, 0.25)
    diffuse = _gaussian_logits(16.0, 16.0, 5.0)
    uniform = torch.zeros_like(spike)
    first = _gaussian_logits(10.0, 16.0, 1.0)
    second = _gaussian_logits(22.0, 16.0, 1.0)
    bimodal = torch.logaddexp(first, second)
    return {
        "spike": spike,
        "diffuse": diffuse,
        "uniform": uniform,
        "equal_separated_modes": bimodal,
    }


def compatibility_prototypes() -> dict[str, torch.Tensor]:
    values = prototypes()
    return {
        "spike_sigma0.35": _gaussian_logits(16.0, 16.0, 0.35),
        "diffuse": values["diffuse"],
        "equal_separated_modes": values["equal_separated_modes"],
    }


def _is_healthy(output: Any) -> bool:
    max_low, max_high = DEADZONE_MAX_PROBABILITY_RANGE
    support_low, support_high = DEADZONE_EFFECTIVE_SUPPORT_RANGE
    return bool(
        torch.all(output.max_probability >= max_low - HEALTH_COMPARISON_TOLERANCE)
        and torch.all(
            output.max_probability <= max_high + HEALTH_COMPARISON_TOLERANCE
        )
        and torch.all(
            output.effective_support_cells
            >= support_low - HEALTH_COMPARISON_TOLERANCE
        )
        and torch.all(
            output.effective_support_cells
            <= support_high + HEALTH_COMPARISON_TOLERANCE
        )
        and torch.all(
            output.dominant_mass_r2
            >= DEADZONE_DOMINANT_MASS_R2_MIN - HEALTH_COMPARISON_TOLERANCE
        )
    )


def _shape_snapshot(logits: torch.Tensor) -> dict[str, float | bool]:
    output = conditional_deadzone_shape(logits)
    return {
        "loss": float(output.loss.detach()),
        "max_probability": float(output.max_probability.detach().item()),
        "effective_support_cells": float(
            output.effective_support_cells.detach().item()
        ),
        "dominant_mass_r2": float(output.dominant_mass_r2.detach().item()),
        "healthy": _is_healthy(output),
    }


def calibrate_weight(data_root: Path) -> dict[str, Any]:
    image, target, frames = load_fixed_batch(data_root, shift=0)
    model = initial_model(CALIBRATION_SEED)
    flat_coordinates, logits = model(image)
    coordinates = flat_coordinates.view(image.shape[0], -1, 2)
    coordinate_loss = F.mse_loss(coordinates, target)
    shape_loss = conditional_deadzone_shape(logits).loss
    coordinate_gradient = torch.autograd.grad(
        coordinate_loss, logits, retain_graph=True
    )[0]
    shape_gradient = torch.autograd.grad(shape_loss, logits)[0]
    coordinate_norm = float(torch.linalg.vector_norm(coordinate_gradient))
    shape_norm = float(torch.linalg.vector_norm(shape_gradient))
    weight = coordinate_norm / max(shape_norm, NORM_FLOOR)
    return {
        "seed": CALIBRATION_SEED,
        "frames": frames,
        "coordinate_loss": float(coordinate_loss.detach()),
        "deadzone_loss": float(shape_loss.detach()),
        "coordinate_logit_gradient_l2": coordinate_norm,
        "deadzone_logit_gradient_l2": shape_norm,
        "gradient_norm_floor": NORM_FLOOR,
        "lambda_deadzone": weight,
        "formula": (
            "norm(d_coordinate_loss/d_logits) / "
            "max(norm(d_deadzone_loss/d_logits), 1e-12)"
        ),
    }


def positive_control_gate(
    data_root: Path, positive_control_root: Path
) -> dict[str, Any]:
    image, _, frames = load_fixed_batch(data_root, shift=0)
    values = {
        "max_probability": [],
        "effective_support_cells": [],
        "dominant_mass_r2": [],
        "per_channel_loss": [],
    }
    aggregate_losses = []
    gradients = []
    for seed in SEEDS:
        checkpoint = (
            positive_control_root
            / f"heatmap_standard64_k10_seed{seed}"
            / "model.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_checkpoint_model(checkpoint)
        model.train()
        _, logits = model(image)
        output = conditional_deadzone_shape(logits)
        gradient = torch.autograd.grad(output.loss, logits)[0]
        aggregate_losses.append(float(output.loss.detach()))
        gradients.append(float(torch.linalg.vector_norm(gradient)))
        values["max_probability"].extend(
            output.max_probability.detach().flatten().cpu().tolist()
        )
        values["effective_support_cells"].extend(
            output.effective_support_cells.detach().flatten().cpu().tolist()
        )
        values["dominant_mass_r2"].extend(
            output.dominant_mass_r2.detach().flatten().cpu().tolist()
        )
        values["per_channel_loss"].extend(
            output.per_channel_loss.detach().flatten().cpu().tolist()
        )
    max_gradient = max(gradients)
    max_loss = max(values["per_channel_loss"])
    return {
        "passed": bool(
            max_loss == 0.0
            and max(aggregate_losses) == 0.0
            and max_gradient <= POSITIVE_CONTROL_GRADIENT_MAX
        ),
        "frames": frames,
        "seeds": list(SEEDS),
        "unit_count": len(values["per_channel_loss"]),
        "max_per_channel_loss": max_loss,
        "max_aggregate_loss": max(aggregate_losses),
        "max_logit_gradient_l2": max_gradient,
        "ranges": {
            key: _quantiles(value)
            for key, value in values.items()
            if key != "per_channel_loss"
        },
    }


def gradient_compatibility(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: float,
) -> dict[str, float | bool]:
    logits = logits.detach().clone().requires_grad_()
    coordinates = spatial_softmax(logits)
    coordinate_loss = F.mse_loss(coordinates, target)
    weighted_shape_loss = weight * conditional_deadzone_shape(logits).loss
    coordinate_gradient = torch.autograd.grad(
        coordinate_loss, logits, retain_graph=True
    )[0]
    shape_gradient = torch.autograd.grad(
        weighted_shape_loss, logits, retain_graph=True
    )[0]
    direct_gradient = torch.autograd.grad(
        coordinate_loss + weighted_shape_loss, logits
    )[0]
    coordinate_norm_sq = float(torch.sum(coordinate_gradient**2))
    if coordinate_norm_sq <= NORM_FLOOR:
        raise ValueError("coordinate gradient vanished in compatibility probe")
    multiplier = float(
        torch.sum(coordinate_gradient * (coordinate_gradient + shape_gradient))
    ) / coordinate_norm_sq
    reconstruction = reconstruction_relative_error(
        direct_gradient.flatten(),
        coordinate_gradient.flatten(),
        shape_gradient.flatten(),
    )
    return {
        "coordinate_descent_multiplier": multiplier,
        "coordinate_gradient_l2": math.sqrt(coordinate_norm_sq),
        "weighted_shape_gradient_l2": float(
            torch.linalg.vector_norm(shape_gradient)
        ),
        "gradient_reconstruction_relative_error": reconstruction,
        "finite": bool(
            torch.isfinite(coordinate_gradient).all()
            and torch.isfinite(shape_gradient).all()
            and torch.isfinite(direct_gradient).all()
        ),
    }


def directional_compatibility(
    logits: torch.Tensor,
    weight: float,
) -> dict[str, dict[str, float | bool]]:
    with torch.no_grad():
        coordinate = spatial_softmax(logits).reshape(1, 1, 2)
    delta = 0.5 * 2.0 / (logits.shape[-1] - 1)
    directions = {
        "+x": (delta, 0.0),
        "-x": (-delta, 0.0),
        "+y": (0.0, delta),
        "-y": (0.0, -delta),
    }
    return {
        name: gradient_compatibility(
            logits,
            coordinate
            + torch.tensor(offset, dtype=coordinate.dtype).reshape(1, 1, 2),
            weight,
        )
        for name, offset in directions.items()
    }


def repair_prototype(logits: torch.Tensor) -> dict[str, Any]:
    variable = torch.nn.Parameter(logits.detach().clone())
    optimizer = torch.optim.Adam((variable,), lr=REPAIR_LR)
    initial = _shape_snapshot(variable)
    initial_output = conditional_deadzone_shape(variable)
    initial_gradient = torch.autograd.grad(initial_output.loss, variable)[0]
    initial_gradient_l2 = float(torch.linalg.vector_norm(initial_gradient))
    completed_steps = 0
    for step in range(1, REPAIR_MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_deadzone_shape(variable).loss
        loss.backward()
        if not torch.isfinite(variable.grad).all():
            break
        optimizer.step()
        completed_steps = step
        if _is_healthy(conditional_deadzone_shape(variable)):
            break
    final_output = conditional_deadzone_shape(variable)
    final_gradient = torch.autograd.grad(final_output.loss, variable)[0]
    final_gradient_l2 = float(torch.linalg.vector_norm(final_gradient))
    final = _shape_snapshot(variable)
    passed = bool(
        float(initial_output.loss) > 0.0
        and math.isfinite(initial_gradient_l2)
        and initial_gradient_l2 > 0.0
        and final["healthy"]
        and float(final["loss"]) <= ZERO_LOSS_MAX
        and final_gradient_l2 <= ZERO_GRADIENT_MAX
        and completed_steps <= REPAIR_MAX_STEPS
    )
    return {
        "passed": passed,
        "completed_steps": completed_steps,
        "initial": initial,
        "initial_gradient_l2": initial_gradient_l2,
        "final": final,
        "final_gradient_l2": final_gradient_l2,
    }


def healthy_translation_gate(weight: float) -> dict[str, Any]:
    positions = ((10.0, 10.0), (16.0, 16.0), (22.0, 20.0))
    rows = []
    for center_x, center_y in positions:
        logits = _gaussian_logits(center_x, center_y, 1.0)
        output = conditional_deadzone_shape(logits.requires_grad_())
        gradient = torch.autograd.grad(output.loss, logits)[0]
        compatibility = directional_compatibility(logits.detach(), weight)
        rows.append({
            "center": [center_x, center_y],
            "loss": float(output.loss.detach()),
            "gradient_l2": float(torch.linalg.vector_norm(gradient)),
            "compatibility": compatibility,
        })
    losses = [row["loss"] for row in rows]
    multipliers = [
        float(value["coordinate_descent_multiplier"])
        for row in rows
        for value in row["compatibility"].values()
    ]
    reconstructions = [
        float(value["gradient_reconstruction_relative_error"])
        for row in rows
        for value in row["compatibility"].values()
    ]
    passed = bool(
        max(losses) - min(losses) <= TRANSLATION_LOSS_TOLERANCE
        and max(losses) <= ZERO_LOSS_MAX
        and max(row["gradient_l2"] for row in rows) <= ZERO_GRADIENT_MAX
        and min(multipliers) >= HEALTHY_DESCENT_MULTIPLIER_MIN
        and max(reconstructions) <= RECONSTRUCTION_ERROR_MAX
    )
    return {
        "passed": passed,
        "minimum_coordinate_descent_multiplier": min(multipliers),
        "maximum_loss_difference": max(losses) - min(losses),
        "maximum_reconstruction_relative_error": max(reconstructions),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> Path:
    calibration = calibrate_weight(args.data_root)
    weight = float(calibration["lambda_deadzone"])
    controls = positive_control_gate(args.data_root, args.positive_control_root)
    prototype_values = prototypes()
    repairs = {
        name: repair_prototype(logits)
        for name, logits in prototype_values.items()
    }
    active_compatibility = {
        name: directional_compatibility(logits, weight)
        for name, logits in compatibility_prototypes().items()
    }
    active_multipliers = [
        float(value["coordinate_descent_multiplier"])
        for prototype in active_compatibility.values()
        for value in prototype.values()
    ]
    active_reconstruction = [
        float(value["gradient_reconstruction_relative_error"])
        for prototype in active_compatibility.values()
        for value in prototype.values()
    ]
    active_finite = all(
        bool(value["finite"])
        for prototype in active_compatibility.values()
        for value in prototype.values()
    )
    active_pass = bool(
        min(active_multipliers) >= ACTIVE_DESCENT_MULTIPLIER_MIN
        and max(active_reconstruction) <= RECONSTRUCTION_ERROR_MAX
        and active_finite
    )
    healthy = healthy_translation_gate(weight)
    all_repairs_pass = all(bool(value["passed"]) for value in repairs.values())
    passed = bool(controls["passed"] and healthy["passed"] and all_repairs_pass and active_pass)
    payload = {
        "stage": "R0_conditional_deadzone_fallback_gate",
        "passed": passed,
        "decision": (
            "synthetic gate passes; three-seed fallback R1 may be implemented"
            if passed
            else "synthetic gate fails; stop this instrument design and review"
        ),
        "calibration": calibration,
        "positive_controls": controls,
        "healthy_translation": healthy,
        "prototype_repairs": repairs,
        "active_compatibility": {
            "passed": active_pass,
            "minimum_coordinate_descent_multiplier": min(active_multipliers),
            "maximum_reconstruction_relative_error": max(active_reconstruction),
            "all_gradients_finite": active_finite,
            "by_prototype": active_compatibility,
        },
        "thresholds": {
            "max_probability_range": list(DEADZONE_MAX_PROBABILITY_RANGE),
            "effective_support_range": list(DEADZONE_EFFECTIVE_SUPPORT_RANGE),
            "dominant_mass_r2_min": DEADZONE_DOMINANT_MASS_R2_MIN,
            "positive_control_gradient_l2_max": POSITIVE_CONTROL_GRADIENT_MAX,
            "healthy_coordinate_descent_multiplier_min": (
                HEALTHY_DESCENT_MULTIPLIER_MIN
            ),
            "active_coordinate_descent_multiplier_min": (
                ACTIVE_DESCENT_MULTIPLIER_MIN
            ),
            "repair_max_steps": REPAIR_MAX_STEPS,
            "repair_final_loss_max": ZERO_LOSS_MAX,
            "repair_final_gradient_l2_max": ZERO_GRADIENT_MAX,
            "gradient_reconstruction_relative_error_max": (
                RECONSTRUCTION_ERROR_MAX
            ),
            "health_comparison_tolerance": HEALTH_COMPARISON_TOLERANCE,
        },
        "statistical_scope": (
            "descriptive synthetic mechanism gate; heatmap positive controls "
            "contain one object, four correlated frames and n=3 optimization "
            "seeds; prototypes are deterministic cases; no hypothesis test or "
            "population inference"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "R0_DEADZONE_FALLBACK_GATE.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--positive-control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
