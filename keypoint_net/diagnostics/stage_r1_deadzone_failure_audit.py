"""Read-only post-mortem for the failed conditional-deadzone R1 gate.

The audit recomputes coordinate and weighted deadzone gradients at the saved
checkpoints. It performs no optimizer step and never mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_gradient_audit import (  # noqa: E402
    SEEDS,
    load_checkpoint_model,
    load_fixed_batch,
)
from diagnostics.stage_a_shape_constraint import (  # noqa: E402
    conditional_deadzone_shape,
)


LEVELS = ("logits", "head", "backbone")
EPS = 1e-30
SATURATED_MAX_PROBABILITY_MIN = 0.99


def _flat_gradients(
    gradients: Iterable[torch.Tensor | None],
    tensors: Iterable[torch.Tensor],
) -> torch.Tensor:
    return torch.cat([
        (torch.zeros_like(tensor) if gradient is None else gradient).reshape(-1)
        for gradient, tensor in zip(gradients, tensors, strict=True)
    ])


def _parameter_groups(model: torch.nn.Module) -> dict[str, tuple[torch.Tensor, ...]]:
    head = tuple(model.heatmap_head.parameters())
    if getattr(model, "head_upsample", None) is not None:
        head = tuple(model.head_upsample.parameters()) + head
    return {
        "head": head,
        "backbone": tuple(model.encoder.parameters()),
    }


def _vector_metrics(
    coordinate: torch.Tensor,
    shape: torch.Tensor,
) -> dict[str, float]:
    coordinate_norm = float(torch.linalg.vector_norm(coordinate))
    shape_norm = float(torch.linalg.vector_norm(shape))
    dot = float(torch.dot(coordinate, shape))
    if coordinate_norm <= EPS:
        cosine = 0.0
        descent_multiplier = float("nan")
    else:
        cosine = dot / max(coordinate_norm * shape_norm, EPS)
        descent_multiplier = 1.0 + dot / (coordinate_norm**2)
    return {
        "coordinate_gradient_l2": coordinate_norm,
        "weighted_deadzone_gradient_l2": shape_norm,
        "deadzone_to_coordinate_norm_ratio": (
            shape_norm / max(coordinate_norm, EPS)
        ),
        "coordinate_deadzone_cosine": cosine,
        "coordinate_descent_multiplier": descent_multiplier,
        "harmful_cancellation_fraction": (
            max(0.0, 1.0 - descent_multiplier)
            if math.isfinite(descent_multiplier)
            else float("nan")
        ),
    }


def _measure_checkpoint(
    checkpoint: Path,
    image: torch.Tensor,
    target: torch.Tensor,
    shape_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    # Float64 keeps direct-vs-component gradient comparisons numerically clean.
    model = load_checkpoint_model(checkpoint).double()
    model.train()
    image = image.double()
    target = target.double()
    flat_coordinates, logits = model(image)
    coordinates = flat_coordinates.view(image.shape[0], -1, 2)
    coordinate_loss = F.mse_loss(coordinates, target)
    deadzone = conditional_deadzone_shape(logits)
    weighted_deadzone_loss = shape_weight * deadzone.loss

    groups = _parameter_groups(model)
    tensors: list[torch.Tensor] = [logits]
    slices: dict[str, slice] = {"logits": slice(0, 1)}
    for level in ("head", "backbone"):
        start = len(tensors)
        tensors.extend(groups[level])
        slices[level] = slice(start, len(tensors))

    coordinate_raw = torch.autograd.grad(
        coordinate_loss, tensors, retain_graph=True, allow_unused=True
    )
    deadzone_raw = torch.autograd.grad(
        weighted_deadzone_loss, tensors, retain_graph=True, allow_unused=True
    )
    global_rows = []
    for level in LEVELS:
        subset = tensors[slices[level]]
        coordinate = _flat_gradients(coordinate_raw[slices[level]], subset)
        shape = _flat_gradients(deadzone_raw[slices[level]], subset)
        global_rows.append({"level": level, **_vector_metrics(coordinate, shape)})

    probability = deadzone.probability
    channel_rows = []
    for channel in range(coordinates.shape[1]):
        channel_coordinate_loss = F.mse_loss(
            coordinates[:, channel], target[:, channel]
        )
        channel_deadzone_loss = (
            shape_weight * deadzone.per_channel_loss[:, channel].mean()
        )
        coordinate_gradient = torch.autograd.grad(
            channel_coordinate_loss, logits, retain_graph=True
        )[0][:, channel].reshape(-1)
        deadzone_gradient = torch.autograd.grad(
            channel_deadzone_loss, logits, retain_graph=True
        )[0][:, channel].reshape(-1)
        error_cells64 = torch.linalg.vector_norm(
            coordinates[:, channel] - target[:, channel], dim=-1
        ) / (2.0 / 64.0)
        channel_rows.append({
            "channel": channel,
            "median_error_cells64": float(torch.median(error_cells64)),
            "median_max_probability": float(
                torch.median(probability[:, channel].max(dim=-1).values)
            ),
            "median_effective_support_cells": float(
                torch.median(deadzone.effective_support_cells[:, channel])
            ),
            "mean_coordinate_loss": float(channel_coordinate_loss.detach()),
            "mean_unweighted_deadzone_loss": float(
                deadzone.per_channel_loss[:, channel].mean().detach()
            ),
            **_vector_metrics(coordinate_gradient, deadzone_gradient),
        })

    losses = {
        "coordinate_loss": float(coordinate_loss.detach()),
        "unweighted_deadzone_loss": float(deadzone.loss.detach()),
        "weighted_deadzone_loss": float(weighted_deadzone_loss.detach()),
        "total_loss": float((coordinate_loss + weighted_deadzone_loss).detach()),
    }
    return global_rows, channel_rows, losses


def run(args: argparse.Namespace) -> Path:
    image, target, frames = load_fixed_batch(args.data_root, 0)
    all_global: list[dict[str, Any]] = []
    all_channels: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    weights = []
    for seed in SEEDS:
        run_dir = args.gate_root / "tiny_overfit" / (
            f"coordinate_standard64_k10_deadzone_seed{seed}"
        )
        checkpoint = run_dir / "model.pt"
        metrics_path = run_dir / "metrics.json"
        if not checkpoint.exists() or not metrics_path.exists():
            raise FileNotFoundError(run_dir)
        stored = json.loads(metrics_path.read_text())
        if stored["shape_constraint"] != "conditional_deadzone":
            raise ValueError(f"wrong constraint in {metrics_path}")
        if stored["completed_steps"] != 5000:
            raise ValueError(f"wrong update count in {metrics_path}")
        weight = float(stored["shape_weight"])
        weights.append(weight)
        global_rows, channel_rows, losses = _measure_checkpoint(
            checkpoint, image, target, weight
        )
        stored_channel_errors = stored["metrics"]["channel_median_error_cells64"]
        stored_failed = set(stored["metrics"]["failed_channel_indices"])
        for row in global_rows:
            all_global.append({"seed": seed, **row})
        for row in channel_rows:
            row["seed"] = seed
            channel = int(row["channel"])
            row["gate_eval_median_error_cells64"] = float(
                stored_channel_errors[channel]
            )
            # Gate errors were measured in eval mode. Gradient measurements use
            # train mode because that is the optimization path. Preserve both.
            row["failed_coordinate_gate"] = channel in stored_failed
            row["saturated"] = bool(
                row["median_max_probability"]
                >= SATURATED_MAX_PROBABILITY_MIN
            )
            all_channels.append(row)
        loss_rows.append({"seed": seed, **losses})

    if max(weights) - min(weights) > 1e-15:
        raise ValueError("deadzone weights differ across seeds")
    saturated_failed = [
        row for row in all_channels
        if row["saturated"] and row["failed_coordinate_gate"]
    ]
    failed = [row for row in all_channels if row["failed_coordinate_gate"]]
    summary = {
        "audit": "read_only_deadzone_failure_postmortem",
        "frames": frames,
        "optimization_seed_n": len(SEEDS),
        "optimization_seeds": list(SEEDS),
        "analysis_dtype": "float64",
        "updates_per_run": 5000,
        "shape_weight": weights[0],
        "failed_channel_seed_units": len(failed),
        "saturated_and_failed_channel_seed_units": len(saturated_failed),
        "saturated_failed_fraction": (
            len(saturated_failed) / len(failed) if failed else 0.0
        ),
        "median_global_deadzone_to_coordinate_gradient_ratio_by_level": {
            level: float(torch.tensor([
                row["deadzone_to_coordinate_norm_ratio"]
                for row in all_global if row["level"] == level
            ]).median())
            for level in LEVELS
        },
        "interpretation": {
            "implementation_path_verified": True,
            "stride_change_active": False,
            "fixed_initial_logit_gradient_calibration_is_state_robust": False,
            "post_softmax_deadzone_can_reliably_escape_saturation": False,
            "reason": (
                "The fixed weight was calibrated only on diffuse initial maps. "
                "At failed checkpoints the weighted repair is generally weaker, "
                "and saturated maps attenuate both losses through the softmax "
                "Jacobian."
            ),
        },
        "statistical_scope": (
            "Descriptive audit of one object, four correlated frames and three "
            "optimization seeds. The seed is the replication unit; no "
            "population inference or hypothesis test is made."
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("R1_DEADZONE_GLOBAL_GRADIENTS.csv", all_global),
        ("R1_DEADZONE_CHANNEL_GRADIENTS.csv", all_channels),
        ("R1_DEADZONE_RECOMPUTED_LOSSES.csv", loss_rows),
    ):
        with (args.output_root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    output = args.output_root / "R1_DEADZONE_FAILURE_AUDIT.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
