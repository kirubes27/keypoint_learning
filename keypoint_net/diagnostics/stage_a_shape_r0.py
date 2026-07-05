"""One-shot Stage-R0 calibration and positive-control measurement."""

from __future__ import annotations

import argparse
import json
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
    heatmap_shape_metrics,
    prediction_centered_js,
)


SIGMA_CELLS = 1.0
CALIBRATION_SEED = 42
NORM_FLOOR = 1e-12


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def calibrate_weight(data_root: Path) -> dict[str, Any]:
    image, target, frames = load_fixed_batch(data_root, shift=0)
    model = initial_model(CALIBRATION_SEED)
    flat_coordinates, logits = model(image)
    coordinates = flat_coordinates.view(image.shape[0], -1, 2)
    coordinate_loss = F.mse_loss(coordinates, target)
    shape = prediction_centered_js(logits, sigma_cells=SIGMA_CELLS)
    coordinate_gradient = torch.autograd.grad(
        coordinate_loss, logits, retain_graph=True
    )[0]
    shape_gradient = torch.autograd.grad(shape.loss, logits)[0]
    coordinate_norm = float(torch.linalg.vector_norm(coordinate_gradient))
    shape_norm = float(torch.linalg.vector_norm(shape_gradient))
    weight = coordinate_norm / max(shape_norm, NORM_FLOOR)
    return {
        "seed": CALIBRATION_SEED,
        "frames": frames,
        "coordinate_loss": float(coordinate_loss.detach()),
        "shape_loss": float(shape.loss.detach()),
        "coordinate_logit_gradient_l2": coordinate_norm,
        "shape_logit_gradient_l2": shape_norm,
        "gradient_norm_floor": NORM_FLOOR,
        "lambda_shape": weight,
        "formula": (
            "norm(d_coordinate_loss/d_logits) / "
            "max(norm(d_shape_loss/d_logits), 1e-12)"
        ),
    }


def positive_control_ranges(
    data_root: Path, positive_control_root: Path
) -> dict[str, Any]:
    image, _, frames = load_fixed_batch(data_root, shift=0)
    result: dict[str, Any] = {"frames": frames, "seeds": list(SEEDS)}
    for mode in ("train", "eval"):
        collected = {
            "max_probability": [],
            "normalized_entropy": [],
            "effective_support_cells": [],
        }
        for seed in SEEDS:
            checkpoint = (
                positive_control_root
                / f"heatmap_standard64_k10_seed{seed}"
                / "model.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = load_checkpoint_model(checkpoint)
            model.train(mode == "train")
            with torch.no_grad():
                _, logits = model(image)
                metrics = heatmap_shape_metrics(logits)
            for key in collected:
                collected[key].extend(metrics[key].flatten().cpu().tolist())
        result[mode] = {
            key: quantiles(values) for key, values in collected.items()
        }
    return result


def run(args: argparse.Namespace) -> Path:
    calibration = calibrate_weight(args.data_root)
    controls = positive_control_ranges(
        args.data_root, args.positive_control_root
    )
    payload = {
        "stage": "R0_shape_constraint_calibration",
        "architecture": "standard64",
        "num_keypoints": 10,
        "sigma_cells": SIGMA_CELLS,
        "shape_constraint": (
            "Jensen-Shannon divergence from predicted spatial probability "
            "to sigma-1 Gaussian centred at detached predicted expectation"
        ),
        "ground_truth_used_by_shape_constraint": False,
        "calibration": calibration,
        "successful_heatmap_positive_controls": controls,
        "frozen_r1_shape_ranges": {
            "per_channel_median_max_probability": [0.08, 0.30],
            "per_channel_median_effective_support_cells": [8.0, 32.0],
        },
        "statistical_scope": (
            "positive controls are descriptive: one object, four correlated "
            "frames, three optimization seeds; no inference"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "PRELAUNCH_R0_CALIBRATION.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError(f"frozen calibration differs from existing {output}")
    else:
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

