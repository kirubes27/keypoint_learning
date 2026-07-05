"""Audit logit gradients behind the Stage-A coordinate-overfit failure.

This is a read-only diagnostic: it reconstructs deterministic initializations,
loads frozen checkpoints, and differentiates coordinate MSE and Gaussian
heatmap CE with respect to the same 64x64 logits. It never updates model
parameters or source artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor, spatial_softmax  # noqa: E402
from diagnostics.day45_supervised_control import (  # noqa: E402
    CELL64_NORM,
    gaussian_target_distribution,
)
from diagnostics.stage_a_supervised_control import (  # noqa: E402
    load_problem,
    make_dataset,
    seed_everything,
)


SEEDS = (42, 43, 44)
NUM_KEYPOINTS = 10
TARGET_RADIUS_CELLS = 1.0
TARGET_PROB_MAX = 1e-3
TARGET_GRAD_FRACTION_RATIO_MAX = 0.1
SUPPORT_FRACTION = 0.75
REJECT_FRACTION = 0.50
COORD_FINAL_INITIAL_SUPPORT_MAX = 0.10
COORD_FINAL_INITIAL_REJECT_MIN = 0.50
HEATMAP_FINAL_INITIAL_SUPPORT_MIN = 0.50


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def spatial_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def logit_gradients(
    logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> dict[str, torch.Tensor]:
    """Return per-frame/channel probabilities and per-unit logit gradients.

    The coordinate gradient corresponds to a per-unit mean squared error over
    x/y. The heatmap gradient corresponds to per-unit Gaussian-target CE.
    """
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    probability = torch.softmax(flat / temperature, dim=-1)
    grid = spatial_grid(
        height, width, device=logits.device, dtype=logits.dtype
    )
    coordinate = torch.einsum("bks,sd->bkd", probability, grid)
    difference = coordinate - target
    centered = grid[None, None, :, :] - coordinate[:, :, None, :]
    coordinate_gradient = (
        probability
        * torch.sum(centered * difference[:, :, None, :], dim=-1)
        / temperature
    )
    gaussian = gaussian_target_distribution(target, height, width)
    heatmap_gradient = probability - gaussian
    return {
        "probability": probability,
        "coordinate": coordinate,
        "coordinate_gradient": coordinate_gradient,
        "heatmap_gradient": heatmap_gradient,
    }


def verify_coordinate_gradient() -> float:
    generator = torch.Generator().manual_seed(20260705)
    logits = torch.randn(2, 3, 5, 4, generator=generator, requires_grad=True)
    target = torch.empty(2, 3, 2).uniform_(-0.8, 0.8, generator=generator)
    coordinate = spatial_softmax(logits, temperature=1.0)
    loss = F.mse_loss(coordinate, target)
    autograd = torch.autograd.grad(loss, logits)[0].reshape(2, 3, -1)
    analytic = logit_gradients(logits.detach(), target)["coordinate_gradient"]
    analytic = analytic / (logits.shape[0] * logits.shape[1])
    return float(torch.max(torch.abs(autograd - analytic)))


def _region_mask(
    center_x: float,
    center_y: float,
    height: int,
    width: int,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2).reshape(-1)


def measure_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    frames: list[int],
    state: str,
    assignment: str,
    seed: int,
    failed_channels: set[int],
) -> list[dict[str, Any]]:
    values = logit_gradients(logits, target)
    probability = values["probability"].detach()
    coordinate = values["coordinate"].detach()
    coordinate_gradient = values["coordinate_gradient"].detach()
    heatmap_gradient = values["heatmap_gradient"].detach()
    _, channels, height, width = logits.shape
    rows: list[dict[str, Any]] = []
    for batch_index, frame in enumerate(frames):
        for channel in range(channels):
            p = probability[batch_index, channel]
            coord_grad = coordinate_gradient[batch_index, channel]
            heat_grad = heatmap_gradient[batch_index, channel]
            target_x = float((target[batch_index, channel, 0] + 1.0) * 0.5 * (width - 1))
            target_y = float((target[batch_index, channel, 1] + 1.0) * 0.5 * (height - 1))
            argmax = int(torch.argmax(p))
            argmax_y, argmax_x = divmod(argmax, width)
            target_mask = _region_mask(
                target_x,
                target_y,
                height,
                width,
                TARGET_RADIUS_CELLS,
                p.device,
            )
            argmax_mask = _region_mask(
                float(argmax_x),
                float(argmax_y),
                height,
                width,
                TARGET_RADIUS_CELLS,
                p.device,
            )
            coord_abs = torch.abs(coord_grad)
            heat_abs = torch.abs(heat_grad)
            coord_total = float(coord_abs.sum())
            heat_total = float(heat_abs.sum())
            entropy = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
            error = float(
                torch.linalg.vector_norm(
                    coordinate[batch_index, channel] - target[batch_index, channel]
                )
                / CELL64_NORM
            )
            row = {
                "state": state,
                "assignment": assignment,
                "seed": seed,
                "frame": int(frame),
                "channel": channel,
                "frozen_failed_channel": channel in failed_channels,
                "coordinate_error_cells64": error,
                "argmax_target_distance_cells64": math.hypot(
                    argmax_x - target_x, argmax_y - target_y
                ),
                "max_probability": float(p.max()),
                "normalized_entropy": entropy / math.log(height * width),
                "effective_support_cells": math.exp(entropy),
                "target_probability_mass_r1": float(p[target_mask].sum()),
                "argmax_probability_mass_r1": float(p[argmax_mask].sum()),
                "coordinate_gradient_l2": float(torch.linalg.vector_norm(coord_grad)),
                "heatmap_gradient_l2": float(torch.linalg.vector_norm(heat_grad)),
                "coordinate_target_gradient_fraction_r1": (
                    float(coord_abs[target_mask].sum()) / coord_total if coord_total else 0.0
                ),
                "coordinate_argmax_gradient_fraction_r1": (
                    float(coord_abs[argmax_mask].sum()) / coord_total if coord_total else 0.0
                ),
                "heatmap_target_gradient_fraction_r1": (
                    float(heat_abs[target_mask].sum()) / heat_total if heat_total else 0.0
                ),
                "heatmap_argmax_gradient_fraction_r1": (
                    float(heat_abs[argmax_mask].sum()) / heat_total if heat_total else 0.0
                ),
            }
            rows.append(row)
    return rows


def _checkpoint_result(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)["result"]


def load_checkpoint_model(path: Path) -> KeypointExtractor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = KeypointExtractor(
        num_keypoints=NUM_KEYPOINTS,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    )
    model.load_state_dict(payload["extractor_state_dict"], strict=True)
    model.train()
    return model


def make_args(data_root: Path, shift: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=data_root,
        split_json=None,
        object="engineers_hammer_vray",
        num_keypoints=NUM_KEYPOINTS,
        target_shift=shift,
        center_x=255.49998435893767,
        center_y=255.50001568508694,
        roll_sign=1,
    )


def load_fixed_batch(data_root: Path, shift: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    args = make_args(data_root, shift)
    problem = load_problem(args)
    indices = list(problem["split"].train[:4])
    dataset = make_dataset(problem, indices, augment=False, seed=42)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    return batch["image"], batch["target"], [int(v) for v in batch["frame"]]


def initial_model(seed: int) -> KeypointExtractor:
    seed_everything(seed)
    model = KeypointExtractor(
        num_keypoints=NUM_KEYPOINTS,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    )
    model.train()
    return model


def model_logits(model: KeypointExtractor, image: torch.Tensor) -> torch.Tensor:
    # The model returns both soft-argmax coordinates and the raw heatmap logits.
    _, logits = model(image)
    return logits


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def add_matched_ratios(rows: list[dict[str, Any]]) -> None:
    initial = {
        (row["assignment"], row["seed"], row["frame"], row["channel"]): row
        for row in rows
        if row["state"] == "initial"
    }
    for row in rows:
        row["coordinate_gradient_final_initial_ratio"] = None
        row["heatmap_gradient_final_initial_ratio"] = None
        if row["state"] != "final_coordinate":
            continue
        key = (row["assignment"], row["seed"], row["frame"], row["channel"])
        base = initial[key]
        row["coordinate_gradient_final_initial_ratio"] = (
            row["coordinate_gradient_l2"] / base["coordinate_gradient_l2"]
            if base["coordinate_gradient_l2"]
            else math.inf
        )
        row["heatmap_gradient_final_initial_ratio"] = (
            row["heatmap_gradient_l2"] / base["heatmap_gradient_l2"]
            if base["heatmap_gradient_l2"]
            else math.inf
        )


def spatial_starvation(row: dict[str, Any]) -> bool:
    heat_fraction = float(row["heatmap_target_gradient_fraction_r1"])
    return bool(
        row["target_probability_mass_r1"] <= TARGET_PROB_MAX
        and row["coordinate_target_gradient_fraction_r1"]
        <= TARGET_GRAD_FRACTION_RATIO_MAX * heat_fraction
    )


def summarize(rows: list[dict[str, Any]], autograd_max_error: float) -> dict[str, Any]:
    by_assignment: dict[str, Any] = {}
    for assignment in ("identity", "shift1"):
        final = [
            row for row in rows
            if row["state"] == "final_coordinate"
            and row["assignment"] == assignment
            and row["frozen_failed_channel"]
        ]
        if not final:
            raise RuntimeError(f"no frozen failed units for {assignment}")
        signature = [spatial_starvation(row) for row in final]
        saturated = [
            row["max_probability"] >= 0.90 or row["normalized_entropy"] <= 0.25
            for row in final
        ]
        by_assignment[assignment] = {
            "failed_frame_channel_units": len(final),
            "spatial_starvation_count": int(sum(signature)),
            "spatial_starvation_fraction": float(np.mean(signature)),
            "saturated_wrong_peak_fraction": float(np.mean(saturated)),
            "median_target_probability_mass_r1": _median(final, "target_probability_mass_r1"),
            "median_coordinate_target_gradient_fraction_r1": _median(
                final, "coordinate_target_gradient_fraction_r1"
            ),
            "median_heatmap_target_gradient_fraction_r1": _median(
                final, "heatmap_target_gradient_fraction_r1"
            ),
            "median_coordinate_gradient_final_initial_ratio": _median(
                final, "coordinate_gradient_final_initial_ratio"
            ),
            "median_heatmap_gradient_final_initial_ratio": _median(
                final, "heatmap_gradient_final_initial_ratio"
            ),
        }

    support = all(
        item["spatial_starvation_fraction"] >= SUPPORT_FRACTION
        and item["median_coordinate_gradient_final_initial_ratio"]
        <= COORD_FINAL_INITIAL_SUPPORT_MAX
        and item["median_heatmap_gradient_final_initial_ratio"]
        >= HEATMAP_FINAL_INITIAL_SUPPORT_MIN
        for item in by_assignment.values()
    )
    reject = any(
        item["spatial_starvation_fraction"] < REJECT_FRACTION
        or item["median_coordinate_gradient_final_initial_ratio"]
        > COORD_FINAL_INITIAL_REJECT_MIN
        for item in by_assignment.values()
    )
    verdict = "supported" if support else "not_supported" if reject else "mixed"
    pooled_failed = [
        row for row in rows
        if row["state"] == "final_coordinate" and row["frozen_failed_channel"]
    ]
    saturated_fraction = float(np.mean([
        row["max_probability"] >= 0.90 or row["normalized_entropy"] <= 0.25
        for row in pooled_failed
    ]))
    heatmap_results = [
        row for row in rows if row["state"] == "final_heatmap"
    ]
    heatmap_channel_medians = {
        str(channel): float(np.median([
            row["coordinate_error_cells64"]
            for row in heatmap_results if row["channel"] == channel
        ]))
        for channel in range(NUM_KEYPOINTS)
    }
    positive_control_pass = max(heatmap_channel_medians.values()) <= 0.20
    if not positive_control_pass:
        verdict = "invalid_positive_control"
    return {
        "verdict": verdict,
        "saturated_wrong_peak_submechanism": saturated_fraction >= SUPPORT_FRACTION,
        "pooled_saturated_wrong_peak_fraction": saturated_fraction,
        "autograd_analytic_max_abs_error": autograd_max_error,
        "autograd_gate_pass": autograd_max_error <= 1e-6,
        "positive_control": {
            "heatmap_trained_channel_median_error_cells64": heatmap_channel_medians,
            "all_channels_at_or_below_0.20": positive_control_pass,
        },
        "by_coordinate_assignment": by_assignment,
        "thresholds": {
            "target_probability_mass_r1_max": TARGET_PROB_MAX,
            "coordinate_to_heatmap_target_gradient_fraction_ratio_max": TARGET_GRAD_FRACTION_RATIO_MAX,
            "support_fraction_min": SUPPORT_FRACTION,
            "coordinate_final_initial_ratio_support_max": COORD_FINAL_INITIAL_SUPPORT_MAX,
            "heatmap_final_initial_ratio_support_min": HEATMAP_FINAL_INITIAL_SUPPORT_MIN,
            "reject_fraction_below": REJECT_FRACTION,
            "coordinate_final_initial_ratio_reject_above": COORD_FINAL_INITIAL_REJECT_MIN,
        },
        "statistical_scope": (
            "descriptive one-object/four-correlated-frame mechanism audit; "
            "n=3 optimization seeds per condition; no error bars, test, or population inference"
        ),
    }


def run(args: argparse.Namespace) -> Path:
    autograd_error = verify_coordinate_gradient()
    if autograd_error > 1e-6:
        raise RuntimeError(f"analytic gradient mismatch: {autograd_error}")
    rows: list[dict[str, Any]] = []
    batches = {
        "identity": load_fixed_batch(args.data_root, 0),
        "shift1": load_fixed_batch(args.data_root, 1),
    }
    for seed in SEEDS:
        init = initial_model(seed)
        for assignment, (image, target, frames) in batches.items():
            logits = model_logits(init, image)
            rows.extend(measure_logits(
                logits,
                target,
                frames=frames,
                state="initial",
                assignment=assignment,
                seed=seed,
                failed_channels=set(),
            ))

        sources = (
            (
                "identity",
                args.baseline_root / f"coordinate_standard64_k10_seed{seed}" / "model.pt",
            ),
            (
                "shift1",
                args.attribution_root / f"coordinate_standard64_k10_shift1_seed{seed}" / "model.pt",
            ),
        )
        for assignment, checkpoint in sources:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            result = _checkpoint_result(checkpoint)
            failed = {int(v) for v in result["metrics"]["failed_channel_indices"]}
            model = load_checkpoint_model(checkpoint)
            image, target, frames = batches[assignment]
            rows.extend(measure_logits(
                model_logits(model, image),
                target,
                frames=frames,
                state="final_coordinate",
                assignment=assignment,
                seed=seed,
                failed_channels=failed,
            ))

        heatmap_checkpoint = (
            args.attribution_root / f"heatmap_standard64_k10_seed{seed}" / "model.pt"
        )
        if not heatmap_checkpoint.exists():
            raise FileNotFoundError(heatmap_checkpoint)
        heatmap_model = load_checkpoint_model(heatmap_checkpoint)
        image, target, frames = batches["identity"]
        rows.extend(measure_logits(
            model_logits(heatmap_model, image),
            target,
            frames=frames,
            state="final_heatmap",
            assignment="identity",
            seed=seed,
            failed_channels=set(),
        ))

    add_matched_ratios(rows)
    result = summarize(rows, autograd_error)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "A0_GRADIENT_AUDIT_UNITS.csv"
    json_path = args.output_root / "A0_GRADIENT_AUDIT_SUMMARY.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--attribution-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

