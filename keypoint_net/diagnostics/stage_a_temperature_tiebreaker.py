"""Read-only temperature sensitivity tiebreaker for the gradient audit."""

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


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.day45_supervised_control import CELL64_NORM  # noqa: E402
from diagnostics.stage_a_gradient_audit import (  # noqa: E402
    NUM_KEYPOINTS,
    SEEDS,
    TARGET_RADIUS_CELLS,
    _region_mask,
    _checkpoint_result,
    load_checkpoint_model,
    load_fixed_batch,
    logit_gradients,
    model_logits,
    write_json,
)


TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
NEAR_DISTANCE_MAX = 1.5


def measure_temperature(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    assignment: str,
    seed: int,
    frames: list[int],
    failed_channels: set[int],
) -> list[dict[str, Any]]:
    values = logit_gradients(logits, target, temperature=temperature)
    probability = values["probability"].detach()
    coordinate = values["coordinate"].detach()
    gradient = values["coordinate_gradient"].detach()
    _, channels, height, width = logits.shape
    rows = []
    for batch_index, frame in enumerate(frames):
        for channel in range(channels):
            if channel not in failed_channels:
                continue
            p = probability[batch_index, channel]
            grad = gradient[batch_index, channel]
            target_x = float((target[batch_index, channel, 0] + 1) * 0.5 * (width - 1))
            target_y = float((target[batch_index, channel, 1] + 1) * 0.5 * (height - 1))
            argmax = int(torch.argmax(p))
            argmax_y, argmax_x = divmod(argmax, width)
            target_mask = _region_mask(
                target_x, target_y, height, width, TARGET_RADIUS_CELLS, p.device
            )
            rows.append({
                "assignment": assignment,
                "seed": seed,
                "frame": int(frame),
                "channel": channel,
                "temperature": temperature,
                "coordinate_error_cells64": float(
                    torch.linalg.vector_norm(
                        coordinate[batch_index, channel] - target[batch_index, channel]
                    ) / CELL64_NORM
                ),
                "coordinate_gradient_l2": float(torch.linalg.vector_norm(grad)),
                "target_probability_mass_r1": float(p[target_mask].sum()),
                "argmax_target_distance_cells64": math.hypot(
                    argmax_x - target_x, argmax_y - target_y
                ),
                "max_probability": float(p.max()),
            })
    return rows


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        (row["assignment"], row["seed"], row["frame"], row["channel"]): row
        for row in rows if row["temperature"] == 1.0
    }
    for row in rows:
        base = baseline[(row["assignment"], row["seed"], row["frame"], row["channel"])]
        row["distance_group"] = (
            "near" if base["argmax_target_distance_cells64"] <= NEAR_DISTANCE_MAX else "far"
        )
        row["error_ratio_to_t1"] = row["coordinate_error_cells64"] / base["coordinate_error_cells64"]
        row["gradient_ratio_to_t1"] = (
            row["coordinate_gradient_l2"] / base["coordinate_gradient_l2"]
            if base["coordinate_gradient_l2"] else math.inf
        )
        row["target_probability_ratio_to_t1"] = (
            row["target_probability_mass_r1"] / base["target_probability_mass_r1"]
            if base["target_probability_mass_r1"] else math.inf
        )

    by_group: dict[str, Any] = {}
    for group in ("near", "far"):
        by_temperature = {}
        for temperature in TEMPERATURES:
            selected = [
                row for row in rows
                if row["distance_group"] == group and row["temperature"] == temperature
            ]
            if not selected:
                continue
            by_temperature[str(int(temperature))] = {
                "units": len(selected),
                "median_error_cells64": _median(selected, "coordinate_error_cells64"),
                "median_error_ratio_to_t1": _median(selected, "error_ratio_to_t1"),
                "median_gradient_ratio_to_t1": _median(selected, "gradient_ratio_to_t1"),
                "median_target_probability_ratio_to_t1": _median(
                    selected, "target_probability_ratio_to_t1"
                ),
                "median_max_probability": _median(selected, "max_probability"),
            }
        by_group[group] = by_temperature

    near_supported = any(
        item["median_error_ratio_to_t1"] <= 0.5
        and item["median_gradient_ratio_to_t1"] >= 10.0
        for key, item in by_group["near"].items() if key != "1"
    )
    far_supported = any(
        item["median_target_probability_ratio_to_t1"] >= 10.0
        and item["median_gradient_ratio_to_t1"] >= 10.0
        for key, item in by_group["far"].items() if key != "1"
    )
    return {
        "near_target_saturation_supported": near_supported,
        "far_wrong_peak_desaturation_supported": far_supported,
        "by_distance_group_and_temperature": by_group,
        "design": {
            "temperatures": list(TEMPERATURES),
            "near_argmax_target_distance_cells64_max": NEAR_DISTANCE_MAX,
            "weights_updated": False,
        },
        "statistical_scope": (
            "descriptive frozen-logit sensitivity; one object, four correlated frames, "
            "n=3 optimization seeds per assignment; no inferential claim"
        ),
    }


def run(args: argparse.Namespace) -> Path:
    batches = {
        "identity": load_fixed_batch(args.data_root, 0),
        "shift1": load_fixed_batch(args.data_root, 1),
    }
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
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
            result = _checkpoint_result(checkpoint)
            failed = {int(v) for v in result["metrics"]["failed_channel_indices"]}
            model = load_checkpoint_model(checkpoint)
            image, target, frames = batches[assignment]
            logits = model_logits(model, image).detach()
            for temperature in TEMPERATURES:
                rows.extend(measure_temperature(
                    logits,
                    target,
                    temperature=temperature,
                    assignment=assignment,
                    seed=seed,
                    frames=frames,
                    failed_channels=failed,
                ))
    result = summarize(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "A0_TEMPERATURE_TIEBREAKER_UNITS.csv"
    json_path = args.output_root / "A0_TEMPERATURE_TIEBREAKER_SUMMARY.json"
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

