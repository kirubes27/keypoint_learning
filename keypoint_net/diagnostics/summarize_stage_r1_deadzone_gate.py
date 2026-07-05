"""Aggregate the three frozen conditional-dead-zone R1 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
RUN_PATTERN = "coordinate_standard64_k10_deadzone_seed{seed}"
SHAPE_WEIGHT = 8.723808430294485e-06


def load_run(root: Path, seed: int) -> dict[str, Any]:
    path = root / "tiny_overfit" / RUN_PATTERN.format(seed=seed) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("run_scope") != "authoritative_r1_tiny_gate":
        raise ValueError(f"non-authoritative fallback R1 artifact: {path}")
    if not payload.get("gate_authoritative"):
        raise ValueError(f"fallback R1 gate not authoritative: {path}")
    if payload.get("shape_constraint") != "conditional_deadzone":
        raise ValueError(f"wrong shape constraint in {path}")
    if abs(float(payload.get("shape_weight", 0.0)) - SHAPE_WEIGHT) > 1e-15:
        raise ValueError(f"wrong shape weight in {path}")
    metrics = payload["metrics"]
    probe = metrics["r1_probe"]
    return {
        "seed": seed,
        "passed": bool(payload["passed"]),
        "coordinate_gate_pass": bool(metrics["coordinate_gate_pass"]),
        "shape_gate_pass": bool(probe["shape_gate_pass"]),
        "counterfactual_gradient_gate_pass": bool(
            probe["counterfactual_gradient_gate_pass"]
        ),
        "completed_steps": int(payload["completed_steps"]),
        "median_error_cells64": float(metrics["median_error_cells64"]),
        "worst_channel_error_cells64": float(
            metrics["max_channel_median_error_cells64"]
        ),
        "failed_channels": [int(v) for v in metrics["failed_channel_indices"]],
        "failed_physical_targets": [
            int(v) for v in metrics["failed_physical_target_indices"]
        ],
        "run_gradient_ratio": float(
            probe["run_median_counterfactual_gradient_final_initial_ratio"]
        ),
        "min_channel_gradient_ratio": min(
            float(v)
            for v in probe[
                "per_channel_median_counterfactual_gradient_final_initial_ratio"
            ]
        ),
        "min_channel_max_probability": min(
            float(v) for v in probe["per_channel_median_max_probability"]
        ),
        "max_channel_max_probability": max(
            float(v) for v in probe["per_channel_median_max_probability"]
        ),
        "min_channel_support": min(
            float(v) for v in probe["per_channel_median_effective_support_cells"]
        ),
        "max_channel_support": max(
            float(v) for v in probe["per_channel_median_effective_support_cells"]
        ),
        "min_channel_dominant_mass_r2": min(
            float(v) for v in probe["per_channel_median_dominant_mass_r2"]
        ),
        "runtime_seconds": float(payload["runtime_seconds"]),
        "source": str(path),
    }


def summarize(root: Path) -> dict[str, Any]:
    rows = [load_run(root, seed) for seed in SEEDS]
    failure_frequency = {
        str(target): sum(target in row["failed_physical_targets"] for row in rows)
        for target in range(10)
    }
    persistent_targets = [
        target for target in range(10) if failure_frequency[str(target)] >= 2
    ]
    pass_count = sum(int(row["passed"]) for row in rows)
    passed = pass_count >= 2 and not persistent_targets
    return {
        "gate": "R1_conditional_deadzone_fallback_tiny_coordinate_overfit",
        "passed": passed,
        "pass_count": pass_count,
        "required": (
            "at least 2/3 joint seed passes and no target fails in >=2/3 seeds"
        ),
        "persistent_failed_physical_targets": persistent_targets,
        "physical_target_failure_frequency_out_of_3": failure_frequency,
        "per_seed": rows,
        "frozen_recipe": {
            "architecture": "standard64",
            "num_keypoints": 10,
            "steps_max": 5000,
            "shape_constraint": "conditional_deadzone",
            "shape_weight": SHAPE_WEIGHT,
            "max_probability_range": [0.08, 0.30],
            "effective_support_range": [8.0, 32.0],
            "dominant_mass_r2_min": 0.70,
        },
        "statistical_scope": (
            "descriptive one-object/four-correlated-frame gate; n=3 optimization "
            "seeds; seed is the replication unit; no hypothesis test or "
            "population inference"
        ),
        "decision": (
            "fallback R1 passes; R2 convergence pilot may start"
            if passed
            else "fallback R1 fails; stop this instrument design and review"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    json_path = args.root / "R1_DEADZONE_GATE_SUMMARY.json"
    csv_path = args.root / "R1_DEADZONE_GATE_RUNS.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    with csv_path.open("w", newline="") as handle:
        fields = list(result["per_seed"][0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["per_seed"]:
            serialized = dict(row)
            serialized["failed_channels"] = json.dumps(row["failed_channels"])
            serialized["failed_physical_targets"] = json.dumps(
                row["failed_physical_targets"]
            )
            writer.writerow(serialized)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
