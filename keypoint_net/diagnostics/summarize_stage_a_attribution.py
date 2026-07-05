"""Summarize K=10 target-permutation versus heatmap-supervision attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
K = 10
CONDITIONS = {
    "baseline_coordinate": ("coordinate_standard64_k10_seed{seed}", 0),
    "coordinate_shift1": ("coordinate_standard64_k10_shift1_seed{seed}", 1),
    "heatmap_identity": ("heatmap_standard64_k10_seed{seed}", 0),
}


def _load_one(root: Path, pattern: str, seed: int, shift: int) -> dict[str, Any]:
    path = root / "tiny_overfit" / pattern.format(seed=seed) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    metrics = payload["metrics"]
    failed_channels = [int(value) for value in metrics["failed_channel_indices"]]
    mapping = [int((channel + shift) % K) for channel in range(K)]
    failed_targets = [mapping[channel] for channel in failed_channels]
    stored_targets = metrics.get("failed_physical_target_indices")
    if stored_targets is not None and [int(v) for v in stored_targets] != failed_targets:
        raise ValueError(f"physical-target mapping mismatch in {path}")
    return {
        "seed": seed,
        "passed": bool(payload["passed"]),
        "median_error_cells64": float(metrics["median_error_cells64"]),
        "worst_error_cells64": float(metrics["max_channel_median_error_cells64"]),
        "failed_channels": failed_channels,
        "failed_physical_targets": failed_targets,
        "failed_fraction": len(failed_channels) / K,
        "runtime_seconds": float(payload["runtime_seconds"]),
        "source": str(path),
    }


def frequency(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        str(index): sum(index in row[field] for row in rows)
        for index in range(K)
    }


def summarize(baseline_root: Path, attribution_root: Path) -> dict[str, Any]:
    rows_by_condition = {}
    for condition, (pattern, shift) in CONDITIONS.items():
        root = baseline_root if condition == "baseline_coordinate" else attribution_root
        rows = [_load_one(root, pattern, seed, shift) for seed in SEEDS]
        for row in rows:
            row["condition"] = condition
        rows_by_condition[condition] = rows

    by_condition = {}
    for condition, rows in rows_by_condition.items():
        by_condition[condition] = {
            "pass_count": sum(int(row["passed"]) for row in rows),
            "channel_failure_frequency_out_of_3": frequency(rows, "failed_channels"),
            "physical_target_failure_frequency_out_of_3": frequency(
                rows, "failed_physical_targets"
            ),
            "raw_runs": rows,
        }

    permuted = by_condition["coordinate_shift1"]
    heatmap = by_condition["heatmap_identity"]
    hard_targets = (3, 6, 9)
    physical_follows = all(
        permuted["physical_target_failure_frequency_out_of_3"][str(index)] >= 2
        for index in hard_targets
    )
    numerical_stays = all(
        permuted["channel_failure_frequency_out_of_3"][str(index)] >= 2
        for index in hard_targets
    )
    heatmap_rescue = heatmap["pass_count"] >= 2
    heatmap_same_targets = all(
        heatmap["physical_target_failure_frequency_out_of_3"][str(index)] >= 2
        for index in hard_targets
    )
    return {
        "design": {
            "k": K,
            "seeds": list(SEEDS),
            "coordinate_permutation": "channel c receives physical target (c+1) mod 10",
            "heatmap_supervision": "Gaussian target CE, sigma 8 input pixels",
            "unchanged_gate": "median <=0.10 cell64 and every channel <=0.20 cell64",
        },
        "by_condition": by_condition,
        "preregistered_pattern_flags": {
            "failure_follows_physical_targets_3_6_9": physical_follows,
            "failure_stays_numerical_channels_3_6_9": numerical_stays,
            "heatmap_supervision_passes_at_least_2_of_3": heatmap_rescue,
            "heatmap_retains_physical_targets_3_6_9": heatmap_same_targets,
        },
        "statistical_scope": (
            "descriptive one-object/four-frame diagnostic; n=3 optimization seeds per "
            "condition; no error bars, hypothesis test or population inference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--attribution-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.baseline_root, args.attribution_root)
    json_path = args.attribution_root / "A0_ATTRIBUTION_SUMMARY.json"
    csv_path = args.attribution_root / "A0_ATTRIBUTION_RUNS.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "condition", "seed", "passed", "median_error_cells64",
            "worst_error_cells64", "failed_fraction", "failed_channels",
            "failed_physical_targets", "runtime_seconds", "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITIONS:
            for row in result["by_condition"][condition]["raw_runs"]:
                serialized = dict(row)
                serialized["failed_channels"] = json.dumps(row["failed_channels"])
                serialized["failed_physical_targets"] = json.dumps(
                    row["failed_physical_targets"]
                )
                writer.writerow(serialized)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
