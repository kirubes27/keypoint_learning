"""Summarize the preregistered Stage-A keypoint-count attribution sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_K = (5, 10, 15, 20)
EXPECTED_SEEDS = (42, 43, 44)


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for k in EXPECTED_K:
        for seed in EXPECTED_SEEDS:
            path = (
                root
                / "tiny_overfit"
                / f"coordinate_standard64_k{k}_seed{seed}"
                / "metrics.json"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            if int(payload["num_keypoints"]) != k:
                raise ValueError(f"K mismatch in {path}")
            metrics = payload["metrics"]
            channel = [float(value) for value in metrics["channel_median_error_cells64"]]
            if len(channel) != k:
                raise ValueError(f"expected {k} channel errors in {path}")
            failed = [index for index, value in enumerate(channel) if value > 0.20]
            if failed != [int(value) for value in metrics["failed_channel_indices"]]:
                raise ValueError(f"failed-channel list inconsistent in {path}")
            rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "passed": bool(payload["passed"]),
                    "median_error_cells64": float(metrics["median_error_cells64"]),
                    "max_channel_error_cells64": float(
                        metrics["max_channel_median_error_cells64"]
                    ),
                    "failed_count": len(failed),
                    "failed_fraction": len(failed) / k,
                    "failed_channels": failed,
                    "runtime_seconds": float(payload["runtime_seconds"]),
                    "source": str(path),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    median_failure_fractions = []
    stable_indices: dict[str, list[int]] = {}
    for k in EXPECTED_K:
        subset = [row for row in rows if row["k"] == k]
        failure_sets = [set(row["failed_channels"]) for row in subset]
        stable = sorted(set.intersection(*failure_sets)) if failure_sets else []
        stable_indices[str(k)] = stable
        frequencies = {
            str(channel): sum(channel in failed for failed in failure_sets)
            for channel in range(k)
        }
        median_fraction = float(np.median([row["failed_fraction"] for row in subset]))
        median_failure_fractions.append(median_fraction)
        by_k[str(k)] = {
            "pass_count": sum(int(row["passed"]) for row in subset),
            "median_failed_fraction": median_fraction,
            "median_worst_channel_error_cells64": float(
                np.median([row["max_channel_error_cells64"] for row in subset])
            ),
            "channels_failed_in_all_three_seeds": stable,
            "channel_failure_frequency_out_of_3": frequencies,
        }

    nondecreasing_steps = sum(
        median_failure_fractions[index + 1] >= median_failure_fractions[index]
        for index in range(len(median_failure_fractions) - 1)
    )
    capacity_pattern = bool(
        nondecreasing_steps >= 2
        and median_failure_fractions[-1] - median_failure_fractions[0] >= 0.20
    )
    stable_index_pattern = any(stable_indices[str(k)] for k in EXPECTED_K)
    k5_fails_majority = by_k["5"]["pass_count"] <= 1
    core_pattern = bool(
        k5_fails_majority
        and median_failure_fractions[-1] - median_failure_fractions[0] < 0.20
    )
    stochastic_pattern = bool(
        any(row["failed_count"] for row in rows)
        and not stable_index_pattern
        and not capacity_pattern
    )
    return {
        "design": {
            "k": list(EXPECTED_K),
            "seeds": list(EXPECTED_SEEDS),
            "resolution": "standard native 64x64 heatmaps (/8 encoder features)",
            "threshold": "every channel <= 0.20 cell64 and median <= 0.10 cell64",
        },
        "by_k": by_k,
        "preregistered_pattern_flags": {
            "stable_channel_or_target_ordering_issue": stable_index_pattern,
            "worsens_with_keypoint_count": capacity_pattern,
            "core_readout_issue_already_at_k5": core_pattern,
            "stochastic_saturation": stochastic_pattern,
        },
        "interpretation_limits": (
            "descriptive one-object tiny-overfit diagnostic; n=3 optimization seeds per K; "
            "no population inference or semantic-keypoint claim"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.root)
    summary = summarize(rows)
    output_json = args.root / "A0_K_SWEEP_SUMMARY.json"
    output_csv = args.root / "A0_K_SWEEP_RUNS.csv"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    with output_csv.open("w", newline="") as handle:
        fieldnames = [
            "k", "seed", "passed", "median_error_cells64",
            "max_channel_error_cells64", "failed_count", "failed_fraction",
            "failed_channels", "runtime_seconds", "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["failed_channels"] = json.dumps(row["failed_channels"])
            writer.writerow(serialized)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
