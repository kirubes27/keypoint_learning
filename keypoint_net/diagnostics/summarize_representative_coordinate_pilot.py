"""Summarize the validation-only representative coordinate pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    training = json.loads((run_dir / "training_summary.json").read_text())
    validation = json.loads((run_dir / "best_validation_metrics.json").read_text())
    probe = json.loads((run_dir / "best_coordinate_path_probe.json").read_text())
    plain = validation["unaugmented"]
    augmented = validation["fixed_augmented"]
    checks = {
        "validation_plateau": training["stop_reason"] == "validation_plateau",
        "both_medians_at_most_0p50": max(
            plain["median_of_channel_medians_cells64"],
            augmented["median_of_channel_medians_cells64"],
        ) <= 0.50,
        "both_p90_at_most_1p50": max(
            plain["p90_error_cells64"], augmented["p90_error_cells64"]
        ) <= 1.50,
        "both_on_mask_at_least_0p95": min(
            plain["on_mask_fraction"], augmented["on_mask_fraction"]
        ) >= 0.95,
        "no_inaccurate_saturated_channel": not probe[
            "inaccurate_saturated_channel_indices"
        ],
        "no_collapsed_gradient_channel": not probe[
            "collapsed_gradient_channel_indices"
        ],
        "test_untouched": not bool(training["test_evaluated"]),
    }
    expected = {
        "seed": 41,
        "architecture": "standard64",
        "num_keypoints": 10,
        "supervision": "coordinate",
        "shape_constraint": "none",
        "min_epochs": 1000,
        "max_epochs": 3000,
        "eval_every": 25,
        "plateau_patience_epochs": 400,
        "relative_improvement": 0.01,
    }
    configuration_matches = all(config[key] == value for key, value in expected.items())
    checks["configuration_matches_lock"] = configuration_matches
    return {
        "gate": "representative_coordinate_pilot_validation_only",
        "viable_for_three_seed_confirmation": all(checks.values()),
        "checks": checks,
        "configuration": {key: config[key] for key in expected},
        "training": training,
        "best_validation": validation,
        "best_coordinate_path_probe": probe,
        "test_evaluated": False,
        "decision": (
            "freeze recipe and authorize three-seed confirmation"
            if all(checks.values())
            else "do not launch three-seed confirmation; review failed checks"
        ),
        "statistical_scope": (
            "Descriptive one-object pilot with one optimization seed; "
            "validation frames belong to one correlated cyclic orbit; no "
            "error bars, hypothesis tests, or population inference."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    result = summarize(args.run_dir)
    output = args.output or (args.run_dir.parent.parent / "REPRESENTATIVE_PILOT_SUMMARY.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
