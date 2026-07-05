import json
import sys
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.summarize_stage_r1_shape_gate import SEEDS, summarize  # noqa: E402


def _write_run(root: Path, seed: int, *, passed: bool, failed: list[int]) -> None:
    run = root / "tiny_overfit" / f"coordinate_standard64_k10_shapejs_seed{seed}"
    run.mkdir(parents=True)
    payload = {
        "passed": passed,
        "run_scope": "authoritative_r1_tiny_gate",
        "gate_authoritative": True,
        "completed_steps": 5000,
        "runtime_seconds": 1.0,
        "metrics": {
            "coordinate_gate_pass": passed,
            "median_error_cells64": 0.05 if passed else 1.0,
            "max_channel_median_error_cells64": 0.1 if passed else 2.0,
            "failed_channel_indices": failed,
            "failed_physical_target_indices": failed,
            "r1_probe": {
                "shape_gate_pass": True,
                "counterfactual_gradient_gate_pass": True,
                "run_median_counterfactual_gradient_final_initial_ratio": 0.5,
                "per_channel_median_counterfactual_gradient_final_initial_ratio": [0.5] * 10,
                "per_channel_median_max_probability": [0.15] * 10,
                "per_channel_median_effective_support_cells": [17.0] * 10,
            },
        },
    }
    (run / "metrics.json").write_text(json.dumps(payload))


def test_r1_summary_requires_two_joint_passes_and_no_persistent_target(tmp_path: Path) -> None:
    _write_run(tmp_path, SEEDS[0], passed=True, failed=[])
    _write_run(tmp_path, SEEDS[1], passed=True, failed=[])
    _write_run(tmp_path, SEEDS[2], passed=False, failed=[3])
    result = summarize(tmp_path)
    assert result["passed"]
    assert result["pass_count"] == 2


def test_r1_summary_rejects_persistent_target_even_if_run_flags_pass(tmp_path: Path) -> None:
    _write_run(tmp_path, SEEDS[0], passed=True, failed=[3])
    _write_run(tmp_path, SEEDS[1], passed=True, failed=[3])
    _write_run(tmp_path, SEEDS[2], passed=True, failed=[])
    result = summarize(tmp_path)
    assert not result["passed"]
    assert result["persistent_failed_physical_targets"] == [3]

