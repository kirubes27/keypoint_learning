import sys
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.summarize_stage_a_k_sweep import EXPECTED_K, EXPECTED_SEEDS, summarize


def _rows(failed_by_k_seed: dict[tuple[int, int], list[int]]) -> list[dict]:
    rows = []
    for k in EXPECTED_K:
        for seed in EXPECTED_SEEDS:
            failed = failed_by_k_seed.get((k, seed), [])
            rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "passed": not failed,
                    "median_error_cells64": 0.05,
                    "max_channel_error_cells64": 0.1 if not failed else 2.0,
                    "failed_count": len(failed),
                    "failed_fraction": len(failed) / k,
                    "failed_channels": failed,
                    "runtime_seconds": 1.0,
                    "source": "synthetic",
                }
            )
    return rows


def test_summary_detects_stable_channel_issue() -> None:
    failures = {(10, seed): [3] for seed in EXPECTED_SEEDS}
    flags = summarize(_rows(failures))["preregistered_pattern_flags"]
    assert flags["stable_channel_or_target_ordering_issue"]


def test_summary_detects_k_dependent_failure_fraction() -> None:
    failures = {}
    for seed in EXPECTED_SEEDS:
        failures[(5, seed)] = []
        failures[(10, seed)] = [0]
        failures[(15, seed)] = [0, 1, 2]
        failures[(20, seed)] = list(range(6))
    flags = summarize(_rows(failures))["preregistered_pattern_flags"]
    assert flags["worsens_with_keypoint_count"]
