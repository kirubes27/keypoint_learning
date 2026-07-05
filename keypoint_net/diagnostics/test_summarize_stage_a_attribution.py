import json
import sys
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.summarize_stage_a_attribution import SEEDS, summarize


def _write_run(root: Path, name: str, failed_channels: list[int], shift: int) -> None:
    run = root / "tiny_overfit" / name
    run.mkdir(parents=True)
    mapping = [(channel + shift) % 10 for channel in range(10)]
    failed_targets = [mapping[channel] for channel in failed_channels]
    payload = {
        "passed": not failed_channels,
        "runtime_seconds": 1.0,
        "metrics": {
            "median_error_cells64": 0.05,
            "max_channel_median_error_cells64": 0.1 if not failed_channels else 2.0,
            "failed_channel_indices": failed_channels,
            "failed_physical_target_indices": failed_targets,
        },
    }
    (run / "metrics.json").write_text(json.dumps(payload))


def test_summary_distinguishes_target_following_from_channel_identity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    attribution = tmp_path / "attribution"
    for seed in SEEDS:
        _write_run(
            baseline,
            f"coordinate_standard64_k10_seed{seed}",
            [3, 6, 9],
            0,
        )
        # Under shift=1, physical targets 3/6/9 are numerical channels 2/5/8.
        _write_run(
            attribution,
            f"coordinate_standard64_k10_shift1_seed{seed}",
            [2, 5, 8],
            1,
        )
        _write_run(
            attribution,
            f"heatmap_standard64_k10_seed{seed}",
            [],
            0,
        )
    flags = summarize(baseline, attribution)["preregistered_pattern_flags"]
    assert flags["failure_follows_physical_targets_3_6_9"]
    assert not flags["failure_stays_numerical_channels_3_6_9"]
    assert flags["heatmap_supervision_passes_at_least_2_of_3"]
