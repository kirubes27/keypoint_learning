import json
import sys
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.summarize_representative_coordinate_pilot import summarize  # noqa: E402


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "coordinate_standard64_k10_seed41"
    run.mkdir(parents=True)
    _write(run / "config.json", {
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
    })
    _write(run / "training_summary.json", {
        "stop_reason": "validation_plateau",
        "test_evaluated": False,
    })
    one = {
        "median_of_channel_medians_cells64": 0.4,
        "p90_error_cells64": 1.2,
        "on_mask_fraction": 0.99,
    }
    _write(run / "best_validation_metrics.json", {
        "unaugmented": dict(one),
        "fixed_augmented": dict(one),
    })
    _write(run / "best_coordinate_path_probe.json", {
        "inaccurate_saturated_channel_indices": [],
        "collapsed_gradient_channel_indices": [],
    })
    return run


def test_representative_pilot_summary_requires_every_check(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    result = summarize(run)
    assert result["viable_for_three_seed_confirmation"]
    probe = json.loads((run / "best_coordinate_path_probe.json").read_text())
    probe["collapsed_gradient_channel_indices"] = [3]
    _write(run / "best_coordinate_path_probe.json", probe)
    result = summarize(run)
    assert not result["viable_for_three_seed_confirmation"]
    assert not result["checks"]["no_collapsed_gradient_channel"]
