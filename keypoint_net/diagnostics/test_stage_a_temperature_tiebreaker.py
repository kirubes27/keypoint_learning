import sys
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_temperature_tiebreaker import summarize  # noqa: E402


def _rows(group_distance: float) -> list[dict]:
    rows = []
    for temperature, error, grad, probability in (
        (1.0, 2.0, 1.0, 0.001),
        (2.0, 0.8, 12.0, 0.020),
        (4.0, 0.7, 15.0, 0.030),
        (8.0, 1.0, 20.0, 0.040),
    ):
        rows.append({
            "assignment": "identity",
            "seed": 42,
            "frame": 0,
            "channel": 0 if group_distance <= 1.5 else 1,
            "temperature": temperature,
            "coordinate_error_cells64": error,
            "coordinate_gradient_l2": grad,
            "target_probability_mass_r1": probability,
            "argmax_target_distance_cells64": group_distance,
            "max_probability": 0.9,
        })
    return rows


def test_summary_detects_both_desaturation_subtypes() -> None:
    result = summarize(_rows(1.0) + _rows(3.0))
    assert result["near_target_saturation_supported"]
    assert result["far_wrong_peak_desaturation_supported"]

