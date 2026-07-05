import sys
from pathlib import Path

import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.stage_a_gradient_audit import (  # noqa: E402
    logit_gradients,
    spatial_starvation,
    verify_coordinate_gradient,
)


def test_analytic_coordinate_gradient_matches_autograd() -> None:
    assert verify_coordinate_gradient() <= 1e-6


def test_wrong_one_hot_peak_starves_coordinate_target_gradient() -> None:
    logits = torch.full((1, 1, 8, 8), -20.0)
    logits[0, 0, 1, 1] = 20.0
    target = torch.tensor([[[1.0, 1.0]]])
    values = logit_gradients(logits, target)
    probability = values["probability"][0, 0]
    coordinate_gradient = values["coordinate_gradient"][0, 0].abs()
    heatmap_gradient = values["heatmap_gradient"][0, 0].abs()
    # Nearest target is the bottom-right cell.
    target_index = 7 * 8 + 7
    row = {
        "target_probability_mass_r1": float(probability[target_index]),
        "coordinate_target_gradient_fraction_r1": float(
            coordinate_gradient[target_index] / coordinate_gradient.sum()
        ),
        "heatmap_target_gradient_fraction_r1": float(
            heatmap_gradient[target_index] / heatmap_gradient.sum()
        ),
    }
    assert spatial_starvation(row)
