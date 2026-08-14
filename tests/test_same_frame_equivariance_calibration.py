from __future__ import annotations

import torch

from keypoint_net.model import PhaseAModel
from keypoint_net.run_same_frame_equivariance_calibration import (
    _coefficient_receipt,
    _training_state_sha256,
)


def test_historical_seed42_initial_states_are_exactly_reproducible() -> None:
    expected = {
        False: "4c5e199e3f8caacc81582463f452e3058adc25fb8adfb501336b0156dcdcd82f",
        True: "64b93ff13f2ade10e1389a9028a774604add9358a0ed21275c7e0018d51330ae",
    }
    for inverse, expected_hash in expected.items():
        torch.manual_seed(42)
        model = PhaseAModel(
            num_keypoints=10,
            base_channels=32,
            temperature=1.0,
            padding_mode="reflect",
            operator_type="shared_affine",
            learn_inverse_operator=inverse,
            heatmap_res=64,
        )
        assert _training_state_sha256(model) == expected_hash


def test_coefficient_rule_uses_six_cell_median_and_cap() -> None:
    cells = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for recipe, local in zip(("task55_clean", "task80_assisted"), (ratios[:3], ratios[3:])):
        for seed, ratio in zip((42, 43, 44), local):
            cells.append({
                "recipe": recipe,
                "seed": seed,
                "auxiliary_to_base_gradient_ratio": ratio,
            })
    receipt = _coefficient_receipt(cells)
    assert receipt["median_unscaled_auxiliary_to_base_ratio_all_six_cells"] == 0.35
    assert receipt["coefficient"] == 0.1 / 0.35
    assert receipt["scaled_median_contribution_all_six_cells"] == 0.1


def test_coefficient_rule_caps_large_requested_weight() -> None:
    cells = [
        {"recipe": recipe, "seed": seed, "auxiliary_to_base_gradient_ratio": 0.01}
        for recipe in ("task55_clean", "task80_assisted")
        for seed in (42, 43, 44)
    ]
    receipt = _coefficient_receipt(cells)
    assert receipt["coefficient"] == 0.5
    assert receipt["status"] == "calibrated_at_safety_cap"
