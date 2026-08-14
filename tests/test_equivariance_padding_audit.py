from __future__ import annotations

import numpy as np
import torch

from keypoint_net.frozen_nuisance_sensitivity import NuisanceSpec
from keypoint_net.run_equivariance_padding_audit import (
    _compare_policies,
    _outer_cell_mass,
    _warp_with_padding,
)
from keypoint_net.same_frame_equivariance import warp_spatial_distribution


def test_zero_padding_path_is_exact_candidate_path() -> None:
    generator = torch.Generator().manual_seed(52)
    probability = torch.rand((2, 3, 19, 19), generator=generator, dtype=torch.float64)
    probability /= probability.sum(dim=(-2, -1), keepdim=True)
    for spec in (
        NuisanceSpec(name="translation", kind="translation", dx_pixels=0.25, dy_pixels=0.25),
        NuisanceSpec(name="rotation", kind="rotation", clockwise_degrees=0.25),
    ):
        observed, _ = _warp_with_padding(probability, spec, padding_mode="zeros")
        assert torch.equal(observed, warp_spatial_distribution(probability, spec))


def test_outer_cell_mass_counts_each_corner_once() -> None:
    probability = torch.zeros((1, 1, 4, 4), dtype=torch.float64)
    probability[0, 0, 0, 0] = 0.25
    probability[0, 0, 0, 1] = 0.25
    probability[0, 0, 1, 0] = 0.25
    probability[0, 0, 1, 1] = 0.25
    assert _outer_cell_mass(probability).item() == 0.75


def test_decision_rule_fails_when_reflection_hides_worst_event() -> None:
    zero = np.arange(1800, dtype=np.float64).reshape(180, 10)
    reflection = zero.copy()
    residual = np.zeros_like(zero)
    residual[179, 9] = 10.0
    reflection[179, 9] = -1.0
    result = _compare_policies(zero, reflection, residual)
    assert result["worst_residual_event"]["reflection_js_rank_percentile"] < 0.5
    assert result["decision_changing"] is True


def test_decision_rule_passes_identical_rankings() -> None:
    zero = np.arange(1800, dtype=np.float64).reshape(180, 10)
    residual = zero.copy()
    result = _compare_policies(zero, zero.copy(), residual)
    assert result["top20_overlap_count"] == 20
    assert result["decision_changing"] is False
