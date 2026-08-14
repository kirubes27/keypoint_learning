from __future__ import annotations

import torch

from keypoint_net.run_equivariance_gradient_decomposition import (
    _cosine,
    _relative_reconstruction_error,
    _summary,
)


def test_cosine_handles_zero_and_opposite_vectors() -> None:
    zero = torch.zeros(2, dtype=torch.float64)
    first = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert _cosine(zero, first) is None
    assert _cosine(first, -first) == -1.0


def test_signed_component_dots_reconstruct_total_conflict() -> None:
    auxiliary = torch.tensor([1.0, 0.0], dtype=torch.float64)
    components = {
        "against": torch.tensor([-3.0, 0.0], dtype=torch.float64),
        "with": torch.tensor([1.0, 0.0], dtype=torch.float64),
    }
    base = sum(components.values(), torch.zeros(2, dtype=torch.float64))
    result = _summary(base, auxiliary, components, {"only": auxiliary})
    assert result["base_auxiliary_dot"] == -2.0
    assert result["base_components"]["against"]["fraction_of_total_base_auxiliary_dot"] == 1.5
    assert result["base_components"]["with"]["fraction_of_total_base_auxiliary_dot"] == -0.5


def test_reconstruction_error_is_relative_to_observed_norm() -> None:
    observed = torch.tensor([3.0, 4.0], dtype=torch.float64)
    reconstructed = torch.tensor([3.0, 4.001], dtype=torch.float64)
    absolute, relative = _relative_reconstruction_error(observed, reconstructed)
    assert abs(absolute - 0.001) < 1e-12
    assert abs(relative - 0.0002) < 1e-12
