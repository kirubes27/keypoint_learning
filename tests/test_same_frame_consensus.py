from __future__ import annotations

import inspect

import torch

from keypoint_net.frozen_nuisance_sensitivity import IMAGE_SIZE, NuisanceSpec
from keypoint_net.run_frozen_wobble_forensics import _infer
from keypoint_net.same_frame_consensus import (
    LOCKED_CONSENSUS_SPECS,
    align_view_probabilities_to_source,
    consensus_probabilities_from_view_logits,
    inverse_geometric_spec,
    spatial_expectation_from_probabilities,
)
from keypoint_net.same_frame_equivariance import (
    spatial_probabilities,
    warp_spatial_distribution,
)


def _gaussian_probability(*, size: int = 64) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float64),
        indexing="ij",
    )
    probability = torch.exp(-((xx - 0.27) ** 2 + (yy + 0.19) ** 2) / (2.0 * 0.055**2))
    probability = probability / probability.sum()
    return probability.view(1, 1, size, size)


def test_locked_consensus_has_five_symmetric_views() -> None:
    assert [spec.name for spec in LOCKED_CONSENSUS_SPECS] == [
        "identity",
        "translation_plus_quarter_pixel_xy",
        "translation_minus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
        "rotation_minus_quarter_degree_clockwise_image",
    ]
    assert sum(spec.kind == "identity" for spec in LOCKED_CONSENSUS_SPECS) == 1
    assert sorted(spec.dx_pixels for spec in LOCKED_CONSENSUS_SPECS if spec.kind == "translation") == [-0.25, 0.25]
    assert sorted(spec.clockwise_degrees for spec in LOCKED_CONSENSUS_SPECS if spec.kind == "rotation") == [-0.25, 0.25]


def test_identity_only_consensus_is_bit_exact_native_probability() -> None:
    generator = torch.Generator().manual_seed(20260815)
    logits = torch.randn((1, 3, 4, 17, 19), generator=generator)
    identity = (NuisanceSpec(name="identity", kind="identity"),)
    expected = spatial_probabilities(logits[0])
    consensus, aligned = consensus_probabilities_from_view_logits(logits, specs=identity)
    assert torch.equal(consensus, expected)
    assert torch.equal(aligned[0], expected)


def test_inverse_spec_exactly_negates_declared_geometry() -> None:
    translation = NuisanceSpec(name="forward_t", kind="translation", dx_pixels=7.0, dy_pixels=-3.0)
    rotation = NuisanceSpec(name="forward_r", kind="rotation", clockwise_degrees=4.0)
    inverse_t = inverse_geometric_spec(translation)
    inverse_r = inverse_geometric_spec(rotation)
    assert (inverse_t.dx_pixels, inverse_t.dy_pixels) == (-7.0, 3.0)
    assert inverse_r.clockwise_degrees == -4.0


def test_planted_translation_and_rotation_are_aligned_in_correct_direction() -> None:
    source = _gaussian_probability()
    source_xy = spatial_expectation_from_probabilities(source)
    specs = (
        NuisanceSpec(name="translation", kind="translation", dx_pixels=8.0, dy_pixels=-5.0),
        NuisanceSpec(name="rotation", kind="rotation", clockwise_degrees=5.0),
    )
    for spec in specs:
        view = warp_spatial_distribution(source, spec, input_coordinate_size=IMAGE_SIZE)
        recovered = align_view_probabilities_to_source(view, spec)
        recovered_xy = spatial_expectation_from_probabilities(recovered)
        wrong = warp_spatial_distribution(view, spec, input_coordinate_size=IMAGE_SIZE)
        wrong_xy = spatial_expectation_from_probabilities(wrong)
        correct_error_cells = torch.linalg.vector_norm(recovered_xy - source_xy).item() * 31.5
        wrong_error_cells = torch.linalg.vector_norm(wrong_xy - source_xy).item() * 31.5
        assert correct_error_cells < 0.03
        assert wrong_error_cells > correct_error_cells + 0.5


def test_perfectly_equivariant_views_recover_normalized_finite_consensus() -> None:
    source = _gaussian_probability()
    logits = []
    for spec in LOCKED_CONSENSUS_SPECS:
        view = source if spec.kind == "identity" else warp_spatial_distribution(source, spec)
        logits.append(torch.log(view.clamp_min(1e-100)))
    consensus, aligned = consensus_probabilities_from_view_logits(torch.stack(logits))
    assert consensus.shape == source.shape
    assert aligned.shape == (5, *source.shape)
    assert torch.isfinite(consensus).all()
    assert (consensus >= 0).all()
    assert torch.allclose(consensus.sum(dim=(-2, -1)), torch.ones((1, 1), dtype=torch.float64), atol=1e-14, rtol=0.0)
    centroid_error_cells = torch.linalg.vector_norm(
        spatial_expectation_from_probabilities(consensus)
        - spatial_expectation_from_probabilities(source)
    ).item() * 31.5
    assert centroid_error_cells < 0.002


def test_frozen_evaluator_keeps_native_as_default_decoder() -> None:
    assert inspect.signature(_infer).parameters["decoder"].default == "native"
