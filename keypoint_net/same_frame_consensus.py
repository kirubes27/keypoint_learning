"""Frozen five-view consensus readout for same-frame wobble diagnosis.

The decoder does not train or update the detector.  It applies five locked tiny
geometric views to one frame, maps every predicted heatmap distribution back to
the original image coordinates, and averages equal probability mass.  This is
an inference-only diagnostic for view-specific readout instability; it is not a
material-identity or generalization solution.
"""

from __future__ import annotations

from typing import Sequence

import torch

try:
    from .frozen_nuisance_sensitivity import IMAGE_SIZE, NuisanceSpec
    from .same_frame_equivariance import spatial_probabilities, warp_spatial_distribution
except ImportError:  # Support direct historical script imports.
    from frozen_nuisance_sensitivity import IMAGE_SIZE, NuisanceSpec  # type: ignore
    from same_frame_equivariance import (  # type: ignore
        spatial_probabilities,
        warp_spatial_distribution,
    )


class SameFrameConsensusError(ValueError):
    """Raised when the frozen decoder semantics are violated."""


LOCKED_CONSENSUS_SPECS = (
    NuisanceSpec(name="identity", kind="identity"),
    NuisanceSpec(
        name="translation_plus_quarter_pixel_xy",
        kind="translation",
        dx_pixels=0.25,
        dy_pixels=0.25,
    ),
    NuisanceSpec(
        name="translation_minus_quarter_pixel_xy",
        kind="translation",
        dx_pixels=-0.25,
        dy_pixels=-0.25,
    ),
    NuisanceSpec(
        name="rotation_plus_quarter_degree_clockwise_image",
        kind="rotation",
        clockwise_degrees=0.25,
    ),
    NuisanceSpec(
        name="rotation_minus_quarter_degree_clockwise_image",
        kind="rotation",
        clockwise_degrees=-0.25,
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameFrameConsensusError(message)


def inverse_geometric_spec(spec: NuisanceSpec) -> NuisanceSpec:
    """Return the exact inverse of one locked geometric view."""
    if spec.kind == "identity":
        return NuisanceSpec(name=f"inverse_{spec.name}", kind="identity")
    if spec.kind == "translation":
        return NuisanceSpec(
            name=f"inverse_{spec.name}",
            kind="translation",
            dx_pixels=-float(spec.dx_pixels),
            dy_pixels=-float(spec.dy_pixels),
        )
    if spec.kind == "rotation":
        return NuisanceSpec(
            name=f"inverse_{spec.name}",
            kind="rotation",
            clockwise_degrees=-float(spec.clockwise_degrees),
        )
    raise SameFrameConsensusError(f"unsupported consensus view kind: {spec.kind}")


def align_view_probabilities_to_source(
    probabilities: torch.Tensor,
    spec: NuisanceSpec,
    *,
    input_coordinate_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Map a view's heatmap mass back into the unmodified source frame."""
    _require(probabilities.ndim == 4, "probabilities must be BxNxHxW")
    _require(bool(torch.isfinite(probabilities).all()), "probabilities are non-finite")
    _require(bool((probabilities >= 0).all()), "probabilities contain negative mass")
    mass = probabilities.sum(dim=(-2, -1), keepdim=True)
    _require(bool((mass > 0).all()), "every heatmap must contain positive mass")
    normalized = probabilities / mass
    if spec.kind == "identity":
        return normalized
    return warp_spatial_distribution(
        normalized,
        inverse_geometric_spec(spec),
        input_coordinate_size=input_coordinate_size,
    )


def consensus_probabilities_from_view_logits(
    view_logits: torch.Tensor,
    *,
    specs: Sequence[NuisanceSpec] = LOCKED_CONSENSUS_SPECS,
    input_coordinate_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equal-weight consensus and each source-aligned view distribution.

    ``view_logits`` is SxBxNxHxW in exactly the order of ``specs``.  The
    identity-only control returns the ordinary softmax tensor bit-for-bit.
    """
    _require(view_logits.ndim == 5, "view logits must be SxBxNxHxW")
    _require(len(specs) >= 1, "at least one view is required")
    _require(view_logits.shape[0] == len(specs), "view/spec count differs")
    _require(bool(torch.isfinite(view_logits).all()), "view logits are non-finite")
    probabilities = torch.stack(
        tuple(spatial_probabilities(view_logits[index]) for index in range(len(specs)))
    )
    if len(specs) == 1 and specs[0].kind == "identity":
        return probabilities[0], probabilities
    aligned = torch.stack(
        tuple(
            align_view_probabilities_to_source(
                probabilities[index],
                spec,
                input_coordinate_size=input_coordinate_size,
            )
            for index, spec in enumerate(specs)
        )
    )
    consensus = aligned.mean(dim=0)
    consensus = consensus / consensus.sum(dim=(-2, -1), keepdim=True)
    _require(bool(torch.isfinite(consensus).all()), "consensus is non-finite")
    _require(bool((consensus >= 0).all()), "consensus contains negative mass")
    return consensus, aligned


def spatial_expectation_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute endpoint-aligned normalized xy from BxNxHxW probability mass."""
    _require(probabilities.ndim == 4, "probabilities must be BxNxHxW")
    _require(bool(torch.isfinite(probabilities).all()), "probabilities are non-finite")
    mass = probabilities.sum(dim=(-2, -1), keepdim=True)
    _require(bool((mass > 0).all()), "every heatmap must contain positive mass")
    normalized = probabilities / mass
    height, width = normalized.shape[-2:]
    y_coordinates = torch.linspace(
        -1.0, 1.0, height, device=normalized.device, dtype=normalized.dtype
    )
    x_coordinates = torch.linspace(
        -1.0, 1.0, width, device=normalized.device, dtype=normalized.dtype
    )
    yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    return torch.stack(
        (
            (normalized * xx).sum(dim=(-2, -1)),
            (normalized * yy).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )


__all__ = [
    "LOCKED_CONSENSUS_SPECS",
    "SameFrameConsensusError",
    "align_view_probabilities_to_source",
    "consensus_probabilities_from_view_logits",
    "inverse_geometric_spec",
    "spatial_expectation_from_probabilities",
]
