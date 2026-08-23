"""RGB-conditioned distribution transport for the free-logit capability gate.

The module deliberately contains no dataset, mask, geometry, model, checkpoint,
or operator dependency.  It turns two RGB frames into a fixed sparse local
correspondence field and evaluates an objective on arbitrary spatial logits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


class MaterialTransportError(ValueError):
    """Raised when an input violates the frozen free-logit gate."""


@dataclass(frozen=True)
class MaterialTransportConfig:
    image_size: int = 512
    grid_size: int = 64
    patch_size: int = 35
    search_radius_cells: int = 4
    correspondence_temperature: float = 0.05
    minimum_patch_rms: float = 1.0e-6
    minimum_motion_cells: float = 0.125
    invalid_site_cost: float = 2.0
    descriptor_chunk_size: int = 256

    def validate(self) -> None:
        if self.image_size < 8:
            raise MaterialTransportError("image_size must be at least eight")
        if self.grid_size < 2:
            raise MaterialTransportError("grid_size must be at least two")
        if self.patch_size <= 1 or self.patch_size % 2 != 1:
            raise MaterialTransportError("patch_size must be an odd integer greater than one")
        if self.patch_size > self.image_size:
            raise MaterialTransportError("patch_size exceeds image_size")
        if self.search_radius_cells < 1:
            raise MaterialTransportError("search_radius_cells must be positive")
        if not math.isfinite(self.correspondence_temperature) or self.correspondence_temperature <= 0.0:
            raise MaterialTransportError("correspondence_temperature must be finite and positive")
        if not math.isfinite(self.minimum_patch_rms) or self.minimum_patch_rms <= 0.0:
            raise MaterialTransportError("minimum_patch_rms must be finite and positive")
        if not math.isfinite(self.minimum_motion_cells) or self.minimum_motion_cells <= 0.0:
            raise MaterialTransportError("minimum_motion_cells must be finite and positive")
        if not math.isfinite(self.invalid_site_cost) or self.invalid_site_cost <= 0.0:
            raise MaterialTransportError("invalid_site_cost must be finite and positive")
        if self.descriptor_chunk_size < 1:
            raise MaterialTransportError("descriptor_chunk_size must be positive")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class MaterialTransportWeights:
    transport: float = 1.0
    site: float = 0.5
    concentration: float = 0.25
    overlap: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise MaterialTransportError(f"{name} weight must be finite and non-negative")

    def as_dict(self) -> dict[str, float]:
        self.validate()
        return asdict(self)


def _require_floating_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise MaterialTransportError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise MaterialTransportError(f"{name} contains non-finite values")


def local_candidate_layout(
    config: MaterialTransportConfig,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return target indices, validity, xy offsets, and the self-offset column."""

    config.validate()
    grid = config.grid_size
    radius = config.search_radius_cells
    offsets = torch.tensor(
        [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
        dtype=torch.long,
        device=device,
    )
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    source_x = xx.reshape(-1, 1)
    source_y = yy.reshape(-1, 1)
    target_x = source_x + offsets[:, 0]
    target_y = source_y + offsets[:, 1]
    valid = (target_x >= 0) & (target_x < grid) & (target_y >= 0) & (target_y < grid)
    indices = target_y.clamp(0, grid - 1) * grid + target_x.clamp(0, grid - 1)
    self_columns = torch.nonzero(
        (offsets[:, 0] == 0) & (offsets[:, 1] == 0), as_tuple=False
    ).reshape(-1)
    if self_columns.numel() != 1:
        raise MaterialTransportError("candidate layout does not contain exactly one self offset")
    return indices.long(), valid.bool(), offsets, int(self_columns.item())


def opposite_offset_columns(offsets_xy: torch.Tensor) -> torch.Tensor:
    """Map every xy offset column to the column containing its negation."""

    if offsets_xy.ndim != 2 or offsets_xy.shape[1] != 2:
        raise MaterialTransportError("offsets_xy must have shape (candidates,2)")
    values = [(int(row[0]), int(row[1])) for row in offsets_xy.detach().cpu().tolist()]
    lookup = {value: index for index, value in enumerate(values)}
    if len(lookup) != len(values):
        raise MaterialTransportError("offset columns are not unique")
    try:
        opposite = [lookup[(-dx, -dy)] for dx, dy in values]
    except KeyError as error:
        raise MaterialTransportError("offset layout is not symmetric") from error
    return torch.tensor(opposite, dtype=torch.long, device=offsets_xy.device)


def extract_endpoint_grid_descriptors(
    image: torch.Tensor,
    config: MaterialTransportConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample centred, normalized RGB patches at endpoint-aligned grid centres.

    Returns `(descriptor, valid, rms)` with shapes `(grid**2, 3*patch**2)`,
    `(grid**2,)`, and `(grid**2,)`.
    """

    config.validate()
    _require_floating_tensor(image, name="image")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.shape != (3, config.image_size, config.image_size):
        raise MaterialTransportError(
            f"image must have shape {(3, config.image_size, config.image_size)}"
        )
    image = image.contiguous()
    device = image.device
    dtype = image.dtype
    grid_axis_px = torch.linspace(
        0.0,
        float(config.image_size - 1),
        config.grid_size,
        dtype=dtype,
        device=device,
    )
    yy, xx = torch.meshgrid(grid_axis_px, grid_axis_px, indexing="ij")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    radius = (config.patch_size - 1) // 2
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    patch_y, patch_x = torch.meshgrid(offsets, offsets, indexing="ij")
    patch_offsets = torch.stack((patch_x, patch_y), dim=-1)
    boundary_valid = (
        (centres[:, 0] >= radius)
        & (centres[:, 0] <= config.image_size - 1 - radius)
        & (centres[:, 1] >= radius)
        & (centres[:, 1] <= config.image_size - 1 - radius)
    )

    descriptors: list[torch.Tensor] = []
    rms_values: list[torch.Tensor] = []
    for start in range(0, centres.shape[0], config.descriptor_chunk_size):
        stop = min(start + config.descriptor_chunk_size, centres.shape[0])
        chunk_centres = centres[start:stop]
        sample_px = chunk_centres[:, None, None, :] + patch_offsets[None, :, :, :]
        sample_grid = sample_px * (2.0 / float(config.image_size - 1)) - 1.0
        patches = F.grid_sample(
            image.unsqueeze(0).expand(stop - start, -1, -1, -1),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        channel_mean = patches.mean(dim=(-2, -1), keepdim=True)
        centred = patches - channel_mean
        flattened = centred.reshape(stop - start, -1)
        rms = flattened.square().mean(dim=1).sqrt()
        norm = flattened.norm(dim=1).clamp_min(torch.finfo(dtype).eps)
        descriptors.append(flattened / norm[:, None])
        rms_values.append(rms)

    descriptor = torch.cat(descriptors, dim=0)
    rms = torch.cat(rms_values, dim=0)
    valid = boundary_valid & (rms >= config.minimum_patch_rms)
    descriptor = torch.where(valid[:, None], descriptor, torch.zeros_like(descriptor))
    if not bool(torch.isfinite(descriptor).all()) or not bool(torch.isfinite(rms).all()):
        raise MaterialTransportError("descriptor extraction produced non-finite values")
    return descriptor, valid, rms


def local_similarity_field(
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    candidate_index: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    candidate_column_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute fixed local centred-cosine similarities without a large gather."""

    _require_floating_tensor(source_descriptor, name="source_descriptor")
    _require_floating_tensor(target_descriptor, name="target_descriptor")
    if source_descriptor.shape != target_descriptor.shape or source_descriptor.ndim != 2:
        raise MaterialTransportError("source and target descriptor shapes differ")
    cells = source_descriptor.shape[0]
    if source_valid.shape != (cells,) or target_valid.shape != (cells,):
        raise MaterialTransportError("descriptor validity shape differs")
    if candidate_index.ndim != 2 or candidate_index.shape[0] != cells:
        raise MaterialTransportError("candidate index shape differs")
    if candidate_valid.shape != candidate_index.shape:
        raise MaterialTransportError("candidate validity shape differs")
    columns = candidate_index.shape[1]
    if candidate_column_order is None:
        order = torch.arange(columns, device=candidate_index.device)
    else:
        order = candidate_column_order.to(device=candidate_index.device, dtype=torch.long)
        if order.shape != (columns,) or not torch.equal(
            torch.sort(order).values, torch.arange(columns, device=order.device)
        ):
            raise MaterialTransportError("candidate_column_order is not a permutation")
    scores = torch.full(
        (cells, columns),
        -torch.inf,
        dtype=source_descriptor.dtype,
        device=source_descriptor.device,
    )
    for column_value in order:
        column = int(column_value.item())
        targets = candidate_index[:, column]
        valid = candidate_valid[:, column] & source_valid & target_valid[targets]
        similarity = torch.sum(source_descriptor * target_descriptor[targets], dim=1)
        scores[:, column] = torch.where(valid, similarity, scores[:, column])
    return scores


def conditional_from_similarity(
    similarity: torch.Tensor,
    candidate_valid: torch.Tensor,
    self_column: int,
    config: MaterialTransportConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert similarities to mass-preserving rows with invalid self-loops."""

    config.validate()
    if similarity.ndim != 2 or candidate_valid.shape != similarity.shape:
        raise MaterialTransportError("similarity and candidate validity shapes differ")
    finite = torch.isfinite(similarity) & candidate_valid
    row_valid = finite.any(dim=1)
    masked = torch.where(finite, similarity, torch.full_like(similarity, -torch.inf))
    safe = masked.clone()
    safe[~row_valid, self_column] = 0.0
    probability = torch.softmax(safe / config.correspondence_temperature, dim=1)
    probability = torch.where(finite, probability, torch.zeros_like(probability))
    probability[~row_valid] = 0.0
    probability[~row_valid, self_column] = 1.0
    row_sum = probability.sum(dim=1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1.0e-6, rtol=1.0e-6):
        raise MaterialTransportError("conditional rows do not preserve unit mass")
    valid_count = finite.sum(dim=1)
    return probability, row_valid, valid_count


def directional_site_cost(
    forward_probability: torch.Tensor,
    reverse_probability: torch.Tensor,
    forward_row_valid: torch.Tensor,
    forward_valid_count: torch.Tensor,
    candidate_index: torch.Tensor,
    candidate_valid: torch.Tensor,
    offsets_xy: torch.Tensor,
    config: MaterialTransportConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ambiguity, reciprocal, and active-motion costs for source rows."""

    config.validate()
    if forward_probability.shape != reverse_probability.shape:
        raise MaterialTransportError("forward and reverse conditional shapes differ")
    if candidate_index.shape != forward_probability.shape or candidate_valid.shape != candidate_index.shape:
        raise MaterialTransportError("candidate layout shape differs from conditionals")
    cells, columns = forward_probability.shape
    if forward_row_valid.shape != (cells,) or forward_valid_count.shape != (cells,):
        raise MaterialTransportError("row metadata shape differs")
    if offsets_xy.shape != (columns, 2):
        raise MaterialTransportError("offset shape differs")
    epsilon = torch.finfo(forward_probability.dtype).eps
    entropy = -torch.sum(
        forward_probability * torch.log(forward_probability.clamp_min(epsilon)), dim=1
    )
    denominator = torch.log(forward_valid_count.to(forward_probability.dtype).clamp_min(2.0))
    ambiguity = torch.where(forward_valid_count > 1, entropy / denominator, torch.zeros_like(entropy))

    opposite = opposite_offset_columns(offsets_xy)
    reciprocal = torch.zeros(cells, dtype=forward_probability.dtype, device=forward_probability.device)
    for column in range(columns):
        target = candidate_index[:, column]
        return_probability = reverse_probability[target, opposite[column]]
        reciprocal = reciprocal + torch.where(
            candidate_valid[:, column],
            forward_probability[:, column] * return_probability,
            torch.zeros_like(return_probability),
        )
    reciprocal_cost = 1.0 - reciprocal.clamp(0.0, 1.0)

    displacement = torch.linalg.vector_norm(offsets_xy.to(forward_probability.dtype), dim=1)
    expected_displacement = torch.sum(forward_probability * displacement[None, :], dim=1)
    inactivity = F.relu(config.minimum_motion_cells - expected_displacement) / config.minimum_motion_cells

    cost = (ambiguity + reciprocal_cost + inactivity) / 3.0
    cost = torch.where(
        forward_row_valid,
        cost,
        torch.full_like(cost, config.invalid_site_cost),
    )
    components = {
        "ambiguity": ambiguity,
        "reciprocal_cost": reciprocal_cost,
        "expected_displacement_cells": expected_displacement,
        "inactivity": inactivity,
        "row_valid": forward_row_valid,
    }
    if not bool(torch.isfinite(cost).all()):
        raise MaterialTransportError("site cost contains non-finite values")
    return cost, components


def build_bidirectional_field(
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    config: MaterialTransportConfig,
    *,
    verify_column_order: bool = False,
) -> dict[str, torch.Tensor | bool]:
    """Build independently computed forward and reverse sparse RGB fields."""

    candidate_index, candidate_valid, offsets_xy, self_column = local_candidate_layout(
        config, device=source_descriptor.device
    )
    forward_similarity = local_similarity_field(
        source_descriptor,
        target_descriptor,
        source_valid,
        target_valid,
        candidate_index,
        candidate_valid,
    )
    reverse_similarity = local_similarity_field(
        target_descriptor,
        source_descriptor,
        target_valid,
        source_valid,
        candidate_index,
        candidate_valid,
    )
    order_exact = True
    if verify_column_order:
        reverse_order = torch.arange(
            candidate_index.shape[1] - 1,
            -1,
            -1,
            device=candidate_index.device,
        )
        forward_reordered = local_similarity_field(
            source_descriptor,
            target_descriptor,
            source_valid,
            target_valid,
            candidate_index,
            candidate_valid,
            candidate_column_order=reverse_order,
        )
        reverse_reordered = local_similarity_field(
            target_descriptor,
            source_descriptor,
            target_valid,
            source_valid,
            candidate_index,
            candidate_valid,
            candidate_column_order=reverse_order,
        )
        order_exact = bool(
            torch.equal(forward_similarity, forward_reordered)
            and torch.equal(reverse_similarity, reverse_reordered)
        )
        if not order_exact:
            raise MaterialTransportError("candidate-column reversal changed a similarity field")

    forward_probability, forward_row_valid, forward_count = conditional_from_similarity(
        forward_similarity, candidate_valid, self_column, config
    )
    reverse_probability, reverse_row_valid, reverse_count = conditional_from_similarity(
        reverse_similarity, candidate_valid, self_column, config
    )
    forward_cost, forward_components = directional_site_cost(
        forward_probability,
        reverse_probability,
        forward_row_valid,
        forward_count,
        candidate_index,
        candidate_valid,
        offsets_xy,
        config,
    )
    reverse_cost, reverse_components = directional_site_cost(
        reverse_probability,
        forward_probability,
        reverse_row_valid,
        reverse_count,
        candidate_index,
        candidate_valid,
        offsets_xy,
        config,
    )
    output: dict[str, torch.Tensor | bool] = {
        "candidate_index": candidate_index,
        "candidate_valid": candidate_valid,
        "offsets_xy": offsets_xy,
        "forward_similarity": forward_similarity,
        "reverse_similarity": reverse_similarity,
        "forward_probability": forward_probability,
        "reverse_probability": reverse_probability,
        "forward_row_valid": forward_row_valid,
        "reverse_row_valid": reverse_row_valid,
        "forward_site_cost": forward_cost,
        "reverse_site_cost": reverse_cost,
        "column_order_reversal_exact": order_exact,
    }
    for name, value in forward_components.items():
        output[f"forward_{name}"] = value
    for name, value in reverse_components.items():
        output[f"reverse_{name}"] = value
    return output


def sparse_transport(
    source_probability: torch.Tensor,
    conditional: torch.Tensor,
    candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Transport `(edges,channels,cells)` probability through sparse rows."""

    _require_floating_tensor(source_probability, name="source_probability")
    _require_floating_tensor(conditional, name="conditional")
    if source_probability.ndim != 3 or conditional.ndim != 3:
        raise MaterialTransportError("source_probability and conditional must be rank three")
    edges, channels, cells = source_probability.shape
    if conditional.shape[:2] != (edges, cells):
        raise MaterialTransportError("conditional leading shape differs")
    if candidate_index.shape != conditional.shape[1:]:
        raise MaterialTransportError("candidate index shape differs")
    columns = conditional.shape[2]
    flat_index = candidate_index.reshape(1, cells * columns).expand(edges, -1)
    transported_channels: list[torch.Tensor] = []
    for channel in range(channels):
        mass = source_probability[:, channel, :, None] * conditional
        transported = torch.zeros(
            (edges, cells),
            dtype=source_probability.dtype,
            device=source_probability.device,
        )
        transported.scatter_add_(1, flat_index, mass.reshape(edges, cells * columns))
        transported_channels.append(transported)
    result = torch.stack(transported_channels, dim=1)
    result_sum = result.sum(dim=2)
    source_sum = source_probability.sum(dim=2)
    if not torch.allclose(result_sum, source_sum, atol=2.0e-5, rtol=2.0e-5):
        raise MaterialTransportError("sparse transport did not preserve heatmap mass")
    return result


def jensen_shannon_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return one bounded Jensen-Shannon value per leading item."""

    _require_floating_tensor(first, name="first")
    _require_floating_tensor(second, name="second")
    if first.shape != second.shape or first.ndim < 1:
        raise MaterialTransportError("Jensen-Shannon inputs must have matching shapes")
    epsilon = torch.finfo(first.dtype).eps
    first = first / first.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    second = second / second.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    midpoint = 0.5 * (first + second)
    first_kl = torch.sum(first * (torch.log(first.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))), dim=-1)
    second_kl = torch.sum(second * (torch.log(second.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))), dim=-1)
    return 0.5 * (first_kl + second_kl)


def channel_overlap_loss(probability: torch.Tensor) -> torch.Tensor:
    """Mean pairwise spatial overlap for `(frames,channels,cells)` distributions."""

    _require_floating_tensor(probability, name="probability")
    if probability.ndim != 3 or probability.shape[1] < 2:
        raise MaterialTransportError("probability must contain at least two channels")
    gram = torch.einsum("fcn,fdn->fcd", probability, probability)
    rows, columns = torch.triu_indices(
        probability.shape[1], probability.shape[1], offset=1, device=probability.device
    )
    return gram[:, rows, columns].mean()


def weighted_site_loss(probability: torch.Tensor, site_cost: torch.Tensor) -> torch.Tensor:
    """Heatmap-weighted site cost; exposed for the uniform-gradient semantic test."""

    _require_floating_tensor(probability, name="probability")
    _require_floating_tensor(site_cost, name="site_cost")
    if probability.ndim != 3 or site_cost.shape != (probability.shape[0], probability.shape[2]):
        raise MaterialTransportError("site cost shape differs from probability")
    return torch.sum(probability * site_cost[:, None, :], dim=2).mean()


def cyclic_material_transport_objective(
    logits: torch.Tensor,
    forward_conditional: torch.Tensor,
    reverse_conditional: torch.Tensor,
    forward_site_cost: torch.Tensor,
    reverse_site_cost: torch.Tensor,
    candidate_index: torch.Tensor,
    *,
    weights: MaterialTransportWeights = MaterialTransportWeights(),
) -> dict[str, torch.Tensor]:
    """Evaluate the locked full-orbit free-logit objective."""

    weights.validate()
    _require_floating_tensor(logits, name="logits")
    if logits.ndim == 4:
        frames, channels, height, width = logits.shape
        if height != width:
            raise MaterialTransportError("spatial logits must be square")
        flat_logits = logits.reshape(frames, channels, height * width)
    elif logits.ndim == 3:
        frames, channels, _ = logits.shape
        flat_logits = logits
    else:
        raise MaterialTransportError("logits must have rank three or four")
    if frames < 2 or channels < 2:
        raise MaterialTransportError("the objective requires at least two frames and channels")
    probability = torch.softmax(flat_logits, dim=2)
    adjacent_probability = torch.roll(probability, shifts=-1, dims=0)
    forward_prediction = sparse_transport(probability, forward_conditional, candidate_index)
    reverse_prediction = sparse_transport(adjacent_probability, reverse_conditional, candidate_index)
    transport = 0.5 * (
        jensen_shannon_divergence(forward_prediction, adjacent_probability).mean()
        + jensen_shannon_divergence(reverse_prediction, probability).mean()
    )
    site = 0.5 * (
        weighted_site_loss(probability, forward_site_cost)
        + weighted_site_loss(adjacent_probability, reverse_site_cost)
    )
    epsilon = torch.finfo(probability.dtype).eps
    concentration = (
        -torch.sum(probability * torch.log(probability.clamp_min(epsilon)), dim=2)
        / math.log(float(probability.shape[2]))
    ).mean()
    overlap = channel_overlap_loss(probability)
    total = (
        weights.transport * transport
        + weights.site * site
        + weights.concentration * concentration
        + weights.overlap * overlap
    )
    values = {
        "total": total,
        "transport": transport,
        "site": site,
        "concentration": concentration,
        "overlap": overlap,
        "probability": probability,
        "forward_prediction": forward_prediction,
        "reverse_prediction": reverse_prediction,
    }
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise MaterialTransportError(f"objective component {name} is non-finite")
    return values


__all__ = [
    "MaterialTransportConfig",
    "MaterialTransportError",
    "MaterialTransportWeights",
    "build_bidirectional_field",
    "channel_overlap_loss",
    "conditional_from_similarity",
    "cyclic_material_transport_objective",
    "directional_site_cost",
    "extract_endpoint_grid_descriptors",
    "jensen_shannon_divergence",
    "local_candidate_layout",
    "local_similarity_field",
    "opposite_offset_columns",
    "sparse_transport",
    "weighted_site_loss",
]
