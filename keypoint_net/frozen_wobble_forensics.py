"""Pure, deterministic metrics for the frozen wobble forensic gate.

This module deliberately contains no training code.  The functions operate on
already-produced coordinates or logits so that their semantics can be proved on
planted oracles before any saved checkpoint is inspected.

Coordinate convention
---------------------
Coordinates are ``(x, y)`` on endpoint-aligned ``[-1, 1]`` grids.  A rendered
world-Z roll by ``theta`` is physically de-rotated with ``R(-theta)`` around the
locked projected pivot ``(0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


GRID_MIN = -1.0
GRID_MAX = 1.0


class ForensicContractError(ValueError):
    """Raised when an input would make a forensic result uninterpretable."""


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ForensicContractError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ForensicContractError(f"{name} contains non-finite values")
    return array


def endpoint_grid(size: int) -> np.ndarray:
    """Return the exact endpoint-aligned coordinate axis used by the model."""
    if int(size) != size or size < 2:
        raise ForensicContractError(f"grid size must be an integer >=2, got {size!r}")
    return np.linspace(GRID_MIN, GRID_MAX, int(size), dtype=np.float64)


def rotate_points(
    points: Any,
    theta_deg: Any,
    *,
    pivot: Iterable[float] = (0.0, 0.0),
) -> np.ndarray:
    """Rotate ``(..., 2)`` points by matching leading ``theta_deg`` values."""
    pts = _finite_array(points, name="points")
    if pts.ndim < 2 or pts.shape[-1] != 2:
        raise ForensicContractError("points must have shape (..., 2)")
    theta = _finite_array(theta_deg, name="theta_deg")
    if theta.ndim == 0:
        theta = np.full(pts.shape[:-1], float(theta), dtype=np.float64)
    else:
        try:
            theta = np.broadcast_to(theta, pts.shape[:-1])
        except ValueError as exc:
            raise ForensicContractError(
                f"theta_deg shape {theta.shape} does not broadcast to {pts.shape[:-1]}"
            ) from exc
    centre = _finite_array(tuple(pivot), name="pivot", ndim=1)
    if centre.shape != (2,):
        raise ForensicContractError("pivot must contain exactly two values")

    radians = np.deg2rad(theta)
    cos_t = np.cos(radians)
    sin_t = np.sin(radians)
    shifted = pts - centre
    x = cos_t * shifted[..., 0] - sin_t * shifted[..., 1]
    y = sin_t * shifted[..., 0] + cos_t * shifted[..., 1]
    return np.stack((x, y), axis=-1) + centre


def canonicalize(
    raw_points: Any,
    theta_deg: Any,
    *,
    pivot: Iterable[float] = (0.0, 0.0),
    sign: int = -1,
) -> np.ndarray:
    """Map raw image coordinates into the locked physical canonical frame."""
    if sign not in (-1, 1):
        raise ForensicContractError("sign must be -1 (physical) or +1 (audit only)")
    points = _finite_array(raw_points, name="raw_points")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ForensicContractError("raw_points must have shape (frames, channels, 2)")
    theta = _finite_array(theta_deg, name="theta_deg", ndim=1)
    if theta.shape[0] != points.shape[0]:
        raise ForensicContractError("theta_deg length must equal the number of frames")
    return rotate_points(points, sign * theta[:, None], pivot=pivot)


def raw_from_canonical(
    canonical_points: Any,
    theta_deg: Any,
    *,
    pivot: Iterable[float] = (0.0, 0.0),
) -> np.ndarray:
    """Render canonical points into the raw rotating image frame."""
    points = _finite_array(canonical_points, name="canonical_points")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ForensicContractError("canonical_points must have shape (frames, channels, 2)")
    theta = _finite_array(theta_deg, name="theta_deg", ndim=1)
    if theta.shape[0] != points.shape[0]:
        raise ForensicContractError("theta_deg length must equal the number of frames")
    return rotate_points(points, theta[:, None], pivot=pivot)


def geometry_smoke() -> dict[str, float]:
    """Analytic sign/pivot smoke, independent of model outputs and masks."""
    theta = np.arange(180, dtype=np.float64) * 2.0
    anchors = np.asarray(((0.08, 0.03), (0.62, -0.21), (-0.41, 0.55)))
    canonical = np.broadcast_to(anchors, (theta.size,) + anchors.shape).copy()
    raw = raw_from_canonical(canonical, theta)
    correct = canonicalize(raw, theta, sign=-1)
    wrong_sign = canonicalize(raw, theta, sign=1)
    wrong_pivot = canonicalize(raw, theta, sign=-1, pivot=(0.17, -0.11))

    def rms_about_reference(value: np.ndarray, reference: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value - reference))))

    return {
        "correct_sign_rms_normalized": rms_about_reference(correct, canonical),
        "wrong_sign_rms_normalized": rms_about_reference(wrong_sign, canonical),
        "wrong_pivot_rms_normalized": rms_about_reference(wrong_pivot, canonical),
    }


def _periodic_interp(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    source_x = x[order]
    source_y = y[order]
    if np.unique(source_x).size != source_x.size:
        raise ForensicContractError("periodic interpolation angles must be unique")
    extended_x = np.concatenate((source_x - 360.0, source_x, source_x + 360.0))
    extended_y = np.concatenate((source_y, source_y, source_y))
    wrapped = np.mod(target, 360.0)
    return np.interp(wrapped, extended_x, extended_y)


def angle_aligned_residue_points(
    canonical_points: Any,
    theta_deg: Any,
    frame_indices: Any | None = None,
    *,
    target_angles_deg: Any | None = None,
) -> np.ndarray:
    """Interpolate the three residue chains to common physical angles.

    Returns an array of shape ``(target_angles, 3, channels, 2)``.
    """
    points = _finite_array(canonical_points, name="canonical_points", ndim=3)
    theta = _finite_array(theta_deg, name="theta_deg", ndim=1)
    if points.shape[0] != theta.size or points.shape[-1] != 2:
        raise ForensicContractError("coordinate and angle shapes are inconsistent")
    if frame_indices is None:
        indices = np.arange(theta.size, dtype=np.int64)
    else:
        raw_indices = np.asarray(frame_indices)
        if raw_indices.ndim != 1 or raw_indices.size != theta.size:
            raise ForensicContractError("frame_indices must have one value per frame")
        if not np.equal(raw_indices, np.round(raw_indices)).all():
            raise ForensicContractError("frame_indices must be integers")
        indices = raw_indices.astype(np.int64)
    target = theta if target_angles_deg is None else _finite_array(
        target_angles_deg, name="target_angles_deg", ndim=1
    )
    aligned = np.empty((target.size, 3, points.shape[1], 2), dtype=np.float64)
    for residue in range(3):
        keep = indices % 3 == residue
        if int(np.count_nonzero(keep)) < 2:
            raise ForensicContractError(f"residue {residue} has fewer than two samples")
        for channel in range(points.shape[1]):
            for coordinate in range(2):
                aligned[:, residue, channel, coordinate] = _periodic_interp(
                    theta[keep], points[keep, channel, coordinate], target
                )
    return aligned


def _energy(value: np.ndarray) -> float:
    return float(np.sum(np.square(value), dtype=np.float64))


def _fraction_removed(before: np.ndarray, after: np.ndarray) -> float:
    denominator = _energy(before)
    if denominator == 0.0:
        return 1.0 if _energy(after) == 0.0 else 0.0
    return float(1.0 - _energy(after) / denominator)


def _period_three_explained_energy(series: np.ndarray, frame_indices: np.ndarray) -> float:
    """Fraction of mean-centred position energy explained by period three."""
    centred = series - np.mean(series, axis=0, keepdims=True)
    denominator = _energy(centred)
    if denominator == 0.0:
        return 0.0
    phase = 2.0 * np.pi * frame_indices.astype(np.float64) / 3.0
    design = np.stack((np.sin(phase), np.cos(phase)), axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, centred, rcond=None)
    fitted = design @ coefficients
    return float(np.clip(_energy(fitted) / denominator, 0.0, 1.0))


@dataclass(frozen=True)
class PhaseChannelMetrics:
    residue_means: list[list[float]]
    maximum_residue_mean_offset: float
    angle_aligned_disagreement_mean: float
    angle_aligned_disagreement_max: float
    period_three_position_energy_fraction: float
    transition_class_step_energy_fraction: float
    position_energy_removed_by_constant_residue_correction: float
    adjacent_step_energy_removed_by_constant_residue_correction: float
    maximum_second_difference: float
    maximum_phase_acceleration_bound: float
    maximum_spike_excess_over_phase_bound: float


def phase_metrics(
    canonical_points: Any,
    theta_deg: Any,
    frame_indices: Any | None = None,
) -> dict[str, Any]:
    """Measure how much wobble can be explained by three phase chains."""
    points = _finite_array(canonical_points, name="canonical_points", ndim=3)
    theta = _finite_array(theta_deg, name="theta_deg", ndim=1)
    if points.shape[0] != theta.size or points.shape[-1] != 2:
        raise ForensicContractError("canonical_points must be (frames, channels, 2)")
    if frame_indices is None:
        indices = np.arange(theta.size, dtype=np.int64)
    else:
        raw_indices = np.asarray(frame_indices)
        if raw_indices.ndim != 1 or raw_indices.size != theta.size:
            raise ForensicContractError("frame_indices must have one value per frame")
        if not np.equal(raw_indices, np.round(raw_indices)).all():
            raise ForensicContractError("frame_indices must be integers")
        indices = raw_indices.astype(np.int64)

    if points.shape[0] < 7:
        raise ForensicContractError("phase metrics require at least seven frames")
    if not np.array_equal(indices, np.arange(indices[0], indices[0] + indices.size)):
        raise ForensicContractError("phase metrics require consecutive frame indices")

    aligned = angle_aligned_residue_points(points, theta, indices)
    channel_rows: list[dict[str, Any]] = []
    adjacent = np.diff(points, axis=0)
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]

    for channel in range(points.shape[1]):
        residue_means = np.stack(
            [np.mean(points[indices % 3 == residue, channel], axis=0) for residue in range(3)]
        )
        pairwise_residue = np.linalg.norm(
            residue_means[:, None, :] - residue_means[None, :, :], axis=-1
        )

        aligned_channel = aligned[:, :, channel, :]
        aligned_pairwise = np.linalg.norm(
            aligned_channel[:, :, None, :] - aligned_channel[:, None, :, :], axis=-1
        )
        disagreement = np.max(aligned_pairwise, axis=(1, 2))

        corrected = points[:, channel] - residue_means[indices % 3]
        corrected += np.mean(points[:, channel], axis=0, keepdims=True)
        original_centred = points[:, channel] - np.mean(
            points[:, channel], axis=0, keepdims=True
        )
        corrected_centred = corrected - np.mean(corrected, axis=0, keepdims=True)

        steps = adjacent[:, channel]
        step_classes = indices[:-1] % 3
        class_means = np.stack(
            [np.mean(steps[step_classes == residue], axis=0) for residue in range(3)]
        )
        fitted_steps = class_means[step_classes]
        transition_fraction = _fraction_removed(steps, steps - fitted_steps)

        corrected_steps = np.diff(corrected, axis=0)
        step_removed = _fraction_removed(steps, corrected_steps)
        position_removed = _fraction_removed(original_centred, corrected_centred)

        # Predict the second difference produced by the fitted *constant*
        # residue offsets.  Using the observed disagreement at the spike angle
        # as its own bound would be circular: an isolated spike would enlarge
        # the bound that is supposed to detect it.  Angle-varying residue
        # disagreement remains visible in the separate aligned-disagreement and
        # correction-residual metrics; it is not silently called constant phase.
        residue_offset = residue_means - np.mean(residue_means, axis=0, keepdims=True)
        centre_residue = indices[1:-1] % 3
        phase_second_vector = (
            residue_offset[(centre_residue + 1) % 3]
            - 2.0 * residue_offset[centre_residue]
            + residue_offset[(centre_residue - 1) % 3]
        )
        local_phase_acceleration_bound = np.linalg.norm(phase_second_vector, axis=-1)
        spike_magnitude = np.linalg.norm(second[:, channel], axis=-1)
        spike_excess = spike_magnitude - local_phase_acceleration_bound

        row = PhaseChannelMetrics(
            residue_means=residue_means.tolist(),
            maximum_residue_mean_offset=float(np.max(pairwise_residue)),
            angle_aligned_disagreement_mean=float(np.mean(disagreement)),
            angle_aligned_disagreement_max=float(np.max(disagreement)),
            period_three_position_energy_fraction=_period_three_explained_energy(
                points[:, channel], indices
            ),
            transition_class_step_energy_fraction=transition_fraction,
            position_energy_removed_by_constant_residue_correction=position_removed,
            adjacent_step_energy_removed_by_constant_residue_correction=step_removed,
            maximum_second_difference=float(np.max(spike_magnitude)),
            maximum_phase_acceleration_bound=float(np.max(local_phase_acceleration_bound)),
            maximum_spike_excess_over_phase_bound=float(np.max(spike_excess)),
        )
        channel_rows.append(row.__dict__)

    return {
        "definition": {
            "frame_unit": "consecutive rendered frames",
            "angle_unit": "degrees",
            "coordinate_unit": "endpoint-aligned normalized coordinate",
            "phase_acceleration_bound": "second-difference magnitude predicted by fitted constant residue offsets; spike itself is not used as its own bound",
            "statistics": "descriptive; frames and differences are correlated",
        },
        "channels": channel_rows,
    }


def coordinate_dynamics(points: Any) -> dict[str, np.ndarray]:
    """Return per-frame/channel coordinate diagnostics without aggregation."""
    array = _finite_array(points, name="points", ndim=3)
    if array.shape[-1] != 2 or array.shape[0] < 7:
        raise ForensicContractError("points must be (at least 7 frames, channels, 2)")
    centre = np.mean(array, axis=0, keepdims=True)
    return {
        "radius_about_mean": np.linalg.norm(array - centre, axis=-1),
        "adjacent_step": np.linalg.norm(np.diff(array, axis=0), axis=-1),
        "within_residue_step": np.linalg.norm(array[3:] - array[:-3], axis=-1),
        "second_difference": np.linalg.norm(
            array[2:] - 2.0 * array[1:-1] + array[:-2], axis=-1
        ),
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=(-2, -1), keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=(-2, -1), keepdims=True)


def spatial_expectation_float32(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Recompute the production soft coordinate using NumPy float32 arithmetic."""
    raw = np.asarray(logits)
    if raw.ndim != 4:
        raise ForensicContractError("logits must have shape (frames, channels, height, width)")
    if not np.isfinite(raw).all():
        raise ForensicContractError("logits contains non-finite values")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ForensicContractError("temperature must be finite and positive")
    values = raw.astype(np.float32, copy=False) / np.float32(temperature)
    values = values - np.max(values, axis=(-2, -1), keepdims=True)
    weights = np.exp(values).astype(np.float32, copy=False)
    weights /= np.sum(weights, axis=(-2, -1), keepdims=True, dtype=np.float32)
    height, width = raw.shape[-2:]
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    x = np.sum(weights * xx, axis=(-2, -1), dtype=np.float32)
    y = np.sum(weights * yy, axis=(-2, -1), dtype=np.float32)
    return np.stack((x, y), axis=-1).astype(np.float64)


def warp_image_by_standard_rotation(
    source: Any,
    theta_deg: float,
    *,
    pivot: Iterable[float] = (0.0, 0.0),
    order: int = 0,
) -> np.ndarray:
    """Predict an image after a standard coordinate rotation by ``theta_deg``.

    Output pixels are inverse-mapped into ``source`` using endpoint-aligned
    normalized coordinates.  This is evaluation-only and is used to verify the
    metadata sign and projected pivot on actual masks.
    """
    try:
        from scipy.ndimage import map_coordinates
    except Exception as exc:  # pragma: no cover - dependency failure is explicit.
        raise ForensicContractError("scipy is required for image rotation audits") from exc
    image = np.asarray(source)
    if image.ndim not in (2, 3):
        raise ForensicContractError("source image must be 2D or 3D")
    if order not in (0, 1):
        raise ForensicContractError("rotation audit supports interpolation order 0 or 1")
    height, width = image.shape[:2]
    y_axis = endpoint_grid(height)
    x_axis = endpoint_grid(width)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    target_points = np.stack((xx, yy), axis=-1)
    # If target p = R(theta) source q, inverse mapping samples q = R(-theta) p.
    source_points = rotate_points(target_points, -float(theta_deg), pivot=pivot)
    source_x = (source_points[..., 0] + 1.0) * 0.5 * (width - 1)
    source_y = (source_points[..., 1] + 1.0) * 0.5 * (height - 1)
    coordinates = np.stack((source_y, source_x), axis=0)
    if image.ndim == 2:
        return map_coordinates(
            image.astype(np.float64), coordinates, order=order, mode="constant", cval=0.0
        )
    channels = [
        map_coordinates(
            image[..., channel].astype(np.float64),
            coordinates,
            order=order,
            mode="constant",
            cval=0.0,
        )
        for channel in range(image.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def _peak_table(
    probabilities: np.ndarray,
    logits: np.ndarray,
    hard_y: np.ndarray,
    hard_x: np.ndarray,
    radius: int,
) -> dict[str, np.ndarray]:
    frames, channels, height, width = probabilities.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    second_y = np.empty((frames, channels), dtype=np.int64)
    second_x = np.empty((frames, channels), dtype=np.int64)
    second_probability = np.empty((frames, channels), dtype=np.float64)
    second_logit = np.empty((frames, channels), dtype=np.float64)
    first_mass = np.empty((frames, channels), dtype=np.float64)
    second_mass = np.empty((frames, channels), dtype=np.float64)
    for frame in range(frames):
        for channel in range(channels):
            distance_sq_first = (
                np.square(yy - hard_y[frame, channel])
                + np.square(xx - hard_x[frame, channel])
            )
            allowed = distance_sq_first > radius * radius
            if not bool(np.any(allowed)):
                raise ForensicContractError("exclusion radius leaves no second-peak candidate")
            candidate = np.where(allowed, probabilities[frame, channel], -np.inf)
            flat_second = int(np.argmax(candidate))
            sy, sx = np.unravel_index(flat_second, (height, width))
            second_y[frame, channel] = sy
            second_x[frame, channel] = sx
            second_probability[frame, channel] = probabilities[frame, channel, sy, sx]
            second_logit[frame, channel] = logits[frame, channel, sy, sx]
            first_mass[frame, channel] = float(
                np.sum(probabilities[frame, channel][distance_sq_first <= radius * radius])
            )
            distance_sq_second = np.square(yy - sy) + np.square(xx - sx)
            second_mass[frame, channel] = float(
                np.sum(probabilities[frame, channel][distance_sq_second <= radius * radius])
            )
    return {
        "second_peak_y_cell": second_y,
        "second_peak_x_cell": second_x,
        "second_peak_probability": second_probability,
        "probability_margin": probabilities[
            np.arange(frames)[:, None],
            np.arange(channels)[None, :],
            hard_y,
            hard_x,
        ]
        - second_probability,
        "logit_margin": logits[
            np.arange(frames)[:, None],
            np.arange(channels)[None, :],
            hard_y,
            hard_x,
        ]
        - second_logit,
        "first_local_probability_mass": first_mass,
        "second_local_probability_mass": second_mass,
    }


def heatmap_topology(logits: Any, *, temperature: float = 1.0) -> dict[str, Any]:
    """Compute soft/hard and separated-peak evidence from heatmap logits.

    ``logits`` has shape ``(frames, channels, height, width)``.  All returned
    arrays preserve frame and channel axes.  Peak exclusion uses Euclidean cell
    distance and is reported for radii 1, 2, and 4.
    """
    raw = _finite_array(logits, name="logits", ndim=4)
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ForensicContractError("temperature must be finite and positive")
    frames, channels, height, width = raw.shape
    if height < 10 or width < 10:
        raise ForensicContractError("heatmaps must be at least 10x10")
    probabilities = _stable_softmax(raw / float(temperature))
    y_axis = endpoint_grid(height)
    x_axis = endpoint_grid(width)
    yy_normalized, xx_normalized = np.meshgrid(y_axis, x_axis, indexing="ij")
    yy_cells, xx_cells = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )

    soft_x = np.sum(probabilities * xx_normalized, axis=(-2, -1))
    soft_y = np.sum(probabilities * yy_normalized, axis=(-2, -1))
    soft_x_cell = np.sum(probabilities * xx_cells, axis=(-2, -1))
    soft_y_cell = np.sum(probabilities * yy_cells, axis=(-2, -1))
    flat_hard = np.argmax(raw.reshape(frames, channels, -1), axis=-1)
    hard_y, hard_x = np.unravel_index(flat_hard, (height, width))
    hard_coordinate = np.stack((x_axis[hard_x], y_axis[hard_y]), axis=-1)
    soft_coordinate = np.stack((soft_x, soft_y), axis=-1)

    eps = np.finfo(np.float64).tiny
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, eps)), axis=(-2, -1))
    dx = xx_cells[None, None] - soft_x_cell[:, :, None, None]
    dy = yy_cells[None, None] - soft_y_cell[:, :, None, None]
    covariance_xx = np.sum(probabilities * dx * dx, axis=(-2, -1))
    covariance_yy = np.sum(probabilities * dy * dy, axis=(-2, -1))
    covariance_xy = np.sum(probabilities * dx * dy, axis=(-2, -1))
    trace = covariance_xx + covariance_yy
    discriminant = np.sqrt(
        np.maximum(np.square(covariance_xx - covariance_yy) + 4.0 * np.square(covariance_xy), 0.0)
    )
    eigen_max = np.maximum(0.5 * (trace + discriminant), 0.0)
    eigen_min = np.maximum(0.5 * (trace - discriminant), 0.0)

    first_probability = probabilities[
        np.arange(frames)[:, None],
        np.arange(channels)[None, :],
        hard_y,
        hard_x,
    ]
    result: dict[str, Any] = {
        "definition": {
            "temperature": float(temperature),
            "grid": "endpoint-aligned [-1,1]",
            "peak_exclusion": "Euclidean heatmap-cell radius",
            "statistics": "per frame and fixed channel; descriptive",
        },
        "probabilities": probabilities,
        "soft_coordinate": soft_coordinate,
        "hard_coordinate": hard_coordinate,
        "hard_x_cell": hard_x,
        "hard_y_cell": hard_y,
        "soft_hard_distance_cells": np.sqrt(
            np.square(soft_x_cell - hard_x) + np.square(soft_y_cell - hard_y)
        ),
        "entropy_nats": entropy,
        "effective_support_cells": np.exp(entropy),
        "rms_width_cells": np.sqrt(np.maximum(trace, 0.0)),
        "anisotropy_ratio": np.sqrt(eigen_max / np.maximum(eigen_min, eps)),
        "first_peak_probability": first_probability,
    }
    result["separated_peaks"] = {
        str(radius): _peak_table(probabilities, raw, hard_y, hard_x, radius)
        for radius in (1, 2, 4)
    }
    return result


def deterministic_spike_frames(
    canonical_points: Any,
    *,
    top_k: int = 5,
) -> list[list[dict[str, float | int]]]:
    """Select the largest non-seam second differences with stable tie-breaking."""
    points = _finite_array(canonical_points, name="canonical_points", ndim=3)
    if top_k <= 0:
        raise ForensicContractError("top_k must be positive")
    magnitude = coordinate_dynamics(points)["second_difference"]
    rows: list[list[dict[str, float | int]]] = []
    centre_frames = np.arange(1, points.shape[0] - 1, dtype=np.int64)
    for channel in range(points.shape[1]):
        order = np.lexsort((centre_frames, -magnitude[:, channel]))[:top_k]
        rows.append(
            [
                {
                    "centre_frame_index": int(centre_frames[index]),
                    "canonical_second_difference": float(magnitude[index, channel]),
                }
                for index in order
            ]
        )
    return rows
