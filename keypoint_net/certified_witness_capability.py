"""Pure contracts for the exact-ten-track supervised capability gate."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SIZE = 512
FEATURE_SIZE = 64
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
EXPECTED_WITNESS_IDS = (
    1857,
    2237,
    2241,
    12601,
    12606,
    12980,
    12993,
    13100,
    13868,
    14394,
)
CELL_SPACING_PX = (IMAGE_SIZE - 1) / (FEATURE_SIZE - 1)
HALF_CELL_DIAGONAL_PX = CELL_SPACING_PX / math.sqrt(2.0)


class CapabilityContractError(RuntimeError):
    """Raised when a semantic or provenance contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"missing file: {resolved}")
    return {
        "absolute_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def pixel_to_normalized(points_px: np.ndarray) -> np.ndarray:
    return np.asarray(points_px, dtype=np.float64) / (IMAGE_SIZE - 1) * 2.0 - 1.0


def normalized_to_pixel(points_normalized: np.ndarray) -> np.ndarray:
    return (np.asarray(points_normalized, dtype=np.float64) + 1.0) * 0.5 * (IMAGE_SIZE - 1)


def gaussian_target_distribution(
    target_normalized: torch.Tensor,
    *,
    height: int = FEATURE_SIZE,
    width: int = FEATURE_SIZE,
    sigma_input_px: float = 8.0,
) -> torch.Tensor:
    """Return a per-channel probability target centred at continuous points."""
    require(sigma_input_px > 0.0, "sigma_input_px must be positive")
    y = torch.linspace(-1.0, 1.0, height, device=target_normalized.device)
    x = torch.linspace(-1.0, 1.0, width, device=target_normalized.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma_normalized = 2.0 * sigma_input_px / (IMAGE_SIZE - 1)
    squared_distance = (
        (xx[None, None] - target_normalized[..., 0, None, None]) ** 2
        + (yy[None, None] - target_normalized[..., 1, None, None]) ** 2
    )
    logits = -0.5 * squared_distance / (sigma_normalized**2)
    return torch.softmax(logits.flatten(-2), dim=-1)


def dense_heatmap_cross_entropy(
    logits: torch.Tensor,
    target_normalized: torch.Tensor,
    *,
    sigma_input_px: float = 8.0,
) -> torch.Tensor:
    require(logits.ndim == 4, f"expected BxKxHxW logits, got {tuple(logits.shape)}")
    require(
        tuple(target_normalized.shape) == (logits.shape[0], logits.shape[1], 2),
        "target/logit shape mismatch",
    )
    target_distribution = gaussian_target_distribution(
        target_normalized,
        height=logits.shape[-2],
        width=logits.shape[-1],
        sigma_input_px=sigma_input_px,
    )
    log_probability = F.log_softmax(logits.flatten(-2), dim=-1)
    return -(target_distribution * log_probability).sum(dim=-1).mean()


def model_state_sha256(module: torch.nn.Module) -> str:
    """Hash a CPU-normalized state dict without relying on pickle metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "summary values invalid")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "q90": float(np.quantile(vector, 0.90)),
        "maximum": float(vector.max()),
    }


def evaluate_predictions(
    prediction_px: np.ndarray,
    target_px: np.ndarray,
    masks: np.ndarray,
    *,
    witness_ids: tuple[int, ...] = EXPECTED_WITNESS_IDS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the immutable localization/identity/distinctness contract."""
    prediction = np.asarray(prediction_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=bool)
    require(prediction.shape == target.shape, "prediction/target shape mismatch")
    require(prediction.ndim == 3 and prediction.shape[1:] == (EXPECTED_WITNESSES, 2), "unexpected point shape")
    require(mask_array.shape == (prediction.shape[0], IMAGE_SIZE, IMAGE_SIZE), "unexpected mask shape")
    require(tuple(witness_ids) == EXPECTED_WITNESS_IDS, "witness identity/order differs")
    require(bool(np.isfinite(prediction).all()), "prediction contains non-finite values")
    require(bool(np.isfinite(target).all()), "target contains non-finite values")

    material_error = np.linalg.norm(prediction - target, axis=-1)
    rounded = np.rint(prediction).astype(np.int64)
    in_image = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < IMAGE_SIZE)
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < IMAGE_SIZE)
    )
    clipped = np.clip(rounded, 0, IMAGE_SIZE - 1)
    frame_indices = np.arange(prediction.shape[0])[:, None]
    on_object = in_image & mask_array[frame_indices, clipped[..., 1], clipped[..., 0]]

    target_distances = np.linalg.norm(
        prediction[:, :, None, :] - target[:, None, :, :], axis=-1
    )
    assigned_identity = np.argmin(target_distances, axis=-1)
    identity_correct = assigned_identity == np.arange(EXPECTED_WITNESSES)[None, :]

    prediction_pair_distance = np.linalg.norm(
        prediction[:, :, None, :] - prediction[:, None, :, :], axis=-1
    )
    target_pair_distance = np.linalg.norm(
        target[:, :, None, :] - target[:, None, :, :], axis=-1
    )
    pair_mask = np.triu(np.ones((EXPECTED_WITNESSES, EXPECTED_WITNESSES), dtype=bool), k=1)
    pair_ratio = prediction_pair_distance[:, pair_mask] / target_pair_distance[:, pair_mask]
    distinct_pair = pair_ratio >= 0.5

    within_half_cell = material_error <= HALF_CELL_DIAGONAL_PX + 1e-12
    strict_pass = bool(
        np.all(within_half_cell)
        and np.all(on_object)
        and np.all(identity_correct)
        and np.all(distinct_pair)
    )
    per_witness: list[dict[str, Any]] = []
    for channel, witness_id in enumerate(witness_ids):
        per_witness.append(
            {
                "channel": channel,
                "witness_id": int(witness_id),
                "material_error_px": _summary(material_error[:, channel]),
                "within_half_cell_rate": float(within_half_cell[:, channel].mean()),
                "on_object_rate": float(on_object[:, channel].mean()),
                "identity_assignment_rate": float(identity_correct[:, channel].mean()),
                "strict_channel_pass": bool(
                    np.all(within_half_cell[:, channel])
                    and np.all(on_object[:, channel])
                    and np.all(identity_correct[:, channel])
                ),
            }
        )
    violations = {
        "outside_half_cell_count": int(np.size(within_half_cell) - np.sum(within_half_cell)),
        "off_object_count": int(np.size(on_object) - np.sum(on_object)),
        "wrong_identity_count": int(np.size(identity_correct) - np.sum(identity_correct)),
        "collapsed_pair_count": int(np.size(distinct_pair) - np.sum(distinct_pair)),
    }
    report = {
        "strict_capability_pass": strict_pass,
        "frame_count": int(prediction.shape[0]),
        "witness_count": EXPECTED_WITNESSES,
        "cell_spacing_px": CELL_SPACING_PX,
        "half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
        "violations": violations,
        "material_error_px": _summary(material_error),
        "within_half_cell_rate": float(within_half_cell.mean()),
        "on_object_rate": float(on_object.mean()),
        "identity_assignment_rate": float(identity_correct.mean()),
        "minimum_predicted_pair_distance_px": float(prediction_pair_distance[:, pair_mask].min()),
        "minimum_predicted_to_physical_pair_ratio": float(pair_ratio.min()),
        "per_witness": per_witness,
        "statistical_scope": {
            "inference": "descriptive_only",
            "sample_unit": "fixed_witness_over_one_180_frame_correlated_orbit",
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    derived = {
        "material_error_px": material_error,
        "within_half_cell": within_half_cell,
        "on_object": on_object,
        "identity_correct": identity_correct,
        "assigned_identity": assigned_identity,
        "prediction_pair_distance_px": prediction_pair_distance,
        "target_pair_distance_px": target_pair_distance,
        "distinct_pair": distinct_pair,
    }
    return report, derived


def evaluation_score(report: dict[str, Any]) -> tuple[int, int, int, int, float, float]:
    """Frozen lexicographic checkpoint score; lower is better."""
    violations = report["violations"]
    return (
        int(violations["outside_half_cell_count"]),
        int(violations["wrong_identity_count"]),
        int(violations["collapsed_pair_count"]),
        int(violations["off_object_count"]),
        float(report["material_error_px"]["maximum"]),
        float(report["material_error_px"]["median"]),
    )


def nearest_r64_grid_prediction(target_px: np.ndarray) -> np.ndarray:
    """Construct the planted-grid positive control for the evaluator."""
    normalized = pixel_to_normalized(target_px)
    cell = np.rint((normalized + 1.0) * 0.5 * (FEATURE_SIZE - 1)).astype(np.int64)
    cell = np.clip(cell, 0, FEATURE_SIZE - 1)
    grid_normalized = -1.0 + 2.0 * cell.astype(np.float64) / (FEATURE_SIZE - 1)
    return normalized_to_pixel(grid_normalized)


def bilinear_planted_logits(target_normalized: np.ndarray) -> torch.Tensor:
    """Create logits whose r64 softmax expectation is the continuous target.

    Probability mass is placed on the four enclosing grid cells with bilinear
    weights. ``-inf`` elsewhere makes the control exact under spatial softmax
    without pretending that a hard argmax cell can remain on a thin silhouette.
    """
    target = np.asarray(target_normalized, dtype=np.float64)
    require(target.ndim == 3 and target.shape[1:] == (EXPECTED_WITNESSES, 2), "unexpected planted target shape")
    require(bool(np.isfinite(target).all()), "planted target contains non-finite values")
    require(float(target.min()) >= -1.0 and float(target.max()) <= 1.0, "planted target outside normalized image")
    cell = (target + 1.0) * 0.5 * (FEATURE_SIZE - 1)
    logits = np.full(
        (target.shape[0], EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
        -np.inf,
        dtype=np.float64,
    )
    for frame in range(target.shape[0]):
        for witness in range(EXPECTED_WITNESSES):
            x_value, y_value = cell[frame, witness]
            x0 = int(np.floor(x_value))
            y0 = int(np.floor(y_value))
            x1 = min(x0 + 1, FEATURE_SIZE - 1)
            y1 = min(y0 + 1, FEATURE_SIZE - 1)
            x_weights: dict[int, float] = {}
            y_weights: dict[int, float] = {}
            x_weights[x0] = x_weights.get(x0, 0.0) + (1.0 - (x_value - x0))
            x_weights[x1] = x_weights.get(x1, 0.0) + (x_value - x0)
            y_weights[y0] = y_weights.get(y0, 0.0) + (1.0 - (y_value - y0))
            y_weights[y1] = y_weights.get(y1, 0.0) + (y_value - y0)
            for x_index, x_weight in x_weights.items():
                for y_index, y_weight in y_weights.items():
                    weight = x_weight * y_weight
                    if weight > 0.0:
                        logits[frame, witness, y_index, x_index] = np.log(weight)
    return torch.from_numpy(logits)


def torch_checkpoint_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a checkpoint deterministically enough for an immediate receipt."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()
