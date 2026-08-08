"""Evaluation-only material-alignment and static-parking audit.

The audit freezes an already selected descriptor-matrix checkpoint.  For each
frozen validation pair it compares the exact attachment-loss coordinate
descent vector with the known +6 degree material transport.  It also measures
whether predicted motion is attenuated relative to the oracle motion.  No
parameter, optimizer, scheduler, checkpoint, or selection state is updated.

This file is intentionally post-hoc: ``source_repo_root`` identifies the clean
historical checkout authorized by the completed-run receipt, while
``audit_repo_root`` identifies the clean checkout containing this audit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from keypoint_net import descriptor_attachment
from keypoint_net import descriptor_attachment_runtime
from keypoint_net import eval_representation
from keypoint_net import representation_fresh_checkpoint_runtime as fresh_runtime


CELL_SCHEMA_VERSION = "descriptor-material-alignment-cell.v1"
DECISION_SCHEMA_VERSION = "descriptor-material-alignment-decision.v1"
EXPECTED_BRANCH = "agent/representation-oracles-20260726"
GRID_STEP_64 = 2.0 / 63.0
NORM_EPSILON = 1e-12
ON_OBJECT_RATE_MIN = 0.75
MIN_DEFINED_ALIGNMENT_FRACTION = 0.90


class MaterialAlignmentError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterialAlignmentError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    absolute = path.expanduser().resolve(strict=True)
    _require(absolute.is_file() and not absolute.is_symlink(), f"invalid file: {absolute}")
    data = absolute.read_bytes()
    return {
        "absolute_path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        _require(written > 0, "exclusive artifact write made no progress")
        offset += written


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_checkout(root: Path, *, expected_commit: str) -> dict[str, str]:
    resolved = root.expanduser().resolve(strict=True)
    observed = {
        "absolute_path": str(resolved),
        "branch": _git(resolved, "branch", "--show-current"),
        "commit": _git(resolved, "rev-parse", "HEAD"),
        "tracked_status": _git(
            resolved, "status", "--porcelain", "--untracked-files=no"
        ),
    }
    _require(observed["branch"] == EXPECTED_BRANCH, "checkout branch differs")
    _require(observed["commit"] == expected_commit, "checkout commit differs")
    _require(observed["tracked_status"] == "", "checkout has tracked changes")
    return observed


def rotation_matrix_degrees(angle: float) -> np.ndarray:
    radians = math.radians(float(angle))
    return np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )


def apply_rotation(points: np.ndarray, angle: float) -> np.ndarray:
    return np.matmul(np.asarray(points, dtype=np.float64), rotation_matrix_degrees(angle).T)


def sample_mask(points: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the production evaluator's endpoint-grid and nearest-pixel rule."""

    values = np.asarray(points, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=bool)
    _require(values.ndim == 3 and values.shape[-1] == 2, "points shape differs")
    _require(mask_values.ndim == 3 and mask_values.shape[0] == values.shape[0],
             "mask shape differs")
    height, width = mask_values.shape[-2:]
    u = np.rint((values[..., 0] + 1.0) * (width - 1) / 2.0).astype(np.int64)
    v = np.rint((values[..., 1] + 1.0) * (height - 1) / 2.0).astype(np.int64)
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside = np.zeros(in_image.shape, dtype=bool)
    for row in range(values.shape[0]):
        valid = np.flatnonzero(in_image[row])
        inside[row, valid] = mask_values[row, v[row, valid], u[row, valid]]
    return inside, in_image


def bbox_diagonals(masks: np.ndarray) -> np.ndarray:
    values = []
    height, width = masks.shape[-2:]
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        _require(rows.size > 0, f"mask {index} is empty")
        diagonal = math.hypot(
            float(columns.max() - columns.min()) / float(width - 1),
            float(rows.max() - rows.min()) / float(height - 1),
        )
        _require(diagonal > 0.0, f"mask {index} has zero diagonal")
        values.append(diagonal)
    return np.asarray(values, dtype=np.float64)


def normalized_coordinate_step(
    candidate: np.ndarray,
    descent: np.ndarray,
    *,
    step_size: float = GRID_STEP_64,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    descent_values = np.asarray(descent, dtype=np.float64)
    norms = np.linalg.norm(descent_values, axis=-1, keepdims=True)
    unit = np.divide(
        descent_values,
        norms,
        out=np.zeros_like(descent_values),
        where=norms > NORM_EPSILON,
    )
    unprojected = candidate_values + float(step_size) * unit
    stepped = np.clip(unprojected, -1.0, 1.0)
    projected = np.any(stepped != unprojected, axis=-1)
    return stepped, projected


def direction_metrics(
    *,
    anchor: np.ndarray,
    candidate: np.ndarray,
    descent: np.ndarray,
    angle_deg: float,
    candidate_masks: np.ndarray,
    candidate_bbox_diagonal: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute geometry, one-cell intervention, and mask metrics."""

    anchor_values = np.asarray(anchor, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    descent_values = np.asarray(descent, dtype=np.float64)
    _require(
        anchor_values.shape == candidate_values.shape == descent_values.shape,
        "direction tensor shapes differ",
    )
    oracle_target = apply_rotation(anchor_values, angle_deg)
    correction = oracle_target - candidate_values
    correction_norm = np.linalg.norm(correction, axis=-1)
    descent_norm = np.linalg.norm(descent_values, axis=-1)
    denominator = correction_norm * descent_norm
    cosine = np.full(correction_norm.shape, np.nan, dtype=np.float64)
    defined = denominator > NORM_EPSILON
    cosine[defined] = np.sum(correction * descent_values, axis=-1)[defined] / denominator[defined]

    stepped, projected = normalized_coordinate_step(candidate_values, descent_values)
    diagonal = np.asarray(candidate_bbox_diagonal, dtype=np.float64)[:, None]
    error_before = np.linalg.norm(correction, axis=-1) / 2.0 / diagonal
    error_after = np.linalg.norm(oracle_target - stepped, axis=-1) / 2.0 / diagonal
    before_inside, before_in_image = sample_mask(candidate_values, candidate_masks)
    after_inside, after_in_image = sample_mask(stepped, candidate_masks)
    return {
        "oracle_target": oracle_target,
        "correction_norm": correction_norm,
        "descent_norm": descent_norm,
        "cosine": cosine,
        "alignment_defined": defined,
        "stepped": stepped,
        "projected": projected,
        "error_before_objdiag": error_before,
        "error_after_objdiag": error_after,
        "error_delta_objdiag": error_after - error_before,
        "on_object_before": before_inside & before_in_image,
        "on_object_after": after_inside & after_in_image,
    }


def _descriptive(values: Iterable[float], *, sample_unit: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0 and bool(np.isfinite(array).all()),
             f"{sample_unit} values are invalid")
    return {
        "sample_unit": sample_unit,
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "inference": "descriptive_correlated_validation_pair_channel_values",
        "sem_computed": False,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]], eligible_channels: Sequence[int]) -> dict[str, Any]:
    eligible = set(int(item) for item in eligible_channels)
    selected = [row for row in rows if int(row["channel"]) in eligible]
    _require(selected, "no eligible material-alignment rows")
    defined = [row for row in selected if row["cosine_alignment"] is not None]
    defined_fraction = len(defined) / len(selected)
    cosine = _descriptive(
        (float(row["cosine_alignment"]) for row in defined),
        sample_unit="eligible_validation_pair_direction_channel_cosine",
    )
    error_before = _descriptive(
        (float(row["error_before_objdiag"]) for row in selected),
        sample_unit="eligible_validation_pair_direction_channel_error_before_objdiag",
    )
    error_after = _descriptive(
        (float(row["error_after_objdiag"]) for row in selected),
        sample_unit="eligible_validation_pair_direction_channel_error_after_objdiag",
    )
    error_delta = _descriptive(
        (float(row["error_delta_objdiag"]) for row in selected),
        sample_unit="eligible_validation_pair_direction_channel_error_delta_objdiag",
    )

    channel_rows = []
    for channel in sorted(eligible):
        local = [row for row in selected if int(row["channel"]) == channel]
        _require(local, f"eligible channel {channel} lacks rows")
        before_rate = float(np.mean([bool(row["on_object_before"]) for row in local]))
        after_rate = float(np.mean([bool(row["on_object_after"]) for row in local]))
        channel_rows.append(
            {
                "channel": channel,
                "sample_count": len(local),
                "on_object_rate_before": before_rate,
                "on_object_rate_after": after_rate,
                "on_object_before": before_rate >= ON_OBJECT_RATE_MIN,
                "on_object_after": after_rate >= ON_OBJECT_RATE_MIN,
            }
        )
    before_count = sum(bool(row["on_object_before"]) for row in channel_rows)
    after_count = sum(bool(row["on_object_after"]) for row in channel_rows)

    forward = [row for row in selected if row["direction"] == "forward"]
    motion_ratios = [
        float(row["motion_attenuation_ratio"])
        for row in forward
        if row["motion_attenuation_ratio"] is not None
    ]
    _require(motion_ratios, "motion attenuation is undefined")
    checkpoint_pass = bool(
        defined_fraction >= MIN_DEFINED_ALIGNMENT_FRACTION
        and cosine["median"] > 0.0
        and error_delta["median"] < 0.0
        and after_count >= before_count
    )
    return {
        "eligibility": {
            "source": "bound_primary_evaluator_active_on_object_indices",
            "channel_indices": sorted(eligible),
            "channel_count": len(eligible),
        },
        "gradient_alignment": {
            "defined_count": len(defined),
            "total_count": len(selected),
            "defined_fraction": defined_fraction,
            "minimum_required_defined_fraction": MIN_DEFINED_ALIGNMENT_FRACTION,
            "cosine": cosine,
        },
        "one_cell_coordinate_step": {
            "step_size_normalized_xy": GRID_STEP_64,
            "error_before": error_before,
            "error_after": error_after,
            "error_delta": error_delta,
            "strict_error_decrease_fraction": float(np.mean([
                float(row["error_delta_objdiag"]) < 0.0 for row in selected
            ])),
            "projected_to_coordinate_domain_count": sum(
                bool(row["coordinate_projection_applied"]) for row in selected
            ),
            "active_on_object_channel_count_before": before_count,
            "active_on_object_channel_count_after": after_count,
            "on_object_rate_min": ON_OBJECT_RATE_MIN,
            "per_channel": channel_rows,
        },
        "static_parking": {
            "motion_attenuation_index": float(np.median(motion_ratios)),
            "motion_attenuation_ratio": _descriptive(
                motion_ratios,
                sample_unit="eligible_forward_validation_pair_channel_observed_over_oracle_motion",
            ),
        },
        "checkpoint_pass": checkpoint_pass,
        "checkpoint_pass_rule": {
            "median_cosine_strictly_positive": True,
            "median_one_cell_error_delta_strictly_negative": True,
            "active_on_object_channel_count_nonworse": True,
            "defined_alignment_fraction_at_least": MIN_DEFINED_ALIGNMENT_FRACTION,
        },
    }


def _load_json(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    try:
        value = json.loads(Path(record["absolute_path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterialAlignmentError(f"{name} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{name} is not an object")
    return value, record


def _validate_evaluator_result(
    path: Path,
    *,
    cell_id: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, record = _load_json(path, name="primary evaluator result")
    claimed = result.get("result_content_sha256")
    payload = dict(result)
    payload.pop("result_content_sha256", None)
    _require(claimed == eval_representation.canonical_sha256(payload),
             "primary evaluator result content hash differs")
    _require(result.get("case_id") == cell_id, "primary evaluator cell differs")
    checkpoint = result.get("validated_checkpoint")
    _require(isinstance(checkpoint, Mapping)
             and checkpoint.get("sha256") == checkpoint_sha256,
             "primary evaluator checkpoint differs")
    health = result.get("channel_health")
    _require(isinstance(health, Mapping)
             and isinstance(health.get("eligible_indices"), list)
             and len(health["eligible_indices"]) >= 2,
             "primary evaluator eligibility is unavailable")
    return result, record


def _validation_pairs(path: Path, *, object_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document, record = _load_json(path, name="validation pair index")
    claimed = document.get("content_hash_sha256")
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    _require(claimed == canonical_sha256(payload), "validation pair index content hash differs")
    pairs = document.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == 21,
             "validation pair index must contain 21 pairs")
    normalized = []
    for row in pairs:
        _require(isinstance(row, Mapping), "validation pair row is invalid")
        _require(
            row.get("model_name") == object_name
            and row.get("split") == "validation"
            and row.get("transform_family") == "roll"
            and row.get("physical_axis") == "world_z"
            and row.get("direction") == "forward"
            and row.get("stride") == 3
            and float(row.get("signed_generator")) == 6.0,
            "validation pair semantics differ",
        )
        source = row.get("src_frame_index")
        target = row.get("dst_frame_index")
        _require(type(source) is int and type(target) is int
                 and 0 <= source < 180 and 0 <= target < 180,
                 "validation frame index differs")
        normalized.append(dict(row))
    return normalized, record


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.detach().cpu()).tobytes(order="C"))
    return digest.hexdigest()


def _preprocess(images: Sequence[np.ndarray], indices: Sequence[int], device: torch.device) -> torch.Tensor:
    array = np.stack([images[index] for index in indices])
    batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std).to(device)


def _append_direction_rows(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    pair_rows: Sequence[Mapping[str, Any]],
    channels: int,
    metrics: Mapping[str, np.ndarray],
    anchor: np.ndarray,
    candidate: np.ndarray,
    descent: np.ndarray,
    angle_deg: float,
    eligible: set[int],
) -> None:
    observed_motion = candidate - anchor
    oracle_motion = apply_rotation(anchor, angle_deg) - anchor
    observed_norm = np.linalg.norm(observed_motion, axis=-1)
    oracle_norm = np.linalg.norm(oracle_motion, axis=-1)
    for batch_row, pair in enumerate(pair_rows):
        source = int(pair["src_frame_index"])
        target = int(pair["dst_frame_index"])
        if direction == "reverse":
            source, target = target, source
        for channel in range(channels):
            ratio = None
            if oracle_norm[batch_row, channel] > NORM_EPSILON:
                ratio = float(observed_norm[batch_row, channel] / oracle_norm[batch_row, channel])
            cosine = metrics["cosine"][batch_row, channel]
            rows.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "direction": direction,
                    "source_frame": source,
                    "target_frame": target,
                    "channel": channel,
                    "eligible_channel": channel in eligible,
                    "coordinate_descent_xy": descent[batch_row, channel].tolist(),
                    "coordinate_descent_norm": float(metrics["descent_norm"][batch_row, channel]),
                    "oracle_correction_norm": float(metrics["correction_norm"][batch_row, channel]),
                    "cosine_alignment": None if not np.isfinite(cosine) else float(cosine),
                    "error_before_objdiag": float(metrics["error_before_objdiag"][batch_row, channel]),
                    "error_after_objdiag": float(metrics["error_after_objdiag"][batch_row, channel]),
                    "error_delta_objdiag": float(metrics["error_delta_objdiag"][batch_row, channel]),
                    "coordinate_projection_applied": bool(metrics["projected"][batch_row, channel]),
                    "on_object_before": bool(metrics["on_object_before"][batch_row, channel]),
                    "on_object_after": bool(metrics["on_object_after"][batch_row, channel]),
                    "observed_motion_norm": float(observed_norm[batch_row, channel]),
                    "oracle_motion_norm": float(oracle_norm[batch_row, channel]),
                    "motion_attenuation_ratio": ratio,
                }
            )


def run_cell(
    *,
    source_repo_root: Path,
    audit_repo_root: Path,
    audit_commit: str,
    receipt_path: Path,
    cell_id: str,
    data_root: Path,
    evaluator_result_path: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    _require(type(batch_size) is int and batch_size >= 1, "batch size must be positive")
    audit_checkout = validate_checkout(audit_repo_root, expected_commit=audit_commit)
    audit_source = _file_record(Path(__file__))
    _require(Path(audit_source["absolute_path"]).is_relative_to(Path(audit_checkout["absolute_path"])),
             "audit code is outside the audit checkout")

    model, capability, checkpoint_load = (
        descriptor_attachment_runtime.load_authorized_descriptor_checkpoint(
            source_repo_root,
            receipt_path,
            cell_id=cell_id,
        )
    )
    source_checkout = validate_checkout(
        source_repo_root, expected_commit=capability.source_commit
    )
    evaluator_result, evaluator_record = _validate_evaluator_result(
        evaluator_result_path,
        cell_id=cell_id,
        checkpoint_sha256=capability.checkpoint_sha256,
    )
    pair_rows, validation_record = _validation_pairs(
        Path(capability.expected_training_arguments["val_pairs_index"]),
        object_name=str(capability.expected_base_config["object"]["name"]),
    )
    images, masks, _ = fresh_runtime._load_full_orbit(  # noqa: SLF001
        Path(data_root).resolve(strict=True), Path(source_checkout["absolute_path"])
    )
    diagonal = bbox_diagonals(masks)
    eligible_channels = [int(item) for item in evaluator_result["channel_health"]["eligible_indices"]]
    eligible = set(eligible_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    before_state = _state_digest(model)
    rows: list[dict[str, Any]] = []
    attachment_loss_batches = []
    for start in range(0, len(pair_rows), batch_size):
        local_pairs = pair_rows[start:start + batch_size]
        source_indices = [int(row["src_frame_index"]) for row in local_pairs]
        target_indices = [int(row["dst_frame_index"]) for row in local_pairs]
        source_batch = _preprocess(images, source_indices, device)
        target_batch = _preprocess(images, target_indices, device)
        with torch.inference_mode():
            source_flat, _, source_features = model.extractor(
                source_batch, return_descriptor_features=True
            )
            target_flat, _, target_features = model.extractor(
                target_batch, return_descriptor_features=True
            )
        source_coordinates = source_flat.reshape(-1, 10, 2).cpu()
        target_coordinates = target_flat.reshape(-1, 10, 2).cpu()
        source_features = source_features.cpu()
        target_features = target_features.cpu()

        target_candidate = target_coordinates.detach().clone().requires_grad_(True)
        forward_loss = descriptor_attachment.descriptor_attachment_one_way(
            source_features,
            source_coordinates,
            target_features,
            target_candidate,
        )
        forward_gradient = torch.autograd.grad(forward_loss, target_candidate)[0]
        source_candidate = source_coordinates.detach().clone().requires_grad_(True)
        reverse_loss = descriptor_attachment.descriptor_attachment_one_way(
            target_features,
            target_coordinates,
            source_features,
            source_candidate,
        )
        reverse_gradient = torch.autograd.grad(reverse_loss, source_candidate)[0]
        attachment_loss_batches.append(
            {
                "batch_start": start,
                "batch_pair_count": len(local_pairs),
                "forward": float(forward_loss.detach()),
                "reverse": float(reverse_loss.detach()),
            }
        )

        source_np = source_coordinates.numpy().astype(np.float64)
        target_np = target_coordinates.numpy().astype(np.float64)
        forward_descent = -forward_gradient.detach().numpy().astype(np.float64)
        reverse_descent = -reverse_gradient.detach().numpy().astype(np.float64)
        target_masks = masks[np.asarray(target_indices)]
        source_masks = masks[np.asarray(source_indices)]
        forward_metrics = direction_metrics(
            anchor=source_np,
            candidate=target_np,
            descent=forward_descent,
            angle_deg=6.0,
            candidate_masks=target_masks,
            candidate_bbox_diagonal=diagonal[np.asarray(target_indices)],
        )
        reverse_metrics = direction_metrics(
            anchor=target_np,
            candidate=source_np,
            descent=reverse_descent,
            angle_deg=-6.0,
            candidate_masks=source_masks,
            candidate_bbox_diagonal=diagonal[np.asarray(source_indices)],
        )
        _append_direction_rows(
            rows,
            direction="forward",
            pair_rows=local_pairs,
            channels=10,
            metrics=forward_metrics,
            anchor=source_np,
            candidate=target_np,
            descent=forward_descent,
            angle_deg=6.0,
            eligible=eligible,
        )
        _append_direction_rows(
            rows,
            direction="reverse",
            pair_rows=local_pairs,
            channels=10,
            metrics=reverse_metrics,
            anchor=target_np,
            candidate=source_np,
            descent=reverse_descent,
            angle_deg=-6.0,
            eligible=eligible,
        )
    after_state = _state_digest(model)
    _require(before_state == after_state, "checkpoint state changed during audit")

    document: dict[str, Any] = {
        "schema_version": CELL_SCHEMA_VERSION,
        "artifact_type": "descriptor_material_alignment_cell",
        "cell_id": cell_id,
        "recipe": cell_id.split("__", 1)[0],
        "arm": capability.arm,
        "seed": int(capability.expected_training_arguments["seed"]),
        "source_checkout": source_checkout,
        "audit_checkout": audit_checkout,
        "audit_source": audit_source,
        "inputs": {
            "completed_run_receipt": _file_record(receipt_path),
            "checkpoint": checkpoint_load,
            "primary_evaluator_result": evaluator_record,
            "validation_pair_index": validation_record,
            "data_root": str(Path(data_root).resolve(strict=True)),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "inference_device": device.type,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "coordinate_gradient_device": "cpu",
            "model_parameters_frozen": True,
            "model_state_sha256_before": before_state,
            "model_state_sha256_after": after_state,
            "optimizer_created": False,
            "training_performed": False,
            "checkpoint_selection_changed": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "protocol": {
            "partition": "frozen_validation",
            "pair_count": len(pair_rows),
            "directions": ["forward", "reverse"],
            "signed_generator_degrees": {"forward": 6.0, "reverse": -6.0},
            "descriptor_gradient": "exact_negative_coordinate_gradient_with_feature_values_detached",
            "one_cell_step": GRID_STEP_64,
            "coordinate_step_projection": "clip_to_closed_normalized_domain_[-1,1]",
            "mask_sampling": "endpoint_grid_nearest_pixel_matches_primary_evaluator",
            "statistical_scope": "descriptive_only_correlated_pairs_and_channels",
        },
        "attachment_loss_batches": attachment_loss_batches,
        "rows": rows,
        "summary": summarize_rows(rows, eligible_channels),
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    supplied_output = output_path.expanduser()
    output = supplied_output.parent.resolve(strict=True) / supplied_output.name
    data = canonical_json_bytes(document) + b"\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise MaterialAlignmentError("cell output must be a new path") from exc
    try:
        _write_all(descriptor, data)
    finally:
        os.close(descriptor)
    return document


def summarize_cells(
    cell_paths: Sequence[Path],
    *,
    primary_decision_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require(len(cell_paths) == 12, "summary requires exactly 12 cell artifacts")
    cells: dict[str, dict[str, Any]] = {}
    records = []
    for path in cell_paths:
        document, record = _load_json(path, name="material-alignment cell")
        claimed = document.get("content_hash_sha256")
        payload = dict(document)
        payload.pop("content_hash_sha256", None)
        _require(document.get("schema_version") == CELL_SCHEMA_VERSION
                 and claimed == canonical_sha256(payload),
                 "material-alignment cell identity differs")
        cell_id = document.get("cell_id")
        _require(isinstance(cell_id, str) and cell_id not in cells,
                 "material-alignment cell IDs differ")
        cells[cell_id] = document
        records.append({"cell_id": cell_id, **record,
                        "content_hash_sha256": claimed})
    expected = {
        f"{recipe}__{arm}__seed{seed}"
        for recipe in ("task55_clean", "task80_assisted")
        for arm in ("control", "attachment")
        for seed in (42, 43, 44)
    }
    _require(set(cells) == expected, "material-alignment cell set differs")

    primary, primary_record = _load_json(primary_decision_path, name="primary descriptor decision")
    claimed_primary = primary.get("content_hash_sha256")
    primary_payload = dict(primary)
    primary_payload.pop("content_hash_sha256", None)
    _require(claimed_primary == canonical_sha256(primary_payload)
             and primary.get("outcome") == "reject_exact_descriptor_attachment",
             "primary descriptor decision differs")

    recipe_arm = {}
    for recipe in ("task55_clean", "task80_assisted"):
        for arm in ("control", "attachment"):
            selected = [cells[f"{recipe}__{arm}__seed{seed}"] for seed in (42, 43, 44)]
            count = sum(bool(item["summary"]["checkpoint_pass"]) for item in selected)
            recipe_arm[f"{recipe}__{arm}"] = {
                "checkpoint_pass_count": count,
                "seed_count": 3,
                "recipe_arm_pass": count >= 2,
            }

    pair_rows = []
    parking_support = 0
    drift_deltas: dict[tuple[str, int], float] = {}
    for pair in primary["pairs"]:
        drift_deltas[(str(pair["recipe"]), int(pair["seed"]))] = (
            float(pair["values"]["attachment"]["canonical_drift"])
            - float(pair["values"]["control"]["canonical_drift"])
        )
    parking_reductions = []
    drift_values = []
    for recipe in ("task55_clean", "task80_assisted"):
        for seed in (42, 43, 44):
            control = cells[f"{recipe}__control__seed{seed}"]["summary"]["static_parking"]["motion_attenuation_index"]
            attachment = cells[f"{recipe}__attachment__seed{seed}"]["summary"]["static_parking"]["motion_attenuation_index"]
            relative_reduction = (float(control) - float(attachment)) / float(control)
            supports = relative_reduction >= 0.20
            parking_support += int(supports)
            drift_delta = drift_deltas[(recipe, seed)]
            parking_reduction = float(control) - float(attachment)
            parking_reductions.append(parking_reduction)
            drift_values.append(drift_delta)
            pair_rows.append({
                "recipe": recipe,
                "seed": seed,
                "control_motion_attenuation_index": float(control),
                "attachment_motion_attenuation_index": float(attachment),
                "attachment_relative_reduction": relative_reduction,
                "at_least_20_percent_reduction": supports,
                "canonical_drift_delta_attachment_minus_control": drift_delta,
            })
    parking_ranks = np.argsort(np.argsort(parking_reductions))
    drift_ranks = np.argsort(np.argsort(drift_values))
    rank_correlation = float(np.corrcoef(parking_ranks, drift_ranks)[0, 1])
    _require(math.isfinite(rank_correlation), "parking/drift rank correlation is undefined")
    parking_confirmed = parking_support >= 5 and rank_correlation > 0.0

    control_passes = all(
        recipe_arm[f"{recipe}__control"]["recipe_arm_pass"]
        for recipe in ("task55_clean", "task80_assisted")
    )
    attachment_passes = all(
        recipe_arm[f"{recipe}__attachment"]["recipe_arm_pass"]
        for recipe in ("task55_clean", "task80_assisted")
    )
    any_control_recipe_fails = any(
        not recipe_arm[f"{recipe}__control"]["recipe_arm_pass"]
        for recipe in ("task55_clean", "task80_assisted")
    )
    if control_passes and attachment_passes:
        branch = "local_direction_sensible_audit_scale_optimizer_and_epoch_matching"
    elif control_passes and not attachment_passes:
        branch = "self_induced_moving_feature_field_pathology"
    elif any_control_recipe_fails:
        branch = "construct_mismatch_do_not_sweep_lambda"
    else:
        branch = "mechanism_unresolved_do_not_progress_automatically"

    document: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "artifact_type": "descriptor_material_alignment_decision",
        "source_commit": next(iter(cells.values()))["source_checkout"]["commit"],
        "audit_commit": next(iter(cells.values()))["audit_checkout"]["commit"],
        "primary_descriptor_decision": primary_record,
        "cell_artifacts": sorted(records, key=lambda item: item["cell_id"]),
        "recipe_arm_results": recipe_arm,
        "static_parking": {
            "pairs": pair_rows,
            "at_least_20_percent_reduction_count": parking_support,
            "required_count": 5,
            "rank_correlation_parking_delta_vs_drift_delta": rank_correlation,
            "ranking_direction_matches": rank_correlation > 0.0,
            "parking_mechanism_confirmed": parking_confirmed,
        },
        "branch": branch,
        "training_authorized": False,
        "lambda_sweep_authorized": False,
        "next_scientific_phase_started": False,
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    supplied_output = output_path.expanduser()
    output = supplied_output.parent.resolve(strict=True) / supplied_output.name
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise MaterialAlignmentError("decision output must be a new path") from exc
    try:
        _write_all(descriptor, canonical_json_bytes(document) + b"\n")
    finally:
        os.close(descriptor)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("cell")
    cell.add_argument("--source-repo-root", type=Path, required=True)
    cell.add_argument("--audit-repo-root", type=Path, required=True)
    cell.add_argument("--audit-commit", required=True)
    cell.add_argument("--receipt", type=Path, required=True)
    cell.add_argument("--cell-id", required=True)
    cell.add_argument("--data-root", type=Path, required=True)
    cell.add_argument("--evaluator-result", type=Path, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--batch-size", type=int, default=7)
    summary = commands.add_parser("summarize")
    summary.add_argument("--cell", type=Path, action="append", required=True)
    summary.add_argument("--primary-decision", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        document = run_cell(
            source_repo_root=args.source_repo_root,
            audit_repo_root=args.audit_repo_root,
            audit_commit=args.audit_commit,
            receipt_path=args.receipt,
            cell_id=args.cell_id,
            data_root=args.data_root,
            evaluator_result_path=args.evaluator_result,
            output_path=args.output,
            batch_size=args.batch_size,
        )
    else:
        document = summarize_cells(
            args.cell,
            primary_decision_path=args.primary_decision,
            output_path=args.output,
        )
    print(json.dumps({
        "output": str(args.output),
        "content_hash_sha256": document["content_hash_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
