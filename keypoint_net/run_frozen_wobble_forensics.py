"""Run source-bound, frozen-checkpoint full-orbit wobble forensics.

This command performs inference only.  It never constructs an optimizer and it
refuses to overwrite an output directory.  The model checkpoint is hashed and
loaded from the same open file descriptor with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

try:
    from .frozen_wobble_forensics import (
        canonicalize,
        coordinate_dynamics,
        deterministic_spike_frames,
        heatmap_topology,
        phase_metrics,
        rotate_points,
        spatial_expectation_float32,
    )
    from .model import PhaseAModel
except ImportError:  # Support direct ``python keypoint_net/...`` execution.
    from frozen_wobble_forensics import (  # type: ignore
        canonicalize,
        coordinate_dynamics,
        deterministic_spike_frames,
        heatmap_topology,
        phase_metrics,
        rotate_points,
        spatial_expectation_float32,
    )
    from model import PhaseAModel  # type: ignore


SCHEMA_VERSION = "frozen_wobble_forensics.v1"
EXPECTED_EVALUATION_COMMIT = "43dee6362c22928b3ce2d481def491d981754ab1"
EXPECTED_TRAINING_COMMIT = "fa7986f4c534718fc2695cd9e8ef73177ef3ff65"
EXPECTED_DATASET_BINDING = "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
EXPECTED_FRAME_COUNT = 180
EXPECTED_CHANNEL_COUNT = 10
EXPECTED_IMAGE_SIZE = 512
EXPECTED_HEATMAP_SIZE = 64
EXPECTED_FRAME_STEP_DEG = 2.0
EXPECTED_STRIDE = 3
EXPECTED_STRIDE_DEG = 6.0
COORDINATE_CONSISTENCY_TOLERANCE = 1e-4
CELL_RE = re.compile(r"^(task55_clean|task80_assisted)__(control|ocr_zncc)__seed(42|43|44)$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
IMPLEMENTATION_SOURCE_RELATIVE_PATHS = (
    "keypoint_net/run_frozen_wobble_forensics.py",
    "keypoint_net/frozen_wobble_forensics.py",
    "keypoint_net/model.py",
)


class FrozenForensicError(ValueError):
    """Fail-closed binding or semantic error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenForensicError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_binding(repo_root: Path) -> tuple[str, dict[str, Any]]:
    """Bind the committed forensic implementation without conflating its HEAD
    with the older outcome evaluator commit.

    The completed outcomes remain locked to ``EXPECTED_EVALUATION_COMMIT`` in
    ``_validate_outcome``.  The forensic implementation may be a later commit,
    but it must descend from that exact commit and its runtime source files must
    be byte-for-byte clean relative to its own HEAD.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_EVALUATION_COMMIT, head],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        ancestry.returncode == 0,
        "forensic implementation HEAD does not descend from the source-bound evaluator commit",
    )
    cleanliness = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCE_RELATIVE_PATHS],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        cleanliness.returncode == 0,
        "forensic runtime source files differ from the recorded implementation HEAD",
    )
    source_records = {
        relative_path: _regular_file_record(repo_root / relative_path)
        for relative_path in IMPLEMENTATION_SOURCE_RELATIVE_PATHS
    }
    return head, source_records


def _regular_file_record(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    metadata = path.stat()
    _require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
    return {
        "absolute_path": str(path),
        "sha256": _file_sha256(path),
        "size_bytes": metadata.st_size,
    }


def _load_json(path_value: Path | str, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _regular_file_record(path_value)
    try:
        value = json.loads(Path(record["absolute_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise FrozenForensicError(f"cannot parse {name}") from exc
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value, record


def _same_fd_checkpoint_load(
    path_value: Path | str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    path = Path(path_value).expanduser().resolve(strict=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "checkpoint is not a regular file")
        _require(metadata.st_size == expected_size, "checkpoint byte size differs from outcome binding")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            observed_sha256 = digest.hexdigest()
            _require(observed_sha256 == expected_sha256, "checkpoint SHA-256 differs from outcome binding")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    finally:
        os.close(descriptor)
    _require(isinstance(payload, Mapping), "checkpoint payload is not a mapping")
    return payload, {
        "absolute_path": str(path),
        "sha256": observed_sha256,
        "size_bytes": metadata.st_size,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(cpu.shape)))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_outcome(
    outcome: Mapping[str, Any], *, checkpoint_role: str
) -> tuple[str, Mapping[str, Any]]:
    cell_id = outcome.get("cell_id")
    _require(isinstance(cell_id, str) and CELL_RE.fullmatch(cell_id) is not None, "outcome cell ID differs")
    _require(outcome.get("artifact_type") == "frozen_paired_ocr_zncc_outcome_cell", "outcome artifact type differs")
    _require(outcome.get("evaluation_status") == "complete", "outcome cell is incomplete")
    _require(outcome.get("evaluation_source_commit") == EXPECTED_EVALUATION_COMMIT, "evaluation source commit differs")
    _require(outcome.get("source_commit") == EXPECTED_TRAINING_COMMIT, "training source commit differs")
    _require(outcome.get("training_or_weight_update_performed") is False, "outcome unexpectedly reports training")
    _require(outcome.get("lineage", {}).get("dataset_binding_sha256") == EXPECTED_DATASET_BINDING, "dataset binding differs")
    results = outcome.get("checkpoint_results")
    _require(isinstance(results, Mapping) and checkpoint_role in results, "checkpoint role is absent from outcome")
    role_result = results[checkpoint_role]
    _require(isinstance(role_result, Mapping), "checkpoint result is invalid")
    checkpoint = role_result.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint binding is missing")
    _require(checkpoint.get("checkpoint_role") == checkpoint_role, "checkpoint role binding differs")
    _require(isinstance(checkpoint.get("sha256"), str) and SHA_RE.fullmatch(checkpoint["sha256"]), "checkpoint SHA-256 is invalid")
    _require(isinstance(checkpoint.get("size_bytes"), int) and checkpoint["size_bytes"] > 0, "checkpoint size is invalid")
    return cell_id, checkpoint


def _validate_calibration(
    calibration: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    _require(record["sha256"] == expected_sha256, "calibration file SHA-256 differs")
    _require(calibration.get("schema_version") == "frozen_wobble_oracle_calibration.v1_2", "calibration schema differs")
    _require(calibration.get("model_checkpoint_opened") is False, "calibration is not model blind")
    _require(calibration.get("all_semantic_assertions_pass") is True, "calibration semantic assertions did not all pass")
    _require(all(calibration.get("semantic_assertions", {}).values()), "calibration contains a failed assertion")
    _require(calibration.get("content_hash_sha256") == _content_hash(calibration), "calibration content hash differs")


def _construct_frozen_model(
    payload: Mapping[str, Any],
    *,
    cell_id: str,
    checkpoint_role: str,
    expected_epoch: int,
) -> tuple[PhaseAModel, Mapping[str, Any]]:
    expected_keys = (
        {
            "epoch", "model_state_dict", "optimizer_state_dict", "loss",
            "selection_loss_name", "base_validation_loss", "attachment_validation_loss",
            "ocr_zncc_validation_loss", "augmented_validation_loss", "config",
        }
        if checkpoint_role == "best_model"
        else {
            "epoch", "model_state_dict", "optimizer_state_dict", "loss",
            "base_training_loss", "attachment_training_loss", "ocr_zncc_training_loss",
            "ocr_zncc_accepted_match_coverage", "config",
        }
    )
    _require(set(payload) == expected_keys, "checkpoint payload keys differ from completed protocol")
    _require(int(payload.get("epoch", -1)) == expected_epoch, "checkpoint epoch differs from outcome binding")
    config = payload.get("config")
    _require(isinstance(config, Mapping), "checkpoint configuration is missing")
    _require(config.get("cell_id") == cell_id, "checkpoint embedded cell ID differs")
    _require(config.get("source_commit") == EXPECTED_TRAINING_COMMIT, "checkpoint embedded source commit differs")
    _require(int(config.get("img_size", -1)) == EXPECTED_IMAGE_SIZE, "checkpoint input size differs")
    _require(int(config.get("heatmap_res", -1)) == EXPECTED_HEATMAP_SIZE, "checkpoint heatmap size differs")
    _require(int(config.get("num_keypoints", -1)) == EXPECTED_CHANNEL_COUNT, "checkpoint channel count differs")
    _require(float(config.get("temperature", float("nan"))) == 1.0, "checkpoint temperature differs")
    _require(int(config.get("frame_skip", -1)) == EXPECTED_STRIDE, "checkpoint frame skip differs")

    model = PhaseAModel(
        num_keypoints=int(config["num_keypoints"]),
        base_channels=int(config["base_channels"]),
        temperature=float(config["temperature"]),
        num_action_classes=int(config["num_action_classes"]),
        padding_mode=str(config["padding_mode"]),
        operator_type=str(config["operator_type"]),
        learn_inverse_operator=bool(config["learn_inverse_operator"]),
        heatmap_res=int(config["heatmap_res"]),
    )
    state = payload.get("model_state_dict")
    _require(isinstance(state, Mapping) and set(state) == set(model.state_dict()), "model-state keys differ")
    for name, target in model.state_dict().items():
        source = state[name]
        _require(torch.is_tensor(source), f"checkpoint tensor {name} is invalid")
        _require(source.device.type == "cpu" and source.shape == target.shape and source.dtype == target.dtype, f"checkpoint tensor {name} metadata differs")
        _require(bool(torch.isfinite(source).all()), f"checkpoint tensor {name} is non-finite")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, config


def _validate_pair_index(pair_index: Mapping[str, Any]) -> dict[str, list[int]]:
    _require(pair_index.get("artifact_type") == "representation_pair_index", "pair index artifact type differs")
    _require(pair_index.get("dataset_binding_sha256") == EXPECTED_DATASET_BINDING, "pair index dataset binding differs")
    transform = pair_index.get("transform")
    _require(isinstance(transform, Mapping), "pair index transform is missing")
    expected_transform = {
        "cyclic": True,
        "direction": "forward",
        "expected_2d_family": "planar_rotation_about_projected_center",
        "family": "roll",
        "generator_units": "degrees",
        "physical_axis": "world_z",
        "signed_generator": EXPECTED_STRIDE_DEG,
        "stride": EXPECTED_STRIDE,
        "stride_units": "frames",
    }
    _require(dict(transform) == expected_transform, "pair index transform semantics differ")
    partition = pair_index.get("frame_partition")
    _require(isinstance(partition, Mapping), "frame partition is missing")
    result = {
        "holdout": [int(value) for value in partition.get("holdout_frame_indices", [])],
        "guard": [int(value) for value in partition.get("guard_frame_indices", [])],
        "train": [int(value) for value in partition.get("train_frame_indices", [])],
    }
    combined = sorted(value for values in result.values() for value in values)
    _require(combined == list(range(EXPECTED_FRAME_COUNT)), "frame partition does not cover 0..179 exactly")
    rows = [row for row in pair_index.get("pairs", []) if row.get("model_name") == "engineers_hammer_vray"]
    _require(len(rows) == 147, "hammer training edge count differs")
    _require(all(int(row["dst_frame_index"]) - int(row["src_frame_index"]) == 3 for row in rows), "hammer pair graph contains a non-3 edge")
    return result


def _load_corpus(
    data_root_value: Path | str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    data_root = Path(data_root_value).expanduser().resolve(strict=True)
    object_root = data_root / "train" / "engineers_hammer_vray"
    meta_path = object_root / "meta.jsonl"
    _require(meta_path.is_file(), "hammer metadata is missing")
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(rows) == EXPECTED_FRAME_COUNT, "metadata row count differs from 180")
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    theta: list[float] = []
    indices: list[int] = []
    inventory_digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        _require(int(row.get("frame_index", -1)) == expected_index, "metadata frame order differs")
        _require(float(row.get("theta_deg", float("nan"))) == expected_index * EXPECTED_FRAME_STEP_DEG, "metadata angle differs")
        _require(row.get("operator_name") == "tdw_world_z_roll", "metadata operator differs")
        _require(row.get("tdw_axis") == "roll" and row.get("is_world") is True and row.get("use_centroid") is True, "metadata world-Z semantics differ")
        _require(row.get("valid") is True, "metadata contains an invalid frame")
        image_path = (object_root / str(row["image_relpath"])).resolve(strict=True)
        mask_path = (object_root / str(row["mask_relpath"])).resolve(strict=True)
        image_record = _regular_file_record(image_path)
        mask_record = _regular_file_record(mask_path)
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
        _require(image.shape == (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE, 3), "RGB frame shape differs")
        _require(mask.shape == (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), "mask shape differs")
        images.append(image)
        masks.append(mask)
        theta.append(float(row["theta_deg"]))
        indices.append(expected_index)
        compact = {
            "frame_index": expected_index,
            "theta_deg": float(row["theta_deg"]),
            "image_relpath": str(row["image_relpath"]),
            "image_sha256": image_record["sha256"],
            "mask_relpath": str(row["mask_relpath"]),
            "mask_sha256": mask_record["sha256"],
        }
        inventory_digest.update(_canonical_bytes(compact))
        records.append(compact)
    return images, np.stack(masks), np.asarray(theta), indices, {
        "data_root": str(data_root),
        "object_root": str(object_root),
        "metadata": _regular_file_record(meta_path),
        "frame_inventory_sha256": inventory_digest.hexdigest(),
        "frame_records": records,
    }


def _infer(model: PhaseAModel, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require(not model.training and all(not p.requires_grad for p in model.parameters()), "model is not frozen in eval mode")
    before = _state_sha256(model)
    means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    supplied_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            array = np.stack(images[start : start + 8])
            batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            batch = batch.sub_(means).div_(stds)
            flattened, logits = model.extractor(batch)
            supplied_batches.append(flattened.reshape(batch.shape[0], EXPECTED_CHANNEL_COUNT, 2).cpu().numpy().copy())
            logit_batches.append(logits.cpu().numpy().copy())
    supplied = np.concatenate(supplied_batches).astype(np.float32, copy=False)
    logits = np.concatenate(logit_batches).astype(np.float32, copy=False)
    _require(supplied.shape == (EXPECTED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, 2), "supplied coordinate shape differs")
    _require(logits.shape == (EXPECTED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, EXPECTED_HEATMAP_SIZE, EXPECTED_HEATMAP_SIZE), "logit shape differs")
    derived = spatial_expectation_float32(logits)
    maximum_error = float(np.max(np.abs(derived - supplied)))
    _require(maximum_error <= COORDINATE_CONSISTENCY_TOLERANCE, "supplied and recomputed coordinates disagree")
    after = _state_sha256(model)
    _require(before == after, "model state changed during inference")
    return derived, logits, {
        "device": "cpu",
        "batch_size": 8,
        "inference_mode": True,
        "optimizer_constructed": False,
        "training_or_weight_update_performed": False,
        "coordinate_estimator": "NumPy float32 endpoint-grid expectation checked against production PyTorch extractor output",
        "maximum_supplied_vs_recomputed_coordinate_error": maximum_error,
        "coordinate_consistency_tolerance": COORDINATE_CONSISTENCY_TOLERANCE,
        "coordinate_consistency_policy_kind": "implementation consistency, not a scientific quality threshold",
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "model_state_unchanged": True,
        "points_sha256": hashlib.sha256(derived.tobytes(order="C")).hexdigest(),
        "logits_sha256": hashlib.sha256(logits.tobytes(order="C")).hexdigest(),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 2, "summary expects (sample, channel)")
    return {
        "n_per_channel": int(array.shape[0]),
        "sample_unit": "correlated frames or overlapping frame differences",
        "descriptive_not_inferential": True,
        "per_channel": [
            {
                "channel": channel,
                "mean": float(np.mean(array[:, channel])),
                "median": float(np.median(array[:, channel])),
                "q90": float(np.quantile(array[:, channel], 0.90)),
                "q95": float(np.quantile(array[:, channel], 0.95)),
                "q99": float(np.quantile(array[:, channel], 0.99)),
                "maximum": float(np.max(array[:, channel])),
            }
            for channel in range(array.shape[1])
        ],
    }


def _coordinate_summary(points: np.ndarray) -> dict[str, Any]:
    dynamics = coordinate_dynamics(points)
    centred = points - np.mean(points, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=0))
    cyclic_second = np.linalg.norm(
        np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0),
        axis=-1,
    )
    return {
        "unit": "endpoint-aligned normalized coordinate; multiply by 255.5 for 512-input pixels or 31.5 for 64-heatmap cells",
        "canonical_rms_per_channel": [float(value) for value in rms],
        "canonical_radius_about_mean": _summary(dynamics["radius_about_mean"]),
        "adjacent_plus2_step": _summary(dynamics["adjacent_step"]),
        "within_residue_plus6_step": _summary(dynamics["within_residue_step"]),
        "noncyclic_second_difference": _summary(dynamics["second_difference"]),
        "cyclic_second_difference": _summary(cyclic_second),
    }


def _stratified_summary(points: np.ndarray, partition: Mapping[str, list[int]]) -> dict[str, Any]:
    result: dict[str, Any] = {"full_nonseam": _coordinate_summary(points)}
    for name in ("holdout", "train"):
        indices = np.asarray(partition[name], dtype=np.int64)
        _require(np.all(np.diff(indices) == 1), f"{name} partition must be contiguous")
        result[name] = _coordinate_summary(points[indices])
    guard_segments = []
    for segment in ((24, 25, 26), (177, 178, 179)):
        values = points[np.asarray(segment, dtype=np.int64)]
        centred = values - np.mean(values, axis=0, keepdims=True)
        guard_segments.append({
            "frame_indices": list(segment),
            "canonical_rms_per_channel": [
                float(value)
                for value in np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=0))
            ],
            "canonical_radius_about_mean": _summary(np.linalg.norm(centred, axis=-1)),
            "adjacent_plus2_step": _summary(np.linalg.norm(np.diff(values, axis=0), axis=-1)),
            "single_second_difference_per_channel": [
                float(value)
                for value in np.linalg.norm(values[2] - 2.0 * values[1] + values[0], axis=-1)
            ],
        })
    result["guard"] = {
        "segments": guard_segments,
        "note": "three-frame guard segments are too short for the complete phase suite",
    }
    result["cyclic_seam_step_per_channel"] = [
        float(value) for value in np.linalg.norm(points[0] - points[-1], axis=-1)
    ]
    return result


def _mask_localization(points: np.ndarray, masks: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:
        raise FrozenForensicError("scipy is required for signed mask distance") from exc
    pixel = (points + 1.0) * 0.5 * (EXPECTED_IMAGE_SIZE - 1)
    inside = np.zeros(points.shape[:2], dtype=bool)
    signed_distance = np.zeros(points.shape[:2], dtype=np.float64)
    border_distance = np.minimum.reduce(
        (pixel[..., 0], pixel[..., 1], (EXPECTED_IMAGE_SIZE - 1) - pixel[..., 0], (EXPECTED_IMAGE_SIZE - 1) - pixel[..., 1])
    )
    bbox_diagonal = np.empty(EXPECTED_FRAME_COUNT, dtype=np.float64)
    for frame in range(EXPECTED_FRAME_COUNT):
        mask = masks[frame]
        ys, xs = np.where(mask)
        _require(xs.size > 0, f"empty object mask at frame {frame}")
        bbox_diagonal[frame] = math.hypot(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        distance_inside = distance_transform_edt(mask)
        distance_outside = distance_transform_edt(~mask)
        for channel in range(EXPECTED_CHANNEL_COUNT):
            x = pixel[frame, channel, 0]
            y = pixel[frame, channel, 1]
            if 0.0 <= x <= EXPECTED_IMAGE_SIZE - 1 and 0.0 <= y <= EXPECTED_IMAGE_SIZE - 1:
                xi = int(np.clip(np.rint(x), 0, EXPECTED_IMAGE_SIZE - 1))
                yi = int(np.clip(np.rint(y), 0, EXPECTED_IMAGE_SIZE - 1))
                inside[frame, channel] = bool(mask[yi, xi])
                signed_distance[frame, channel] = (
                    float(distance_inside[yi, xi]) if inside[frame, channel]
                    else -float(distance_outside[yi, xi])
                )
            else:
                signed_distance[frame, channel] = -float(EXPECTED_IMAGE_SIZE)
    return {
        "on_object_rate_per_channel": [float(value) for value in np.mean(inside, axis=0)],
        "minimum_signed_mask_distance_px_per_channel": [float(value) for value in np.min(signed_distance, axis=0)],
        "minimum_image_border_distance_px_per_channel": [float(value) for value in np.min(border_distance, axis=0)],
        "out_of_bounds_rate_per_channel": [float(value) for value in np.mean(border_distance < 0.0, axis=0)],
        "mask_bbox_diagonal_px": {
            "minimum": float(np.min(bbox_diagonal)),
            "median": float(np.median(bbox_diagonal)),
            "maximum": float(np.max(bbox_diagonal)),
        },
        "statistics": "descriptive across 180 correlated frames",
    }, {
        "on_object": inside,
        "signed_mask_distance_px": signed_distance,
        "image_border_distance_px": border_distance,
        "mask_bbox_diagonal_px": bbox_diagonal,
    }


def _duplicate_diagnostics(points: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pairs = [(left, right) for left in range(EXPECTED_CHANNEL_COUNT) for right in range(left + 1, EXPECTED_CHANNEL_COUNT)]
    distance = np.stack([np.linalg.norm(points[:, left] - points[:, right], axis=-1) for left, right in pairs], axis=1)
    overlap = np.stack([
        np.sum(np.minimum(probabilities[:, left], probabilities[:, right]), axis=(-2, -1))
        for left, right in pairs
    ], axis=1)
    pair_rows = []
    for index, (left, right) in enumerate(pairs):
        pair_rows.append({
            "left_channel": left,
            "right_channel": right,
            "distance_normalized_min": float(np.min(distance[:, index])),
            "distance_normalized_median": float(np.median(distance[:, index])),
            "heatmap_probability_overlap_mean": float(np.mean(overlap[:, index])),
            "heatmap_probability_overlap_max": float(np.max(overlap[:, index])),
        })
    return {
        "fixed_channel_only_for_pass_fail": True,
        "pair_count": len(pairs),
        "pairs": pair_rows,
        "nearest_other_channel_distance_per_channel": [
            float(min(row["distance_normalized_median"] for row in pair_rows if channel in (row["left_channel"], row["right_channel"])))
            for channel in range(EXPECTED_CHANNEL_COUNT)
        ],
        "statistics": "descriptive across 180 correlated frames; no Hungarian reassignment is used to pass a channel",
    }, {
        "pair_channel_indices": np.asarray(pairs, dtype=np.int64),
        "pairwise_distance_normalized": distance,
        "pairwise_heatmap_probability_overlap": overlap,
    }


def _operator_diagnostics(model: PhaseAModel, points: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    operator = model.operator
    _require(hasattr(operator, "A") and hasattr(operator, "bias"), "operator is not the expected shared affine form")
    learned_A = operator.A.detach().cpu().numpy().astype(np.float64)
    learned_b = operator.bias.detach().cpu().numpy().astype(np.float64)
    radians = math.radians(EXPECTED_STRIDE_DEG)
    expected_A = np.asarray(((math.cos(radians), -math.sin(radians)), (math.sin(radians), math.cos(radians))))
    source = points[:-EXPECTED_STRIDE]
    target = points[EXPECTED_STRIDE:]
    predicted = source @ learned_A.T + learned_b
    physical = rotate_points(source, EXPECTED_STRIDE_DEG)
    prediction_error = np.linalg.norm(predicted - target, axis=-1)
    physical_search_centre_error = np.linalg.norm(predicted - physical, axis=-1)
    determinant = float(np.linalg.det(learned_A))
    learned_angle = math.degrees(math.atan2(learned_A[1, 0], learned_A[0, 0]))
    return {
        "learned_A": learned_A.tolist(),
        "learned_bias": learned_b.tolist(),
        "expected_plus6_rotation_A": expected_A.tolist(),
        "matrix_frobenius_error": float(np.linalg.norm(learned_A - expected_A, ord="fro")),
        "learned_angle_deg_atan2": learned_angle,
        "angle_error_deg": learned_angle - EXPECTED_STRIDE_DEG,
        "determinant": determinant,
        "bias_norm_normalized": float(np.linalg.norm(learned_b)),
        "checkpoint_prediction_error": _summary(prediction_error),
        "physical_search_centre_error": _summary(physical_search_centre_error),
    }, {
        "operator_predicted_plus6_coordinate": predicted,
        "physical_plus6_coordinate": physical,
        "operator_prediction_error": prediction_error,
        "operator_physical_search_centre_error": physical_search_centre_error,
    }


def _heatmap_summary(topology: Mapping[str, Any], hard_canonical: np.ndarray, hard_thresholds: Mapping[str, float]) -> dict[str, Any]:
    hard_dynamics = coordinate_dynamics(hard_canonical)
    jump_threshold = float(hard_thresholds["adjacent_step_max"])
    return {
        "soft_hard_distance_cells": _summary(np.asarray(topology["soft_hard_distance_cells"])),
        "entropy_nats": _summary(np.asarray(topology["entropy_nats"])),
        "effective_support_cells": _summary(np.asarray(topology["effective_support_cells"])),
        "rms_width_cells": _summary(np.asarray(topology["rms_width_cells"])),
        "anisotropy_ratio": _summary(np.asarray(topology["anisotropy_ratio"])),
        "first_peak_probability": _summary(np.asarray(topology["first_peak_probability"])),
        "separated_peak_logit_margin_radius4": _summary(np.asarray(topology["separated_peaks"]["4"]["logit_margin"])),
        "hard_canonical_adjacent_step": _summary(hard_dynamics["adjacent_step"]),
        "hard_jump_threshold_normalized_from_r64_oracle": jump_threshold,
        "hard_jump_rate_per_channel": [
            float(value) for value in np.mean(hard_dynamics["adjacent_step"] > jump_threshold, axis=0)
        ],
    }


def _save_raw_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    np.savez_compressed(path, **arrays)
    return _regular_file_record(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    head, source_records = _implementation_binding(repo_root)

    calibration, calibration_record = _load_json(args.calibration, name="oracle calibration")
    _validate_calibration(calibration, calibration_record, expected_sha256=args.expected_calibration_sha256)
    outcome, outcome_record = _load_json(args.outcome_json, name="outcome JSON")
    cell_id, checkpoint_binding = _validate_outcome(outcome, checkpoint_role=args.checkpoint_role)
    pair_index, pair_index_record = _load_json(args.pair_index, name="pair index")
    partition = _validate_pair_index(pair_index)

    payload, checkpoint_record = _same_fd_checkpoint_load(
        args.checkpoint,
        expected_sha256=str(checkpoint_binding["sha256"]),
        expected_size=int(checkpoint_binding["size_bytes"]),
    )
    model, checkpoint_config = _construct_frozen_model(
        payload,
        cell_id=cell_id,
        checkpoint_role=args.checkpoint_role,
        expected_epoch=int(checkpoint_binding["epoch"]),
    )
    _require(_state_sha256(model) == checkpoint_binding["model_state_sha256"], "loaded model-state hash differs from prior bound evaluator")

    images, masks, theta, frame_indices, corpus = _load_corpus(args.data_root)
    points, logits, inference = _infer(model, images)
    canonical = canonicalize(points, theta)
    wrong_sign = canonicalize(points, theta, sign=1)
    topology = heatmap_topology(logits)
    hard = np.asarray(topology["hard_coordinate"])
    hard_canonical = canonicalize(hard, theta)
    hard_wrong_sign = canonicalize(hard, theta, sign=1)

    localization, localization_arrays = _mask_localization(points, masks)
    duplicates, duplicate_arrays = _duplicate_diagnostics(canonical, np.asarray(topology["probabilities"]))
    operator, operator_arrays = _operator_diagnostics(model, points)
    hard_thresholds = calibration["frozen_thresholds"]["hard_r64_quantized"]
    soft_thresholds = calibration["frozen_thresholds"]["soft_single_gaussian"]

    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_arrays = {
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "theta_deg": theta,
        "soft_coordinate": points,
        "soft_canonical_coordinate": canonical,
        "hard_coordinate": hard,
        "hard_canonical_coordinate": hard_canonical,
        "logits": logits,
        "soft_hard_distance_cells": np.asarray(topology["soft_hard_distance_cells"]),
        "entropy_nats": np.asarray(topology["entropy_nats"]),
        "effective_support_cells": np.asarray(topology["effective_support_cells"]),
        "rms_width_cells": np.asarray(topology["rms_width_cells"]),
        "anisotropy_ratio": np.asarray(topology["anisotropy_ratio"]),
        "first_peak_probability": np.asarray(topology["first_peak_probability"]),
        "hard_x_cell": np.asarray(topology["hard_x_cell"]),
        "hard_y_cell": np.asarray(topology["hard_y_cell"]),
        **localization_arrays,
        **duplicate_arrays,
        **operator_arrays,
    }
    for radius, rows in topology["separated_peaks"].items():
        for name, value in rows.items():
            raw_arrays[f"separated_peak_r{radius}_{name}"] = np.asarray(value)
    raw_record = _save_raw_arrays(output_dir / "raw_forensic_arrays.npz", raw_arrays)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "source_bound_frozen_checkpoint_full_orbit_forensics",
        "cell_id": cell_id,
        "recipe": outcome["recipe"],
        "arm": outcome["arm"],
        "seed": outcome["seed"],
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_epoch": int(checkpoint_binding["epoch"]),
        "training_or_weight_update_performed": False,
        "classification_status": "descriptive_forensic_smoke_not_yet_final_success_contract",
        "bindings": {
            "implementation_head": head,
            "required_evaluation_ancestor_commit": EXPECTED_EVALUATION_COMMIT,
            "implementation_sources": source_records,
            "outcome_json": outcome_record,
            "checkpoint": checkpoint_record,
            "calibration": calibration_record,
            "pair_index": pair_index_record,
            "corpus": corpus,
            "raw_arrays": raw_record,
        },
        "checkpoint_configuration": {
            key: checkpoint_config[key]
            for key in (
                "cell_id", "source_commit", "seed", "img_size", "heatmap_res", "temperature",
                "frame_skip", "num_keypoints", "base_channels", "padding_mode", "operator_type",
                "learn_inverse_operator", "lambda_smooth", "lambda_disp", "lambda_ent", "lambda_inv",
                "lambda_cycle", "lambda_attach", "lambda_ocr_zncc",
            )
        },
        "geometry": {
            "physical_transform": "TDW world-Z roll",
            "theta_metadata_deg": [float(value) for value in theta],
            "pivot_normalized": [0.0, 0.0],
            "canonical_operation": "R(-theta) p(theta)",
            "wrong_sign_audit": {
                "soft_canonical_rms_per_channel": _coordinate_summary(wrong_sign)["canonical_rms_per_channel"],
                "hard_canonical_rms_per_channel": _coordinate_summary(hard_wrong_sign)["canonical_rms_per_channel"],
            },
            "frame_partition": partition,
            "cyclic_seam_reported_separately": True,
        },
        "inference": inference,
        "soft_coordinate_metrics": _stratified_summary(canonical, partition),
        "hard_coordinate_metrics": _stratified_summary(hard_canonical, partition),
        "phase_metrics": {
            "full": phase_metrics(canonical, theta, frame_indices),
            "holdout": phase_metrics(canonical[:24], theta[:24], frame_indices[:24]),
            "train": phase_metrics(canonical[27:177], theta[27:177], frame_indices[27:177]),
        },
        "heatmap_topology": _heatmap_summary(topology, hard_canonical, hard_thresholds),
        "localization": localization,
        "distinctness": duplicates,
        "operator": operator,
        "spike_frames": {
            "selector": "top five non-seam canonical second differences per fixed channel; magnitude descending then frame index ascending",
            "soft": deterministic_spike_frames(canonical, top_k=5),
            "hard": deterministic_spike_frames(hard_canonical, top_k=5),
        },
        "oracle_envelope_exceedance": {
            "hard_thresholds": hard_thresholds,
            "soft_single_gaussian_thresholds": soft_thresholds,
            "single_gaussian_heatmap_topology_thresholds": calibration["frozen_thresholds"]["single_gaussian_heatmap_topology"],
            "phase_dominance_thresholds": calibration["frozen_thresholds"]["phase_dominance"],
            "activity_thresholds": calibration["frozen_thresholds"]["activity"],
            "grounding_and_distinctness_thresholds": calibration["frozen_thresholds"]["grounding_and_distinctness"],
            "note": "Exceedance identifies a clean-oracle failure; it is not by itself a causal mechanism label.",
            "soft_canonical_rms_exceeds_per_channel": [
                bool(value > soft_thresholds["canonical_rms_max"])
                for value in _coordinate_summary(canonical)["canonical_rms_per_channel"]
            ],
            "hard_canonical_rms_exceeds_per_channel": [
                bool(value > hard_thresholds["canonical_rms_max"])
                for value in _coordinate_summary(hard_canonical)["canonical_rms_per_channel"]
            ],
        },
        "statistical_scope": {
            "statistics": "mean, median, empirical q90/q95/q99, maximum, and RMS",
            "sample_units": "fixed keypoint channel and correlated rendered frames or overlapping differences",
            "n_frames": EXPECTED_FRAME_COUNT,
            "n_channels": EXPECTED_CHANNEL_COUNT,
            "descriptive_not_inferential": True,
            "correlation_caveat": "adjacent and second-difference samples overlap and frames share one orbit; no naive SEM or population CI is reported",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    result_path = output_dir / "forensic_report.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "result": _regular_file_record(result_path),
        "raw_arrays": raw_record,
        "cell_id": cell_id,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_epoch": int(checkpoint_binding["epoch"]),
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-json", type=Path, required=True)
    parser.add_argument("--checkpoint-role", choices=("best_model", "final_model"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(SHA_RE.fullmatch(args.expected_calibration_sha256) is not None, "expected calibration SHA-256 is invalid")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
