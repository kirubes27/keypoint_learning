"""Fresh-only checkpoint loader and production-evaluator adapter."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from keypoint_net import eval_representation as evaluator
from keypoint_net import model as model_module
from keypoint_net import representation_array_codec as codec
from keypoint_net import representation_evaluation_provenance as provenance_contract
from keypoint_net import representation_fresh_checkpoint_authorization as authorization


class FreshCheckpointRuntimeError(RuntimeError):
    pass


_LOADED_CHECKPOINT: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "fresh_loaded_checkpoint", default=None
)
_PENDING_EVALUATOR: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "fresh_pending_evaluator", default=None
)
_PENDING_PROVENANCE: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "fresh_pending_provenance", default=None
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshCheckpointRuntimeError(message)


def _read_regular(path: Path, *, name: str) -> tuple[bytes, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreshCheckpointRuntimeError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), metadata.st_size
    finally:
        os.close(descriptor)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_authorized_fresh_checkpoint(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    cell_id: str,
) -> tuple[torch.nn.Module, authorization.FreshCheckpointCapability, dict[str, Any]]:
    """Hash and deserialize an authorized checkpoint through one descriptor."""

    capability = authorization.authorize_completed_fresh_run(
        repo_root, receipt_path, expected_cell_id=cell_id
    )
    path = Path(capability.checkpoint_absolute_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreshCheckpointRuntimeError("cannot open fresh checkpoint") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "checkpoint is not a regular file")
        _require(metadata.st_size == capability.checkpoint_size_bytes,
                 "checkpoint size changed at load")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _require(digest.hexdigest() == capability.checkpoint_sha256,
                     "checkpoint hash changed at load")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except Exception as exc:
        if isinstance(exc, FreshCheckpointRuntimeError):
            raise
        raise FreshCheckpointRuntimeError("safe checkpoint load failed") from exc
    finally:
        os.close(descriptor)

    _require(isinstance(payload, Mapping), "checkpoint payload is not a mapping")
    _require(
        set(payload) == {"epoch", "model_state_dict", "optimizer_state_dict", "loss", "config"},
        "checkpoint payload keys differ",
    )
    authorization.validate_loaded_checkpoint(capability, payload)
    training = capability.expected_embedded_config["training"]
    model = model_module.PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=0,
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=training["learn_inverse_operator"],
        heatmap_res=training["heatmap_res"],
    )
    state = payload["model_state_dict"]
    _require(isinstance(state, Mapping) and set(state) == set(model.state_dict()),
             "checkpoint state keys differ")
    for name, target in model.state_dict().items():
        source = state[name]
        _require(torch.is_tensor(source) and source.device.type == "cpu",
                 f"checkpoint tensor {name} is invalid")
        _require(source.shape == target.shape and source.dtype == target.dtype
                 and bool(torch.isfinite(source).all()),
                 f"checkpoint tensor {name} shape/dtype/finiteness differs")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    load_record = {
        "absolute_path": capability.checkpoint_absolute_path,
        "file_sha256": capability.checkpoint_sha256,
        "size_bytes": capability.checkpoint_size_bytes,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }
    register_fresh_checkpoint_load(capability, load_record)
    return model, capability, load_record


def register_fresh_checkpoint_load(
    capability: authorization.FreshCheckpointCapability,
    checkpoint_record: Mapping[str, Any],
) -> None:
    checked = authorization.require_fresh_checkpoint_capability(capability)
    _require(_LOADED_CHECKPOINT.get() is None and _PENDING_EVALUATOR.get() is None,
             "a fresh checkpoint context is already pending")
    expected = {
        "absolute_path": checked.checkpoint_absolute_path,
        "file_sha256": checked.checkpoint_sha256,
        "size_bytes": checked.checkpoint_size_bytes,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }
    _require(dict(checkpoint_record) == expected, "fresh load record differs")
    _LOADED_CHECKPOINT.set({"capability": checked, "load_record": expected})


def arm_fresh_checkpoint_evaluator(
    bundle: Mapping[str, Any],
    *,
    capability: authorization.FreshCheckpointCapability,
    inference_record: Mapping[str, Any],
) -> None:
    loaded = _LOADED_CHECKPOINT.get()
    _LOADED_CHECKPOINT.set(None)
    checked = authorization.require_fresh_checkpoint_capability(capability)
    _require(isinstance(loaded, Mapping) and loaded.get("capability") is checked,
             "fresh evaluator requires the registered checkpoint load")
    _require(inference_record.get("state_unchanged") is True
             and inference_record.get("gradients_enabled_inside_inference") is False,
             "fresh inference did not preserve state/no-gradient semantics")
    _require(bundle.get("case_kind") == "checkpoint"
             and bundle.get("checkpoint_authority") == "fresh_run"
             and bundle.get("case_id") == checked.cell_id,
             "fresh bundle identity differs")
    claimed_hash = bundle.get("bundle_content_sha256")
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(claimed_hash == evaluator.canonical_sha256(payload),
             "fresh bundle content hash differs")
    _PENDING_EVALUATOR.set({"bundle_content_sha256": claimed_hash, "loaded": loaded})


def validate_fresh_checkpoint_evaluator_authorization(
    bundle: Mapping[str, Any],
) -> Mapping[str, Any]:
    pending = _PENDING_EVALUATOR.get()
    _PENDING_EVALUATOR.set(None)
    _require(isinstance(pending, Mapping),
             "fresh evaluation requires a live one-shot context")
    claimed_hash = bundle.get("bundle_content_sha256")
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(claimed_hash == pending["bundle_content_sha256"]
             and claimed_hash == evaluator.canonical_sha256(payload),
             "fresh bundle changed after authorization")
    loaded = pending["loaded"]
    capability = authorization.require_fresh_checkpoint_capability(
        loaded["capability"]
    )
    _require(_PENDING_PROVENANCE.get() is None,
             "a fresh provenance receipt is already pending")
    _PENDING_PROVENANCE.set(loaded)
    return MappingProxyType({
        "checkpoint_evaluation_authorized": True,
        "source_commit": capability.source_commit,
        "cell_id": capability.cell_id,
        "checkpoint_sha256": capability.checkpoint_sha256,
        "completed_run_receipt_sha256": capability.receipt_file_sha256,
        "training_or_weight_update_authorized": False,
        "selection_use_authorized": True,
    })


def consume_fresh_checkpoint_provenance_load_receipt(
    *, source_commit: str, checkpoint_record: Mapping[str, Any]
) -> Mapping[str, Any]:
    loaded = _PENDING_PROVENANCE.get()
    _PENDING_PROVENANCE.set(None)
    _require(isinstance(loaded, Mapping), "fresh provenance lacks a load receipt")
    capability = authorization.require_fresh_checkpoint_capability(
        loaded["capability"]
    )
    expected = {
        "role": "checkpoint",
        "absolute_path": capability.checkpoint_absolute_path,
        "file_sha256": capability.checkpoint_sha256,
        "size_bytes": capability.checkpoint_size_bytes,
    }
    _require(source_commit == capability.source_commit
             and dict(checkpoint_record) == expected,
             "fresh provenance checkpoint binding differs")
    return MappingProxyType({
        **expected,
        "task_id": 55 if capability.cell_id.startswith("task55_") else 80,
        "fixture_id": capability.cell_id,
        "source_commit": capability.source_commit,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    })


def _committed_file_records(repo_root: Path) -> list[dict[str, Any]]:
    records = []
    for role, relative_path in provenance_contract.FRESH_CHECKPOINT_ROLE_PATHS.items():
        data, _ = _read_regular((repo_root / relative_path).absolute(), name=role)
        records.append(
            {
                "role": role,
                "repo_relative_path": relative_path,
                "file_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def build_evaluation_provenance(
    repo_root: Path,
    capability: authorization.FreshCheckpointCapability,
) -> dict[str, Any]:
    def external(role: str, path: str, digest: str, size: int) -> dict[str, Any]:
        return {
            "role": role,
            "absolute_path": path,
            "file_sha256": digest,
            "size_bytes": size,
        }

    return {
        "schema_version": provenance_contract.PROVENANCE_SCHEMA_VERSION,
        "source_commit": capability.source_commit,
        "committed_files": _committed_file_records(repo_root),
        "external_files": [
            external("checkpoint", capability.checkpoint_absolute_path,
                     capability.checkpoint_sha256, capability.checkpoint_size_bytes),
            external("checkpoint_config", capability.config_absolute_path,
                     capability.config_sha256, capability.config_size_bytes),
            external("checkpoint_metadata", capability.history_absolute_path,
                     capability.history_sha256, capability.history_size_bytes),
            external("completed_run_receipt", capability.receipt_absolute_path,
                     capability.receipt_file_sha256, capability.receipt_size_bytes),
        ],
    }


def _bbox_diagonals(masks: np.ndarray) -> list[float]:
    values = []
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        _require(rows.size > 0, f"mask {index} is empty")
        values.append(math.hypot(
            float(columns.max() - columns.min()) / 511.0,
            float(rows.max() - rows.min()) / 511.0,
        ))
    return values


def _evaluation_config(*, full_primary_roll: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "protocol": "full_primary_roll" if full_primary_roll else "generic",
        "representation_thresholds": {
            "close_distance_objdiag": 0.06,
            "persistent_fraction": 0.50,
            "recurrent_fraction": 0.10,
            "transient_longest_fraction": 0.10,
            "clustered_median_objdiag": 0.12,
        },
        "motion_reference_magnitude_image01": 0.1,
        "motion_fraction_min": 0.1,
        "on_object_rate_min": 0.75,
        "minimum_eligible_channels": 2,
        "operator_composition_horizons": [1, 2, 10],
    }
    if full_primary_roll:
        config.update(
            {
                "full_rollout_horizons": list(range(1, 60)),
                "role_scoped_holdout_frame_ids": list(range(24)),
                "holdout_rollout_horizons": list(range(1, 8)),
                "closure_horizon": 60,
            }
        )
    return config


def build_fresh_checkpoint_bundle(
    repo_root: Path,
    capability: authorization.FreshCheckpointCapability,
    model: torch.nn.Module,
    *,
    points: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    frame_ids: Sequence[int],
    physical_states: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    partition: str,
    full_primary_roll: bool,
) -> dict[str, Any]:
    training = capability.expected_embedded_config["training"]
    A = model.operator.A.detach().cpu().numpy().astype(np.float64)
    b = model.operator.bias.detach().cpu().numpy().astype(np.float64)
    bundle: dict[str, Any] = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": capability.cell_id,
        "case_kind": "checkpoint",
        "checkpoint_authority": "fresh_run",
        "provenance": build_evaluation_provenance(repo_root, capability),
        "evaluation_config": _evaluation_config(full_primary_roll=full_primary_roll),
        "estimator_metadata": {
            "input_height": 512,
            "input_width": 512,
            "heatmap_height": training["heatmap_res"],
            "heatmap_width": training["heatmap_res"],
            "endpoint_grid": True,
            "temperature": training["temperature"],
            "logit_dtype": "float32",
            "softmax_dtype": "float32",
            "crop": None,
            "resize": [512, 512],
            "align_corners": None,
        },
        "transform": {
            **dict(capability.expected_embedded_config["transform"]),
            "expected_2d_family": "planar_rotation_about_projected_center",
            "projected_centre_xy": [0.0, 0.0],
        },
        "evaluation": {
            "object_id": capability.expected_embedded_config["object"]["name"],
            "seed": capability.expected_embedded_config["seed"],
            "partition": partition,
            "frame_ids": list(frame_ids),
            "points": codec.encode_float32_array(points),
            "physical_states": [dict(item) for item in physical_states],
            "bbox_diagonal_image01": _bbox_diagonals(masks),
            "visibility": codec.encode_bool_packbits_array(
                np.ones(points.shape[:2], dtype=bool)
            ),
            "masks": {
                "values": codec.encode_bool_packbits_array(masks),
                "geometry": {
                    "input_height": 512,
                    "input_width": 512,
                    "crop": None,
                    "resize": [512, 512],
                    "align_corners": None,
                },
            },
            "pair_rows": [dict(item) for item in pair_rows],
            "logits": codec.encode_float32_array(logits),
        },
        "operator": {"mode": "supplied", "A": A.tolist(), "b": b.tolist()},
    }
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    return bundle


def _load_full_orbit(data_root: Path, repo_root: Path) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, float]]]:
    frame_manifest = json.loads((repo_root / provenance_contract.FRESH_CHECKPOINT_ROLE_PATHS[
        "replay_frame_mask_inventory"
    ]).read_text())
    metadata_manifest = json.loads((repo_root / provenance_contract.FRESH_CHECKPOINT_ROLE_PATHS[
        "replay_metadata_manifest"
    ]).read_text())
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in frame_manifest["records"]:
        decoded = []
        for role in ("image", "mask"):
            binding = record[role]
            path = (data_root / binding["relative_path"]).absolute()
            data, size = _read_regular(path, name=f"frame {record['frame_index']} {role}")
            _require(size == binding["size_bytes"]
                     and hashlib.sha256(data).hexdigest() == binding["sha256"],
                     f"frame {record['frame_index']} {role} binding differs")
            with Image.open(io.BytesIO(data)) as handle:
                decoded.append(np.asarray(handle.convert("RGB" if role == "image" else "L")).copy())
        images.append(decoded[0])
        masks.append(decoded[1] > 0)
    _require(len(images) == len(masks) == 180, "full hammer orbit is incomplete")
    states = [
        {"theta_deg": float(record["physical_state"]["theta_deg"])}
        for record in metadata_manifest["frame_records"]
    ]
    return images, np.stack(masks), states


def _infer(model: torch.nn.Module, images: Sequence[np.ndarray], *, heatmap_res: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    before = _state_digest(model)
    point_batches = []
    logit_batches = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            array = np.stack(images[start:start + 8])
            batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255)
            batch = batch.sub_(mean).div_(std)
            flattened, logits = model.extractor(batch)
            point_batches.append(flattened.reshape(batch.shape[0], 10, 2).numpy().copy())
            logit_batches.append(logits.numpy().copy())
    points = np.concatenate(point_batches)
    logits = np.concatenate(logit_batches)
    _require(points.shape == (len(images), 10, 2), "point output shape differs")
    _require(logits.shape == (len(images), 10, heatmap_res, heatmap_res),
             "logit output shape differs")
    after = _state_digest(model)
    _require(before == after, "model state changed during inference")
    return points, logits, {
        "state_unchanged": True,
        "gradients_enabled_inside_inference": False,
    }


def evaluate_authorized_fresh_run(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    cell_id: str,
    data_root: Path | str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    model, capability, _ = load_authorized_fresh_checkpoint(
        root, receipt_path, cell_id=cell_id
    )
    images, masks, states = _load_full_orbit(Path(data_root).resolve(strict=True), root)
    heatmap_res = capability.expected_embedded_config["training"]["heatmap_res"]
    points, logits, inference = _infer(model, images, heatmap_res=heatmap_res)
    pairs = [
        {
            "pair_id": f"full_corpus:{source}:{(source + 3) % 180}",
            "source_frame": source,
            "target_frame": (source + 3) % 180,
            "direction": "forward",
            "stride": 3,
            "partition": "full_corpus",
        }
        for source in range(180)
    ]
    bundle = build_fresh_checkpoint_bundle(
        root, capability, model,
        points=points, logits=logits, masks=masks,
        frame_ids=list(range(180)), physical_states=states, pair_rows=pairs,
        partition="full_corpus", full_primary_roll=True,
    )
    arm_fresh_checkpoint_evaluator(
        bundle, capability=capability, inference_record=inference
    )
    return evaluator.evaluate_bundle(bundle)


__all__ = [
    "FreshCheckpointRuntimeError",
    "build_evaluation_provenance",
    "build_fresh_checkpoint_bundle",
    "evaluate_authorized_fresh_run",
    "consume_fresh_checkpoint_provenance_load_receipt",
    "load_authorized_fresh_checkpoint",
    "validate_fresh_checkpoint_evaluator_authorization",
]
