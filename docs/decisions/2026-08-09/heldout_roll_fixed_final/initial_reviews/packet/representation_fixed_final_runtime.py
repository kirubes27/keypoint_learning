"""One-pass held-out roll finalizer for an authorized fixed-final run."""

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
from keypoint_net import representation_corpus_inventory as corpus_inventory
from keypoint_net import representation_evaluation_provenance as provenance_contract
from keypoint_net import representation_fixed_final_authorization as authorization
from keypoint_net import representation_split_adapter as split_adapter


class FixedFinalRuntimeError(RuntimeError):
    """The saved model, test-opening boundary, or evidence chain is invalid."""


_LOADED: ContextVar[dict[str, Any] | None] = ContextVar(
    "fixed_final_loaded_checkpoint", default=None
)
_PENDING_EVALUATOR: ContextVar[dict[str, Any] | None] = ContextVar(
    "fixed_final_pending_evaluator", default=None
)
_PENDING_PROVENANCE: ContextVar[dict[str, Any] | None] = ContextVar(
    "fixed_final_pending_provenance", default=None
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedFinalRuntimeError(message)


def _read_regular(path: Path, *, name: str) -> tuple[bytes, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FixedFinalRuntimeError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks), metadata.st_size
    finally:
        os.close(descriptor)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _safe_load_checkpoint(
    capability: authorization.FixedFinalCheckpointCapability,
) -> Mapping[str, Any]:
    path = Path(capability.checkpoint_absolute_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FixedFinalRuntimeError("cannot open fixed-final checkpoint") from exc
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
        if isinstance(exc, FixedFinalRuntimeError):
            raise
        raise FixedFinalRuntimeError("safe fixed-final checkpoint load failed") from exc
    finally:
        os.close(descriptor)
    _require(isinstance(payload, Mapping), "checkpoint payload is not a mapping")
    expected_keys = {
        "epoch", "model_state_dict", "optimizer_state_dict", "loss",
        "base_training_loss", "attachment_training_loss", "config",
    }
    _require(set(payload) == expected_keys, "fixed-final checkpoint keys differ")
    return payload


def load_authorized_fixed_final_checkpoint(
    repo_root: Path | str,
    binding: authorization.FixedFinalRunBinding,
    receipt_path: Path | str,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    authorization.FixedFinalCheckpointCapability,
    Mapping[str, Any],
]:
    capability = authorization.authorize_training_receipt(
        repo_root, binding, receipt_path
    )
    payload = _safe_load_checkpoint(capability)
    training = dict(capability.binding.expected_training_arguments)
    _require(payload["epoch"] == capability.binding.frozen_epochs,
             "checkpoint epoch differs from frozen epoch")
    config = payload["config"]
    _require(isinstance(config, Mapping), "checkpoint config is invalid")
    expected_contract = {
        "run_id": capability.binding.run_id,
        "recipe_id": capability.binding.recipe_id,
        "manifest_file_sha256": capability.binding.manifest_file_sha256,
        "manifest_content_sha256": capability.binding.manifest_content_sha256,
        "object_id": capability.binding.object_id,
        "object_role": capability.binding.object_role,
        "seed": capability.binding.seed,
        "frozen_epochs": capability.binding.frozen_epochs,
    }
    _require(config.get("fixed_final_contract") == expected_contract,
             "checkpoint fixed-final contract differs")
    _require(config.get("training_mode") == "fixed-final"
             and config.get("effective_epochs") == capability.binding.frozen_epochs,
             "checkpoint training policy differs")
    model = model_module.PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=(
            training["num_action_classes"] if training["lambda_act"] > 0.0 else 0
        ),
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=(
            training["learn_inverse_operator"]
            or training["lambda_inv"] > 0.0
            or training["lambda_cycle"] > 0.0
        ),
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
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    load_record = {
        "absolute_path": capability.checkpoint_absolute_path,
        "file_sha256": capability.checkpoint_sha256,
        "size_bytes": capability.checkpoint_size_bytes,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }
    _require(_LOADED.get() is None and _PENDING_EVALUATOR.get() is None,
             "a fixed-final checkpoint context is already pending")
    _LOADED.set(
        {
            "capability": capability,
            "model": model,
            "load_record": load_record,
            "state_sha256": _state_digest(model),
            "inference_record": None,
            "built_bundle_sha256": None,
            "opening_ledger": None,
        }
    )
    return model, capability, load_record


def _load_inventory(
    repo_root: Path,
    capability: authorization.FixedFinalCheckpointCapability,
) -> dict[str, Mapping[str, Any]]:
    path = repo_root / capability.binding.corpus_inventory_repo_relative_path
    raw, _ = _read_regular(path.resolve(strict=True), name="roll corpus inventory")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixedFinalRuntimeError("roll corpus inventory is invalid") from exc
    _require(isinstance(document, Mapping), "roll corpus inventory is invalid")
    _require(corpus_inventory.inventory_content_hash(document)
             == document.get("content_hash_sha256"),
             "roll corpus inventory content hash differs")
    records = document.get("files")
    _require(isinstance(records, list) and records, "roll corpus inventory has no files")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(isinstance(record, Mapping)
                 and set(record) == {"relative_path", "sha256", "size_bytes"},
                 "roll corpus inventory file record differs")
        relative = record["relative_path"]
        _require(isinstance(relative, str) and relative not in by_path,
                 "roll corpus inventory path is invalid or duplicated")
        by_path[relative] = record
    return by_path


def _read_bound_dataset_file(
    data_root: Path,
    relative_path: str,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    _require(relative_path in inventory, f"{name} is absent from corpus inventory")
    record = inventory[relative_path]
    path = (data_root / relative_path).resolve(strict=True)
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise FixedFinalRuntimeError(f"{name} escapes the dataset root") from exc
    raw, size = _read_regular(path, name=name)
    digest = hashlib.sha256(raw).hexdigest()
    _require(size == record["size_bytes"] and digest == record["sha256"],
             f"{name} differs from the committed corpus inventory")
    return raw, {
        "relative_path": relative_path,
        "absolute_path": str(path),
        "sha256": digest,
        "size_bytes": size,
    }


def _open_test_frames_once(
    repo_root: Path,
    data_root: Path,
    capability: authorization.FixedFinalCheckpointCapability,
    test_manifest: Any,
) -> tuple[list[np.ndarray], np.ndarray, list[int], list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any], dict[str, Any]]:
    loaded = _LOADED.get()
    _require(isinstance(loaded, dict) and loaded.get("opening_ledger") is None,
             "test contents can be opened only once")
    adapter = split_adapter.build_index_manifest_adapter_rows(
        manifest=test_manifest,
        object_id=capability.binding.object_id,
        geometry_binding_path=(repo_root / capability.binding.geometry_repo_relative_path).resolve(strict=True),
    )
    _require(adapter["stratum"]["transform_family"] == "roll"
             and adapter["stratum"]["evaluation_partition"] == "test",
             "fixed-final finalizer accepts only the held-out roll stratum")
    inventory = _load_inventory(repo_root, capability)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frames = adapter["evaluation"]["frames"]
    opened: list[dict[str, Any]] = []
    for frame in frames:
        image_raw, image_record = _read_bound_dataset_file(
            data_root, frame["image_relpath"], inventory,
            name=f"test frame {frame['frame_id']} image",
        )
        mask_raw, mask_record = _read_bound_dataset_file(
            data_root, frame["mask_relpath"], inventory,
            name=f"test frame {frame['frame_id']} mask",
        )
        try:
            with Image.open(io.BytesIO(image_raw)) as handle:
                image = np.asarray(handle.convert("RGB")).copy()
            with Image.open(io.BytesIO(mask_raw)) as handle:
                mask = np.asarray(handle.convert("L")).copy() > 0
        except Exception as exc:
            raise FixedFinalRuntimeError(
                f"test frame {frame['frame_id']} cannot be decoded"
            ) from exc
        _require(image.shape == (512, 512, 3) and mask.shape == (512, 512),
                 "fixed-final image/mask geometry must be 512x512")
        images.append(image)
        masks.append(mask)
        opened.extend(
            [
                {"frame_id": frame["frame_id"], "role": "image", **image_record},
                {"frame_id": frame["frame_id"], "role": "mask", **mask_record},
            ]
        )
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    ledger = {
        "test_content_open_phase_count": 1,
        "unique_test_frame_count": len(frame_ids),
        "unique_test_image_open_count": len(images),
        "unique_test_mask_open_count": len(masks),
        "each_unique_test_image_opened_once": True,
        "each_unique_test_mask_opened_once": True,
        "second_test_open_prevented": True,
        "opened_files": opened,
    }
    loaded["opening_ledger"] = ledger
    return (
        images,
        np.stack(masks),
        frame_ids,
        [dict(frame["physical_state"]) for frame in frames],
        [dict(row) for row in adapter["evaluation"]["rows"]],
        adapter["evaluator_transform"],
        ledger,
    )


def _infer_once(
    model: torch.nn.Module,
    images: Sequence[np.ndarray],
    *,
    device: torch.device,
    num_keypoints: int,
    heatmap_res: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    loaded = _LOADED.get()
    _require(isinstance(loaded, dict) and loaded.get("model") is model
             and loaded.get("inference_record") is None,
             "fixed-final inference requires the live one-shot context")
    before = _state_digest(model)
    _require(before == loaded["state_sha256"], "model state changed before inference")
    point_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            array = np.stack(images[start:start + 8])
            batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2)
            batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
            batch = batch.sub_(mean).div_(std)
            flattened, logits = model.extractor(batch)
            point_batches.append(
                flattened.reshape(batch.shape[0], num_keypoints, 2)
                .detach().cpu().numpy().astype(np.float32, copy=True)
            )
            logit_batches.append(
                logits.detach().cpu().numpy().astype(np.float32, copy=True)
            )
    points = np.concatenate(point_batches)
    logits = np.concatenate(logit_batches)
    _require(points.shape == (len(images), num_keypoints, 2),
             "fixed-final point output shape differs")
    _require(logits.shape == (len(images), num_keypoints, heatmap_res, heatmap_res),
             "fixed-final logit output shape differs")
    after = _state_digest(model)
    _require(before == after, "model state changed during held-out inference")
    record = {
        "frame_count": len(images),
        "model_state_sha256": after,
        "points_sha256": _array_digest(points),
        "logits_sha256": _array_digest(logits),
        "gradients_enabled_inside_inference": False,
        "state_unchanged": True,
        "inference_token": object(),
    }
    loaded["inference_record"] = record
    return points, logits, record


def _bbox_diagonals(masks: np.ndarray) -> list[float]:
    values: list[float] = []
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        _require(rows.size > 0, f"test mask {index} is empty")
        values.append(
            math.hypot(
                float(columns.max() - columns.min()) / 511.0,
                float(rows.max() - rows.min()) / 511.0,
            )
        )
    return values


def _evaluation_config() -> dict[str, Any]:
    return {
        "protocol": "generic",
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


def _committed_file_records(
    repo_root: Path,
    capability: authorization.FixedFinalCheckpointCapability,
) -> list[dict[str, str]]:
    paths = dict(provenance_contract.FIXED_FINAL_CHECKPOINT_ROLE_PATHS)
    binding = capability.binding
    paths.update(
        {
            "fixed_final_manifest": binding.manifest_repo_relative_path,
            "training_pair_artifact": binding.train_pair_repo_relative_path,
            "evaluation_pair_artifact": binding.test_pair_repo_relative_path,
            "corpus_inventory": binding.corpus_inventory_repo_relative_path,
            "geometry_manifest": binding.geometry_repo_relative_path,
            "decision_spec": binding.decision_spec_repo_relative_path,
            "pro_review": binding.pro_review_repo_relative_path,
            "fable_review": binding.fable_review_repo_relative_path,
            "user_approval": binding.user_approval_repo_relative_path,
        }
    )
    records: list[dict[str, str]] = []
    for role, relative in paths.items():
        raw, _ = _read_regular((repo_root / relative).resolve(strict=True), name=role)
        records.append(
            {"role": role, "repo_relative_path": relative,
             "file_sha256": hashlib.sha256(raw).hexdigest()}
        )
    return records


def build_evaluation_provenance(
    repo_root: Path,
    capability: authorization.FixedFinalCheckpointCapability,
) -> dict[str, Any]:
    checked = authorization.require_checkpoint_capability(capability)

    def external(role: str, path: str, digest: str, size: int) -> dict[str, Any]:
        return {"role": role, "absolute_path": path,
                "file_sha256": digest, "size_bytes": size}

    return {
        "schema_version": provenance_contract.PROVENANCE_SCHEMA_VERSION,
        "source_commit": checked.binding.source_commit,
        "committed_files": _committed_file_records(repo_root, checked),
        "external_files": [
            external("checkpoint", checked.checkpoint_absolute_path,
                     checked.checkpoint_sha256, checked.checkpoint_size_bytes),
            external("checkpoint_config", checked.config_absolute_path,
                     checked.config_sha256, checked.config_size_bytes),
            external("checkpoint_metadata", checked.history_absolute_path,
                     checked.history_sha256, checked.history_size_bytes),
            external("completed_run_receipt", checked.receipt_absolute_path,
                     checked.receipt_file_sha256, checked.receipt_size_bytes),
        ],
    }


def _build_bundle(
    repo_root: Path,
    capability: authorization.FixedFinalCheckpointCapability,
    model: torch.nn.Module,
    *,
    points: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    frame_ids: Sequence[int],
    physical_states: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    transform: Mapping[str, Any],
) -> dict[str, Any]:
    loaded = _LOADED.get()
    _require(isinstance(loaded, dict) and loaded.get("capability") is capability
             and loaded.get("model") is model,
             "fixed-final bundle requires the exact loaded checkpoint")
    inference = loaded.get("inference_record")
    _require(isinstance(inference, Mapping)
             and inference.get("points_sha256") == _array_digest(points)
             and inference.get("logits_sha256") == _array_digest(logits)
             and inference.get("model_state_sha256") == _state_digest(model),
             "fixed-final arrays/model differ from the one-shot inference")
    training = capability.binding.expected_training_arguments
    A = model.operator.A.detach().cpu().numpy().astype(np.float64)
    b = model.operator.bias.detach().cpu().numpy().astype(np.float64)
    bundle: dict[str, Any] = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": capability.binding.run_id,
        "case_kind": "checkpoint",
        "checkpoint_authority": "fixed_final",
        "provenance": build_evaluation_provenance(repo_root, capability),
        "evaluation_config": _evaluation_config(),
        "estimator_metadata": {
            "input_height": 512, "input_width": 512,
            "heatmap_height": training["heatmap_res"],
            "heatmap_width": training["heatmap_res"],
            "endpoint_grid": True, "temperature": training["temperature"],
            "logit_dtype": "float32", "softmax_dtype": "float32",
            "crop": None, "resize": [512, 512], "align_corners": None,
        },
        "transform": dict(transform),
        "evaluation": {
            "object_id": capability.binding.object_id,
            "seed": capability.binding.seed,
            "partition": "test",
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
                    "input_height": 512, "input_width": 512, "crop": None,
                    "resize": [512, 512], "align_corners": None,
                },
            },
            "pair_rows": [dict(item) for item in pair_rows],
            "logits": codec.encode_float32_array(logits),
        },
        "operator": {"mode": "supplied", "A": A.tolist(), "b": b.tolist()},
    }
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    loaded["built_bundle_sha256"] = bundle["bundle_content_sha256"]
    return bundle


def _arm_evaluator(
    bundle: Mapping[str, Any],
    capability: authorization.FixedFinalCheckpointCapability,
    inference_record: Mapping[str, Any],
) -> None:
    loaded = _LOADED.get()
    _LOADED.set(None)
    _require(isinstance(loaded, dict) and loaded.get("capability") is capability,
             "fixed-final evaluator requires the registered checkpoint")
    _require(inference_record is loaded.get("inference_record"),
             "fixed-final evaluator requires the actual inference record")
    claimed = bundle.get("bundle_content_sha256")
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(bundle.get("checkpoint_authority") == "fixed_final"
             and bundle.get("case_id") == capability.binding.run_id
             and claimed == evaluator.canonical_sha256(payload)
             and claimed == loaded.get("built_bundle_sha256"),
             "fixed-final bundle identity or content differs")
    _PENDING_EVALUATOR.set({"bundle_content_sha256": claimed, "loaded": loaded})


def validate_fixed_final_checkpoint_evaluator_authorization(
    bundle: Mapping[str, Any],
) -> Mapping[str, Any]:
    pending = _PENDING_EVALUATOR.get()
    _PENDING_EVALUATOR.set(None)
    _require(isinstance(pending, dict),
             "fixed-final evaluation requires a live one-shot context")
    claimed = bundle.get("bundle_content_sha256")
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(claimed == pending["bundle_content_sha256"]
             and claimed == evaluator.canonical_sha256(payload),
             "fixed-final bundle changed after authorization")
    loaded = pending["loaded"]
    capability = authorization.require_checkpoint_capability(loaded["capability"])
    _require(_PENDING_PROVENANCE.get() is None,
             "a fixed-final provenance receipt is already pending")
    _PENDING_PROVENANCE.set(loaded)
    return MappingProxyType(
        {
            "checkpoint_evaluation_authorized": True,
            "source_commit": capability.binding.source_commit,
            "run_id": capability.binding.run_id,
            "checkpoint_sha256": capability.checkpoint_sha256,
            "completed_run_receipt_sha256": capability.receipt_file_sha256,
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": True,
        }
    )


def consume_fixed_final_checkpoint_provenance_load_receipt(
    *, source_commit: str, checkpoint_record: Mapping[str, Any]
) -> Mapping[str, Any]:
    loaded = _PENDING_PROVENANCE.get()
    _PENDING_PROVENANCE.set(None)
    _require(isinstance(loaded, dict),
             "fixed-final provenance lacks a checkpoint load receipt")
    capability = authorization.require_checkpoint_capability(loaded["capability"])
    expected = {
        "role": "checkpoint",
        "absolute_path": capability.checkpoint_absolute_path,
        "file_sha256": capability.checkpoint_sha256,
        "size_bytes": capability.checkpoint_size_bytes,
    }
    _require(source_commit == capability.binding.source_commit
             and dict(checkpoint_record) == expected,
             "fixed-final provenance checkpoint binding differs")
    return MappingProxyType(
        {
            **expected,
            "task_id": 55 if capability.binding.recipe_id == "task55_clean" else 80,
            "fixture_id": capability.binding.run_id,
            "source_commit": capability.binding.source_commit,
            "same_open_file_descriptor_hash_and_load": True,
            "weights_only": True,
        }
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(
            descriptor,
            json.dumps(
                value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
            ).encode("utf-8") + b"\n",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finalize_fixed_final_roll(
    repo_root: Path | str,
    binding: authorization.FixedFinalRunBinding,
    training_receipt_path: Path | str,
    *,
    data_root: Path | str,
    test_manifest: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Open held-out content once and emit one bound representation result."""

    root = Path(repo_root).resolve(strict=True)
    dataset_root = Path(data_root).expanduser().resolve(strict=True)
    model, capability, _ = load_authorized_fixed_final_checkpoint(
        root, binding, training_receipt_path, device=device
    )
    (
        images, masks, frame_ids, physical_states, pair_rows, transform, ledger,
    ) = _open_test_frames_once(root, dataset_root, capability, test_manifest)
    training = capability.binding.expected_training_arguments
    points, logits, inference = _infer_once(
        model, images, device=device, num_keypoints=training["num_keypoints"],
        heatmap_res=training["heatmap_res"],
    )
    bundle = _build_bundle(
        root, capability, model, points=points, logits=logits, masks=masks,
        frame_ids=frame_ids, physical_states=physical_states, pair_rows=pair_rows,
        transform=transform,
    )
    _arm_evaluator(bundle, capability, inference)
    result = evaluator.evaluate_bundle(bundle)
    _require(result.get("stratum", {}).get("partition") == "test",
             "fixed-final evaluator did not return test-stratum evidence")
    _require(result.get("scientific_threshold_status") == "not_defined_by_evaluator",
             "fixed-final evaluator attempted an automatic scientific decision")
    run_dir = Path(capability.binding.run_directory).resolve(strict=True)
    bundle_path = run_dir / "fixed_final_test_bundle.json"
    result_path = run_dir / "fixed_final_test_representation_result.json"
    _write_json_exclusive(bundle_path, bundle)
    _write_json_exclusive(result_path, result)
    evidence_receipt = authorization.write_evidence_receipt(
        capability, bundle_path=bundle_path, result_path=result_path,
        opening_ledger=ledger,
    )
    return {
        "bundle_path": str(bundle_path),
        "result_path": str(result_path),
        "evidence_receipt_path": str(evidence_receipt),
        "result": result,
        "test_boundary": ledger,
    }


__all__ = [
    "FixedFinalRuntimeError", "build_evaluation_provenance",
    "consume_fixed_final_checkpoint_provenance_load_receipt",
    "finalize_fixed_final_roll", "load_authorized_fixed_final_checkpoint",
    "validate_fixed_final_checkpoint_evaluator_authorization",
]
