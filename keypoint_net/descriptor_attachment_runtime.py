"""Authorized descriptor-checkpoint loader and production evaluator adapter."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import torch

from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net import eval_representation as evaluator
from keypoint_net import model as model_module
from keypoint_net import representation_fresh_checkpoint_runtime as fresh_runtime


class DescriptorAttachmentRuntimeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DescriptorAttachmentRuntimeError(message)


def load_authorized_descriptor_checkpoint(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    cell_id: str,
) -> tuple[
    torch.nn.Module,
    authorization.DescriptorCompletedRunCapability,
    dict[str, Any],
]:
    """Hash and safely load the exact checkpoint through one open descriptor."""

    capability = authorization.authorize_completed_run(
        repo_root, receipt_path, expected_cell_id=cell_id
    )
    path = Path(capability.checkpoint_absolute_path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise DescriptorAttachmentRuntimeError("cannot open descriptor checkpoint") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "descriptor checkpoint is not regular")
        _require(metadata.st_size == capability.checkpoint_size_bytes,
                 "descriptor checkpoint size changed at load")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _require(digest.hexdigest() == capability.checkpoint_sha256,
                     "descriptor checkpoint hash changed at load")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except Exception as exc:
        if isinstance(exc, DescriptorAttachmentRuntimeError):
            raise
        raise DescriptorAttachmentRuntimeError("safe descriptor checkpoint load failed") from exc
    finally:
        os.close(descriptor)

    _require(isinstance(payload, Mapping), "descriptor checkpoint is not a mapping")
    _require(set(payload) == {
        "epoch", "model_state_dict", "optimizer_state_dict", "loss",
        "selection_loss_name", "base_validation_loss",
        "attachment_validation_loss", "augmented_validation_loss", "config",
    }, "descriptor checkpoint payload keys differ")
    training = capability.expected_training_arguments
    model = model_module.PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=training["num_action_classes"],
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=training["learn_inverse_operator"],
        heatmap_res=training["heatmap_res"],
    )
    state = payload["model_state_dict"]
    _require(isinstance(state, Mapping) and set(state) == set(model.state_dict()),
             "descriptor checkpoint state keys differ")
    for name, target in model.state_dict().items():
        source = state[name]
        _require(torch.is_tensor(source) and source.device.type == "cpu",
                 f"descriptor checkpoint tensor {name} is invalid")
        _require(source.shape == target.shape and source.dtype == target.dtype
                 and bool(torch.isfinite(source).all()),
                 f"descriptor checkpoint tensor {name} differs")
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
    fresh_runtime.register_descriptor_checkpoint_load(
        capability, load_record, model=model
    )
    return model, capability, load_record


def evaluate_authorized_descriptor_run(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    cell_id: str,
    data_root: Path | str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    model, capability, _ = load_authorized_descriptor_checkpoint(
        root, receipt_path, cell_id=cell_id
    )
    images, masks, states = fresh_runtime._load_full_orbit(  # noqa: SLF001
        Path(data_root).resolve(strict=True), root
    )
    heatmap_res = int(capability.expected_training_arguments["heatmap_res"])
    points, logits, inference = fresh_runtime._infer(  # noqa: SLF001
        model, images, heatmap_res=heatmap_res
    )
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
    bundle = fresh_runtime.build_descriptor_checkpoint_bundle(
        root, capability, model,
        points=points, logits=logits, masks=masks,
        frame_ids=list(range(180)), physical_states=states, pair_rows=pairs,
        partition="full_corpus", full_primary_roll=True,
    )
    fresh_runtime.arm_descriptor_checkpoint_evaluator(
        bundle, capability=capability, inference_record=inference
    )
    return evaluator.evaluate_bundle(bundle)


__all__ = [
    "DescriptorAttachmentRuntimeError",
    "evaluate_authorized_descriptor_run",
    "load_authorized_descriptor_checkpoint",
]
