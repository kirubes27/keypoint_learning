"""Fail-closed training authorization for the descriptor-attachment experiment.

The historical 64-versus-128 manifest remains immutable.  This boundary uses a
new experiment manifest and the CPU calibration receipt, both exact-file and
content-hash bound, so neither code nor execution depends on Markdown.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from .descriptor_attachment import LOSS_SPEC_SHA256, canonical_sha256
    from .descriptor_attachment_calibration import (
        CHECKPOINT_SELECTOR,
        SCHEMA_VERSION as WEIGHT_RECEIPT_SCHEMA_VERSION,
    )
    from . import representation_fresh_checkpoint_authorization as fresh
    from . import fresh_roll_determinism
except ImportError:  # Historical direct-script execution.
    from descriptor_attachment import LOSS_SPEC_SHA256, canonical_sha256  # type: ignore
    from descriptor_attachment_calibration import (  # type: ignore
        CHECKPOINT_SELECTOR,
        SCHEMA_VERSION as WEIGHT_RECEIPT_SCHEMA_VERSION,
    )
    import representation_fresh_checkpoint_authorization as fresh  # type: ignore
    import fresh_roll_determinism  # type: ignore


EXPERIMENT_SCHEMA_VERSION = "descriptor_attachment_experiment_manifest.v1"
EXPERIMENT_ARTIFACT_TYPE = "descriptor_attachment_experiment_manifest"
EXPECTED_BRANCH = "agent/representation-oracles-20260726"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DESCRIPTOR_CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__(control|attachment)__seed(42|43|44)$"
)
COMPLETED_RUN_RECEIPT_NAME = "DESCRIPTOR_COMPLETED_RUN_RECEIPT.json"
COMPLETED_RUN_SCHEMA_VERSION = "descriptor_attachment_completed_run.v1"

# Every argument that can alter scientific data, model, optimization, loss,
# selection, or evaluation semantics must appear exactly once in each cell.
BOUND_TRAINING_ARGUMENTS = frozenset({
    "data_root",
    "object",
    "pairs_index",
    "indexed_mode",
    "train_pairs_index",
    "val_pairs_index",
    "test_pairs_index",
    "dataset_binding_sha256",
    "img_size",
    "num_keypoints",
    "base_channels",
    "heatmap_res",
    "temperature",
    "padding_mode",
    "operator_type",
    "learn_inverse_operator",
    "lambda_smooth",
    "lambda_disp",
    "lambda_ent",
    "lambda_act",
    "lambda_loc",
    "lambda_inv",
    "lambda_cycle",
    "lambda_attach",
    "sigma",
    "loc_bg_threshold",
    "num_action_classes",
    "frame_skip",
    "yaw_step_deg",
    "center_crop",
    "epochs",
    "frozen_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "output_dir",
    "save_every",
    "log_every",
    "auto_eval",
    "auto_eval_checkpoint",
    "eval_frames_dir",
    "eval_max_k",
})

DESCRIPTOR_RUN_SOURCE_PATHS = (
    "keypoint_net/descriptor_attachment.py",
    "keypoint_net/descriptor_attachment_authorization.py",
    "keypoint_net/model.py",
    "keypoint_net/train.py",
    "keypoint_net/dataset.py",
    "keypoint_net/fresh_roll_determinism.py",
    "keypoint_net/descriptor_attachment_runtime.py",
    "keypoint_net/representation_fresh_checkpoint_runtime.py",
    "keypoint_net/eval_representation.py",
    "keypoint_net/representation_evaluation_provenance.py",
    "keypoint_net/representation_array_codec.py",
    "keypoint_net/run_descriptor_attachment_matrix.py",
    "keypoint_net/descriptor_attachment_decision.py",
)
_VERIFIED_DESCRIPTOR_CAPABILITY = object()


@dataclass(frozen=True)
class DescriptorAttachmentBinding:
    cell_id: str
    arm: str
    lambda_attach: float
    run_directory: str
    experiment_manifest_file_sha256: str
    experiment_manifest_content_sha256: str
    weight_receipt_file_sha256: str
    weight_receipt_content_sha256: str


@dataclass(frozen=True)
class DescriptorCompletedRunCapability:
    cell_id: str
    arm: str
    base_cell_id: str
    source_commit: str
    checkpoint_absolute_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    config_absolute_path: str
    config_sha256: str
    config_size_bytes: int
    history_absolute_path: str
    history_sha256: str
    history_size_bytes: int
    receipt_absolute_path: str
    receipt_file_sha256: str
    receipt_size_bytes: int
    expected_training_arguments: Mapping[str, Any]
    expected_base_config: Mapping[str, Any]
    _capability: object


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path, *, name: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().absolute()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        _require(size == metadata.st_size, f"{name} size changed while reading")
        return {
            "absolute_path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        }
    finally:
        os.close(descriptor)


def _load_json_value(path_value: str | Path, *, name: str) -> tuple[Any, dict[str, Any]]:
    record = _file_record(path_value, name=name)
    path = Path(record["absolute_path"])
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    return value, record


def _read_regular_json(path_value: str | Path, *, name: str) -> tuple[dict[str, Any], str]:
    path = Path(path_value).expanduser().resolve(strict=True)
    mode = path.stat().st_mode
    _require(stat.S_ISREG(mode), f"{name} is not a regular file")
    data = path.read_bytes()
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    _require(isinstance(document, dict), f"{name} must be a JSON object")
    return document, hashlib.sha256(data).hexdigest()


def _validate_content_hash(document: Mapping[str, Any], *, name: str) -> str:
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        f"{name} content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash mismatch")
    return claimed


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_state(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    _require(_COMMIT_RE.fullmatch(commit) is not None, "Git HEAD is invalid")
    _require(branch == EXPECTED_BRANCH, f"wrong source branch {branch!r}")
    tracked = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    _require(not tracked, "tracked source is modified")
    return commit, branch


def validate_weight_receipt(
    repo_root: Path | str,
    receipt_path: str | Path,
    *,
    expected_file_sha256: str,
) -> tuple[Mapping[str, Any], str, str]:
    root = Path(repo_root).resolve(strict=True)
    document, file_hash = _read_regular_json(receipt_path, name="weight receipt")
    _require(
        _SHA256_RE.fullmatch(expected_file_sha256 or "") is not None,
        "weight receipt expected file hash is invalid",
    )
    _require(file_hash == expected_file_sha256, "weight receipt file hash mismatch")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "source",
        "loss_spec_sha256",
        "checkpoint_selector",
        "calibration_protocol",
        "dataset",
        "cell_norms",
        "median_gradient_ratio",
        "maximum_gradient_ratio",
        "lambda_attach",
        "scaled_median_contribution",
        "scaled_maximum_contribution",
        "caps",
        "runtime",
        "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "weight receipt fields differ from schema")
    content_hash = _validate_content_hash(document, name="weight receipt")
    _require(
        document["schema_version"] == WEIGHT_RECEIPT_SCHEMA_VERSION
        and document["artifact_type"] == "descriptor_attachment_weight_receipt",
        "wrong weight receipt schema",
    )
    _require(document["status"] == "authorized", "weight receipt did not authorize lambda")
    _require(document["loss_spec_sha256"] == LOSS_SPEC_SHA256, "loss specification changed")
    _require(
        document["checkpoint_selector"] == CHECKPOINT_SELECTOR,
        "checkpoint selector changed",
    )
    lambda_attach = document["lambda_attach"]
    _require(
        type(lambda_attach) is float and math.isfinite(lambda_attach) and lambda_attach > 0.0,
        "weight receipt lambda_attach is invalid",
    )
    source = document["source"]
    _require(
        isinstance(source, dict)
        and set(source) == {"commit", "branch", "module_hashes"},
        "weight receipt source record is invalid",
    )
    _require(source["branch"] == EXPECTED_BRANCH, "weight receipt branch changed")
    module_hashes = source["module_hashes"]
    expected_modules = {
        "keypoint_net/descriptor_attachment.py",
        "keypoint_net/descriptor_attachment_calibration.py",
        "keypoint_net/model.py",
        "keypoint_net/dataset.py",
        "aggregate_sha256",
    }
    _require(
        isinstance(module_hashes, dict) and set(module_hashes) == expected_modules,
        "weight receipt module hash set changed",
    )
    current_module_hashes = {
        relative: _sha256_file(root / relative)
        for relative in sorted(expected_modules - {"aggregate_sha256"})
    }
    _require(
        module_hashes["aggregate_sha256"]
        == canonical_sha256(current_module_hashes),
        "weight receipt aggregate code hash is invalid",
    )
    for relative, current_hash in current_module_hashes.items():
        _require(module_hashes[relative] == current_hash, f"bound code changed: {relative}")
    return document, file_hash, content_hash


def bind_training_namespace(
    repo_root: Path | str,
    args: argparse.Namespace,
) -> DescriptorAttachmentBinding:
    """Bind one control/attachment cell to exact code, manifest, and weight."""

    root = Path(repo_root).resolve(strict=True)
    _require(isinstance(args.descriptor_cell_id, str), "descriptor_cell_id is required")
    _require(
        isinstance(args.descriptor_experiment_manifest, str),
        "descriptor experiment manifest is required",
    )
    _require(
        isinstance(args.descriptor_experiment_manifest_sha256, str),
        "descriptor experiment manifest hash is required",
    )
    _require(
        isinstance(args.descriptor_weight_receipt, str),
        "descriptor weight receipt is required",
    )
    _require(
        isinstance(args.descriptor_weight_receipt_sha256, str),
        "descriptor weight receipt hash is required",
    )
    _require(
        args.descriptor_loss_spec_sha256 == LOSS_SPEC_SHA256,
        "descriptor loss specification hash differs from code",
    )
    _require(args.fresh_cell_id is None, "descriptor and historical fresh-cell paths cannot mix")

    manifest, manifest_file_hash = _read_regular_json(
        args.descriptor_experiment_manifest,
        name="descriptor experiment manifest",
    )
    _require(
        _SHA256_RE.fullmatch(args.descriptor_experiment_manifest_sha256) is not None,
        "descriptor experiment manifest expected hash is invalid",
    )
    _require(
        manifest_file_hash == args.descriptor_experiment_manifest_sha256,
        "descriptor experiment manifest file hash mismatch",
    )
    expected_manifest_keys = {
        "schema_version",
        "artifact_type",
        "source_commit",
        "source_branch",
        "loss_spec_sha256",
        "checkpoint_selector",
        "weight_receipt",
        "cells",
        "content_hash_sha256",
    }
    _require(set(manifest) == expected_manifest_keys, "experiment manifest fields differ")
    manifest_content_hash = _validate_content_hash(
        manifest,
        name="descriptor experiment manifest",
    )
    _require(
        manifest["schema_version"] == EXPERIMENT_SCHEMA_VERSION
        and manifest["artifact_type"] == EXPERIMENT_ARTIFACT_TYPE,
        "wrong descriptor experiment manifest schema",
    )
    _require(manifest["loss_spec_sha256"] == LOSS_SPEC_SHA256, "manifest loss spec changed")
    _require(
        manifest["checkpoint_selector"] == CHECKPOINT_SELECTOR,
        "manifest checkpoint selector changed",
    )
    source_commit, source_branch = _source_state(root)
    _require(
        manifest["source_commit"] == source_commit
        and manifest["source_branch"] == source_branch,
        "experiment manifest source differs from exact checkout",
    )

    receipt_record = manifest["weight_receipt"]
    _require(
        isinstance(receipt_record, dict)
        and set(receipt_record)
        == {"absolute_path", "file_sha256", "content_hash_sha256"},
        "manifest weight receipt binding is invalid",
    )
    supplied_receipt_path = str(
        Path(args.descriptor_weight_receipt).expanduser().resolve(strict=True)
    )
    _require(
        receipt_record["absolute_path"] == supplied_receipt_path,
        "weight receipt path differs from manifest",
    )
    _require(
        receipt_record["file_sha256"] == args.descriptor_weight_receipt_sha256,
        "weight receipt CLI hash differs from manifest",
    )
    receipt, receipt_file_hash, receipt_content_hash = validate_weight_receipt(
        root,
        supplied_receipt_path,
        expected_file_sha256=args.descriptor_weight_receipt_sha256,
    )
    _require(
        receipt_record["content_hash_sha256"] == receipt_content_hash,
        "weight receipt content hash differs from manifest",
    )
    _require(receipt["source"]["commit"] == source_commit, "weight receipt source commit changed")

    cells = manifest["cells"]
    _require(isinstance(cells, list) and cells, "experiment manifest cells are empty")
    matches = [cell for cell in cells if cell.get("cell_id") == args.descriptor_cell_id]
    _require(len(matches) == 1, "descriptor cell ID is missing or duplicated")
    cell = matches[0]
    _require(
        isinstance(cell, dict)
        and set(cell) == {"cell_id", "arm", "training_arguments"},
        "descriptor cell fields differ from schema",
    )
    _require(cell["arm"] in {"control", "attachment"}, "descriptor cell arm is invalid")
    expected_arguments = cell["training_arguments"]
    _require(
        isinstance(expected_arguments, dict)
        and set(expected_arguments) == BOUND_TRAINING_ARGUMENTS,
        "descriptor cell does not bind every scientific training argument",
    )
    for name, expected in expected_arguments.items():
        actual = getattr(args, name, object())
        _require(
            type(actual) is type(expected) and actual == expected,
            f"descriptor argument --{name} differs from frozen cell",
        )
    lambda_attach = expected_arguments["lambda_attach"]
    if cell["arm"] == "control":
        _require(type(lambda_attach) is float and lambda_attach == 0.0, "control arm requires lambda_attach=0")
    else:
        _require(
            type(lambda_attach) is float
            and lambda_attach == receipt["lambda_attach"]
            and lambda_attach > 0.0,
            "attachment arm lambda differs from the weight receipt",
        )
    run_directory = (
        Path(expected_arguments["output_dir"]).expanduser().resolve()
        / args.descriptor_cell_id
    ).absolute()
    _require(not run_directory.exists(), "descriptor cell output already exists")
    return DescriptorAttachmentBinding(
        cell_id=args.descriptor_cell_id,
        arm=cell["arm"],
        lambda_attach=lambda_attach,
        run_directory=str(run_directory),
        experiment_manifest_file_sha256=manifest_file_hash,
        experiment_manifest_content_sha256=manifest_content_hash,
        weight_receipt_file_sha256=receipt_file_hash,
        weight_receipt_content_sha256=receipt_content_hash,
    )


def _descriptor_cell_identity(cell_id: str) -> tuple[str, str, int, str]:
    match = _DESCRIPTOR_CELL_RE.fullmatch(cell_id)
    _require(match is not None, "descriptor completed cell ID is invalid")
    recipe, arm, seed_text = match.groups()
    seed = int(seed_text)
    return recipe, arm, seed, f"{recipe}__r64__seed{seed}"


def _manifest_cell_from_config(
    repo_root: Path,
    config: Mapping[str, Any],
    *,
    expected_cell_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = config.get("descriptor_experiment_manifest")
    manifest_hash = config.get("descriptor_experiment_manifest_sha256")
    receipt_path = config.get("descriptor_weight_receipt")
    receipt_hash = config.get("descriptor_weight_receipt_sha256")
    _require(
        all(isinstance(value, str) and value for value in (
            manifest_path,
            manifest_hash,
            receipt_path,
            receipt_hash,
        )),
        "completed config lacks descriptor manifest/receipt bindings",
    )
    manifest, file_hash = _read_regular_json(
        str(manifest_path), name="descriptor experiment manifest"
    )
    _require(file_hash == manifest_hash, "completed manifest file hash differs")
    _require(
        manifest.get("schema_version") == EXPERIMENT_SCHEMA_VERSION
        and manifest.get("artifact_type") == EXPERIMENT_ARTIFACT_TYPE,
        "completed manifest schema differs",
    )
    _validate_content_hash(manifest, name="descriptor experiment manifest")
    source_commit, source_branch = _source_state(repo_root)
    _require(
        manifest.get("source_commit") == source_commit
        and manifest.get("source_branch") == source_branch,
        "completed manifest source differs",
    )
    _require(
        manifest.get("loss_spec_sha256") == LOSS_SPEC_SHA256
        and manifest.get("checkpoint_selector") == CHECKPOINT_SELECTOR,
        "completed manifest scientific contract differs",
    )
    receipt, _, receipt_content_hash = validate_weight_receipt(
        repo_root,
        str(receipt_path),
        expected_file_sha256=str(receipt_hash),
    )
    receipt_record = manifest.get("weight_receipt")
    _require(
        isinstance(receipt_record, Mapping)
        and receipt_record.get("absolute_path")
        == str(Path(str(receipt_path)).expanduser().resolve(strict=True))
        and receipt_record.get("file_sha256") == receipt_hash
        and receipt_record.get("content_hash_sha256") == receipt_content_hash,
        "completed manifest weight receipt differs",
    )
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "completed manifest cells are invalid")
    matches = [cell for cell in cells if cell.get("cell_id") == expected_cell_id]
    _require(len(matches) == 1, "completed manifest cell is missing or duplicated")
    cell = matches[0]
    expected_arguments = cell.get("training_arguments")
    _require(
        isinstance(expected_arguments, Mapping)
        and set(expected_arguments) == BOUND_TRAINING_ARGUMENTS,
        "completed cell argument contract differs",
    )
    for name, expected in expected_arguments.items():
        actual = config.get(name, object())
        _require(
            type(actual) is type(expected) and actual == expected,
            f"completed config argument {name} differs",
        )
    _require(
        config.get("descriptor_cell_id") == expected_cell_id
        and config.get("descriptor_loss_spec_sha256") == LOSS_SPEC_SHA256,
        "completed descriptor identity differs",
    )
    return manifest, cell, receipt


def _validation_selection(
    history: Any,
    *,
    epochs: int,
    log_every: int,
) -> tuple[Mapping[str, Any], list[int]]:
    _require(isinstance(history, list) and history, "completed history is empty")
    rows = [row for row in history if isinstance(row, Mapping) and "val_base_loss" in row]
    expected_epochs = [1, *range(log_every, epochs + 1, log_every)]
    _require(
        [row.get("epoch") for row in rows] == expected_epochs,
        "completed validation schedule differs",
    )
    for row in rows:
        for key in ("val_base_loss", "val_attachment_loss", "val_train_loss"):
            value = row.get(key)
            _require(
                type(value) in (int, float)
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"completed history {key} is invalid",
            )
    best = min(rows, key=lambda row: float(row["val_base_loss"]))
    return best, expected_epochs


def _expected_optimizer_steps(
    base_config: Mapping[str, Any], expected_arguments: Mapping[str, Any]
) -> int:
    train_pair_count = base_config["split"]["train"][
        "pair_count_after_object_filter"
    ]
    return int(expected_arguments["epochs"]) * math.ceil(
        int(train_pair_count) / int(expected_arguments["batch_size"])
    )


def _validate_checkpoint_config(
    checkpoint_config: Any,
    *,
    config: Mapping[str, Any],
    expected_arguments: Mapping[str, Any],
    expected_cell_id: str,
) -> None:
    _require(isinstance(checkpoint_config, Mapping), "descriptor checkpoint config is invalid")
    exact_keys = {
        "source_commit": config.get("source_commit"),
        "cell_id": expected_cell_id,
        "descriptor_attachment_binding": config.get("descriptor_attachment_binding"),
        "descriptor_loss_spec_sha256": LOSS_SPEC_SHA256,
        "num_keypoints": expected_arguments["num_keypoints"],
        "base_channels": expected_arguments["base_channels"],
        "heatmap_res": expected_arguments["heatmap_res"],
        "temperature": expected_arguments["temperature"],
        "padding_mode": expected_arguments["padding_mode"],
        "operator_type": expected_arguments["operator_type"],
        "lambda_attach": expected_arguments["lambda_attach"],
        "seed": expected_arguments["seed"],
        "data_root": expected_arguments["data_root"],
        "object": expected_arguments["object"],
        "train_pairs_index": expected_arguments["train_pairs_index"],
        "val_pairs_index": expected_arguments["val_pairs_index"],
        "test_pairs_index": expected_arguments["test_pairs_index"],
        "dataset_binding_sha256": expected_arguments["dataset_binding_sha256"],
        "training_mode": expected_arguments["indexed_mode"],
        "effective_epochs": expected_arguments["epochs"],
    }
    for key, expected in exact_keys.items():
        _require(
            key in checkpoint_config
            and type(checkpoint_config[key]) is type(expected)
            and checkpoint_config[key] == expected,
            f"descriptor checkpoint config {key} differs",
        )
    _require(
        checkpoint_config.get("checkpoint_policy") == config.get("checkpoint_policy")
        and checkpoint_config.get("index_provenance") == config.get("index_provenance")
        and checkpoint_config.get("determinism") == config.get("determinism")
        and checkpoint_config.get("determinism_amendment")
        == config.get("determinism_amendment"),
        "descriptor checkpoint embedded execution contract differs",
    )


def _validate_execution(
    root: Path,
    execution: Any,
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    expected_arguments: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    expected_keys = {
        "training_completed",
        "device",
        "optimizer_step_count",
        "determinism",
        "nondeterminism_evidence",
        "full_command",
        "runtime_environment",
    }
    _require(
        isinstance(execution, Mapping) and set(execution) == expected_keys,
        "descriptor execution fields differ",
    )
    _require(execution["training_completed"] is True, "descriptor training is incomplete")
    device = execution["device"]
    _require(isinstance(device, str) and device.split(":", 1)[0] == "cuda",
             "descriptor full run requires CUDA")
    _require(
        execution["optimizer_step_count"]
        == _expected_optimizer_steps(base_config, expected_arguments),
        "descriptor optimizer step count differs",
    )
    full_command = execution["full_command"]
    _require(
        isinstance(full_command, list)
        and full_command
        and all(isinstance(value, str) and value for value in full_command)
        and config.get("full_command") == full_command,
        "descriptor full command differs",
    )
    runtime = execution["runtime_environment"]
    expected_runtime_keys = {
        "python_implementation", "python_version", "pytorch_version",
        "torchvision_version", "numpy_version", "pytorch_cuda_version",
        "cudnn_version", "device_type", "gpu_name", "nvidia_driver_version",
        "driver_visible_cuda_version", "slurm_job_id",
        "primary_matrix_launch_lock", "environment_lock", "slurm_job_script",
    }
    _require(
        isinstance(runtime, Mapping)
        and set(runtime) == expected_runtime_keys
        and runtime.get("device_type") == "cuda"
        and isinstance(runtime.get("slurm_job_id"), str)
        and bool(runtime["slurm_job_id"])
        and config.get("runtime_environment") == runtime,
        "descriptor CUDA runtime environment differs",
    )
    for key in (
        "python_implementation", "python_version", "pytorch_version",
        "torchvision_version", "numpy_version", "pytorch_cuda_version",
        "gpu_name", "nvidia_driver_version", "driver_visible_cuda_version",
    ):
        _require(isinstance(runtime.get(key), str) and bool(runtime[key]),
                 f"descriptor runtime {key} is invalid")
    policy = fresh_roll_determinism.policy_for_heatmap_resolution(
        root, int(expected_arguments["heatmap_res"])
    )
    fresh_roll_determinism.validate_determinism_record(
        execution["determinism"],
        expected_seed=int(expected_arguments["seed"]),
        expected_policy=policy,
    )
    fresh_roll_determinism.validate_final_warning_evidence(
        execution["nondeterminism_evidence"],
        policy=policy,
        device_type="cuda",
    )
    _require(
        config.get("determinism") == execution["determinism"]
        and config.get("determinism_amendment")
        == fresh_roll_determinism.amendment_record(root),
        "descriptor determinism binding differs",
    )
    captures = []
    for name, record in (
        ("matrix lock", runtime["primary_matrix_launch_lock"]),
        ("environment lock", runtime["environment_lock"]),
        ("Slurm script", runtime["slurm_job_script"]),
    ):
        _require(
            isinstance(record, Mapping)
            and set(record) == {"absolute_path", "sha256", "size_bytes"}
            and _file_record(record["absolute_path"], name=name) == dict(record),
            f"descriptor {name} provenance differs",
        )
        captures.append(record)
    return captures[0], captures[1], captures[2]


def _safe_checkpoint_payload(path: Path, *, expected_record: Mapping[str, Any]) -> Mapping[str, Any]:
    live = _file_record(path, name="descriptor best checkpoint")
    _require(live == dict(expected_record), "descriptor checkpoint live file differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("descriptor checkpoint safe load failed") from exc
    _require(isinstance(payload, Mapping), "descriptor checkpoint is not a mapping")
    return payload


def write_completed_run_receipt(
    repo_root: Path | str,
    binding: DescriptorAttachmentBinding,
    *,
    device: str,
    optimizer_step_count: int,
    determinism: Mapping[str, Any],
    nondeterminism_evidence: Mapping[str, Any],
    full_command: list[str],
    runtime_environment: Mapping[str, Any],
) -> Path:
    """Exclusively write one checkpoint-bound descriptor completion receipt."""

    root = Path(repo_root).resolve(strict=True)
    source_commit, source_branch = _source_state(root)
    run_directory = Path(binding.run_directory).resolve(strict=True)
    config, config_record = _load_json_value(run_directory / "config.json", name="descriptor config")
    history, history_record = _load_json_value(run_directory / "history.json", name="descriptor history")
    _require(isinstance(config, Mapping), "descriptor config is not an object")
    _, cell, weight_receipt = _manifest_cell_from_config(
        root, config, expected_cell_id=binding.cell_id
    )
    expected_arguments = cell["training_arguments"]
    _, _, _, base_cell_id = _descriptor_cell_identity(binding.cell_id)
    base_config = fresh.resolve_fresh_cell(root, base_cell_id).expected_config
    best, validation_epochs = _validation_selection(
        history,
        epochs=int(expected_arguments["epochs"]),
        log_every=int(expected_arguments["log_every"]),
    )
    checkpoint_path = run_directory / "best_model.pt"
    checkpoint_record = _file_record(checkpoint_path, name="descriptor best checkpoint")
    payload = _safe_checkpoint_payload(checkpoint_path, expected_record=checkpoint_record)
    required_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "loss",
        "selection_loss_name",
        "base_validation_loss",
        "attachment_validation_loss",
        "augmented_validation_loss",
        "config",
    }
    _require(set(payload) == required_checkpoint_keys, "descriptor checkpoint keys differ")
    _require(
        payload["epoch"] == best["epoch"]
        and payload["selection_loss_name"] == "base_validation_loss"
        and float(payload["base_validation_loss"]) == float(best["val_base_loss"])
        and float(payload["attachment_validation_loss"])
        == float(best["val_attachment_loss"])
        and float(payload["augmented_validation_loss"])
        == float(best["val_train_loss"])
        and float(payload["loss"]) == float(best["val_base_loss"]),
        "descriptor checkpoint is not the first minimum base validation epoch",
    )
    _validate_checkpoint_config(
        payload["config"],
        config=config,
        expected_arguments=expected_arguments,
        expected_cell_id=binding.cell_id,
    )
    _require(
        isinstance(optimizer_step_count, int)
        and optimizer_step_count == _expected_optimizer_steps(base_config, expected_arguments),
        "descriptor optimizer step count differs",
    )
    matrix_record, environment_record, slurm_record = _validate_execution(
        root,
        {
            "training_completed": True,
            "device": device,
            "optimizer_step_count": optimizer_step_count,
            "determinism": dict(determinism),
            "nondeterminism_evidence": dict(nondeterminism_evidence),
            "full_command": list(full_command),
            "runtime_environment": dict(runtime_environment),
        },
        config=config,
        base_config=base_config,
        expected_arguments=expected_arguments,
    )
    source_files = [
        {
            "repo_relative_path": relative,
            **_file_record(root / relative, name=relative),
        }
        for relative in DESCRIPTOR_RUN_SOURCE_PATHS
    ]
    document: dict[str, Any] = {
        "schema_version": COMPLETED_RUN_SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_completed_run_receipt",
        "cell_id": binding.cell_id,
        "arm": binding.arm,
        "base_cell_id": base_cell_id,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "loss_spec_sha256": LOSS_SPEC_SHA256,
        "lambda_attach": binding.lambda_attach,
        "weight_receipt_content_sha256": weight_receipt["content_hash_sha256"],
        "artifacts": {
            "checkpoint": checkpoint_record,
            "config": config_record,
            "history": history_record,
        },
        "selection": {
            "selection_loss_name": "base_validation_loss",
            "checkpoint_epoch": best["epoch"],
            "base_validation_loss": best["val_base_loss"],
            "attachment_validation_loss": best["val_attachment_loss"],
            "augmented_validation_loss": best["val_train_loss"],
            "validation_epochs": validation_epochs,
        },
        "execution": {
            "training_completed": True,
            "device": device,
            "optimizer_step_count": optimizer_step_count,
            "determinism": dict(determinism),
            "nondeterminism_evidence": dict(nondeterminism_evidence),
            "full_command": list(full_command),
            "runtime_environment": dict(runtime_environment),
        },
        "source_files": source_files,
        "matrix_lock": dict(matrix_record),
        "environment_lock": dict(environment_record),
        "slurm_script": dict(slurm_record),
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    receipt_path = run_directory / COMPLETED_RUN_RECEIPT_NAME
    data = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise ValueError("descriptor completed receipt must be new") from exc
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return receipt_path


def authorize_completed_run(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    expected_cell_id: str,
) -> DescriptorCompletedRunCapability:
    """Validate a completed descriptor run and return a narrow load capability."""

    root = Path(repo_root).resolve(strict=True)
    source_commit, source_branch = _source_state(root)
    document, receipt_record = _load_json_value(receipt_path, name="descriptor completed receipt")
    _require(isinstance(document, Mapping), "descriptor completed receipt is not an object")
    required_receipt_keys = {
        "schema_version", "artifact_type", "cell_id", "arm", "base_cell_id",
        "source_commit", "source_branch", "loss_spec_sha256", "lambda_attach",
        "weight_receipt_content_sha256", "artifacts", "selection", "execution",
        "source_files", "matrix_lock", "environment_lock", "slurm_script",
        "content_hash_sha256",
    }
    _require(set(document) == required_receipt_keys,
             "descriptor completed receipt fields differ")
    _require(
        document.get("schema_version") == COMPLETED_RUN_SCHEMA_VERSION
        and document.get("artifact_type") == "descriptor_attachment_completed_run_receipt",
        "descriptor completed receipt schema differs",
    )
    _validate_content_hash(document, name="descriptor completed receipt")
    _require(
        document.get("cell_id") == expected_cell_id
        and document.get("source_commit") == source_commit
        and document.get("source_branch") == source_branch
        and document.get("loss_spec_sha256") == LOSS_SPEC_SHA256,
        "descriptor completed receipt identity differs",
    )
    _, arm, _, base_cell_id = _descriptor_cell_identity(expected_cell_id)
    _require(
        document.get("arm") == arm and document.get("base_cell_id") == base_cell_id,
        "descriptor completed receipt arm/base differs",
    )
    artifacts = document.get("artifacts")
    _require(isinstance(artifacts, Mapping) and set(artifacts) == {"checkpoint", "config", "history"},
             "descriptor completed artifact records differ")
    config, live_config = _load_json_value(artifacts["config"]["absolute_path"], name="descriptor config")
    history, live_history = _load_json_value(artifacts["history"]["absolute_path"], name="descriptor history")
    _require(live_config == artifacts["config"] and live_history == artifacts["history"],
             "descriptor config/history live files differ")
    _require(isinstance(config, Mapping), "descriptor config is invalid")
    _, cell, weight_receipt = _manifest_cell_from_config(
        root, config, expected_cell_id=expected_cell_id
    )
    expected_arguments = cell["training_arguments"]
    _require(
        document.get("lambda_attach") == expected_arguments["lambda_attach"]
        and document.get("weight_receipt_content_sha256")
        == weight_receipt["content_hash_sha256"],
        "descriptor completed weight binding differs",
    )
    base_config = fresh.resolve_fresh_cell(root, base_cell_id).expected_config
    best, validation_epochs = _validation_selection(
        history,
        epochs=int(expected_arguments["epochs"]),
        log_every=int(expected_arguments["log_every"]),
    )
    selection = document.get("selection")
    _require(
        isinstance(selection, Mapping)
        and selection.get("selection_loss_name") == "base_validation_loss"
        and selection.get("checkpoint_epoch") == best["epoch"]
        and selection.get("base_validation_loss") == best["val_base_loss"]
        and selection.get("attachment_validation_loss") == best["val_attachment_loss"]
        and selection.get("augmented_validation_loss") == best["val_train_loss"]
        and selection.get("validation_epochs") == validation_epochs,
        "descriptor completed selection differs",
    )
    checkpoint_record = artifacts["checkpoint"]
    checkpoint_path = Path(checkpoint_record["absolute_path"])
    payload = _safe_checkpoint_payload(checkpoint_path, expected_record=checkpoint_record)
    _require(
        set(payload) == {
            "epoch", "model_state_dict", "optimizer_state_dict", "loss",
            "selection_loss_name", "base_validation_loss",
            "attachment_validation_loss", "augmented_validation_loss", "config",
        }
        and payload.get("epoch") == best["epoch"]
        and payload.get("selection_loss_name") == "base_validation_loss"
        and float(payload.get("base_validation_loss")) == float(best["val_base_loss"]),
        "descriptor completed checkpoint selection differs",
    )
    _validate_checkpoint_config(
        payload["config"],
        config=config,
        expected_arguments=expected_arguments,
        expected_cell_id=expected_cell_id,
    )
    matrix_record, environment_record, slurm_record = _validate_execution(
        root,
        document.get("execution"),
        config=config,
        base_config=base_config,
        expected_arguments=expected_arguments,
    )
    _require(
        document.get("matrix_lock") == matrix_record
        and document.get("environment_lock") == environment_record
        and document.get("slurm_script") == slurm_record,
        "descriptor top-level execution bindings differ",
    )
    for record_name in ("matrix_lock", "environment_lock", "slurm_script"):
        record = document.get(record_name)
        _require(
            isinstance(record, Mapping)
            and _file_record(record["absolute_path"], name=record_name) == dict(record),
            f"descriptor completed {record_name} differs",
        )
    source_files = document.get("source_files")
    _require(isinstance(source_files, list), "descriptor source files are invalid")
    observed_paths = [record.get("repo_relative_path") for record in source_files
                      if isinstance(record, Mapping)]
    _require(
        observed_paths == list(DESCRIPTOR_RUN_SOURCE_PATHS),
        "descriptor source file set/order differs",
    )
    for source_record in source_files:
        relative = source_record.get("repo_relative_path")
        _require(relative in DESCRIPTOR_RUN_SOURCE_PATHS, "descriptor source path is unbound")
        live = {"repo_relative_path": relative, **_file_record(root / relative, name=relative)}
        _require(live == source_record, f"descriptor source file changed: {relative}")
    return DescriptorCompletedRunCapability(
        cell_id=expected_cell_id,
        arm=arm,
        base_cell_id=base_cell_id,
        source_commit=source_commit,
        checkpoint_absolute_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_record["sha256"],
        checkpoint_size_bytes=checkpoint_record["size_bytes"],
        config_absolute_path=artifacts["config"]["absolute_path"],
        config_sha256=artifacts["config"]["sha256"],
        config_size_bytes=artifacts["config"]["size_bytes"],
        history_absolute_path=artifacts["history"]["absolute_path"],
        history_sha256=artifacts["history"]["sha256"],
        history_size_bytes=artifacts["history"]["size_bytes"],
        receipt_absolute_path=receipt_record["absolute_path"],
        receipt_file_sha256=receipt_record["sha256"],
        receipt_size_bytes=receipt_record["size_bytes"],
        expected_training_arguments=dict(expected_arguments),
        expected_base_config=dict(base_config),
        _capability=_VERIFIED_DESCRIPTOR_CAPABILITY,
    )


def require_completed_run_capability(
    value: DescriptorCompletedRunCapability,
) -> DescriptorCompletedRunCapability:
    _require(
        isinstance(value, DescriptorCompletedRunCapability)
        and value._capability is _VERIFIED_DESCRIPTOR_CAPABILITY,
        "descriptor completed-run capability is not authentic",
    )
    return value


def reject_unbound_descriptor_arguments(args: argparse.Namespace) -> None:
    """Keep all non-experiment entry paths exactly on the legacy zero arm."""

    _require(
        type(args.lambda_attach) is float and args.lambda_attach == 0.0,
        "lambda_attach requires an authorized descriptor_cell_id",
    )
    for name in (
        "descriptor_experiment_manifest",
        "descriptor_experiment_manifest_sha256",
        "descriptor_weight_receipt",
        "descriptor_weight_receipt_sha256",
        "descriptor_loss_spec_sha256",
    ):
        _require(getattr(args, name) is None, f"--{name} is unbound without descriptor_cell_id")


__all__ = [
    "BOUND_TRAINING_ARGUMENTS",
    "COMPLETED_RUN_RECEIPT_NAME",
    "COMPLETED_RUN_SCHEMA_VERSION",
    "DescriptorAttachmentBinding",
    "DescriptorCompletedRunCapability",
    "EXPECTED_BRANCH",
    "EXPERIMENT_ARTIFACT_TYPE",
    "EXPERIMENT_SCHEMA_VERSION",
    "authorize_completed_run",
    "bind_training_namespace",
    "reject_unbound_descriptor_arguments",
    "require_completed_run_capability",
    "validate_weight_receipt",
    "write_completed_run_receipt",
]
