"""Run the bounded CUDA gate for the frozen roll head-package matrix.

This is implementation evidence only.  It executes two identical three-step
replicas for the four seed-42 recipe/head combinations, records repeatability
descriptively, safely reloads one checkpoint per cell on CPU, and checks the
restored model on CUDA.  It never writes a completed-run receipt and therefore
cannot authorize selection or scientific use of the checkpoints it creates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
KEYPOINT_ROOT = REPO_ROOT / "keypoint_net"
if str(KEYPOINT_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model as model_module
import fresh_roll_determinism as determinism_module
import representation_fresh_checkpoint_authorization as fresh
import train as train_module


EXPECTED_BRANCH = "agent/representation-oracles-20260726"
CELL_IDS = (
    "task55_clean__r64__seed42",
    "task55_clean__r128__seed42",
    "task80_assisted__r64__seed42",
    "task80_assisted__r128__seed42",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _build_model(training: Mapping[str, Any]) -> model_module.PhaseAModel:
    return model_module.PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=0,
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=training["learn_inverse_operator"],
        heatmap_res=training["heatmap_res"],
    )


def _head_parameter_count(model: model_module.PhaseAModel) -> int:
    count = sum(
        parameter.numel()
        for parameter in model.extractor.heatmap_head.parameters()
    )
    if model.extractor.head_upsample is not None:
        count += sum(
            parameter.numel()
            for parameter in model.extractor.head_upsample.parameters()
        )
    return count


def _safe_reload(path: Path) -> tuple[Mapping[str, Any], str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("smoke checkpoint is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise RuntimeError("smoke checkpoint payload is not a mapping")
    return payload, digest.hexdigest(), metadata.st_size


def _write_json_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)


def _required_environment(name: str, pattern: re.Pattern[str] | None = None) -> str:
    value = os.environ.get(name)
    if not value or (pattern is not None and pattern.fullmatch(value) is None):
        raise RuntimeError(f"required environment value {name} is invalid")
    return value


def _preflight_cells(data_root: Path, output_root: Path) -> list[tuple[Any, Any]]:
    prepared = []
    for cell_id in CELL_IDS:
        arguments = fresh.training_arguments(
            REPO_ROOT,
            cell_id,
            data_root=data_root,
            output_root=output_root / "unclaimed_fresh_output_roots" / cell_id,
        )
        namespace = argparse.Namespace(**arguments)
        binding = fresh.bind_training_namespace(REPO_ROOT, namespace)
        train_module._validate_training_arguments(namespace)
        plan = train_module._prepare_training_data(
            namespace,
            include_backward=False,
        )
        train_module._validate_fresh_data_plan(
            plan,
            binding,
            data_root=str(data_root),
        )
        if len(plan.train_dataset) != 147 or len(plan.val_dataset) != 21:
            raise RuntimeError(f"{cell_id} hammer pair counts differ")
        if Path(binding.run_directory).exists():
            raise RuntimeError(f"{cell_id} smoke preflight created a run directory")
        prepared.append((binding, plan))
    return prepared


def _run_replica(
    *,
    binding: Any,
    plan: Any,
    device: torch.device,
    source_commit: str,
    encoder_key_reference: tuple[str, ...] | None,
    replica_index: int,
    checkpoint_path: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    torch.Tensor,
    tuple[str, ...],
    int,
]:
    cell_id = binding.cell.cell_id
    training = dict(binding.cell.expected_config["training"])
    policy = determinism_module.policy_for_heatmap_resolution(
        REPO_ROOT,
        training["heatmap_res"],
    )
    determinism = train_module._configure_determinism(
        binding.cell.seed,
        policy=policy,
    )
    determinism["data_loader_workers"] = 0
    determinism_module.validate_determinism_record(
        determinism,
        expected_seed=binding.cell.seed,
        expected_policy=policy,
    )
    warning_evidence = determinism_module.new_warning_evidence(policy)

    model = _build_model(training).to(device)
    encoder_keys = tuple(model.extractor.encoder.state_dict().keys())
    if encoder_key_reference is not None and encoder_keys != encoder_key_reference:
        raise RuntimeError("64/128 encoder state keys differ")
    head_count = _head_parameter_count(model)
    expected_head_count = 1290 if training["heatmap_res"] == 64 else 74570
    if head_count != expected_head_count:
        raise RuntimeError("head parameter count differs")

    before_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    before_digest = _state_digest(model)
    optimizer = Adam(
        model.parameters(),
        lr=training["lr"],
        weight_decay=training["weight_decay"],
    )
    step_metrics = []
    for sample_index in range(3):
        loader = DataLoader(
            Subset(plan.train_dataset, [sample_index]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        metrics = train_module.train_epoch(
            model,
            loader,
            optimizer,
            device,
            training["lambda_smooth"],
            training["lambda_disp"],
            training["lambda_ent"],
            training["lambda_act"],
            training["lambda_loc"],
            training["lambda_inv"],
            training["lambda_cycle"],
            training["loc_bg_threshold"],
            training["sigma"],
            training["num_keypoints"],
            nondeterminism_policy=policy,
            nondeterminism_evidence=warning_evidence,
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("non-finite smoke metric")
        step_metrics.append({key: float(value) for key, value in metrics.items()})
    torch.cuda.synchronize(device)
    completed_warning_evidence = determinism_module.finalize_warning_evidence(
        warning_evidence,
        policy=policy,
        device_type="cuda",
    )
    changed_parameters = sorted(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before_parameters[name], parameter.detach().cpu())
    )
    after_digest = _state_digest(model)
    if not changed_parameters or before_digest == after_digest:
        raise RuntimeError("three CUDA optimizer steps did not change parameters")
    if not optimizer.state:
        raise RuntimeError("optimizer has no state after the CUDA update")
    if not all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise RuntimeError("model contains a non-finite parameter after update")

    checkpoint_record = None
    inference_model = model
    if checkpoint_path is not None:
        torch.save(
            {
                "artifact_type": "fresh_roll_cuda_smoke_checkpoint",
                "scope": "implementation_smoke_only_not_training_completion",
                "cell_id": cell_id,
                "source_commit": source_commit,
                "replica_index": replica_index,
                "optimizer_steps": 3,
                "determinism": determinism,
                "nondeterminism_evidence": completed_warning_evidence,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        payload, checkpoint_hash, checkpoint_size = _safe_reload(checkpoint_path)
        if payload.get("scope") != "implementation_smoke_only_not_training_completion":
            raise RuntimeError("smoke checkpoint scope differs")
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping) or not all(
            torch.is_tensor(value) and value.device.type == "cpu"
            for value in state.values()
        ):
            raise RuntimeError("safe smoke reload did not map every weight to CPU")
        restored = _build_model(training).cpu()
        restored.load_state_dict(state, strict=True)
        restored.requires_grad_(False)
        restored.eval()
        if _state_digest(restored) != after_digest:
            raise RuntimeError("restored smoke checkpoint weights differ")
        restored.to(device)
        inference_model = restored
        checkpoint_record = {
            "absolute_path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint_size,
            "scope": payload["scope"],
            "safe_same_descriptor_reload": True,
            "all_reloaded_weights_on_cpu": True,
            "weights_exact_after_reload": True,
        }

    validation_loader = DataLoader(
        Subset(plan.val_dataset, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    validation_batch = next(iter(validation_loader))
    x_t = validation_batch["x_t"].to(device, non_blocking=True)
    x_t1 = validation_batch["x_t1"].to(device, non_blocking=True)
    with torch.inference_mode():
        points, logits = inference_model.extractor(x_t)
        outputs = inference_model(x_t, x_t1)
    torch.cuda.synchronize(device)
    expected_resolution = training["heatmap_res"]
    if tuple(logits.shape) != (
        1,
        training["num_keypoints"],
        expected_resolution,
        expected_resolution,
    ):
        raise RuntimeError("restored heatmap shape differs")
    if tuple(points.shape) != (1, 2 * training["num_keypoints"]):
        raise RuntimeError("restored point shape differs")
    if tuple(outputs["p_hat_t1"].shape) != tuple(points.shape):
        raise RuntimeError("restored operator output shape differs")

    parameter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    heatmaps = logits.detach().cpu().clone()
    record = {
        "replica_index": replica_index,
        "seed_reset_to": binding.cell.seed,
        "optimizer_steps": 3,
        "sample_indices_in_order": [0, 1, 2],
        "step_metrics": step_metrics,
        "changed_parameter_count": len(changed_parameters),
        "before_state_sha256": before_digest,
        "after_state_sha256": after_digest,
        "checkpoint": checkpoint_record,
        "heatmap_shape": list(logits.shape),
        "point_shape": list(points.shape),
        "operator_output_shape": list(outputs["p_hat_t1"].shape),
        "determinism": determinism,
        "nondeterminism_evidence": completed_warning_evidence,
        "passed": True,
    }

    del before_parameters, optimizer, outputs, points, logits
    del validation_batch, validation_loader, loader, inference_model, model
    del x_t, x_t1
    gc.collect()
    torch.cuda.empty_cache()
    return record, parameter_state, heatmaps, encoder_keys, head_count


def _run_cell(
    *,
    binding: Any,
    plan: Any,
    output_root: Path,
    device: torch.device,
    source_commit: str,
    encoder_key_reference: tuple[str, ...] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    cell_id = binding.cell.cell_id
    training = dict(binding.cell.expected_config["training"])
    checkpoint_dir = output_root / "smoke_checkpoints" / cell_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    replicas = []
    parameter_states = []
    heatmaps = []
    encoder_keys = None
    head_count = None
    for replica_index in range(2):
        replica, parameters, fixed_heatmaps, current_encoder_keys, current_head_count = (
            _run_replica(
                binding=binding,
                plan=plan,
                device=device,
                source_commit=source_commit,
                encoder_key_reference=encoder_key_reference,
                replica_index=replica_index,
                checkpoint_path=(
                    checkpoint_dir / "replica0_three_step_cuda_smoke.pt"
                    if replica_index == 0
                    else None
                ),
            )
        )
        if encoder_keys is not None and current_encoder_keys != encoder_keys:
            raise RuntimeError("replica encoder state keys differ")
        if head_count is not None and current_head_count != head_count:
            raise RuntimeError("replica head parameter counts differ")
        encoder_keys = current_encoder_keys
        head_count = current_head_count
        replicas.append(replica)
        parameter_states.append(parameters)
        heatmaps.append(fixed_heatmaps)

    if set(parameter_states[0]) != set(parameter_states[1]):
        raise RuntimeError("replica parameter names differ")
    per_parameter_max_abs_difference = {
        name: float(
            (parameter_states[0][name] - parameter_states[1][name])
            .abs()
            .max()
            .item()
        )
        for name in sorted(parameter_states[0])
    }
    loss_differences = [
        abs(
            replicas[0]["step_metrics"][index]["loss"]
            - replicas[1]["step_metrics"][index]["loss"]
        )
        for index in range(3)
    ]
    comparison = {
        "decision_use": "descriptive_only",
        "numeric_acceptance_threshold": None,
        "state_digest_equal": (
            replicas[0]["after_state_sha256"]
            == replicas[1]["after_state_sha256"]
        ),
        "per_parameter_max_abs_difference": per_parameter_max_abs_difference,
        "maximum_parameter_abs_difference": max(
            per_parameter_max_abs_difference.values()
        ),
        "per_step_loss_absolute_difference": loss_differences,
        "maximum_fixed_batch_heatmap_absolute_difference": float(
            (heatmaps[0] - heatmaps[1]).abs().max().item()
        ),
        "used_as_pass_fail_criterion": False,
    }
    record = {
        "cell_id": cell_id,
        "recipe": binding.cell.recipe,
        "head_package": binding.cell.head_package,
        "seed": binding.cell.seed,
        "device": "cuda",
        "data_role": binding.cell.expected_config["object"],
        "train_pair_count": len(plan.train_dataset),
        "validation_pair_count": len(plan.val_dataset),
        "batch_size_used_for_bounded_smoke": 1,
        "frozen_full_run_batch_size": training["batch_size"],
        "replica_count": 2,
        "optimizer_steps_per_replica": 3,
        "head_parameter_count": head_count,
        "encoder_state_keys": list(encoder_keys),
        "determinism_amendment": determinism_module.amendment_record(
            REPO_ROOT
        ),
        "replicas": replicas,
        "repeatability_comparison": comparison,
        "fresh_cell_output_directory_created": False,
        "passed": True,
    }
    return record, encoder_keys


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().absolute()
    if COMMIT_RE.fullmatch(args.expected_commit) is None:
        raise RuntimeError("expected commit is invalid")
    if output_root.exists():
        raise RuntimeError(f"smoke output root already exists: {output_root}")
    if _git("rev-parse", "HEAD") != args.expected_commit:
        raise RuntimeError("HEAD differs from expected smoke commit")
    if _git("rev-parse", "--abbrev-ref", "HEAD") != EXPECTED_BRANCH:
        raise RuntimeError("branch differs from frozen fresh-run branch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is dirty")

    slurm_job_id = _required_environment("SLURM_JOB_ID")
    slurm_script_hash = _required_environment(
        "FRESH_ROLL_SLURM_SCRIPT_SHA256", SHA256_RE
    )
    environment_lock_hash = _required_environment(
        "FRESH_ROLL_ENV_LOCK_SHA256", SHA256_RE
    )
    script_hash = _sha256_file(Path(__file__).resolve())
    prepared = _preflight_cells(data_root, output_root)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before source/data preflight completed")

    output_root.mkdir(parents=True, exist_ok=False)
    report_path = output_root / "CUDA_SMOKE_REPORT.json"
    report: dict[str, Any] = {
        "schema_version": "fresh_roll_cuda_smoke.v2",
        "artifact_type": "fresh_roll_cuda_training_smoke_report",
        "scope": "deterministic_implementation_evidence_only_not_scientific_result",
        "source_commit": args.expected_commit,
        "source_branch": EXPECTED_BRANCH,
        "diagnostic_script": {
            "repo_relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "file_sha256": script_hash,
        },
        "slurm": {
            "job_id": slurm_job_id,
            "script_sha256": slurm_script_hash,
        },
        "environment_lock_sha256": environment_lock_hash,
        "determinism_amendment": determinism_module.amendment_record(
            REPO_ROOT
        ),
        "repeatability_contract": {
            "replicas_per_cell": 2,
            "optimizer_steps_per_replica": 3,
            "decision_use": "descriptive_only",
            "numeric_acceptance_threshold": None,
        },
        "constraints": {
            "full_training_run": False,
            "completed_run_receipt_issued_for_smoke_checkpoint": False,
            "selection_or_scientific_use_authorized": False,
            "repeatability_difference_used_as_pass_fail_criterion": False,
            "cuda_used": True,
            "protected_untracked_calibration_file_accessed": False,
            "cleanup_or_deletion_performed": False,
        },
        "cells": [],
    }
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the GPU allocation")
        device = torch.device("cuda:0")
        runtime_environment = train_module._runtime_environment(device)
        if runtime_environment["device_type"] != "cuda":
            raise RuntimeError("runtime environment did not record CUDA")
        report["environment"] = {
            **runtime_environment,
            "python_version_full": sys.version,
            "platform": platform.platform(),
            "numpy_version": str(np.__version__),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
        }
        encoder_keys = None
        for binding, plan in prepared:
            print(f"[cuda-smoke] start {binding.cell.cell_id}", flush=True)
            cell_record, encoder_keys = _run_cell(
                binding=binding,
                plan=plan,
                output_root=output_root,
                device=device,
                source_commit=args.expected_commit,
                encoder_key_reference=encoder_keys,
            )
            report["cells"].append(cell_record)
            print(f"[cuda-smoke] pass {binding.cell.cell_id}", flush=True)
        counts = {
            record["head_package"]: record["head_parameter_count"]
            for record in report["cells"]
        }
        report["head_parameter_delta_128_minus_64"] = counts["128"] - counts["64"]
        if report["head_parameter_delta_128_minus_64"] != 73280:
            raise RuntimeError("head parameter delta differs")
        report["passed"] = True
        report["stop_or_next_gate"] = "pass_to_substantive_review_before_full_matrix"
    except Exception as exc:
        report["passed"] = False
        report["stop_or_next_gate"] = "stop_before_full_matrix"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json_exclusive(report_path, report)
        raise
    _write_json_exclusive(report_path, report)
    print(f"[cuda-smoke] report {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
