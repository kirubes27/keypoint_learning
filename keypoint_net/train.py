"""
Phase A Training Script

Trains the keypoint extractor + linear operator as specified in the Action Plan.

Usage:
    # Single object (recommended for debugging)
    python train.py --data_root ./phase_a_yaw_only --object coffeemug --epochs 1000
    
    # All objects in train split
    python train.py --data_root ./phase_a_yaw_only --epochs 1000
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import PhaseAModel, compute_losses
from dataset import (
    IndexPairDataset,
    IndexPairManifest,
    PoseSequenceDataset,
    SingleObjectDataset,
    inspect_index_pair_manifest,
)
import representation_fresh_checkpoint_authorization as fresh_authorization
import representation_corpus_inventory


@dataclass(frozen=True)
class TrainingDataPlan:
    """Fully validated data/checkpoint policy prepared before run creation."""

    mode: str
    effective_epochs: int
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_manifest: Optional[IndexPairManifest]
    index_provenance: dict
    checkpoint_policy: dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str:
    repository_root = Path(__file__).resolve().parent.parent
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not bind the run to an exact Git commit") from exc


def _select_device() -> torch.device:
    """Prefer cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_determinism(seed: int) -> dict:
    """Enforce and report the frozen deterministic-algorithm policy."""
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config not in (None, ":4096:8"):
        raise ValueError(
            "fresh runs require CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cudnn_benchmark": bool(
            getattr(torch.backends.cudnn, "benchmark", False)
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _nvidia_driver_facts() -> tuple[str, str]:
    """Return driver and driver-visible CUDA versions from nvidia-smi."""
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("CUDA run cannot record nvidia-smi provenance") from exc
    driver = re.search(r"Driver Version:\s*([0-9.]+)", completed.stdout)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", completed.stdout)
    if driver is None or cuda is None:
        raise RuntimeError("CUDA run cannot parse nvidia-smi provenance")
    return driver.group(1), cuda.group(1)


def _runtime_environment(device: torch.device) -> dict:
    """Record the reviewed runtime fields without importing torchvision."""
    driver_version = None
    driver_visible_cuda_version = None
    gpu_name = None
    if device.type == "cuda":
        driver_version, driver_visible_cuda_version = _nvidia_driver_facts()
        gpu_name = torch.cuda.get_device_name(device)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_script_hash = os.environ.get("FRESH_ROLL_SLURM_SCRIPT_SHA256")
    if (slurm_job_id is None) != (slurm_script_hash is None):
        raise RuntimeError(
            "Slurm runs require both SLURM_JOB_ID and "
            "FRESH_ROLL_SLURM_SCRIPT_SHA256"
        )
    if slurm_script_hash is not None and re.fullmatch(
        r"[0-9a-f]{64}", slurm_script_hash
    ) is None:
        raise RuntimeError("FRESH_ROLL_SLURM_SCRIPT_SHA256 is invalid")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": str(np.__version__),
        "pytorch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "gpu_name": gpu_name,
        "nvidia_driver_version": driver_version,
        "driver_visible_cuda_version": driver_visible_cuda_version,
        "slurm_job_id": slurm_job_id,
        "slurm_job_script_sha256": slurm_script_hash,
    }


def _infer_eval_frames_dir(data_root: str, object_name: str):
    """
    Infer frames directory for post-training eval.
    """
    if not object_name:
        return None

    train_frames = Path(data_root) / "train" / object_name / "frames" / "a"
    test_frames = Path(data_root) / "test" / object_name / "frames" / "a"
    if train_frames.exists():
        return train_frames
    if test_frames.exists():
        return test_frames
    return None


def _run_auto_eval(
    run_dir: Path,
    checkpoint_path: Path,
    frames_dir: Path,
    img_size: int,
    frame_skip: int,
    eval_max_k: int,
    seed: int,
) -> None:
    """
    Run ablations, compositionality, and visualization scripts after training.
    """
    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    jobs = [
        (
            "ablations",
            [
                py,
                str(script_dir / "ablations.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--frames_dir",
                str(frames_dir),
                "--img_size",
                str(img_size),
                "--frame_skip",
                str(frame_skip),
                "--seed",
                str(seed),
                "--output_dir",
                str(run_dir / "ablations"),
            ],
        ),
        (
            "compositionality",
            [
                py,
                str(script_dir / "eval_compositionality.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--frames_dir",
                str(frames_dir),
                "--img_size",
                str(img_size),
                "--max_k",
                str(eval_max_k),
                "--output_dir",
                str(run_dir / "compositionality"),
            ],
        ),
        (
            "visualizations",
            [
                py,
                str(script_dir / "visualize.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--frames_dir",
                str(frames_dir),
                "--img_size",
                str(img_size),
                "--output_dir",
                str(run_dir / "visualizations"),
            ],
        ),
    ]

    print("\n[AutoEval] Running post-training evaluation...")
    print(f"[AutoEval] Checkpoint: {checkpoint_path}")
    print(f"[AutoEval] Frames: {frames_dir}")
    for job_name, cmd in jobs:
        print(f"[AutoEval] {job_name}...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"[AutoEval] WARNING: {job_name} failed with exit code "
                f"{exc.returncode}. Continuing."
            )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_smooth: float,
    lambda_disp: float,
    lambda_ent: float,
    lambda_act: float,
    lambda_loc: float,
    lambda_inv: float,
    lambda_cycle: float,
    loc_bg_threshold: float,
    sigma: float,
    num_keypoints: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_pred = 0.0
    total_smooth = 0.0
    total_disp = 0.0
    total_ent = 0.0
    total_act = 0.0
    total_loc = 0.0
    total_inv = 0.0
    total_cycle = 0.0
    total_act_acc = 0.0
    n_batches = 0

    for batch in loader:
        x_t = batch['x_t'].to(device)
        x_t1 = batch['x_t1'].to(device)
        action_labels = batch['action_label'].to(device).long()

        # Forward pass
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=lambda_smooth,
            lambda_disp=lambda_disp,
            lambda_ent=lambda_ent,
            lambda_act=lambda_act,
            lambda_loc=lambda_loc,
            lambda_inv=lambda_inv,
            lambda_cycle=lambda_cycle,
            sigma=sigma,
            num_keypoints=num_keypoints,
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=loc_bg_threshold,
        )

        # Backward pass
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += losses['loss'].item()
        total_pred += losses['l_pred'].item()
        total_smooth += losses['l_smooth'].item()
        total_disp += losses['l_disp'].item()
        total_ent += losses['l_ent'].item()
        total_act += losses['l_act'].item()
        total_loc += losses['l_loc'].item()
        total_inv += losses['l_inv'].item()
        total_cycle += losses['l_cycle'].item()
        total_act_acc += losses['act_acc'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'l_pred': total_pred / n_batches,
        'l_smooth': total_smooth / n_batches,
        'l_disp': total_disp / n_batches,
        'l_ent': total_ent / n_batches,
        'l_act': total_act / n_batches,
        'l_loc': total_loc / n_batches,
        'l_inv': total_inv / n_batches,
        'l_cycle': total_cycle / n_batches,
        'act_acc': total_act_acc / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_smooth: float,
    lambda_disp: float,
    lambda_ent: float,
    lambda_act: float,
    lambda_loc: float,
    lambda_inv: float,
    lambda_cycle: float,
    loc_bg_threshold: float,
    sigma: float,
    num_keypoints: int,
) -> dict:
    """Evaluate one explicitly supplied held-out loader."""
    model.eval()
    
    total_loss = 0.0
    total_pred = 0.0
    total_act = 0.0
    total_loc = 0.0
    total_inv = 0.0
    total_cycle = 0.0
    total_act_acc = 0.0
    n_batches = 0

    for batch in loader:
        x_t = batch['x_t'].to(device)
        x_t1 = batch['x_t1'].to(device)
        action_labels = batch['action_label'].to(device).long()

        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=lambda_smooth,
            lambda_disp=lambda_disp,
            lambda_ent=lambda_ent,
            lambda_act=lambda_act,
            lambda_loc=lambda_loc,
            lambda_inv=lambda_inv,
            lambda_cycle=lambda_cycle,
            sigma=sigma,
            num_keypoints=num_keypoints,
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=loc_bg_threshold,
        )

        total_loss += losses['loss'].item()
        total_pred += losses['l_pred'].item()
        total_act += losses['l_act'].item()
        total_loc += losses['l_loc'].item()
        total_inv += losses['l_inv'].item()
        total_cycle += losses['l_cycle'].item()
        total_act_acc += losses['act_acc'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'l_pred': total_pred / n_batches,
        'l_act': total_act / n_batches,
        'l_loc': total_loc / n_batches,
        'l_inv': total_inv / n_batches,
        'l_cycle': total_cycle / n_batches,
        'act_acc': total_act_acc / n_batches,
    }


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_training_arguments(args: argparse.Namespace) -> None:
    """Fail closed on argument contradictions before device/run allocation."""
    if args.lambda_act > 0 and args.num_action_classes < 2:
        raise ValueError("--num_action_classes must be >= 2 when --lambda_act > 0")
    for name in (
        "epochs",
        "batch_size",
        "save_every",
        "log_every",
        "img_size",
        "frame_skip",
        "eval_max_k",
        "num_keypoints",
        "base_channels",
    ):
        _validate_positive_int(getattr(args, name), f"--{name}")
    if args.frozen_epochs is not None:
        _validate_positive_int(args.frozen_epochs, "--frozen_epochs")
    if args.center_crop is not None:
        _validate_positive_int(args.center_crop, "--center_crop")
    for name in ("lr", "temperature", "sigma", "yaw_step_deg"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight_decay must be finite and non-negative")
    for name in (
        "lambda_smooth",
        "lambda_disp",
        "lambda_ent",
        "lambda_act",
        "lambda_loc",
        "lambda_inv",
        "lambda_cycle",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if (
        not np.isfinite(args.loc_bg_threshold)
        or not 0 <= args.loc_bg_threshold <= 255
    ):
        raise ValueError("--loc_bg_threshold must be finite and within [0, 255]")
    if (
        isinstance(args.num_action_classes, bool)
        or not isinstance(args.num_action_classes, int)
        or args.num_action_classes < 0
    ):
        raise ValueError("--num_action_classes must be a non-negative integer")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("--seed must be an integer")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("--output_dir must be a non-empty path")


def _resolve_index_argument_policy(args: argparse.Namespace) -> dict:
    """
    Resolve the explicit indexed-mode argument matrix.

    The historical ``--pairs_index`` spelling remains available only as a
    deprecated alias for ``--train_pairs_index`` inside an explicit safe mode.
    It can no longer create a train-as-validation indexed run.
    """
    indexed_values = (
        args.indexed_mode,
        args.pairs_index,
        args.train_pairs_index,
        args.val_pairs_index,
        args.test_pairs_index,
        args.dataset_binding_sha256,
        args.frozen_epochs,
    )
    if not any(value is not None for value in indexed_values):
        return {"mode": "legacy"}

    if args.indexed_mode is None:
        raise ValueError(
            "Indexed pair files require explicit --indexed_mode "
            "development or fixed-final; no train-as-validation fallback exists"
        )
    if args.pairs_index is not None and args.train_pairs_index is not None:
        raise ValueError(
            "Pass only one of --pairs_index and --train_pairs_index"
        )
    train_index = args.train_pairs_index or args.pairs_index
    if train_index is None:
        raise ValueError("Indexed mode requires --train_pairs_index")
    if args.object is None:
        raise ValueError(
            "Indexed mode requires --object so each object-specific model has "
            "one frozen recipe role"
        )
    if args.dataset_binding_sha256 is None:
        raise ValueError(
            "Indexed mode requires independently verified "
            "--dataset_binding_sha256"
        )
    if args.auto_eval:
        raise ValueError(
            "--auto_eval is forbidden for indexed modes because the legacy "
            "whole-directory evaluator is not partition-aware"
        )

    holdout_count = sum(
        value is not None
        for value in (args.val_pairs_index, args.test_pairs_index)
    )
    if holdout_count != 1:
        raise ValueError(
            "Indexed mode requires exactly one of --val_pairs_index or "
            "--test_pairs_index"
        )

    if args.indexed_mode == "development":
        if args.val_pairs_index is None or args.test_pairs_index is not None:
            raise ValueError(
                "development mode requires train+validation indices and no "
                "test index"
            )
        if args.frozen_epochs is not None:
            raise ValueError(
                "--frozen_epochs is reserved for fixed-final mode"
            )
        return {
            "mode": "development",
            "train_index": train_index,
            "holdout_index": args.val_pairs_index,
            "holdout_split": "validation",
            "effective_epochs": args.epochs,
        }

    if args.indexed_mode == "fixed-final":
        if args.test_pairs_index is None or args.val_pairs_index is not None:
            raise ValueError(
                "fixed-final mode requires train+test indices and no "
                "validation index"
            )
        if args.frozen_epochs is None:
            raise ValueError("fixed-final mode requires --frozen_epochs")
        if args.epochs != args.frozen_epochs:
            raise ValueError(
                "--epochs must exactly equal --frozen_epochs in fixed-final mode"
            )
        return {
            "mode": "fixed-final",
            "train_index": train_index,
            "holdout_index": args.test_pairs_index,
            "holdout_split": "test",
            "effective_epochs": args.frozen_epochs,
        }

    raise ValueError(f"Unknown indexed mode: {args.indexed_mode!r}")


def _normalise_role(role: str) -> str:
    return role.strip().lower().replace("-", "_").replace(" ", "_")


def _validate_manifest_pair(
    train_manifest: IndexPairManifest,
    holdout_manifest: IndexPairManifest,
    *,
    object_name: str,
    mode: str,
) -> None:
    """Require one corpus/family and frame-disjoint endpoints for both roles."""
    train_meta = train_manifest.metadata
    holdout_meta = holdout_manifest.metadata
    for key in (
        "dataset_basename",
        "dataset_binding_sha256",
        "dataset_semantic_lock_sha256",
        "source_pair_index_relpath",
        "source_pair_index_sha256",
        "unversioned_source",
        "generator",
        "object_roles",
        "transform",
        "frame_partition",
    ):
        if train_meta.get(key) != holdout_meta.get(key):
            raise ValueError(
                f"Train and holdout pair indices disagree on {key!r}"
            )

    object_roles = train_meta["object_roles"]
    if object_name not in object_roles:
        raise ValueError(
            f"Object {object_name!r} is absent from the pair-index role lock"
        )
    role = _normalise_role(object_roles[object_name])
    if mode == "development" and role != "development":
        raise ValueError(
            f"development mode requires a development object, got role={role!r}"
        )
    if mode == "fixed-final" and role not in {"confirmation", "final_test"}:
        raise ValueError(
            "fixed-final mode requires a confirmation or final_test object, "
            f"got role={role!r}"
        )

    overlap = train_manifest.endpoint_ids & holdout_manifest.endpoint_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "Train and holdout indices share source/target frame endpoints; "
            f"overlap_count={len(overlap)}, examples={examples}"
        )


def _manifest_provenance(manifest: IndexPairManifest) -> dict:
    return {
        "resolved_path": str(manifest.index_path),
        "file_sha256": manifest.index_sha256,
        "content_hash_sha256": manifest.content_hash_sha256,
        "dataset_binding_sha256": manifest.dataset_binding_sha256,
        "split": manifest.split,
        "pair_count_after_object_filter": len(manifest.pairs),
        "endpoint_count_after_object_filter": len(manifest.endpoint_ids),
        "transform": manifest.transform,
    }


def _validate_fresh_data_plan(
    data_plan: TrainingDataPlan,
    binding: fresh_authorization.FreshRunBinding,
    *,
    data_root: str,
) -> None:
    """Cross-bind the prepared datasets to the manifest cell."""
    if data_plan.mode != "development" or data_plan.test_manifest is not None:
        raise ValueError("fresh cell must use development mode without a test loader")
    expected = binding.cell.expected_config
    for split in ("train", "validation"):
        actual = data_plan.index_provenance.get(split)
        frozen = expected["split"][split]
        expected_path = (
            binding.cell.train_pairs_path
            if split == "train"
            else binding.cell.validation_pairs_path
        )
        actual_transform = actual.get("transform") if isinstance(actual, dict) else None
        frozen_transform = expected["transform"]
        transform_matches = (
            isinstance(actual_transform, dict)
            and set(actual_transform) == set(frozen_transform) | {"expected_2d_family"}
            and all(
                actual_transform.get(key) == value
                for key, value in frozen_transform.items()
            )
            and actual_transform.get("expected_2d_family")
            == "planar_rotation_about_projected_center"
        )
        if not isinstance(actual, dict) or (
            actual.get("resolved_path") != expected_path
            or actual.get("file_sha256") != frozen["file_sha256"]
            or actual.get("content_hash_sha256") != frozen["content_hash_sha256"]
            or actual.get("dataset_binding_sha256")
            != expected["dataset"]["binding_sha256"]
            or actual.get("pair_count_after_object_filter")
            != frozen["pair_count_after_object_filter"]
            or not transform_matches
        ):
            raise ValueError(f"fresh {split} split provenance differs")
    inventory_record = expected["dataset"]["corpus_inventory"]
    inventory_path = Path(__file__).resolve().parent.parent / inventory_record[
        "repo_relative_path"
    ]
    inventory = representation_corpus_inventory.validate_corpus_inventory(
        inventory_path.read_bytes(),
        "roll",
        data_root,
    )
    if inventory.content_hash_sha256 != expected["dataset"]["binding_sha256"]:
        raise ValueError("fresh live corpus binding differs")


def _prepare_training_data(
    args: argparse.Namespace,
    *,
    include_backward: bool,
) -> TrainingDataPlan:
    """
    Validate every data input before a run directory or device is created.

    In fixed-final mode this inspects test metadata, paths, hashes, and endpoint
    sets, but deliberately does not instantiate the test Dataset.
    """
    policy = _resolve_index_argument_policy(args)
    mode = policy["mode"]

    if mode in {"development", "fixed-final"}:
        train_manifest = inspect_index_pair_manifest(
            data_root=args.data_root,
            index_path=policy["train_index"],
            expected_split="train",
            object_name=args.object,
            strict_metadata=True,
            expected_dataset_binding_sha256=args.dataset_binding_sha256,
            validate_paths=True,
        )
        holdout_manifest = inspect_index_pair_manifest(
            data_root=args.data_root,
            index_path=policy["holdout_index"],
            expected_split=policy["holdout_split"],
            object_name=args.object,
            strict_metadata=True,
            expected_dataset_binding_sha256=args.dataset_binding_sha256,
            validate_paths=True,
        )
        _validate_manifest_pair(
            train_manifest,
            holdout_manifest,
            object_name=args.object,
            mode=mode,
        )

        train_dataset = IndexPairDataset(
            data_root=args.data_root,
            index_path=str(train_manifest.index_path),
            img_size=args.img_size,
            center_crop=args.center_crop,
            include_backward=include_backward,
            object_name=args.object,
            strict_metadata=True,
            expected_split="train",
            expected_index_sha256=train_manifest.index_sha256,
            expected_dataset_binding_sha256=args.dataset_binding_sha256,
            manifest=train_manifest,
        )
        index_provenance = {
            "train": _manifest_provenance(train_manifest),
            policy["holdout_split"]: _manifest_provenance(holdout_manifest),
        }

        if mode == "development":
            val_dataset = IndexPairDataset(
                data_root=args.data_root,
                index_path=str(holdout_manifest.index_path),
                img_size=args.img_size,
                center_crop=args.center_crop,
                include_backward=include_backward,
                object_name=args.object,
                strict_metadata=True,
                expected_split="validation",
                expected_index_sha256=holdout_manifest.index_sha256,
                expected_dataset_binding_sha256=args.dataset_binding_sha256,
                manifest=holdout_manifest,
            )
            checkpoint_policy = {
                "mode": "development",
                "selection": "minimum_total_validation_loss",
                "authoritative_checkpoint": "best_model.pt",
                "best_model_written": True,
                "validation_loader": True,
                "test_loader": False,
                "post_training_test_evaluations": 0,
                "epochs": policy["effective_epochs"],
            }
            return TrainingDataPlan(
                mode=mode,
                effective_epochs=policy["effective_epochs"],
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_manifest=None,
                index_provenance=index_provenance,
                checkpoint_policy=checkpoint_policy,
            )

        checkpoint_policy = {
            "mode": "fixed-final",
            "selection": "none",
            "authoritative_checkpoint": "final_model.pt",
            "best_model_written": False,
            "validation_loader": False,
            "test_loader": "constructed_after_final_model",
            "post_training_test_evaluations": 1,
            "epochs": policy["effective_epochs"],
            "epochs_source": "--frozen_epochs",
        }
        return TrainingDataPlan(
            mode=mode,
            effective_epochs=policy["effective_epochs"],
            train_dataset=train_dataset,
            val_dataset=None,
            test_manifest=holdout_manifest,
            index_provenance=index_provenance,
            checkpoint_policy=checkpoint_policy,
        )

    # Historical whole-directory modes remain available and retain their
    # historical checkpoint semantics. No index file can reach this branch.
    if args.object:
        train_frames = Path(args.data_root) / "train" / args.object / "frames" / "a"
        test_frames = Path(args.data_root) / "test" / args.object / "frames" / "a"

        if train_frames.exists():
            train_dataset = SingleObjectDataset(
                str(train_frames),
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
                include_backward=include_backward,
            )
            val_dataset = train_dataset
        elif test_frames.exists():
            train_dataset = SingleObjectDataset(
                str(test_frames),
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
                include_backward=include_backward,
            )
            val_dataset = train_dataset
        else:
            train_dataset = PoseSequenceDataset(
                args.data_root,
                split="train",
                object_name=args.object,
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
                include_backward=include_backward,
            )
            val_dataset = train_dataset
    else:
        train_dataset = PoseSequenceDataset(
            args.data_root,
            split="train",
            img_size=args.img_size,
            frame_skip=args.frame_skip,
            center_crop=args.center_crop,
            include_backward=include_backward,
        )
        try:
            val_dataset = PoseSequenceDataset(
                args.data_root,
                split="test",
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
                include_backward=include_backward,
            )
        except (FileNotFoundError, ValueError, KeyError):
            print("No test split found, using train for legacy validation")
            val_dataset = train_dataset

    return TrainingDataPlan(
        mode="legacy",
        effective_epochs=args.epochs,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_manifest=None,
        index_provenance={},
        checkpoint_policy={
            "mode": "legacy",
            "selection": "minimum_legacy_validation_loss",
            "authoritative_checkpoint": "best_model.pt",
            "best_model_written": True,
            "validation_loader": True,
            "test_loader": False,
            "post_training_test_evaluations": 0,
            "epochs": args.epochs,
        },
    )


def _build_fixed_final_test_loader(
    plan: TrainingDataPlan,
    args: argparse.Namespace,
    *,
    final_model_path: Path,
    use_cuda: bool,
    n_workers: int,
) -> DataLoader:
    """Construct the still-hashed test Dataset only after final_model exists."""
    if plan.mode != "fixed-final" or plan.test_manifest is None:
        raise ValueError("Fixed-final test loader requested for a non-final plan")
    if not final_model_path.is_file():
        raise RuntimeError(
            "Authoritative final_model.pt must exist before test Dataset creation"
        )
    manifest = plan.test_manifest
    test_dataset = IndexPairDataset(
        data_root=args.data_root,
        index_path=str(manifest.index_path),
        img_size=args.img_size,
        center_crop=args.center_crop,
        include_backward=args.lambda_act > 0.0,
        object_name=args.object,
        strict_metadata=True,
        expected_split="test",
        expected_index_sha256=manifest.index_sha256,
        expected_dataset_binding_sha256=args.dataset_binding_sha256,
    )
    overlap = plan.train_dataset.endpoint_ids & test_dataset.endpoint_ids
    if overlap:
        raise ValueError(
            "Test endpoints changed after preflight and now overlap training"
        )
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=use_cuda,
    )


def main():
    parser = argparse.ArgumentParser(description="Phase A Training")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument(
        "--fresh_cell_id",
        type=str,
        default=None,
        help=(
            "Internal exact-cell binding used only by run_fresh_roll_cell.py; "
            "all scientific arguments must match the committed cell."
        ),
    )
    parser.add_argument("--object", type=str, default=None, help="Single object name (e.g., coffeemug)")
    parser.add_argument("--pairs_index", type=str, default=None,
                        help="Deprecated alias for --train_pairs_index. It is "
                             "accepted only with an explicit --indexed_mode and "
                             "matching validation/test index; index-only legacy "
                             "calls fail closed.")
    parser.add_argument(
        "--indexed_mode",
        type=str,
        default=None,
        choices=["development", "fixed-final"],
        help=(
            "Strict indexed protocol. development requires explicit train and "
            "validation indices; fixed-final requires train and test indices."
        ),
    )
    parser.add_argument(
        "--train_pairs_index",
        type=str,
        default=None,
        help="Strict train pair-index JSON, absolute or relative to --data_root.",
    )
    parser.add_argument(
        "--val_pairs_index",
        type=str,
        default=None,
        help="Strict validation pair-index JSON for development mode.",
    )
    parser.add_argument(
        "--test_pairs_index",
        type=str,
        default=None,
        help="Strict one-shot test pair-index JSON for fixed-final mode.",
    )
    parser.add_argument(
        "--dataset_binding_sha256",
        type=str,
        default=None,
        help=(
            "Independently verified full-corpus binding required by strict "
            "indexed modes; must match every supplied pair index."
        ),
    )
    parser.add_argument("--img_size", type=int, default=256)
    
    # Model
    parser.add_argument("--num_keypoints", type=int, default=10, help="N keypoints")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--heatmap_res", type=int, default=64, choices=[64, 128],
                        help="Heatmap resolution at 512 input: 64 = legacy /8 head; "
                             "128 = feature-upsample + 3x3 conv head at /4 "
                             "(halves the soft-argmax cell size)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Soft-argmax temperature")
    parser.add_argument(
        "--padding_mode",
        type=str,
        default="reflect",
        choices=["zeros", "reflect", "replicate", "circular"],
        help="Conv2d padding_mode used in keypoint extractor",
    )
    parser.add_argument(
        "--operator_type",
        type=str,
        default="dense",
        choices=["dense", "shared_affine"],
        help=(
            "Transition operator family. 'dense' is the legacy 2N x 2N map; "
            "'shared_affine' applies the same 2D affine map to every keypoint."
        ),
    )
    parser.add_argument(
        "--learn_inverse_operator",
        action="store_true",
        help="Learn a separate inverse operator W- for inverse/cycle losses.",
    )
    
    # Loss weights -- Run 0 defaults: only L_pred + L_disp
    parser.add_argument("--lambda_smooth", type=float, default=0.0, help="λ_s for L_smooth (0 for Run 0)")
    parser.add_argument("--lambda_disp", type=float, default=0.1, help="λ_d for L_disp")
    parser.add_argument("--lambda_ent", type=float, default=0.0, help="λ_e for L_ent (0 for Run 0)")
    parser.add_argument("--lambda_act", type=float, default=0.0, help="λ_a for L_act action classification loss")
    parser.add_argument("--lambda_loc", type=float, default=0.0, help="λ_l for L_loc localization loss")
    parser.add_argument("--lambda_inv", type=float, default=0.0, help="λ_i for inverse prediction W- K_{t+1} ~= K_t")
    parser.add_argument("--lambda_cycle", type=float, default=0.0, help="λ_c for one-step W- W+ and W+ W- cycle consistency")
    parser.add_argument("--sigma", type=float, default=0.1, help="σ for L_disp length scale")
    parser.add_argument("--loc_bg_threshold", type=float, default=30.0,
                        help="Background subtraction threshold for L_loc (0-255 scale)")
    parser.add_argument("--num_action_classes", type=int, default=2, help="Number of action classes for action head")
    
    # Dataset (10.4)
    parser.add_argument("--frame_skip", type=int, default=3, help="Frame skip for angular step (1=2 deg, 3=6 deg if yaw_step=2)")
    parser.add_argument("--yaw_step_deg", type=float, default=2.0, help="Yaw step in degrees used when generating the dataset")
    parser.add_argument("--center_crop", type=int, default=None, help="Center crop size before resize (reduces background)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--frozen_epochs",
        type=int,
        default=None,
        help=(
            "Frozen epoch count for fixed-final mode. It must exactly equal "
            "--epochs; the final checkpoint is authoritative."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10, help="Log metrics every N epochs")
    parser.add_argument("--auto_eval", action="store_true",
                        help="Run ablations/compositionality/visualizations after training")
    parser.add_argument("--auto_eval_checkpoint", type=str, default="best",
                        choices=["best", "final"],
                        help="Checkpoint to use for auto eval")
    parser.add_argument("--eval_frames_dir", type=str, default=None,
                        help="Frames directory for auto eval. If omitted and --object is set, inferred from data_root.")
    parser.add_argument("--eval_max_k", type=int, default=10,
                        help="max_k for compositionality when --auto_eval is enabled")
    
    args = parser.parse_args()

    # All fresh-cell, indexed-mode, path/hash, and endpoint checks happen
    # before device selection and before any output directory is created.
    repo_root = Path(__file__).resolve().parent.parent
    fresh_binding = None
    if args.fresh_cell_id is not None:
        fresh_binding = fresh_authorization.bind_training_namespace(
            repo_root,
            args,
        )
    _validate_training_arguments(args)
    include_backward = args.lambda_act > 0.0
    data_plan = _prepare_training_data(
        args,
        include_backward=include_backward,
    )
    if fresh_binding is not None:
        _validate_fresh_data_plan(
            data_plan,
            fresh_binding,
            data_root=args.data_root,
        )
    train_dataset = data_plan.train_dataset
    val_dataset = data_plan.val_dataset
    args.epochs = data_plan.effective_epochs
    source_commit = _source_commit()

    learn_inverse_operator = (
        args.learn_inverse_operator
        or args.lambda_inv > 0.0
        or args.lambda_cycle > 0.0
    )
    
    # Claim the fresh cell atomically after every source/data preflight and
    # before any CUDA query can initialize a device context.
    if fresh_binding is not None:
        run_dir = Path(fresh_binding.run_directory)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)

    # Reproducibility. Preserve legacy behavior outside the dedicated path.
    if fresh_binding is not None:
        determinism = _configure_determinism(args.seed)
    else:
        _seed_everything(args.seed)
        determinism = None
    print(f"Seed: {args.seed}")
    
    # Setup device
    device = _select_device()
    print(f"Using device: {device}")

    # pin_memory only helps on CUDA; num_workers=0 is safest on MPS/CPU.
    use_cuda = device.type == "cuda"
    n_workers = 4 if use_cuda else 0
    if determinism is not None:
        determinism["data_loader_workers"] = n_workers
    runtime_environment = (
        _runtime_environment(device) if fresh_binding is not None else None
    )
    full_command = (
        [sys.executable, *sys.argv] if fresh_binding is not None else None
    )
    
    # Create output directory
    # Include pid (and microseconds) to avoid collisions when launching
    # multiple runs in parallel.
    if fresh_binding is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        obj_name = args.object or "all"
        run_name = f"phase_a_{obj_name}_{timestamp}_seed{args.seed}_pid{os.getpid()}"
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = dict(vars(args))
    config['device'] = str(device)
    config['learn_inverse_operator_effective'] = learn_inverse_operator
    config['training_mode'] = data_plan.mode
    config['effective_epochs'] = data_plan.effective_epochs
    config['checkpoint_policy'] = data_plan.checkpoint_policy
    config['index_provenance'] = data_plan.index_provenance
    config['source_commit'] = source_commit
    if fresh_binding is not None:
        config['determinism'] = determinism
        config['full_command'] = full_command
        config['runtime_environment'] = runtime_environment
        config['cell_id'] = fresh_binding.cell.cell_id
        config['fresh_run_contract'] = dict(fresh_binding.cell.expected_config)
        config['experiment_manifest'] = {
            'file_sha256': fresh_binding.cell.manifest_file_sha256,
            'content_hash_sha256': (
                fresh_authorization.EXPERIMENT_MANIFEST_CONTENT_SHA256
            ),
        }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Output directory: {run_dir}")
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")
    elif data_plan.mode == "fixed-final":
        print("Validation samples: none (fixed-final checkpoint policy)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=use_cuda,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=use_cuda,
        )
    
    # Create model
    model_num_action_classes = args.num_action_classes if args.lambda_act > 0 else 0
    model = PhaseAModel(
        num_keypoints=args.num_keypoints,
        base_channels=args.base_channels,
        temperature=args.temperature,
        num_action_classes=model_num_action_classes,
        padding_mode=args.padding_mode,
        operator_type=args.operator_type,
        learn_inverse_operator=learn_inverse_operator,
        heatmap_res=args.heatmap_res,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    history = []
    best_loss = float('inf')
    optimizer_step_count = 0
    
    # Full config dict saved into every checkpoint for reproducibility
    ckpt_config = {
        'num_keypoints': args.num_keypoints,
        'base_channels': args.base_channels,
        'heatmap_res': args.heatmap_res,
        'temperature': args.temperature,
        'padding_mode': args.padding_mode,
        'operator_type': args.operator_type,
        'learn_inverse_operator': learn_inverse_operator,
        'frame_skip': args.frame_skip,
        'yaw_step_deg': args.yaw_step_deg,
        'lambda_smooth': args.lambda_smooth,
        'lambda_disp': args.lambda_disp,
        'lambda_ent': args.lambda_ent,
        'lambda_act': args.lambda_act,
        'lambda_loc': args.lambda_loc,
        'lambda_inv': args.lambda_inv,
        'lambda_cycle': args.lambda_cycle,
        'num_action_classes': model_num_action_classes,
        'sigma': args.sigma,
        'loc_bg_threshold': args.loc_bg_threshold,
        'img_size': args.img_size,
        'center_crop': args.center_crop,
        'seed': args.seed,
        'data_root': args.data_root,
        'resolved_data_root': str(Path(args.data_root).expanduser().resolve()),
        'object': args.object,
        'pairs_index': args.pairs_index,
        'train_pairs_index': args.train_pairs_index,
        'val_pairs_index': args.val_pairs_index,
        'test_pairs_index': args.test_pairs_index,
        'dataset_binding_sha256': args.dataset_binding_sha256,
        'training_mode': data_plan.mode,
        'effective_epochs': data_plan.effective_epochs,
        'frozen_epochs': args.frozen_epochs,
        'checkpoint_policy': data_plan.checkpoint_policy,
        'index_provenance': data_plan.index_provenance,
        'indexed_transform': (
            train_dataset.transform_metadata
            if isinstance(train_dataset, IndexPairDataset)
            else None
        ),
        'source_commit': source_commit,
    }
    if fresh_binding is not None:
        ckpt_config['cell_id'] = fresh_binding.cell.cell_id
        ckpt_config['fresh_run_contract'] = dict(
            fresh_binding.cell.expected_config
        )
        ckpt_config['determinism'] = determinism
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training mode: {data_plan.mode}")
    print(f"Checkpoint policy: {data_plan.checkpoint_policy}")
    print(f"Operator: {args.operator_type} (learn_inverse_operator={learn_inverse_operator})")
    print(f"Loss weights: lambda_smooth={args.lambda_smooth}, lambda_disp={args.lambda_disp}, "
          f"lambda_ent={args.lambda_ent}, lambda_act={args.lambda_act}, lambda_loc={args.lambda_loc}, "
          f"lambda_inv={args.lambda_inv}, lambda_cycle={args.lambda_cycle}, sigma={args.sigma}")
    if isinstance(train_dataset, IndexPairDataset):
        print(
            "Indexed transform (no inferred geometry): "
            f"{train_dataset.transform_metadata}"
        )
    else:
        eff_deg = args.frame_skip * args.yaw_step_deg
        print(
            f"Frame skip: {args.frame_skip} "
            f"(legacy effective step: {eff_deg:.0f} deg)"
        )
    print(f"Include backward pairs: {include_backward}")
    print("-" * 70)
    
    # ---- First-batch diagnostic (one-time) ----
    with torch.no_grad():
        diag_batch = next(iter(train_loader))
        diag_x_t = diag_batch['x_t'].to(device)
        diag_x_t1 = diag_batch['x_t1'].to(device)
        diag_action_labels = diag_batch['action_label'].to(device).long()
        diag_out = model(diag_x_t, diag_x_t1)
        diag_losses = compute_losses(
            diag_out, lambda_smooth=args.lambda_smooth, lambda_disp=args.lambda_disp,
            lambda_ent=args.lambda_ent, lambda_act=args.lambda_act, lambda_loc=args.lambda_loc,
            lambda_inv=args.lambda_inv, lambda_cycle=args.lambda_cycle,
            sigma=args.sigma, num_keypoints=args.num_keypoints, action_labels=diag_action_labels,
            x_t=diag_x_t, x_t1=diag_x_t1, loc_bg_threshold=args.loc_bg_threshold,
        )
        print(f"\n[Diagnostic] First batch (before training):")
        print(f"  p_t  range: [{diag_out['p_t'].min().item():.4f}, {diag_out['p_t'].max().item():.4f}]")
        print(f"  l_pred:  {diag_losses['l_pred'].item():.6f}")
        print(f"  l_disp:  {diag_losses['l_disp'].item():.6f}")
        print(f"  l_smooth:{diag_losses['l_smooth'].item():.6f}")
        print(f"  l_ent:   {diag_losses['l_ent'].item():.6f}")
        print(f"  l_act:   {diag_losses['l_act'].item():.6f}")
        print(f"  l_loc:   {diag_losses['l_loc'].item():.6f}")
        print(f"  l_inv:   {diag_losses['l_inv'].item():.6f}")
        print(f"  l_cycle: {diag_losses['l_cycle'].item():.6f}")
        print(f"  act_acc: {diag_losses['act_acc'].item():.4f}")
        print(f"  total:   {diag_losses['loss'].item():.6f}")
        print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda_smooth, args.lambda_disp, args.lambda_ent, args.lambda_act,
            args.lambda_loc, args.lambda_inv, args.lambda_cycle,
            args.loc_bg_threshold, args.sigma, args.num_keypoints
        )
        optimizer_step_count += len(train_loader)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if epoch % args.log_every == 0 or epoch == 1:
            log_line = (
                f"Epoch {epoch:4d} | "
                f"Loss: {train_metrics['loss']:.5f} | "
                f"L_pred: {train_metrics['l_pred']:.5f} | "
                f"L_smooth: {train_metrics['l_smooth']:.5f} | "
                f"L_disp: {train_metrics['l_disp']:.5f} | "
                f"L_ent: {train_metrics['l_ent']:.5f} | "
                f"L_act: {train_metrics['l_act']:.5f} | "
                f"L_loc: {train_metrics['l_loc']:.5f} | "
                f"L_inv: {train_metrics['l_inv']:.5f} | "
                f"L_cycle: {train_metrics['l_cycle']:.5f} | "
                f"ActAcc: {train_metrics['act_acc']:.3f}"
            )
            history_entry = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_pred': train_metrics['l_pred'],
                'train_smooth': train_metrics['l_smooth'],
                'train_disp': train_metrics['l_disp'],
                'train_ent': train_metrics['l_ent'],
                'train_act': train_metrics['l_act'],
                'train_loc': train_metrics['l_loc'],
                'train_inv': train_metrics['l_inv'],
                'train_cycle': train_metrics['l_cycle'],
                'train_act_acc': train_metrics['act_acc'],
                'lr': scheduler.get_last_lr()[0],
            }

            if val_loader is not None:
                val_metrics = evaluate(
                    model, val_loader, device,
                    args.lambda_smooth, args.lambda_disp, args.lambda_ent,
                    args.lambda_act, args.lambda_loc, args.lambda_inv,
                    args.lambda_cycle, args.loc_bg_threshold, args.sigma,
                    args.num_keypoints
                )
                log_line += f" | Val: {val_metrics['loss']:.5f}"
                history_entry.update({
                    'val_loss': val_metrics['loss'],
                    'val_pred': val_metrics['l_pred'],
                    'val_act': val_metrics['l_act'],
                    'val_loc': val_metrics['l_loc'],
                    'val_inv': val_metrics['l_inv'],
                    'val_cycle': val_metrics['l_cycle'],
                    'val_act_acc': val_metrics['act_acc'],
                })

                # Development/legacy only: validation selects best_model.pt.
                if val_metrics['loss'] < best_loss:
                    best_loss = val_metrics['loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'config': ckpt_config,
                    }, run_dir / "best_model.pt")

            print(log_line)
            history.append(history_entry)
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'config': ckpt_config,
            }, run_dir / f"checkpoint_{epoch:05d}.pt")
    
    # Save final model. In fixed-final mode this write is the authority boundary:
    # the test Dataset is not constructed until this file exists.
    final_model_path = run_dir / "final_model.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_metrics['loss'],
        'config': ckpt_config,
    }, final_model_path)

    fixed_final_test_report = None
    if data_plan.mode == "fixed-final":
        if not final_model_path.is_file():
            raise RuntimeError(
                "Authoritative final_model.pt is missing; refusing to open test data"
            )
        test_loader = _build_fixed_final_test_loader(
            data_plan,
            args,
            final_model_path=final_model_path,
            use_cuda=use_cuda,
            n_workers=n_workers,
        )
        # This is the only test evaluate() call in the training entry point.
        test_metrics = evaluate(
            model, test_loader, device,
            args.lambda_smooth, args.lambda_disp, args.lambda_ent,
            args.lambda_act, args.lambda_loc, args.lambda_inv,
            args.lambda_cycle, args.loc_bg_threshold, args.sigma,
            args.num_keypoints,
        )
        fixed_final_test_report = {
            "schema_version": 1,
            "training_mode": "fixed-final",
            "evaluation_count": 1,
            "authoritative_checkpoint": str(final_model_path.resolve()),
            "authoritative_checkpoint_sha256": _sha256_file(final_model_path),
            "test_index": data_plan.index_provenance["test"],
            "metrics": test_metrics,
        }
        with open(run_dir / "fixed_final_test_metrics.json", "w") as f:
            json.dump(fixed_final_test_report, f, indent=2)
    
    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if fresh_binding is not None:
        receipt_path = fresh_authorization.write_completed_run_receipt(
            repo_root,
            fresh_binding,
            device=str(device),
            optimizer_step_count=optimizer_step_count,
            determinism=determinism,
            full_command=full_command,
            runtime_environment=runtime_environment,
        )
        print(f"Completed-run receipt: {receipt_path}")
    
    print("-" * 60)
    if val_loader is not None:
        print(f"Training complete! Best validation loss: {best_loss:.6f}")
    else:
        print(
            "Training complete! Fixed-final policy used final_model.pt; "
            f"one post-training test loss: "
            f"{fixed_final_test_report['metrics']['loss']:.6f}"
        )
    print(f"Outputs saved to: {run_dir}")

    if args.auto_eval:
        checkpoint_name = "best_model.pt" if args.auto_eval_checkpoint == "best" else "final_model.pt"
        checkpoint_path = run_dir / checkpoint_name

        if args.eval_frames_dir is not None:
            frames_dir = Path(args.eval_frames_dir)
        else:
            frames_dir = _infer_eval_frames_dir(args.data_root, args.object)

        if frames_dir is None:
            print(
                "[AutoEval] Skipping: could not infer frames directory. "
                "Pass --eval_frames_dir explicitly."
            )
        elif not frames_dir.exists():
            print(f"[AutoEval] Skipping: frames directory does not exist: {frames_dir}")
        else:
            _run_auto_eval(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                frames_dir=frames_dir,
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                eval_max_k=args.eval_max_k,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
