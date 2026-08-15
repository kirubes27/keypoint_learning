"""Prepare or execute the paired seed-42 one-epoch GPU smoke.

Preparation writes an external manifest after the source commit exists.
Execution runs control and candidate sequentially on the same allocated GPU.
This module never submits a Slurm job and never authorizes a longer pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from keypoint_net import (
    same_frame_equivariance_smoke_authorization as authorization,
)


RECIPES = ("task55_clean", "task80_assisted")
ARMS = ("control", "self_equivariance")
OBJECT_NAME = "engineers_hammer_vray"
PAIR_RECEIPT_SCHEMA_VERSION = "same_frame_equivariance_smoke_pair_receipt.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path_value: str | Path) -> dict[str, Any]:
    return authorization._file_record(  # noqa: SLF001
        path_value, name="bound self-equivariance smoke file"
    )


def _arguments(
    *,
    recipe: str,
    arm: str,
    data_root: Path,
    train_index: Path,
    validation_index: Path,
    output_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    _require(recipe in RECIPES, "unknown smoke recipe")
    _require(arm in ARMS, "unknown smoke arm")
    values = {
        "data_root": str(data_root),
        "object": OBJECT_NAME,
        "pairs_index": None,
        "indexed_mode": "development",
        "train_pairs_index": str(train_index),
        "val_pairs_index": str(validation_index),
        "test_pairs_index": None,
        "dataset_binding_sha256": authorization.DATASET_BINDING_SHA256,
        "img_size": 512,
        "num_keypoints": 10,
        "base_channels": 32,
        "heatmap_res": 64,
        "temperature": 1.0,
        "padding_mode": "reflect",
        "operator_type": "shared_affine",
        "lambda_attach": 0.0,
        "lambda_ocr_zncc": 0.0,
        "lambda_self_equivariance": (
            0.0 if arm == "control" else authorization.CALIBRATED_COEFFICIENT
        ),
        "ocr_peak_margin_exclusion_radius_cells": 1,
        "sigma": 0.1,
        "loc_bg_threshold": 30.0,
        "num_action_classes": 0,
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "center_crop": None,
        "epochs": 1,
        "frozen_epochs": None,
        "batch_size": batch_size,
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "seed": 42,
        "output_dir": str(output_root),
        "save_every": 1,
        "log_every": 1,
        "auto_eval": False,
        "auto_eval_checkpoint": "best",
        "eval_frames_dir": None,
        "eval_max_k": 10,
        **authorization.TASK_RECIPES[recipe],
    }
    _require(
        set(values) == authorization.BOUND_ARGUMENTS,
        "constructed smoke training argument set differs",
    )
    return values


def prepare(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    commit, branch = authorization._source_state(repo_root)  # noqa: SLF001
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    train_index = Path(args.train_index).expanduser().resolve(strict=True)
    validation_index = Path(args.validation_index).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().absolute()
    manifest_path = Path(args.manifest_path).expanduser().absolute()
    calibration_record = _record(args.calibration_receipt)
    audit_record = _record(args.preoptimizer_audit)
    _require(_sha256(train_index) == authorization.TRAIN_INDEX_SHA256, "train index differs")
    _require(
        _sha256(validation_index) == authorization.VALIDATION_INDEX_SHA256,
        "validation index differs",
    )
    _require(1 <= args.batch_size <= 16, "batch size must be within [1, 16]")
    _require(not manifest_path.exists(), f"manifest already exists: {manifest_path}")
    _require(not output_root.exists(), f"smoke output root already exists: {output_root}")

    calibration = json.loads(
        Path(calibration_record["absolute_path"]).read_text(encoding="utf-8")
    )
    audit = json.loads(Path(audit_record["absolute_path"]).read_text(encoding="utf-8"))
    _require(
        calibration.get("schema_version") == authorization.CALIBRATION_SCHEMA_VERSION
        and calibration.get("calibration", {}).get("coefficient")
        == authorization.CALIBRATED_COEFFICIENT,
        "calibration receipt does not bind coefficient 0.5",
    )
    _require(
        audit.get("schema_version") == "same_frame_equivariance_preoptimizer_audit.v1"
        and audit.get("status") == "pass"
        and audit.get("semantic_lock", {}).get("optimizer_steps") == 0,
        "preoptimizer audit has not passed",
    )
    cells = []
    for recipe in RECIPES:
        for arm in ARMS:
            cell_id = f"{recipe}__{arm}__seed42__smoke"
            cells.append(
                {
                    "cell_id": cell_id,
                    "recipe": recipe,
                    "arm": arm,
                    "seed": 42,
                    "stage": "one_epoch_gpu_smoke",
                    "training_arguments": _arguments(
                        recipe=recipe,
                        arm=arm,
                        data_root=data_root,
                        train_index=train_index,
                        validation_index=validation_index,
                        output_root=output_root,
                        batch_size=args.batch_size,
                    ),
                }
            )
    manifest = {
        "schema_version": authorization.MANIFEST_SCHEMA_VERSION,
        "artifact_type": "paired_same_frame_equivariance_one_epoch_gpu_smoke",
        "status": "prepared_not_executed",
        "authorization_scope": {
            "only_seed": 42,
            "only_epoch_count": 1,
            "recipes": list(RECIPES),
            "arms": list(ARMS),
            "longer_pilot_authorized": False,
            "scientific_array_authorized": False,
        },
        "source_commit": commit,
        "source_branch": branch,
        "source_files": {
            relative: _record(repo_root / relative)
            for relative in authorization.RUN_SOURCE_PATHS
        },
        "dataset": {
            "root": str(data_root),
            "binding_sha256": authorization.DATASET_BINDING_SHA256,
            "train_index": _record(train_index),
            "validation_index": _record(validation_index),
            "object": OBJECT_NAME,
        },
        "calibration_receipt": calibration_record,
        "preoptimizer_audit": audit_record,
        "coefficient": authorization.CALIBRATED_COEFFICIENT,
        "fairness_lock": {
            "common_auxiliary_forward_executed_in_both_arms": True,
            "common_auxiliary_forward_image_count_per_pair_batch": "6B",
            "only_arm_difference": "lambda_self_equivariance 0.0 versus 0.5",
            "control_runs_before_candidate_on_same_allocation": True,
        },
        "cells": cells,
    }
    manifest["content_hash_sha256"] = authorization.canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": "prepared_not_executed",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "content_hash_sha256": manifest["content_hash_sha256"],
        "cells": len(cells),
    }, indent=2))
    return manifest_path


def _training_command(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    calibration_path: Path,
    calibration_sha256: str,
    cell: Mapping[str, Any],
) -> list[str]:
    arguments = dict(cell["training_arguments"])
    command = [
        sys.executable,
        str(repo_root / "keypoint_net/train.py"),
        "--self_equivariance_cell_id",
        str(cell["cell_id"]),
        "--self_equivariance_experiment_manifest",
        str(manifest_path),
        "--self_equivariance_experiment_manifest_sha256",
        manifest_sha256,
        "--self_equivariance_calibration_receipt",
        str(calibration_path),
        "--self_equivariance_calibration_receipt_sha256",
        calibration_sha256,
    ]
    boolean_flags = {"learn_inverse_operator", "auto_eval"}
    skip_none = {
        "pairs_index",
        "test_pairs_index",
        "frozen_epochs",
        "center_crop",
        "eval_frames_dir",
    }
    for name, value in arguments.items():
        if name in boolean_flags:
            if value:
                command.append(f"--{name}")
            continue
        if name in skip_none and value is None:
            continue
        command.extend((f"--{name}", str(value)))
    return command


def _load_bound_manifest(
    repo_root: Path, manifest_path: Path, expected_hash: str
) -> tuple[dict[str, Any], str]:
    record = _record(manifest_path)
    _require(record["sha256"] == expected_hash, "smoke manifest file hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.get("content_hash_sha256")
    payload = dict(manifest)
    payload.pop("content_hash_sha256", None)
    _require(
        authorization.canonical_sha256(payload) == claimed,
        "smoke manifest content hash differs",
    )
    commit, branch = authorization._source_state(repo_root)  # noqa: SLF001
    _require(
        manifest.get("source_commit") == commit
        and manifest.get("source_branch") == branch,
        "smoke manifest source differs",
    )
    return manifest, record["sha256"]


def _paired_receipt(
    *,
    recipe: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> Path:
    cells = {
        cell["arm"]: cell
        for cell in manifest["cells"]
        if cell["recipe"] == recipe
    }
    _require(set(cells) == set(ARMS), "paired smoke cells differ")
    run_root = Path(cells["control"]["training_arguments"]["output_dir"])
    loaded: dict[str, dict[str, Any]] = {}
    for arm, cell in cells.items():
        run_dir = run_root / cell["cell_id"]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        sampler = json.loads((run_dir / "sampler_order.json").read_text(encoding="utf-8"))
        receipt_path = run_dir / authorization.COMPLETED_RUN_RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        loaded[arm] = {
            "run_directory": str(run_dir),
            "config": config,
            "sampler": sampler,
            "completed_receipt": receipt,
            "completed_receipt_record": _record(receipt_path),
        }
    control = loaded["control"]
    candidate = loaded["self_equivariance"]
    checks = {
        "initial_model_state_identical": (
            control["config"]["initial_model_state_sha256"]
            == candidate["config"]["initial_model_state_sha256"]
        ),
        "post_diagnostic_model_state_identical": (
            control["config"]["post_diagnostic_model_state_sha256"]
            == candidate["config"]["post_diagnostic_model_state_sha256"]
        ),
        "diagnostic_pair_order_identical": (
            control["config"]["diagnostic_pair_order_sha256"]
            == candidate["config"]["diagnostic_pair_order_sha256"]
        ),
        "epoch_sampler_order_identical": (
            control["sampler"]["epochs"] == candidate["sampler"]["epochs"]
        ),
        "same_slurm_allocation": (
            control["completed_receipt"]["runtime_environment"]["slurm_job_id"]
            == candidate["completed_receipt"]["runtime_environment"]["slurm_job_id"]
        ),
        "same_visible_gpu": (
            control["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["cuda_visible_devices"]
            == candidate["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["cuda_visible_devices"]
        ),
        "same_pair_wrapper": (
            control["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["wrapper_pid"]
            == candidate["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["wrapper_pid"]
        ),
        "control_precedes_candidate": (
            control["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["position"]
            == "1_control"
            and candidate["completed_receipt"]["runtime_environment"]
            ["self_equivariance_pair_execution"]["position"]
            == "2_self_equivariance"
        ),
    }
    _require(all(checks.values()), f"paired smoke receipt checks failed: {checks}")
    document = {
        "schema_version": PAIR_RECEIPT_SCHEMA_VERSION,
        "status": "completed_paired_one_epoch_smoke_not_scientific_outcome",
        "recipe": recipe,
        "manifest": _record(manifest_path),
        "manifest_expected_sha256": manifest_sha256,
        "checks": checks,
        "cells": loaded,
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    destination = run_root / f"{recipe}__PAIRED_SMOKE_RECEIPT.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def execute(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    manifest_path = Path(args.manifest_path).expanduser().resolve(strict=True)
    manifest, manifest_sha256 = _load_bound_manifest(
        repo_root, manifest_path, args.manifest_sha256
    )
    _require(args.recipe in RECIPES, "unknown smoke recipe")
    cells = [
        cell for cell in manifest["cells"] if cell.get("recipe") == args.recipe
    ]
    cells_by_arm = {cell["arm"]: cell for cell in cells}
    _require(set(cells_by_arm) == set(ARMS), "recipe smoke pair differs")
    calibration_path = Path(
        manifest["calibration_receipt"]["absolute_path"]
    ).resolve(strict=True)
    calibration_sha256 = manifest["calibration_receipt"]["sha256"]

    pair_id = f"{args.recipe}__seed42__same_allocation"
    wrapper_pid = str(os.getpid())
    for position, arm in (
        ("1_control", "control"),
        ("2_self_equivariance", "self_equivariance"),
    ):
        environment = dict(os.environ)
        environment.update(
            {
                "SELF_EQUIVARIANCE_PAIR_ID": pair_id,
                "SELF_EQUIVARIANCE_PAIR_POSITION": position,
                "SELF_EQUIVARIANCE_PAIR_WRAPPER_PID": wrapper_pid,
            }
        )
        command = _training_command(
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            calibration_path=calibration_path,
            calibration_sha256=calibration_sha256,
            cell=cells_by_arm[arm],
        )
        subprocess.run(command, check=True, cwd=repo_root, env=environment)
    receipt = _paired_receipt(
        recipe=args.recipe,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    print(json.dumps({
        "status": "completed_paired_one_epoch_smoke_not_scientific_outcome",
        "recipe": args.recipe,
        "paired_receipt": str(receipt),
        "paired_receipt_sha256": _sha256(receipt),
    }, indent=2))
    return receipt


def write_environment_lock(args: argparse.Namespace) -> Path:
    destination = Path(args.output_path).expanduser().absolute()
    _require(not destination.exists(), f"environment lock already exists: {destination}")
    document = {
        "schema_version": "same_frame_equivariance_cluster_environment.v1",
        "artifact_type": "live_gpu_smoke_environment_lock",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": str(np.__version__),
        "pytorch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _require(document["cuda_available"] is True, "environment lock requires CUDA")
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "environment_lock": str(destination),
        "sha256": _sha256(destination),
    }, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", required=True)
    prepare_parser.add_argument("--data-root", required=True)
    prepare_parser.add_argument("--train-index", required=True)
    prepare_parser.add_argument("--validation-index", required=True)
    prepare_parser.add_argument("--calibration-receipt", required=True)
    prepare_parser.add_argument("--preoptimizer-audit", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--manifest-path", required=True)
    prepare_parser.add_argument("--batch-size", type=int, default=4)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--repo-root", required=True)
    execute_parser.add_argument("--manifest-path", required=True)
    execute_parser.add_argument("--manifest-sha256", required=True)
    execute_parser.add_argument("--recipe", required=True, choices=RECIPES)

    environment_parser = subparsers.add_parser("write-environment-lock")
    environment_parser.add_argument("--output-path", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "execute":
        execute(args)
    else:
        write_environment_lock(args)


if __name__ == "__main__":
    main()
