"""Fail-closed binding for the paired one-epoch equivariance GPU smoke.

This module intentionally authorizes only four seed-42 smoke cells.  It does
not authorize a longer pilot or a scientific training array.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "agent/sliding-wobble-eradication-20260814"
AUTHORIZED_BASE_COMMIT = "43dee6362c22928b3ce2d481def491d981754ab1"
MANIFEST_SCHEMA_VERSION = "same_frame_equivariance_gpu_smoke_manifest.v1"
CALIBRATION_SCHEMA_VERSION = "same_frame_equivariance_gradient_calibration.v1"
COMPLETED_RUN_SCHEMA_VERSION = "same_frame_equivariance_gpu_smoke_completed_run.v1"
COMPLETED_RUN_RECEIPT_NAME = "SELF_EQUIVARIANCE_COMPLETED_RUN_RECEIPT.json"
CALIBRATED_COEFFICIENT = 0.5
TRAIN_INDEX_SHA256 = "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
VALIDATION_INDEX_SHA256 = "3e71c4f862a99d1882a8704140f8706612460d804a6ed7a833c8dee9f35514a4"
DATASET_BINDING_SHA256 = "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__(control|self_equivariance)__seed42__smoke$"
)

TASK_RECIPES = {
    "task55_clean": {
        "lambda_smooth": 0.0,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.0,
        "lambda_cycle": 0.0,
        "learn_inverse_operator": False,
    },
    "task80_assisted": {
        "lambda_smooth": 0.001,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.5,
        "lambda_cycle": 0.5,
        "learn_inverse_operator": True,
    },
}

FIXED_ARGUMENTS = {
    "object": "engineers_hammer_vray",
    "pairs_index": None,
    "indexed_mode": "development",
    "test_pairs_index": None,
    "dataset_binding_sha256": DATASET_BINDING_SHA256,
    "img_size": 512,
    "num_keypoints": 10,
    "base_channels": 32,
    "heatmap_res": 64,
    "temperature": 1.0,
    "padding_mode": "reflect",
    "operator_type": "shared_affine",
    "lambda_attach": 0.0,
    "lambda_ocr_zncc": 0.0,
    "ocr_peak_margin_exclusion_radius_cells": 1,
    "sigma": 0.1,
    "loc_bg_threshold": 30.0,
    "num_action_classes": 0,
    "frame_skip": 3,
    "yaw_step_deg": 2.0,
    "center_crop": None,
    "epochs": 1,
    "frozen_epochs": None,
    "lr": 0.0001,
    "weight_decay": 0.00001,
    "seed": 42,
    "save_every": 1,
    "log_every": 1,
    "auto_eval": False,
    "auto_eval_checkpoint": "best",
    "eval_frames_dir": None,
    "eval_max_k": 10,
}

BOUND_ARGUMENTS = frozenset(
    {
        *FIXED_ARGUMENTS,
        *TASK_RECIPES["task55_clean"],
        "data_root",
        "train_pairs_index",
        "val_pairs_index",
        "batch_size",
        "output_dir",
        "lambda_self_equivariance",
    }
)

RUN_SOURCE_PATHS = (
    "keypoint_net/model.py",
    "keypoint_net/dataset.py",
    "keypoint_net/train.py",
    "keypoint_net/fresh_roll_determinism.py",
    "keypoint_net/frozen_nuisance_sensitivity.py",
    "keypoint_net/same_frame_equivariance.py",
    "keypoint_net/same_frame_equivariance_smoke_authorization.py",
    "keypoint_net/run_same_frame_equivariance_preoptimizer_audit.py",
    "keypoint_net/run_same_frame_equivariance_smoke.py",
    "cluster/same_frame_equivariance_smoke.slurm",
)


@dataclass(frozen=True)
class SelfEquivarianceSmokeBinding:
    cell_id: str
    recipe: str
    arm: str
    run_directory: str
    manifest_absolute_path: str
    manifest_file_sha256: str
    manifest_content_sha256: str
    calibration_absolute_path: str
    calibration_file_sha256: str
    expected_training_arguments: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _load_json(path_value: str | Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path_value, name=name)
    try:
        value = json.loads(Path(record["absolute_path"]).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value, record


def _validate_content_hash(document: Mapping[str, Any], *, name: str) -> str:
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        f"{name} content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash differs")
    return claimed


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    _require(_COMMIT_RE.fullmatch(commit) is not None, "source commit is invalid")
    _require(branch == EXPECTED_BRANCH, f"wrong source branch {branch!r}")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                AUTHORIZED_BASE_COMMIT,
                commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("authorized evaluator base is not an ancestor") from exc
    # Deliberately ignore unrelated untracked research artifacts.  Every file
    # that can affect this smoke must be tracked and byte-identical to HEAD.
    for relative in RUN_SOURCE_PATHS:
        _git(repo_root, "ls-files", "--error-unmatch", relative)
        live = (repo_root / relative).read_bytes()
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        _require(live == committed, f"live smoke source differs from HEAD: {relative}")
    return commit, branch


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    return actual == expected


def bind_training_namespace(repo_root: Path, args: Any) -> SelfEquivarianceSmokeBinding:
    commit, branch = _source_state(repo_root)
    identity = _CELL_RE.fullmatch(args.self_equivariance_cell_id or "")
    _require(identity is not None, "self-equivariance smoke cell ID is invalid")
    identity_recipe, identity_arm = identity.groups()

    _require(
        isinstance(args.self_equivariance_experiment_manifest_sha256, str)
        and _SHA256_RE.fullmatch(args.self_equivariance_experiment_manifest_sha256)
        is not None,
        "self-equivariance manifest expected hash is invalid",
    )
    manifest, manifest_record = _load_json(
        args.self_equivariance_experiment_manifest,
        name="self-equivariance smoke manifest",
    )
    _require(
        manifest_record["sha256"] == args.self_equivariance_experiment_manifest_sha256,
        "self-equivariance smoke manifest file hash differs",
    )
    manifest_content_hash = _validate_content_hash(
        manifest, name="self-equivariance smoke manifest"
    )
    _require(
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "self-equivariance smoke manifest schema differs",
    )
    _require(
        manifest.get("source_commit") == commit
        and manifest.get("source_branch") == branch,
        "self-equivariance smoke source lineage differs",
    )

    calibration, calibration_record = _load_json(
        args.self_equivariance_calibration_receipt,
        name="self-equivariance calibration receipt",
    )
    _require(
        calibration_record["sha256"]
        == args.self_equivariance_calibration_receipt_sha256,
        "self-equivariance calibration receipt file hash differs",
    )
    _validate_content_hash(calibration, name="self-equivariance calibration receipt")
    _require(
        calibration.get("schema_version") == CALIBRATION_SCHEMA_VERSION,
        "self-equivariance calibration schema differs",
    )
    calibration_payload = calibration.get("calibration", {})
    _require(
        calibration_payload.get("status") == "calibrated_at_safety_cap"
        and calibration_payload.get("coefficient") == CALIBRATED_COEFFICIENT
        and calibration_payload.get("optimizer_steps") == 0
        and calibration_payload.get("training_or_weight_update_performed") is False,
        "self-equivariance calibration decision differs",
    )
    manifest_calibration = manifest.get("calibration_receipt")
    _require(
        isinstance(manifest_calibration, dict)
        and manifest_calibration.get("sha256") == calibration_record["sha256"],
        "manifest calibration binding differs",
    )

    cells = manifest.get("cells")
    _require(isinstance(cells, list), "self-equivariance smoke cells are invalid")
    matches = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_id") == args.self_equivariance_cell_id
    ]
    _require(len(matches) == 1, "self-equivariance smoke cell is absent or duplicated")
    cell = matches[0]
    expected = cell.get("training_arguments")
    _require(
        isinstance(expected, dict) and set(expected) == BOUND_ARGUMENTS,
        "self-equivariance smoke training argument set differs",
    )
    for name, value in expected.items():
        _require(
            hasattr(args, name) and _same(value, getattr(args, name)),
            f"self-equivariance-bound --{name} differs",
        )
    _require(cell.get("stage") == "one_epoch_gpu_smoke", "smoke stage differs")
    _require(
        cell.get("recipe") == identity_recipe
        and cell.get("arm") == identity_arm
        and cell.get("seed") == 42,
        "self-equivariance cell identity differs",
    )
    for name, value in FIXED_ARGUMENTS.items():
        _require(_same(value, expected.get(name)), f"fixed smoke --{name} differs")
    for name, value in TASK_RECIPES[identity_recipe].items():
        _require(_same(value, expected.get(name)), f"recipe smoke --{name} differs")
    _require(
        _file_record(args.train_pairs_index, name="smoke train index")["sha256"]
        == TRAIN_INDEX_SHA256,
        "smoke train index hash differs",
    )
    _require(
        _file_record(args.val_pairs_index, name="smoke validation index")["sha256"]
        == VALIDATION_INDEX_SHA256,
        "smoke validation index hash differs",
    )
    expected_weight = 0.0 if identity_arm == "control" else CALIBRATED_COEFFICIENT
    _require(
        float(args.lambda_self_equivariance) == expected_weight,
        "smoke arm and self-equivariance coefficient disagree",
    )
    _require(args.lambda_attach == 0.0, "smoke cannot use descriptor attachment")
    _require(args.lambda_ocr_zncc == 0.0, "smoke cannot use OCR-ZNCC")
    _require(
        isinstance(args.batch_size, int) and 1 <= args.batch_size <= 16,
        "smoke batch size must be within [1, 16]",
    )

    run_directory = (
        Path(expected["output_dir"]).expanduser().absolute()
        / args.self_equivariance_cell_id
    )
    _require(
        not run_directory.exists(),
        f"self-equivariance smoke run directory already exists: {run_directory}",
    )
    return SelfEquivarianceSmokeBinding(
        cell_id=args.self_equivariance_cell_id,
        recipe=identity_recipe,
        arm=identity_arm,
        run_directory=str(run_directory),
        manifest_absolute_path=manifest_record["absolute_path"],
        manifest_file_sha256=manifest_record["sha256"],
        manifest_content_sha256=manifest_content_hash,
        calibration_absolute_path=calibration_record["absolute_path"],
        calibration_file_sha256=calibration_record["sha256"],
        expected_training_arguments=dict(expected),
    )


def reject_unbound_arguments(args: Any) -> None:
    supplied = (
        args.self_equivariance_experiment_manifest,
        args.self_equivariance_experiment_manifest_sha256,
        args.self_equivariance_calibration_receipt,
        args.self_equivariance_calibration_receipt_sha256,
    )
    _require(
        not any(item is not None for item in supplied),
        "self-equivariance binding files require --self_equivariance_cell_id",
    )
    _require(
        float(args.lambda_self_equivariance) == 0.0,
        "positive self-equivariance loss requires a bound smoke cell",
    )


def write_completed_run_receipt(
    repo_root: Path,
    binding: SelfEquivarianceSmokeBinding,
    *,
    device: str,
    optimizer_step_count: int,
    full_command: Sequence[str],
    runtime_environment: Mapping[str, Any],
    determinism: Mapping[str, Any],
    nondeterminism_evidence: Mapping[str, Any],
) -> Path:
    commit, branch = _source_state(repo_root)
    run_directory = Path(binding.run_directory).resolve(strict=True)
    history_path = run_directory / "history.json"
    config_path = run_directory / "config.json"
    final_path = run_directory / "final_model.pt"
    best_path = run_directory / "best_model.pt"
    sampler_path = run_directory / "sampler_order.json"
    for path in (history_path, config_path, final_path, best_path, sampler_path):
        _require(path.is_file(), f"completed smoke artifact is missing: {path.name}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    _require(
        isinstance(history, list)
        and len(history) == 1
        and history[0].get("epoch") == 1,
        "completed smoke history differs",
    )
    finite_fields = (
        "train_loss",
        "train_base_loss",
        "train_self_equivariance_loss",
        "val_loss",
        "val_base_loss",
        "val_self_equivariance_loss",
    )
    _require(
        all(
            isinstance(history[0].get(name), (int, float))
            and not isinstance(history[0].get(name), bool)
            and math.isfinite(float(history[0][name]))
            for name in finite_fields
        ),
        "completed smoke history contains a non-finite metric",
    )
    _require(optimizer_step_count > 0, "completed smoke has no optimizer step")
    receipt = {
        "schema_version": COMPLETED_RUN_SCHEMA_VERSION,
        "status": "completed_one_epoch_gpu_smoke_not_scientific_outcome",
        "cell_id": binding.cell_id,
        "recipe": binding.recipe,
        "arm": binding.arm,
        "source_commit": commit,
        "source_branch": branch,
        "manifest_file_sha256": binding.manifest_file_sha256,
        "calibration_file_sha256": binding.calibration_file_sha256,
        "device": device,
        "optimizer_step_count": optimizer_step_count,
        "full_command": list(full_command),
        "runtime_environment": dict(runtime_environment),
        "determinism": dict(determinism),
        "nondeterminism_evidence": dict(nondeterminism_evidence),
        "artifacts": {
            path.name: _file_record(path, name=path.name)
            for path in (history_path, config_path, final_path, best_path, sampler_path)
        },
    }
    receipt["content_hash_sha256"] = canonical_sha256(receipt)
    receipt_path = run_directory / COMPLETED_RUN_RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt_path


__all__ = [
    "BOUND_ARGUMENTS",
    "CALIBRATED_COEFFICIENT",
    "COMPLETED_RUN_RECEIPT_NAME",
    "FIXED_ARGUMENTS",
    "MANIFEST_SCHEMA_VERSION",
    "SelfEquivarianceSmokeBinding",
    "TASK_RECIPES",
    "bind_training_namespace",
    "canonical_sha256",
    "reject_unbound_arguments",
    "write_completed_run_receipt",
]
