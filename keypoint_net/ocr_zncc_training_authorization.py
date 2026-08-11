"""Fail-closed binding for the paired OCR-ZNCC training experiment.

The experiment manifest is generated only after the source is committed, so it
can bind the exact commit, branch, evidence artifacts, scientific arguments,
and one exclusive output directory per cell without a circular Git hash.
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

import torch


EXPECTED_BRANCH = "agent/ocr-zncc-training-20260811"
AUTHORIZED_BASE_COMMIT = "4521538621b410421b1058603d090e2b09ec4178"
EXPERIMENT_SCHEMA_VERSION = "ocr_zncc_paired_training_manifest.v1"
COMPLETED_RUN_SCHEMA_VERSION = "ocr_zncc_completed_run.v1"
COMPLETED_RUN_RECEIPT_NAME = "OCR_ZNCC_COMPLETED_RUN_RECEIPT.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__(control|ocr_zncc)__seed(42|43|44)(__smoke)?$"
)
TASK_RECIPES = {
    "task55_clean": {
        "lambda_smooth": 0.0, "lambda_disp": 0.1, "lambda_ent": 0.01,
        "lambda_act": 0.0, "lambda_loc": 0.0, "lambda_inv": 0.0,
        "lambda_cycle": 0.0, "learn_inverse_operator": False,
    },
    "task80_assisted": {
        "lambda_smooth": 0.001, "lambda_disp": 0.1, "lambda_ent": 0.01,
        "lambda_act": 0.0, "lambda_loc": 0.0, "lambda_inv": 0.5,
        "lambda_cycle": 0.5, "learn_inverse_operator": True,
    },
}
FIXED_TRAINING_ARGUMENTS = {
    "object": "engineers_hammer_vray",
    "pairs_index": None,
    "indexed_mode": "development",
    "test_pairs_index": None,
    "dataset_binding_sha256": "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93",
    "img_size": 512,
    "num_keypoints": 10,
    "base_channels": 32,
    "heatmap_res": 64,
    "temperature": 1.0,
    "padding_mode": "reflect",
    "operator_type": "shared_affine",
    "lambda_attach": 0.0,
    "ocr_peak_margin_exclusion_radius_cells": 1,
    "sigma": 0.1,
    "loc_bg_threshold": 30.0,
    "num_action_classes": 0,
    "frame_skip": 3,
    "yaw_step_deg": 2.0,
    "center_crop": None,
    "epochs": 1000,
    "frozen_epochs": None,
    "batch_size": 16,
    "lr": 0.0001,
    "weight_decay": 0.00001,
    "save_every": 100,
    "log_every": 10,
    "auto_eval": False,
    "auto_eval_checkpoint": "best",
    "eval_frames_dir": None,
    "eval_max_k": 10,
}
TRAIN_INDEX_SHA256 = "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
VALIDATION_INDEX_SHA256 = "3e71c4f862a99d1882a8704140f8706612460d804a6ed7a833c8dee9f35514a4"

BOUND_ARGUMENTS = frozenset(
    {
        "data_root", "object", "pairs_index", "indexed_mode",
        "train_pairs_index", "val_pairs_index", "test_pairs_index",
        "dataset_binding_sha256", "img_size", "num_keypoints",
        "base_channels", "heatmap_res", "temperature", "padding_mode",
        "operator_type", "learn_inverse_operator", "lambda_smooth",
        "lambda_disp", "lambda_ent", "lambda_act", "lambda_loc",
        "lambda_inv", "lambda_cycle", "lambda_attach", "lambda_ocr_zncc",
        "ocr_peak_margin_exclusion_radius_cells", "sigma",
        "loc_bg_threshold", "num_action_classes", "frame_skip",
        "yaw_step_deg", "center_crop", "epochs", "frozen_epochs",
        "batch_size", "lr", "weight_decay", "seed", "output_dir",
        "save_every", "log_every", "auto_eval", "auto_eval_checkpoint",
        "eval_frames_dir", "eval_max_k",
    }
)

RUN_SOURCE_PATHS = (
    "keypoint_net/ocr_zncc_transport.py",
    "keypoint_net/ocr_zncc_training_authorization.py",
    "keypoint_net/ocr_zncc_training_decision.py",
    "keypoint_net/ocr_zncc_fable_review.py",
    "keypoint_net/ocr_zncc_task80_training_authorization.py",
    "keypoint_net/run_ocr_zncc_stage0.py",
    "keypoint_net/run_ocr_zncc_exact_gradient_direction_audit.py",
    "keypoint_net/run_ocr_zncc_minibatch_gradient_audit.py",
    "keypoint_net/ocr_zncc_outcome_evaluator.py",
    "keypoint_net/ocr_zncc_task55_outcome_decision.py",
    "keypoint_net/ocr_zncc_launch_validation.py",
    "keypoint_net/model.py",
    "keypoint_net/train.py",
    "keypoint_net/dataset.py",
    "keypoint_net/fresh_roll_determinism.py",
    "keypoint_net/representation_corpus_inventory.py",
    "keypoint_net/run_ocr_zncc_paired_training.py",
    "keypoint_net/diagnostics/run_ocr_zncc_gpu_smoke.py",
    "cluster/ocr_zncc_gpu_smoke.slurm",
    "cluster/ocr_zncc_paired_stage.slurm",
)


@dataclass(frozen=True)
class OCRTrainingBinding:
    cell_id: str
    recipe: str
    arm: str
    run_directory: str
    manifest_absolute_path: str
    manifest_file_sha256: str
    manifest_content_sha256: str
    expected_training_arguments: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
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
            "absolute_path": str(path), "sha256": digest.hexdigest(),
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
    _require(isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
             f"{name} content hash is invalid")
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash differs")
    return claimed


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _source_state(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    _require(_COMMIT_RE.fullmatch(commit) is not None, "source commit is invalid")
    _require(branch == EXPECTED_BRANCH, f"wrong source branch {branch!r}")
    _require(not _git(repo_root, "status", "--porcelain"),
             "source checkout is not completely clean")
    try:
        subprocess.run(
            [
                "git", "-C", str(repo_root), "merge-base", "--is-ancestor",
                AUTHORIZED_BASE_COMMIT, commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("authorized clean research base is not an ancestor") from exc
    for relative in RUN_SOURCE_PATHS:
        _git(repo_root, "ls-files", "--error-unmatch", relative)
        live = (repo_root / relative).read_bytes()
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            check=True, capture_output=True,
        ).stdout
        _require(live == committed, f"live source differs from HEAD: {relative}")
    return commit, branch


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) \
            and math.isfinite(float(actual)) and float(actual) == expected
    return actual == expected


def bind_training_namespace(repo_root: Path, args: Any) -> OCRTrainingBinding:
    commit, branch = _source_state(repo_root)
    identity = _CELL_RE.fullmatch(args.ocr_cell_id or "")
    _require(identity is not None, "OCR cell ID is invalid")
    identity_recipe, identity_arm, identity_seed, smoke_suffix = identity.groups()
    is_smoke = smoke_suffix == "__smoke"
    _require(isinstance(args.ocr_experiment_manifest_sha256, str)
             and _SHA256_RE.fullmatch(args.ocr_experiment_manifest_sha256) is not None,
             "OCR manifest expected hash is invalid")
    manifest, record = _load_json(args.ocr_experiment_manifest, name="OCR experiment manifest")
    _require(record["sha256"] == args.ocr_experiment_manifest_sha256,
             "OCR experiment manifest file hash differs")
    content_hash = _validate_content_hash(manifest, name="OCR experiment manifest")
    _require(manifest.get("schema_version") == EXPERIMENT_SCHEMA_VERSION,
             "OCR experiment manifest schema differs")
    _require(manifest.get("source_commit") == commit and manifest.get("source_branch") == branch,
             "OCR manifest source lineage differs")
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "OCR manifest cells are invalid")
    matches = [cell for cell in cells if isinstance(cell, dict)
               and cell.get("cell_id") == args.ocr_cell_id]
    _require(len(matches) == 1, "OCR cell is absent or duplicated")
    cell = matches[0]
    expected = cell.get("training_arguments")
    _require(isinstance(expected, dict) and set(expected) == BOUND_ARGUMENTS,
             "OCR cell training argument set differs")
    for name, value in expected.items():
        _require(hasattr(args, name) and _same(value, getattr(args, name)),
                 f"OCR-bound --{name} differs")
    arm = cell.get("arm")
    recipe = cell.get("recipe")
    _require(arm in {"control", "ocr_zncc"}, "OCR cell arm differs")
    _require(recipe in {"task55_clean", "task80_assisted"}, "OCR recipe differs")
    _require(recipe == identity_recipe and arm == identity_arm,
             "OCR cell ID recipe/arm differs from payload")
    _require(int(identity_seed) == int(args.seed) == cell.get("seed"),
             "OCR cell ID seed differs from payload")
    for name, value in TASK_RECIPES[recipe].items():
        _require(_same(value, expected.get(name)), f"OCR {recipe} --{name} differs")
    fixed_contract = dict(FIXED_TRAINING_ARGUMENTS)
    if is_smoke:
        fixed_contract.update({"epochs": 1, "save_every": 1, "log_every": 1})
        _require(cell.get("stage") == "gpu_execution_smoke",
                 "OCR smoke cell stage differs")
        _require(int(identity_seed) == 42,
                 "OCR smoke is limited to seed 42 of its authorized recipe")
    else:
        _require(cell.get("stage") in {"task55_primary", "task80_conditional"},
                 "OCR scientific cell stage differs")
    for name, value in fixed_contract.items():
        _require(_same(value, expected.get(name)), f"OCR fixed --{name} differs")
    _require(_file_record(args.train_pairs_index, name="OCR train index")["sha256"]
             == TRAIN_INDEX_SHA256, "OCR train index hash differs")
    _require(_file_record(args.val_pairs_index, name="OCR validation index")["sha256"]
             == VALIDATION_INDEX_SHA256, "OCR validation index hash differs")
    _require((float(args.lambda_ocr_zncc) == 0.0) == (arm == "control"),
             "OCR arm and coefficient disagree")
    _require(args.lambda_attach == 0.0, "OCR experiment cannot use descriptor attachment")
    _require(args.ocr_peak_margin_exclusion_radius_cells == 1,
             "OCR experiment requires distinct-competitor radius one")
    decision_record = manifest.get("authorization_decision")
    _require(isinstance(decision_record, dict), "OCR authorization decision is missing")
    decision, live_decision_record = _load_json(
        decision_record.get("absolute_path", ""), name="OCR authorization decision"
    )
    _require(live_decision_record["sha256"] == decision_record.get("sha256"),
             "OCR authorization decision file hash differs")
    _validate_content_hash(decision, name="OCR authorization decision")
    decision_contract = {
        "task55_clean": (
            "ocr_zncc_task55_training_authorization.v1",
            "authorize_task55_matched_experiment",
        ),
        "task80_assisted": (
            "ocr_zncc_task80_training_authorization.v1",
            "authorize_task80_matched_experiment",
        ),
    }[recipe]
    _require(decision.get("schema_version") == decision_contract[0],
             "OCR authorization decision schema differs")
    _require(decision.get("status") == decision_contract[1],
             f"OCR authorization decision does not authorize {recipe}")
    _require(decision.get("source_commit") == commit
             and decision.get("source_branch") == branch,
             "OCR authorization decision source differs")
    if recipe == "task55_clean":
        from keypoint_net import ocr_zncc_training_decision as decision_validator
    else:
        from keypoint_net import (
            ocr_zncc_task80_training_authorization as decision_validator,
        )
    deep_validation = decision_validator.validate_authorization_document(
        repo_root, decision
    )
    _require(
        isinstance(deep_validation, Mapping)
        and deep_validation.get("authorization_valid") is True
        and deep_validation.get("status") == decision_contract[1]
        and deep_validation.get("source_commit") == commit
        and deep_validation.get("source_branch") == branch,
        f"OCR authorization decision failed deep validation for {recipe}",
    )
    coefficient = decision.get("coefficient")
    _require(isinstance(coefficient, (int, float)) and not isinstance(coefficient, bool)
             and math.isfinite(float(coefficient)) and 0.0 < float(coefficient) <= 0.5,
             "OCR authorization coefficient is invalid")
    _require(float(manifest.get("ocr_zncc", {}).get("coefficient")) == float(coefficient),
             "OCR manifest coefficient differs from authorization")
    _require(
        _same(float(coefficient), deep_validation.get("coefficient")),
        "OCR authorization coefficient differs from deep validation",
    )
    _require(float(args.lambda_ocr_zncc) == (0.0 if arm == "control" else float(coefficient)),
             "OCR arm coefficient differs from authorization")
    _require(decision.get("ocr_zncc_config", {}).get(
        "peak_margin_exclusion_radius_cells") == 1,
        "OCR authorization matcher configuration differs")
    run_directory = Path(expected["output_dir"]).expanduser().absolute() / args.ocr_cell_id
    _require(not run_directory.exists(), f"OCR run directory already exists: {run_directory}")
    return OCRTrainingBinding(
        cell_id=args.ocr_cell_id,
        recipe=recipe,
        arm=arm,
        run_directory=str(run_directory),
        manifest_absolute_path=record["absolute_path"],
        manifest_file_sha256=record["sha256"],
        manifest_content_sha256=content_hash,
        expected_training_arguments=dict(expected),
    )


def reject_unbound_ocr_arguments(args: Any) -> None:
    supplied = [
        args.ocr_experiment_manifest,
        args.ocr_experiment_manifest_sha256,
    ]
    _require(not any(item is not None for item in supplied),
             "OCR manifest arguments require --ocr_cell_id")
    _require(float(args.lambda_ocr_zncc) == 0.0,
             "positive OCR loss requires --ocr_cell_id")


def _validate_completed_validation_history(
    history: Any,
    binding: OCRTrainingBinding,
) -> list[Mapping[str, Any]]:
    """Validate every retained validation row used by min-base selection."""

    _require(isinstance(history, list) and history, "completed OCR history is empty")
    validation_rows = [row for row in history if isinstance(row, Mapping)
                       and "val_base_loss" in row]
    expected_epochs = (
        [1]
        if int(binding.expected_training_arguments["epochs"]) == 1
        else [1, *range(
            int(binding.expected_training_arguments["log_every"]),
            int(binding.expected_training_arguments["epochs"]) + 1,
            int(binding.expected_training_arguments["log_every"]),
        )]
    )
    _require(
        len(validation_rows) == len(history)
        and [row.get("epoch") for row in validation_rows] == expected_epochs
        and len({row.get("epoch") for row in validation_rows}) == len(expected_epochs),
        "completed OCR validation epoch sequence differs",
    )
    expected_aggregation = {
        "sample_count": 21,
        "batch_count": 2,
        "base_and_standard_metrics": "sample_weighted_complete_loader_mean",
        "ocr_zncc_loss": "accepted_match_weighted_complete_loader_mean",
    }
    finite_fields = (
        "val_loss", "val_train_loss", "val_base_loss",
        "val_attachment_loss", "val_ocr_zncc_loss", "val_pred", "val_act",
        "val_loc", "val_inv", "val_cycle", "val_act_acc",
    )
    for row in validation_rows:
        _require(
            row.get("val_aggregation") == expected_aggregation,
            "completed OCR validation aggregation differs",
        )
        _require(
            all(
                isinstance(row.get(name), (int, float))
                and not isinstance(row.get(name), bool)
                and math.isfinite(float(row[name]))
                for name in finite_fields
            ),
            "completed OCR validation row is non-finite",
        )
        accepted = row.get("val_ocr_zncc_accepted_match_count")
        possible = row.get("val_ocr_zncc_possible_match_count")
        zero_batches = row.get("val_ocr_zncc_zero_accept_batch_count")
        _require(
            isinstance(accepted, int) and not isinstance(accepted, bool)
            and isinstance(possible, int) and not isinstance(possible, bool)
            and isinstance(zero_batches, int) and not isinstance(zero_batches, bool),
            "completed OCR validation matcher counts are invalid",
        )
        if binding.arm == "control":
            _require(
                accepted == 0 and possible == 0 and zero_batches == 0
                and row.get("val_ocr_zncc_accepted_match_coverage") is None,
                "completed control validation OCR metrics differ",
            )
        else:
            coverage = row.get("val_ocr_zncc_accepted_match_coverage")
            _require(
                possible == 210 and 0 <= accepted <= possible
                and 0 <= zero_batches <= 2
                and isinstance(coverage, (int, float))
                and not isinstance(coverage, bool)
                and math.isfinite(float(coverage))
                and math.isclose(
                    float(coverage), accepted / possible,
                    rel_tol=1e-12, abs_tol=1e-12,
                ),
                "completed OCR validation coverage differs",
            )
    return validation_rows


def write_completed_run_receipt(
    repo_root: Path,
    binding: OCRTrainingBinding,
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
    records = {
        name: _file_record(run_directory / name, name=name)
        for name in (
            "config.json", "history.json", "sampler_order.json",
            "best_model.pt", "final_model.pt",
        )
    }
    config = json.loads((run_directory / "config.json").read_bytes())
    history = json.loads((run_directory / "history.json").read_bytes())
    sampler_order = json.loads((run_directory / "sampler_order.json").read_bytes())
    _require(isinstance(config, dict) and config.get("cell_id") == binding.cell_id,
             "completed OCR config identity differs")
    _require(config.get("source_commit") == commit,
             "completed OCR config source differs")
    _require(config.get("checkpoint_policy", {}).get("selection")
             == "minimum_base_validation_loss",
             "completed OCR checkpoint selector differs")
    validation_rows = _validate_completed_validation_history(history, binding)
    first_minimum = min(
        validation_rows,
        key=lambda row: (float(row["val_base_loss"]), int(row["epoch"])),
    )
    best = torch.load(run_directory / "best_model.pt", map_location="cpu",
                      weights_only=False)
    final = torch.load(run_directory / "final_model.pt", map_location="cpu",
                       weights_only=False)
    _require(isinstance(best, dict) and isinstance(final, dict),
             "completed OCR checkpoint payload is invalid")
    _require(best.get("selection_loss_name") == "base_validation_loss",
             "best OCR checkpoint selector label differs")
    _require(int(best.get("epoch", -1)) == int(first_minimum["epoch"]),
             "best OCR checkpoint is not the first minimum base-validation epoch")
    _require(float(best.get("base_validation_loss"))
             == float(first_minimum["val_base_loss"]),
             "best OCR checkpoint base-validation value differs")
    _require(int(final.get("epoch", -1)) == int(binding.expected_training_arguments["epochs"]),
             "final OCR checkpoint epoch differs")
    for payload_name, payload in (("best", best), ("final", final)):
        payload_config = payload.get("config")
        _require(isinstance(payload_config, dict)
                 and payload_config.get("cell_id") == binding.cell_id
                 and payload_config.get("source_commit") == commit,
                 f"{payload_name} OCR checkpoint lineage differs")
    _require(sampler_order.get("schema_version") == "ocr_zncc_sampler_order.v1",
             "OCR sampler-order schema differs")
    epoch_orders = sampler_order.get("epochs")
    _require(isinstance(epoch_orders, list)
             and len(epoch_orders) == int(binding.expected_training_arguments["epochs"]),
             "OCR sampler-order epoch count differs")
    train_count = config.get("index_provenance", {}).get("train", {}).get(
        "pair_count_after_object_filter"
    )
    _require(isinstance(train_count, int) and train_count == 147,
             "OCR completed train pair count differs")
    _require(all(row.get("sample_count") == train_count for row in epoch_orders),
             "OCR sampler-order sample count differs")
    source_files = {
        relative: _file_record(repo_root / relative, name=relative)
        for relative in RUN_SOURCE_PATHS
    }
    expected_steps = int(binding.expected_training_arguments["epochs"]) * math.ceil(
        train_count / int(binding.expected_training_arguments["batch_size"])
    )
    _require(optimizer_step_count == expected_steps, "optimizer step count differs")
    document = {
        "schema_version": COMPLETED_RUN_SCHEMA_VERSION,
        "artifact_type": "ocr_zncc_completed_run_receipt",
        "cell_id": binding.cell_id,
        "recipe": binding.recipe,
        "arm": binding.arm,
        "source_commit": commit,
        "source_branch": branch,
        "manifest": {
            "absolute_path": binding.manifest_absolute_path,
            "file_sha256": binding.manifest_file_sha256,
            "content_hash_sha256": binding.manifest_content_sha256,
        },
        "expected_training_arguments": dict(binding.expected_training_arguments),
        "run_directory": str(run_directory),
        "artifacts": records,
        "source_files": source_files,
        "execution": {
            "device": device,
            "optimizer_step_count": optimizer_step_count,
            "scheduler_step_count": int(binding.expected_training_arguments["epochs"]),
            "full_command": list(full_command),
            "runtime_environment": dict(runtime_environment),
            "determinism": dict(determinism),
            "nondeterminism_evidence": dict(nondeterminism_evidence),
        },
        "checkpoint_selection": "minimum_base_validation_loss",
        "checkpoint_validation": {
            "best_is_first_minimum_base_validation": True,
            "best_epoch": int(best["epoch"]),
            "final_is_fixed_epoch": True,
            "final_epoch": int(final["epoch"]),
        },
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    output = run_directory / COMPLETED_RUN_RECEIPT_NAME
    data = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise ValueError("cannot create OCR completed-run receipt") from exc
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return output


def command_arguments(arguments: Mapping[str, Any]) -> list[str]:
    """Convert a complete manifest argument map into train.py CLI arguments."""
    result: list[str] = []
    for name in sorted(arguments):
        value = arguments[name]
        flag = f"--{name}"
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.extend([flag, str(value)])
    return result
