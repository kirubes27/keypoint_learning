"""Frozen paired outcome evaluation for completed OCR-ZNCC runs.

This module performs evaluation only.  It validates a completed-run receipt,
the exact experiment manifest, every receipt-bound training artifact, the
committed training/evaluation sources, and the frozen validation index before
loading either checkpoint.  It then evaluates ``best_model.pt`` and
``final_model.pt`` on validation frames 0..23, using all ten soft-coordinate
channels and the production representation evaluator's canonical-drift,
channel-health, and operator metric implementations.

No optimizer is constructed, no parameter is updated, and output is created
exclusively.  The generic evaluation configuration is frozen by its canonical
SHA-256 rather than selected from observed outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from keypoint_net import dataset as dataset_module
from keypoint_net import eval_representation as representation_evaluator
from keypoint_net import model as model_module
from keypoint_net import ocr_zncc_training_authorization as training_authorization
from keypoint_net import representation_corpus_inventory as corpus_inventory


SCHEMA_VERSION = "ocr_zncc_paired_outcome_cell.v1"
GENERIC_EVALUATION_CONFIG_SHA256 = (
    "151c973f56d123c085d7c1ab29fb6d720b69243915e06ad05fe1cb3590423e98"
)
VALIDATION_FRAME_IDS = tuple(range(24))
SOFT_COORDINATE_CHANNELS = tuple(range(10))
CHECKPOINT_ROLES = ("best_model", "final_model")
ROLL_CORPUS_INVENTORY_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_splits/"
    "inventories/CORPUS_INVENTORY__roll.json"
)
ROLL_CORPUS_CONTENT_SHA256 = (
    "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
)
SOURCE_RELATIVE_PATHS = (
    "keypoint_net/ocr_zncc_outcome_evaluator.py",
    "keypoint_net/ocr_zncc_training_authorization.py",
    "keypoint_net/eval_representation.py",
    "keypoint_net/model.py",
    "keypoint_net/dataset.py",
    "keypoint_net/representation_corpus_inventory.py",
)
_CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__(control|ocr_zncc)__seed(42|43|44)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OCROutcomeEvaluationError(ValueError):
    """Raised when outcome evidence cannot be bound or evaluated safely."""


@dataclass(frozen=True)
class ValidatedRun:
    cell_id: str
    recipe: str
    arm: str
    seed: int
    source_commit: str
    source_branch: str
    repo_root: Path
    receipt: Mapping[str, Any]
    receipt_record: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_record: Mapping[str, Any]
    run_directory: Path
    artifacts: Mapping[str, Mapping[str, Any]]
    evaluation_source_files: Mapping[str, Mapping[str, Any]]
    best_epoch: int
    best_base_validation_loss: float
    pairing_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedOutcomeResult:
    """A result document independently regenerated from its live run inputs."""

    document: Mapping[str, Any]
    result_record: Mapping[str, Any]
    run: ValidatedRun


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCROutcomeEvaluationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_regular(path_value: str | Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    path = Path(path_value).expanduser().absolute()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise OCROutcomeEvaluationError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
            size += len(block)
        _require(size == metadata.st_size, f"{name} size changed while reading")
        return b"".join(chunks), {
            "absolute_path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        }
    finally:
        os.close(descriptor)


def _strict_json(data: bytes, *, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"{name} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise OCROutcomeEvaluationError(f"{name} contains {token}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except OCROutcomeEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCROutcomeEvaluationError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _load_json(path: str | Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data, record = _read_regular(path, name=name)
    return _strict_json(data, name=name), record


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


def _committed_file_bytes_and_record(
    repo_root: Path,
    relative_path: str,
    *,
    source_commit: str,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    path = (repo_root / relative_path).resolve(strict=True)
    live, record = _read_regular(path, name=name)
    committed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{source_commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    _require(live == committed, f"committed file differs from commit: {relative_path}")
    return live, {"relative_path": relative_path, **record}


def _committed_source_record(
    repo_root: Path, relative_path: str, *, source_commit: str
) -> dict[str, Any]:
    _data, record = _committed_file_bytes_and_record(
        repo_root,
        relative_path,
        source_commit=source_commit,
        name=f"evaluation source {relative_path}",
    )
    return record


def _validate_checkout(repo_root: Path, *, source_commit: str, source_branch: str) -> dict[str, Any]:
    root = repo_root.expanduser().resolve(strict=True)
    _require(_git(root, "rev-parse", "HEAD") == source_commit, "checkout commit differs")
    _require(_git(root, "branch", "--show-current") == source_branch, "checkout branch differs")
    _require(
        _git(root, "status", "--porcelain") == "",
        "checkout is not completely clean",
    )
    try:
        subprocess.run(
            [
                "git", "-C", str(root), "merge-base", "--is-ancestor",
                training_authorization.AUTHORIZED_BASE_COMMIT, source_commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise OCROutcomeEvaluationError(
            "authorized clean research base is not an ancestor"
        ) from exc
    records = {
        relative: _committed_source_record(root, relative, source_commit=source_commit)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "absolute_path": str(root),
        "source_commit": source_commit,
        "source_branch": source_branch,
        "complete_status_clean": True,
        "authorized_base_commit": training_authorization.AUTHORIZED_BASE_COMMIT,
        "authorized_base_is_ancestor": True,
        "source_files": records,
    }


def _verify_bound_record(
    claimed: Mapping[str, Any], *, expected_path: Path, name: str
) -> dict[str, Any]:
    _require(set(claimed) == {"absolute_path", "sha256", "size_bytes"},
             f"{name} record keys differ")
    data, observed = _read_regular(expected_path, name=name)
    del data
    _require(observed == dict(claimed), f"{name} bytes/path differ from binding")
    return observed


def _validate_manifest(
    manifest_record: Mapping[str, Any], *, run_receipt: Mapping[str, Any], cell_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        set(manifest_record)
        == {"absolute_path", "file_sha256", "content_hash_sha256"},
        "receipt manifest record keys differ",
    )
    manifest, observed = _load_json(
        str(manifest_record.get("absolute_path", "")), name="OCR experiment manifest"
    )
    _require(
        observed.get("sha256") == manifest_record.get("file_sha256"),
        "experiment manifest file binding differs",
    )
    content_hash = _validate_content_hash(manifest, name="OCR experiment manifest")
    _require(
        content_hash == manifest_record.get("content_hash_sha256"),
        "experiment manifest content binding differs",
    )
    _require(
        manifest.get("schema_version") == training_authorization.EXPERIMENT_SCHEMA_VERSION,
        "experiment manifest schema differs",
    )
    _require(
        manifest.get("source_commit") == run_receipt.get("source_commit")
        and manifest.get("source_branch") == run_receipt.get("source_branch"),
        "experiment manifest source lineage differs",
    )
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "experiment manifest cells are invalid")
    matches = [
        row for row in cells
        if isinstance(row, Mapping) and row.get("cell_id") == cell_id
    ]
    _require(len(matches) == 1, "experiment manifest cell is absent or duplicated")
    _require(
        matches[0].get("training_arguments")
        == run_receipt.get("expected_training_arguments"),
        "receipt training arguments differ from experiment manifest",
    )
    recipe = str(run_receipt.get("recipe"))
    exploratory_stage = str(matches[0].get("stage", "")).endswith(
        "_exploratory_override"
    )
    _require(
        manifest.get("decision_lock")
        == {
            "authorized_recipe": recipe,
            "authorization_mode": (
                "exploratory_user_override_failed_coverage"
                if exploratory_stage else "legacy_staged_authorization"
            ),
            "checkpoint_selector": "minimum_base_validation_loss",
            "validation_aggregation": "sample_weighted_complete_21_pair_mean",
            "fixed_final_estimand": "epoch_1000_final_model",
            "test_split_used": False,
            "pair_execution": "control_then_ocr_sequential_on_same_allocated_gpu",
            "partial_failure_policy": "fail_closed_new_attempt_root_required_no_resume",
        },
        "manifest decision lock differs",
    )
    _require(
        manifest.get("primary_outcome")
        == {
            "field": "canonical_drift.median_rms_objdiag",
            "coordinate_type": "soft_spatial_expectation",
            "population": "all_channels_0_through_9_on_validation_frames_0_through_23",
            "frame_count": 24,
            "pair_count": 21,
            "normalization": (
                "per_frame_binary_mask_bbox_diagonal_on_endpoint_aligned_0_1_grid"
            ),
            "canonicalization": (
                "inverse_known_world_z_roll_about_projected_center_0_0"
            ),
            "aggregation": (
                "per_channel_rms_over_24_frames_then_median_over_10_channels"
            ),
            "generic_evaluation_config_sha256": GENERIC_EVALUATION_CONFIG_SHA256,
            "lower_is_better": True,
            "ocr_acceptance_or_channel_filtering_used": False,
        },
        "manifest primary outcome lock differs",
    )
    _require(
        manifest.get("frozen_stage_criteria")
        == {
            "selected_checkpoint_drift_ratio_at_most": 0.80,
            "selected_checkpoint_seed_passes_required": 2,
            "final_checkpoint_ocr_not_worse_seed_passes_required": 2,
            "ocr_correct_roll_sign_required_all_seeds": True,
            "ocr_angle_error_not_worse_seed_passes_required": 2,
            "active_on_object_count_not_lower_any_seed": True,
            "reflection_or_nonfinite_or_provenance_failure_blocks": True,
            "sample_inference": "descriptive_paired_seeds_not_population_inference",
        },
        "manifest frozen Task55/80 criterion lock differs",
    )
    decision_claim = manifest.get("authorization_decision")
    _require(isinstance(decision_claim, Mapping), "manifest authorization decision is absent")
    decision, decision_observed = _load_json(
        str(decision_claim.get("absolute_path", "")), name="training authorization decision"
    )
    _require(
        decision_observed == dict(decision_claim),
        "live training authorization decision differs from manifest binding",
    )
    decision_content = _validate_content_hash(
        decision, name="training authorization decision"
    )
    del decision_content
    expected_decision = {
        "task55_clean": (
            "ocr_zncc_task55_training_authorization.v1",
            "authorize_task55_matched_experiment",
        ),
        "task80_assisted": (
            "ocr_zncc_task80_training_authorization.v1",
            "authorize_task80_matched_experiment",
        ),
    }[recipe]
    _require(
        decision.get("schema_version") == expected_decision[0]
        and decision.get("status") == expected_decision[1]
        and decision.get("source_commit") == run_receipt.get("source_commit")
        and decision.get("source_branch") == run_receipt.get("source_branch"),
        "training authorization decision lineage/status differs",
    )
    authorized_coefficient = decision.get("coefficient")
    _require(
        isinstance(authorized_coefficient, (int, float))
        and not isinstance(authorized_coefficient, bool)
        and math.isfinite(float(authorized_coefficient))
        and 0.0 < float(authorized_coefficient) <= 0.5
        and float(manifest.get("ocr_zncc", {}).get("coefficient"))
        == float(authorized_coefficient),
        "training authorization coefficient differs from manifest",
    )
    normal_rows = [
        row for row in cells
        if isinstance(row, Mapping) and _CELL_RE.fullmatch(str(row.get("cell_id")))
    ]
    normal_cells = {
        row.get("cell_id"): row
        for row in normal_rows
    }
    expected_normal_ids = {
        f"{recipe}__{arm}__seed{seed}"
        for arm in ("control", "ocr_zncc")
        for seed in (42, 43, 44)
    }
    _require(
        len(normal_rows) == len(normal_cells) == 6
        and set(normal_cells) == expected_normal_ids,
        "manifest scientific cell set differs",
    )
    for normal_id, row in normal_cells.items():
        arguments = row.get("training_arguments")
        _require(isinstance(arguments, Mapping), f"manifest cell {normal_id} arguments are invalid")
        identity = _CELL_RE.fullmatch(str(normal_id))
        _require(identity is not None, f"manifest cell {normal_id} identity is invalid")
        identity_recipe, identity_arm, identity_seed = identity.groups()
        arm = str(row.get("arm"))
        expected_lambda = 0.0 if arm == "control" else float(authorized_coefficient)
        _require(
            identity_recipe == row.get("recipe") == recipe
            and identity_arm == arm
            and int(identity_seed) == row.get("seed") == arguments.get("seed")
            and arguments.get("lambda_ocr_zncc") == expected_lambda,
            f"manifest cell {normal_id} coefficient/recipe differs",
        )
    dataset_record = manifest.get("dataset")
    _require(isinstance(dataset_record, Mapping), "manifest dataset record is invalid")
    expected_arguments = run_receipt.get("expected_training_arguments")
    _require(isinstance(expected_arguments, Mapping), "receipt training arguments are invalid")
    _require(
        dataset_record.get("binding_sha256")
        == training_authorization.FIXED_TRAINING_ARGUMENTS["dataset_binding_sha256"]
        and dataset_record.get("object") == "engineers_hammer_vray",
        "manifest dataset identity differs",
    )
    _require(
        Path(str(dataset_record.get("absolute_path", ""))).expanduser().resolve(strict=True)
        == Path(str(expected_arguments.get("data_root", ""))).expanduser().resolve(strict=True),
        "manifest data root differs from cell arguments",
    )
    validation_record = dataset_record.get("validation_index")
    _require(isinstance(validation_record, Mapping), "manifest validation index is absent")
    validation_path = Path(str(validation_record.get("absolute_path", "")))
    observed_validation = _verify_bound_record(
        validation_record, expected_path=validation_path, name="validation pair index"
    )
    _require(
        observed_validation["sha256"] == training_authorization.VALIDATION_INDEX_SHA256,
        "validation pair-index hash differs from the frozen OCR binding",
    )
    _require(
        validation_path.expanduser().resolve(strict=True)
        == Path(str(expected_arguments.get("val_pairs_index", ""))).expanduser().resolve(strict=True),
        "manifest validation index path differs from cell arguments",
    )
    return manifest, {
        **observed,
        "content_hash_sha256": content_hash,
        "validation_index": observed_validation,
    }


def _validate_runtime_pair_execution(
    *,
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    source_commit: str,
    source_branch: str,
    recipe: str,
    arm: str,
    seed: int,
    manifest_record: Mapping[str, Any],
) -> dict[str, Any]:
    execution = receipt.get("execution")
    _require(isinstance(execution, Mapping), "receipt execution record is invalid")
    runtime = execution.get("runtime_environment")
    _require(
        isinstance(runtime, Mapping)
        and config.get("runtime_environment") == runtime,
        "config/receipt runtime environment differs",
    )
    pair = runtime.get("ocr_pair_execution")
    expected_position = "1_control" if arm == "control" else "2_ocr_zncc"
    expected_pair_id = f"{recipe}__seed{seed}"
    _require(
        isinstance(pair, Mapping)
        and set(pair)
        == {
            "pair_id",
            "position",
            "wrapper_pid",
            "slurm_job_id",
            "cuda_visible_devices",
            "allocated_gpu_count",
        }
        and pair.get("pair_id") == expected_pair_id
        and pair.get("position") == expected_position
        and isinstance(pair.get("wrapper_pid"), int)
        and not isinstance(pair.get("wrapper_pid"), bool)
        and int(pair["wrapper_pid"]) > 0
        and isinstance(pair.get("slurm_job_id"), str)
        and bool(pair["slurm_job_id"])
        and pair.get("slurm_job_id") == runtime.get("slurm_job_id")
        and isinstance(pair.get("cuda_visible_devices"), str)
        and bool(pair["cuda_visible_devices"])
        and "," not in pair["cuda_visible_devices"]
        and pair.get("allocated_gpu_count") == 1
        and execution.get("device") == "cuda"
        and runtime.get("device_type") == "cuda",
        "OCR run lacks exact single-GPU sequential pair execution provenance",
    )
    artifact_fields = {
        "stage_launch_lock": "ocr_stage_launch_lock",
        "environment_lock": "environment_lock",
        "slurm_script": "slurm_job_script",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for output_name, runtime_name in artifact_fields.items():
        claim = runtime.get(runtime_name)
        _require(
            isinstance(claim, Mapping),
            f"runtime {runtime_name} binding is absent",
        )
        artifacts[output_name] = _verify_bound_record(
            claim,
            expected_path=Path(str(claim.get("absolute_path", ""))),
            name=f"runtime {runtime_name}",
        )
    claimed_sources = receipt.get("source_files")
    _require(isinstance(claimed_sources, Mapping), "receipt source files are invalid")
    _require(
        artifacts["slurm_script"]
        == claimed_sources.get("cluster/ocr_zncc_paired_stage.slurm"),
        "runtime Slurm script differs from receipt-bound committed source",
    )
    stage, stage_observed = _load_json(
        artifacts["stage_launch_lock"]["absolute_path"],
        name="OCR paired stage launch lock",
    )
    stage_hash = _validate_content_hash(stage, name="OCR paired stage launch lock")
    _require(
        stage_observed == artifacts["stage_launch_lock"]
        and stage.get("schema_version") == "ocr_zncc_paired_stage_launch_lock.v1"
        and stage.get("artifact_type") == "ocr_zncc_paired_stage_launch_lock"
        and stage.get("source_commit") == source_commit
        and stage.get("source_branch") == source_branch
        and stage.get("recipe") == recipe
        and stage.get("environment_lock") == artifacts["environment_lock"]
        and stage.get("slurm_script") == artifacts["slurm_script"],
        "runtime stage-lock identity or artifact binding differs",
    )
    stage_manifest = stage.get("manifest")
    _require(
        isinstance(stage_manifest, Mapping)
        and stage_manifest.get("absolute_path") == manifest_record.get("absolute_path")
        and stage_manifest.get("sha256") == manifest_record.get("sha256")
        and stage_manifest.get("size_bytes") == manifest_record.get("size_bytes")
        and stage_manifest.get("content_hash_sha256")
        == manifest_record.get("content_hash_sha256"),
        "runtime stage lock binds a different experiment manifest",
    )
    _require(
        stage.get("pair_order")
        == [
            {
                "seed": item_seed,
                "first": f"{recipe}__control__seed{item_seed}",
                "second": f"{recipe}__ocr_zncc__seed{item_seed}",
            }
            for item_seed in (42, 43, 44)
        ],
        "runtime stage-lock pair order differs",
    )
    return {
        **dict(pair),
        "stage_launch_lock_content_hash_sha256": stage_hash,
        "runtime_provenance_files": artifacts,
    }


def validate_completed_run(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    expected_cell_id: str,
) -> ValidatedRun:
    """Validate a completed OCR run without opening checkpoint payloads."""

    identity = _CELL_RE.fullmatch(expected_cell_id)
    _require(identity is not None, "expected OCR cell ID is invalid")
    expected_recipe, expected_arm, expected_seed_text = identity.groups()
    receipt, receipt_record = _load_json(receipt_path, name="OCR completed-run receipt")
    _validate_content_hash(receipt, name="OCR completed-run receipt")
    _require(
        receipt.get("schema_version") == training_authorization.COMPLETED_RUN_SCHEMA_VERSION,
        "OCR completed-run receipt schema differs",
    )
    _require(receipt.get("cell_id") == expected_cell_id, "receipt cell ID differs")
    _require(
        receipt.get("recipe") == expected_recipe
        and receipt.get("arm") == expected_arm,
        "receipt recipe/arm differs",
    )
    source_commit = receipt.get("source_commit")
    source_branch = receipt.get("source_branch")
    _require(isinstance(source_commit, str), "receipt source commit is invalid")
    _require(
        source_branch == training_authorization.EXPECTED_BRANCH,
        "receipt source branch differs",
    )
    root = Path(repo_root).expanduser().resolve(strict=True)
    checkout = _validate_checkout(
        root, source_commit=source_commit, source_branch=source_branch
    )

    run_directory = Path(str(receipt.get("run_directory", ""))).expanduser().resolve(strict=True)
    _require(
        Path(receipt_record["absolute_path"])
        == run_directory / training_authorization.COMPLETED_RUN_RECEIPT_NAME,
        "completed-run receipt path differs from run directory",
    )
    expected_arguments = receipt.get("expected_training_arguments")
    _require(
        isinstance(expected_arguments, Mapping)
        and set(expected_arguments) == training_authorization.BOUND_ARGUMENTS,
        "receipt training argument set differs",
    )
    seed = int(expected_seed_text)
    _require(
        expected_arguments.get("seed") == seed
        and expected_arguments.get("object") == "engineers_hammer_vray"
        and expected_arguments.get("num_keypoints") == 10
        and expected_arguments.get("heatmap_res") == 64
        and expected_arguments.get("img_size") == 512
        and expected_arguments.get("center_crop") is None,
        "receipt evaluator-critical training arguments differ",
    )
    _require(
        all(
            expected_arguments.get(name) == value
            for name, value in training_authorization.TASK_RECIPES[expected_recipe].items()
        ),
        "receipt recipe weights differ",
    )
    coefficient = float(expected_arguments.get("lambda_ocr_zncc", -1.0))
    _require(
        math.isfinite(coefficient)
        and ((coefficient == 0.0) == (expected_arm == "control"))
        and (coefficient == 0.0 or coefficient > 0.0)
        and (coefficient <= 0.5),
        "receipt arm/coefficient binding differs",
    )
    _require(
        expected_arguments.get("val_pairs_index") is not None
        and expected_arguments.get("dataset_binding_sha256")
        == training_authorization.FIXED_TRAINING_ARGUMENTS["dataset_binding_sha256"],
        "receipt validation data binding differs",
    )
    _require(
        receipt.get("checkpoint_selection") == "minimum_base_validation_loss"
        and receipt.get("checkpoint_validation", {}).get(
            "best_is_first_minimum_base_validation"
        ) is True
        and receipt.get("checkpoint_validation", {}).get("final_is_fixed_epoch") is True,
        "receipt checkpoint policy differs",
    )

    manifest_claim = receipt.get("manifest")
    _require(isinstance(manifest_claim, Mapping), "receipt manifest binding is invalid")
    manifest, manifest_record = _validate_manifest(
        manifest_claim, run_receipt=receipt, cell_id=expected_cell_id
    )

    claimed_artifacts = receipt.get("artifacts")
    expected_names = {
        "config.json", "history.json", "sampler_order.json",
        "best_model.pt", "final_model.pt",
    }
    _require(
        isinstance(claimed_artifacts, Mapping) and set(claimed_artifacts) == expected_names,
        "receipt artifact set differs",
    )
    artifacts = {
        name: _verify_bound_record(
            claimed_artifacts[name], expected_path=run_directory / name, name=name
        )
        for name in sorted(expected_names)
    }
    config, _ = _load_json(run_directory / "config.json", name="run config")
    history, _history_record = _load_json_list(
        run_directory / "history.json", name="run history"
    )
    sampler, _ = _load_json(run_directory / "sampler_order.json", name="sampler order")
    _require(
        config.get("cell_id") == expected_cell_id
        and config.get("source_commit") == source_commit
        and config.get("checkpoint_policy", {}).get("selection")
        == "minimum_base_validation_loss",
        "run config lineage/checkpoint policy differs",
    )
    completed_binding = training_authorization.OCRTrainingBinding(
        cell_id=expected_cell_id,
        recipe=expected_recipe,
        arm=expected_arm,
        run_directory=str(run_directory),
        manifest_absolute_path=manifest_record["absolute_path"],
        manifest_file_sha256=manifest_record["sha256"],
        manifest_content_sha256=manifest_record["content_hash_sha256"],
        expected_training_arguments=dict(expected_arguments),
    )
    validation_rows = training_authorization._validate_completed_validation_history(
        history,
        completed_binding,
    )
    first_minimum = min(
        validation_rows,
        key=lambda row: (float(row["val_base_loss"]), int(row["epoch"])),
    )
    best_epoch = int(first_minimum["epoch"])
    best_base_validation_loss = float(first_minimum["val_base_loss"])
    _require(
        receipt.get("checkpoint_validation", {}).get("best_epoch") == best_epoch
        and receipt.get("checkpoint_validation", {}).get("final_epoch")
        == int(expected_arguments["epochs"]) == 1000,
        "receipt checkpoint epochs disagree with history/frozen final epoch",
    )
    _require(
        sampler.get("schema_version") == "ocr_zncc_sampler_order.v1",
        "sampler-order schema differs",
    )
    epoch_orders = sampler.get("epochs")
    _require(
        isinstance(epoch_orders, list) and len(epoch_orders) == 1000,
        "sampler-order epoch count differs",
    )
    _require(
        all(
            isinstance(row, Mapping)
            and row.get("epoch") == index
            and row.get("sample_count") == 147
            and isinstance(row.get("pair_order_sha256"), str)
            and _SHA256_RE.fullmatch(row["pair_order_sha256"]) is not None
            for index, row in enumerate(epoch_orders, start=1)
        ),
        "sampler-order epoch identity/count/hash differs",
    )
    execution = receipt.get("execution")
    _require(
        isinstance(execution, Mapping)
        and execution.get("optimizer_step_count") == 10_000
        and execution.get("scheduler_step_count") == 1000,
        "receipt optimizer/scheduler step counts differ",
    )
    pairing_fields = {
        "initial_model_state_sha256": config.get("initial_model_state_sha256"),
        "post_diagnostic_model_state_sha256": config.get(
            "post_diagnostic_model_state_sha256"
        ),
        "diagnostic_pair_order_sha256": config.get("diagnostic_pair_order_sha256"),
    }
    _require(
        all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
            for value in pairing_fields.values()
        ),
        "run config matched-pair hashes are invalid",
    )
    _require(
        config.get("diagnostic_pair_order_sample_count") == 16
        and sampler.get("diagnostic_pair_order_sha256")
        == pairing_fields["diagnostic_pair_order_sha256"]
        and sampler.get("diagnostic_pair_order_sample_count") == 16
        and sampler.get("generator_seed")
        == int(expected_arguments["seed"]) + 10_000_019,
        "run config/sampler matched-pair evidence differs",
    )
    runtime_pair_execution = _validate_runtime_pair_execution(
        receipt=receipt,
        config=config,
        source_commit=source_commit,
        source_branch=source_branch,
        recipe=expected_recipe,
        arm=expected_arm,
        seed=seed,
        manifest_record=manifest_record,
    )
    pairing_evidence = {
        **pairing_fields,
        "diagnostic_pair_order_sample_count": 16,
        "train_sampler_generator_seed": sampler["generator_seed"],
        "epoch_pair_order_sequence_sha256": canonical_sha256([
            {
                "epoch": row["epoch"],
                "pair_order_sha256": row["pair_order_sha256"],
                "sample_count": row["sample_count"],
            }
            for row in epoch_orders
        ]),
        "execution_device": execution.get("device"),
        "runtime_pair_execution": runtime_pair_execution,
    }
    _require(
        isinstance(pairing_evidence["execution_device"], str)
        and bool(pairing_evidence["execution_device"]),
        "receipt execution device is invalid",
    )

    claimed_sources = receipt.get("source_files")
    _require(
        isinstance(claimed_sources, Mapping)
        and set(claimed_sources) == set(training_authorization.RUN_SOURCE_PATHS),
        "receipt training-source role set differs",
    )
    for relative, claimed in claimed_sources.items():
        observed = _verify_bound_record(
            claimed, expected_path=root / relative, name=f"training source {relative}"
        )
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{source_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        _require(
            hashlib.sha256(committed).hexdigest() == observed["sha256"],
            f"training source does not match commit: {relative}",
        )

    return ValidatedRun(
        cell_id=expected_cell_id,
        recipe=expected_recipe,
        arm=expected_arm,
        seed=seed,
        source_commit=source_commit,
        source_branch=source_branch,
        repo_root=root,
        receipt=receipt,
        receipt_record={
            **receipt_record,
            "content_hash_sha256": receipt["content_hash_sha256"],
        },
        manifest=manifest,
        manifest_record=manifest_record,
        run_directory=run_directory,
        artifacts=artifacts,
        evaluation_source_files=checkout["source_files"],
        best_epoch=best_epoch,
        best_base_validation_loss=best_base_validation_loss,
        pairing_evidence=pairing_evidence,
    )


def _load_json_list(path: Path, *, name: str) -> tuple[list[Any], dict[str, Any]]:
    data, record = _read_regular(path, name=name)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"{name} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise OCROutcomeEvaluationError(f"{name} contains {token}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except OCROutcomeEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCROutcomeEvaluationError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, list), f"{name} is not a JSON list")
    return value, record


def evaluation_config() -> dict[str, Any]:
    config = {
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
    _require(
        representation_evaluator.canonical_sha256(config)
        == GENERIC_EVALUATION_CONFIG_SHA256,
        "generic evaluation configuration hash differs",
    )
    return config


def _relative_dataset_path(root: Path, relative: str, *, name: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OCROutcomeEvaluationError(f"{name} escapes data root") from exc
    return path


def _register_endpoint(
    records: dict[int, dict[str, Any]],
    *,
    frame_id: Any,
    image_relpath: Any,
    mask_relpath: Any,
    state: Any,
) -> None:
    _require(
        isinstance(frame_id, int) and not isinstance(frame_id, bool),
        "validation frame ID is not an integer",
    )
    _require(isinstance(image_relpath, str) and image_relpath, "image relpath is invalid")
    _require(isinstance(mask_relpath, str) and mask_relpath, "mask relpath is invalid")
    _require(isinstance(state, Mapping), "validation physical state is invalid")
    normalized_state = dict(state)
    _require(set(normalized_state) == {"theta_deg"}, "roll state keys differ")
    theta = normalized_state["theta_deg"]
    _require(
        isinstance(theta, (int, float))
        and not isinstance(theta, bool)
        and math.isfinite(float(theta)),
        "roll state angle is invalid",
    )
    candidate = {
        "frame_id": frame_id,
        "image_relpath": image_relpath,
        "mask_relpath": mask_relpath,
        "physical_state": {"theta_deg": float(theta)},
    }
    if frame_id in records:
        _require(records[frame_id] == candidate, f"frame {frame_id} metadata is inconsistent")
    else:
        records[frame_id] = candidate


def _load_frame_arrays(
    data_root: Path, frame_record: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path = _relative_dataset_path(
        data_root, str(frame_record["image_relpath"]), name="validation image"
    )
    mask_path = _relative_dataset_path(
        data_root, str(frame_record["mask_relpath"]), name="validation mask"
    )
    image_data, image_record = _read_regular(image_path, name="validation image")
    mask_data, mask_record = _read_regular(mask_path, name="validation mask")
    try:
        with Image.open(io.BytesIO(image_data)) as handle:
            _require(handle.size == (512, 512), "validation image resolution differs")
            image = np.asarray(handle.convert("RGB")).copy()
        with Image.open(io.BytesIO(mask_data)) as handle:
            _require(handle.size == (512, 512), "validation mask resolution differs")
            mask = np.asarray(handle.convert("L")).copy() > 0
    except (OSError, ValueError) as exc:
        if isinstance(exc, OCROutcomeEvaluationError):
            raise
        raise OCROutcomeEvaluationError("validation image/mask decode failed") from exc
    _require(image.shape == (512, 512, 3), "validation RGB array shape differs")
    _require(mask.shape == (512, 512) and bool(np.any(mask)), "validation mask is invalid")
    return image, mask, {
        "frame_id": int(frame_record["frame_id"]),
        "image": {"relative_path": frame_record["image_relpath"], **image_record},
        "mask": {"relative_path": frame_record["mask_relpath"], **mask_record},
        "physical_state": dict(frame_record["physical_state"]),
    }


def _validate_roll_corpus_inventory(
    run: ValidatedRun, *, data_root: Path
) -> dict[str, Any]:
    inventory_bytes, inventory_record = _committed_file_bytes_and_record(
        run.repo_root,
        ROLL_CORPUS_INVENTORY_RELATIVE_PATH,
        source_commit=run.source_commit,
        name="committed roll corpus inventory",
    )
    try:
        validated = corpus_inventory.validate_corpus_inventory(
            inventory_bytes, "roll", data_root
        )
    except Exception as exc:
        raise OCROutcomeEvaluationError(
            f"live roll corpus differs from committed inventory: {exc}"
        ) from exc
    _require(
        validated.content_hash_sha256 == ROLL_CORPUS_CONTENT_SHA256
        and validated.content_hash_sha256
        == training_authorization.FIXED_TRAINING_ARGUMENTS[
            "dataset_binding_sha256"
        ],
        "roll corpus inventory content hash differs from frozen dataset binding",
    )
    validator_source = run.evaluation_source_files.get(
        "keypoint_net/representation_corpus_inventory.py"
    )
    _require(
        isinstance(validator_source, Mapping),
        "committed corpus-inventory validator source is absent",
    )
    return {
        "dataset_key": "roll",
        "validated_data_root": str(data_root),
        "content_hash_sha256": validated.content_hash_sha256,
        "exact_live_corpus_match": True,
        "inventory_file": {
            **inventory_record,
            "content_hash_sha256": validated.content_hash_sha256,
        },
        "validator_source": dict(validator_source),
    }


def load_frozen_validation_data(
    run: ValidatedRun, *, data_root: Path | str
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve(strict=True)
    arguments = run.receipt["expected_training_arguments"]
    _require(
        root == Path(str(arguments["data_root"])).expanduser().resolve(strict=True),
        "evaluation data root differs from the receipt-bound training corpus",
    )
    inventory_binding = _validate_roll_corpus_inventory(run, data_root=root)
    manifest = dataset_module.inspect_index_pair_manifest(
        data_root=str(root),
        index_path=str(arguments["val_pairs_index"]),
        expected_split="validation",
        object_name="engineers_hammer_vray",
        strict_metadata=True,
        expected_index_sha256=training_authorization.VALIDATION_INDEX_SHA256,
        expected_dataset_binding_sha256=training_authorization.FIXED_TRAINING_ARGUMENTS[
            "dataset_binding_sha256"
        ],
        validate_paths=True,
    )
    _require(
        manifest.index_sha256 == run.manifest_record["validation_index"]["sha256"],
        "loaded validation index differs from experiment manifest",
    )
    transform = manifest.transform
    expected_transform = {
        "family": "roll",
        "physical_axis": "world_z",
        "direction": "forward",
        "signed_generator": 6.0,
        "generator_units": "degrees",
        "stride": 3,
        "stride_units": "frames",
        "cyclic": True,
        "expected_2d_family": "planar_rotation_about_projected_center",
    }
    _require(transform == expected_transform, "validation transform metadata differs")

    endpoints: dict[int, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for pair in manifest.pairs:
        _require(pair.get("model_name") == "engineers_hammer_vray", "validation object differs")
        _require(pair.get("split") == "validation", "validation pair split differs")
        pair_id = pair.get("pair_id")
        _require(isinstance(pair_id, str) and pair_id not in seen_pair_ids,
                 "validation pair ID is invalid or duplicated")
        seen_pair_ids.add(pair_id)
        _register_endpoint(
            endpoints,
            frame_id=pair["src_frame_index"],
            image_relpath=pair["src_image_relpath"],
            mask_relpath=pair["src_mask_relpath"],
            state=pair["src_state"],
        )
        _register_endpoint(
            endpoints,
            frame_id=pair["dst_frame_index"],
            image_relpath=pair["dst_image_relpath"],
            mask_relpath=pair["dst_mask_relpath"],
            state=pair["dst_state"],
        )
        _require(
            abs(float(pair["dst_state"]["theta_deg"])
                - float(pair["src_state"]["theta_deg"]) - 6.0) <= 1e-12,
            "validation pair state increment differs from +6 degrees",
        )
        pair_rows.append({
            "pair_id": pair_id,
            "source_frame": int(pair["src_frame_index"]),
            "target_frame": int(pair["dst_frame_index"]),
            "direction": "forward",
            "stride": 3,
            "partition": "validation",
        })
    _require(tuple(sorted(endpoints)) == VALIDATION_FRAME_IDS,
             "validation endpoint frame set differs from frozen 0..23")
    _require(
        len(pair_rows) == 21
        and {
            (row["source_frame"], row["target_frame"])
            for row in pair_rows
        } == {(source, source + 3) for source in range(21)},
        "validation pair graph differs from the frozen 21 forward +3 edges",
    )

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    input_records: list[dict[str, Any]] = []
    states: list[dict[str, float]] = []
    for frame_id in VALIDATION_FRAME_IDS:
        image, mask, record = _load_frame_arrays(root, endpoints[frame_id])
        images.append(image)
        masks.append(mask)
        input_records.append(record)
        states.append(dict(endpoints[frame_id]["physical_state"]))
    return {
        "data_root": str(root),
        "frame_ids": list(VALIDATION_FRAME_IDS),
        "images": images,
        "masks": np.stack(masks),
        "physical_states": states,
        "pair_rows": pair_rows,
        "input_records": input_records,
        "validation_index": {
            "absolute_path": str(manifest.index_path),
            "sha256": manifest.index_sha256,
            "content_hash_sha256": manifest.content_hash_sha256,
        },
        "dataset_binding_sha256": manifest.dataset_binding_sha256,
        "corpus_inventory": inventory_binding,
        "transform": expected_transform,
    }


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(cpu.shape)))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _safe_load_checkpoint(
    run: ValidatedRun, *, checkpoint_role: str
) -> tuple[torch.nn.Module, Mapping[str, Any], dict[str, Any]]:
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role is invalid")
    filename = f"{checkpoint_role}.pt"
    binding = run.artifacts[filename]
    path = Path(str(binding["absolute_path"]))
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise OCROutcomeEvaluationError(f"cannot open {checkpoint_role}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{checkpoint_role} is not regular")
        _require(metadata.st_size == binding["size_bytes"],
                 f"{checkpoint_role} size changed at load")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _require(digest.hexdigest() == binding["sha256"],
                     f"{checkpoint_role} hash changed at load")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except OCROutcomeEvaluationError:
        raise
    except Exception as exc:
        raise OCROutcomeEvaluationError(f"safe {checkpoint_role} load failed") from exc
    finally:
        os.close(descriptor)

    _require(isinstance(payload, Mapping), f"{checkpoint_role} payload is not a mapping")
    expected_keys = (
        {
            "epoch", "model_state_dict", "optimizer_state_dict", "loss",
            "selection_loss_name", "base_validation_loss",
            "attachment_validation_loss", "ocr_zncc_validation_loss",
            "augmented_validation_loss", "config",
        }
        if checkpoint_role == "best_model"
        else {
            "epoch", "model_state_dict", "optimizer_state_dict", "loss",
            "base_training_loss", "attachment_training_loss",
            "ocr_zncc_training_loss", "ocr_zncc_accepted_match_coverage", "config",
        }
    )
    _require(set(payload) == expected_keys, f"{checkpoint_role} payload keys differ")
    config = payload.get("config")
    arguments = run.receipt["expected_training_arguments"]
    _require(
        isinstance(config, Mapping)
        and config.get("cell_id") == run.cell_id
        and config.get("source_commit") == run.source_commit
        and config.get("seed") == run.seed,
        f"{checkpoint_role} embedded lineage differs",
    )
    if checkpoint_role == "best_model":
        selected_value = payload.get("base_validation_loss")
        _require(
            payload.get("selection_loss_name") == "base_validation_loss"
            and int(payload.get("epoch", -1)) == run.best_epoch
            and isinstance(selected_value, (int, float))
            and not isinstance(selected_value, bool)
            and math.isfinite(float(selected_value))
            and float(selected_value) == run.best_base_validation_loss,
            "best checkpoint selector/epoch/base-validation value differs",
        )
    else:
        _require(
            int(payload.get("epoch", -1))
            == int(run.receipt["checkpoint_validation"]["final_epoch"])
            == int(arguments["epochs"]),
            "final checkpoint epoch differs",
        )

    model = model_module.PhaseAModel(
        num_keypoints=int(arguments["num_keypoints"]),
        base_channels=int(arguments["base_channels"]),
        temperature=float(arguments["temperature"]),
        num_action_classes=int(arguments["num_action_classes"]),
        padding_mode=str(arguments["padding_mode"]),
        operator_type=str(arguments["operator_type"]),
        learn_inverse_operator=bool(arguments["learn_inverse_operator"]),
        heatmap_res=int(arguments["heatmap_res"]),
    )
    state = payload.get("model_state_dict")
    _require(isinstance(state, Mapping) and set(state) == set(model.state_dict()),
             f"{checkpoint_role} model-state keys differ")
    for name, target in model.state_dict().items():
        source = state[name]
        _require(
            torch.is_tensor(source)
            and source.device.type == "cpu"
            and source.shape == target.shape
            and source.dtype == target.dtype
            and bool(torch.isfinite(source).all()),
            f"{checkpoint_role} tensor {name} is invalid",
        )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    load_record = {
        **dict(binding),
        "checkpoint_role": checkpoint_role,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
        "epoch": int(payload["epoch"]),
        "model_state_sha256": _model_state_sha256(model),
    }
    return model, payload, load_record


def _infer_soft_coordinates(
    model: torch.nn.Module, images: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require(not model.training, "outcome inference requires model.eval()")
    _require(
        all(not parameter.requires_grad for parameter in model.parameters()),
        "outcome inference requires frozen model parameters",
    )
    before = _model_state_sha256(model)
    means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    point_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            array = np.stack(images[start:start + 8])
            batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            batch = batch.sub_(means).div_(stds)
            flattened, logits = model.extractor(batch)
            point_batches.append(
                flattened.reshape(batch.shape[0], len(SOFT_COORDINATE_CHANNELS), 2)
                .detach().cpu().numpy().copy()
            )
            logit_batches.append(logits.detach().cpu().numpy().copy())
    supplied_points = np.concatenate(point_batches).astype(np.float32, copy=False)
    logits = np.concatenate(logit_batches).astype(np.float32, copy=False)
    _require(
        supplied_points.shape == (len(VALIDATION_FRAME_IDS), 10, 2)
        and logits.shape == (len(VALIDATION_FRAME_IDS), 10, 64, 64),
        "inference output shape differs",
    )
    _require(
        bool(np.isfinite(supplied_points).all()) and bool(np.isfinite(logits).all()),
        "inference output contains non-finite values",
    )
    derived_points = representation_evaluator.spatial_expectation(
        logits, temperature=1.0, softmax_dtype="float32"
    ).astype(np.float64)
    maximum_error = float(np.max(np.abs(derived_points - supplied_points)))
    _require(maximum_error <= 1e-6, "soft coordinates disagree with logits")
    _require(bool(np.all((derived_points >= -1.0) & (derived_points <= 1.0))),
             "soft coordinates lie outside [-1,1]")
    after = _model_state_sha256(model)
    _require(before == after, "model state changed during outcome inference")
    return derived_points, logits, {
        "frame_count": len(VALIDATION_FRAME_IDS),
        "channel_indices": list(SOFT_COORDINATE_CHANNELS),
        "coordinate_estimator": "production_spatial_expectation_from_float32_logits",
        "maximum_supplied_vs_recomputed_coordinate_error": maximum_error,
        "model_state_unchanged": True,
        "gradients_enabled_inside_inference": False,
        "model_state_sha256": after,
        "points_sha256": hashlib.sha256(derived_points.tobytes(order="C")).hexdigest(),
        "logits_sha256": hashlib.sha256(logits.tobytes(order="C")).hexdigest(),
    }


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    return np.asarray(
        [[math.cos(radians), -math.sin(radians)],
         [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )


def _numeric_values_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_values_finite(item) for item in value)
    return False


def _evaluate_checkpoint(
    run: ValidatedRun,
    validation: Mapping[str, Any],
    *,
    checkpoint_role: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model, payload, load_record = _safe_load_checkpoint(run, checkpoint_role=checkpoint_role)
    points, logits, inference = _infer_soft_coordinates(model, validation["images"])
    masks = np.asarray(validation["masks"], dtype=bool)
    frame_ids = list(VALIDATION_FRAME_IDS)
    frame_index_by_id = {frame_id: frame_id for frame_id in frame_ids}
    visibility = np.ones((len(frame_ids), len(SOFT_COORDINATE_CHANNELS)), dtype=bool)
    state_A = np.stack([
        _rotation_matrix(float(state["theta_deg"]))
        for state in validation["physical_states"]
    ])
    state_b = np.zeros((len(frame_ids), 2), dtype=np.float64)
    bbox = representation_evaluator._bbox_diagonals_from_masks(masks)  # noqa: SLF001
    estimator_metadata = {
        "input_height": 512,
        "input_width": 512,
        "heatmap_height": 64,
        "heatmap_width": 64,
        "endpoint_grid": True,
        "temperature": 1.0,
        "logit_dtype": "float32",
        "softmax_dtype": "float32",
        "crop": None,
        "resize": [512, 512],
        "align_corners": None,
    }
    canonical_drift = representation_evaluator._canonical_drift(  # noqa: SLF001
        points, state_A, state_b, bbox, visibility
    )
    channel_health = representation_evaluator._channel_health(  # noqa: SLF001
        points,
        masks,
        logits,
        visibility,
        validation["pair_rows"],
        frame_index_by_id,
        estimator_metadata,
        config,
    )
    transform = {
        **dict(validation["transform"]),
        "projected_centre_xy": [0.0, 0.0],
    }
    target_A = _rotation_matrix(6.0)
    target_b = np.zeros(2, dtype=np.float64)
    learned_A = model.operator.A.detach().cpu().numpy().astype(np.float64)
    learned_b = model.operator.bias.detach().cpu().numpy().astype(np.float64)
    operator = representation_evaluator._operator_metrics(  # noqa: SLF001
        "roll", learned_A, learned_b, target_A, target_b, transform, config
    )
    critical_failures: list[str] = []
    if operator.get("improper_or_reflection") is True:
        critical_failures.append("operator_is_improper_or_reflection")
    result = {
        "checkpoint_role": checkpoint_role,
        "checkpoint": load_record,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_selector": (
            "minimum_base_validation_loss" if checkpoint_role == "best_model"
            else "fixed_epoch_1000"
        ),
        "inference": inference,
        "operator": operator,
        "canonical_drift": canonical_drift,
        "channel_health": channel_health,
        "critical_failures": critical_failures,
    }
    _require(_numeric_values_finite(result), "outcome metrics contain non-finite values")
    result["all_numeric_outputs_finite"] = True
    return result


def _build_result_document(
    run: ValidatedRun, validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Regenerate the complete frozen result without reading claimed metrics."""

    config = evaluation_config()
    checkpoint_results = {
        role: _evaluate_checkpoint(
            run, validation, checkpoint_role=role, config=config
        )
        for role in CHECKPOINT_ROLES
    }
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_paired_ocr_zncc_outcome_cell",
        "evaluation_status": "complete",
        "cell_id": run.cell_id,
        "recipe": run.recipe,
        "arm": run.arm,
        "seed": run.seed,
        "source_commit": run.source_commit,
        "source_branch": run.source_branch,
        "metric_lock": {
            "primary_metric": "canonical_drift.median_rms_objdiag",
            "implementation": "keypoint_net.eval_representation._canonical_drift",
            "evaluation_config": config,
            "evaluation_config_sha256": GENERIC_EVALUATION_CONFIG_SHA256,
            "validation_frame_ids": list(VALIDATION_FRAME_IDS),
            "soft_coordinate_channel_indices": list(SOFT_COORDINATE_CHANNELS),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "threshold_selected_from_outcomes": False,
        },
        "lineage": {
            "completed_run_receipt": dict(run.receipt_record),
            "experiment_manifest": dict(run.manifest_record),
            "receipt_bound_run_artifacts": {
                name: dict(record) for name, record in run.artifacts.items()
            },
            "receipt_bound_training_sources": {
                name: dict(record)
                for name, record in run.receipt["source_files"].items()
            },
            "committed_evaluation_sources": {
                name: dict(record)
                for name, record in run.evaluation_source_files.items()
            },
            "dataset_binding_sha256": validation["dataset_binding_sha256"],
            "corpus_inventory": validation["corpus_inventory"],
            "validation_index": validation["validation_index"],
            "validation_inputs": validation["input_records"],
            "matched_pairing_evidence": dict(run.pairing_evidence),
        },
        "checkpoint_results": checkpoint_results,
        "statistical_scope": {
            "seed_is_descriptive_paired_repetition": True,
            "frames_and_channels_are_correlated": True,
            "inferential_test_performed": False,
            "sample_units": {
                "primary_cell_summary": "checkpoint_role_within_cell",
                "canonical_drift_summary": "ten_channel_canonical_RMS_values",
            },
        },
        "training_or_weight_update_performed": False,
        "selection_performed_by_evaluator": False,
    }
    _require(_numeric_values_finite(document), "cell result contains non-finite values")
    document["content_hash_sha256"] = canonical_sha256(document)
    return document


def validate_live_result(
    repo_root: Path | str, result_path: Path | str
) -> ValidatedOutcomeResult:
    """Recompute a claimed outcome cell from its exact live primary evidence.

    The supplied metric scalars are never inputs to recomputation.  The bound
    completed-run receipt identifies the exact checkpoints and validation data;
    both checkpoints are reloaded on CPU and the complete 24-frame, ten-channel
    result is regenerated.  Only byte-canonical equality with that regenerated
    document is accepted.
    """

    supplied, observed = _load_json(result_path, name="OCR outcome cell result")
    claimed_hash = _validate_content_hash(supplied, name="OCR outcome cell result")
    _require(
        supplied.get("schema_version") == SCHEMA_VERSION
        and supplied.get("artifact_type")
        == "frozen_paired_ocr_zncc_outcome_cell"
        and supplied.get("evaluation_status") == "complete",
        "OCR outcome cell identity/status differs",
    )
    cell_id = supplied.get("cell_id")
    _require(
        isinstance(cell_id, str) and _CELL_RE.fullmatch(cell_id) is not None,
        "OCR outcome cell ID is invalid",
    )
    lineage = supplied.get("lineage")
    _require(isinstance(lineage, Mapping), "OCR outcome cell lineage is invalid")
    receipt_claim = lineage.get("completed_run_receipt")
    _require(
        isinstance(receipt_claim, Mapping),
        "OCR outcome cell receipt lineage is invalid",
    )
    run = validate_completed_run(
        repo_root,
        str(receipt_claim.get("absolute_path", "")),
        expected_cell_id=cell_id,
    )
    _require(
        dict(receipt_claim) == dict(run.receipt_record),
        "OCR outcome cell receipt record differs from live run",
    )
    arguments = run.receipt.get("expected_training_arguments")
    _require(isinstance(arguments, Mapping), "live run training arguments are invalid")
    validation = load_frozen_validation_data(
        run, data_root=str(arguments.get("data_root", ""))
    )
    regenerated = _build_result_document(run, validation)
    _require(
        canonical_json_bytes(supplied) == canonical_json_bytes(regenerated),
        "OCR outcome cell differs from exact live metric recomputation",
    )
    return ValidatedOutcomeResult(
        document=regenerated,
        result_record={
            **observed,
            "content_hash_sha256": claimed_hash,
            "cell_id": cell_id,
        },
        run=run,
    )


def _write_exclusive(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(document) + b"\n"
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCROutcomeEvaluationError(f"cannot exclusively create {absolute}") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "exclusive result write made no progress")
            offset += written
    finally:
        os.close(descriptor)
    return {
        "absolute_path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "write_mode": "exclusive_create_no_overwrite",
    }


def evaluate_completed_cell(
    *,
    repo_root: Path | str,
    receipt_path: Path | str,
    cell_id: str,
    data_root: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    run = validate_completed_run(repo_root, receipt_path, expected_cell_id=cell_id)
    validation = load_frozen_validation_data(run, data_root=data_root)
    document = _build_result_document(run, validation)
    output_record = _write_exclusive(Path(output_path), document)
    return {"document": document, "output_record": output_record}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_completed_cell(
        repo_root=args.repo_root,
        receipt_path=args.receipt,
        cell_id=args.cell_id,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(json.dumps(result["output_record"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_ROLES",
    "GENERIC_EVALUATION_CONFIG_SHA256",
    "OCROutcomeEvaluationError",
    "SCHEMA_VERSION",
    "SOFT_COORDINATE_CHANNELS",
    "ValidatedOutcomeResult",
    "VALIDATION_FRAME_IDS",
    "canonical_sha256",
    "evaluate_completed_cell",
    "evaluation_config",
    "load_frozen_validation_data",
    "validate_completed_run",
    "validate_live_result",
]
