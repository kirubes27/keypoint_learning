"""Prepare or execute the staged paired OCR-ZNCC GPU experiment.

Preparation creates a new external manifest bound to the committed source and
exact evidence files.  Execution runs one matched control/OCR pair sequentially
on the same allocated device.  This module never submits a cluster job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import ocr_zncc_launch_validation as launch_validation


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_BINDING_SHA256 = "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
TRAIN_INDEX = Path(
    "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
    "roll__world_z__forward__train.json"
)
VALIDATION_INDEX = Path(
    "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
    "roll__world_z__forward__validation.json"
)
PRIMARY_MANIFEST = Path(
    "docs/decisions/2026-07-29/roll_head_package_training/"
    "EXPERIMENT_MANIFEST_v1.json"
)
PRIMARY_MANIFEST_FILE_SHA256 = "a754174e8692fdc21fe2629dcc50d3ee83f00ff521366495a4280cff03594899"
PRIMARY_MANIFEST_CONTENT_SHA256 = "2089c039f3dd7cab220e530834342188d282d8513a22fc395008c213a7913625"
GENERIC_EVALUATION_CONFIG_SHA256 = "151c973f56d123c085d7c1ab29fb6d720b69243915e06ad05fe1cb3590423e98"
OBJECT_NAME = "engineers_hammer_vray"
RECIPES = ("task55_clean", "task80_assisted")
SEEDS = (42, 43, 44)
EXPECTED_ENVIRONMENT_VERSIONS = {
    "python_implementation": "CPython",
    "python_version": "3.11.14",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "numpy": "2.2.6",
    "Pillow": "12.1.0",
    "pytorch_cuda": "12.1",
    "cudnn": 90100,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return authorization._file_record(path, name="bound experiment file")  # noqa: SLF001


def _committed_script_record(path: Path, *, relative: str) -> dict[str, Any]:
    expected = (REPO_ROOT / relative).resolve(strict=True)
    supplied = path.expanduser().resolve(strict=True)
    _require(supplied == expected, f"launch script must be committed {relative}")
    authorization._source_state(REPO_ROOT)  # noqa: SLF001
    return _record(expected)


def _environment_lock_record(path: Path) -> dict[str, Any]:
    record = _record(path)
    value = json.loads(Path(record["absolute_path"]).read_bytes())
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == "fresh_roll_cluster_environment.v1"
        and value.get("artifact_type") == "fresh_roll_pinned_cluster_environment"
        and value.get("versions") == EXPECTED_ENVIRONMENT_VERSIONS
        and isinstance(value.get("cuda_visible_during_setup"), bool)
        and value.get("scope")
        == "shared_unchanged_environment_for_cuda_smoke_and_frozen_matrix",
        "environment lock schema or pinned versions differ",
    )
    setup = value.get("setup_script")
    expected_setup = _record(REPO_ROOT / "cluster/setup_fresh_roll_env.sh")
    _require(
        isinstance(setup, dict) and setup.get("sha256") == expected_setup["sha256"],
        "environment setup-script hash differs from committed source",
    )
    freeze = value.get("pip_freeze")
    _require(isinstance(freeze, dict), "environment pip-freeze record is missing")
    live_freeze = _record(Path(str(freeze.get("absolute_path", ""))))
    _require(
        live_freeze["sha256"] == freeze.get("sha256"),
        "environment pip-freeze record differs",
    )
    return record


def _load_primary_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    path = (REPO_ROOT / PRIMARY_MANIFEST).resolve(strict=True)
    record = _record(path)
    _require(record["sha256"] == PRIMARY_MANIFEST_FILE_SHA256,
             "historical primary manifest file hash differs")
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict)
             and value.get("content_hash_sha256") == PRIMARY_MANIFEST_CONTENT_SHA256,
             "historical primary manifest content identity differs")
    payload = dict(value)
    payload.pop("content_hash_sha256")
    _require(authorization.canonical_sha256(payload) == PRIMARY_MANIFEST_CONTENT_SHA256,
             "historical primary manifest content hash differs")
    return value, record


def _arguments(
    *, recipe: str, seed: int, arm: str, coefficient: float,
    data_root: Path, output_root: Path, primary: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    _require(arm in {"control", "ocr_zncc"}, "invalid arm")
    fixed = primary["fixed_training"]
    recipe_values = primary["recipes"][recipe]
    head = primary["head_packages"]["64"]
    values = {
        "data_root": str(data_root),
        "object": primary["object"]["name"],
        "pairs_index": None,
        "indexed_mode": "development",
        "train_pairs_index": str((REPO_ROOT / TRAIN_INDEX).resolve(strict=True)),
        "val_pairs_index": str((REPO_ROOT / VALIDATION_INDEX).resolve(strict=True)),
        "test_pairs_index": None,
        "dataset_binding_sha256": primary["dataset"]["binding_sha256"],
        "img_size": fixed["img_size"],
        "num_keypoints": fixed["num_keypoints"],
        "base_channels": fixed["base_channels"],
        "heatmap_res": head["heatmap_res"],
        "temperature": fixed["temperature"],
        "padding_mode": fixed["padding_mode"],
        "operator_type": fixed["operator_type"],
        "lambda_attach": 0.0,
        "lambda_ocr_zncc": 0.0 if arm == "control" else coefficient,
        "ocr_peak_margin_exclusion_radius_cells": 1,
        "sigma": fixed["sigma"],
        "loc_bg_threshold": fixed["loc_bg_threshold"],
        "num_action_classes": fixed["num_action_classes"],
        "frame_skip": primary["transform"]["stride"],
        "yaw_step_deg": 2.0,
        "center_crop": fixed["center_crop"],
        "epochs": 1 if smoke else fixed["epochs"],
        "frozen_epochs": None,
        "batch_size": fixed["batch_size"],
        "lr": fixed["lr"],
        "weight_decay": fixed["weight_decay"],
        "seed": seed,
        "output_dir": str(output_root),
        "save_every": 1 if smoke else fixed["save_every"],
        "log_every": 1 if smoke else fixed["log_every"],
        "auto_eval": fixed["auto_eval"],
        "auto_eval_checkpoint": "best",
        "eval_frames_dir": None,
        "eval_max_k": 10,
        **recipe_values,
    }
    _require(set(values) == authorization.BOUND_ARGUMENTS,
             "constructed training argument set differs")
    return values


def prepare(
    *, matrix_root: Path, data_root: Path, authorization_decision: Path,
) -> Path:
    commit, branch = authorization._source_state(REPO_ROOT)
    resolved_data = data_root.expanduser().resolve(strict=True)
    root = matrix_root.expanduser().absolute()
    _require(not root.exists(), f"matrix root already exists: {root}")
    _require(root.parent.resolve(strict=True).is_dir(), "matrix parent is missing")
    destination = root / "OCR_ZNCC_PAIRED_TRAINING_MANIFEST.json"
    primary, primary_record = _load_primary_manifest()
    _require(primary["dataset"]["binding_sha256"] == DATASET_BINDING_SHA256,
             "historical dataset binding differs")
    _require(primary["split_bundle"]["train_pairs"]["file_sha256"]
             == authorization.TRAIN_INDEX_SHA256,
             "historical train split hash differs")
    _require(primary["split_bundle"]["validation_pairs"]["file_sha256"]
             == authorization.VALIDATION_INDEX_SHA256,
             "historical validation split hash differs")
    decision_record = _record(authorization_decision)
    decision = json.loads(Path(decision_record["absolute_path"]).read_bytes())
    _require(isinstance(decision, dict), "authorization decision is not an object")
    claimed = decision.get("content_hash_sha256")
    payload = dict(decision)
    payload.pop("content_hash_sha256", None)
    _require(authorization.canonical_sha256(payload) == claimed,
             "authorization decision content hash differs")
    _require(decision.get("source_commit") == commit
             and decision.get("source_branch") == branch,
             "authorization decision source differs")
    status_to_recipe = {
        "authorize_task55_matched_experiment": "task55_clean",
        "authorize_task80_matched_experiment": "task80_assisted",
    }
    recipe = status_to_recipe.get(decision.get("status"))
    _require(recipe in RECIPES, "authorization decision status is not executable")
    expected_schema = (
        "ocr_zncc_task55_training_authorization.v1"
        if recipe == "task55_clean"
        else "ocr_zncc_task80_training_authorization.v1"
    )
    _require(decision.get("schema_version") == expected_schema,
             "authorization decision schema differs")
    if recipe == "task55_clean":
        from keypoint_net import ocr_zncc_training_decision as decision_validator
    else:
        from keypoint_net import (
            ocr_zncc_task80_training_authorization as decision_validator,
        )
    deep_validation = decision_validator.validate_authorization_document(
        REPO_ROOT, decision
    )
    _require(
        isinstance(deep_validation, Mapping)
        and deep_validation.get("authorization_valid") is True
        and deep_validation.get("status") == decision.get("status")
        and deep_validation.get("source_commit") == commit
        and deep_validation.get("source_branch") == branch,
        "authorization decision failed deep live validation",
    )
    coefficient = float(decision.get("coefficient"))
    _require(0.0 < coefficient <= 0.5, "authorized coefficient is invalid")
    _require(
        float(deep_validation.get("coefficient")) == coefficient,
        "authorization coefficient differs from deep live validation",
    )
    train_record = _record(REPO_ROOT / TRAIN_INDEX)
    validation_record = _record(REPO_ROOT / VALIDATION_INDEX)
    cells = []
    for seed in SEEDS:
        for arm in ("control", "ocr_zncc"):
            cells.append({
                "cell_id": f"{recipe}__{arm}__seed{seed}",
                "recipe": recipe,
                "seed": seed,
                "arm": arm,
                "stage": "task55_primary" if recipe == "task55_clean"
                         else "task80_conditional",
                "training_arguments": _arguments(
                    recipe=recipe, seed=seed, arm=arm,
                    coefficient=coefficient, data_root=resolved_data,
                    output_root=root / "runs", primary=primary, smoke=False,
                ),
            })
    for arm in ("control", "ocr_zncc"):
        cells.append({
            "cell_id": f"{recipe}__{arm}__seed42__smoke",
            "recipe": recipe,
            "seed": 42,
            "arm": arm,
            "stage": "gpu_execution_smoke",
            "training_arguments": _arguments(
                recipe=recipe, seed=42, arm=arm,
                coefficient=coefficient, data_root=resolved_data,
                output_root=root / "smoke_runs", primary=primary, smoke=True,
            ),
        })
    document = {
        "schema_version": authorization.EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "single_stage_paired_ocr_zncc_training_manifest",
        "source_commit": commit,
        "source_branch": branch,
        "dataset": {
            "absolute_path": str(resolved_data),
            "binding_sha256": DATASET_BINDING_SHA256,
            "object": OBJECT_NAME,
            "train_index": train_record,
            "validation_index": validation_record,
        },
        "historical_recipe_source": {
            **primary_record,
            "content_hash_sha256": PRIMARY_MANIFEST_CONTENT_SHA256,
            "intentional_change": (
                "checkpoint selection is minimum base validation loss rather than "
                "the historical manifest's minimum total validation loss"
            ),
        },
        "authorization_decision": decision_record,
        "decision_lock": {
            "authorized_recipe": recipe,
            "checkpoint_selector": "minimum_base_validation_loss",
            "validation_aggregation": "sample_weighted_complete_21_pair_mean",
            "fixed_final_estimand": "epoch_1000_final_model",
            "test_split_used": False,
            "pair_execution": "control_then_ocr_sequential_on_same_allocated_gpu",
            "partial_failure_policy": "fail_closed_new_attempt_root_required_no_resume",
        },
        "primary_outcome": {
            "field": "canonical_drift.median_rms_objdiag",
            "coordinate_type": "soft_spatial_expectation",
            "population": "all_channels_0_through_9_on_validation_frames_0_through_23",
            "frame_count": 24,
            "pair_count": 21,
            "normalization": "per_frame_binary_mask_bbox_diagonal_on_endpoint_aligned_0_1_grid",
            "canonicalization": "inverse_known_world_z_roll_about_projected_center_0_0",
            "aggregation": "per_channel_rms_over_24_frames_then_median_over_10_channels",
            "generic_evaluation_config_sha256": GENERIC_EVALUATION_CONFIG_SHA256,
            "lower_is_better": True,
            "ocr_acceptance_or_channel_filtering_used": False,
        },
        "frozen_stage_criteria": {
            "selected_checkpoint_drift_ratio_at_most": 0.80,
            "selected_checkpoint_seed_passes_required": 2,
            "final_checkpoint_ocr_not_worse_seed_passes_required": 2,
            "ocr_correct_roll_sign_required_all_seeds": True,
            "ocr_angle_error_not_worse_seed_passes_required": 2,
            "active_on_object_count_not_lower_any_seed": True,
            "reflection_or_nonfinite_or_provenance_failure_blocks": True,
            "sample_inference": "descriptive_paired_seeds_not_population_inference",
        },
        "ocr_zncc": {
            "coefficient": coefficient,
            "peak_margin_exclusion_radius_cells": 1,
            "coefficient_is_single_shared_value": True,
            "raw_lambda_sweep_performed": False,
        },
        "cells": cells,
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    # No filesystem mutation occurs before every source/data/decision preflight above.
    os.mkdir(root, 0o755)
    os.mkdir(root / "runs", 0o755)
    os.mkdir(root / "logs", 0o755)
    os.mkdir(root / "smoke_runs", 0o755)
    with destination.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "manifest": str(destination),
        "file_sha256": _sha256_file(destination),
        "source_commit": commit,
        "source_branch": branch,
        "cell_count": len(cells),
    }, indent=2, sort_keys=True))
    return destination


def _load_manifest(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    absolute = path.expanduser().resolve(strict=True)
    _require(_sha256_file(absolute) == expected_sha256, "manifest file hash differs")
    value = json.loads(absolute.read_bytes())
    _require(isinstance(value, dict), "manifest is not an object")
    claimed = value.get("content_hash_sha256")
    payload = dict(value)
    payload.pop("content_hash_sha256", None)
    _require(authorization.canonical_sha256(payload) == claimed,
             "manifest content hash differs")
    return value


def _write_hashed_json(path: Path, document: dict[str, Any]) -> Path:
    _require(not path.exists(), f"refusing to overwrite {path}")
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    with path.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def prepare_smoke_lock(
    *, manifest_path: Path, manifest_sha256: str, environment_lock: Path,
    slurm_script: Path, output_path: Path,
) -> Path:
    commit, branch = authorization._source_state(REPO_ROOT)
    manifest = _load_manifest(manifest_path, manifest_sha256)
    _require(manifest.get("source_commit") == commit
             and manifest.get("source_branch") == branch,
             "smoke-lock manifest source differs")
    recipe = manifest.get("decision_lock", {}).get("authorized_recipe")
    _require(recipe in RECIPES, "GPU smoke manifest recipe is invalid")
    expected_smoke = {
        f"{recipe}__{arm}__seed42__smoke"
        for arm in ("control", "ocr_zncc")
    }
    actual_smoke = {cell["cell_id"] for cell in manifest["cells"]
                    if cell.get("stage") == "gpu_execution_smoke"}
    _require(actual_smoke == expected_smoke, "GPU smoke cell set differs")
    document = {
        "schema_version": "ocr_zncc_gpu_smoke_launch_lock.v1",
        "artifact_type": "ocr_zncc_gpu_smoke_launch_lock",
        "source_commit": commit,
        "source_branch": branch,
        "recipe": recipe,
        "manifest": {**_record(manifest_path),
                     "content_hash_sha256": manifest["content_hash_sha256"]},
        "environment_lock": _environment_lock_record(environment_lock),
        "slurm_script": _committed_script_record(
            slurm_script, relative="cluster/ocr_zncc_gpu_smoke.slurm"
        ),
        "cells": sorted(expected_smoke),
        "scientific_result_authorized": False,
    }
    return _write_hashed_json(output_path, document)


def prepare_stage_lock(
    *, manifest_path: Path, manifest_sha256: str, smoke_receipt: Path,
    environment_lock: Path, slurm_script: Path, output_path: Path,
) -> Path:
    commit, branch = authorization._source_state(REPO_ROOT)
    manifest = _load_manifest(manifest_path, manifest_sha256)
    _require(manifest.get("source_commit") == commit
             and manifest.get("source_branch") == branch,
             "stage-lock manifest source differs")
    recipe = manifest.get("decision_lock", {}).get("authorized_recipe")
    _require(recipe in RECIPES, "stage recipe is invalid")
    smoke_record = _record(smoke_receipt)
    smoke = json.loads(Path(smoke_record["absolute_path"]).read_bytes())
    _require(isinstance(smoke, dict)
             and smoke.get("schema_version") == "ocr_zncc_gpu_smoke.v2"
             and smoke.get("status") == "pass"
             and smoke.get("scientific_result_authorized") is False,
             "GPU smoke receipt did not pass")
    smoke_payload = dict(smoke)
    smoke_claimed = smoke_payload.pop("content_hash_sha256", None)
    _require(authorization.canonical_sha256(smoke_payload) == smoke_claimed,
             "GPU smoke receipt content hash differs")
    _require(smoke.get("source_commit") == commit
             and smoke.get("source_branch") == branch
             and smoke.get("recipe") == recipe
             and smoke.get("manifest", {}).get("file_sha256") == manifest_sha256
             and smoke.get("manifest", {}).get("content_hash_sha256")
             == manifest.get("content_hash_sha256"),
             "GPU smoke source/manifest differs")
    expected_smoke_order = [
        f"{recipe}__control__seed42__smoke",
        f"{recipe}__ocr_zncc__seed42__smoke",
    ]
    checks = smoke.get("checks", {})
    production = smoke.get("production_path", {})
    production_cells = production.get("cells", [])
    _require(
        checks.get("identical_initial_state") is True
        and checks.get("identical_base_forward") is True
        and checks.get("accepted_source_patches_inside") is True
        and isinstance(checks.get("auxiliary_extractor_gradient_norm"), (int, float))
        and not isinstance(checks.get("auxiliary_extractor_gradient_norm"), bool)
        and math.isfinite(float(checks["auxiliary_extractor_gradient_norm"]))
        and float(checks["auxiliary_extractor_gradient_norm"]) > 0.0
        and checks.get("auxiliary_forbidden_gradient_norm") == 0.0
        and production.get("execution_order") == expected_smoke_order
        and production.get("cell_count") == 2
        and isinstance(production_cells, list)
        and [cell.get("cell_id") for cell in production_cells] == expected_smoke_order
        and production_cells[0].get("matched_pairing_evidence")
        == production_cells[1].get("matched_pairing_evidence")
        and [cell.get("pair_position") for cell in production_cells]
        == ["1_control", "2_ocr_zncc"]
        and all(
            cell.get("optimizer_step_count") == 10
            and cell.get("final_epoch") == 1
            and isinstance(cell.get("completed_run_receipt"), dict)
            and isinstance(cell.get("matched_pairing_evidence"), dict)
            for cell in production_cells
        ),
        "GPU smoke semantic or production-path checks differ",
    )
    expected_cells = {
        f"{recipe}__{arm}__seed{seed}"
        for seed in SEEDS for arm in ("control", "ocr_zncc")
    }
    actual_cells = {cell["cell_id"] for cell in manifest["cells"]
                    if cell.get("stage") != "gpu_execution_smoke"}
    _require(actual_cells == expected_cells, "stage scientific cell set differs")
    document = {
        "schema_version": "ocr_zncc_paired_stage_launch_lock.v1",
        "artifact_type": "ocr_zncc_paired_stage_launch_lock",
        "source_commit": commit,
        "source_branch": branch,
        "recipe": recipe,
        "manifest": {**_record(manifest_path),
                     "content_hash_sha256": manifest["content_hash_sha256"]},
        "gpu_smoke_receipt": smoke_record,
        "environment_lock": _environment_lock_record(environment_lock),
        "slurm_script": _committed_script_record(
            slurm_script, relative="cluster/ocr_zncc_paired_stage.slurm"
        ),
        "pair_order": [
            {"seed": seed, "first": f"{recipe}__control__seed{seed}",
             "second": f"{recipe}__ocr_zncc__seed{seed}"}
            for seed in SEEDS
        ],
        "partial_failure_policy": "new_attempt_root_required_no_resume",
    }
    return _write_hashed_json(output_path, document)


def _load_stage_lock(
    path: Path, expected_sha256: str, *, manifest_path: Path,
    manifest_sha256: str,
    recipe: str,
) -> Mapping[str, Any]:
    commit, branch = authorization._source_state(REPO_ROOT)
    record = _record(path)
    _require(record["sha256"] == expected_sha256, "stage-lock file hash differs")
    value = json.loads(Path(record["absolute_path"]).read_bytes())
    _require(isinstance(value, dict), "stage lock is not an object")
    payload = dict(value)
    claimed = payload.pop("content_hash_sha256", None)
    _require(authorization.canonical_sha256(payload) == claimed,
             "stage-lock content hash differs")
    _require(value.get("schema_version") == "ocr_zncc_paired_stage_launch_lock.v1"
             and value.get("recipe") == recipe
             and value.get("source_commit") == commit
             and value.get("source_branch") == branch
             and value.get("manifest", {}).get("sha256") == manifest_sha256,
             "stage-lock identity differs")
    _require(os.environ.get("OCR_STAGE_LAUNCH_LOCK") == record["absolute_path"]
             and os.environ.get("OCR_STAGE_LAUNCH_LOCK_SHA256") == expected_sha256,
             "live stage-lock environment binding differs")
    _require(bool(os.environ.get("SLURM_JOB_ID")), "paired run requires SLURM_JOB_ID")
    runtime_bindings = {
        "environment_lock": ("OCR_ENV_LOCK", "OCR_ENV_LOCK_SHA256"),
        "slurm_script": ("OCR_SLURM_SCRIPT", "OCR_SLURM_SCRIPT_SHA256"),
    }
    for field, (path_variable, hash_variable) in runtime_bindings.items():
        expected = value.get(field)
        _require(isinstance(expected, dict), f"stage-lock {field} is missing")
        _require(
            os.environ.get(path_variable) == expected.get("absolute_path")
            and os.environ.get(hash_variable) == expected.get("sha256"),
            f"live {field} environment binding differs from stage lock",
        )
    deep = launch_validation.validate_paired_stage_launch_from_environment(
        repo_root=REPO_ROOT,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage_lock_path=Path(record["absolute_path"]),
        stage_lock_sha256=expected_sha256,
        expected_recipe=recipe,
    )
    _require(
        deep.get("authorization_valid") is True
        and deep.get("recipe") == recipe
        and deep.get("source_commit") == commit
        and deep.get("source_branch") == branch,
        "stage lock failed deep completed-smoke validation",
    )
    return value


def _cell_command(
    *, cell: Mapping[str, Any], manifest_path: Path, manifest_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str((REPO_ROOT / "keypoint_net/train.py").resolve(strict=True)),
        *authorization.command_arguments(cell["training_arguments"]),
        "--ocr_cell_id", str(cell["cell_id"]),
        "--ocr_experiment_manifest", str(manifest_path.resolve(strict=True)),
        "--ocr_experiment_manifest_sha256", manifest_sha256,
    ]


def run_pair(
    *, manifest_path: Path, manifest_sha256: str, recipe: str, seed: int,
    stage_lock_path: Path, stage_lock_sha256: str, print_only: bool,
) -> None:
    _require(recipe in RECIPES and seed in SEEDS, "pair identity is invalid")
    manifest = _load_manifest(manifest_path, manifest_sha256)
    _require(manifest.get("decision_lock", {}).get("authorized_recipe") == recipe,
             "manifest does not authorize requested recipe")
    _load_stage_lock(
        stage_lock_path, stage_lock_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256, recipe=recipe,
    )
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    pair_ids = [f"{recipe}__{arm}__seed{seed}"
                for arm in ("control", "ocr_zncc")]
    _require(all(cell_id in cells for cell_id in pair_ids), "paired cells are missing")
    run_directories = [
        Path(cells[cell_id]["training_arguments"]["output_dir"]) / cell_id
        for cell_id in pair_ids
    ]
    _require(not any(path.exists() for path in run_directories),
             "paired run is non-resumable; a new attempt root is required")
    pair_id = f"{recipe}__seed{seed}"
    for position, arm in enumerate(("control", "ocr_zncc"), start=1):
        cell_id = f"{recipe}__{arm}__seed{seed}"
        cell = cells[cell_id]
        command = _cell_command(
            cell=cell, manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        if print_only:
            print(" ".join(command))
        else:
            child_environment = os.environ.copy()
            child_environment.update({
                "OCR_PAIR_ID": pair_id,
                "OCR_PAIR_POSITION": (
                    "1_control" if position == 1 else "2_ocr_zncc"
                ),
                "OCR_PAIR_WRAPPER_PID": str(os.getpid()),
            })
            subprocess.run(command, check=True, env=child_environment)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--matrix-root", type=Path, required=True)
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--authorization-decision", type=Path, required=True)
    smoke_lock_parser = subparsers.add_parser("prepare-smoke-lock")
    smoke_lock_parser.add_argument("--manifest", type=Path, required=True)
    smoke_lock_parser.add_argument("--manifest-sha256", required=True)
    smoke_lock_parser.add_argument("--environment-lock", type=Path, required=True)
    smoke_lock_parser.add_argument("--slurm-script", type=Path, required=True)
    smoke_lock_parser.add_argument("--output", type=Path, required=True)
    stage_lock_parser = subparsers.add_parser("prepare-stage-lock")
    stage_lock_parser.add_argument("--manifest", type=Path, required=True)
    stage_lock_parser.add_argument("--manifest-sha256", required=True)
    stage_lock_parser.add_argument("--smoke-receipt", type=Path, required=True)
    stage_lock_parser.add_argument("--environment-lock", type=Path, required=True)
    stage_lock_parser.add_argument("--slurm-script", type=Path, required=True)
    stage_lock_parser.add_argument("--output", type=Path, required=True)
    pair_parser = subparsers.add_parser("run-pair")
    pair_parser.add_argument("--manifest", type=Path, required=True)
    pair_parser.add_argument("--manifest-sha256", required=True)
    pair_parser.add_argument("--recipe", choices=RECIPES, required=True)
    pair_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    pair_parser.add_argument("--stage-launch-lock", type=Path, required=True)
    pair_parser.add_argument("--stage-launch-lock-sha256", required=True)
    pair_parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            matrix_root=args.matrix_root, data_root=args.data_root,
            authorization_decision=args.authorization_decision,
        )
    elif args.command == "prepare-smoke-lock":
        prepare_smoke_lock(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            environment_lock=args.environment_lock,
            slurm_script=args.slurm_script,
            output_path=args.output,
        )
    elif args.command == "prepare-stage-lock":
        prepare_stage_lock(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            smoke_receipt=args.smoke_receipt,
            environment_lock=args.environment_lock,
            slurm_script=args.slurm_script,
            output_path=args.output,
        )
    else:
        run_pair(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            recipe=args.recipe,
            seed=args.seed,
            stage_lock_path=args.stage_launch_lock,
            stage_lock_sha256=args.stage_launch_lock_sha256,
            print_only=args.print_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
