"""Shared deep launch-lineage validation for scientific OCR-ZNCC pairs.

The paired Slurm entrypoint, the pair wrapper, and ``train.py`` all call this
module before a scientific run directory can be claimed.  A stage-lock JSON is
therefore not accepted merely because it is self-consistent: the validator
reopens the completed GPU smoke, its launch lock, both production-path smoke
receipts, every receipt-bound artifact, and the pinned runtime environment.

Smoke-cell execution is intentionally outside this validator because the
completed smoke receipt does not exist until those two cells finish.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from keypoint_net import ocr_zncc_training_authorization as authorization
except ModuleNotFoundError:  # Direct execution through keypoint_net/train.py.
    import ocr_zncc_training_authorization as authorization


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
RECIPES = ("task55_clean", "task80_assisted")
SEEDS = (42, 43, 44)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OCRLaunchValidationError(ValueError):
    """Raised when a scientific launch lacks exact completed-smoke lineage."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRLaunchValidationError(message)


def _record(
    path_value: Path | str, *, name: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = Path(path_value)
    _require(path.is_absolute(), f"{name} path is not absolute")
    try:
        observed = authorization._file_record(path, name=name)  # noqa: SLF001
    except Exception as exc:
        raise OCRLaunchValidationError(f"cannot validate {name}: {exc}") from exc
    if expected_sha256 is not None:
        _require(
            _SHA256_RE.fullmatch(expected_sha256) is not None
            and observed["sha256"] == expected_sha256,
            f"{name} hash differs",
        )
    return observed


def _load_hashed_json(
    path_value: Path | str,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path_value, name=name, expected_sha256=expected_sha256)
    try:
        value = json.loads(Path(record["absolute_path"]).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCRLaunchValidationError(f"{name} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    try:
        content_hash = authorization._validate_content_hash(  # noqa: SLF001
            value, name=name
        )
    except Exception as exc:
        raise OCRLaunchValidationError(f"cannot validate {name}: {exc}") from exc
    return value, {**record, "content_hash_sha256": content_hash}


def _expected_cell_command(
    *,
    repo_root: Path,
    cell: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str((repo_root / "keypoint_net/train.py").resolve(strict=True)),
        *authorization.command_arguments(cell["training_arguments"]),
        "--ocr_cell_id",
        str(cell["cell_id"]),
        "--ocr_experiment_manifest",
        str(manifest_path),
        "--ocr_experiment_manifest_sha256",
        manifest_sha256,
    ]


def _validate_live_environment(environment_record: Mapping[str, Any]) -> None:
    lock_path = Path(str(environment_record.get("absolute_path", "")))
    lock, observed = _load_hashed_json_without_content(
        lock_path, name="cluster environment lock"
    )
    _require(observed == dict(environment_record), "environment-lock file record differs")
    _require(
        lock.get("schema_version") == "fresh_roll_cluster_environment.v1"
        and lock.get("artifact_type") == "fresh_roll_pinned_cluster_environment"
        and lock.get("scope")
        == "shared_unchanged_environment_for_cuda_smoke_and_frozen_matrix"
        and lock.get("versions") == EXPECTED_ENVIRONMENT_VERSIONS,
        "environment-lock semantics or versions differ",
    )
    live = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": str(np.__version__),
        "Pillow": importlib.metadata.version("Pillow"),
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    _require(live == lock["versions"], f"live environment versions differ: {live!r}")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.encode()
    freeze_record = lock.get("pip_freeze")
    _require(
        isinstance(freeze_record, Mapping)
        and hashlib.sha256(freeze).hexdigest() == freeze_record.get("sha256"),
        "live pip freeze differs from environment lock",
    )


def _load_hashed_json_without_content(
    path_value: Path | str, *, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path_value, name=name)
    try:
        value = json.loads(Path(record["absolute_path"]).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCRLaunchValidationError(f"{name} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value, record


def validate_paired_stage_launch(
    *,
    repo_root: Path | str,
    expected_commit: str,
    manifest_path: Path | str,
    manifest_sha256: str,
    stage_lock_path: Path | str,
    stage_lock_sha256: str,
    environment_lock_path: Path | str,
    environment_lock_sha256: str,
    slurm_script_path: Path | str,
    slurm_script_sha256: str,
    expected_recipe: str | None = None,
) -> dict[str, Any]:
    """Deeply validate one stage lock and all completed-smoke evidence."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    _require(bool(os.environ.get("SLURM_JOB_ID")), "scientific OCR launch lacks SLURM_JOB_ID")
    try:
        live_commit, live_branch = authorization._source_state(root)  # noqa: SLF001
    except Exception as exc:
        raise OCRLaunchValidationError(f"source validation failed: {exc}") from exc
    _require(
        live_commit == expected_commit
        and live_branch == authorization.EXPECTED_BRANCH,
        "scientific launch source identity differs",
    )

    manifest_absolute = Path(manifest_path).expanduser().absolute()
    stage_absolute = Path(stage_lock_path).expanduser().absolute()
    environment_absolute = Path(environment_lock_path).expanduser().absolute()
    script_absolute = Path(slurm_script_path).expanduser().absolute()
    _require(
        script_absolute == (root / "cluster/ocr_zncc_paired_stage.slurm").absolute(),
        "scientific launch Slurm path is not the committed paired-stage script",
    )
    manifest, manifest_record_with_content = _load_hashed_json(
        manifest_absolute,
        name="paired-stage manifest",
        expected_sha256=manifest_sha256,
    )
    manifest_record = {
        key: manifest_record_with_content[key]
        for key in ("absolute_path", "sha256", "size_bytes")
    }
    stage, stage_record_with_content = _load_hashed_json(
        stage_absolute,
        name="paired-stage launch lock",
        expected_sha256=stage_lock_sha256,
    )
    stage_record = {
        key: stage_record_with_content[key]
        for key in ("absolute_path", "sha256", "size_bytes")
    }
    environment_record = _record(
        environment_absolute,
        name="cluster environment lock",
        expected_sha256=environment_lock_sha256,
    )
    script_record = _record(
        script_absolute,
        name="paired-stage Slurm script",
        expected_sha256=slurm_script_sha256,
    )

    recipe = stage.get("recipe")
    _require(recipe in RECIPES, "paired-stage recipe is invalid")
    if expected_recipe is not None:
        _require(recipe == expected_recipe, "paired-stage recipe differs from requested recipe")
    expected_pair_order = [
        {
            "seed": seed,
            "first": f"{recipe}__control__seed{seed}",
            "second": f"{recipe}__ocr_zncc__seed{seed}",
        }
        for seed in SEEDS
    ]
    _require(
        stage.get("schema_version") == "ocr_zncc_paired_stage_launch_lock.v1"
        and stage.get("artifact_type") == "ocr_zncc_paired_stage_launch_lock"
        and stage.get("source_commit") == expected_commit
        and stage.get("source_branch") == authorization.EXPECTED_BRANCH
        and stage.get("manifest")
        == {**manifest_record, "content_hash_sha256": manifest["content_hash_sha256"]}
        and stage.get("environment_lock") == environment_record
        and stage.get("slurm_script") == script_record
        and stage.get("pair_order") == expected_pair_order
        and stage.get("partial_failure_policy")
        == "new_attempt_root_required_no_resume",
        "paired-stage launch-lock semantics differ",
    )

    expected_smoke_order = [
        f"{recipe}__control__seed42__smoke",
        f"{recipe}__ocr_zncc__seed42__smoke",
    ]
    expected_scientific_cells = {
        f"{recipe}__{arm}__seed{seed}"
        for seed in SEEDS
        for arm in ("control", "ocr_zncc")
    }
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "paired-stage manifest cells are invalid")
    manifest_by_id = {
        cell["cell_id"]: cell
        for cell in cells
        if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
    }
    scientific_ids = [
        cell.get("cell_id")
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("stage") != "gpu_execution_smoke"
    ]
    smoke_ids = [
        cell.get("cell_id")
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("stage") == "gpu_execution_smoke"
    ]
    _require(
        manifest.get("source_commit") == expected_commit
        and manifest.get("source_branch") == authorization.EXPECTED_BRANCH
        and manifest.get("decision_lock", {}).get("authorized_recipe") == recipe
        and len(manifest_by_id) == len(cells)
        and len(scientific_ids) == len(set(scientific_ids)) == 6
        and set(scientific_ids) == expected_scientific_cells
        and smoke_ids == expected_smoke_order,
        "paired-stage manifest authorization differs",
    )

    smoke_claim = stage.get("gpu_smoke_receipt")
    _require(isinstance(smoke_claim, Mapping), "paired-stage smoke receipt is missing")
    smoke_path = Path(str(smoke_claim.get("absolute_path", "")))
    smoke, smoke_record_with_content = _load_hashed_json(
        smoke_path,
        name="GPU smoke receipt",
        expected_sha256=str(smoke_claim.get("sha256", "")),
    )
    smoke_record = {
        key: smoke_record_with_content[key]
        for key in ("absolute_path", "sha256", "size_bytes")
    }
    _require(dict(smoke_claim) == smoke_record, "paired-stage smoke file record differs")
    smoke_checks = smoke.get("checks")
    one_step = smoke.get("one_step")
    production = smoke.get("production_path")
    _require(isinstance(smoke_checks, Mapping), "GPU smoke checks are invalid")
    _require(isinstance(one_step, Mapping), "GPU smoke one-step evidence is invalid")
    _require(isinstance(production, Mapping), "GPU smoke production evidence is invalid")
    production_cells = production.get("cells")
    smoke_job_id = smoke.get("cuda", {}).get("slurm_job_id")
    auxiliary_norm = smoke_checks.get("auxiliary_extractor_gradient_norm")
    forbidden_norm = smoke_checks.get("auxiliary_forbidden_gradient_norm")
    _require(
        smoke.get("schema_version") == "ocr_zncc_gpu_smoke.v2"
        and smoke.get("artifact_type") == "ocr_zncc_gpu_execution_smoke"
        and smoke.get("status") == "pass"
        and smoke.get("scientific_result_authorized") is False
        and smoke.get("source_commit") == expected_commit
        and smoke.get("source_branch") == authorization.EXPECTED_BRANCH
        and smoke.get("recipe") == recipe
        and isinstance(smoke_job_id, str)
        and bool(smoke_job_id)
        and smoke.get("manifest")
        == {
            "absolute_path": manifest_record["absolute_path"],
            "file_sha256": manifest_record["sha256"],
            "content_hash_sha256": manifest["content_hash_sha256"],
        }
        and smoke_checks.get("identical_initial_state") is True
        and smoke_checks.get("identical_base_forward") is True
        and smoke_checks.get("accepted_source_patches_inside") is True
        and isinstance(auxiliary_norm, (int, float))
        and not isinstance(auxiliary_norm, bool)
        and math.isfinite(float(auxiliary_norm))
        and float(auxiliary_norm) > 0.0
        and forbidden_norm == 0.0
        and set(one_step) == {"control", "ocr_zncc"}
        and all(
            isinstance(step, Mapping)
            and step.get("optimizer_steps") == 1
            and step.get("model_state_finite") is True
            and isinstance(step.get("optimization_loss"), (int, float))
            and not isinstance(step.get("optimization_loss"), bool)
            and math.isfinite(float(step["optimization_loss"]))
            for step in one_step.values()
        )
        and one_step.get("ocr_zncc", {}).get("accepted_match_count", 0) > 0
        and production.get("entrypoint")
        == str((root / "keypoint_net/train.py").resolve(strict=True))
        and production.get("execution_order") == expected_smoke_order
        and production.get("cell_count") == 2
        and isinstance(production_cells, list)
        and [item.get("cell_id") for item in production_cells]
        == expected_smoke_order,
        "paired-stage GPU smoke evidence differs",
    )

    runtime_provenance = smoke.get("runtime_provenance")
    _require(
        isinstance(runtime_provenance, Mapping)
        and runtime_provenance.get("environment_lock") == environment_record,
        "GPU smoke runtime provenance differs",
    )
    smoke_script_record = _record(
        (root / "cluster/ocr_zncc_gpu_smoke.slurm").absolute(),
        name="GPU smoke Slurm script",
    )
    _require(
        runtime_provenance.get("slurm_script") == smoke_script_record,
        "GPU smoke Slurm provenance differs",
    )
    smoke_lock_claim = runtime_provenance.get("smoke_launch_lock")
    _require(isinstance(smoke_lock_claim, Mapping), "GPU smoke launch lock is missing")
    smoke_lock_path = Path(str(smoke_lock_claim.get("absolute_path", "")))
    smoke_lock, smoke_lock_record_with_content = _load_hashed_json(
        smoke_lock_path,
        name="GPU smoke launch lock",
        expected_sha256=str(smoke_lock_claim.get("sha256", "")),
    )
    expected_smoke_lock_claim = {
        **smoke_lock_record_with_content,
        "schema_version": "ocr_zncc_gpu_smoke_launch_lock.v1",
    }
    _require(
        dict(smoke_lock_claim) == expected_smoke_lock_claim,
        "GPU smoke launch-lock file record differs",
    )
    _require(
        smoke_lock.get("schema_version") == "ocr_zncc_gpu_smoke_launch_lock.v1"
        and smoke_lock.get("artifact_type") == "ocr_zncc_gpu_smoke_launch_lock"
        and smoke_lock.get("source_commit") == expected_commit
        and smoke_lock.get("source_branch") == authorization.EXPECTED_BRANCH
        and smoke_lock.get("recipe") == recipe
        and smoke_lock.get("manifest")
        == {**manifest_record, "content_hash_sha256": manifest["content_hash_sha256"]}
        and smoke_lock.get("environment_lock") == environment_record
        and smoke_lock.get("slurm_script") == smoke_script_record
        and smoke_lock.get("cells") == expected_smoke_order
        and smoke_lock.get("scientific_result_authorized") is False,
        "GPU smoke launch-lock semantics differ",
    )

    production_pairing: dict[str, Mapping[str, Any]] = {}
    for summary in production_cells:
        _require(isinstance(summary, Mapping), "GPU smoke cell summary is invalid")
        cell_id = str(summary.get("cell_id", ""))
        cell = manifest_by_id.get(cell_id)
        _require(
            isinstance(cell, Mapping)
            and cell_id in expected_smoke_order
            and cell.get("stage") == "gpu_execution_smoke"
            and cell.get("recipe") == recipe
            and cell.get("seed") == 42,
            f"{cell_id} smoke manifest cell differs",
        )
        receipt_claim = summary.get("completed_run_receipt")
        _require(
            isinstance(receipt_claim, Mapping),
            f"{cell_id} completed-run receipt summary is missing",
        )
        receipt_path = Path(str(receipt_claim.get("absolute_path", "")))
        receipt, receipt_record_with_content = _load_hashed_json(
            receipt_path,
            name=f"{cell_id} completed-run receipt",
            expected_sha256=str(receipt_claim.get("sha256", "")),
        )
        _require(
            dict(receipt_claim) == receipt_record_with_content,
            f"{cell_id} completed-run receipt record differs",
        )
        run_directory = (
            Path(str(cell["training_arguments"]["output_dir"])) / cell_id
        ).resolve(strict=True)
        execution = receipt.get("execution")
        _require(isinstance(execution, Mapping), f"{cell_id} execution is invalid")
        runtime = execution.get("runtime_environment")
        _require(isinstance(runtime, Mapping), f"{cell_id} runtime is invalid")
        runtime_pair = runtime.get("ocr_pair_execution")
        _require(isinstance(runtime_pair, Mapping), f"{cell_id} runtime pair is invalid")
        expected_arm = str(cell.get("arm"))
        expected_position = "1_control" if expected_arm == "control" else "2_ocr_zncc"
        expected_command = _expected_cell_command(
            repo_root=root,
            cell=cell,
            manifest_path=manifest_absolute,
            manifest_sha256=manifest_sha256,
        )
        _require(
            receipt.get("schema_version") == authorization.COMPLETED_RUN_SCHEMA_VERSION
            and receipt.get("artifact_type") == "ocr_zncc_completed_run_receipt"
            and receipt.get("cell_id") == cell_id
            and receipt.get("recipe") == recipe
            and receipt.get("arm") == expected_arm
            and receipt.get("source_commit") == expected_commit
            and receipt.get("source_branch") == authorization.EXPECTED_BRANCH
            and receipt.get("manifest")
            == {
                "absolute_path": manifest_record["absolute_path"],
                "file_sha256": manifest_record["sha256"],
                "content_hash_sha256": manifest["content_hash_sha256"],
            }
            and receipt.get("expected_training_arguments")
            == cell.get("training_arguments")
            and receipt.get("run_directory") == str(run_directory)
            and execution.get("device") == "cuda"
            and execution.get("optimizer_step_count") == 10
            and execution.get("scheduler_step_count") == 1
            and execution.get("full_command") == expected_command
            and runtime.get("slurm_job_id") == smoke_job_id
            and runtime.get("environment_lock") == environment_record
            and runtime.get("slurm_job_script") == smoke_script_record
            and runtime.get("ocr_stage_launch_lock")
            == {
                key: smoke_lock_record_with_content[key]
                for key in ("absolute_path", "sha256", "size_bytes")
            }
            and runtime_pair.get("pair_id") == f"{recipe}__seed42__smoke"
            and runtime_pair.get("position") == expected_position
            and runtime_pair.get("slurm_job_id") == smoke_job_id
            and runtime_pair.get("allocated_gpu_count") == 1
            and summary.get("runtime_environment") == runtime
            and summary.get("pair_position") == expected_position
            and receipt.get("checkpoint_selection")
            == "minimum_base_validation_loss"
            and receipt.get("checkpoint_validation", {}).get("final_epoch") == 1,
            f"{cell_id} completed-run semantics differ",
        )
        elapsed = summary.get("elapsed_seconds")
        peak = summary.get("cuda_memory", {}).get("observed_peak_mib")
        _require(
            isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(float(elapsed))
            and float(elapsed) > 0.0
            and isinstance(peak, (int, float))
            and not isinstance(peak, bool)
            and math.isfinite(float(peak))
            and float(peak) > 0.0,
            f"{cell_id} elapsed-time/CUDA-memory evidence differs",
        )

        artifacts = receipt.get("artifacts")
        artifact_names = {
            "config.json",
            "history.json",
            "sampler_order.json",
            "best_model.pt",
            "final_model.pt",
        }
        _require(
            isinstance(artifacts, Mapping) and set(artifacts) == artifact_names,
            f"{cell_id} completion artifact set differs",
        )
        for artifact_name in sorted(artifact_names):
            _require(
                artifacts[artifact_name]
                == _record(
                    (run_directory / artifact_name).absolute(),
                    name=f"{cell_id}/{artifact_name}",
                ),
                f"{cell_id}/{artifact_name} record differs",
            )

        config, _ = _load_hashed_json_without_content(
            run_directory / "config.json", name=f"{cell_id} config"
        )
        history_data = json.loads((run_directory / "history.json").read_bytes())
        sampler, _ = _load_hashed_json_without_content(
            run_directory / "sampler_order.json", name=f"{cell_id} sampler order"
        )
        binding = authorization.OCRTrainingBinding(
            cell_id=cell_id,
            recipe=str(recipe),
            arm=expected_arm,
            run_directory=str(run_directory),
            manifest_absolute_path=manifest_record["absolute_path"],
            manifest_file_sha256=manifest_record["sha256"],
            manifest_content_sha256=manifest["content_hash_sha256"],
            expected_training_arguments=dict(cell["training_arguments"]),
        )
        try:
            validation_rows = authorization._validate_completed_validation_history(  # noqa: SLF001
                history_data, binding
            )
        except Exception as exc:
            raise OCRLaunchValidationError(
                f"{cell_id} completed validation history differs: {exc}"
            ) from exc
        epoch_orders = sampler.get("epochs")
        _require(
            config.get("cell_id") == cell_id
            and config.get("source_commit") == expected_commit
            and config.get("checkpoint_policy", {}).get("selection")
            == "minimum_base_validation_loss"
            and sampler.get("schema_version") == "ocr_zncc_sampler_order.v1"
            and isinstance(epoch_orders, list)
            and len(epoch_orders) == 1
            and epoch_orders[0].get("epoch") == 1
            and epoch_orders[0].get("sample_count") == 147
            and isinstance(epoch_orders[0].get("pair_order_sha256"), str)
            and _SHA256_RE.fullmatch(epoch_orders[0]["pair_order_sha256"]) is not None,
            f"{cell_id} config/history/sampler semantics differ",
        )
        pairing = {
            "initial_model_state_sha256": config.get("initial_model_state_sha256"),
            "post_diagnostic_model_state_sha256": config.get(
                "post_diagnostic_model_state_sha256"
            ),
            "diagnostic_pair_order_sha256": config.get("diagnostic_pair_order_sha256"),
            "diagnostic_pair_order_sample_count": config.get(
                "diagnostic_pair_order_sample_count"
            ),
            "train_sampler_seed": sampler.get("generator_seed"),
            "epoch_pair_orders": epoch_orders,
            "device": execution.get("device"),
            "pair_execution_common": {
                key: runtime_pair.get(key)
                for key in (
                    "pair_id",
                    "wrapper_pid",
                    "slurm_job_id",
                    "cuda_visible_devices",
                    "allocated_gpu_count",
                )
            },
        }
        _require(
            summary.get("matched_pairing_evidence") == pairing
            and pairing["diagnostic_pair_order_sample_count"] == 16
            and pairing["train_sampler_seed"]
            == int(cell["training_arguments"]["seed"]) + 10_000_019,
            f"{cell_id} production pairing evidence differs",
        )
        first_minimum = min(
            validation_rows,
            key=lambda row: (float(row["val_base_loss"]), int(row["epoch"])),
        )
        try:
            best = torch.load(
                run_directory / "best_model.pt", map_location="cpu", weights_only=True
            )
            final = torch.load(
                run_directory / "final_model.pt", map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise OCRLaunchValidationError(f"{cell_id} checkpoint load failed") from exc
        _require(
            isinstance(best, Mapping)
            and isinstance(final, Mapping)
            and all(
                isinstance(payload.get("model_state_dict"), Mapping)
                and bool(payload["model_state_dict"])
                and all(
                    torch.is_tensor(value) and bool(torch.isfinite(value).all())
                    for value in payload["model_state_dict"].values()
                )
                for payload in (best, final)
            )
            and best.get("selection_loss_name") == "base_validation_loss"
            and best.get("epoch") == first_minimum["epoch"]
            and float(best.get("base_validation_loss"))
            == float(first_minimum["val_base_loss"])
            and final.get("epoch") == 1
            and best.get("config", {}).get("cell_id") == cell_id
            and final.get("config", {}).get("cell_id") == cell_id
            and best.get("config", {}).get("source_commit") == expected_commit
            and final.get("config", {}).get("source_commit") == expected_commit,
            f"{cell_id} checkpoint semantics differ",
        )
        source_files = receipt.get("source_files")
        _require(
            isinstance(source_files, Mapping)
            and set(source_files) == set(authorization.RUN_SOURCE_PATHS),
            f"{cell_id} source-file set differs",
        )
        for relative in authorization.RUN_SOURCE_PATHS:
            _require(
                source_files[relative]
                == _record((root / relative).absolute(), name=f"source {relative}"),
                f"{cell_id} source {relative} differs",
            )
        production_pairing[expected_arm] = pairing

    _require(
        set(production_pairing) == {"control", "ocr_zncc"}
        and production_pairing["control"] == production_pairing["ocr_zncc"],
        "production smoke control/OCR pairing evidence differs",
    )
    _validate_live_environment(environment_record)
    return {
        "authorization_valid": True,
        "recipe": recipe,
        "source_commit": expected_commit,
        "source_branch": authorization.EXPECTED_BRANCH,
        "manifest": {
            **manifest_record,
            "content_hash_sha256": manifest["content_hash_sha256"],
        },
        "stage_launch_lock": stage_record_with_content,
        "gpu_smoke_receipt": smoke_record_with_content,
        "environment_lock": environment_record,
        "slurm_script": script_record,
        "scientific_training_authorized": True,
    }


def validate_paired_stage_launch_from_environment(
    *,
    repo_root: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    stage_lock_path: Path | str,
    stage_lock_sha256: str,
    expected_recipe: str | None = None,
) -> dict[str, Any]:
    """Validate the scientific stage using the exact live OCR runtime bindings."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    try:
        expected_commit = authorization._git(root, "rev-parse", "HEAD")  # noqa: SLF001
    except Exception as exc:
        raise OCRLaunchValidationError(f"cannot resolve live source commit: {exc}") from exc
    _require(
        os.environ.get("OCR_STAGE_LAUNCH_LOCK")
        == str(Path(stage_lock_path).expanduser().absolute())
        and os.environ.get("OCR_STAGE_LAUNCH_LOCK_SHA256") == stage_lock_sha256,
        "live stage-lock environment binding differs",
    )
    required = {
        "environment_lock_path": os.environ.get("OCR_ENV_LOCK"),
        "environment_lock_sha256": os.environ.get("OCR_ENV_LOCK_SHA256"),
        "slurm_script_path": os.environ.get("OCR_SLURM_SCRIPT"),
        "slurm_script_sha256": os.environ.get("OCR_SLURM_SCRIPT_SHA256"),
    }
    _require(all(required.values()), "live OCR runtime provenance environment is incomplete")
    return validate_paired_stage_launch(
        repo_root=root,
        expected_commit=expected_commit,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage_lock_path=stage_lock_path,
        stage_lock_sha256=stage_lock_sha256,
        environment_lock_path=str(required["environment_lock_path"]),
        environment_lock_sha256=str(required["environment_lock_sha256"]),
        slurm_script_path=str(required["slurm_script_path"]),
        slurm_script_sha256=str(required["slurm_script_sha256"]),
        expected_recipe=expected_recipe,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage-launch-lock", type=Path, required=True)
    parser.add_argument("--stage-launch-lock-sha256", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--environment-lock-sha256", required=True)
    parser.add_argument("--slurm-script", type=Path, required=True)
    parser.add_argument("--slurm-script-sha256", required=True)
    parser.add_argument("--expected-recipe", choices=RECIPES)
    args = parser.parse_args(argv)
    result = validate_paired_stage_launch(
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        stage_lock_path=args.stage_launch_lock,
        stage_lock_sha256=args.stage_launch_lock_sha256,
        environment_lock_path=args.environment_lock,
        environment_lock_sha256=args.environment_lock_sha256,
        slurm_script_path=args.slurm_script,
        slurm_script_sha256=args.slurm_script_sha256,
        expected_recipe=args.expected_recipe,
    )
    print(result["recipe"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OCRLaunchValidationError",
    "validate_paired_stage_launch",
    "validate_paired_stage_launch_from_environment",
]
