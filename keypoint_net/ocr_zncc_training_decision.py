"""Deep, fail-closed Task-55 OCR-ZNCC training authorization.

The artifact is not trusted because it says ``authorize``.  Creation and every
train-time use reopen the raw no-training evidence, recompute all summaries and
gates from rows, bind the exact committed source, and revalidate the single
Fable High invocation.  This module performs no training or GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from keypoint_net import ocr_zncc_fable_review as fable_review
from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import run_ocr_zncc_exact_gradient_direction_audit as exact_audit
from keypoint_net import run_ocr_zncc_minibatch_gradient_audit as minibatch_audit
from keypoint_net import run_ocr_zncc_stage0 as stage0
from keypoint_net.ocr_zncc_transport import OCRZNCCConfig


SCHEMA_VERSION = "ocr_zncc_task55_training_authorization.v1"
STATUS = "authorize_task55_matched_experiment"
# Compatibility exports for Task80's pre-existing tests and evidence utilities;
# the canonical definitions live in the dedicated review module.
FABLE_HELPER_SHA256 = fable_review.FABLE_HELPER_SHA256
FABLE_PROMPT_PREFIX = fable_review.FABLE_PROMPT_PREFIX
EXPECTED_CELL_IDS = tuple(stage0.CELL_IDS)
EXPECTED_DATASET_BASENAME = "_tdw_world_z_roll_base_panel_512_v2"
EXPECTED_EXECUTION_BOUNDARY = {
    "authorized_recipe": "task55_clean",
    "matched_arms": ["control", "ocr_zncc"],
    "seeds": [42, 43, 44],
    "epochs": 1000,
    "task80_authorized": False,
    "other_objects_or_operators_authorized": False,
    "test_split_authorized": False,
    "training_or_launch_performed_by_this_module": False,
    "gpu_execution_smoke_required_before_scientific_training": True,
}
EXPECTED_REVIEW_EXECUTION = {
    "reviewer": "Fable",
    "model_alias": "fable",
    "effort": "high",
    "review_mode": "calibrated_adversarial_red_team",
    "permission_mode": "plan_read_only",
    "session_persistence": False,
    "single_call_no_retry": True,
    "codex_draft_or_conclusion_supplied": False,
    "prior_reports_are_advisory_stale_examples": True,
    "agreement_is_not_evidence": True,
}
EXPECTED_PROVENANCE_BOUNDARY = {
    "cpu_only": True,
    "optimizer_steps": 0,
    "gpu_jobs_submitted": False,
    "training_authorized": False,
    "task55_and_task80_direction_diagnostics_authorized": True,
    "combined_task55_task80_gradient_calibration_authorized_only_after_both_direction_gates_pass": True,
}
EXPECTED_MINIBATCH_BOUNDARY = {
    "optimizer_steps": 0,
    "training_performed": False,
    "gpu_jobs_submitted": 0,
    "training_authorized_by_this_artifact_alone": False,
}
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version", "artifact_type", "status", "source_commit",
    "source_branch", "research_base", "source_binding", "decision",
    "coverage_gate_status", "coefficient", "coefficient_rule",
    "ocr_zncc_config", "evidence", "primary_recomputation",
    "execution_boundary", "review_interpretation", "review_execution",
    "content_hash_sha256",
}


class OCRTask55AuthorizationError(ValueError):
    """Raised when raw evidence does not authorize the bounded Task55 run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRTask55AuthorizationError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, *, name: str) -> float:
    _require(_is_number(value) and math.isfinite(float(value)), f"{name} is non-finite")
    return float(value)


def _equivalent(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    return left == right


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "logical_name": record.get("logical_name"),
        "sha256": record.get("sha256"),
        "size_bytes": record.get("size_bytes"),
    }
    if "content_hash_sha256" in record:
        result["content_hash_sha256"] = record.get("content_hash_sha256")
    return result


def _embedded_file_identity(record: Mapping[str, Any]) -> tuple[Any, Any]:
    return record.get("sha256"), record.get("size_bytes")


def _validate_embedded_record(record: Any, *, name: str) -> None:
    _require(isinstance(record, Mapping), f"{name} record is absent")
    sha = record.get("sha256")
    size = record.get("size_bytes")
    _require(
        isinstance(sha, str) and authorization._SHA256_RE.fullmatch(sha) is not None,  # noqa: SLF001
        f"{name} SHA-256 is invalid",
    )
    _require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
             f"{name} byte size is invalid")


def _json_record(path: Path | str, *, logical_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document, observed = authorization._load_json(path, name=logical_name)  # noqa: SLF001
    content_hash = authorization._validate_content_hash(document, name=logical_name)  # noqa: SLF001
    return document, {
        "logical_name": logical_name,
        "absolute_path": observed["absolute_path"],
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
        "content_hash_sha256": content_hash,
    }


def _plain_record(path: Path | str, *, logical_name: str) -> dict[str, Any]:
    observed = authorization._file_record(path, name=logical_name)  # noqa: SLF001
    return {
        "logical_name": logical_name,
        "absolute_path": observed["absolute_path"],
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }


def _load_claimed_json(record: Any, *, logical_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(record, Mapping), f"{logical_name} evidence record is invalid")
    _require(record.get("logical_name") == logical_name, f"{logical_name} logical name differs")
    document, observed = _json_record(str(record.get("absolute_path", "")), logical_name=logical_name)
    _require(dict(record) == observed, f"{logical_name} live evidence bytes differ")
    return document, observed


def _load_claimed_plain(record: Any, *, logical_name: str) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{logical_name} evidence record is invalid")
    _require(record.get("logical_name") == logical_name, f"{logical_name} logical name differs")
    observed = _plain_record(str(record.get("absolute_path", "")), logical_name=logical_name)
    _require(dict(record) == observed, f"{logical_name} live evidence bytes differ")
    return observed


def _source_binding(repo_root: Path) -> dict[str, Any]:
    commit, branch = authorization._source_state(repo_root)  # noqa: SLF001
    sources = []
    for relative in authorization.RUN_SOURCE_PATHS:
        observed = authorization._file_record(repo_root / relative, name=relative)  # noqa: SLF001
        sources.append({
            "logical_name": f"committed_source:{relative}",
            "relative_path": relative,
            "sha256": observed["sha256"],
            "size_bytes": observed["size_bytes"],
        })
    return {
        "source_commit": commit,
        "source_branch": branch,
        "full_status_clean": True,
        "research_base_commit": authorization.AUTHORIZED_BASE_COMMIT,
        "research_base_is_ancestor": True,
        "run_sources": sources,
    }


def _validate_checkout(value: Any, *, commit: str, branch: str, name: str) -> None:
    _require(isinstance(value, Mapping), f"{name} checkout is invalid")
    _require(
        value.get("commit") == commit and value.get("branch") == branch,
        f"{name} checkout source differs",
    )
    _require(
        value.get("authorized_base_commit") == authorization.AUTHORIZED_BASE_COMMIT
        and value.get("authorized_base_is_ancestor") is True,
        f"{name} research-base lineage differs",
    )
    _require(
        value.get("complete_status") == "",
        f"{name} complete checkout was dirty",
    )


def _validate_source_record(
    repo_root: Path, embedded: Any, relative: str, *, name: str
) -> None:
    _validate_embedded_record(embedded, name=name)
    live = authorization._file_record(repo_root / relative, name=name)  # noqa: SLF001
    _require(
        _embedded_file_identity(embedded) == (live["sha256"], live["size_bytes"]),
        f"{name} does not match final committed source",
    )


def _expected_training_pair_ids(repo_root: Path) -> set[str]:
    index_path = repo_root / stage0.TRAIN_INDEX_RELPATH
    index, record = authorization._load_json(index_path, name="training pair index")  # noqa: SLF001
    _require(record["sha256"] == stage0.TRAIN_INDEX_SHA256, "training split file hash differs")
    authorization._validate_content_hash(index, name="training pair index")  # noqa: SLF001
    _require(
        index.get("split") == "train"
        and index.get("dataset_binding_sha256") == stage0.DATASET_BINDING_SHA256,
        "training split semantics differ",
    )
    pair_ids = {
        str(row["pair_id"])
        for row in index.get("pairs", [])
        if isinstance(row, Mapping) and row.get("model_name") == stage0.OBJECT_NAME
    }
    _require(len(pair_ids) == 147, "hammer training split pair set differs")
    return pair_ids


def _validate_provenance(
    repo_root: Path,
    document: Mapping[str, Any],
    *,
    commit: str,
    branch: str,
) -> dict[str, Mapping[str, Any]]:
    _require(
        document.get("schema_version") == f"{stage0.SCHEMA_VERSION}.provenance"
        and document.get("artifact_type") == "ocr_zncc_stage0_provenance",
        "Stage0 provenance identity differs",
    )
    _validate_checkout(document.get("checkout"), commit=commit, branch=branch, name="Stage0")
    sources = document.get("source_files")
    _require(isinstance(sources, Mapping) and set(sources) == {
        "ocr_zncc_transport", "stage0_runner", "model", "dataset", "train_index"
    }, "Stage0 source-file set differs")
    source_paths = {
        "ocr_zncc_transport": "keypoint_net/ocr_zncc_transport.py",
        "stage0_runner": "keypoint_net/run_ocr_zncc_stage0.py",
        "model": "keypoint_net/model.py",
        "dataset": "keypoint_net/dataset.py",
        "train_index": stage0.TRAIN_INDEX_RELPATH,
    }
    for role, relative in source_paths.items():
        _validate_source_record(repo_root, sources[role], relative, name=f"Stage0 {role}")
    _require(
        document.get("selector_resolution") == {
            "historical_frozen_controls": "minimum_total_validation_loss",
            "future_ocr_zncc_auxiliary_experiments": "minimum_base_validation_loss",
            "historical_evidence_rewritten": False,
            "ambiguity_status": "resolved_before_stage0_execution",
        },
        "Stage0 selector resolution differs",
    )
    _require(document.get("authorization_boundary") == EXPECTED_PROVENANCE_BOUNDARY,
             "Stage0 zero-step/GPU/training boundary differs")
    _require(Path(str(document.get("dataset_root", ""))).name == EXPECTED_DATASET_BASENAME,
             "Stage0 dataset root basename differs")
    controls = document.get("control_cells")
    _require(isinstance(controls, list) and len(controls) == 6,
             "Stage0 control-cell count differs")
    by_id: dict[str, Mapping[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    for control in controls:
        _require(isinstance(control, Mapping), "Stage0 control cell is invalid")
        cell_id = control.get("cell_id")
        _require(cell_id in EXPECTED_CELL_IDS and cell_id not in by_id,
                 "Stage0 control-cell set is duplicated or invalid")
        recipe, _, seed_text = str(cell_id).partition("__r64__seed")
        _require(control.get("recipe") == recipe and control.get("seed") == int(seed_text),
                 f"Stage0 control identity differs: {cell_id}")
        _require(
            control.get("historical_checkpoint_selector") == "minimum_total_validation_loss"
            and control.get("future_auxiliary_checkpoint_selector")
            == "minimum_base_validation_loss"
            and control.get("strict_state_dict_load") is True,
            f"Stage0 checkpoint semantics differ: {cell_id}",
        )
        for role in ("receipt", "config", "history", "checkpoint", "evaluation"):
            _validate_embedded_record(control.get(role), name=f"{cell_id} {role}")
        _require(
            isinstance(control.get("evaluation_result_content_sha256"), str)
            and authorization._SHA256_RE.fullmatch(  # noqa: SLF001
                str(control["evaluation_result_content_sha256"])
            ) is not None,
            f"Stage0 evaluation content hash differs: {cell_id}",
        )
        checkpoint_hashes.add(str(control["checkpoint"]["sha256"]))
        by_id[str(cell_id)] = control
    _require(set(by_id) == set(EXPECTED_CELL_IDS), "Stage0 control-cell set differs")
    _require(len(checkpoint_hashes) == 6, "Stage0 checkpoint set is not six distinct files")
    return by_id


def _validate_rows_cover_pairs(
    rows: Sequence[Mapping[str, Any]], channels: Sequence[int], pair_ids: set[str], *, name: str
) -> None:
    _require(len(rows) == len(channels) * len(pair_ids), f"{name} row count differs")
    for channel in channels:
        local = [row for row in rows if row.get("channel") == channel]
        _require(
            len(local) == len(pair_ids)
            and {str(row.get("pair_id")) for row in local} == pair_ids,
            f"{name} does not cover the exact full split for channel {channel}",
        )


def _validate_direction_cell(
    document: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    pair_ids: set[str],
) -> dict[str, Any]:
    cell_id = str(control["cell_id"])
    recipe, _, seed_text = cell_id.partition("__r64__seed")
    _require(
        document.get("schema_version") == f"{stage0.SCHEMA_VERSION}.direction_cell"
        and document.get("artifact_type") == "ocr_zncc_direction_cell"
        and document.get("cell_id") == cell_id
        and document.get("recipe") == recipe
        and document.get("seed") == int(seed_text),
        f"Stage0 direction-cell identity differs: {cell_id}",
    )
    _require(document.get("control_provenance") == control,
             f"Stage0 direction/control provenance differs: {cell_id}")
    channels = document.get("eligible_channels")
    _require(
        isinstance(channels, list) and len(channels) >= 2
        and len(set(channels)) == len(channels)
        and all(isinstance(item, int) and 0 <= item < 10 for item in channels),
        f"Stage0 eligible-channel set differs: {cell_id}",
    )
    dataset = document.get("dataset")
    _require(
        isinstance(dataset, Mapping)
        and dataset.get("split") == "train"
        and dataset.get("sample_count") == 147
        and dataset.get("index_sha256") == stage0.TRAIN_INDEX_SHA256
        and dataset.get("index_content_sha256") == stage0.TRAIN_INDEX_CONTENT_SHA256
        and dataset.get("dataset_binding_sha256") == stage0.DATASET_BINDING_SHA256
        and dataset.get("material_oracle_use")
        == "evaluation_only_known_positive_6_degree_world_z_roll",
        f"Stage0 direction split/oracle binding differs: {cell_id}",
    )
    execution = document.get("execution")
    _require(
        isinstance(execution, Mapping)
        and execution.get("device") == "cpu"
        and execution.get("model_mode") == "eval"
        and execution.get("optimizer_steps") == 0
        and execution.get("scheduler_steps") == 0
        and execution.get("training_performed") is False
        and execution.get("state_unchanged") is True
        and execution.get("state_sha256_before") == execution.get("state_sha256_after"),
        f"Stage0 direction execution boundary differs: {cell_id}",
    )
    _require(
        document.get("ocr_zncc_config")
        == OCRZNCCConfig(peak_margin_exclusion_radius_cells=1).as_dict(),
        f"Stage0 direction matcher config differs: {cell_id}",
    )
    rows = document.get("rows")
    _require(isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows),
             f"Stage0 direction rows are invalid: {cell_id}")
    _validate_rows_cover_pairs(rows, channels, pair_ids, name=f"Stage0 direction {cell_id}")
    for row in rows:
        _require(row.get("eligible_channel") is True,
                 f"Stage0 direction row is not eligible: {cell_id}")
        if row.get("usable_direction") is True:
            _require(row.get("accepted_match") is True,
                     f"Stage0 usable row is not accepted: {cell_id}")
            _finite(row.get("direction_cosine"), name=f"{cell_id} direction cosine")
            _finite(
                row.get("one_cell_material_error_delta_objdiag"),
                name=f"{cell_id} material-error delta",
            )
        else:
            _require(
                row.get("direction_cosine") is None
                and row.get("one_cell_material_error_delta_objdiag") is None,
                f"Stage0 unusable row carries direction evidence: {cell_id}",
            )
    recomputed = stage0._summarize_direction_rows(rows, channels)  # noqa: SLF001
    _require(_equivalent(document.get("summary"), recomputed),
             f"Stage0 direction summary does not recompute: {cell_id}")
    _require(
        recomputed["cell_pass"] is False
        and recomputed["usable_direction_coverage"] < stage0.USABLE_COVERAGE_MINIMUM
        and recomputed["direction_cosine"] is not None
        and recomputed["direction_cosine"]["median"] > 0.0
        and recomputed["one_cell_material_error_delta_objdiag"] is not None
        and recomputed["one_cell_material_error_delta_objdiag"]["median"] < 0.0
        and recomputed["active_on_object"]["change"] >= 0,
        f"Stage0 accepted material-direction evidence differs: {cell_id}",
    )
    return recomputed


def _validate_direction_summary(
    document: Mapping[str, Any],
    *,
    cells: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        document.get("schema_version") == f"{stage0.SCHEMA_VERSION}.direction_summary"
        and document.get("artifact_type") == "ocr_zncc_task55_task80_direction_summary",
        "Stage0 direction-summary identity differs",
    )
    recomputed_recipes: dict[str, Any] = {}
    for recipe in stage0.RECIPES:
        local_ids = [cell_id for cell_id in EXPECTED_CELL_IDS if cell_id.startswith(recipe)]
        pass_count = sum(bool(cells[cell_id]["cell_pass"]) for cell_id in local_ids)
        recipe_pass = pass_count >= stage0.RECIPE_PASS_COUNT_MINIMUM
        recomputed_recipes[recipe] = {
            "cell_pass_count": pass_count,
            "cell_count": len(local_ids),
            "recipe_pass": recipe_pass,
            "preregistered_engineering_rule":
                f"at_least_{stage0.RECIPE_PASS_COUNT_MINIMUM}_of_3_cells_pass",
            "cells": [
                {"cell_id": cell_id, "summary": cells[cell_id]}
                for cell_id in local_ids
            ],
        }
    _require(_equivalent(document.get("recipe_results"), recomputed_recipes),
             "Stage0 recipe direction summaries do not recompute")
    both_pass = all(item["recipe_pass"] for item in recomputed_recipes.values())
    _require(
        both_pass is False
        and document.get("both_recipe_gates_pass") is False
        and document.get("calibration_authorized_by_stage0_rule") is False,
        "Stage0 failed 50-percent gate was rewritten",
    )
    claims = document.get("direction_cell_artifacts")
    _require(isinstance(claims, list) and len(claims) == 6,
             "Stage0 direction-summary cell records differ")
    _require(
        {_embedded_file_identity(item) for item in claims}
        == {(record["sha256"], record["size_bytes"]) for record in records.values()},
        "Stage0 direction-summary artifact bytes differ",
    )
    return {
        "recipe_results": recomputed_recipes,
        "both_recipe_gates_pass": both_pass,
        "calibration_authorized_by_stage0_rule": False,
    }


def _validate_exact_audit(
    repo_root: Path,
    document: Mapping[str, Any],
    *,
    commit: str,
    branch: str,
    controls: Mapping[str, Mapping[str, Any]],
    direction_channels: Mapping[str, Sequence[int]],
    pair_ids: set[str],
) -> dict[str, Any]:
    _require(
        document.get("schema_version") == exact_audit.SCHEMA_VERSION
        and document.get("artifact_type")
        == "ocr_zncc_exact_coordinate_gradient_direction_audit",
        "exact-gradient audit identity differs",
    )
    _validate_checkout(document.get("checkout"), commit=commit, branch=branch,
                       name="exact-gradient audit")
    inputs = document.get("inputs")
    _require(isinstance(inputs, Mapping), "exact-gradient inputs are invalid")
    for role, relative in {
        "audit_source": "keypoint_net/run_ocr_zncc_exact_gradient_direction_audit.py",
        "ocr_zncc_source": "keypoint_net/ocr_zncc_transport.py",
        "stage0_source": "keypoint_net/run_ocr_zncc_stage0.py",
        "train_index": stage0.TRAIN_INDEX_RELPATH,
    }.items():
        _validate_source_record(repo_root, inputs.get(role), relative,
                                name=f"exact-gradient {role}")
    _require(inputs.get("dataset_binding_sha256") == stage0.DATASET_BINDING_SHA256,
             "exact-gradient dataset binding differs")
    cells = document.get("cells")
    _require(isinstance(cells, list) and len(cells) == 24,
             "exact-gradient cell count differs")
    observed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    summaries: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    expected_config = {
        radius: OCRZNCCConfig(peak_margin_exclusion_radius_cells=radius).as_dict()
        for radius in (0, 1)
    }
    for cell in cells:
        _require(isinstance(cell, Mapping), "exact-gradient cell is invalid")
        cell_id = cell.get("cell_id")
        stage = cell.get("stage")
        config = cell.get("ocr_zncc_config")
        radius = config.get("peak_margin_exclusion_radius_cells") if isinstance(config, Mapping) else None
        key = (str(cell_id), str(stage), int(radius) if isinstance(radius, int) else -1)
        _require(
            cell_id in EXPECTED_CELL_IDS
            and stage in {"frozen_control", "random_initialization"}
            and radius in {0, 1}
            and key not in observed
            and config == expected_config[int(radius)],
            "exact-gradient cell identity/configuration differs",
        )
        channels = cell.get("channel_indices")
        expected_channels = (
            list(direction_channels[str(cell_id)])
            if stage == "frozen_control" else list(range(10))
        )
        _require(
            isinstance(channels, list) and len(channels) >= 2
            and len(set(channels)) == len(channels)
            and all(isinstance(item, int) and 0 <= item < 10 for item in channels),
            "exact-gradient channel set differs",
        )
        if stage == "random_initialization":
            _require(channels == list(range(10)), "initial exact-gradient channels differ")
        else:
            _require(channels == expected_channels,
                     "frozen exact-gradient channels differ from Stage0 direction evidence")
        execution = cell.get("execution")
        _require(
            isinstance(execution, Mapping)
            and execution.get("device") == "cpu"
            and execution.get("optimizer_steps") == 0
            and execution.get("scheduler_steps") == 0
            and execution.get("parameter_grad_fields_unchanged") is True,
            "exact-gradient zero-step boundary differs",
        )
        if stage == "frozen_control":
            _require(
                execution.get("model_mode") == "eval"
                and execution.get("model_state_sha256_before")
                == execution.get("model_state_sha256_after"),
                "frozen exact-gradient model state differs",
            )
        else:
            _require(execution.get("model_mode") == "train",
                     "initial exact-gradient model mode differs")
        rows = cell.get("rows")
        _require(isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows),
                 "exact-gradient rows are invalid")
        _validate_rows_cover_pairs(rows, channels, pair_ids,
                                   name=f"exact-gradient {key}")
        for row in rows:
            if row.get("accepted_match") is True:
                _require(row.get("source_patch_inside") is True,
                         "exact-gradient accepted an invalid source patch")
            if row.get("usable_direction") is True:
                _require(row.get("accepted_match") is True,
                         "exact-gradient usable row is not accepted")
                _finite(row.get("direction_cosine"), name="exact-gradient cosine")
                _finite(row.get("one_cell_material_error_delta_objdiag"),
                        name="exact-gradient one-cell delta")
        summary = exact_audit._summary(rows, channels)  # noqa: SLF001
        _require(_equivalent(cell.get("summary"), summary),
                 f"exact-gradient summary does not recompute: {key}")
        _require(summary["accepted_source_patch_outside_image_count"] == 0,
                 "exact-gradient boundary validity failed")
        observed[key] = cell
        summaries[key] = summary
    expected_keys = {
        (cell_id, stage, radius)
        for cell_id in EXPECTED_CELL_IDS
        for stage in ("frozen_control", "random_initialization")
        for radius in (0, 1)
    }
    _require(set(observed) == expected_keys, "exact-gradient cell set differs")
    recomputed = {
        "all_frozen_infinitesimal_direction_gates": all(
            summary["infinitesimal_direction_gate"]
            for (cell_id, stage, radius), summary in summaries.items()
            if stage == "frozen_control"
        ),
        "all_initial_infinitesimal_direction_gates": all(
            summary["infinitesimal_direction_gate"]
            for (cell_id, stage, radius), summary in summaries.items()
            if stage == "random_initialization"
        ),
        "all_frozen_one_cell_finite_step_gates": all(
            summary["one_cell_finite_step_gate"]
            for (cell_id, stage, radius), summary in summaries.items()
            if stage == "frozen_control"
        ),
        "all_initial_one_cell_finite_step_gates": all(
            summary["one_cell_finite_step_gate"]
            for (cell_id, stage, radius), summary in summaries.items()
            if stage == "random_initialization"
        ),
        "frozen_50_percent_coverage_gate_status": "remains_failed_unchanged",
        "training_authorized_by_artifact": False,
    }
    _require(_equivalent(document.get("decision_summary"), recomputed),
             "exact-gradient decision summary does not recompute")
    _require(
        recomputed["all_frozen_infinitesimal_direction_gates"] is True
        and recomputed["all_initial_infinitesimal_direction_gates"] is True
        and recomputed["all_frozen_one_cell_finite_step_gates"] is True
        and recomputed["all_initial_one_cell_finite_step_gates"] is False,
        "exact-gradient frozen gate outcome differs",
    )
    _require(
        document.get("execution")
        == {"optimizer_steps": 0, "scheduler_steps": 0, "gpu_jobs_submitted": 0},
        "exact-gradient top-level execution boundary differs",
    )
    return recomputed


def _describe(values: Sequence[float]) -> dict[str, Any]:
    _require(bool(values) and all(math.isfinite(value) for value in values),
             "minibatch summary inputs are invalid")
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _validate_minibatch_audit(
    repo_root: Path,
    document: Mapping[str, Any],
    *,
    commit: str,
    branch: str,
) -> dict[str, Any]:
    _require(
        document.get("schema_version") == minibatch_audit.SCHEMA_VERSION
        and document.get("artifact_type")
        == "zero_step_training_minibatch_total_gradient_audit",
        "minibatch-gradient audit identity differs",
    )
    _validate_checkout(document.get("checkout"), commit=commit, branch=branch,
                       name="minibatch-gradient audit")
    sources = document.get("source_files")
    _require(isinstance(sources, Mapping) and set(sources) == {
        "runner", "ocr_zncc_transport", "stage0_runner", "model",
        "loss_implementation", "train_index",
    }, "minibatch-gradient source set differs")
    for role, relative in {
        "runner": "keypoint_net/run_ocr_zncc_minibatch_gradient_audit.py",
        "ocr_zncc_transport": "keypoint_net/ocr_zncc_transport.py",
        "stage0_runner": "keypoint_net/run_ocr_zncc_stage0.py",
        "model": "keypoint_net/model.py",
        "loss_implementation": "keypoint_net/model.py",
        "train_index": stage0.TRAIN_INDEX_RELPATH,
    }.items():
        _validate_source_record(repo_root, sources[role], relative,
                                name=f"minibatch-gradient {role}")
    dataset = document.get("dataset")
    _require(
        isinstance(dataset, Mapping)
        and dataset.get("split") == "train"
        and dataset.get("pair_count") == 147
        and dataset.get("dataset_binding_sha256") == stage0.DATASET_BINDING_SHA256
        and dataset.get("train_index_sha256") == stage0.TRAIN_INDEX_SHA256,
        "minibatch-gradient dataset/split binding differs",
    )
    expected_config = OCRZNCCConfig(peak_margin_exclusion_radius_cells=1).as_dict()
    _require(document.get("ocr_zncc_config") == expected_config,
             "minibatch-gradient matcher config differs")
    expected_rule = {
        "formula": "min(0.5, 0.10 / median_of_six_cell_median_unscaled_batch_ratios)",
        "target_median_contribution": minibatch_audit.TARGET_MEDIAN_CONTRIBUTION,
        "coefficient_safety_cap": minibatch_audit.COEFFICIENT_CAP,
        "raw_lambda_sweep_performed": False,
    }
    _require(document.get("coefficient_rule") == expected_rule,
             "minibatch-gradient coefficient rule differs")
    cells = document.get("cells")
    _require(isinstance(cells, list) and len(cells) == 6,
             "minibatch-gradient cell count differs")
    medians: list[float] = []
    observed_ids: set[str] = set()
    parameter_names: list[str] | None = None
    for cell in cells:
        _require(isinstance(cell, Mapping), "minibatch-gradient cell is invalid")
        cell_id = cell.get("cell_id")
        recipe = cell.get("recipe")
        seed = cell.get("seed")
        _require(
            cell_id in EXPECTED_CELL_IDS and cell_id not in observed_ids
            and cell_id == f"{recipe}__r64__seed{seed}"
            and recipe in stage0.RECIPES and seed in stage0.SEEDS
            and cell.get("model_initialization_seed") == seed,
            "minibatch-gradient cell identity differs",
        )
        observed_ids.add(str(cell_id))
        names = cell.get("parameter_names")
        _require(
            isinstance(names, list) and names
            and len(names) == len(set(names))
            and all(str(name).startswith(("extractor.encoder.", "extractor.heatmap_head."))
                    for name in names)
            and cell.get("parameter_set")
            == "all_trainable_extractor.encoder_and_extractor.heatmap_head",
            f"minibatch-gradient parameter set differs: {cell_id}",
        )
        if parameter_names is None:
            parameter_names = list(names)
        _require(names == parameter_names, "minibatch-gradient parameter names differ across cells")
        rows = cell.get("rows")
        _require(isinstance(rows, list) and len(rows) == 10,
                 f"minibatch-gradient batch count differs: {cell_id}")
        nonzero: list[Mapping[str, Any]] = []
        expected_starts = list(range(0, 144, 16)) + [144]
        expected_counts = [16] * 9 + [3]
        for index, (row, start, count) in enumerate(zip(rows, expected_starts, expected_counts)):
            _require(
                isinstance(row, Mapping)
                and row.get("batch_index") == index
                and row.get("sample_start") == start
                and row.get("sample_count") == count
                and row.get("possible_match_count") == count * 10
                and row.get("accepted_source_patch_outside_image_count") == 0,
                f"minibatch-gradient row partition/boundary differs: {cell_id}",
            )
            accepted = row.get("accepted_match_count")
            _require(isinstance(accepted, int) and 0 <= accepted <= count * 10,
                     f"minibatch accepted count differs: {cell_id}")
            base = _finite(row.get("base_gradient_norm"), name="base gradient norm")
            aux = _finite(row.get("auxiliary_gradient_norm"), name="auxiliary gradient norm")
            _require(base > minibatch_audit.NORM_FLOOR, "minibatch base gradient is zero")
            if accepted == 0:
                _require(
                    aux <= minibatch_audit.NORM_FLOOR
                    and row.get("unscaled_auxiliary_to_base_norm_ratio") is None
                    and row.get("cap_scaled_auxiliary_to_base_norm_ratio") is None
                    and row.get("base_auxiliary_gradient_cosine") is None,
                    "zero-accept minibatch gradient semantics differ",
                )
            else:
                ratio = _finite(
                    row.get("unscaled_auxiliary_to_base_norm_ratio"),
                    name="unscaled auxiliary/base ratio",
                )
                cap_ratio = _finite(
                    row.get("cap_scaled_auxiliary_to_base_norm_ratio"),
                    name="cap-scaled auxiliary/base ratio",
                )
                cosine = _finite(
                    row.get("base_auxiliary_gradient_cosine"),
                    name="base/auxiliary gradient cosine",
                )
                _require(
                    aux > minibatch_audit.NORM_FLOOR
                    and math.isclose(ratio, aux / base, rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(
                        cap_ratio,
                        minibatch_audit.COEFFICIENT_CAP * ratio,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and -1.0 - 1e-12 <= cosine <= 1.0 + 1e-12,
                    "minibatch gradient row arithmetic differs",
                )
                nonzero.append(row)
        unscaled = [float(row["unscaled_auxiliary_to_base_norm_ratio"]) for row in nonzero]
        cap_scaled = [float(row["cap_scaled_auxiliary_to_base_norm_ratio"]) for row in nonzero]
        cosines = [float(row["base_auxiliary_gradient_cosine"]) for row in nonzero]
        recomputed_summary = {
            "batch_count": len(rows),
            "nonzero_accept_batch_count": len(nonzero),
            "zero_accept_batch_count": len(rows) - len(nonzero),
            "zero_accept_batch_fraction": (len(rows) - len(nonzero)) / len(rows),
            "accepted_match_count": sum(int(row["accepted_match_count"]) for row in rows),
            "possible_match_count": 1470,
            "accepted_match_coverage":
                sum(int(row["accepted_match_count"]) for row in rows) / 1470,
            "unscaled_auxiliary_to_base_norm_ratio_nonzero_batches": _describe(unscaled),
            "cap_scaled_auxiliary_to_base_norm_ratio_nonzero_batches": _describe(cap_scaled),
            "base_auxiliary_gradient_cosine_nonzero_batches": _describe(cosines),
            "accepted_source_patch_outside_image_count": 0,
        }
        _require(_equivalent(cell.get("summary"), recomputed_summary),
                 f"minibatch-gradient summary does not recompute: {cell_id}")
        execution = cell.get("execution")
        _require(
            isinstance(execution, Mapping)
            and execution.get("device") == "cpu"
            and execution.get("model_mode") == "train"
            and execution.get("batch_size") == 16
            and execution.get("shuffle") is False
            and execution.get("optimizer_steps") == 0
            and execution.get("scheduler_steps") == 0
            and execution.get("batchnorm_state_change_expected") is True,
            f"minibatch-gradient execution boundary differs: {cell_id}",
        )
        medians.append(float(
            recomputed_summary["unscaled_auxiliary_to_base_norm_ratio_nonzero_batches"]["median"]
        ))
    _require(observed_ids == set(EXPECTED_CELL_IDS), "minibatch-gradient cell set differs")
    median_unscaled = float(statistics.median(medians))
    coefficient = min(
        minibatch_audit.COEFFICIENT_CAP,
        minibatch_audit.TARGET_MEDIAN_CONTRIBUTION / median_unscaled,
    )
    combined = {
        "median_of_six_cell_median_unscaled_batch_ratios": median_unscaled,
        "minimum_cell_median_unscaled_batch_ratio": min(medians),
        "maximum_cell_median_unscaled_batch_ratio": max(medians),
        "recommended_scaled_median_contribution": coefficient * median_unscaled,
        "all_finite": True,
        "all_base_gradients_nonzero": True,
        "all_accepted_source_patches_inside_image": True,
    }
    _require(_equivalent(document.get("combined_summary"), combined),
             "minibatch-gradient combined summary does not recompute")
    _require(
        _is_number(document.get("recommended_coefficient"))
        and math.isclose(float(document["recommended_coefficient"]), coefficient,
                         rel_tol=1e-12, abs_tol=1e-12)
        and 0.0 < coefficient <= 0.5,
        "minibatch-gradient coefficient does not recompute",
    )
    _require(document.get("authorization_boundary") == EXPECTED_MINIBATCH_BOUNDARY,
             "minibatch-gradient zero-step/GPU boundary differs")
    return {
        "combined_summary": combined,
        "recommended_coefficient": coefficient,
        "coefficient_rule": expected_rule,
        "ocr_zncc_config": expected_config,
    }


def _validate_primary_evidence(
    repo_root: Path,
    *,
    provenance: Mapping[str, Any],
    direction_summary: Mapping[str, Any],
    direction_cells: Mapping[str, Mapping[str, Any]],
    direction_records: Mapping[str, Mapping[str, Any]],
    exact: Mapping[str, Any],
    minibatch: Mapping[str, Any],
) -> dict[str, Any]:
    commit, branch = authorization._source_state(repo_root)  # noqa: SLF001
    pair_ids = _expected_training_pair_ids(repo_root)
    controls = _validate_provenance(
        repo_root, provenance, commit=commit, branch=branch
    )
    _require(set(direction_cells) == set(EXPECTED_CELL_IDS),
             "Stage0 direction-cell set differs")
    recomputed_cells = {
        cell_id: _validate_direction_cell(
            direction_cells[cell_id], control=controls[cell_id], pair_ids=pair_ids
        )
        for cell_id in EXPECTED_CELL_IDS
    }
    direction_result = _validate_direction_summary(
        direction_summary, cells=recomputed_cells, records=direction_records
    )
    exact_result = _validate_exact_audit(
        repo_root, exact, commit=commit, branch=branch,
        controls=controls,
        direction_channels={
            cell_id: direction_cells[cell_id]["eligible_channels"]
            for cell_id in EXPECTED_CELL_IDS
        },
        pair_ids=pair_ids,
    )
    minibatch_result = _validate_minibatch_audit(
        repo_root, minibatch, commit=commit, branch=branch
    )
    return {
        "stage0_direction": direction_result,
        "exact_gradient_decision": exact_result,
        "minibatch_gradient": minibatch_result,
        "source_commit": commit,
        "source_branch": branch,
        "optimizer_steps_in_primary_evidence": 0,
        "gpu_jobs_submitted_in_primary_evidence": 0,
        "training_performed_in_primary_evidence": False,
    }


def _load_evidence(document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = document.get("evidence")
    _require(isinstance(evidence, Mapping) and set(evidence) == {
        "stage0_provenance", "stage0_direction_summary", "stage0_direction_cells",
        "exact_gradient_audit", "minibatch_gradient_audit", "fable_final_review",
    }, "Task55 evidence field set differs")
    provenance, provenance_record = _load_claimed_json(
        evidence["stage0_provenance"], logical_name="stage0_provenance"
    )
    direction_summary, summary_record = _load_claimed_json(
        evidence["stage0_direction_summary"], logical_name="stage0_direction_summary"
    )
    exact, exact_record = _load_claimed_json(
        evidence["exact_gradient_audit"], logical_name="exact_gradient_audit"
    )
    minibatch, minibatch_record = _load_claimed_json(
        evidence["minibatch_gradient_audit"], logical_name="minibatch_gradient_audit"
    )
    claims = evidence["stage0_direction_cells"]
    _require(isinstance(claims, list) and len(claims) == 6,
             "Task55 direction evidence records differ")
    direction_cells: dict[str, Mapping[str, Any]] = {}
    direction_records: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        _require(isinstance(claim, Mapping), "Task55 direction evidence record is invalid")
        logical_name = str(claim.get("logical_name", ""))
        cell_id = logical_name.removeprefix("stage0_direction_cell:")
        _require(cell_id in EXPECTED_CELL_IDS and cell_id not in direction_cells,
                 "Task55 direction evidence cell set differs")
        cell, record = _load_claimed_json(claim, logical_name=logical_name)
        direction_cells[cell_id] = cell
        direction_records[cell_id] = record
    return {
        "provenance": provenance,
        "direction_summary": direction_summary,
        "direction_cells": direction_cells,
        "exact": exact,
        "minibatch": minibatch,
    }, {
        "stage0_provenance": provenance_record,
        "stage0_direction_summary": summary_record,
        "stage0_direction_cells": direction_records,
        "exact_gradient_audit": exact_record,
        "minibatch_gradient_audit": minibatch_record,
        "fable_final_review": evidence["fable_final_review"],
    }


def _validate_fable(
    repo_root: Path,
    claim: Any,
    *,
    primary_records: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(claim, Mapping) and set(claim) == {
        "briefing", "receipt", "compiled_prompt", "raw_review", "helper"
    }, "Task55 Fable evidence field set differs")
    briefing, briefing_record = _load_claimed_json(
        claim["briefing"], logical_name="fable_briefing"
    )
    del briefing
    receipt, receipt_record = _load_claimed_json(
        claim["receipt"], logical_name="fable_invocation_receipt"
    )
    prompt_record = _load_claimed_plain(
        claim["compiled_prompt"], logical_name="fable_compiled_prompt"
    )
    raw_record = _load_claimed_plain(
        claim["raw_review"], logical_name="fable_raw_review"
    )
    helper_record = _load_claimed_plain(
        claim["helper"], logical_name="fable_high_helper"
    )
    primary_identities = {
        "stage0_provenance": primary_records["stage0_provenance"],
        "stage0_direction_summary": primary_records["stage0_direction_summary"],
        "exact_gradient_audit": primary_records["exact_gradient_audit"],
        "minibatch_gradient_audit": primary_records["minibatch_gradient_audit"],
        **{
            f"stage0_direction_cell:{cell_id}": record
            for cell_id, record in primary_records["stage0_direction_cells"].items()
        },
    }
    validated = fable_review.validate_review_bundle(
        repo_root=repo_root,
        receipt=receipt,
        briefing_path=briefing_record["absolute_path"],
        prompt_path=prompt_record["absolute_path"],
        raw_path=raw_record["absolute_path"],
        helper_path=helper_record["absolute_path"],
        primary_identities=primary_identities,
    )
    return {
        **validated,
        "receipt": _record_identity(receipt_record),
    }


def validate_authorization_document(
    repo_root: Path | str, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Deeply revalidate Task55 authorization at the train-time boundary."""

    _require(isinstance(document, Mapping), "Task55 authorization is not a mapping")
    _require(set(document) == EXPECTED_TOP_LEVEL_FIELDS,
             "Task55 authorization field set differs")
    authorization._validate_content_hash(document, name="Task55 authorization")  # noqa: SLF001
    _require(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("artifact_type")
        == "bounded_task55_ocr_zncc_training_authorization"
        and document.get("status") == STATUS,
        "Task55 authorization identity/status differs",
    )
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _source_binding(root)
    _require(document.get("source_binding") == source,
             "Task55 authorization source binding differs")
    _require(
        document.get("source_commit") == source["source_commit"]
        and document.get("source_branch") == source["source_branch"],
        "Task55 authorization source differs",
    )
    _require(
        document.get("research_base") == {
            "commit": authorization.AUTHORIZED_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "Task55 research-base binding differs",
    )
    loaded, records = _load_evidence(document)
    recomputed = _validate_primary_evidence(
        root,
        provenance=loaded["provenance"],
        direction_summary=loaded["direction_summary"],
        direction_cells=loaded["direction_cells"],
        direction_records=records["stage0_direction_cells"],
        exact=loaded["exact"],
        minibatch=loaded["minibatch"],
    )
    _require(_equivalent(document.get("primary_recomputation"), recomputed),
             "Task55 stored primary recomputation differs from live rows")
    fable = _validate_fable(
        root, records["fable_final_review"], primary_records=records
    )
    _require(fable.get("review_valid") is True, "Task55 Fable review is invalid")
    coefficient = recomputed["minibatch_gradient"]["recommended_coefficient"]
    _require(
        _is_number(document.get("coefficient"))
        and math.isclose(float(document["coefficient"]), coefficient,
                         rel_tol=1e-12, abs_tol=1e-12)
        and document.get("coefficient_rule")
        == recomputed["minibatch_gradient"]["coefficient_rule"]
        and document.get("ocr_zncc_config")
        == recomputed["minibatch_gradient"]["ocr_zncc_config"],
        "Task55 coefficient/configuration differs from live minibatch rows",
    )
    _require(document.get("coverage_gate_status") == {
        "preregistered_50_percent_gate": "failed_unchanged",
        "interpretation": (
            "The threshold remains failed. This authorization is only for a bounded "
            "paired causal experiment because the accepted-match material direction "
            "and exact local training gradient independently revalidate."
        ),
    }, "Task55 coverage-gate interpretation differs")
    _require(document.get("execution_boundary") == EXPECTED_EXECUTION_BOUNDARY,
             "Task55 execution boundary differs")
    _require(document.get("review_execution") == EXPECTED_REVIEW_EXECUTION,
             "Task55 review execution contract differs")
    return {
        "authorization_valid": True,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "source_commit": source["source_commit"],
        "source_branch": source["source_branch"],
        "coefficient": coefficient,
        "ocr_zncc_config": recomputed["minibatch_gradient"]["ocr_zncc_config"],
        "coverage_gate_passed": False,
        "fable_review_valid": True,
        "training_or_launch_performed": False,
    }


def _write_exclusive(path: Path | str, document: Mapping[str, Any]) -> None:
    absolute = Path(path).expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCRTask55AuthorizationError(f"cannot exclusively create {absolute}") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "Task55 authorization write made no progress")
            offset += written
    finally:
        os.close(descriptor)


def create(
    *,
    repo_root: Path | str,
    stage0_provenance_path: Path | str,
    direction_summary_path: Path | str,
    direction_cell_paths: Sequence[Path | str],
    exact_gradient_path: Path | str,
    minibatch_gradient_path: Path | str,
    fable_briefing_path: Path | str,
    fable_receipt_path: Path | str,
    fable_prompt_path: Path | str,
    fable_raw_path: Path | str,
    fable_helper_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Create a cross-machine, live-evidence-bound Task55 authorization."""

    output = Path(output_path).expanduser().absolute()
    _require(not output.exists(), f"refusing to overwrite {output}")
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _source_binding(root)
    provenance, provenance_record = _json_record(
        stage0_provenance_path, logical_name="stage0_provenance"
    )
    direction_summary, summary_record = _json_record(
        direction_summary_path, logical_name="stage0_direction_summary"
    )
    exact, exact_record = _json_record(
        exact_gradient_path, logical_name="exact_gradient_audit"
    )
    minibatch, minibatch_record = _json_record(
        minibatch_gradient_path, logical_name="minibatch_gradient_audit"
    )
    _require(len(direction_cell_paths) == 6, "exactly six direction cells are required")
    direction_cells: dict[str, Mapping[str, Any]] = {}
    direction_records: dict[str, Mapping[str, Any]] = {}
    for path in direction_cell_paths:
        temporary, temporary_record = _json_record(path, logical_name="temporary_direction_cell")
        cell_id = temporary.get("cell_id")
        _require(cell_id in EXPECTED_CELL_IDS and cell_id not in direction_cells,
                 "direction-cell creation set differs")
        logical_name = f"stage0_direction_cell:{cell_id}"
        temporary_record["logical_name"] = logical_name
        direction_cells[str(cell_id)] = temporary
        direction_records[str(cell_id)] = temporary_record
    recomputed = _validate_primary_evidence(
        root,
        provenance=provenance,
        direction_summary=direction_summary,
        direction_cells=direction_cells,
        direction_records=direction_records,
        exact=exact,
        minibatch=minibatch,
    )
    briefing, briefing_record = _json_record(
        fable_briefing_path, logical_name="fable_briefing"
    )
    del briefing
    receipt, receipt_record = _json_record(
        fable_receipt_path, logical_name="fable_invocation_receipt"
    )
    del receipt
    prompt_record = _plain_record(
        fable_prompt_path, logical_name="fable_compiled_prompt"
    )
    raw_record = _plain_record(fable_raw_path, logical_name="fable_raw_review")
    helper_record = _plain_record(fable_helper_path, logical_name="fable_high_helper")
    fable_claim = {
        "briefing": briefing_record,
        "receipt": receipt_record,
        "compiled_prompt": prompt_record,
        "raw_review": raw_record,
        "helper": helper_record,
    }
    primary_records = {
        "stage0_provenance": provenance_record,
        "stage0_direction_summary": summary_record,
        "stage0_direction_cells": direction_records,
        "exact_gradient_audit": exact_record,
        "minibatch_gradient_audit": minibatch_record,
    }
    fable_validation = _validate_fable(
        root, fable_claim, primary_records=primary_records
    )
    _require(fable_validation.get("review_valid") is True, "Fable review did not validate")
    coefficient = recomputed["minibatch_gradient"]["recommended_coefficient"]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "bounded_task55_ocr_zncc_training_authorization",
        "status": STATUS,
        "source_commit": source["source_commit"],
        "source_branch": source["source_branch"],
        "research_base": {
            "commit": authorization.AUTHORIZED_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "source_binding": source,
        "decision": (
            "Test whether OCR-ZNCC at the row-recomputed formula coefficient reduces "
            "validation material drift relative to an exact matched Task55 control."
        ),
        "coverage_gate_status": {
            "preregistered_50_percent_gate": "failed_unchanged",
            "interpretation": (
                "The threshold remains failed. This authorization is only for a bounded "
                "paired causal experiment because the accepted-match material direction "
                "and exact local training gradient independently revalidate."
            ),
        },
        "coefficient": coefficient,
        "coefficient_rule": recomputed["minibatch_gradient"]["coefficient_rule"],
        "ocr_zncc_config": recomputed["minibatch_gradient"]["ocr_zncc_config"],
        "evidence": {
            "stage0_provenance": provenance_record,
            "stage0_direction_summary": summary_record,
            "stage0_direction_cells": [
                direction_records[cell_id] for cell_id in EXPECTED_CELL_IDS
            ],
            "exact_gradient_audit": exact_record,
            "minibatch_gradient_audit": minibatch_record,
            "fable_final_review": fable_claim,
        },
        "primary_recomputation": recomputed,
        "execution_boundary": EXPECTED_EXECUTION_BOUNDARY,
        "review_interpretation": (
            "Fable is an advisory challenger, not authority. The final report is bound "
            "to the exact briefing and primary bytes; every decision-changing numeric "
            "claim is independently recomputed here from raw evidence rows."
        ),
        "review_execution": EXPECTED_REVIEW_EXECUTION,
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    # Revalidate in memory before creating an immutable output.  This catches any
    # accidental mismatch between creation and train-time validation contracts.
    validate_authorization_document(root, document)
    _write_exclusive(output, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage0-provenance", type=Path, required=True)
    parser.add_argument("--direction-summary", type=Path, required=True)
    parser.add_argument("--direction-cell", action="append", type=Path, required=True)
    parser.add_argument("--exact-gradient", type=Path, required=True)
    parser.add_argument("--minibatch-gradient", type=Path, required=True)
    parser.add_argument("--fable-briefing", type=Path, required=True)
    parser.add_argument("--fable-receipt", type=Path, required=True)
    parser.add_argument("--fable-prompt", type=Path, required=True)
    parser.add_argument("--fable-raw", type=Path, required=True)
    parser.add_argument("--fable-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = create(
        repo_root=args.repo_root,
        stage0_provenance_path=args.stage0_provenance,
        direction_summary_path=args.direction_summary,
        direction_cell_paths=args.direction_cell,
        exact_gradient_path=args.exact_gradient,
        minibatch_gradient_path=args.minibatch_gradient,
        fable_briefing_path=args.fable_briefing,
        fable_receipt_path=args.fable_receipt,
        fable_prompt_path=args.fable_prompt,
        fable_raw_path=args.fable_raw,
        fable_helper_path=args.fable_helper,
        output_path=args.output,
    )
    print(json.dumps({
        "status": document["status"],
        "coefficient": document["coefficient"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_EXECUTION_BOUNDARY",
    "SCHEMA_VERSION",
    "STATUS",
    "create",
    "validate_authorization_document",
]
