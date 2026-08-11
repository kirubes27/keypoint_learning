"""Create a fail-closed Task-80 OCR-ZNCC training authorization.

This module must be committed before Task-55 outcomes are observed.  At use
time it validates the exact Task-55 training authorization, re-loads and
re-validates all six live Task-55 outcome-cell artifacts, recomputes the frozen
paired Task-55 decision in memory, and requires the supplied Task-55 decision
artifact to agree exactly with that recomputation.  A fabricated or merely
self-hashed pass status cannot authorize Task 80.

The output only authorizes the bounded matched Task-80 experiment with the
same calibrated OCR-ZNCC coefficient and distinct-competitor radius one.  This
module has no training, optimizer, experiment-launch, Slurm, or GPU interface;
its subprocess use is limited to read-only Git provenance checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import ocr_zncc_outcome_evaluator as outcome_evaluator
from keypoint_net import ocr_zncc_task55_outcome_decision as task55_decision
from keypoint_net import ocr_zncc_training_authorization as training_authorization
from keypoint_net import ocr_zncc_training_decision as task55_training_decision


SCHEMA_VERSION = "ocr_zncc_task80_training_authorization.v1"
STATUS = "authorize_task80_matched_experiment"
EXPECTED_BRANCH = "agent/ocr-zncc-training-20260811"
RESEARCH_BASE_COMMIT = "4521538621b410421b1058603d090e2b09ec4178"
SOURCE_RELATIVE_PATHS = (
    "keypoint_net/ocr_zncc_task80_training_authorization.py",
    "keypoint_net/ocr_zncc_task55_outcome_decision.py",
    "keypoint_net/ocr_zncc_outcome_evaluator.py",
    "keypoint_net/ocr_zncc_training_authorization.py",
    "keypoint_net/ocr_zncc_training_decision.py",
)
EXPECTED_TASK55_DECISION_RULE = {
    "paired_repetition_unit": "seed",
    "seed_ids": [42, 43, 44],
    "descriptive_not_inferential": True,
    "selected_checkpoint_scope": (
        "each arm's own first minimum base-validation-loss best_model.pt"
    ),
    "selected_checkpoint_criteria": {
        "ocr_to_control_canonical_drift_ratio": "at_most_0.80_in_at_least_2_of_3",
        "ocr_roll_sign": "positive_in_all_3",
        "ocr_absolute_angle_error": "no_worse_than_control_in_at_least_2_of_3",
        "ocr_active_on_object_count": "not_lower_than_control_in_any_seed",
    },
    "final_checkpoint_scope": "fixed epoch-1000 final_model.pt",
    "final_checkpoint_decision_criterion": {
        "ocr_canonical_drift": "no_worse_than_control_in_at_least_2_of_3"
    },
    "final_operator_and_channel_health": "diagnostic_only_not_a_pass_substitute",
    "provenance_nonfinite_or_reflection_failure": "blocks",
    "criterion_scope_frozen_before_outcome_inspection": True,
    "no_significance_test": True,
}
TASK80_DECISION_TEXT = (
    "Run the predeclared matched Task80 control/OCR-ZNCC experiment only "
    "because the independently selected Task55 stage passed every frozen gate."
)
EXPECTED_TASK80_EXECUTION_BOUNDARY = {
    "authorized_recipe": "task80_assisted",
    "matched_arms": ["control", "ocr_zncc"],
    "seeds": [42, 43, 44],
    "epochs": 1000,
    "task55_rerun_authorized": False,
    "other_objects_or_operators_authorized": False,
    "test_split_authorized": False,
    "training_or_launch_performed_by_this_module": False,
}
EXPECTED_STATISTICAL_SCOPE = {
    "task55_seeds_are_descriptive_paired_repetitions": True,
    "inferential_test_performed": False,
}
_RECOMPUTED_FIELDS = (
    "outcome",
    "task55_pass",
    "task80_training_authorized_by_this_artifact",
    "criterion_results",
    "support_counts",
    "critical_blockers",
    "pairs",
    "final_checkpoint_diagnostics_not_used_for_decision",
)


class OCRTask80AuthorizationError(ValueError):
    """Raised when Task-55 evidence cannot authorize Task 80."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRTask80AuthorizationError(message)


def _load_json(path: Path | str, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return outcome_evaluator._load_json(path, name=name)  # noqa: SLF001
    except Exception as exc:
        if isinstance(exc, OCRTask80AuthorizationError):
            raise
        raise OCRTask80AuthorizationError(f"cannot validate {name}: {exc}") from exc


def _validate_content_hash(document: Mapping[str, Any], *, name: str) -> str:
    claimed = document.get("content_hash_sha256")
    _require(isinstance(claimed, str), f"{name} content hash is absent")
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    _require(
        outcome_evaluator.canonical_sha256(payload) == claimed,
        f"{name} content hash differs",
    )
    return claimed


def _git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _validate_source(
    repo_root: Path, *, expected_commit: str, expected_branch: str
) -> dict[str, Any]:
    _require(expected_branch == EXPECTED_BRANCH, "source branch is not the OCR experiment branch")
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    branch = _git(repo_root, "branch", "--show-current").stdout.strip()
    status = _git(repo_root, "status", "--porcelain").stdout.strip()
    _require(head == expected_commit, "live source commit differs from Task55 evidence")
    _require(branch == expected_branch, "live source branch differs from Task55 evidence")
    _require(status == "", "live checkout is not completely clean")
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        RESEARCH_BASE_COMMIT,
        expected_commit,
        check=False,
    )
    _require(ancestry.returncode == 0, "verified research base is not an ancestor")
    records = {
        relative: outcome_evaluator._committed_source_record(  # noqa: SLF001
            repo_root, relative, source_commit=expected_commit
        )
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "source_commit": expected_commit,
        "source_branch": expected_branch,
        "research_base_commit": RESEARCH_BASE_COMMIT,
        "research_base_is_ancestor": True,
        "full_status_clean": True,
        "committed_sources": records,
    }


def _validate_task55_training_authorization(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    repo_root: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    _require(repo_root is not None, "Task55 authorization requires live repository validation")
    try:
        validated = task55_training_decision.validate_authorization_document(
            repo_root, document
        )
    except Exception as exc:
        raise OCRTask80AuthorizationError(
            f"Task55 deep training authorization failed: {exc}"
        ) from exc
    _require(
        validated.get("authorization_valid") is True
        and validated.get("status") == "authorize_task55_matched_experiment"
        and validated.get("source_commit") == source_commit
        and validated.get("source_branch") == source_branch,
        "Task55 deep authorization lineage/status differs",
    )
    coefficient = float(validated["coefficient"])
    config = validated.get("ocr_zncc_config")
    _require(
        0.0 < coefficient <= 0.5
        and isinstance(config, Mapping)
        and config.get("peak_margin_exclusion_radius_cells") == 1,
        "Task55 deep coefficient/configuration differs",
    )
    return coefficient, {
        "coefficient_rule": dict(document["coefficient_rule"]),
        "ocr_zncc_config": dict(config),
    }


def _validate_bound_json_record(
    record: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{name} record is invalid")
    document, observed = _load_json(str(record.get("absolute_path", "")), name=name)
    _validate_content_hash(document, name=name)
    _require(
        dict(record) == _record_with_content_hash(observed, document),
        f"{name} record differs from live bytes",
    )
    return document


def _validate_bound_file_record(
    record: Mapping[str, Any], *, name: str
) -> tuple[bytes, dict[str, Any]]:
    _require(isinstance(record, Mapping), f"{name} record is invalid")
    try:
        data, observed = outcome_evaluator._read_regular(  # noqa: SLF001
            str(record.get("absolute_path", "")), name=name
        )
    except Exception as exc:
        raise OCRTask80AuthorizationError(f"cannot validate {name}: {exc}") from exc
    _require(dict(record) == observed, f"{name} record differs from live bytes")
    return data, observed


def _deep_validate_task55_training_authorization(
    repo_root: Path,
    document: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    coefficient: float,
) -> None:
    try:
        live_commit, live_branch = training_authorization._source_state(  # noqa: SLF001
            repo_root
        )
    except Exception as exc:
        raise OCRTask80AuthorizationError(
            f"Task55 deep source validation failed: {exc}"
        ) from exc
    _require(
        live_commit == source_commit and live_branch == source_branch,
        "Task55 deep source validation differs from authorization lineage",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "source_commit",
        "source_branch",
        "decision",
        "coverage_gate_status",
        "coefficient",
        "coefficient_rule",
        "ocr_zncc_config",
        "evidence",
        "execution_boundary",
        "review_interpretation",
        "review_execution",
        "content_hash_sha256",
    }
    _require(set(document) == expected_fields, "Task55 training authorization field set differs")
    _require(
        document.get("decision")
        == (
            "Test whether OCR-ZNCC at a formula-calibrated coefficient reduces "
            "validation material drift relative to an exact matched Task-55 control."
        ),
        "Task55 scientific decision statement differs",
    )
    _require(
        document.get("coverage_gate_status")
        == {
            "preregistered_50_percent_gate": "failed_unchanged",
            "interpretation": (
                "The threshold remains failed. Training is authorized only as a bounded "
                "paired causal experiment because accepted matches have correct material "
                "direction and the exact local training gradient passes."
            ),
        },
        "Task55 failed coverage-gate status differs",
    )
    evidence = document.get("evidence")
    expected_evidence_roles = {
        "stage0_provenance",
        "stage0_direction_summary",
        "exact_gradient_audit",
        "minibatch_gradient_audit",
        "fable_original_briefing",
        "fable_compiled_prompt",
        "fable_raw_review",
        "fable_high_helper",
    }
    _require(
        isinstance(evidence, Mapping) and set(evidence) == expected_evidence_roles,
        "Task55 evidence role set differs",
    )
    provenance = _validate_bound_json_record(
        evidence["stage0_provenance"], name="Task55 Stage0 provenance"
    )
    direction = _validate_bound_json_record(
        evidence["stage0_direction_summary"], name="Task55 Stage0 direction summary"
    )
    exact = _validate_bound_json_record(
        evidence["exact_gradient_audit"], name="Task55 exact-gradient audit"
    )
    minibatch = _validate_bound_json_record(
        evidence["minibatch_gradient_audit"], name="Task55 minibatch-gradient audit"
    )

    _require(
        provenance.get("schema_version")
        == "ocr_zncc_stage0.v2_distinct_competitor.provenance"
        and provenance.get("checkout", {}).get("commit") == source_commit
        and provenance.get("checkout", {}).get("branch") == source_branch
        and provenance.get("checkout", {}).get("authorized_base_commit")
        == training_authorization.AUTHORIZED_BASE_COMMIT
        and provenance.get("checkout", {}).get("authorized_base_is_ancestor") is True,
        "Task55 Stage0 source/base lineage differs",
    )
    _require(
        provenance.get("authorization_boundary", {}).get("optimizer_steps") == 0
        and provenance.get("authorization_boundary", {}).get("training_authorized") is False,
        "Task55 Stage0 execution boundary differs",
    )
    _transport_data, transport_record = outcome_evaluator._read_regular(  # noqa: SLF001
        repo_root / "keypoint_net/ocr_zncc_transport.py",
        name="live OCR-ZNCC transport source",
    )
    _require(
        provenance.get("source_files", {}).get("ocr_zncc_transport", {}).get("sha256")
        == transport_record["sha256"],
        "Task55 Stage0 matcher source differs from live committed source",
    )
    _require(
        direction.get("schema_version")
        == "ocr_zncc_stage0.v2_distinct_competitor.direction_summary"
        and direction.get("both_recipe_gates_pass") is False
        and direction.get("calibration_authorized_by_stage0_rule") is False,
        "Task55 failed Stage0 direction/coverage status differs",
    )
    direction_cells = []
    for recipe in ("task55_clean", "task80_assisted"):
        recipe_result = direction.get("recipe_results", {}).get(recipe)
        _require(
            isinstance(recipe_result, Mapping) and recipe_result.get("cell_count") == 3,
            f"Task55 Stage0 {recipe} cell count differs",
        )
        for cell in recipe_result.get("cells", []):
            summary = cell.get("summary", {})
            _require(
                summary.get("cell_pass") is False
                and float(summary["usable_direction_coverage"]) < 0.5
                and float(summary["direction_cosine"]["median"]) > 0.0
                and float(summary["one_cell_material_error_delta_objdiag"]["median"]) < 0.0
                and summary["active_on_object"]["change"] >= 0,
                "Task55 Stage0 direction cell gate differs",
            )
            direction_cells.append(cell.get("cell_id"))
    _require(len(direction_cells) == 6, "Task55 Stage0 direction cell set differs")

    _require(
        exact.get("schema_version")
        == "ocr_zncc_exact_coordinate_gradient_direction.v2"
        and exact.get("checkout", {}).get("commit") == source_commit,
        "Task55 exact-gradient source/schema differs",
    )
    exact_summary = exact.get("decision_summary", {})
    _require(
        exact_summary.get("all_frozen_infinitesimal_direction_gates") is True
        and exact_summary.get("all_initial_infinitesimal_direction_gates") is True
        and exact_summary.get("all_frozen_one_cell_finite_step_gates") is True
        and exact_summary.get("all_initial_one_cell_finite_step_gates") is False,
        "Task55 exact-gradient semantic gates differ",
    )
    radius_one_cells = [
        cell for cell in exact.get("cells", [])
        if cell.get("ocr_zncc_config", {}).get(
            "peak_margin_exclusion_radius_cells"
        ) == 1
    ]
    _require(
        len(radius_one_cells) == 12
        and all(
            cell.get("summary", {}).get(
                "accepted_source_patch_outside_image_count"
            ) == 0
            for cell in radius_one_cells
        ),
        "Task55 exact-gradient radius-one/source-geometry gate differs",
    )

    combined = minibatch.get("combined_summary", {})
    _require(
        minibatch.get("schema_version")
        == "ocr_zncc_training_minibatch_total_gradient_audit.v2"
        and minibatch.get("checkout", {}).get("commit") == source_commit
        and minibatch.get("ocr_zncc_config", {}).get(
            "peak_margin_exclusion_radius_cells"
        ) == 1
        and combined.get("all_finite") is True
        and combined.get("all_base_gradients_nonzero") is True
        and combined.get("all_accepted_source_patches_inside_image") is True,
        "Task55 minibatch-gradient validity/source/config differs",
    )
    _require(
        minibatch.get("recommended_coefficient") == coefficient
        and document.get("coefficient_rule") == minibatch.get("coefficient_rule")
        and document.get("ocr_zncc_config") == minibatch.get("ocr_zncc_config")
        and abs(float(combined.get("recommended_scaled_median_contribution")) - 0.10)
        <= 1e-12,
        "Task55 coefficient/configuration differs from live minibatch calibration",
    )

    briefing, _ = _validate_bound_file_record(
        evidence["fable_original_briefing"], name="Task55 Fable briefing"
    )
    prompt, _ = _validate_bound_file_record(
        evidence["fable_compiled_prompt"], name="Task55 Fable compiled prompt"
    )
    raw, raw_record = _validate_bound_file_record(
        evidence["fable_raw_review"], name="Task55 Fable raw review"
    )
    _helper, helper_record = _validate_bound_file_record(
        evidence["fable_high_helper"], name="Task55 Fable High helper"
    )
    _require(
        helper_record["sha256"] == task55_training_decision.FABLE_HELPER_SHA256,
        "Task55 Fable helper source differs",
    )
    _require(
        prompt == task55_training_decision.FABLE_PROMPT_PREFIX + briefing,
        "Task55 Fable prompt is not the exact compiled independent briefing",
    )
    try:
        raw_text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OCRTask80AuthorizationError("Task55 Fable raw review is not UTF-8") from exc
    _require(
        raw_record["size_bytes"] >= 1000
        and raw_text.strip().lower()
        not in {
            "credit balance is too low",
            "not logged in",
            "authentication failed",
            "rate limit",
        },
        "Task55 Fable review is missing, non-substantive, or a service error",
    )
    _require(
        document.get("review_execution")
        == {
            "reviewer": "Fable",
            "model_alias": "fable",
            "effort": "high",
            "review_mode": "calibrated_adversarial_red_team",
            "permission_mode": "plan_read_only",
            "session_persistence": False,
            "single_call_no_retry": True,
            "codex_draft_or_conclusion_supplied": False,
            "prior_external_reviewer_examples_may_be_supplied": True,
            "agreement_is_not_evidence": True,
        },
        "Task55 Fable review execution contract differs",
    )
    _require(
        document.get("review_interpretation")
        == (
            "The calibrated Fable High report is advisory. Every decision-changing "
            "claim above was revalidated from the bound primary artifacts."
        ),
        "Task55 review interpretation differs",
    )


def _paired_evidence_matches(
    results_by_cell: Mapping[str, Mapping[str, Any]], *, seed: int
) -> bool:
    control = results_by_cell[f"task55_clean__control__seed{seed}"]
    ocr = results_by_cell[f"task55_clean__ocr_zncc__seed{seed}"]
    control_pairing = control["lineage"]["matched_pairing_evidence"]
    ocr_pairing = ocr["lineage"]["matched_pairing_evidence"]
    task55_decision.validate_matched_pairing_evidence(
        control_pairing,
        ocr_pairing,
        recipe="task55_clean",
        seed=seed,
    )
    return True


def _recompute_task55(
    repo_root: Path, result_paths: Sequence[Path | str]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[str, Any],
    list[task55_decision.CellEvidence],
]:
    _require(len(result_paths) == 6, "exactly six Task55 outcome cells are required")
    cells: list[task55_decision.CellEvidence] = []
    results_by_cell: dict[str, Mapping[str, Any]] = {}
    result_records: list[dict[str, Any]] = []
    manifest_records: list[Mapping[str, Any]] = []
    for path in result_paths:
        try:
            evidence, result, record = task55_decision._load_and_bind_result(  # noqa: SLF001
                repo_root, Path(path)
            )
        except Exception as exc:
            raise OCRTask80AuthorizationError(
                f"Task55 outcome cell failed live validation: {exc}"
            ) from exc
        _require(evidence.cell_id not in results_by_cell, "Task55 outcome cell is duplicated")
        cells.append(evidence)
        results_by_cell[evidence.cell_id] = result
        result_records.append(record)
        manifest_records.append(result["lineage"]["experiment_manifest"])
    expected_ids = {
        f"task55_clean__{arm}__seed{seed}"
        for arm in ("control", "ocr_zncc")
        for seed in (42, 43, 44)
    }
    _require(set(results_by_cell) == expected_ids, "Task55 outcome cell set differs")
    _require(
        all(record == manifest_records[0] for record in manifest_records),
        "Task55 outcome cells bind different manifests",
    )
    _require(
        all(_paired_evidence_matches(results_by_cell, seed=seed) for seed in (42, 43, 44)),
        "Task55 control/OCR paired execution evidence differs",
    )
    recomputed = task55_decision.decide(cells)
    return (
        recomputed,
        sorted(result_records, key=lambda row: row["cell_id"]),
        manifest_records[0],
        cells,
    )


def _validate_supplied_task55_decision(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    recomputed: Mapping[str, Any],
    result_records: Sequence[Mapping[str, Any]],
    manifest_record: Mapping[str, Any],
    decision_source_record: Mapping[str, Any],
) -> None:
    _validate_content_hash(document, name="Task55 paired outcome decision")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "source_commit",
        "source_branch",
        "decision_rule",
        "experiment_manifest",
        "cell_result_records",
        "committed_decision_source",
        *_RECOMPUTED_FIELDS,
        "next_phase_started",
        "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "Task55 outcome decision field set differs")
    _require(
        document.get("schema_version") == task55_decision.SCHEMA_VERSION
        and document.get("artifact_type")
        == "frozen_task55_paired_ocr_zncc_outcome_decision",
        "Task55 outcome decision identity differs",
    )
    _require(
        document.get("source_commit") == source_commit
        and document.get("source_branch") == source_branch,
        "Task55 outcome decision source differs",
    )
    _require(
        document.get("decision_rule") == EXPECTED_TASK55_DECISION_RULE,
        "Task55 outcome decision rule differs",
    )
    _require(
        document.get("experiment_manifest") == manifest_record,
        "Task55 outcome decision manifest binding differs",
    )
    _require(
        document.get("cell_result_records") == list(result_records),
        "Task55 outcome decision cell-result bindings differ",
    )
    _require(
        document.get("committed_decision_source") == decision_source_record,
        "Task55 outcome decision source-file binding differs",
    )
    for field in _RECOMPUTED_FIELDS:
        _require(
            document.get(field) == recomputed.get(field),
            f"Task55 outcome decision disagrees with live recomputation: {field}",
        )
    _require(
        recomputed.get("task55_pass") is True
        and recomputed.get("outcome")
        == "task55_pass_candidate_for_separate_task80_authorization"
        and recomputed.get("critical_blockers") == []
        and recomputed.get("criterion_results", {}).get(
            "no_provenance_nonfinite_or_reflection_blocker"
        ) is True,
        "live Task55 recomputation does not pass without blockers",
    )
    _require(
        document.get("next_phase_started") is False
        and document.get("task80_training_authorized_by_this_artifact") is False,
        "Task55 decision boundary differs",
    )


def _record_with_content_hash(
    observed: Mapping[str, Any], document: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(observed),
        "content_hash_sha256": document["content_hash_sha256"],
    }


def _revalidate_task55_prerequisites(
    repo_root: Path,
    prerequisites: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
) -> dict[str, Any]:
    expected_keys = {
        "training_authorization",
        "outcome_cells",
        "paired_outcome_decision",
        "live_recomputation",
        "experiment_manifest",
    }
    _require(set(prerequisites) == expected_keys, "Task55 prerequisite field set differs")

    training_claim = prerequisites.get("training_authorization")
    _require(isinstance(training_claim, Mapping), "Task55 training prerequisite is invalid")
    training_document, training_observed = _load_json(
        str(training_claim.get("absolute_path", "")),
        name="Task55 prerequisite training authorization",
    )
    _validate_content_hash(
        training_document, name="Task55 prerequisite training authorization"
    )
    _require(
        dict(training_claim)
        == _record_with_content_hash(training_observed, training_document),
        "Task55 training prerequisite record differs from live bytes",
    )
    coefficient, inherited = _validate_task55_training_authorization(
        training_document,
        source_commit=source_commit,
        source_branch=source_branch,
        repo_root=repo_root,
    )

    outcome_claims = prerequisites.get("outcome_cells")
    _require(
        isinstance(outcome_claims, list) and len(outcome_claims) == 6,
        "Task55 outcome prerequisite set is invalid",
    )
    outcome_paths: list[str] = []
    for claim in outcome_claims:
        _require(isinstance(claim, Mapping), "Task55 outcome prerequisite record is invalid")
        absolute_path = claim.get("absolute_path")
        _require(
            isinstance(absolute_path, str) and Path(absolute_path).is_absolute(),
            "Task55 outcome prerequisite path is not absolute",
        )
        outcome_paths.append(absolute_path)
    recomputed, result_records, manifest_record, _cells = _recompute_task55(
        repo_root, outcome_paths
    )
    _require(
        outcome_claims == result_records,
        "Task55 outcome prerequisite records differ from live validation",
    )
    _require(
        prerequisites.get("experiment_manifest") == manifest_record,
        "Task55 prerequisite manifest differs from live outcomes",
    )

    manifest, observed_manifest = _load_json(
        str(manifest_record.get("absolute_path", "")),
        name="Task55 prerequisite experiment manifest",
    )
    _validate_content_hash(manifest, name="Task55 prerequisite experiment manifest")
    _require(
        observed_manifest.get("sha256") == manifest_record.get("sha256")
        and observed_manifest.get("size_bytes") == manifest_record.get("size_bytes")
        and manifest.get("source_commit") == source_commit
        and manifest.get("source_branch") == source_branch,
        "Task55 prerequisite manifest live/source binding differs",
    )
    _require(
        manifest.get("authorization_decision") == training_observed,
        "Task55 prerequisite manifest does not bind its live training authorization",
    )

    decision_claim = prerequisites.get("paired_outcome_decision")
    _require(isinstance(decision_claim, Mapping), "Task55 decision prerequisite is invalid")
    decision_document, decision_observed = _load_json(
        str(decision_claim.get("absolute_path", "")),
        name="Task55 prerequisite paired outcome decision",
    )
    _validate_content_hash(
        decision_document, name="Task55 prerequisite paired outcome decision"
    )
    _require(
        dict(decision_claim)
        == _record_with_content_hash(decision_observed, decision_document),
        "Task55 decision prerequisite record differs from live bytes",
    )
    decision_source_record = outcome_evaluator._committed_source_record(  # noqa: SLF001
        repo_root,
        "keypoint_net/ocr_zncc_task55_outcome_decision.py",
        source_commit=source_commit,
    )
    _validate_supplied_task55_decision(
        decision_document,
        source_commit=source_commit,
        source_branch=source_branch,
        recomputed=recomputed,
        result_records=result_records,
        manifest_record=manifest_record,
        decision_source_record=decision_source_record,
    )

    expected_recomputation = {
        "outcome": recomputed["outcome"],
        "task55_pass": recomputed["task55_pass"],
        "criterion_results": recomputed["criterion_results"],
        "support_counts": recomputed["support_counts"],
        "critical_blockers": recomputed["critical_blockers"],
        "supplied_decision_matches_recomputation": True,
    }
    _require(
        prerequisites.get("live_recomputation") == expected_recomputation,
        "stored Task55 recomputation summary differs from live recomputation",
    )
    return {
        "coefficient": coefficient,
        "coefficient_rule": inherited["coefficient_rule"],
        "ocr_zncc_config": inherited["ocr_zncc_config"],
        "task55_training_authorization": _record_with_content_hash(
            training_observed, training_document
        ),
        "task55_outcome_records": result_records,
        "task55_outcome_decision": _record_with_content_hash(
            decision_observed, decision_document
        ),
        "task55_manifest": manifest_record,
        "task55_recomputation": expected_recomputation,
    }


def validate_authorization_document(
    repo_root: Path | str, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate a Task-80 authorization at the train-time use boundary.

    All paths embedded in the authorization are reopened and rehashed.  The
    six Task-55 cell results are re-evaluated through their completed-run
    receipts, the paired Task-55 decision is recomputed, and the inherited
    coefficient/configuration and exact Task-80 boundary are checked again.
    """

    _require(isinstance(document, Mapping), "Task80 authorization is not a mapping")
    _validate_content_hash(document, name="Task80 training authorization")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "source_commit",
        "source_branch",
        "research_base",
        "decision",
        "coefficient",
        "coefficient_rule",
        "ocr_zncc_config",
        "task55_prerequisites",
        "source_binding",
        "execution_boundary",
        "statistical_scope",
        "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "Task80 authorization field set differs")
    _require(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("artifact_type")
        == "bounded_task80_ocr_zncc_training_authorization"
        and document.get("status") == STATUS,
        "Task80 authorization identity/status differs",
    )
    source_commit = document.get("source_commit")
    source_branch = document.get("source_branch")
    _require(isinstance(source_commit, str), "Task80 source commit is invalid")
    _require(isinstance(source_branch, str), "Task80 source branch is invalid")
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _validate_source(
        root, expected_commit=source_commit, expected_branch=source_branch
    )
    _require(document.get("source_binding") == source, "Task80 source binding differs")
    _require(
        document.get("research_base")
        == {
            "commit": RESEARCH_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "Task80 research-base binding differs",
    )
    _require(document.get("decision") == TASK80_DECISION_TEXT,
             "Task80 decision statement differs")
    prerequisites = document.get("task55_prerequisites")
    _require(isinstance(prerequisites, Mapping), "Task55 prerequisites are invalid")
    live = _revalidate_task55_prerequisites(
        root,
        prerequisites,
        source_commit=source_commit,
        source_branch=source_branch,
    )
    _require(
        document.get("coefficient") == live["coefficient"],
        "Task80 coefficient differs from live Task55 authorization",
    )
    _require(
        document.get("coefficient_rule") == live["coefficient_rule"],
        "Task80 coefficient rule differs from live Task55 authorization",
    )
    _require(
        document.get("ocr_zncc_config") == live["ocr_zncc_config"]
        and document.get("ocr_zncc_config", {}).get(
            "peak_margin_exclusion_radius_cells"
        ) == 1,
        "Task80 OCR-ZNCC configuration differs from radius-one Task55",
    )
    _require(
        document.get("execution_boundary") == EXPECTED_TASK80_EXECUTION_BOUNDARY,
        "Task80 execution boundary differs",
    )
    _require(
        document.get("statistical_scope") == EXPECTED_STATISTICAL_SCOPE,
        "Task80 statistical scope differs",
    )
    return {
        "authorization_valid": True,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "research_base_commit": RESEARCH_BASE_COMMIT,
        "coefficient": live["coefficient"],
        "ocr_zncc_config": live["ocr_zncc_config"],
        "task55_outcome": live["task55_recomputation"]["outcome"],
        "task55_pass": live["task55_recomputation"]["task55_pass"],
        "critical_blockers": live["task55_recomputation"]["critical_blockers"],
        "training_or_launch_performed": False,
    }


def _write_exclusive(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = outcome_evaluator.canonical_json_bytes(document) + b"\n"
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCRTask80AuthorizationError(f"cannot exclusively create {absolute}") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "exclusive Task80 authorization write made no progress")
            offset += written
    finally:
        os.close(descriptor)
    return {
        "absolute_path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "write_mode": "exclusive_create_no_overwrite",
    }


def create(
    *,
    repo_root: Path | str,
    task55_training_authorization_path: Path | str,
    task55_result_paths: Sequence[Path | str],
    task55_outcome_decision_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    task55_training, task55_training_record = _load_json(
        task55_training_authorization_path, name="Task55 training authorization"
    )
    source_commit = task55_training.get("source_commit")
    source_branch = task55_training.get("source_branch")
    _require(isinstance(source_commit, str), "Task55 source commit is invalid")
    _require(isinstance(source_branch, str), "Task55 source branch is invalid")
    source = _validate_source(
        root, expected_commit=source_commit, expected_branch=source_branch
    )
    coefficient, inherited = _validate_task55_training_authorization(
        task55_training,
        source_commit=source_commit,
        source_branch=source_branch,
        repo_root=root,
    )

    recomputed, result_records, manifest_record, _cells = _recompute_task55(
        root, task55_result_paths
    )
    manifest, observed_manifest = _load_json(
        str(manifest_record.get("absolute_path", "")), name="Task55 experiment manifest"
    )
    _validate_content_hash(manifest, name="Task55 experiment manifest")
    _require(
        observed_manifest.get("sha256") == manifest_record.get("sha256")
        and manifest.get("source_commit") == source_commit
        and manifest.get("source_branch") == source_branch,
        "Task55 manifest live/source binding differs",
    )
    _require(
        manifest.get("authorization_decision") == task55_training_record,
        "Task55 manifest does not bind the supplied training authorization",
    )

    supplied_decision, supplied_decision_record = _load_json(
        task55_outcome_decision_path, name="Task55 paired outcome decision"
    )
    decision_source_record = outcome_evaluator._committed_source_record(  # noqa: SLF001
        root,
        "keypoint_net/ocr_zncc_task55_outcome_decision.py",
        source_commit=source_commit,
    )
    _validate_supplied_task55_decision(
        supplied_decision,
        source_commit=source_commit,
        source_branch=source_branch,
        recomputed=recomputed,
        result_records=result_records,
        manifest_record=manifest_record,
        decision_source_record=decision_source_record,
    )

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "bounded_task80_ocr_zncc_training_authorization",
        "status": STATUS,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "research_base": {
            "commit": RESEARCH_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "decision": TASK80_DECISION_TEXT,
        "coefficient": coefficient,
        "coefficient_rule": inherited["coefficient_rule"],
        "ocr_zncc_config": inherited["ocr_zncc_config"],
        "task55_prerequisites": {
            "training_authorization": {
                **task55_training_record,
                "content_hash_sha256": task55_training["content_hash_sha256"],
            },
            "outcome_cells": list(result_records),
            "paired_outcome_decision": {
                **supplied_decision_record,
                "content_hash_sha256": supplied_decision["content_hash_sha256"],
            },
            "live_recomputation": {
                "outcome": recomputed["outcome"],
                "task55_pass": recomputed["task55_pass"],
                "criterion_results": recomputed["criterion_results"],
                "support_counts": recomputed["support_counts"],
                "critical_blockers": recomputed["critical_blockers"],
                "supplied_decision_matches_recomputation": True,
            },
            "experiment_manifest": dict(manifest_record),
        },
        "source_binding": source,
        "execution_boundary": dict(EXPECTED_TASK80_EXECUTION_BOUNDARY),
        "statistical_scope": dict(EXPECTED_STATISTICAL_SCOPE),
    }
    document["content_hash_sha256"] = outcome_evaluator.canonical_sha256(document)
    output_record = _write_exclusive(Path(output_path), document)
    return {"document": document, "output_record": output_record}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task55-training-authorization", type=Path, required=True)
    parser.add_argument("--task55-result", type=Path, action="append", required=True)
    parser.add_argument("--task55-outcome-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = create(
        repo_root=args.repo_root,
        task55_training_authorization_path=args.task55_training_authorization,
        task55_result_paths=args.task55_result,
        task55_outcome_decision_path=args.task55_outcome_decision,
        output_path=args.output,
    )
    print(json.dumps(
        {
            **result["output_record"],
            "schema_version": result["document"]["schema_version"],
            "status": result["document"]["status"],
            "coefficient": result["document"]["coefficient"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_BRANCH",
    "EXPECTED_TASK80_EXECUTION_BOUNDARY",
    "EXPECTED_TASK55_DECISION_RULE",
    "OCRTask80AuthorizationError",
    "RESEARCH_BASE_COMMIT",
    "SCHEMA_VERSION",
    "STATUS",
    "create",
    "validate_authorization_document",
]
