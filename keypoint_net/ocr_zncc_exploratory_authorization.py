"""Explicit user-authorized OCR-ZNCC exploration for Task 55 and Task 80.

This is deliberately separate from the preregistered Stage-0 authorization.
The 50% coverage gate remains failed.  The only authority added here is the
user's explicit request to run a bounded matched experiment anyway; every
scientific input is still recomputed from the exact zero-step evidence rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import ocr_zncc_training_decision as task55_decision


SCHEMA_VERSION = "ocr_zncc_dual_recipe_exploratory_authorization.v1"
STATUS = "authorize_dual_recipe_exploratory_matched_experiment"
USER_OVERRIDE_SCHEMA_VERSION = "ocr_zncc_user_exploratory_override.v1"
USER_OVERRIDE_STATUS = "explicit_user_authorized_dual_recipe_exploration"
SOURCE_THREAD_ID = "019fce51-33ee-7c80-94bc-0deedcd793b3"
USER_AUTHORIZATION_TEXT = (
    "uk what just start teh trainig and lets see what happens anyway, just "
    "introdoce hte loss on both 55 and 80 and lets see what happens, but yea "
    "do it properly uk, can you point me again to the results wher we trained "
    "the decscriptor loss becasue i didnt really see ti or forgor to see the "
    "visualisations, but yea train it wiht this loss and lets see if it improves"
)
AUTHORIZED_RECIPES = ("task55_clean", "task80_assisted")
EXPECTED_CELL_IDS = tuple(task55_decision.EXPECTED_CELL_IDS)

EXPERIMENT_SCOPE = {
    "object": "engineers_hammer_vray",
    "transform_family": "roll",
    "physical_axis": "world_z",
    "signed_generator_degrees": 6.0,
    "stride_frames": 3,
    "head_resolution": 64,
    "recipes": list(AUTHORIZED_RECIPES),
    "seeds": [42, 43, 44],
    "arms": ["control", "ocr_zncc"],
    "scientific_cell_count": 12,
    "epochs_per_cell": 1000,
    "test_split_used": False,
}

EXECUTION_BOUNDARY = {
    "purpose": "exploratory_causal_test_after_failed_coverage_gate",
    "coverage_gate": "failed_unchanged_not_relabelled_as_pass",
    "task80_is_not_conditioned_on_task55_outcome": True,
    "gpu_smoke_required_separately_for_each_recipe": True,
    "pair_execution": "control_then_ocr_sequential_on_same_allocated_gpu",
    "checkpoint_selector": "minimum_base_validation_loss_first_minimum",
    "fixed_final_estimand": "epoch_1000_final_model",
    "validation_aggregation": "sample_weighted_complete_21_pair_mean",
    "raw_lambda_sweep_performed": False,
    "other_objects_or_transforms_authorized": False,
}

STATISTICAL_SCOPE = {
    "sample_unit": "matched_training_seed",
    "paired_seed_count_per_recipe": 3,
    "inference": "descriptive_exploratory_not_population_inference",
    "primary_outcome": "canonical_drift.median_rms_objdiag",
    "primary_population": "all_10_channels_on_24_frozen_validation_frames",
}


class OCRExploratoryAuthorizationError(ValueError):
    """Raised when the explicit exploratory authorization is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRExploratoryAuthorizationError(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_exclusive(path: Path | str, document: Mapping[str, Any]) -> None:
    absolute = Path(path).expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCRExploratoryAuthorizationError(
            f"cannot exclusively create {absolute}"
        ) from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "exclusive exploratory write made no progress")
            offset += written
    finally:
        os.close(descriptor)


def _source_binding(root: Path) -> dict[str, Any]:
    return task55_decision._source_binding(root)  # noqa: SLF001


def _validate_user_override_document(
    document: Mapping[str, Any], *, source_commit: str, source_branch: str
) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "artifact_type", "status", "source_commit",
        "source_branch", "source_thread_id", "user_authorization_text",
        "decision", "experiment_scope", "coverage_acknowledgement",
        "created_utc", "content_hash_sha256",
    }
    _require(set(document) == expected_fields, "user override field set differs")
    authorization._validate_content_hash(  # noqa: SLF001
        document, name="OCR exploratory user override"
    )
    _require(
        document.get("schema_version") == USER_OVERRIDE_SCHEMA_VERSION
        and document.get("artifact_type")
        == "explicit_user_ocr_zncc_exploratory_override"
        and document.get("status") == USER_OVERRIDE_STATUS,
        "user override identity/status differs",
    )
    _require(
        document.get("source_commit") == source_commit
        and document.get("source_branch") == source_branch
        and document.get("source_thread_id") == SOURCE_THREAD_ID
        and document.get("user_authorization_text") == USER_AUTHORIZATION_TEXT,
        "user override source/instruction differs",
    )
    _require(
        document.get("decision")
        == "Run matched control-versus-OCR training for both Task 55 and Task 80."
        and document.get("experiment_scope") == EXPERIMENT_SCOPE
        and document.get("coverage_acknowledgement")
        == {
            "preregistered_50_percent_gate": "failed_unchanged",
            "override_meaning": (
                "permission_for_an_exploratory_training_test_not_evidence_that_"
                "the_stage0_semantic_gate_passed"
            ),
        },
        "user override decision/scope differs",
    )
    created = document.get("created_utc")
    _require(
        isinstance(created, str) and created.endswith("Z") and "T" in created,
        "user override timestamp is invalid",
    )
    return {
        "user_override_valid": True,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "authorized_recipes": list(AUTHORIZED_RECIPES),
    }


def create_user_override(
    *, repo_root: Path | str, output_path: Path | str
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _source_binding(root)
    document: dict[str, Any] = {
        "schema_version": USER_OVERRIDE_SCHEMA_VERSION,
        "artifact_type": "explicit_user_ocr_zncc_exploratory_override",
        "status": USER_OVERRIDE_STATUS,
        "source_commit": source["source_commit"],
        "source_branch": source["source_branch"],
        "source_thread_id": SOURCE_THREAD_ID,
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "decision": (
            "Run matched control-versus-OCR training for both Task 55 and Task 80."
        ),
        "experiment_scope": EXPERIMENT_SCOPE,
        "coverage_acknowledgement": {
            "preregistered_50_percent_gate": "failed_unchanged",
            "override_meaning": (
                "permission_for_an_exploratory_training_test_not_evidence_that_"
                "the_stage0_semantic_gate_passed"
            ),
        },
        "created_utc": _utc_now(),
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    _validate_user_override_document(
        document,
        source_commit=source["source_commit"],
        source_branch=source["source_branch"],
    )
    _write_exclusive(output_path, document)
    return document


def _load_primary_evidence(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_fields = {
        "stage0_provenance", "stage0_direction_summary",
        "stage0_direction_cells", "exact_gradient_audit",
        "minibatch_gradient_audit",
    }
    _require(set(evidence) == expected_fields, "exploratory evidence field set differs")
    loaded: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for field, logical_name in (
        ("stage0_provenance", "stage0_provenance"),
        ("stage0_direction_summary", "stage0_direction_summary"),
        ("exact_gradient_audit", "exact_gradient_audit"),
        ("minibatch_gradient_audit", "minibatch_gradient_audit"),
    ):
        document, record = task55_decision._load_claimed_json(  # noqa: SLF001
            evidence[field], logical_name=logical_name
        )
        loaded[field] = document
        records[field] = record
    cell_claims = evidence["stage0_direction_cells"]
    _require(
        isinstance(cell_claims, list) and len(cell_claims) == len(EXPECTED_CELL_IDS),
        "exploratory direction-cell record count differs",
    )
    cells: dict[str, Mapping[str, Any]] = {}
    cell_records: dict[str, Mapping[str, Any]] = {}
    for claim in cell_claims:
        _require(isinstance(claim, Mapping), "direction-cell record is invalid")
        logical_name = str(claim.get("logical_name", ""))
        cell_id = logical_name.removeprefix("stage0_direction_cell:")
        _require(
            cell_id in EXPECTED_CELL_IDS and cell_id not in cells,
            "exploratory direction-cell identity differs",
        )
        cell, record = task55_decision._load_claimed_json(  # noqa: SLF001
            claim, logical_name=logical_name
        )
        cells[cell_id] = cell
        cell_records[cell_id] = record
    loaded["stage0_direction_cells"] = cells
    records["stage0_direction_cells"] = cell_records
    return loaded, records


def _recompute_primary(
    root: Path, loaded: Mapping[str, Any], records: Mapping[str, Any]
) -> dict[str, Any]:
    result = task55_decision._validate_primary_evidence(  # noqa: SLF001
        root,
        provenance=loaded["stage0_provenance"],
        direction_summary=loaded["stage0_direction_summary"],
        direction_cells=loaded["stage0_direction_cells"],
        direction_records=records["stage0_direction_cells"],
        exact=loaded["exact_gradient_audit"],
        minibatch=loaded["minibatch_gradient_audit"],
    )
    _require(
        result["stage0_direction"]["both_recipe_gates_pass"] is False
        and result["stage0_direction"]["calibration_authorized_by_stage0_rule"]
        is False,
        "failed Stage0 coverage status was not preserved",
    )
    exact = result["exact_gradient_decision"]
    _require(
        exact["all_frozen_infinitesimal_direction_gates"] is True
        and exact["all_frozen_one_cell_finite_step_gates"] is True
        and exact["all_initial_infinitesimal_direction_gates"] is True
        and exact["frozen_50_percent_coverage_gate_status"]
        == "remains_failed_unchanged"
        and exact["training_authorized_by_artifact"] is False,
        "exact-gradient exploratory evidence differs",
    )
    minibatch = result["minibatch_gradient"]
    coefficient = minibatch["recommended_coefficient"]
    _require(
        isinstance(coefficient, (int, float))
        and not isinstance(coefficient, bool)
        and math.isfinite(float(coefficient))
        and 0.0 < float(coefficient) <= 0.5,
        "exploratory coefficient is invalid",
    )
    return result


def validate_authorization_document(
    repo_root: Path | str, document: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "artifact_type", "status", "source_commit",
        "source_branch", "research_base", "source_binding", "decision",
        "authorized_recipes", "experiment_scope", "coverage_gate_status",
        "coefficient", "coefficient_rule", "ocr_zncc_config", "evidence",
        "primary_recomputation", "user_override", "execution_boundary",
        "statistical_scope", "content_hash_sha256",
    }
    _require(set(document) == expected_fields, "dual authorization field set differs")
    authorization._validate_content_hash(  # noqa: SLF001
        document, name="dual exploratory authorization"
    )
    _require(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("artifact_type")
        == "dual_recipe_exploratory_ocr_zncc_training_authorization"
        and document.get("status") == STATUS,
        "dual exploratory authorization identity/status differs",
    )
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _source_binding(root)
    _require(
        document.get("source_commit") == source["source_commit"]
        and document.get("source_branch") == source["source_branch"]
        and document.get("source_binding") == source,
        "dual exploratory source binding differs",
    )
    _require(
        document.get("research_base")
        == {
            "commit": authorization.AUTHORIZED_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "dual exploratory research-base binding differs",
    )
    _require(
        document.get("decision")
        == (
            "Run both Task 55 and Task 80 as matched exploratory control-versus-"
            "OCR experiments without relabelling the failed Stage0 coverage gate."
        )
        and document.get("authorized_recipes") == list(AUTHORIZED_RECIPES)
        and document.get("experiment_scope") == EXPERIMENT_SCOPE
        and document.get("coverage_gate_status")
        == {
            "preregistered_50_percent_gate": "failed_unchanged",
            "training_interpretation": "exploratory_user_override",
        }
        and document.get("execution_boundary") == EXECUTION_BOUNDARY
        and document.get("statistical_scope") == STATISTICAL_SCOPE,
        "dual exploratory decision/scope differs",
    )
    evidence = document.get("evidence")
    _require(isinstance(evidence, Mapping), "dual exploratory evidence is invalid")
    loaded, records = _load_primary_evidence(evidence)
    recomputed = _recompute_primary(root, loaded, records)
    _require(
        task55_decision._equivalent(  # noqa: SLF001
            document.get("primary_recomputation"), recomputed
        ),
        "dual stored primary recomputation differs from live rows",
    )
    minibatch = recomputed["minibatch_gradient"]
    coefficient = float(minibatch["recommended_coefficient"])
    _require(
        isinstance(document.get("coefficient"), (int, float))
        and not isinstance(document.get("coefficient"), bool)
        and math.isclose(
            float(document["coefficient"]), coefficient,
            rel_tol=1e-12, abs_tol=1e-12,
        )
        and document.get("coefficient_rule") == minibatch["coefficient_rule"]
        and document.get("ocr_zncc_config") == minibatch["ocr_zncc_config"],
        "dual exploratory coefficient/configuration differs",
    )
    override_claim = document.get("user_override")
    _require(isinstance(override_claim, Mapping), "user override record is invalid")
    override, observed_override = task55_decision._load_claimed_json(  # noqa: SLF001
        override_claim, logical_name="user_exploratory_override"
    )
    del observed_override
    _validate_user_override_document(
        override,
        source_commit=source["source_commit"],
        source_branch=source["source_branch"],
    )
    return {
        "authorization_valid": True,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "source_commit": source["source_commit"],
        "source_branch": source["source_branch"],
        "authorized_recipes": list(AUTHORIZED_RECIPES),
        "coefficient": coefficient,
        "ocr_zncc_config": minibatch["ocr_zncc_config"],
        "coverage_gate_passed": False,
        "training_interpretation": "exploratory_user_override",
        "training_or_launch_performed": False,
    }


def create_authorization(
    *,
    repo_root: Path | str,
    stage0_provenance_path: Path | str,
    direction_summary_path: Path | str,
    direction_cell_paths: Sequence[Path | str],
    exact_gradient_path: Path | str,
    minibatch_gradient_path: Path | str,
    user_override_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    source = _source_binding(root)
    evidence: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for field, logical_name, path in (
        ("stage0_provenance", "stage0_provenance", stage0_provenance_path),
        ("stage0_direction_summary", "stage0_direction_summary", direction_summary_path),
        ("exact_gradient_audit", "exact_gradient_audit", exact_gradient_path),
        ("minibatch_gradient_audit", "minibatch_gradient_audit", minibatch_gradient_path),
    ):
        value, record = task55_decision._json_record(  # noqa: SLF001
            path, logical_name=logical_name
        )
        evidence[field] = record
        loaded[field] = value
        records[field] = record
    _require(
        len(direction_cell_paths) == len(EXPECTED_CELL_IDS),
        "exactly six exploratory direction cells are required",
    )
    cells: dict[str, Mapping[str, Any]] = {}
    cell_records: dict[str, Mapping[str, Any]] = {}
    for path in direction_cell_paths:
        value, record = task55_decision._json_record(  # noqa: SLF001
            path, logical_name="temporary_direction_cell"
        )
        cell_id = str(value.get("cell_id", ""))
        _require(
            cell_id in EXPECTED_CELL_IDS and cell_id not in cells,
            "exploratory direction-cell creation set differs",
        )
        record["logical_name"] = f"stage0_direction_cell:{cell_id}"
        cells[cell_id] = value
        cell_records[cell_id] = record
    loaded["stage0_direction_cells"] = cells
    records["stage0_direction_cells"] = cell_records
    evidence["stage0_direction_cells"] = [
        cell_records[cell_id] for cell_id in EXPECTED_CELL_IDS
    ]
    recomputed = _recompute_primary(root, loaded, records)
    override, override_record = task55_decision._json_record(  # noqa: SLF001
        user_override_path, logical_name="user_exploratory_override"
    )
    _validate_user_override_document(
        override,
        source_commit=source["source_commit"],
        source_branch=source["source_branch"],
    )
    minibatch = recomputed["minibatch_gradient"]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "dual_recipe_exploratory_ocr_zncc_training_authorization",
        "status": STATUS,
        "source_commit": source["source_commit"],
        "source_branch": source["source_branch"],
        "research_base": {
            "commit": authorization.AUTHORIZED_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "source_binding": source,
        "decision": (
            "Run both Task 55 and Task 80 as matched exploratory control-versus-"
            "OCR experiments without relabelling the failed Stage0 coverage gate."
        ),
        "authorized_recipes": list(AUTHORIZED_RECIPES),
        "experiment_scope": EXPERIMENT_SCOPE,
        "coverage_gate_status": {
            "preregistered_50_percent_gate": "failed_unchanged",
            "training_interpretation": "exploratory_user_override",
        },
        "coefficient": minibatch["recommended_coefficient"],
        "coefficient_rule": minibatch["coefficient_rule"],
        "ocr_zncc_config": minibatch["ocr_zncc_config"],
        "evidence": evidence,
        "primary_recomputation": recomputed,
        "user_override": override_record,
        "execution_boundary": EXECUTION_BOUNDARY,
        "statistical_scope": STATISTICAL_SCOPE,
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    validate_authorization_document(root, document)
    _write_exclusive(output_path, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    user = subparsers.add_parser("create-user-override")
    user.add_argument("--repo-root", type=Path, required=True)
    user.add_argument("--output", type=Path, required=True)
    create = subparsers.add_parser("create-authorization")
    create.add_argument("--repo-root", type=Path, required=True)
    create.add_argument("--stage0-provenance", type=Path, required=True)
    create.add_argument("--direction-summary", type=Path, required=True)
    create.add_argument("--direction-cell", action="append", type=Path, required=True)
    create.add_argument("--exact-gradient", type=Path, required=True)
    create.add_argument("--minibatch-gradient", type=Path, required=True)
    create.add_argument("--user-override", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create-user-override":
        document = create_user_override(
            repo_root=args.repo_root, output_path=args.output
        )
    else:
        document = create_authorization(
            repo_root=args.repo_root,
            stage0_provenance_path=args.stage0_provenance,
            direction_summary_path=args.direction_summary,
            direction_cell_paths=args.direction_cell,
            exact_gradient_path=args.exact_gradient,
            minibatch_gradient_path=args.minibatch_gradient,
            user_override_path=args.user_override,
            output_path=args.output,
        )
    print(json.dumps({
        "status": document["status"],
        "source_commit": document["source_commit"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_RECIPES",
    "SCHEMA_VERSION",
    "STATUS",
    "USER_OVERRIDE_SCHEMA_VERSION",
    "USER_OVERRIDE_STATUS",
    "create_authorization",
    "create_user_override",
    "validate_authorization_document",
]
