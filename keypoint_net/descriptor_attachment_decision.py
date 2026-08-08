"""Apply the frozen paired decision rule to 12 descriptor evaluator results.

The three seeds per recipe are descriptive paired repetitions, not independent
population samples.  This module performs no significance test and cannot add
seeds, tune the loss, retrain, or start a later phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import descriptor_attachment as attachment
from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net import roll_head_package_decision as package_decision
from keypoint_net import run_descriptor_attachment_matrix as matrix


SCHEMA_VERSION = "descriptor_attachment_primary_decision.v1"
RECIPES = ("task55_clean", "task80_assisted")
SEEDS = (42, 43, 44)
ARMS = ("control", "attachment")


class DescriptorDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class DescriptorEvidence:
    cell_id: str
    recipe: str
    arm: str
    seed: int
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    angle_error_deg: float | None
    validation_auc: float | None
    canonical_drift: float | None
    persistent_duplicate_count: int | None
    recurrent_duplicate_count: int | None
    active_on_object_count: int | None
    source_commit: str
    result_content_sha256: str
    checkpoint_sha256: str
    completed_run_receipt_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DescriptorDecisionError(message)


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    data = matrix._read_regular(path.absolute(), name=name)  # noqa: SLF001
    value = matrix._strict_json(data, name=name)  # noqa: SLF001
    return value


def _cell_id(recipe: str, arm: str, seed: int) -> str:
    return f"{recipe}__{arm}__seed{seed}"


def _adapt_for_existing_metric_extractor(
    result: Mapping[str, Any], *, recipe: str, seed: int
) -> Mapping[str, Any]:
    """Change only identity fields in-memory to reuse the frozen metric parser."""

    adapted = copy.deepcopy(dict(result))
    synthetic_id = f"{recipe}__r64__seed{seed}"
    adapted["case_id"] = synthetic_id
    checkpoint_authorization = dict(adapted["checkpoint_authorization"])
    checkpoint_authorization["cell_id"] = synthetic_id
    adapted["checkpoint_authorization"] = checkpoint_authorization
    adapted.pop("result_content_sha256", None)
    adapted["result_content_sha256"] = package_decision.canonical_sha256(adapted)
    return adapted


def extract_descriptor_evidence(
    result: Mapping[str, Any], *, repo_root: Path, expected_cell_id: str,
) -> DescriptorEvidence:
    claimed_hash = result.get("result_content_sha256")
    _require(isinstance(claimed_hash, str), "descriptor result hash is absent")
    unhashed = dict(result)
    unhashed.pop("result_content_sha256", None)
    _require(package_decision.canonical_sha256(unhashed) == claimed_hash,
             "descriptor result content hash differs")
    _require(result.get("case_id") == expected_cell_id,
             "descriptor result cell identity differs")
    recipe, arm, seed_text = expected_cell_id.split("__")
    seed = int(seed_text.removeprefix("seed"))
    _require(recipe in RECIPES and arm in ARMS and seed in SEEDS,
             "descriptor result cell is outside the primary matrix")
    provenance = result.get("provenance")
    _require(isinstance(provenance, Mapping), "descriptor provenance is invalid")
    external = provenance.get("external_files")
    _require(isinstance(external, list), "descriptor external provenance is invalid")
    by_role = {record.get("role"): record for record in external
               if isinstance(record, Mapping)}
    _require(set(by_role) == {"checkpoint", "checkpoint_config",
                              "checkpoint_metadata", "completed_run_receipt"},
             "descriptor external provenance roles differ")
    receipt_record = by_role["completed_run_receipt"]
    receipt_path = Path(str(receipt_record["absolute_path"]))
    capability = authorization.authorize_completed_run(
        repo_root, receipt_path, expected_cell_id=expected_cell_id
    )
    _require(receipt_record.get("file_sha256") == capability.receipt_file_sha256
             and by_role["checkpoint"].get("file_sha256")
             == capability.checkpoint_sha256,
             "descriptor result-to-receipt binding differs")
    parsed = package_decision.extract_cell_evidence(
        _adapt_for_existing_metric_extractor(result, recipe=recipe, seed=seed)
    )
    return DescriptorEvidence(
        cell_id=expected_cell_id,
        recipe=recipe,
        arm=arm,
        seed=seed,
        eligible=parsed.eligible,
        ineligibility_reasons=parsed.ineligibility_reasons,
        angle_error_deg=parsed.angle_error_deg,
        validation_auc=parsed.validation_auc,
        canonical_drift=parsed.canonical_drift,
        persistent_duplicate_count=parsed.persistent_duplicate_count,
        recurrent_duplicate_count=parsed.recurrent_duplicate_count,
        active_on_object_count=parsed.active_on_object_count,
        source_commit=capability.source_commit,
        result_content_sha256=claimed_hash,
        checkpoint_sha256=capability.checkpoint_sha256,
        completed_run_receipt_sha256=capability.receipt_file_sha256,
    )


def decide(cells: Sequence[DescriptorEvidence]) -> dict[str, Any]:
    by_id = {cell.cell_id: cell for cell in cells}
    expected = {_cell_id(recipe, arm, seed) for recipe in RECIPES
                for arm in ARMS for seed in SEEDS}
    _require(len(cells) == len(by_id) == 12 and set(by_id) == expected,
             "descriptor decision requires exactly 12 unique cells")
    source_commits = {cell.source_commit for cell in cells}
    _require(len(source_commits) == 1, "descriptor result source commits differ")

    pairs = []
    summaries: dict[str, Any] = {}
    all_eligible = all(cell.eligible for cell in cells)
    for recipe in RECIPES:
        counts = {
            "canonical_drift_strict_wins": 0,
            "validation_auc_nonworse": 0,
            "duplicate_active_guardrail_support": 0,
            "operator_angle_guardrail_support": 0,
        }
        for seed in SEEDS:
            control = by_id[_cell_id(recipe, "control", seed)]
            intervention = by_id[_cell_id(recipe, "attachment", seed)]
            comparable = control.eligible and intervention.eligible
            drift_win = bool(
                comparable
                and intervention.canonical_drift < control.canonical_drift  # type: ignore[operator]
            )
            auc_support = bool(
                comparable
                and intervention.validation_auc <= control.validation_auc  # type: ignore[operator]
            )
            duplicate_support = bool(
                comparable
                and intervention.persistent_duplicate_count
                <= control.persistent_duplicate_count  # type: ignore[operator]
                and intervention.recurrent_duplicate_count
                <= control.recurrent_duplicate_count  # type: ignore[operator]
                and intervention.active_on_object_count
                >= control.active_on_object_count  # type: ignore[operator]
            )
            angle_support = bool(
                comparable
                and intervention.angle_error_deg <= control.angle_error_deg  # type: ignore[operator]
            )
            counts["canonical_drift_strict_wins"] += int(drift_win)
            counts["validation_auc_nonworse"] += int(auc_support)
            counts["duplicate_active_guardrail_support"] += int(duplicate_support)
            counts["operator_angle_guardrail_support"] += int(angle_support)
            pairs.append({
                "recipe": recipe, "seed": seed, "comparable": comparable,
                "control_cell": control.cell_id,
                "attachment_cell": intervention.cell_id,
                "support": {
                    "canonical_drift_strict_win": drift_win,
                    "validation_auc_nonworse": auc_support,
                    "duplicate_active_guardrail": duplicate_support,
                    "operator_angle_guardrail": angle_support,
                },
                "values": {
                    "control": {
                        "canonical_drift": control.canonical_drift,
                        "validation_auc": control.validation_auc,
                        "persistent_duplicate_count": control.persistent_duplicate_count,
                        "recurrent_duplicate_count": control.recurrent_duplicate_count,
                        "active_on_object_count": control.active_on_object_count,
                        "angle_error_deg": control.angle_error_deg,
                    },
                    "attachment": {
                        "canonical_drift": intervention.canonical_drift,
                        "validation_auc": intervention.validation_auc,
                        "persistent_duplicate_count": intervention.persistent_duplicate_count,
                        "recurrent_duplicate_count": intervention.recurrent_duplicate_count,
                        "active_on_object_count": intervention.active_on_object_count,
                        "angle_error_deg": intervention.angle_error_deg,
                    },
                },
            })
        recipe_pass = all(counts[key] >= 2 for key in counts)
        summaries[recipe] = {"paired_seed_count": 3, "support_counts": counts,
                             "recipe_pass": recipe_pass}

    if not all_eligible:
        outcome = "stop_ineligible_result"
    elif all(summary["recipe_pass"] for summary in summaries.values()):
        outcome = "retain_descriptor_attachment"
    else:
        outcome = "reject_exact_descriptor_attachment"
    return {
        "outcome": outcome,
        "all_twelve_evaluator_results_eligible": all_eligible,
        "rule": {
            "unit": "paired_seed_within_recipe",
            "seed_count_per_recipe": 3,
            "descriptive_not_inferential": True,
            "each_recipe_requires_at_least_two_of_three_support_on_every_axis": [
                "canonical_drift_strict_win",
                "validation_auc_nonworse",
                "persistent_and_recurrent_duplicates_nonworse_and_active_on_object_nonworse",
                "operator_angle_error_nonworse",
            ],
            "both_recipes_must_pass": True,
            "no_extension_seed_or_hyperparameter_tuning_authorized": True,
        },
        "recipe_summaries": summaries,
        "pairs": pairs,
        "ineligible_cells": [
            {"cell_id": cell.cell_id,
             "reasons": list(cell.ineligibility_reasons)}
            for cell in cells if not cell.eligible
        ],
    }


def run_decision(*, matrix_root: Path, output: Path) -> Path:
    root = matrix_root.resolve(strict=True)
    lock = matrix.validate_matrix(
        root,
        expected_commit=str(_read_json(root / matrix.LOCK_NAME,
                                       name="paired matrix lock")["source_commit"]),
        slurm_script_sha256=_read_json(root / matrix.LOCK_NAME,
                                      name="paired matrix lock")["slurm_script"]["sha256"],
        environment_lock_sha256=_read_json(root / matrix.LOCK_NAME,
                                           name="paired matrix lock")["environment_lock"]["sha256"],
    )
    cells = []
    result_records = []
    for recipe in RECIPES:
        for arm in ARMS:
            for seed in SEEDS:
                cell_id = _cell_id(recipe, arm, seed)
                path = root / "evaluations" / f"{cell_id}.json"
                result = _read_json(path, name=f"descriptor result {cell_id}")
                evidence = extract_descriptor_evidence(
                    result, repo_root=matrix.REPO_ROOT, expected_cell_id=cell_id
                )
                cells.append(evidence)
                data = matrix._read_regular(path.absolute(), name=cell_id)  # noqa: SLF001
                result_records.append({"cell_id": cell_id,
                                       "absolute_path": str(path.absolute()),
                                       "file_sha256": hashlib.sha256(data).hexdigest(),
                                       "result_content_sha256": evidence.result_content_sha256})
    payload = decide(cells)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_primary_decision",
        "source_commit": lock["source_commit"],
        "matrix_lock": matrix._file_record(root / matrix.LOCK_NAME,
                                           name="paired matrix lock"),  # noqa: SLF001
        "result_records": result_records,
        **payload,
        "next_phase_started": False,
    }
    document["content_hash_sha256"] = attachment.canonical_sha256(document)
    _require(not output.exists(), "descriptor decision output already exists")
    matrix._write_exclusive(output.absolute(), document,
                            name="descriptor primary decision")  # noqa: SLF001
    return output.absolute()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_decision(matrix_root=args.matrix_root, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
