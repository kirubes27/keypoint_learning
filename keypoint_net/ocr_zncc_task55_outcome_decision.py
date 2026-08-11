"""Apply the frozen Task-55 OCR-ZNCC paired outcome rule.

The three seeds are descriptive paired repetitions.  The selected-checkpoint
criteria are evaluated at each arm's own ``best_model.pt`` selected by minimum
base validation loss.  The only decision criterion at ``final_model.pt`` is
canonical-drift non-regression at epoch 1000; final operator and channel-health
values are recorded as diagnostics and cannot be substituted into the frozen
selected-checkpoint safety criteria.

This module performs no training, checkpoint selection, significance test, or
Task-80 launch.  A passing result is only an input to a separate Task-80
authorization step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import ocr_zncc_outcome_evaluator as outcome_evaluator


SCHEMA_VERSION = "ocr_zncc_task55_paired_outcome_decision.v1"
SEEDS = (42, 43, 44)
ARMS = ("control", "ocr_zncc")


class OCRTask55DecisionError(ValueError):
    """Raised when the six result artifacts do not form valid paired evidence."""


@dataclass(frozen=True)
class CheckpointEvidence:
    canonical_drift: float
    proper_rotation_angle_deg: float | None
    absolute_angle_error_deg: float | None
    active_on_object_count: int
    improper_or_reflection: bool
    all_numeric_outputs_finite: bool
    critical_failures: tuple[str, ...]


@dataclass(frozen=True)
class CellEvidence:
    cell_id: str
    arm: str
    seed: int
    source_commit: str
    result_content_sha256: str
    best_model: CheckpointEvidence
    final_model: CheckpointEvidence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRTask55DecisionError(message)


def _cell_id(arm: str, seed: int) -> str:
    return f"task55_clean__{arm}__seed{seed}"


def _finite_float(value: Any, *, name: str, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} is not finite",
    )
    result = float(value)
    if nonnegative:
        _require(result >= 0.0, f"{name} is negative")
    return result


def _optional_finite_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _extract_checkpoint(
    result: Mapping[str, Any], *, checkpoint_role: str
) -> CheckpointEvidence:
    checkpoints = result.get("checkpoint_results")
    _require(isinstance(checkpoints, Mapping), "cell checkpoint_results is invalid")
    row = checkpoints.get(checkpoint_role)
    _require(isinstance(row, Mapping), f"cell lacks {checkpoint_role}")
    _require(row.get("checkpoint_role") == checkpoint_role,
             f"{checkpoint_role} identity differs")
    expected_selector = (
        "minimum_base_validation_loss"
        if checkpoint_role == "best_model"
        else "fixed_epoch_1000"
    )
    _require(row.get("checkpoint_selector") == expected_selector,
             f"{checkpoint_role} selector differs")
    if checkpoint_role == "final_model":
        _require(row.get("checkpoint_epoch") == 1000,
                 "final checkpoint is not epoch 1000")
    canonical = row.get("canonical_drift")
    operator = row.get("operator")
    health = row.get("channel_health")
    _require(isinstance(canonical, Mapping), "canonical drift is invalid")
    _require(isinstance(operator, Mapping), "operator metrics are invalid")
    _require(isinstance(health, Mapping), "channel-health metrics are invalid")
    count = health.get("active_on_object_count")
    _require(isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 10,
             "active/on-object count is invalid")
    failures = row.get("critical_failures")
    _require(
        isinstance(failures, list) and all(isinstance(item, str) for item in failures),
        "critical failure list is invalid",
    )
    return CheckpointEvidence(
        canonical_drift=_finite_float(
            canonical.get("median_rms_objdiag"),
            name=f"{checkpoint_role} canonical drift",
            nonnegative=True,
        ),
        proper_rotation_angle_deg=_optional_finite_float(
            operator.get("proper_rotation_angle_deg"),
            name=f"{checkpoint_role} proper rotation angle",
        ),
        absolute_angle_error_deg=_optional_finite_float(
            operator.get("absolute_angle_error_deg"),
            name=f"{checkpoint_role} absolute angle error",
        ),
        active_on_object_count=count,
        improper_or_reflection=operator.get("improper_or_reflection") is True,
        all_numeric_outputs_finite=row.get("all_numeric_outputs_finite") is True,
        critical_failures=tuple(failures),
    )


def extract_cell_evidence(
    validated: outcome_evaluator.ValidatedOutcomeResult,
) -> CellEvidence:
    """Extract decision scalars only from an independently regenerated result."""

    _require(
        isinstance(validated, outcome_evaluator.ValidatedOutcomeResult),
        "cell evidence requires a live deep-validated outcome result",
    )
    result = validated.document
    claimed = result.get("content_hash_sha256")
    _require(isinstance(claimed, str), "cell result content hash is absent")
    _require(
        validated.result_record.get("content_hash_sha256") == claimed
        and validated.result_record.get("cell_id") == result.get("cell_id"),
        "live result-validation receipt differs from regenerated document",
    )
    payload = dict(result)
    payload.pop("content_hash_sha256", None)
    _require(
        outcome_evaluator.canonical_sha256(payload) == claimed,
        "cell result content hash differs",
    )
    _require(
        result.get("schema_version") == outcome_evaluator.SCHEMA_VERSION
        and result.get("artifact_type") == "frozen_paired_ocr_zncc_outcome_cell"
        and result.get("evaluation_status") == "complete",
        "cell outcome schema/status differs",
    )
    _require(result.get("recipe") == "task55_clean", "decision received non-Task55 cell")
    arm = result.get("arm")
    seed = result.get("seed")
    _require(arm in ARMS and seed in SEEDS, "cell arm/seed is invalid")
    expected_id = _cell_id(str(arm), int(seed))
    _require(result.get("cell_id") == expected_id, "cell identity differs")
    lock = result.get("metric_lock")
    _require(isinstance(lock, Mapping), "cell metric lock is invalid")
    _require(
        lock.get("primary_metric") == "canonical_drift.median_rms_objdiag"
        and lock.get("evaluation_config_sha256")
        == outcome_evaluator.GENERIC_EVALUATION_CONFIG_SHA256
        and lock.get("validation_frame_ids")
        == list(outcome_evaluator.VALIDATION_FRAME_IDS)
        and lock.get("soft_coordinate_channel_indices")
        == list(outcome_evaluator.SOFT_COORDINATE_CHANNELS)
        and lock.get("checkpoint_roles") == list(outcome_evaluator.CHECKPOINT_ROLES),
        "cell metric lock differs",
    )
    source_commit = result.get("source_commit")
    _require(isinstance(source_commit, str), "cell source commit is invalid")
    return CellEvidence(
        cell_id=expected_id,
        arm=str(arm),
        seed=int(seed),
        source_commit=source_commit,
        result_content_sha256=claimed,
        best_model=_extract_checkpoint(result, checkpoint_role="best_model"),
        final_model=_extract_checkpoint(result, checkpoint_role="final_model"),
    )


def _correct_positive_roll_sign(checkpoint: CheckpointEvidence) -> bool:
    angle = checkpoint.proper_rotation_angle_deg
    return bool(
        not checkpoint.improper_or_reflection
        and angle is not None
        and angle * 6.0 > 0.0
    )


def _critical_blockers(cells: Sequence[CellEvidence]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for cell in cells:
        for role in outcome_evaluator.CHECKPOINT_ROLES:
            checkpoint = getattr(cell, role)
            if not checkpoint.all_numeric_outputs_finite:
                blockers.append({
                    "cell_id": cell.cell_id,
                    "checkpoint_role": role,
                    "reason": "nonfinite_evaluator_output",
                })
            if checkpoint.improper_or_reflection:
                blockers.append({
                    "cell_id": cell.cell_id,
                    "checkpoint_role": role,
                    "reason": "operator_is_improper_or_reflection",
                })
            for failure in checkpoint.critical_failures:
                if failure != "operator_is_improper_or_reflection":
                    blockers.append({
                        "cell_id": cell.cell_id,
                        "checkpoint_role": role,
                        "reason": f"unrecognized_critical_failure:{failure}",
                    })
    return blockers


def decide(cells: Sequence[CellEvidence]) -> dict[str, Any]:
    by_id = {cell.cell_id: cell for cell in cells}
    expected = {_cell_id(arm, seed) for arm in ARMS for seed in SEEDS}
    _require(len(cells) == len(by_id) == 6 and set(by_id) == expected,
             "Task55 decision requires exactly six unique cells")
    _require(len({cell.source_commit for cell in cells}) == 1,
             "Task55 cell source commits differ")

    blockers = _critical_blockers(cells)
    pairs: list[dict[str, Any]] = []
    selected_ratio_support = 0
    final_drift_support = 0
    selected_magnitude_support = 0
    selected_active_all = True
    selected_sign_all = True

    final_diagnostic_sign_all = True
    final_diagnostic_magnitude_support = 0
    final_diagnostic_active_all = True

    for seed in SEEDS:
        control = by_id[_cell_id("control", seed)]
        ocr = by_id[_cell_id("ocr_zncc", seed)]
        control_best = control.best_model
        ocr_best = ocr.best_model
        control_final = control.final_model
        ocr_final = ocr.final_model

        if control_best.canonical_drift <= 0.0:
            ratio = None
            ratio_support = False
            blockers.append({
                "seed": seed,
                "checkpoint_role": "best_model",
                "reason": "selected_control_drift_ratio_denominator_nonpositive",
            })
        else:
            ratio = ocr_best.canonical_drift / control_best.canonical_drift
            ratio_support = ratio <= 0.80
        final_support = ocr_final.canonical_drift <= control_final.canonical_drift
        selected_sign = _correct_positive_roll_sign(ocr_best)
        selected_magnitude = bool(
            ocr_best.absolute_angle_error_deg is not None
            and control_best.absolute_angle_error_deg is not None
            and ocr_best.absolute_angle_error_deg
            <= control_best.absolute_angle_error_deg
        )
        selected_active = (
            ocr_best.active_on_object_count >= control_best.active_on_object_count
        )

        final_sign = _correct_positive_roll_sign(ocr_final)
        final_magnitude = bool(
            ocr_final.absolute_angle_error_deg is not None
            and control_final.absolute_angle_error_deg is not None
            and ocr_final.absolute_angle_error_deg
            <= control_final.absolute_angle_error_deg
        )
        final_active = (
            ocr_final.active_on_object_count >= control_final.active_on_object_count
        )

        selected_ratio_support += int(ratio_support)
        final_drift_support += int(final_support)
        selected_magnitude_support += int(selected_magnitude)
        selected_active_all = selected_active_all and selected_active
        selected_sign_all = selected_sign_all and selected_sign
        final_diagnostic_sign_all = final_diagnostic_sign_all and final_sign
        final_diagnostic_magnitude_support += int(final_magnitude)
        final_diagnostic_active_all = final_diagnostic_active_all and final_active

        pairs.append({
            "seed": seed,
            "control_cell_id": control.cell_id,
            "ocr_zncc_cell_id": ocr.cell_id,
            "selected_checkpoint": {
                "scope": "each_arm_own_minimum_base_validation_loss_checkpoint",
                "values": {
                    "control": {
                        "canonical_drift_median_rms_objdiag": control_best.canonical_drift,
                        "proper_rotation_angle_deg": control_best.proper_rotation_angle_deg,
                        "absolute_angle_error_deg": control_best.absolute_angle_error_deg,
                        "active_on_object_count": control_best.active_on_object_count,
                    },
                    "ocr_zncc": {
                        "canonical_drift_median_rms_objdiag": ocr_best.canonical_drift,
                        "proper_rotation_angle_deg": ocr_best.proper_rotation_angle_deg,
                        "absolute_angle_error_deg": ocr_best.absolute_angle_error_deg,
                        "active_on_object_count": ocr_best.active_on_object_count,
                    },
                    "ocr_to_control_drift_ratio": ratio,
                },
                "decision_support": {
                    "drift_ratio_at_most_0p80": ratio_support,
                    "ocr_positive_roll_sign": selected_sign,
                    "ocr_absolute_angle_error_nonworse": selected_magnitude,
                    "ocr_active_on_object_count_nonlower": selected_active,
                },
            },
            "final_checkpoint": {
                "scope": "fixed_epoch_1000",
                "values": {
                    "control": {
                        "canonical_drift_median_rms_objdiag": control_final.canonical_drift,
                        "proper_rotation_angle_deg": control_final.proper_rotation_angle_deg,
                        "absolute_angle_error_deg": control_final.absolute_angle_error_deg,
                        "active_on_object_count": control_final.active_on_object_count,
                    },
                    "ocr_zncc": {
                        "canonical_drift_median_rms_objdiag": ocr_final.canonical_drift,
                        "proper_rotation_angle_deg": ocr_final.proper_rotation_angle_deg,
                        "absolute_angle_error_deg": ocr_final.absolute_angle_error_deg,
                        "active_on_object_count": ocr_final.active_on_object_count,
                    },
                },
                "decision_support": {
                    "ocr_drift_nonregression": final_support,
                },
                "diagnostics_not_used_for_decision": {
                    "ocr_positive_roll_sign": final_sign,
                    "ocr_absolute_angle_error_nonworse": final_magnitude,
                    "ocr_active_on_object_count_nonlower": final_active,
                },
            },
        })

    criterion_results = {
        "selected_drift_ratio_at_most_0p80_in_at_least_two_of_three": (
            selected_ratio_support >= 2
        ),
        "final_epoch_1000_drift_nonregression_in_at_least_two_of_three": (
            final_drift_support >= 2
        ),
        "selected_ocr_positive_roll_sign_in_all_three": selected_sign_all,
        "selected_ocr_absolute_angle_error_nonworse_in_at_least_two_of_three": (
            selected_magnitude_support >= 2
        ),
        "selected_ocr_active_on_object_count_nonlower_in_all_three": selected_active_all,
        "no_provenance_nonfinite_or_reflection_blocker": not blockers,
    }
    passed = all(criterion_results.values())
    if blockers:
        outcome = "stop_invalid_or_critical_evaluator_evidence"
    elif passed:
        outcome = "task55_pass_candidate_for_separate_task80_authorization"
    else:
        outcome = "task55_fail_stop_before_task80"
    return {
        "outcome": outcome,
        "task55_pass": passed,
        "task80_training_authorized_by_this_artifact": False,
        "criterion_results": criterion_results,
        "support_counts": {
            "paired_seed_count": 3,
            "selected_drift_ratio_at_most_0p80": selected_ratio_support,
            "final_epoch_1000_drift_nonregression": final_drift_support,
            "selected_absolute_angle_error_nonworse": selected_magnitude_support,
        },
        "critical_blockers": blockers,
        "pairs": pairs,
        "final_checkpoint_diagnostics_not_used_for_decision": {
            "ocr_positive_roll_sign_in_all_three": final_diagnostic_sign_all,
            "ocr_absolute_angle_error_nonworse_support_count": (
                final_diagnostic_magnitude_support
            ),
            "ocr_active_on_object_count_nonlower_in_all_three": (
                final_diagnostic_active_all
            ),
        },
    }


def _load_and_bind_result(
    repo_root: Path, path: Path
) -> tuple[CellEvidence, dict[str, Any], dict[str, Any]]:
    validated = outcome_evaluator.validate_live_result(repo_root, path)
    result = dict(validated.document)
    result_record = dict(validated.result_record)
    run = validated.run
    evidence = extract_cell_evidence(validated)
    lineage = result.get("lineage")
    _require(isinstance(lineage, Mapping), "cell lineage is invalid")
    receipt_record = lineage.get("completed_run_receipt")
    _require(isinstance(receipt_record, Mapping), "cell receipt lineage is invalid")
    _require(
        evidence.source_commit == run.source_commit
        and result.get("source_branch") == run.source_branch,
        "cell result source lineage differs from completed run",
    )
    _require(
        dict(receipt_record) == dict(run.receipt_record),
        "cell result receipt record differs from current bound receipt",
    )
    manifest_record = lineage.get("experiment_manifest")
    _require(
        isinstance(manifest_record, Mapping)
        and dict(manifest_record) == dict(run.manifest_record),
        "cell result manifest record differs from current bound manifest",
    )
    _require(
        lineage.get("committed_evaluation_sources") == run.evaluation_source_files,
        "cell result evaluation-source binding differs",
    )
    _require(
        lineage.get("matched_pairing_evidence") == run.pairing_evidence,
        "cell result matched-pairing evidence differs from completed run",
    )
    for role in outcome_evaluator.CHECKPOINT_ROLES:
        filename = f"{role}.pt"
        checkpoint = result["checkpoint_results"][role]["checkpoint"]
        _require(
            checkpoint.get("absolute_path") == run.artifacts[filename]["absolute_path"]
            and checkpoint.get("sha256") == run.artifacts[filename]["sha256"]
            and checkpoint.get("size_bytes") == run.artifacts[filename]["size_bytes"],
            f"cell {role} binding differs from completed run",
        )
    _require(
        result_record.get("content_hash_sha256") == evidence.result_content_sha256
        and result_record.get("cell_id") == evidence.cell_id,
        "deep result-validation receipt differs from extracted evidence",
    )
    return evidence, result, result_record


def validate_matched_pairing_evidence(
    control: Mapping[str, Any],
    ocr: Mapping[str, Any],
    *,
    recipe: str,
    seed: int,
) -> None:
    """Require one allocation/wrapper with control-first and OCR-second."""

    common_fields = {
        "initial_model_state_sha256",
        "post_diagnostic_model_state_sha256",
        "diagnostic_pair_order_sha256",
        "diagnostic_pair_order_sample_count",
        "train_sampler_generator_seed",
        "epoch_pair_order_sequence_sha256",
        "execution_device",
    }
    expected_fields = common_fields | {"runtime_pair_execution"}
    _require(
        set(control) == set(ocr) == expected_fields
        and all(control[field] == ocr[field] for field in common_fields),
        f"{recipe} seed {seed} common matched-pair evidence differs",
    )
    control_runtime = control.get("runtime_pair_execution")
    ocr_runtime = ocr.get("runtime_pair_execution")
    runtime_fields = {
        "pair_id",
        "position",
        "wrapper_pid",
        "slurm_job_id",
        "cuda_visible_devices",
        "allocated_gpu_count",
        "stage_launch_lock_content_hash_sha256",
        "runtime_provenance_files",
    }
    _require(
        isinstance(control_runtime, Mapping)
        and isinstance(ocr_runtime, Mapping)
        and set(control_runtime) == set(ocr_runtime) == runtime_fields,
        f"{recipe} seed {seed} runtime pair evidence schema differs",
    )
    runtime_common = runtime_fields - {"position"}
    _require(
        all(control_runtime[field] == ocr_runtime[field] for field in runtime_common)
        and control_runtime.get("pair_id") == f"{recipe}__seed{seed}"
        and control_runtime.get("position") == "1_control"
        and ocr_runtime.get("position") == "2_ocr_zncc"
        and control_runtime.get("allocated_gpu_count") == 1
        and control.get("execution_device") == ocr.get("execution_device") == "cuda",
        f"{recipe} seed {seed} is not one control-first/OCR-second GPU pair",
    )


def _write_exclusive(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = outcome_evaluator.canonical_json_bytes(document) + b"\n"
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCRTask55DecisionError(f"cannot exclusively create {absolute}") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "exclusive decision write made no progress")
            offset += written
    finally:
        os.close(descriptor)
    return {
        "absolute_path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "write_mode": "exclusive_create_no_overwrite",
    }


def run_decision(
    *, repo_root: Path | str, result_paths: Sequence[Path | str], output_path: Path | str
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    _require(len(result_paths) == 6, "exactly six Task55 result paths are required")
    cells: list[CellEvidence] = []
    result_records: list[dict[str, Any]] = []
    manifest_records: list[Mapping[str, Any]] = []
    pairing_records: dict[str, Mapping[str, Any]] = {}
    for supplied in result_paths:
        evidence, result, record = _load_and_bind_result(root, Path(supplied))
        cells.append(evidence)
        result_records.append(record)
        manifest_records.append(result["lineage"]["experiment_manifest"])
        pairing_records[evidence.cell_id] = result["lineage"]["matched_pairing_evidence"]
    _require(
        all(record == manifest_records[0] for record in manifest_records),
        "Task55 cell results bind different experiment manifests",
    )
    for seed in SEEDS:
        control = pairing_records[_cell_id("control", seed)]
        ocr = pairing_records[_cell_id("ocr_zncc", seed)]
        validate_matched_pairing_evidence(
            control,
            ocr,
            recipe="task55_clean",
            seed=seed,
        )
    payload = decide(cells)
    source_commit = cells[0].source_commit
    decision_source = outcome_evaluator._committed_source_record(  # noqa: SLF001
        root,
        "keypoint_net/ocr_zncc_task55_outcome_decision.py",
        source_commit=source_commit,
    )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_task55_paired_ocr_zncc_outcome_decision",
        "source_commit": source_commit,
        "source_branch": "agent/ocr-zncc-training-20260811",
        "decision_rule": {
            "paired_repetition_unit": "seed",
            "seed_ids": list(SEEDS),
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
        },
        "experiment_manifest": dict(manifest_records[0]),
        "cell_result_records": sorted(result_records, key=lambda row: row["cell_id"]),
        "committed_decision_source": decision_source,
        **payload,
        "next_phase_started": False,
    }
    document["content_hash_sha256"] = outcome_evaluator.canonical_sha256(document)
    output_record = _write_exclusive(Path(output_path), document)
    return {"document": document, "output_record": output_record}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_decision(
        repo_root=args.repo_root,
        result_paths=args.result,
        output_path=args.output,
    )
    print(json.dumps(
        {
            **result["output_record"],
            "outcome": result["document"]["outcome"],
            "task55_pass": result["document"]["task55_pass"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CellEvidence",
    "CheckpointEvidence",
    "OCRTask55DecisionError",
    "SCHEMA_VERSION",
    "SEEDS",
    "decide",
    "extract_cell_evidence",
    "run_decision",
    "validate_matched_pairing_evidence",
]
