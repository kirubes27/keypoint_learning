"""Apply the frozen primary-seed 64-versus-128 package decision rule.

The module consumes exactly the twelve receipted production-evaluator results
for seeds 42--44.  It cannot start training, add seeds, or change a scientific
threshold.  A primary outcome is one of: adopt 128, retain 64, or request the
already-preregistered seed-45/46 extension.
"""

from __future__ import annotations

import argparse
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


RESULT_SCHEMA_VERSION = "representation-evaluation-result-v2"
DECISION_SCHEMA_VERSION = "roll-head-package-primary-decision-v1"
RECIPES = ("task55_clean", "task80_assisted")
HEAD_PACKAGES = ("64", "128")
PRIMARY_SEEDS = (42, 43, 44)
EXTENSION_SEEDS = (45, 46)
CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__r(64|128)__seed(42|43|44)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PackageDecisionError(ValueError):
    """The evaluator inputs cannot support the frozen package decision."""


@dataclass(frozen=True)
class CellEvidence:
    cell_id: str
    recipe: str
    head_package: str
    seed: int
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    angle_error_deg: float | None
    validation_auc: float | None
    canonical_drift: float | None
    persistent_duplicate_count: int | None
    recurrent_duplicate_count: int | None
    active_on_object_count: int | None
    source_commit: str | None
    evaluation_config_sha256: str
    result_content_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageDecisionError(message)


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return data + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _strict_json(data: bytes, *, name: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        _require(math.isfinite(value), f"{name} contains a non-finite number")
        return value

    def reject_constant(token: str) -> None:
        raise PackageDecisionError(f"{name} contains forbidden constant {token!r}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except PackageDecisionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PackageDecisionError(f"{name} is not strict JSON") from exc
    return _mapping(value, name=name)


def _read_regular(path: Path, *, name: str) -> bytes:
    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PackageDecisionError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def extract_cell_evidence(result: Mapping[str, Any]) -> CellEvidence:
    """Validate one hashed evaluator result and extract decision axes."""

    _require(result.get("schema_version") == RESULT_SCHEMA_VERSION,
             "evaluator result schema differs")
    claimed_hash = result.get("result_content_sha256")
    _require(isinstance(claimed_hash, str) and SHA256_RE.fullmatch(claimed_hash) is not None,
             "evaluator result content hash is invalid")
    unhashed = dict(result)
    unhashed.pop("result_content_sha256", None)
    _require(canonical_sha256(unhashed) == claimed_hash,
             "evaluator result content hash differs")

    cell_id = result.get("case_id")
    match = CELL_RE.fullmatch(cell_id) if isinstance(cell_id, str) else None
    _require(match is not None, "evaluator result cell id is not a primary cell")
    recipe, head_package, seed_text = match.groups()
    seed = int(seed_text)
    _require(result.get("case_kind") == "checkpoint",
             f"{cell_id}: case kind differs")

    stratum = _mapping(result.get("stratum"), name=f"{cell_id} stratum")
    _require(stratum.get("object_id") == "engineers_hammer_vray",
             f"{cell_id}: object role differs")
    _require(stratum.get("seed") == seed,
             f"{cell_id}: seed differs")
    _require(stratum.get("partition") == "full_corpus",
             f"{cell_id}: partition differs")
    _require(
        stratum.get("transform_family") == "roll"
        and stratum.get("direction") == "forward"
        and stratum.get("stride") == 3,
        f"{cell_id}: transform stratum differs",
    )
    transform = _mapping(result.get("transform"), name=f"{cell_id} transform")
    _require(
        transform.get("family") == "roll"
        and transform.get("physical_axis") == "world_z"
        and transform.get("direction") == "forward"
        and transform.get("stride") == 3
        and transform.get("cyclic") is True
        and transform.get("signed_generator") == 6.0,
        f"{cell_id}: frozen roll geometry differs",
    )
    evaluation_config = _mapping(
        result.get("evaluation_config"), name=f"{cell_id} evaluation config"
    )
    evaluation_config_sha256 = result.get("evaluation_config_sha256")
    _require(
        isinstance(evaluation_config_sha256, str)
        and SHA256_RE.fullmatch(evaluation_config_sha256) is not None
        and evaluation_config_sha256 == canonical_sha256(evaluation_config),
        f"{cell_id}: evaluation config hash differs",
    )
    _require(
        evaluation_config.get("protocol") == "full_primary_roll"
        and evaluation_config.get("role_scoped_holdout_frame_ids") == list(range(24))
        and evaluation_config.get("holdout_rollout_horizons") == list(range(1, 8))
        and evaluation_config.get("full_rollout_horizons") == list(range(1, 60))
        and evaluation_config.get("closure_horizon") == 60,
        f"{cell_id}: frozen evaluation scope differs",
    )

    authorization = _mapping(
        result.get("checkpoint_authorization"),
        name=f"{cell_id} checkpoint authorization",
    )
    _require(
        authorization.get("checkpoint_evaluation_authorized") is True
        and authorization.get("selection_use_authorized") is True
        and authorization.get("training_or_weight_update_authorized") is False
        and authorization.get("cell_id") == cell_id,
        f"{cell_id}: fresh selection authorization differs",
    )
    source_commit = authorization.get("source_commit")
    _require(isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None,
             f"{cell_id}: source commit is invalid")
    for key in ("checkpoint_sha256", "completed_run_receipt_sha256"):
        value = authorization.get(key)
        _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                 f"{cell_id}: authorization {key} is invalid")

    reasons: list[str] = []
    critical_failures = result.get("critical_failures")
    if not isinstance(critical_failures, list):
        reasons.append("critical_failure_record_missing")
    elif critical_failures:
        reasons.append("evaluator_critical_failure")

    operator = result.get("operator")
    angle_error = None
    if not isinstance(operator, Mapping):
        reasons.append("operator_metrics_missing")
    else:
        if operator.get("wrong_locked_sign") is not False:
            reasons.append("wrong_roll_sign")
        if operator.get("improper_or_reflection") is not False:
            reasons.append("improper_or_reflection_operator")
        angle_error = _finite_number(operator.get("absolute_angle_error_deg"))
        if angle_error is None or angle_error < 0.0:
            angle_error = None
            reasons.append("operator_angle_unavailable")

    collapse = result.get("collapse_evidence")
    if not isinstance(collapse, Mapping):
        reasons.append("collapse_evidence_missing")
    elif (
        collapse.get(
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead"
        ) != "available"
        or collapse.get(
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ) is not False
    ):
        reasons.append("structural_negative_control_collapse_or_void")

    active_count = _nonnegative_integer(
        _nested(result, "channel_health", "active_on_object_count")
    )
    eligible_count = _nonnegative_integer(
        _nested(result, "trajectory_separation", "eligible_channel_count")
    )
    if (
        active_count is None
        or eligible_count is None
        or active_count != eligible_count
        or eligible_count < 2
    ):
        reasons.append("fewer_than_two_eligible_active_on_object_channels")

    eligible_categories = _nested(
        result,
        "trajectory_separation",
        "eligible_active_on_object",
        "category_counts",
    )
    persistent = None
    recurrent = None
    if isinstance(eligible_categories, Mapping):
        persistent = _nonnegative_integer(
            eligible_categories.get("persistent_duplicate")
        )
        recurrent = _nonnegative_integer(
            eligible_categories.get("recurrent_close_pair")
        )
    if persistent is None or recurrent is None:
        reasons.append("eligible_duplicate_counts_missing")

    validation_auc = _finite_number(
        _nested(
            result,
            "rollout",
            "role_scoped_holdout_identity_normalized_auc",
            "value",
        )
    )
    if validation_auc is None or validation_auc < 0.0:
        validation_auc = None
        reasons.append("role_scoped_validation_auc_void")

    canonical_drift = _finite_number(
        _nested(result, "canonical_drift", "median_rms_objdiag")
    )
    if canonical_drift is None or canonical_drift < 0.0:
        canonical_drift = None
        reasons.append("canonical_drift_missing")

    provenance = result.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_commit") != source_commit
        or not isinstance(provenance.get("committed_files"), list)
        or not isinstance(provenance.get("external_files"), list)
    ):
        reasons.append("provenance_missing_or_contradictory")

    return CellEvidence(
        cell_id=cell_id,
        recipe=recipe,
        head_package=head_package,
        seed=seed,
        eligible=not reasons,
        ineligibility_reasons=tuple(sorted(set(reasons))),
        angle_error_deg=angle_error,
        validation_auc=validation_auc,
        canonical_drift=canonical_drift,
        persistent_duplicate_count=persistent,
        recurrent_duplicate_count=recurrent,
        active_on_object_count=active_count,
        source_commit=source_commit,
        evaluation_config_sha256=evaluation_config_sha256,
        result_content_sha256=claimed_hash,
    )


def _expected_cell_ids() -> set[str]:
    return {
        f"{recipe}__r{head}__seed{seed}"
        for recipe in RECIPES
        for head in HEAD_PACKAGES
        for seed in PRIMARY_SEEDS
    }


def decide_primary_package(cells: Sequence[CellEvidence]) -> dict[str, Any]:
    """Apply only the frozen three-primary-seed decision and extension trigger."""

    by_id: dict[str, CellEvidence] = {}
    for cell in cells:
        match = CELL_RE.fullmatch(cell.cell_id)
        _require(match is not None, f"invalid primary cell {cell.cell_id}")
        _require(
            (cell.recipe, cell.head_package, cell.seed)
            == (match.group(1), match.group(2), int(match.group(3))),
            f"cell identity fields differ for {cell.cell_id}",
        )
        _require(
            isinstance(cell.source_commit, str)
            and COMMIT_RE.fullmatch(cell.source_commit) is not None,
            f"source commit is invalid for {cell.cell_id}",
        )
        _require(
            SHA256_RE.fullmatch(cell.evaluation_config_sha256) is not None,
            f"evaluation config hash is invalid for {cell.cell_id}",
        )
        if cell.eligible:
            _require(
                not cell.ineligibility_reasons
                and all(
                    value is not None
                    for value in (
                        cell.angle_error_deg,
                        cell.validation_auc,
                        cell.canonical_drift,
                        cell.persistent_duplicate_count,
                        cell.recurrent_duplicate_count,
                        cell.active_on_object_count,
                    )
                ),
                f"eligible cell metrics are incomplete for {cell.cell_id}",
            )
        else:
            _require(bool(cell.ineligibility_reasons),
                     f"ineligible cell lacks reasons for {cell.cell_id}")
        _require(cell.cell_id not in by_id, f"duplicate cell {cell.cell_id}")
        by_id[cell.cell_id] = cell
    expected = _expected_cell_ids()
    _require(set(by_id) == expected,
             f"primary matrix cell set differs; missing={sorted(expected - set(by_id))}, "
             f"unexpected={sorted(set(by_id) - expected)}")

    pairs: list[dict[str, Any]] = []
    recipe_summaries: dict[str, Any] = {}
    for recipe in RECIPES:
        counts = {
            "validation_auc_strict_wins": 0,
            "canonical_drift_strict_wins": 0,
            "duplicate_active_guardrail_support": 0,
            "angle_guardrail_support": 0,
        }
        for seed in PRIMARY_SEEDS:
            low = by_id[f"{recipe}__r64__seed{seed}"]
            high = by_id[f"{recipe}__r128__seed{seed}"]
            comparable = low.eligible and high.eligible
            auc_win = bool(
                comparable and high.validation_auc < low.validation_auc  # type: ignore[operator]
            )
            drift_win = bool(
                comparable and high.canonical_drift < low.canonical_drift  # type: ignore[operator]
            )
            count_support = bool(
                comparable
                and high.persistent_duplicate_count
                <= low.persistent_duplicate_count  # type: ignore[operator]
                and high.active_on_object_count >= low.active_on_object_count  # type: ignore[operator]
            )
            angle_support = bool(
                comparable and high.angle_error_deg <= low.angle_error_deg  # type: ignore[operator]
            )
            counts["validation_auc_strict_wins"] += int(auc_win)
            counts["canonical_drift_strict_wins"] += int(drift_win)
            counts["duplicate_active_guardrail_support"] += int(count_support)
            counts["angle_guardrail_support"] += int(angle_support)
            pairs.append({
                "recipe": recipe,
                "seed": seed,
                "comparable": comparable,
                "cell_64": low.cell_id,
                "cell_128": high.cell_id,
                "values": {
                    "64": {
                        "validation_auc": low.validation_auc,
                        "canonical_drift": low.canonical_drift,
                        "persistent_duplicate_count": low.persistent_duplicate_count,
                        "recurrent_duplicate_count": low.recurrent_duplicate_count,
                        "active_on_object_count": low.active_on_object_count,
                        "angle_error_deg": low.angle_error_deg,
                    },
                    "128": {
                        "validation_auc": high.validation_auc,
                        "canonical_drift": high.canonical_drift,
                        "persistent_duplicate_count": high.persistent_duplicate_count,
                        "recurrent_duplicate_count": high.recurrent_duplicate_count,
                        "active_on_object_count": high.active_on_object_count,
                        "angle_error_deg": high.angle_error_deg,
                    },
                },
                "supports_128": {
                    "validation_auc_strictly_lower": auc_win,
                    "canonical_drift_strictly_lower": drift_win,
                    "persistent_duplicates_no_more_and_active_channels_no_fewer": (
                        count_support
                    ),
                    "angle_error_no_worse": angle_support,
                },
            })
        recipe_summaries[recipe] = counts

    ineligible = {
        cell.cell_id: list(cell.ineligibility_reasons)
        for cell in by_id.values()
        if not cell.eligible
    }
    source_commits = sorted(
        {cell.source_commit for cell in by_id.values() if cell.source_commit}
    )
    _require(len(source_commits) == 1,
             "primary matrix must use exactly one source commit")
    evaluation_config_hashes = {
        cell.evaluation_config_sha256 for cell in by_id.values()
    }
    _require(len(evaluation_config_hashes) == 1,
             "primary matrix evaluation configs differ")
    all_eligible = not ineligible
    guardrails_pass = all(
        summary["duplicate_active_guardrail_support"] >= 2
        and summary["angle_guardrail_support"] >= 2
        for summary in recipe_summaries.values()
    )
    required_counts = [
        summary[name]
        for summary in recipe_summaries.values()
        for name in (
            "validation_auc_strict_wins",
            "canonical_drift_strict_wins",
        )
    ]

    if not all_eligible:
        decision = "retain_64"
        reasons = ["one_or_more_primary_cells_ineligible"]
    elif not guardrails_pass:
        decision = "retain_64"
        reasons = ["count_or_angle_guardrail_not_supported_in_two_of_three"]
    elif all(count == 3 for count in required_counts):
        decision = "adopt_128"
        reasons = ["all_frozen_primary_adoption_conditions_pass"]
    elif all(count >= 2 for count in required_counts):
        decision = "run_extension_45_46"
        reasons = ["auc_or_drift_has_exactly_two_of_three_support"]
    else:
        decision = "retain_64"
        reasons = ["required_auc_or_drift_consistency_not_met"]

    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_scope": "primary_seeds_42_43_44_only",
        "decision": decision,
        "selected_head_package": (
            "128" if decision == "adopt_128" else "64" if decision == "retain_64" else None
        ),
        "extension_required": decision == "run_extension_45_46",
        "extension_seeds": (
            list(EXTENSION_SEEDS) if decision == "run_extension_45_46" else []
        ),
        "decision_reasons": reasons,
        "all_primary_cells_eligible": all_eligible,
        "ineligible_cells": dict(sorted(ineligible.items())),
        "recipe_summaries": recipe_summaries,
        "paired_values": pairs,
        "input_result_hashes": {
            cell_id: by_id[cell_id].result_content_sha256 for cell_id in sorted(by_id)
        },
        "input_source_commits": source_commits,
        "evaluation_config_sha256": next(iter(evaluation_config_hashes)),
        "rule_constraints": {
            "paired_by_recipe_and_seed": True,
            "exact_equality_is_not_a_strict_win": True,
            "post_outcome_epsilon": None,
            "scalar_score": None,
            "count_or_categorical_guardrails_never_trigger_extension": True,
            "extension_outcome_not_decided_by_this_primary_tool": True,
        },
        "statistical_scope": {
            "sample_unit": "independent_training_seed_within_recipe",
            "primary_n_per_recipe": 3,
            "inference": "descriptive_frozen_decision_rule",
            "sem_computed": False,
        },
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "--no-replace-objects", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageDecisionError("cannot establish decision-tool Git provenance") from exc


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _require(path.is_absolute(), "decision output path must be absolute")
    data = canonical_json_bytes(value, newline=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise PackageDecisionError("cannot exclusively create decision output") from exc
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require(len(args.result) == 12, "exactly twelve --result paths are required")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    cells = []
    input_files = []
    for index, supplied in enumerate(args.result):
        path = supplied.expanduser().absolute()
        data = _read_regular(path, name=f"evaluator result {index + 1}")
        cells.append(extract_cell_evidence(_strict_json(data, name=str(path))))
        input_files.append({
            "absolute_path": str(path),
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })

    decision = decide_primary_package(cells)
    decision["input_files"] = sorted(input_files, key=lambda row: row["absolute_path"])
    decision["decision_tool"] = {
        "repo_relative_path": str(script_path.relative_to(repo_root)),
        "file_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "source_commit": _git(repo_root, "rev-parse", "HEAD"),
        "tracked_worktree_clean": not bool(
            _git(repo_root, "status", "--porcelain", "--untracked-files=no")
        ),
    }
    decision["decision_content_sha256"] = canonical_sha256(decision)
    _write_exclusive(args.output.expanduser().absolute(), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
