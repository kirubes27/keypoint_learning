"""Definition-identical comparisons for registered checkpoint replays.

This module deliberately compares only the fields frozen in
``REPLAY_REGISTRY_v1.json``.  It does not reinterpret newer representation
diagnostics and it never opens a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from keypoint_net import representation_checkpoint_replay as replay_registry


__representation_import_sha256__ = hashlib.sha256(
    Path(__file__).absolute().read_bytes()
).hexdigest()


class ReplayComparisonError(ValueError):
    """Raised when historical replay evidence is missing or disagrees."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayComparisonError(message)


def _strict_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplayComparisonError(
                    f"{name} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ReplayComparisonError(
            f"{name} has non-standard JSON constant {value!r}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayComparisonError(f"{name} is not strict UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{name} top level must be an object")
    return value


def _read_registered_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    """Read and hash one exact regular file through one no-follow descriptor."""

    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayComparisonError(f"{name} cannot be opened: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} must be a regular file")
        _require(
            metadata.st_size == expected_size_bytes,
            f"{name} size differs from replay registry",
        )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
            size += len(block)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    actual_sha256 = digest.hexdigest()
    _require(size == expected_size_bytes, f"{name} byte count changed while reading")
    _require(
        actual_sha256 == expected_sha256,
        f"{name} SHA-256 differs from replay registry",
    )
    return data, {
        "absolute_path": str(path),
        "size_bytes": size,
        "sha256": actual_sha256,
        "verification_status": "size_and_sha256_verified_from_single_open_descriptor",
    }


def _finite_number(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite number",
    )
    return float(value)


def _numeric_comparison(
    historical: Any,
    production: Any,
    *,
    absolute_max: float,
    relative_max: float,
    name: str,
) -> dict[str, Any]:
    reference = _finite_number(historical, name=f"{name} historical")
    observed = _finite_number(production, name=f"{name} production")
    absolute_difference = abs(observed - reference)
    relative_difference = (
        None if reference == 0.0 else absolute_difference / abs(reference)
    )
    absolute_pass = absolute_difference <= absolute_max
    relative_pass = (
        relative_difference is not None
        and relative_difference <= relative_max
    )
    return {
        "historical": reference,
        "production": observed,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "absolute_predicate_passed": absolute_pass,
        "relative_predicate_passed": relative_pass,
        "passed": absolute_pass or relative_pass,
    }


def _elementwise_comparison(
    historical: Any,
    production: Any,
    *,
    absolute_max: float,
    relative_max: float,
    name: str,
) -> dict[str, Any]:
    if isinstance(historical, list):
        _require(
            isinstance(production, list) and len(production) == len(historical),
            f"{name} list shape differs",
        )
        children = [
            _elementwise_comparison(
                reference,
                observed,
                absolute_max=absolute_max,
                relative_max=relative_max,
                name=f"{name}/{index}",
            )
            for index, (reference, observed) in enumerate(
                zip(historical, production)
            )
        ]
        leaves = [
            leaf
            for child in children
            for leaf in child["leaf_comparisons"]
        ]
        return {
            "passed": all(bool(leaf["passed"]) for leaf in leaves),
            "leaf_count": len(leaves),
            "leaf_comparisons": leaves,
        }
    leaf = _numeric_comparison(
        historical,
        production,
        absolute_max=absolute_max,
        relative_max=relative_max,
        name=name,
    )
    leaf["component"] = name
    return {
        "passed": bool(leaf["passed"]),
        "leaf_count": 1,
        "leaf_comparisons": [leaf],
    }


def _selected_fixture(
    registry_document: Mapping[str, Any],
    *,
    task_id: int,
) -> Mapping[str, Any]:
    fixtures = registry_document.get("fixtures")
    _require(isinstance(fixtures, list), "replay registry fixtures are invalid")
    matches = [
        fixture
        for fixture in fixtures
        if isinstance(fixture, Mapping) and fixture.get("task_id") == task_id
    ]
    _require(len(matches) == 1, f"Task {task_id} is not uniquely registered")
    return matches[0]


def compare_definition_identical_metrics(
    *,
    evaluator_result: Mapping[str, Any],
    registry_document: Mapping[str, Any],
    historical_rollout: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    """Compare the exact 245 registry-defined historical field records."""

    _require(
        isinstance(evaluator_result, Mapping),
        "evaluator result must be a mapping",
    )
    _require(
        isinstance(registry_document, Mapping),
        "registry document must be a mapping",
    )
    _require(
        isinstance(historical_rollout, Mapping),
        "historical rollout must be a mapping",
    )
    _require(
        isinstance(task_id, int) and not isinstance(task_id, bool),
        "task_id must be an integer",
    )
    # Validate every logical registry value before using its pointer contract.
    registry_bytes = replay_registry.canonical_json_bytes(registry_document)
    replay_registry.validate_replay_registry_bytes(registry_bytes)
    fixture = _selected_fixture(registry_document, task_id=task_id)

    pointer_contract = replay_registry.validate_production_pointer_contract(
        evaluator_result
    )
    _require(
        pointer_contract["resolved_value_count"] == 245,
        "production pointer contract did not resolve 245 frozen records",
    )

    replay_rule = registry_document["replay_rule"]
    absolute_max = float(replay_rule["absolute_difference_max"])
    relative_max = float(replay_rule["relative_difference_max"])
    definitions = registry_document["definition_identical_fields"]
    records: list[dict[str, Any]] = []
    scalar_component_count = 0

    for definition in definitions["operator"]:
        historical_pointer = definition["historical_pointer"]
        production_pointer = definition["production_pointer"]
        historical_value = replay_registry.resolve_json_pointer(
            historical_rollout,
            historical_pointer,
        )
        production_value = replay_registry.resolve_json_pointer(
            evaluator_result,
            production_pointer,
        )
        comparison = _elementwise_comparison(
            historical_value,
            production_value,
            absolute_max=absolute_max,
            relative_max=relative_max,
            name=f"operator:{production_pointer}",
        )
        scalar_component_count += int(comparison["leaf_count"])
        records.append(
            {
                "kind": "operator",
                "historical_pointer": historical_pointer,
                "production_pointer": production_pointer,
                "comparison": definition["comparison"],
                **comparison,
            }
        )

    rollout = definitions["cyclic_rollout"]
    production_location = rollout["production_location"]
    intermediate = replay_registry.resolve_json_pointer(
        evaluator_result,
        production_location["intermediate_horizons"]["list_pointer"],
    )
    _require(isinstance(intermediate, list), "production horizons must be a list")
    selector = production_location["intermediate_horizons"]["selector_field"]
    by_k: dict[int, Mapping[str, Any]] = {}
    for record in intermediate:
        _require(isinstance(record, Mapping), "production horizon must be an object")
        k = record.get(selector)
        _require(
            isinstance(k, int) and not isinstance(k, bool) and k not in by_k,
            "production horizon selector is invalid or duplicated",
        )
        by_k[k] = record
    closure = replay_registry.resolve_json_pointer(
        evaluator_result,
        production_location["closure"]["record_pointer"],
    )
    _require(isinstance(closure, Mapping), "production closure must be an object")

    field_mapping = rollout["field_mapping"]
    for k in range(
        int(rollout["k_values"]["first"]),
        int(rollout["k_values"]["last"]) + 1,
    ):
        production_record = (
            closure
            if k == int(rollout["k_values"]["closure_k"])
            else by_k.get(k)
        )
        _require(
            isinstance(production_record, Mapping),
            f"production result lacks rollout k={k}",
        )
        for historical_field, production_field in field_mapping.items():
            historical_pointer = rollout["historical_pointer_pattern"].format(
                k=k,
                field=historical_field,
            )
            production_pointer = (
                rollout["production_metric_subpointer_pattern"].format(
                    mapped_field=production_field
                )
            )
            historical_value = replay_registry.resolve_json_pointer(
                historical_rollout,
                historical_pointer,
            )
            production_value = replay_registry.resolve_json_pointer(
                production_record,
                production_pointer,
            )
            if historical_field == "n_samples":
                _require(
                    isinstance(historical_value, int)
                    and not isinstance(historical_value, bool)
                    and isinstance(production_value, int)
                    and not isinstance(production_value, bool),
                    f"rollout k={k} sample count must be an integer",
                )
                comparison_record: dict[str, Any] = {
                    "historical": historical_value,
                    "production": production_value,
                    "passed": production_value == historical_value,
                }
            else:
                comparison_record = _numeric_comparison(
                    historical_value,
                    production_value,
                    absolute_max=absolute_max,
                    relative_max=relative_max,
                    name=f"rollout:k={k}:{production_field}",
                )
            scalar_component_count += 1
            records.append(
                {
                    "kind": "rollout",
                    "k": k,
                    "historical_field": historical_field,
                    "production_field": production_field,
                    "comparison": (
                        "exact_integer"
                        if historical_field == "n_samples"
                        else "replay_rule"
                    ),
                    **comparison_record,
                }
            )

    _require(len(records) == 245, "comparison did not produce 245 frozen records")
    failures = [
        {
            "record_index": index,
            "kind": record["kind"],
            "k": record.get("k"),
            "historical_pointer": record.get("historical_pointer"),
            "production_pointer": record.get("production_pointer"),
        }
        for index, record in enumerate(records)
        if not bool(record["passed"])
    ]
    return {
        "fixture_id": fixture["fixture_id"],
        "task_id": task_id,
        "fixture_role": fixture["role"],
        "candidate_eligibility": fixture["expected_new_classification"][
            "candidate_eligibility"
        ],
        "replay_rule": dict(replay_rule),
        "frozen_record_count": len(records),
        "scalar_component_count": scalar_component_count,
        "passed_record_count": len(records) - len(failures),
        "failed_record_count": len(failures),
        "all_definition_identical_fields_passed": not failures,
        "failures": failures,
        "records": records,
    }


def verify_registered_historical_outputs_and_compare(
    *,
    evaluator_result: Mapping[str, Any],
    registry_path: str | os.PathLike[str],
    task_id: int,
) -> dict[str, Any]:
    """Verify all historical output files, then compare the rollout record."""

    path = Path(registry_path)
    _require(path.is_absolute(), "registry_path must be absolute")
    registry_bytes = path.read_bytes()
    replay_registry.validate_replay_registry_bytes(
        registry_bytes,
        registry_absolute_path=str(path),
    )
    registry_document = _strict_json_bytes(
        registry_bytes,
        name="replay registry",
    )
    fixture = _selected_fixture(registry_document, task_id=task_id)
    run_directory = Path(fixture["run_directory"])
    _require(run_directory.is_absolute(), "fixture run_directory must be absolute")

    verified_outputs: list[dict[str, Any]] = []
    rollout_document: dict[str, Any] | None = None
    for index, record in enumerate(fixture["historical_outputs"]):
        relative_path = record["relative_path"]
        _require(
            isinstance(relative_path, str)
            and relative_path
            and not Path(relative_path).is_absolute()
            and ".." not in Path(relative_path).parts,
            f"historical output {index} has unsafe relative_path",
        )
        output_path = run_directory / relative_path
        data, verification = _read_registered_file(
            output_path,
            expected_sha256=record["sha256"],
            expected_size_bytes=record["size_bytes"],
            name=f"historical output {index}",
        )
        verification["relative_path"] = relative_path
        verified_outputs.append(verification)
        if relative_path == "rollout/rollout_metrics.json":
            rollout_document = _strict_json_bytes(
                data,
                name="historical rollout metrics",
            )
    _require(
        rollout_document is not None,
        "fixture lacks registered rollout/rollout_metrics.json",
    )
    comparison = compare_definition_identical_metrics(
        evaluator_result=evaluator_result,
        registry_document=registry_document,
        historical_rollout=rollout_document,
        task_id=task_id,
    )
    comparison["verified_historical_outputs"] = verified_outputs
    return comparison


__all__ = [
    "ReplayComparisonError",
    "compare_definition_identical_metrics",
    "verify_registered_historical_outputs_and_compare",
]
