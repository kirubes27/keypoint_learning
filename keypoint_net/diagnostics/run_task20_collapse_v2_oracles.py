"""Run the bounded Task-20 structural-collapse-v2 planted oracle suite.

This command constructs only seven synthetic heatmap/trajectory cases declared
in one exact committed v2 manifest.  It has no checkpoint, model, dataset,
split, optimizer, or training interface.  Every bundle is evaluated twice by
``keypoint_net.eval_representation.evaluate_bundle`` and outputs are written
only after all committed categorical assertions and byte-determinism checks
pass.

The command is an execution boundary, not execution authorization.  The exact
committed candidate still requires the project's read-only review and planted
v2 provenance profile before an official run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from keypoint_net import eval_representation as evaluator
from keypoint_net import representation_evaluation_provenance as provenance


MANIFEST_SCHEMA_VERSION = "representation-oracle-case-manifest-v2"
CASE_SCHEMA_VERSION = "representation-oracle-planted-case-v2"
SUITE_RESULT_SCHEMA_VERSION = "representation-oracle-planted-suite-result-v2"
SCOPE = "task20_structural_negative_control_collapse_v2_only"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_RELATIVE_PATH = (
    "keypoint_net/diagnostics/run_task20_collapse_v2_oracles.py"
)
MANIFEST_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/"
    "TASK20_COLLAPSE_ORACLE_MANIFEST_v2.json"
)
CASE_PREFIX = (
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
)
RESULT_PREFIX = (
    "docs/decisions/2026-07-29/representation_oracle_results_v2"
)
V1_CASE_PREFIX = "docs/decisions/2026-07-26/representation_oracle_cases"
V1_RESULT_PREFIX = "docs/decisions/2026-07-26/representation_oracle_results"
NUMERIC_REGISTRY_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "NUMERIC_CALIBRATION_v1_1.json"
)
CALIBRATION_FIXTURE_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "CALIBRATION_FIXTURE_MANIFEST.json"
)
FABLE_REVIEW_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "FABLE_5_HIGH_CROSS_BACKEND_TOLERANCE_V3_EXECUTION_REVIEW_2026-07-29.md"
)
RUNTIME_SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v3.json"
)
CHECKPOINT_PREFLIGHT_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "CHECKPOINT_HASH_PREFLIGHT_RESULT_2026-07-27.json"
)
SEMANTIC_LOCK_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "TASK20_STRUCTURAL_COLLAPSE_V2_SEMANTIC_LOCK.md"
)
DECISION_AMENDMENT_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "DECISION_SYNTHESIS_v2_8_AMENDMENT_2026-07-29.md"
)
CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "TASK55_COORDINATE_BACKEND_BRIDGE_SEMANTIC_LOCK.md"
)
IMMUTABLE_TASK20_V1_RESULT_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/results/"
    "TASK20_CHECKPOINT_REPLAY_RESULT_v1.json"
)
IMMUTABLE_TASK20_V1_FILE_SHA256 = (
    "e44ec8b839d6b39377a8acf8b5b2997334ad3c518530675cadcc322e696d2675"
)
FABLE_REVIEW_BINDING_SCHEMA = (
    "cross_backend_tolerance_v3_fable_execution_review_binding.v1"
)
DEFAULT_MANIFEST = REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH
SAFE_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SAFE_ATTEMPT_DIRECTORY = re.compile(r"^attempt_[0-9]{4}$")
HEX_DIGITS = frozenset("0123456789abcdef")
EXPECTED_CASES = (
    (
        "separated_nonflat_no_confirmed_flat_dead",
        "separated_nonflat_no_confirmed_flat_dead",
        "positive",
    ),
    (
        "healthy_separated_plus_confirmed_flat_dead",
        "healthy_separated_plus_confirmed_flat_dead",
        "positive",
    ),
    (
        "task20_shape_nine_duplicate_plus_confirmed_flat_dead",
        "task20_shape_nine_duplicate_plus_confirmed_flat_dead",
        "falsifier",
    ),
    (
        "duplicate_cluster_plus_distinct_nonflat_plus_confirmed_flat_dead",
        "duplicate_cluster_plus_distinct_nonflat_plus_confirmed_flat_dead",
        "near_miss",
    ),
    (
        "all_nonflat_duplicate",
        "all_nonflat_duplicate",
        "falsifier",
    ),
    (
        "below_minimum_nonflat_void",
        "below_minimum_nonflat_void",
        "evidence_void",
    ),
    (
        "coordinate_only_missing_logits_retains_all_channels",
        "coordinate_only_missing_logits_retains_all_channels",
        "evidence_void",
    ),
)
REVIEWED_PLANTED_INPUT_PATHS = (
    SEMANTIC_LOCK_RELATIVE_PATH,
    DECISION_AMENDMENT_RELATIVE_PATH,
    CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_RELATIVE_PATH,
    HARNESS_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    *(
        CASE_PREFIX + f"{case_id}.json"
        for case_id, _pattern, _role in EXPECTED_CASES
    ),
    RUNTIME_SOURCE_MANIFEST_RELATIVE_PATH,
    CHECKPOINT_PREFLIGHT_RELATIVE_PATH,
    IMMUTABLE_TASK20_V1_RESULT_RELATIVE_PATH,
    "keypoint_net/diagnostics/build_checkpoint_replay_manifests.py",
    "keypoint_net/diagnostics/test_run_task20_collapse_v2_oracles_contract.py",
    "keypoint_net/eval_representation.py",
    "keypoint_net/representation_array_codec.py",
    "keypoint_net/representation_checkpoint_authorization.py",
    "keypoint_net/representation_checkpoint_runtime.py",
    "keypoint_net/representation_evaluation_provenance.py",
    "keypoint_net/test_eval_representation_contract.py",
    "keypoint_net/test_representation_evaluation_provenance.py",
    "tests/test_checkpoint_replay_manifests.py",
    "tests/test_representation_checkpoint_runtime.py",
)
FROZEN_EXECUTION_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.10.19",
    "numpy_version": "2.2.6",
}
# This matrix is deliberately independent of the case-JSON assertions.  It is
# the literal semantic truth table against which official suite results are
# checked a second time.
INDEPENDENT_OUTCOME_MATRIX = {
    "separated_nonflat_no_confirmed_flat_dead": {
        "confirmed_flat_dead_channel_indices": [],
        "channels_not_confirmed_flat_dead_indices": [0, 1, 2, 3],
        "possible_retained_pair_count": 6,
        "reported_retained_pair_count": 6,
        "reported_evaluable_retained_pair_count": 6,
        "legacy_collapse": False,
        "v2_collapse": False,
        "v2_status": "available",
    },
    "healthy_separated_plus_confirmed_flat_dead": {
        "confirmed_flat_dead_channel_indices": [4],
        "channels_not_confirmed_flat_dead_indices": [0, 1, 2, 3],
        "possible_retained_pair_count": 6,
        "reported_retained_pair_count": 6,
        "reported_evaluable_retained_pair_count": 6,
        "legacy_collapse": False,
        "v2_collapse": False,
        "v2_status": "available",
    },
    "task20_shape_nine_duplicate_plus_confirmed_flat_dead": {
        "confirmed_flat_dead_channel_indices": [8],
        "channels_not_confirmed_flat_dead_indices": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
        ],
        "possible_retained_pair_count": 36,
        "reported_retained_pair_count": 36,
        "reported_evaluable_retained_pair_count": 36,
        "legacy_collapse": False,
        "v2_collapse": True,
        "v2_status": "available",
    },
    "duplicate_cluster_plus_distinct_nonflat_plus_confirmed_flat_dead": {
        "confirmed_flat_dead_channel_indices": [4],
        "channels_not_confirmed_flat_dead_indices": [0, 1, 2, 3],
        "possible_retained_pair_count": 6,
        "reported_retained_pair_count": 6,
        "reported_evaluable_retained_pair_count": 6,
        "legacy_collapse": False,
        "v2_collapse": False,
        "v2_status": "available",
    },
    "all_nonflat_duplicate": {
        "confirmed_flat_dead_channel_indices": [],
        "channels_not_confirmed_flat_dead_indices": [0, 1, 2, 3],
        "possible_retained_pair_count": 6,
        "reported_retained_pair_count": 6,
        "reported_evaluable_retained_pair_count": 6,
        "legacy_collapse": True,
        "v2_collapse": True,
        "v2_status": "available",
    },
    "below_minimum_nonflat_void": {
        "confirmed_flat_dead_channel_indices": [1, 2, 3],
        "channels_not_confirmed_flat_dead_indices": [0],
        "possible_retained_pair_count": 0,
        "reported_retained_pair_count": None,
        "reported_evaluable_retained_pair_count": None,
        "legacy_collapse": False,
        "v2_collapse": None,
        "v2_status": "void_below_minimum_channels_not_confirmed_flat_dead",
    },
    "coordinate_only_missing_logits_retains_all_channels": {
        "confirmed_flat_dead_channel_indices": [],
        "channels_not_confirmed_flat_dead_indices": [0, 1, 2, 3],
        "possible_retained_pair_count": 6,
        "reported_retained_pair_count": 6,
        "reported_evaluable_retained_pair_count": 6,
        "legacy_collapse": True,
        "v2_collapse": True,
        "v2_status": "available",
    },
}
_EXPECTED_BY_CASE_ID = {
    case_id: {"pattern": pattern, "semantic_role": role}
    for case_id, pattern, role in EXPECTED_CASES
}
_LOADED_HARNESS_SOURCE_DIGEST = provenance.capture_loaded_source_digest(
    "oracle_harness_source",
    Path(__file__).absolute(),
)


class Task20CollapseV2OracleError(RuntimeError):
    """Raised when the bounded suite contract or an assertion fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task20CollapseV2OracleError(message)


def frozen_execution_environment_record() -> dict[str, str]:
    """Return the exact reviewed interpreter/runtime or fail closed."""

    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    _require(
        observed == FROZEN_EXECUTION_ENVIRONMENT,
        "Task-20 collapse-v2 execution environment differs: "
        f"expected={FROZEN_EXECUTION_ENVIRONMENT!r}, observed={observed!r}",
    )
    return observed


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    _require(
        not missing and not extra,
        f"{name} keys differ: missing={missing}, extra={extra}",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_hash_sha256", None)
    return evaluator.canonical_sha256(payload)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Task20CollapseV2OracleError(
                f"strict JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Task20CollapseV2OracleError(
        f"strict JSON contains non-finite constant {value!r}"
    )


def _require_finite_json(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"strict JSON number at {path} is non-finite")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")


def _load_strict_json(path: Path, *, name: str) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Task20CollapseV2OracleError(
            f"cannot read strict JSON for {name}: {path}"
        ) from exc
    _require_finite_json(value)
    return value


def _safe_committed_path(relative_path: str, *, name: str) -> Path:
    _require(
        isinstance(relative_path, str) and bool(relative_path),
        f"{name} path is empty",
    )
    _require(
        "\\" not in relative_path and ":" not in relative_path,
        f"{name} path contains a forbidden separator",
    )
    pure = PurePosixPath(relative_path)
    _require(
        not pure.is_absolute()
        and bool(pure.parts)
        and "." not in pure.parts
        and ".." not in pure.parts
        and str(pure) == relative_path,
        f"{name} path is unsafe: {relative_path}",
    )
    cursor = REPOSITORY_ROOT
    try:
        for part in pure.parts:
            cursor = cursor / part
            metadata = cursor.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                f"{name} path contains a symlink: {relative_path}",
            )
    except FileNotFoundError as exc:
        raise Task20CollapseV2OracleError(
            f"{name} path does not exist: {relative_path}"
        ) from exc
    _require(cursor.is_file(), f"{name} is not a regular file: {relative_path}")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise Task20CollapseV2OracleError(
            f"{name} escapes the repository: {relative_path}"
        ) from exc
    return resolved


def _validate_file_binding(
    record: object,
    *,
    name: str,
    expected_relative_path: str | None = None,
    require_content_hash: bool = False,
) -> Path:
    _require(isinstance(record, Mapping), f"{name} binding is not a mapping")
    required = {"relative_path", "file_sha256"}
    if require_content_hash:
        required.add("content_hash_sha256")
    _exact_keys(record, required=required, name=f"{name} binding")
    relative_path = record["relative_path"]
    if expected_relative_path is not None:
        _require(
            relative_path == expected_relative_path,
            f"{name} path differs from its exact binding",
        )
    path = _safe_committed_path(relative_path, name=name)
    _require(
        _file_sha256(path) == record["file_sha256"],
        f"{name} file hash mismatch",
    )
    if require_content_hash:
        parsed = _load_strict_json(path, name=name)
        _require(isinstance(parsed, Mapping), f"{name} must contain a mapping")
        declared = parsed.get(
            "content_hash_sha256",
            parsed.get("content_sha256"),
        )
        _require(
            declared == record["content_hash_sha256"],
            f"{name} content hash differs from its binding",
        )
    return path


def _validate_assertion(
    value: object,
    *,
    case_id: str,
    index: int,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping),
        f"{case_id} assertion {index} is not a mapping",
    )
    _exact_keys(
        value,
        required={"json_pointer", "comparator", "expected"},
        name=f"{case_id} assertion {index}",
    )
    _require(
        isinstance(value["json_pointer"], str)
        and value["json_pointer"].startswith("/"),
        f"{case_id} assertion {index} has an invalid JSON pointer",
    )
    _require(
        value["comparator"] == "exact",
        f"{case_id} assertion {index} must use the exact comparator",
    )
    return dict(value)


def _validate_case(
    value: object,
    *,
    manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = manifest_entry["case_id"]
    _require(isinstance(value, Mapping), f"case {case_id} is not a mapping")
    _exact_keys(
        value,
        required={
            "schema_version",
            "case_id",
            "case_kind",
            "group",
            "semantic_role",
            "execution",
            "construction",
            "expectations",
            "content_hash_sha256",
        },
        name=f"case {case_id}",
    )
    _require(value["schema_version"] == CASE_SCHEMA_VERSION, "case schema differs")
    _require(value["case_id"] == case_id, f"{case_id} manifest/case id differs")
    _require(
        SAFE_CASE_ID.fullmatch(case_id) is not None,
        f"{case_id} is not a safe case id",
    )
    _require(
        value["case_kind"] == "planted"
        and value["group"] == "structural_collapse_v2",
        f"{case_id} escaped the bounded planted group",
    )
    expected_case = _EXPECTED_BY_CASE_ID[case_id]
    _require(
        value["semantic_role"] == expected_case["semantic_role"],
        f"{case_id} semantic role differs",
    )
    execution = value["execution"]
    _require(isinstance(execution, Mapping), f"{case_id} execution is invalid")
    _exact_keys(
        execution,
        required={"entrypoint", "expected_disposition", "expected_error_code"},
        name=f"{case_id} execution",
    )
    _require(
        execution
        == {
            "entrypoint": "keypoint_net.eval_representation.evaluate_bundle",
            "expected_disposition": "result",
            "expected_error_code": None,
        },
        f"{case_id} execution contract differs",
    )
    construction = value["construction"]
    _require(
        isinstance(construction, Mapping),
        f"{case_id} construction is invalid",
    )
    _exact_keys(
        construction,
        required={"builder_id", "parameters"},
        name=f"{case_id} construction",
    )
    _require(
        construction["builder_id"] == "task20_collapse_v2_pattern",
        f"{case_id} builder differs",
    )
    _require(
        construction["parameters"]
        == {"pattern": expected_case["pattern"], "resolution": 8},
        f"{case_id} construction parameters differ",
    )
    expectations = value["expectations"]
    _require(
        isinstance(expectations, Mapping),
        f"{case_id} expectations are invalid",
    )
    _exact_keys(
        expectations,
        required={
            "expected_critical_failures",
            "assertions",
            "forbidden_pointers",
        },
        name=f"{case_id} expectations",
    )
    _require(
        expectations["expected_critical_failures"] == [],
        f"{case_id} must not expect an evaluator critical failure",
    )
    _require(
        isinstance(expectations["assertions"], list)
        and bool(expectations["assertions"]),
        f"{case_id} has no assertions",
    )
    assertions = [
        _validate_assertion(assertion, case_id=case_id, index=index)
        for index, assertion in enumerate(expectations["assertions"])
    ]
    forbidden = expectations["forbidden_pointers"]
    _require(
        isinstance(forbidden, list)
        and all(
            isinstance(pointer, str) and pointer.startswith("/")
            for pointer in forbidden
        )
        and len(forbidden) == len(set(forbidden)),
        f"{case_id} forbidden pointers are invalid",
    )
    _require(
        _content_hash(value)
        == value["content_hash_sha256"]
        == manifest_entry["content_hash_sha256"],
        f"{case_id} content hash mismatch",
    )
    normalized = copy.deepcopy(dict(value))
    normalized["expectations"]["assertions"] = assertions
    return normalized


def load_manifest(
    path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the one exact v2 manifest and its seven hash-bound cases."""

    resolved = path.resolve(strict=True)
    _require(
        resolved == DEFAULT_MANIFEST.resolve(strict=True),
        "only the authoritative Task-20 collapse-v2 manifest is accepted",
    )
    manifest = _load_strict_json(resolved, name="Task-20 collapse-v2 manifest")
    _require(isinstance(manifest, Mapping), "v2 manifest is not a mapping")
    _exact_keys(
        manifest,
        required={
            "schema_version",
            "status",
            "scope",
            "case_schema_version",
            "production_entrypoint",
            "existing_data_boundary",
            "semantic_lock",
            "bindings",
            "ordered_case_definitions",
            "determinism_rules",
            "immutable_v1_boundaries",
            "content_hash_sha256",
        },
        name="Task-20 collapse-v2 manifest",
    )
    _require(
        manifest["schema_version"] == MANIFEST_SCHEMA_VERSION,
        "v2 manifest schema differs",
    )
    _require(
        manifest["status"] == "candidate_requires_fable_review",
        "v2 manifest status differs",
    )
    _require(manifest["scope"] == SCOPE, "v2 manifest scope differs")
    _require(
        manifest["case_schema_version"] == CASE_SCHEMA_VERSION,
        "v2 manifest case schema differs",
    )
    _require(
        manifest["production_entrypoint"]
        == "keypoint_net.eval_representation.evaluate_bundle",
        "v2 manifest does not bind the production evaluator",
    )
    _require(
        manifest["existing_data_boundary"]
        == "synthetic_planted_only_no_dataset_checkpoint_model_or_training",
        "v2 manifest execution boundary differs",
    )
    _require(
        manifest["semantic_lock"]
        == {
            "excluded_channel_predicate": "heatmap_flat_dead_is_exactly_true",
            "retained_channel_predicate": "heatmap_flat_dead_is_not_exactly_true",
            "forbidden_channel_filters": [
                "motion_active",
                "on_object",
                "health_label",
                "eligible_indices",
            ],
            "minimum_retained_channel_source": (
                "evaluation_config.minimum_eligible_channels"
            ),
            "below_minimum_disposition": "void_null_not_false",
            "collapse_predicate": (
                "every_possible_retained_pair_is_evaluable_and_"
                "persistent_duplicate"
            ),
            "numeric_threshold_change": False,
            "saved_checkpoint_execution_authority": False,
            "task_20_55_80_scientific_authority": False,
        },
        "v2 semantic lock differs",
    )
    _require(
        manifest["determinism_rules"]
        == {
            "construction_repetitions": 2,
            "evaluation_repetitions": 2,
            "bundle_byte_comparison": "canonical_json_exact",
            "result_byte_comparison": "canonical_json_exact",
            "output_encoding": "canonical_json_utf8_single_lf",
        },
        "v2 determinism rules differ",
    )
    _require(
        manifest["immutable_v1_boundaries"]
        == {
            "case_root": V1_CASE_PREFIX,
            "result_root": V1_RESULT_PREFIX,
            "mutation_authorized": False,
        },
        "v1 immutability boundary differs",
    )
    _require(
        list(INDEPENDENT_OUTCOME_MATRIX)
        == [case_id for case_id, _pattern, _role in EXPECTED_CASES],
        "independent outcome matrix does not exactly cover the ordered cases",
    )
    _require(
        _content_hash(manifest) == manifest["content_hash_sha256"],
        "v2 manifest content hash mismatch",
    )

    bindings = manifest["bindings"]
    _require(isinstance(bindings, Mapping), "v2 manifest bindings are invalid")
    _exact_keys(
        bindings,
        required={"numeric_registry", "calibration_fixture_manifest"},
        name="v2 manifest bindings",
    )
    _validate_file_binding(
        bindings["numeric_registry"],
        name="numeric registry",
        expected_relative_path=NUMERIC_REGISTRY_RELATIVE_PATH,
        require_content_hash=True,
    )
    _validate_file_binding(
        bindings["calibration_fixture_manifest"],
        name="calibration fixture manifest",
        expected_relative_path=CALIBRATION_FIXTURE_RELATIVE_PATH,
    )

    entries = manifest["ordered_case_definitions"]
    _require(
        isinstance(entries, list) and len(entries) == len(EXPECTED_CASES),
        "v2 manifest must contain exactly seven cases",
    )
    _require(
        [entry.get("case_id") for entry in entries]
        == [case_id for case_id, _pattern, _role in EXPECTED_CASES],
        "v2 manifest case order differs",
    )
    cases: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"case entry {index} is invalid")
        _exact_keys(
            entry,
            required={
                "case_id",
                "relative_path",
                "file_sha256",
                "content_hash_sha256",
                "expected_evaluation_config_sha256",
            },
            name=f"case entry {index}",
        )
        case_id = entry["case_id"]
        expected_path = f"{CASE_PREFIX}{case_id}.json"
        _require(
            entry["relative_path"] == expected_path,
            f"{case_id} path differs from the exact v2 case path",
        )
        _require(
            entry["expected_evaluation_config_sha256"]
            == evaluator.canonical_sha256(_base_config()),
            f"{case_id} evaluation-config hash differs",
        )
        case_path = _safe_committed_path(expected_path, name=f"case {case_id}")
        _require(
            _file_sha256(case_path) == entry["file_sha256"],
            f"{case_id} file hash mismatch",
        )
        cases.append(
            _validate_case(
                _load_strict_json(case_path, name=f"case {case_id}"),
                manifest_entry=entry,
            )
        )
        paths.append(case_path)
    _require(len(paths) == len(set(paths)), "duplicate v2 case path")
    return dict(manifest), cases


def _git_bytes(arguments: Sequence[str], *, name: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(REPOSITORY_ROOT),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Task20CollapseV2OracleError(
            name
        ) from exc
    return result.stdout


def _git_head() -> str:
    commit = _git_bytes(
        ["rev-parse", "--verify", "HEAD^{commit}"],
        name="cannot resolve the exact source commit",
    ).decode("ascii", errors="strict").strip()
    _require(
        len(commit) == 40 and all(character in HEX_DIGITS for character in commit),
        "Git returned an invalid source commit",
    )
    return commit


def _git_file_at_commit(commit: str, relative_path: str) -> bytes:
    return _git_bytes(
        ["show", f"{commit}:{relative_path}"],
        name=f"cannot read committed file {relative_path} at {commit}",
    )


def _review_marker(lines: Sequence[str], label: str) -> str:
    prefix = f"{label}:"
    matches = [line for line in lines if line.startswith(prefix)]
    _require(
        len(matches) == 1,
        f"Fable review must contain exactly one {label} marker",
    )
    expected_prefix = f"{label}: "
    _require(
        matches[0].startswith(expected_prefix),
        f"Fable review marker {label} is malformed",
    )
    return matches[0][len(expected_prefix) :]


def _reviewed_fileset_binding(
    *,
    reviewed_candidate_commit: str,
    execution_commit: str,
) -> tuple[str, dict[str, bytes]]:
    records: list[dict[str, str]] = []
    data_by_path: dict[str, bytes] = {}
    for relative_path in REVIEWED_PLANTED_INPUT_PATHS:
        working_path = _safe_committed_path(
            relative_path,
            name=f"reviewed input {relative_path}",
        )
        working_data = working_path.read_bytes()
        candidate_data = _git_file_at_commit(
            reviewed_candidate_commit,
            relative_path,
        )
        execution_data = _git_file_at_commit(
            execution_commit,
            relative_path,
        )
        _require(
            working_data == candidate_data == execution_data,
            f"reviewed input changed after Fable review: {relative_path}",
        )
        records.append(
            {
                "repo_relative_path": relative_path,
                "file_sha256": hashlib.sha256(working_data).hexdigest(),
            }
        )
        data_by_path[relative_path] = working_data
    fileset_sha256 = evaluator.canonical_sha256(
        sorted(records, key=lambda record: record["repo_relative_path"])
    )
    return fileset_sha256, data_by_path


def _validate_fable_review_authority() -> dict[str, str]:
    """Bind the official planted run to one exact read-only Fable review."""

    execution_commit = _git_head()
    review_path = _safe_committed_path(
        FABLE_REVIEW_RELATIVE_PATH,
        name="Fable execution review",
    )
    review_data = review_path.read_bytes()
    _require(
        review_data
        == _git_file_at_commit(execution_commit, FABLE_REVIEW_RELATIVE_PATH),
        "working Fable review differs from the committed execution review",
    )
    try:
        review_text = review_data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Task20CollapseV2OracleError(
            "Fable execution review is not valid UTF-8"
        ) from exc
    lines = review_text.splitlines()
    reviewed_candidate_commit = _review_marker(
        lines,
        "REVIEWED_CANDIDATE_COMMIT",
    )
    _require(
        len(reviewed_candidate_commit) == 40
        and all(
            character in HEX_DIGITS
            for character in reviewed_candidate_commit
        ),
        "Fable reviewed candidate commit is invalid",
    )
    _git_bytes(
        [
            "merge-base",
            "--is-ancestor",
            reviewed_candidate_commit,
            execution_commit,
        ],
        name="Fable reviewed candidate is not an execution ancestor",
    )
    changed_paths = _git_bytes(
        [
            "diff",
            "--name-only",
            f"{reviewed_candidate_commit}..{execution_commit}",
        ],
        name="cannot inspect post-review commit changes",
    ).decode("utf-8", errors="strict").splitlines()
    _require(
        changed_paths == [FABLE_REVIEW_RELATIVE_PATH],
        "only the committed Fable review may differ after the reviewed "
        f"candidate; observed={changed_paths!r}",
    )
    fileset_sha256, data_by_path = _reviewed_fileset_binding(
        reviewed_candidate_commit=reviewed_candidate_commit,
        execution_commit=execution_commit,
    )
    runtime_manifest = _load_strict_json(
        REPOSITORY_ROOT / RUNTIME_SOURCE_MANIFEST_RELATIVE_PATH,
        name="v2 runtime source manifest",
    )
    expected_markers = {
        "FABLE_REVIEW_BINDING_SCHEMA": FABLE_REVIEW_BINDING_SCHEMA,
        "REVIEWED_CANDIDATE_COMMIT": reviewed_candidate_commit,
        "RUNTIME_SOURCE_MANIFEST_FILE_SHA256": hashlib.sha256(
            data_by_path[RUNTIME_SOURCE_MANIFEST_RELATIVE_PATH]
        ).hexdigest(),
        "RUNTIME_SOURCE_MANIFEST_CONTENT_SHA256": runtime_manifest[
            "content_hash_sha256"
        ],
        "CHECKPOINT_HASH_PREFLIGHT_FILE_SHA256": hashlib.sha256(
            data_by_path[CHECKPOINT_PREFLIGHT_RELATIVE_PATH]
        ).hexdigest(),
        "TASK20_V2_SEMANTIC_LOCK_FILE_SHA256": hashlib.sha256(
            data_by_path[SEMANTIC_LOCK_RELATIVE_PATH]
        ).hexdigest(),
        "DECISION_SYNTHESIS_V2_8_AMENDMENT_FILE_SHA256": hashlib.sha256(
            data_by_path[DECISION_AMENDMENT_RELATIVE_PATH]
        ).hexdigest(),
        "CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_FILE_SHA256": hashlib.sha256(
            data_by_path[CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_RELATIVE_PATH]
        ).hexdigest(),
        "TASK20_V2_ORACLE_HARNESS_FILE_SHA256": hashlib.sha256(
            data_by_path[HARNESS_RELATIVE_PATH]
        ).hexdigest(),
        "TASK20_V2_ORACLE_MANIFEST_FILE_SHA256": hashlib.sha256(
            data_by_path[MANIFEST_RELATIVE_PATH]
        ).hexdigest(),
        "PLANTED_V2_REVIEWED_FILESET_SHA256": fileset_sha256,
        "IMMUTABLE_TASK20_V1_FILE_SHA256": (
            IMMUTABLE_TASK20_V1_FILE_SHA256
        ),
        "MODEL": "fable",
        "EFFORT": "high",
        "PERMISSION_MODE": "read_only",
        "SESSION_PERSISTENCE": "false",
        "VERDICT": "PASS_TO_RUN",
        "UNRESOLVED_P0_COUNT": "0",
        "UNRESOLVED_P1_COUNT": "0",
    }
    for label, expected in expected_markers.items():
        _require(
            _review_marker(lines, label) == expected,
            f"Fable review marker {label} differs",
        )
    _require(
        hashlib.sha256(
            data_by_path[IMMUTABLE_TASK20_V1_RESULT_RELATIVE_PATH]
        ).hexdigest()
        == IMMUTABLE_TASK20_V1_FILE_SHA256,
        "immutable Task 20 v1 file hash differs",
    )
    return {
        "reviewed_candidate_commit": reviewed_candidate_commit,
        "execution_commit": execution_commit,
        "review_repo_relative_path": FABLE_REVIEW_RELATIVE_PATH,
        "review_file_sha256": hashlib.sha256(review_data).hexdigest(),
        "reviewed_fileset_sha256": fileset_sha256,
        "verdict": "PASS_TO_RUN",
    }


def _committed_record(role: str, relative_path: str) -> dict[str, str]:
    path = _safe_committed_path(relative_path, name=f"provenance role {role}")
    return {
        "role": role,
        "repo_relative_path": relative_path,
        "file_sha256": _file_sha256(path),
    }


def _provenance_record(
    case_path: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    try:
        case_relative = case_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise Task20CollapseV2OracleError(
            "v2 case definition is outside the repository"
        ) from exc
    role_paths = dict(provenance.CORE_ROLE_PATHS)
    role_paths["oracle_harness_source"] = HARNESS_RELATIVE_PATH
    role_paths["oracle_case_manifest"] = MANIFEST_RELATIVE_PATH
    role_paths.update(
        {
            "calibration_fixture_manifest": CALIBRATION_FIXTURE_RELATIVE_PATH,
            "planted_case_definition": case_relative,
        }
    )
    committed_files = [
        _committed_record(role, relative_path)
        for role, relative_path in sorted(role_paths.items())
    ]
    harness_record = next(
        record
        for record in committed_files
        if record["role"] == "oracle_harness_source"
    )
    _require(
        _LOADED_HARNESS_SOURCE_DIGEST["absolute_path"]
        == str((REPOSITORY_ROOT / HARNESS_RELATIVE_PATH).resolve())
        and _LOADED_HARNESS_SOURCE_DIGEST["sha256"]
        == harness_record["file_sha256"],
        "executing v2 harness bytes differ from its provenance-bound file",
    )
    return {
        "schema_version": provenance.PROVENANCE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "committed_files": committed_files,
        "external_files": [],
    }


def _rotation(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    return np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )


def _base_config() -> dict[str, Any]:
    return {
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


def _pattern_channels(pattern: str) -> tuple[np.ndarray, frozenset[int]]:
    duplicate = np.asarray([0.45, 0.25], dtype=np.float64)
    if pattern == "separated_nonflat_no_confirmed_flat_dead":
        canonical = np.asarray(
            [
                [-0.65, -0.55],
                [0.65, -0.55],
                [0.65, 0.55],
                [-0.65, 0.55],
            ],
            dtype=np.float64,
        )
        dead = frozenset()
    elif pattern == "healthy_separated_plus_confirmed_flat_dead":
        canonical = np.asarray(
            [
                [-0.65, -0.55],
                [0.65, -0.55],
                [0.65, 0.55],
                [-0.65, 0.55],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        dead = frozenset({4})
    elif pattern == "task20_shape_nine_duplicate_plus_confirmed_flat_dead":
        canonical = np.repeat(duplicate[None, :], 10, axis=0)
        canonical[8] = np.zeros(2, dtype=np.float64)
        dead = frozenset({8})
    elif (
        pattern
        == "duplicate_cluster_plus_distinct_nonflat_plus_confirmed_flat_dead"
    ):
        canonical = np.asarray(
            [
                duplicate,
                duplicate,
                duplicate,
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        dead = frozenset({4})
    elif pattern == "all_nonflat_duplicate":
        canonical = np.repeat(duplicate[None, :], 4, axis=0)
        dead = frozenset()
    elif pattern == "below_minimum_nonflat_void":
        canonical = np.asarray(
            [
                duplicate,
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        dead = frozenset({1, 2, 3})
    elif pattern == "coordinate_only_missing_logits_retains_all_channels":
        canonical = np.repeat(duplicate[None, :], 4, axis=0)
        dead = frozenset()
    else:
        raise Task20CollapseV2OracleError(f"unknown v2 pattern {pattern!r}")
    return canonical, dead


def _mixed_logits(
    points: np.ndarray,
    *,
    dead_indices: frozenset[int],
    resolution: int,
    temperature: float,
) -> list[Any]:
    _require(resolution == 8, "v2 planted heatmap resolution must be eight")
    logits = np.full(
        (*points.shape[:-1], resolution, resolution),
        -np.inf,
        dtype=np.float64,
    )
    frame_count, channel_count, _ = points.shape
    for frame_index in range(frame_count):
        for channel in range(channel_count):
            if channel in dead_indices:
                logits[frame_index, channel] = 0.0
                continue
            x, y = (float(item) for item in points[frame_index, channel])
            u_float = (x + 1.0) * float(resolution - 1) / 2.0
            v_float = (y + 1.0) * float(resolution - 1) / 2.0
            _require(
                0.0 <= u_float <= resolution - 1
                and 0.0 <= v_float <= resolution - 1,
                "v2 planted point lies outside its heatmap",
            )
            u0 = int(math.floor(u_float))
            v0 = int(math.floor(v_float))
            u1 = min(u0 + 1, resolution - 1)
            v1 = min(v0 + 1, resolution - 1)
            fx = u_float - u0
            fy = v_float - v0
            weights: dict[tuple[int, int], float] = {}
            for v, weight_y in ((v0, 1.0 - fy), (v1, fy)):
                for u, weight_x in ((u0, 1.0 - fx), (u1, fx)):
                    weights[(v, u)] = (
                        weights.get((v, u), 0.0) + weight_y * weight_x
                    )
            for (v, u), weight in weights.items():
                if weight > 0.0:
                    logits[frame_index, channel, v, u] = (
                        temperature * math.log(weight)
                    )
    nested = logits.tolist()

    def replace_negative_infinity(value: Any) -> Any:
        if isinstance(value, list):
            return [replace_negative_infinity(item) for item in value]
        return "-inf" if value == -math.inf else float(value)

    return replace_negative_infinity(nested)


def _mask_with_central_hole(frame_count: int) -> np.ndarray:
    mask = np.ones((frame_count, 16, 16), dtype=bool)
    mask[:, 6:10, 6:10] = False
    return mask


def _bbox_diagonal(masks: np.ndarray) -> list[float]:
    values: list[float] = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        values.append(
            math.hypot(
                float(columns.max() - columns.min()) / 15.0,
                float(rows.max() - rows.min()) / 15.0,
            )
        )
    return values


def construct_case_bundle(
    case: Mapping[str, Any],
    *,
    case_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Construct one hard-coded synthetic v2 case."""

    pattern = case["construction"]["parameters"]["pattern"]
    resolution = int(case["construction"]["parameters"]["resolution"])
    canonical, dead_indices = _pattern_channels(pattern)
    angles = [0.0, 6.0, 12.0, 18.0]
    frames: list[np.ndarray] = []
    for angle in angles:
        rotation = _rotation(angle)
        frame = np.matmul(canonical, rotation.T)
        for channel in dead_indices:
            frame[channel] = np.zeros(2, dtype=np.float64)
        frames.append(frame)
    points = np.stack(frames)
    masks = _mask_with_central_hole(len(angles))
    include_logits = (
        pattern != "coordinate_only_missing_logits_retains_all_channels"
    )
    channel_count = points.shape[1]
    evaluation = {
        "object_id": "synthetic_task20_collapse_v2",
        "seed": 0,
        "partition": "validation",
        "frame_ids": list(range(len(angles))),
        "points": points.tolist(),
        "physical_states": [{"theta_deg": angle} for angle in angles],
        "bbox_diagonal_image01": _bbox_diagonal(masks),
        "visibility": np.ones(
            (len(angles), channel_count),
            dtype=bool,
        ).tolist(),
        "pair_rows": [
            {
                "pair_id": f"validation:{source}:{source + 1}",
                "source_frame": source,
                "target_frame": source + 1,
                "direction": "forward",
                "stride": 1,
                "partition": "validation",
            }
            for source in range(len(angles) - 1)
        ],
        "masks": {
            "values": masks.tolist(),
            "geometry": {
                "input_height": 16,
                "input_width": 16,
                "crop": None,
                "resize": [16, 16],
                "align_corners": None,
            },
        },
    }
    if include_logits:
        evaluation["logits"] = _mixed_logits(
            points,
            dead_indices=dead_indices,
            resolution=resolution,
            temperature=1.0,
        )
    bundle: dict[str, Any] = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "case_kind": "planted",
        "provenance": _provenance_record(
            case_path,
            source_commit=source_commit,
        ),
        "evaluation_config": _base_config(),
        "estimator_metadata": {
            "input_height": 16,
            "input_width": 16,
            "heatmap_height": resolution,
            "heatmap_width": resolution,
            "endpoint_grid": True,
            "temperature": 1.0,
            "logit_dtype": "float64",
            "softmax_dtype": "float64",
            "crop": None,
            "resize": [16, 16],
            "align_corners": None,
        },
        "transform": {
            "family": "roll",
            "physical_axis": "world_z",
            "direction": "forward",
            "signed_generator": 6.0,
            "generator_units": "degrees",
            "stride": 1,
            "stride_units": "frames",
            "cyclic": True,
            "expected_2d_family": "planar_rotation_about_projected_center",
            "projected_centre_xy": [0.0, 0.0],
        },
        "evaluation": evaluation,
        "operator": {
            "mode": "supplied",
            "A": _rotation(6.0).tolist(),
            "b": [0.0, 0.0],
        },
    }
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    return bundle


_MISSING = object()


def _pointer(value: Any, pointer: str) -> Any:
    cursor = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(cursor, Mapping):
            if token not in cursor:
                return _MISSING
            cursor = cursor[token]
        elif isinstance(cursor, list) and token.isdigit():
            index = int(token)
            if index >= len(cursor):
                return _MISSING
            cursor = cursor[index]
        else:
            return _MISSING
    return cursor


def _same_json_value(left: Any, right: Any) -> bool:
    return evaluator.canonical_json_bytes(left) == evaluator.canonical_json_bytes(
        right
    )


def validate_case_result(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the committed exact assertions to one evaluator result."""

    case_id = case["case_id"]
    _require(result.get("case_id") == case_id, f"{case_id} result id differs")
    _require(
        result.get("critical_failures")
        == case["expectations"]["expected_critical_failures"],
        f"{case_id} critical failures differ",
    )
    for pointer in case["expectations"]["forbidden_pointers"]:
        _require(
            _pointer(result, pointer) is _MISSING,
            f"{case_id} emitted forbidden pointer {pointer}",
        )
    rows: list[dict[str, Any]] = []
    for index, assertion in enumerate(case["expectations"]["assertions"]):
        pointer = assertion["json_pointer"]
        observed = _pointer(result, pointer)
        _require(
            observed is not _MISSING,
            f"{case_id} assertion {index} pointer is absent: {pointer}",
        )
        expected = assertion["expected"]
        _require(
            _same_json_value(observed, expected),
            f"{case_id} assertion {index} failed at {pointer}: "
            f"expected={expected!r}, observed={observed!r}",
        )
        rows.append(
            {
                "json_pointer": pointer,
                "comparator": "exact",
                "expected": expected,
                "observed": observed,
                "passed": True,
            }
        )
    return rows


def validate_independent_outcome(
    case_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Check a result against the literal matrix, never the case JSON."""

    _require(
        case_id in INDEPENDENT_OUTCOME_MATRIX,
        f"{case_id} is absent from the independent outcome matrix",
    )
    expected = INDEPENDENT_OUTCOME_MATRIX[case_id]
    _exact_keys(
        expected,
        required={
            "confirmed_flat_dead_channel_indices",
            "channels_not_confirmed_flat_dead_indices",
            "possible_retained_pair_count",
            "reported_retained_pair_count",
            "reported_evaluable_retained_pair_count",
            "legacy_collapse",
            "v2_collapse",
            "v2_status",
        },
        name=f"{case_id} independent outcome",
    )
    collapse = result.get("collapse_evidence")
    trajectory = result.get("trajectory_separation")
    _require(
        isinstance(collapse, Mapping)
        and isinstance(trajectory, Mapping),
        f"{case_id} lacks collapse or trajectory evidence",
    )
    retained = collapse.get("channels_not_confirmed_flat_dead_indices")
    confirmed_dead = collapse.get("confirmed_flat_dead_channel_indices")
    _require(
        isinstance(retained, list)
        and all(isinstance(item, int) for item in retained),
        f"{case_id} retained-channel subset is invalid",
    )
    _require(
        isinstance(confirmed_dead, list)
        and all(isinstance(item, int) for item in confirmed_dead),
        f"{case_id} confirmed-flat-dead subset is invalid",
    )
    _require(
        trajectory.get("channels_not_confirmed_flat_dead_indices")
        == retained
        and trajectory.get("confirmed_flat_dead_channel_indices")
        == confirmed_dead,
        f"{case_id} collapse and trajectory subsets disagree",
    )
    retained_metric = trajectory.get(
        "channels_not_confirmed_flat_dead",
        _MISSING,
    )
    expected_reported_pairs = expected["reported_retained_pair_count"]
    if expected_reported_pairs is None:
        _require(
            retained_metric is None
            and trajectory.get(
                "channels_not_confirmed_flat_dead_metric_void"
            )
            is True,
            f"{case_id} retained-pair metric should be void",
        )
        reported_pair_count = None
        reported_evaluable_pair_count = None
    else:
        _require(
            isinstance(retained_metric, Mapping)
            and trajectory.get(
                "channels_not_confirmed_flat_dead_metric_void"
            )
            is False,
            f"{case_id} retained-pair metric should be available",
        )
        reported_pair_count = retained_metric.get("pair_count")
        reported_evaluable_pair_count = retained_metric.get(
            "evaluable_pair_count"
        )
    legacy_collapse = collapse.get(
        "structural_negative_control_collapse",
        _MISSING,
    )
    v2_collapse = collapse.get(
        "structural_negative_control_collapse_v2_"
        "excluding_confirmed_flat_dead",
        _MISSING,
    )
    v2_status = collapse.get(
        "structural_negative_control_status_v2_"
        "excluding_confirmed_flat_dead",
        _MISSING,
    )
    _require(
        legacy_collapse is not _MISSING
        and v2_collapse is not _MISSING
        and v2_status is not _MISSING,
        f"{case_id} lacks one or more collapse verdict fields",
    )
    observed = {
        "confirmed_flat_dead_channel_indices": confirmed_dead,
        "channels_not_confirmed_flat_dead_indices": retained,
        "possible_retained_pair_count": len(retained)
        * (len(retained) - 1)
        // 2,
        "reported_retained_pair_count": reported_pair_count,
        "reported_evaluable_retained_pair_count": (
            reported_evaluable_pair_count
        ),
        "legacy_collapse": legacy_collapse,
        "v2_collapse": v2_collapse,
        "v2_status": v2_status,
    }
    _require(
        _same_json_value(observed, expected),
        f"{case_id} differs from the independent literal outcome matrix: "
        f"expected={expected!r}, observed={observed!r}",
    )
    return {
        "matrix_source": "hard_coded_in_v2_harness_not_case_json",
        "expected": copy.deepcopy(expected),
        "observed": copy.deepcopy(observed),
        "passed": True,
    }


def validate_result_evaluation_config(
    case_id: str,
    result: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Recompute and compare the production result's configuration hash."""

    evaluation_config = result.get("evaluation_config")
    declared_sha256 = result.get("evaluation_config_sha256")
    _require(
        isinstance(evaluation_config, Mapping),
        f"{case_id} result lacks its evaluation config",
    )
    recomputed_sha256 = evaluator.canonical_sha256(evaluation_config)
    _require(
        expected_sha256 == evaluator.canonical_sha256(_base_config()),
        f"{case_id} expected evaluation-config hash is not the frozen config",
    )
    _require(
        declared_sha256 == recomputed_sha256 == expected_sha256,
        f"{case_id} result evaluation-config hash differs: "
        f"expected={expected_sha256}, declared={declared_sha256}, "
        f"recomputed={recomputed_sha256}",
    )
    return {
        "expected_sha256": expected_sha256,
        "declared_sha256": declared_sha256,
        "recomputed_sha256": recomputed_sha256,
        "passed": True,
    }


def _case_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    return {
        entry["case_id"]: _safe_committed_path(
            entry["relative_path"],
            name=f"case {entry['case_id']}",
        )
        for entry in manifest["ordered_case_definitions"]
    }


def _validated_output_directory(
    output_directory: Path,
    *,
    source_commit: str,
) -> Path:
    _require(
        len(source_commit) == 40
        and all(character in HEX_DIGITS for character in source_commit),
        "source commit for v2 output is invalid",
    )
    expected_parent = (
        REPOSITORY_ROOT / RESULT_PREFIX / source_commit
    ).absolute()
    actual = output_directory.absolute()
    _require(
        actual.parent == expected_parent
        and SAFE_ATTEMPT_DIRECTORY.fullmatch(actual.name) is not None,
        "v2 output directory must be one new attempt directory below "
        f"{expected_parent}",
    )
    _require(
        not str(actual).startswith(
            str((REPOSITORY_ROOT / V1_CASE_PREFIX).absolute())
        )
        and not str(actual).startswith(
            str((REPOSITORY_ROOT / V1_RESULT_PREFIX).absolute())
        ),
        "v2 output may not overlap a v1 artifact root",
    )
    return actual


def _require_real_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            f"cannot inspect {name}: {path}"
        ) from exc
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{name} must be a real non-symlink directory: {path}",
    )


def _create_output_directory_exclusive(destination: Path) -> int:
    """Create the exact result directory and return a bound directory fd."""

    result_root = (REPOSITORY_ROOT / RESULT_PREFIX).absolute()
    reviewed_commit_root = destination.parent
    _require(
        reviewed_commit_root.parent == result_root,
        "v2 output destination has an unexpected parent",
    )
    _require_real_directory(
        result_root.parent,
        name="v2 output parent",
    )
    try:
        result_root.lstat()
    except FileNotFoundError:
        try:
            os.mkdir(result_root, 0o755)
        except OSError as exc:
            raise Task20CollapseV2OracleError(
                f"cannot exclusively create v2 result root: {result_root}"
            ) from exc
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            f"cannot inspect v2 result root: {result_root}"
        ) from exc
    _require_real_directory(result_root, name="v2 result root")
    try:
        reviewed_commit_root.lstat()
    except FileNotFoundError:
        try:
            os.mkdir(reviewed_commit_root, 0o755)
        except OSError as exc:
            raise Task20CollapseV2OracleError(
                "cannot exclusively create reviewed-commit result root: "
                f"{reviewed_commit_root}"
            ) from exc
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            "cannot inspect reviewed-commit result root: "
            f"{reviewed_commit_root}"
        ) from exc
    _require_real_directory(
        reviewed_commit_root,
        name="reviewed-commit result root",
    )

    try:
        os.mkdir(destination, 0o755)
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            f"refusing to reuse or overwrite v2 output directory {destination}"
        ) from exc
    _require_real_directory(destination, name="v2 result directory")

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(destination, flags)
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            f"cannot bind the new v2 output directory: {destination}"
        ) from exc


def _write_new_file_exclusive(
    *,
    directory_fd: int,
    filename: str,
    payload: bytes,
) -> None:
    _require(
        PurePosixPath(filename).name == filename
        and filename not in {"", ".", ".."},
        f"unsafe v2 output filename: {filename!r}",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            filename,
            flags,
            0o644,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise Task20CollapseV2OracleError(
            f"refusing to overwrite v2 output file: {filename}"
        ) from exc
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            _require(count > 0, f"short write for v2 output file: {filename}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_suite(*, output_directory: Path) -> dict[str, Any]:
    """Evaluate two fresh constructions per case and write verified v2 outputs."""

    execution_environment = frozen_execution_environment_record()
    review_authority = _validate_fable_review_authority()
    source_commit = review_authority["reviewed_candidate_commit"]
    execution_commit = review_authority["execution_commit"]
    manifest, cases = load_manifest()
    _require(
        _git_head() == execution_commit,
        "repository HEAD changed during reviewed suite load",
    )
    destination = _validated_output_directory(
        output_directory,
        source_commit=source_commit,
    )
    case_paths = _case_paths(manifest)
    expected_config_hashes = {
        entry["case_id"]: entry["expected_evaluation_config_sha256"]
        for entry in manifest["ordered_case_definitions"]
    }
    pending_outputs: list[tuple[Path, bytes]] = []
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        bundles = [
            construct_case_bundle(
                case,
                case_path=case_paths[case_id],
                source_commit=source_commit,
            )
            for _ in range(2)
        ]
        bundle_bytes = [
            evaluator.canonical_json_bytes(bundle) for bundle in bundles
        ]
        _require(
            bundle_bytes[0] == bundle_bytes[1],
            f"{case_id} bundle construction is not byte deterministic",
        )
        results = [evaluator.evaluate_bundle(bundle) for bundle in bundles]
        result_bytes = [
            evaluator.canonical_json_bytes(result) for result in results
        ]
        _require(
            result_bytes[0] == result_bytes[1],
            f"{case_id} result is not byte deterministic",
        )
        evaluation_config_checks = [
            validate_result_evaluation_config(
                case_id,
                result,
                expected_sha256=expected_config_hashes[case_id],
            )
            for result in results
        ]
        assertions = validate_case_result(case, results[0])
        independent_outcome = validate_independent_outcome(
            case_id,
            results[0],
        )
        pending_outputs.extend(
            [
                (
                    destination / f"{case_id}.bundle.json",
                    bundle_bytes[0] + b"\n",
                ),
                (
                    destination / f"{case_id}.result.json",
                    result_bytes[0] + b"\n",
                ),
            ]
        )
        summaries.append(
            {
                "case_id": case_id,
                "semantic_role": case["semantic_role"],
                "disposition": "result",
                "bundle_content_sha256": bundles[0]["bundle_content_sha256"],
                "result_content_sha256": results[0]["result_content_sha256"],
                "evaluation_config_sha256": results[0][
                    "evaluation_config_sha256"
                ],
                "evaluation_config_hash_recheck": (
                    evaluation_config_checks[0]
                ),
                "bundle_byte_deterministic": True,
                "result_byte_deterministic": True,
                "independent_literal_outcome_matrix_check": (
                    independent_outcome
                ),
                "assertions": assertions,
            }
        )
    _require(
        not destination.exists(),
        f"refusing to overwrite existing v2 output directory {destination}",
    )
    suite_result: dict[str, Any] = {
        "schema_version": SUITE_RESULT_SCHEMA_VERSION,
        "scope": SCOPE,
        "source_commit": source_commit,
        "execution_commit": execution_commit,
        "fable_review_authority": review_authority,
        "output_attempt": destination.name,
        "manifest_file_sha256": _file_sha256(DEFAULT_MANIFEST),
        "manifest_content_hash_sha256": manifest["content_hash_sha256"],
        "execution_environment": execution_environment,
        "independent_outcome_matrix_sha256": evaluator.canonical_sha256(
            INDEPENDENT_OUTCOME_MATRIX
        ),
        "case_count": len(summaries),
        "cases": summaries,
        "dataset_or_checkpoint_opened": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "training_run": False,
        "v1_artifact_mutation_authorized": False,
        "completion_status": "complete_suite_result_written_last",
        "passed": True,
        "statistical_scope": (
            "deterministic_implementation_correctness_not_inference"
        ),
    }
    suite_result["content_hash_sha256"] = _content_hash(suite_result)
    pending_outputs.append(
        (
            destination / "SUITE_RESULT.json",
            evaluator.canonical_json_bytes(suite_result) + b"\n",
        )
    )
    _require(
        _git_head() == execution_commit,
        "repository HEAD changed before v2 outputs were written",
    )
    directory_fd = _create_output_directory_exclusive(destination)
    try:
        for output_path, payload in pending_outputs:
            _write_new_file_exclusive(
                directory_fd=directory_fd,
                filename=output_path.name,
                payload=payload,
            )
    finally:
        os.close(directory_fd)
    return suite_result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help=(
            "Exact new attempt_#### directory below the reviewed source "
            "commit's v2 result root."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_suite(output_directory=args.output_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
