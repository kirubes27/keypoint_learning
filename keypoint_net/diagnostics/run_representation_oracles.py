"""Run the committed, planted-only representation oracle suite.

This command deliberately has no checkpoint, model, dataset, split, or
training argument.  It constructs only the synthetic cases declared in the
committed immediate-roll manifest and sends every constructed bundle through
``keypoint_net.eval_representation.evaluate_bundle``.  It does not contain a
second implementation of the representation metrics.

The command is an execution boundary, not an authorization to execute.  The
project's v2.7 amendment requires a substantive read-only review of the exact
committed candidate before this command may be run officially.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from keypoint_net import eval_representation as evaluator
from keypoint_net import representation_evaluation_provenance as provenance


MANIFEST_SCHEMA_VERSION = "representation-oracle-case-manifest-v1"
CASE_SCHEMA_VERSION = "representation-oracle-planted-case-v1"
SUITE_RESULT_SCHEMA_VERSION = "representation-oracle-planted-suite-result-v1"
IMMEDIATE_SCOPE = "world_z_roll_and_representation_health_v2_7"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/decisions/2026-07-26/representation_oracle_cases/"
    "ORACLE_CASE_MANIFEST.json"
)
NUMERIC_REGISTRY_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "NUMERIC_CALIBRATION_v1_1.json"
)
CALIBRATION_FIXTURE_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "CALIBRATION_FIXTURE_MANIFEST.json"
)
ALLOWED_COMPARATORS = frozenset(
    {
        "exact",
        "absent",
        "present",
        "contains_exact",
        "abs_error_le_registry",
        "elementwise_abs_error_le_registry",
    }
)
BLOCKED_DATASET_BACKED_FAMILIES = ("scale", "translation", "yaw", "pitch")
HEX_DIGITS = frozenset("0123456789abcdef")
SAFE_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_LOADED_HARNESS_SOURCE_DIGEST = provenance.capture_loaded_source_digest(
    "oracle_harness_source",
    Path(__file__).absolute(),
)


class OracleHarnessError(RuntimeError):
    """Raised when the planted suite boundary or an expected result fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleHarnessError(message)


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    _require(
        not missing and not extra,
        f"{name} keys differ: missing={missing}, extra={extra}",
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _strict_float(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), f"JSON contains non-finite number {token!r}")
    return value


def _reject_constant(token: str) -> None:
    raise OracleHarnessError(f"JSON contains non-finite constant {token!r}")


def _load_strict_json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=_strict_float,
            parse_constant=_reject_constant,
        )
    except OracleHarnessError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OracleHarnessError(f"{name} is not strict valid JSON: {path}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_committed_path(relative_path: object, *, name: str) -> Path:
    _require(
        isinstance(relative_path, str) and bool(relative_path),
        f"{name} path is empty",
    )
    _require("\\" not in relative_path and ":" not in relative_path, f"{name} path is unsafe")
    pure = PurePosixPath(relative_path)
    _require(
        not pure.is_absolute()
        and bool(pure.parts)
        and "." not in pure.parts
        and ".." not in pure.parts
        and str(pure) == relative_path,
        f"{name} path is not a normalized repository-relative path",
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
    except OSError as exc:
        raise OracleHarnessError(f"{name} path cannot be inspected: {relative_path}") from exc
    _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise OracleHarnessError(f"{name} path escapes the repository") from exc
    return resolved


def _verify_bound_file(record: object, *, name: str) -> Path:
    _require(isinstance(record, Mapping), f"{name} must be a mapping")
    _exact_keys(
        record,
        required={"relative_path", "file_sha256"},
        optional={"content_hash_sha256"},
        name=name,
    )
    _require(_is_sha256(record["file_sha256"]), f"{name} file hash is invalid")
    path = _safe_committed_path(record["relative_path"], name=name)
    _require(_file_hash(path) == record["file_sha256"], f"{name} file hash mismatch")
    if "content_hash_sha256" in record:
        _require(
            _is_sha256(record["content_hash_sha256"]),
            f"{name} content hash is invalid",
        )
        parsed = _load_strict_json(path, name=name)
        _require(isinstance(parsed, Mapping), f"{name} top level must be a mapping")
        _require(
            parsed.get("content_sha256", parsed.get("content_hash_sha256"))
            == record["content_hash_sha256"],
            f"{name} declared content hash differs from its binding",
        )
    return path


def _validate_assertion(record: object, *, case_id: str, index: int) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{case_id} assertion {index} is not a mapping")
    _exact_keys(
        record,
        required={"json_pointer", "comparator", "expected"},
        optional={"tolerance_registry_key"},
        name=f"{case_id} assertion {index}",
    )
    pointer = record["json_pointer"]
    comparator = record["comparator"]
    _require(
        isinstance(pointer, str) and (pointer == "" or pointer.startswith("/")),
        f"{case_id} assertion {index} has an invalid JSON pointer",
    )
    _require(
        comparator in ALLOWED_COMPARATORS,
        f"{case_id} assertion {index} has an unsupported comparator",
    )
    tolerance_required = comparator in {
        "abs_error_le_registry",
        "elementwise_abs_error_le_registry",
    }
    _require(
        ("tolerance_registry_key" in record) == tolerance_required,
        f"{case_id} assertion {index} tolerance binding is invalid",
    )
    if tolerance_required:
        _require(
            isinstance(record["tolerance_registry_key"], str)
            and bool(record["tolerance_registry_key"]),
            f"{case_id} assertion {index} tolerance key is empty",
        )
    return dict(record)


def _validate_case(
    value: object,
    *,
    expected_case_id: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"case {expected_case_id} is not a mapping")
    required = {
        "schema_version",
        "case_id",
        "case_kind",
        "group",
        "semantic_role",
        "execution",
        "construction",
        "expectations",
        "content_hash_sha256",
    }
    _exact_keys(value, required=required, name=f"case {expected_case_id}")
    _require(value["schema_version"] == CASE_SCHEMA_VERSION, "case schema differs")
    _require(value["case_id"] == expected_case_id, "manifest/case case_id mismatch")
    _require(
        isinstance(value["case_id"], str)
        and SAFE_CASE_ID.fullmatch(value["case_id"]) is not None,
        "case_id must be a safe lowercase slug",
    )
    _require(value["case_kind"] == "planted", f"{expected_case_id} is not planted")
    _require(
        value["group"] in {
            "coordinate_estimator",
            "roll_geometry",
            "representation_health",
            "heatmap_evidence",
        },
        f"{expected_case_id} group is invalid",
    )
    _require(
        value["semantic_role"] in {"positive", "falsifier", "evidence_void"},
        f"{expected_case_id} semantic_role is invalid",
    )
    execution = value["execution"]
    _require(isinstance(execution, Mapping), f"{expected_case_id} execution is invalid")
    _exact_keys(
        execution,
        required={"entrypoint", "expected_disposition", "expected_error_code"},
        name=f"{expected_case_id} execution",
    )
    _require(
        execution["entrypoint"] == "keypoint_net.eval_representation.evaluate_bundle",
        f"{expected_case_id} does not use the production evaluator",
    )
    _require(
        execution["expected_disposition"] in {"result", "contract_rejection"},
        f"{expected_case_id} disposition is invalid",
    )
    if execution["expected_disposition"] == "result":
        _require(
            execution["expected_error_code"] is None,
            f"{expected_case_id} result case cannot name an error code",
        )
    else:
        _require(
            isinstance(execution["expected_error_code"], str)
            and bool(execution["expected_error_code"]),
            f"{expected_case_id} rejection case lacks an error code",
        )
    construction = value["construction"]
    _require(isinstance(construction, Mapping), f"{expected_case_id} construction is invalid")
    _exact_keys(
        construction,
        required={"builder_id", "parameters"},
        name=f"{expected_case_id} construction",
    )
    _require(
        isinstance(construction["builder_id"], str) and bool(construction["builder_id"]),
        f"{expected_case_id} builder_id is empty",
    )
    _require(
        isinstance(construction["parameters"], Mapping),
        f"{expected_case_id} parameters are invalid",
    )
    expectations = value["expectations"]
    _require(isinstance(expectations, Mapping), f"{expected_case_id} expectations are invalid")
    _exact_keys(
        expectations,
        required={
            "expected_critical_failures",
            "assertions",
            "forbidden_pointers",
        },
        name=f"{expected_case_id} expectations",
    )
    _require(
        isinstance(expectations["expected_critical_failures"], list)
        and all(isinstance(item, str) and bool(item) for item in expectations["expected_critical_failures"])
        and len(expectations["expected_critical_failures"])
        == len(set(expectations["expected_critical_failures"])),
        f"{expected_case_id} critical failure list is invalid",
    )
    _require(
        isinstance(expectations["assertions"], list)
        and bool(expectations["assertions"]),
        f"{expected_case_id} has no declarative expectations",
    )
    normalized_assertions = [
        _validate_assertion(item, case_id=expected_case_id, index=index)
        for index, item in enumerate(expectations["assertions"])
    ]
    _require(
        isinstance(expectations["forbidden_pointers"], list)
        and all(
            isinstance(pointer, str) and pointer.startswith("/")
            for pointer in expectations["forbidden_pointers"]
        )
        and len(expectations["forbidden_pointers"])
        == len(set(expectations["forbidden_pointers"])),
        f"{expected_case_id} forbidden pointer list is invalid",
    )
    _require(_is_sha256(value["content_hash_sha256"]), "case content hash is invalid")
    _require(
        _content_hash(value) == value["content_hash_sha256"] == expected_content_hash,
        f"{expected_case_id} content hash mismatch",
    )
    normalized = copy.deepcopy(dict(value))
    normalized["expectations"]["assertions"] = normalized_assertions
    return normalized


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and cross-bind the exact committed manifest and case definitions."""

    resolved_manifest = path.resolve(strict=True)
    _require(
        resolved_manifest == DEFAULT_MANIFEST.resolve(strict=True),
        "only the authoritative oracle case manifest is accepted",
    )
    manifest = _load_strict_json(resolved_manifest, name="oracle case manifest")
    _require(isinstance(manifest, Mapping), "oracle case manifest is not a mapping")
    _exact_keys(
        manifest,
        required={
            "schema_version",
            "status",
            "scope",
            "case_schema_version",
            "production_entrypoint",
            "existing_data_boundary",
            "blocked_dataset_backed_families",
            "planted_threshold_semantics",
            "legacy_role_subset_semantics",
            "parent_specifications",
            "bindings",
            "ordered_case_definitions",
            "determinism_rules",
            "content_hash_sha256",
        },
        name="oracle case manifest",
    )
    _require(manifest["schema_version"] == MANIFEST_SCHEMA_VERSION, "manifest schema differs")
    _require(manifest["status"] == "candidate_requires_fable_review", "manifest status is invalid")
    _require(manifest["scope"] == IMMEDIATE_SCOPE, "manifest scope differs")
    _require(manifest["case_schema_version"] == CASE_SCHEMA_VERSION, "case schema binding differs")
    _require(
        manifest["production_entrypoint"]
        == "keypoint_net.eval_representation.evaluate_bundle",
        "manifest does not bind the production evaluator",
    )
    _require(
        manifest["existing_data_boundary"]
        == "synthetic_planted_diagnostics_only_no_dataset_mutation_or_training",
        "manifest existing-data boundary differs",
    )
    _require(
        manifest["blocked_dataset_backed_families"]
        == list(BLOCKED_DATASET_BACKED_FAMILIES),
        "manifest must explicitly block scale, translation, yaw, and pitch",
    )
    _require(
        manifest["planted_threshold_semantics"]
        == {
            "status": "planted_logic_fixture_only",
            "motion_reference_magnitude_image01": 0.1,
            "motion_fraction_min": 0.1,
            "on_object_rate_min": 0.75,
            "scientific_authority": "none",
            "task_20_55_80_pass_fail_authority": False,
        },
        "planted thresholds must be labelled as logic fixtures with no "
        "scientific authority",
    )
    _require(
        manifest["legacy_role_subset_semantics"]
        == {
            "status": "not_heldout_without_proven_non_exposure",
            "task_20_55_80_role_subset_is_scientific_holdout": False,
        },
        "legacy endpoint subsets must not be promoted to held-out evidence",
    )
    _require(_is_sha256(manifest["content_hash_sha256"]), "manifest content hash is invalid")
    _require(
        _content_hash(manifest) == manifest["content_hash_sha256"],
        "manifest content hash mismatch",
    )

    parents = manifest["parent_specifications"]
    _require(isinstance(parents, list) and bool(parents), "parent specifications are empty")
    parent_paths = [_verify_bound_file(item, name=f"parent specification {index}") for index, item in enumerate(parents)]
    _require(len(parent_paths) == len(set(parent_paths)), "duplicate parent specification path")

    bindings = manifest["bindings"]
    _require(isinstance(bindings, Mapping), "manifest bindings are invalid")
    _exact_keys(
        bindings,
        required={"numeric_registry", "calibration_fixture_manifest"},
        name="manifest bindings",
    )
    numeric_path = _verify_bound_file(bindings["numeric_registry"], name="numeric registry")
    calibration_path = _verify_bound_file(
        bindings["calibration_fixture_manifest"],
        name="calibration fixture manifest",
    )
    _require(
        numeric_path == (REPOSITORY_ROOT / NUMERIC_REGISTRY_RELATIVE_PATH).resolve(),
        "numeric registry binding uses the wrong path",
    )
    _require(
        calibration_path
        == (REPOSITORY_ROOT / CALIBRATION_FIXTURE_RELATIVE_PATH).resolve(),
        "calibration fixture binding uses the wrong path",
    )

    rules = manifest["determinism_rules"]
    _require(isinstance(rules, Mapping), "determinism rules are invalid")
    _exact_keys(
        rules,
        required={
            "construction_repetitions",
            "evaluation_repetitions",
            "bundle_byte_comparison",
            "result_byte_comparison",
            "output_encoding",
        },
        name="determinism rules",
    )
    _require(
        rules
        == {
            "construction_repetitions": 2,
            "evaluation_repetitions": 2,
            "bundle_byte_comparison": "canonical_json_exact",
            "result_byte_comparison": "canonical_json_exact",
            "output_encoding": "canonical_json_utf8_single_lf",
        },
        "determinism rules differ from the frozen two-construction protocol",
    )

    entries = manifest["ordered_case_definitions"]
    _require(isinstance(entries, list) and bool(entries), "manifest has no cases")
    case_ids: list[str] = []
    case_paths: list[Path] = []
    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"case entry {index} is not a mapping")
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
        _require(isinstance(case_id, str) and bool(case_id), f"case entry {index} id is empty")
        _require(_is_sha256(entry["file_sha256"]), f"{case_id} file hash is invalid")
        _require(_is_sha256(entry["content_hash_sha256"]), f"{case_id} content hash is invalid")
        _require(
            _is_sha256(entry["expected_evaluation_config_sha256"]),
            f"{case_id} evaluation config hash is invalid",
        )
        case_path = _safe_committed_path(entry["relative_path"], name=f"case {case_id}")
        _require(_file_hash(case_path) == entry["file_sha256"], f"{case_id} file hash mismatch")
        raw_case = _load_strict_json(case_path, name=f"case {case_id}")
        cases.append(
            _validate_case(
                raw_case,
                expected_case_id=case_id,
                expected_content_hash=entry["content_hash_sha256"],
            )
        )
        case_ids.append(case_id)
        case_paths.append(case_path)
    _require(len(case_ids) == len(set(case_ids)), "duplicate case_id in manifest")
    _require(len(case_paths) == len(set(case_paths)), "duplicate case path in manifest")
    return dict(manifest), cases


def _git_head() -> str:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
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
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OracleHarnessError("cannot resolve the exact source commit") from exc
    commit = result.stdout.decode("ascii", errors="strict").strip()
    _require(
        len(commit) == 40 and all(character in HEX_DIGITS for character in commit),
        "Git returned an invalid source commit",
    )
    return commit


def _committed_record(role: str, relative_path: str) -> dict[str, str]:
    path = _safe_committed_path(relative_path, name=f"provenance role {role}")
    return {
        "role": role,
        "repo_relative_path": relative_path,
        "file_sha256": _file_hash(path),
    }


def _provenance_record(case_path: Path, *, source_commit: str) -> dict[str, Any]:
    try:
        case_relative = case_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise OracleHarnessError("case definition is outside the repository") from exc
    role_paths = dict(provenance.CORE_ROLE_PATHS)
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
        == str(
            (
                REPOSITORY_ROOT
                / provenance.CORE_ROLE_PATHS["oracle_harness_source"]
            ).resolve()
        )
        and _LOADED_HARNESS_SOURCE_DIGEST["sha256"]
        == harness_record["file_sha256"],
        "executing harness bytes differ from the provenance-bound harness file",
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


def _apply_about_centre(
    points: np.ndarray,
    *,
    angle_deg: float,
    centre: np.ndarray,
) -> np.ndarray:
    rotation = _rotation(angle_deg)
    return np.matmul(points, rotation.T) + centre - rotation @ centre


def _full_mask(frame_count: int, *, size: int = 16) -> np.ndarray:
    return np.ones((frame_count, size, size), dtype=bool)


def _central_mask(frame_count: int, *, size: int = 16) -> np.ndarray:
    mask = np.zeros((frame_count, size, size), dtype=bool)
    mask[:, 3:13, 3:13] = True
    return mask


def _bbox_diagonal(mask: np.ndarray) -> list[float]:
    height, width = mask.shape[-2:]
    values: list[float] = []
    for frame in mask:
        rows, columns = np.nonzero(frame)
        _require(rows.size > 0, "planted mask is empty")
        values.append(
            math.hypot(
                float(columns.max() - columns.min()) / float(width - 1),
                float(rows.max() - rows.min()) / float(height - 1),
            )
        )
    return values


def _json_logits_for_points(
    points: np.ndarray,
    *,
    resolution: int,
    temperature: float,
) -> list[Any]:
    """Encode exact bilinear probability mass as finite logits plus ``-inf``."""

    _require(resolution in {8, 64, 128}, "unsupported planted heatmap resolution")
    _require(temperature > 0.0, "planted temperature must be positive")
    logits = np.full(
        (*points.shape[:-1], resolution, resolution),
        -np.inf,
        dtype=np.float64,
    )
    for index in np.ndindex(points.shape[:-1]):
        x, y = (float(item) for item in points[index])
        u_float = (x + 1.0) * float(resolution - 1) / 2.0
        v_float = (y + 1.0) * float(resolution - 1) / 2.0
        _require(
            0.0 <= u_float <= resolution - 1
            and 0.0 <= v_float <= resolution - 1,
            "planted point is outside its heatmap",
        )
        u0 = int(math.floor(u_float))
        v0 = int(math.floor(v_float))
        u1 = min(u0 + 1, resolution - 1)
        v1 = min(v0 + 1, resolution - 1)
        fx = u_float - u0
        fy = v_float - v0
        weights: dict[tuple[int, int], float] = {}
        for v, y_weight in ((v0, 1.0 - fy), (v1, fy)):
            for u, x_weight in ((u0, 1.0 - fx), (u1, fx)):
                weights[(v, u)] = weights.get((v, u), 0.0) + x_weight * y_weight
        for (v, u), weight in weights.items():
            if weight > 0.0:
                logits[index + (v, u)] = temperature * math.log(weight)
    nested = logits.tolist()

    def replace_negative_infinity(value: Any) -> Any:
        if isinstance(value, list):
            return [replace_negative_infinity(item) for item in value]
        return "-inf" if value == -math.inf else float(value)

    return replace_negative_infinity(nested)


def _flat_logits(
    *,
    frame_count: int,
    channel_count: int,
    resolution: int,
) -> list[Any]:
    return np.zeros(
        (frame_count, channel_count, resolution, resolution),
        dtype=np.float64,
    ).tolist()


def _estimator_fixture_logits(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    resolution: int,
    temperature: float,
    logit_dtype: str,
) -> list[Any]:
    """Reproduce the preregistered estimator constructions exactly."""

    dtype = np.dtype(np.float32 if logit_dtype == "float32" else np.float64)
    channel_logits: list[np.ndarray] = []
    for fixture in fixtures:
        _require(
            float(fixture["temperature"]) == temperature,
            "estimator case temperature differs from its preregistered fixture",
        )
        values = np.full((resolution, resolution), -np.inf, dtype=dtype)
        v = int(fixture["v"])
        u = int(fixture["u"])
        if fixture["construction"] == "single_finite_grid_cell":
            values[v, u] = dtype.type(0.0)
        else:
            fraction_y = float(fixture["fraction_y"])
            fraction_x = float(fixture["fraction_x"])
            weights = np.asarray(
                [
                    [
                        (1.0 - fraction_y) * (1.0 - fraction_x),
                        (1.0 - fraction_y) * fraction_x,
                    ],
                    [
                        fraction_y * (1.0 - fraction_x),
                        fraction_y * fraction_x,
                    ],
                ],
                dtype=dtype,
            )
            positive = weights > 0
            local = np.full((2, 2), -np.inf, dtype=dtype)
            local[positive] = (
                dtype.type(temperature) * np.log(weights[positive])
                + dtype.type(float(fixture["logit_constant"]))
            )
            values[v : v + 2, u : u + 2] = local
        channel_logits.append(values)
    stacked = np.stack(channel_logits)
    repeated = np.stack([stacked, stacked])
    nested = repeated.tolist()

    def replace_negative_infinity(value: Any) -> Any:
        if isinstance(value, list):
            return [replace_negative_infinity(item) for item in value]
        return "-inf" if value == -math.inf else float(value)

    return replace_negative_infinity(nested)


def _base_config(*, full_primary: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "protocol": "full_primary_roll" if full_primary else "generic",
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
    if full_primary:
        config.update(
            {
                "full_rollout_horizons": list(range(1, 60)),
                "role_scoped_holdout_frame_ids": list(range(24)),
                "holdout_rollout_horizons": list(range(1, 8)),
                "closure_horizon": 60,
            }
        )
    return config


def _estimator_metadata(
    *,
    resolution: int = 64,
    logit_dtype: str = "float64",
    softmax_dtype: str = "float64",
    temperature: float = 0.37,
) -> dict[str, Any]:
    return {
        "input_height": 16,
        "input_width": 16,
        "heatmap_height": resolution,
        "heatmap_width": resolution,
        "endpoint_grid": True,
        "temperature": temperature,
        "logit_dtype": logit_dtype,
        "softmax_dtype": softmax_dtype,
        "crop": None,
        "resize": [16, 16],
        "align_corners": None,
    }


def _roll_transform(*, centre: np.ndarray, stride: int = 1) -> dict[str, Any]:
    return {
        "family": "roll",
        "physical_axis": "world_z",
        "direction": "forward",
        "signed_generator": 6.0,
        "generator_units": "degrees",
        "stride": stride,
        "stride_units": "frames",
        "cyclic": True,
        "expected_2d_family": "planar_rotation_about_projected_center",
        "projected_centre_xy": centre.tolist(),
    }


def _pair_rows(frame_count: int, *, stride: int, partition: str) -> list[dict[str, Any]]:
    if partition == "full_corpus":
        sources = range(frame_count)
        targets = [(source + stride) % frame_count for source in sources]
    else:
        sources = range(frame_count - 1)
        targets = [source + 1 for source in sources]
    return [
        {
            "pair_id": f"{partition}:{source}:{target}",
            "source_frame": int(source),
            "target_frame": int(target),
            "direction": "forward",
            "stride": stride,
            "partition": partition,
        }
        for source, target in zip(sources, targets)
    ]


def _bundle(
    *,
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
    points: np.ndarray,
    angles_deg: Sequence[float],
    centre: np.ndarray,
    operator_A: np.ndarray,
    operator_b: np.ndarray,
    masks: np.ndarray,
    estimator_metadata: Mapping[str, Any] | None = None,
    logits: list[Any] | None = None,
    full_primary: bool = False,
) -> dict[str, Any]:
    frame_count, channel_count, coordinate_count = points.shape
    _require(coordinate_count == 2 and channel_count >= 2, "invalid planted point array")
    _require(len(angles_deg) == frame_count, "angle count differs from frame count")
    _require(masks.shape == (frame_count, 16, 16), "planted mask shape differs")
    stride = 3 if full_primary else 1
    partition = "full_corpus" if full_primary else "validation"
    evaluation: dict[str, Any] = {
        "object_id": "synthetic_planted_roll",
        "seed": 0,
        "partition": partition,
        "frame_ids": list(range(frame_count)),
        "points": points.tolist(),
        "physical_states": [
            {"theta_deg": float(angle)}
            for angle in angles_deg
        ],
        "bbox_diagonal_image01": _bbox_diagonal(masks),
        "visibility": np.ones((frame_count, channel_count), dtype=bool).tolist(),
        "pair_rows": _pair_rows(
            frame_count,
            stride=stride,
            partition=partition,
        ),
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
    if logits is not None:
        evaluation["logits"] = logits
    bundle: dict[str, Any] = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "case_kind": "planted",
        "provenance": copy.deepcopy(dict(provenance_record)),
        "evaluation_config": _base_config(full_primary=full_primary),
        "estimator_metadata": dict(
            estimator_metadata
            if estimator_metadata is not None
            else _estimator_metadata()
        ),
        "transform": _roll_transform(centre=centre, stride=stride),
        "evaluation": evaluation,
        "operator": {
            "mode": "supplied",
            "A": operator_A.tolist(),
            "b": operator_b.tolist(),
        },
    }
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    return bundle


def _base_points() -> np.ndarray:
    return np.asarray(
        [
            [-0.45, -0.28],
            [0.48, -0.24],
            [0.42, 0.46],
            [-0.38, 0.51],
        ],
        dtype=np.float64,
    )


def _build_estimator(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(
        parameters,
        required={"resolution", "logit_dtype", "softmax_dtype", "temperature"},
        name=f"{case['case_id']} parameters",
    )
    resolution = int(parameters["resolution"])
    logit_dtype = str(parameters["logit_dtype"])
    softmax_dtype = str(parameters["softmax_dtype"])
    temperature = float(parameters["temperature"])
    _require(resolution in {64, 128}, "estimator case resolution must be 64 or 128")
    _require(logit_dtype in {"float32", "float64"}, "invalid planted logit dtype")
    _require(softmax_dtype == logit_dtype, "planted estimator dtypes must match")
    centre = np.zeros(2, dtype=np.float64)
    fixture_manifest = _load_strict_json(
        REPOSITORY_ROOT / CALIBRATION_FIXTURE_RELATIVE_PATH,
        name="calibration fixture manifest",
    )
    _require(
        isinstance(fixture_manifest, Mapping)
        and fixture_manifest.get("schema_version")
        == "representation-oracle-calibration-fixtures-v1",
        "calibration fixture manifest schema differs",
    )
    raw_fixtures = fixture_manifest.get("estimator_fixtures")
    _require(isinstance(raw_fixtures, list), "estimator fixture list is missing")
    selected = [
        fixture
        for fixture in raw_fixtures
        if isinstance(fixture, Mapping)
        and fixture.get("resolution") == resolution
    ]
    _require(
        len(selected) == 8,
        f"resolution {resolution} must bind all eight preregistered estimator fixtures",
    )
    expected_ids = [
        f"estimator_grid_{resolution}_corner_top_left",
        f"estimator_grid_{resolution}_corner_bottom_right",
        f"estimator_grid_{resolution}_centre",
        f"estimator_grid_{resolution}_edge_top_noncorner",
        f"estimator_grid_{resolution}_asymmetric",
        f"estimator_bilinear_{resolution}_edge_cell",
        f"estimator_bilinear_{resolution}_centre_asymmetric",
        f"estimator_bilinear_{resolution}_last_cell",
    ]
    _require(
        [fixture.get("fixture_id") for fixture in selected] == expected_ids,
        f"resolution {resolution} estimator fixture order or identity differs",
    )
    fixture_points: list[list[float]] = []
    for fixture in selected:
        construction = fixture.get("construction")
        u = fixture.get("u")
        v = fixture.get("v")
        _require(
            isinstance(u, int)
            and not isinstance(u, bool)
            and isinstance(v, int)
            and not isinstance(v, bool),
            "estimator fixture grid index is invalid",
        )
        if construction == "single_finite_grid_cell":
            fraction_x = 0.0
            fraction_y = 0.0
        else:
            _require(
                construction == "analytic_bilinear_probability_mass",
                "estimator fixture construction is unsupported",
            )
            fraction_x = float(fixture["fraction_x"])
            fraction_y = float(fixture["fraction_y"])
        fixture_points.append(
            [
                -1.0
                + 2.0 * (float(u) + fraction_x) / float(resolution - 1),
                -1.0
                + 2.0 * (float(v) + fraction_y) / float(resolution - 1),
            ]
        )
    base = np.asarray(fixture_points, dtype=np.float64)
    # Both frames intentionally replay the same preregistered estimator set.
    # This case tests only coordinate extraction; roll geometry has separate
    # planted cases and is not inferred from these fixture points.
    points = np.stack([base, base])
    rotation = _rotation(6.0)
    logits = _estimator_fixture_logits(
        selected,
        resolution=resolution,
        temperature=temperature,
        logit_dtype=logit_dtype,
    )
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=[0.0, 6.0],
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(2),
        estimator_metadata=_estimator_metadata(
            resolution=resolution,
            logit_dtype=logit_dtype,
            softmax_dtype=softmax_dtype,
            temperature=temperature,
        ),
        logits=logits,
    )


def _build_roll_centered_full_primary(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"physical_step_deg"}, name=f"{case['case_id']} parameters")
    _require(float(parameters["physical_step_deg"]) == 2.0, "full roll step must be 2 degrees")
    centre = np.zeros(2, dtype=np.float64)
    angles = [2.0 * frame for frame in range(180)]
    base = _base_points()
    points = np.stack(
        [
            _apply_about_centre(base, angle_deg=angle, centre=centre)
            for angle in angles
        ]
    )
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(180),
        full_primary=True,
    )


def _build_roll_offcentre(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"projected_centre_xy"}, name=f"{case['case_id']} parameters")
    centre = np.asarray(parameters["projected_centre_xy"], dtype=np.float64)
    _require(centre.shape == (2,), "off-centre case centre must be xy")
    base = _base_points() + centre
    angles = [0.0, 6.0, 12.0, 18.0]
    points = np.stack(
        [
            _apply_about_centre(base, angle_deg=angle, centre=centre)
            for angle in angles
        ]
    )
    rotation = _rotation(6.0)
    bias = centre - rotation @ centre
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=bias,
        masks=_full_mask(4),
    )


def _correct_roll_points(frame_count: int, *, centre: np.ndarray) -> tuple[np.ndarray, list[float]]:
    angles = [6.0 * frame for frame in range(frame_count)]
    base = _base_points()
    return (
        np.stack(
            [
                _apply_about_centre(base, angle_deg=angle, centre=centre)
                for angle in angles
            ]
        ),
        angles,
    )


def _build_wrong_sign(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    points, angles = _correct_roll_points(4, centre=centre)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=_rotation(-6.0),
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_reflection(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    points, angles = _correct_roll_points(4, centre=centre)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_separated(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    points, angles = _correct_roll_points(4, centre=centre)
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_sliding(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"channel", "canonical_delta_xy_per_frame"}, name=f"{case['case_id']} parameters")
    channel = int(parameters["channel"])
    delta = np.asarray(parameters["canonical_delta_xy_per_frame"], dtype=np.float64)
    _require(channel == 0 and delta.shape == (2,), "controlled sliding parameters differ")
    centre = np.zeros(2, dtype=np.float64)
    base = _base_points()
    angles = [0.0, 6.0, 12.0, 18.0]
    frames: list[np.ndarray] = []
    for frame_index, angle in enumerate(angles):
        canonical = base.copy()
        canonical[channel] += frame_index * delta
        frames.append(_apply_about_centre(canonical, angle_deg=angle, centre=centre))
    points = np.stack(frames)
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_coincident(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"canonical_xy", "channel_count"}, name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    channel_count = int(parameters["channel_count"])
    canonical = np.asarray(parameters["canonical_xy"], dtype=np.float64)
    _require(channel_count == 4 and canonical.shape == (2,), "coincident parameters differ")
    base = np.repeat(canonical[None, :], channel_count, axis=0)
    angles = [0.0, 6.0, 12.0, 18.0]
    points = np.stack(
        [
            _apply_about_centre(base, angle_deg=angle, centre=centre)
            for angle in angles
        ]
    )
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_static(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    angles = [0.0, 6.0, 12.0, 18.0]
    points = np.repeat(_base_points()[None, :, :], len(angles), axis=0)
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(4),
    )


def _build_off_object(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    base = np.asarray(
        [[0.85, 0.0], [0.0, 0.85], [-0.85, 0.0], [0.0, -0.85]],
        dtype=np.float64,
    )
    angles = [0.0, 6.0, 12.0, 18.0]
    points = np.stack(
        [
            _apply_about_centre(base, angle_deg=angle, centre=centre)
            for angle in angles
        ]
    )
    _require(bool(np.all(np.abs(points) <= 1.0)), "off-object fixture leaves the image")
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_central_mask(4),
    )


def _build_peak_switch(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"swapped_channels", "resolution"}, name=f"{case['case_id']} parameters")
    swapped = list(parameters["swapped_channels"])
    resolution = int(parameters["resolution"])
    _require(swapped == [0, 1] and resolution == 64, "peak-switch parameters differ")
    centre = np.zeros(2, dtype=np.float64)
    source = _base_points()
    target = _apply_about_centre(source, angle_deg=6.0, centre=centre)
    target[[0, 1]] = target[[1, 0]]
    points = np.stack([source, target])
    temperature = 0.37
    logits = _json_logits_for_points(
        points,
        resolution=resolution,
        temperature=temperature,
    )
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=[0.0, 6.0],
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(2),
        estimator_metadata=_estimator_metadata(
            resolution=resolution,
            temperature=temperature,
        ),
        logits=logits,
    )


def _build_flat_dead(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = case["construction"]["parameters"]
    _exact_keys(parameters, required={"channel_count", "resolution"}, name=f"{case['case_id']} parameters")
    channel_count = int(parameters["channel_count"])
    resolution = int(parameters["resolution"])
    _require(channel_count == 4 and resolution == 8, "flat-dead parameters differ")
    centre = np.zeros(2, dtype=np.float64)
    points = np.zeros((2, channel_count, 2), dtype=np.float64)
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=[0.0, 6.0],
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(2),
        estimator_metadata=_estimator_metadata(
            resolution=resolution,
            temperature=1.0,
        ),
        logits=_flat_logits(
            frame_count=2,
            channel_count=channel_count,
            resolution=resolution,
        ),
    )


def _build_coordinate_only_void(
    case: Mapping[str, Any],
    provenance_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(case["construction"]["parameters"], required=set(), name=f"{case['case_id']} parameters")
    centre = np.zeros(2, dtype=np.float64)
    points, angles = _correct_roll_points(2, centre=centre)
    rotation = _rotation(6.0)
    return _bundle(
        case=case,
        provenance_record=provenance_record,
        points=points,
        angles_deg=angles,
        centre=centre,
        operator_A=rotation,
        operator_b=np.zeros(2, dtype=np.float64),
        masks=_full_mask(2),
    )


BUILDERS: Mapping[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
] = {
    "estimator_exact": _build_estimator,
    "roll_centered_full_primary_exact": _build_roll_centered_full_primary,
    "roll_offcentre_exact": _build_roll_offcentre,
    "roll_wrong_sign": _build_wrong_sign,
    "roll_reflection": _build_reflection,
    "representation_separated_attached": _build_separated,
    "representation_controlled_sliding": _build_sliding,
    "representation_all_coincident": _build_coincident,
    "representation_static_inactive": _build_static,
    "representation_off_object": _build_off_object,
    "heatmap_peak_channel_switch": _build_peak_switch,
    "heatmap_flat_dead": _build_flat_dead,
    "coordinate_only_evidence_void": _build_coordinate_only_void,
}

BUILDER_CONTRACTS: Mapping[str, dict[str, Any]] = {
    "estimator_exact": {
        "group": "coordinate_estimator",
        "semantic_role": "positive",
        "logits": "required",
    },
    "roll_centered_full_primary_exact": {
        "group": "roll_geometry",
        "semantic_role": "positive",
        "logits": "optional",
    },
    "roll_offcentre_exact": {
        "group": "roll_geometry",
        "semantic_role": "positive",
        "logits": "optional",
    },
    "roll_wrong_sign": {
        "group": "roll_geometry",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "roll_reflection": {
        "group": "roll_geometry",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "representation_separated_attached": {
        "group": "representation_health",
        "semantic_role": "positive",
        "logits": "optional",
    },
    "representation_controlled_sliding": {
        "group": "representation_health",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "representation_all_coincident": {
        "group": "representation_health",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "representation_static_inactive": {
        "group": "representation_health",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "representation_off_object": {
        "group": "representation_health",
        "semantic_role": "falsifier",
        "logits": "optional",
    },
    "heatmap_peak_channel_switch": {
        "group": "heatmap_evidence",
        "semantic_role": "falsifier",
        "logits": "required",
    },
    "heatmap_flat_dead": {
        "group": "heatmap_evidence",
        "semantic_role": "falsifier",
        "logits": "required",
    },
    "coordinate_only_evidence_void": {
        "group": "heatmap_evidence",
        "semantic_role": "evidence_void",
        "logits": "forbidden",
    },
}


def _case_path_by_id(manifest: Mapping[str, Any]) -> dict[str, Path]:
    return {
        entry["case_id"]: _safe_committed_path(
            entry["relative_path"],
            name=f"case {entry['case_id']}",
        )
        for entry in manifest["ordered_case_definitions"]
    }


_MISSING = object()


def _pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    _require(pointer.startswith("/"), f"invalid JSON pointer {pointer!r}")
    cursor = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(cursor, Mapping):
            if token not in cursor:
                return _MISSING
            cursor = cursor[token]
        elif isinstance(cursor, list):
            if not token.isdigit():
                return _MISSING
            index = int(token)
            if index >= len(cursor):
                return _MISSING
            cursor = cursor[index]
        else:
            return _MISSING
    return cursor


def _registry_tolerance(registry: Mapping[str, Any], key: str) -> float:
    registries = registry.get("metric_registries")
    _require(isinstance(registries, Mapping) and key in registries, f"unknown tolerance key {key!r}")
    record = registries[key]
    _require(isinstance(record, Mapping), f"tolerance record {key!r} is invalid")
    tolerance = record.get("tolerance")
    _require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) >= 0.0,
        f"tolerance {key!r} is invalid",
    )
    return float(tolerance)


def _same_json_value(actual: Any, expected: Any) -> bool:
    try:
        return _canonical_bytes(actual) == _canonical_bytes(expected)
    except (TypeError, ValueError):
        return False


def _numeric_leaf_errors(actual: Any, expected: Any) -> list[float]:
    if isinstance(expected, list):
        _require(isinstance(actual, list), "elementwise comparison shape differs")
        _require(len(actual) == len(expected), "elementwise comparison length differs")
        errors: list[float] = []
        for actual_item, expected_item in zip(actual, expected):
            errors.extend(_numeric_leaf_errors(actual_item, expected_item))
        return errors
    _require(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool),
        "elementwise comparison encountered a non-numeric leaf",
    )
    _require(
        math.isfinite(float(actual)) and math.isfinite(float(expected)),
        "elementwise comparison encountered a non-finite leaf",
    )
    return [abs(float(actual) - float(expected))]


def _evaluate_assertion(
    result: Mapping[str, Any],
    assertion: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    case_id: str,
    index: int,
) -> dict[str, Any]:
    pointer = assertion["json_pointer"]
    comparator = assertion["comparator"]
    expected = assertion["expected"]
    actual = _pointer(result, pointer)
    passed = False
    observed: Any = None if actual is _MISSING else actual
    tolerance: float | None = None
    if comparator == "absent":
        passed = actual is _MISSING
    elif comparator == "present":
        passed = actual is not _MISSING
    elif comparator == "exact":
        passed = actual is not _MISSING and _same_json_value(actual, expected)
    elif comparator == "contains_exact":
        passed = (
            isinstance(actual, list)
            and any(_same_json_value(item, expected) for item in actual)
        )
    elif comparator == "abs_error_le_registry":
        tolerance = _registry_tolerance(registry, assertion["tolerance_registry_key"])
        passed = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and abs(float(actual) - float(expected)) <= tolerance
        )
    elif comparator == "elementwise_abs_error_le_registry":
        tolerance = _registry_tolerance(registry, assertion["tolerance_registry_key"])
        try:
            errors = (
                _numeric_leaf_errors(actual, expected)
                if actual is not _MISSING
                else []
            )
            passed = bool(errors) and max(errors) <= tolerance
        except OracleHarnessError:
            passed = False
    else:
        raise OracleHarnessError(f"{case_id} assertion {index} comparator escaped validation")
    _require(
        passed,
        f"{case_id} assertion {index} failed at {pointer}: "
        f"comparator={comparator}, expected={expected!r}, observed={observed!r}, "
        f"tolerance={tolerance!r}",
    )
    return {
        "json_pointer": pointer,
        "comparator": comparator,
        "expected": expected,
        "observed": observed,
        "tolerance_registry_key": assertion.get("tolerance_registry_key"),
        "tolerance": tolerance,
        "passed": True,
    }


def _validate_case_result(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    numeric_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case_id = case["case_id"]
    expected = case["expectations"]
    _require(result.get("case_id") == case_id, f"{case_id} result case_id differs")
    _require(
        _same_json_value(
            result.get("critical_failures"),
            expected["expected_critical_failures"],
        ),
        f"{case_id} critical failures differ from the committed definition",
    )
    for pointer in expected["forbidden_pointers"]:
        _require(
            _pointer(result, pointer) is _MISSING,
            f"{case_id} emitted forbidden result pointer {pointer}",
        )
    return [
        _evaluate_assertion(
            result,
            assertion,
            registry=numeric_registry,
            case_id=case_id,
            index=index,
        )
        for index, assertion in enumerate(expected["assertions"])
    ]


def _require_case_specific_evidence(case: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    builder_id = case["construction"]["builder_id"]
    evaluation = bundle["evaluation"]
    _require(builder_id in BUILDER_CONTRACTS, f"{case['case_id']} lacks a builder contract")
    contract = BUILDER_CONTRACTS[builder_id]
    _require(
        case["group"] == contract["group"]
        and case["semantic_role"] == contract["semantic_role"],
        f"{case['case_id']} group or semantic role conflicts with its builder",
    )
    if contract["logits"] == "required":
        _require("logits" in evaluation, f"{case['case_id']} requires planted logits")
    elif contract["logits"] == "forbidden":
        _require("logits" not in evaluation, f"{case['case_id']} must omit logits")
    _require(bundle["case_kind"] == "planted", f"{case['case_id']} escaped planted case kind")
    _require(bundle["transform"]["family"] == "roll", f"{case['case_id']} escaped world-Z roll")
    _require(bundle["transform"]["physical_axis"] == "world_z", f"{case['case_id']} escaped world-Z")
    _require("fit" not in bundle, f"{case['case_id']} unexpectedly contains fit data")
    _require(bundle["operator"]["mode"] == "supplied", f"{case['case_id']} unexpectedly fits an operator")


def construct_case_bundle(
    case: Mapping[str, Any],
    *,
    case_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Construct one planted bundle after its exact case file was cross-bound."""

    builder_id = case["construction"]["builder_id"]
    _require(builder_id in BUILDERS, f"{case['case_id']} uses unknown builder {builder_id!r}")
    provenance_record = _provenance_record(case_path, source_commit=source_commit)
    bundle = BUILDERS[builder_id](case, provenance_record)
    _require_case_specific_evidence(case, bundle)
    return bundle


def run_suite(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_directory: Path,
) -> dict[str, Any]:
    """Run two fresh constructions/evaluations and write only verified outputs."""

    source_commit = _git_head()
    manifest, cases = load_manifest(manifest_path)
    _require(
        _git_head() == source_commit,
        "repository HEAD changed while the manifest and cases were loaded",
    )
    case_paths = _case_path_by_id(manifest)
    expected_config_hashes = {
        entry["case_id"]: entry["expected_evaluation_config_sha256"]
        for entry in manifest["ordered_case_definitions"]
    }
    numeric_registry = _load_strict_json(
        REPOSITORY_ROOT / NUMERIC_REGISTRY_RELATIVE_PATH,
        name="numeric registry",
    )
    _require(isinstance(numeric_registry, Mapping), "numeric registry top level differs")

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
        bundle_bytes = [evaluator.canonical_json_bytes(bundle) for bundle in bundles]
        _require(
            bundle_bytes[0] == bundle_bytes[1],
            f"{case_id} bundle construction is not byte deterministic",
        )
        _require(
            evaluator.canonical_sha256(bundles[0]["evaluation_config"])
            == expected_config_hashes[case_id],
            f"{case_id} evaluation_config differs from its manifest binding",
        )

        expected_disposition = case["execution"]["expected_disposition"]
        results: list[dict[str, Any]] = []
        rejection_codes: list[str] = []
        for bundle in bundles:
            try:
                # Production execution intentionally has no provenance seam.
                result = evaluator.evaluate_bundle(bundle)
            except evaluator.EvaluationContractError as exc:
                rejection_codes.append(exc.code)
            else:
                results.append(result)

        if expected_disposition == "contract_rejection":
            expected_code = case["execution"]["expected_error_code"]
            _require(
                rejection_codes == [expected_code, expected_code] and not results,
                f"{case_id} did not reproduce its committed contract rejection",
            )
            summary = {
                "case_id": case_id,
                "semantic_role": case["semantic_role"],
                "disposition": "contract_rejection",
                "error_code": expected_code,
                "bundle_content_sha256": bundles[0]["bundle_content_sha256"],
                "evaluation_config_sha256": expected_config_hashes[case_id],
                "bundle_byte_deterministic": True,
                "result_byte_deterministic": True,
                "assertions": [],
            }
        else:
            _require(
                len(results) == 2 and not rejection_codes,
                f"{case_id} did not return two evaluator results",
            )
            _require(
                all(
                    result.get("evaluation_config_sha256")
                    == expected_config_hashes[case_id]
                    and evaluator.canonical_sha256(result["evaluation_config"])
                    == expected_config_hashes[case_id]
                    for result in results
                ),
                f"{case_id} result evaluation_config or emitted hash "
                "differs from its manifest binding",
            )
            result_bytes = [evaluator.canonical_json_bytes(result) for result in results]
            _require(
                result_bytes[0] == result_bytes[1],
                f"{case_id} result is not byte deterministic",
            )
            assertions = _validate_case_result(
                case,
                results[0],
                numeric_registry=numeric_registry,
            )
            pending_outputs.extend(
                [
                    (
                        output_directory / f"{case_id}.bundle.json",
                        bundle_bytes[0] + b"\n",
                    ),
                    (
                        output_directory / f"{case_id}.result.json",
                        result_bytes[0] + b"\n",
                    ),
                ]
            )
            summary = {
                "case_id": case_id,
                "semantic_role": case["semantic_role"],
                "disposition": "result",
                "error_code": None,
                "bundle_content_sha256": bundles[0]["bundle_content_sha256"],
                "result_content_sha256": results[0]["result_content_sha256"],
                "evaluation_config_sha256": expected_config_hashes[case_id],
                "bundle_byte_deterministic": True,
                "result_byte_deterministic": True,
                "assertions": assertions,
            }
        summaries.append(summary)

    output_directory = output_directory.absolute()
    if output_directory.exists():
        _require(output_directory.is_dir(), "output path exists and is not a directory")
        _require(not any(output_directory.iterdir()), "output directory must be empty")
    else:
        output_directory.mkdir(parents=True, exist_ok=False)
    suite_result: dict[str, Any] = {
        "schema_version": SUITE_RESULT_SCHEMA_VERSION,
        "scope": IMMEDIATE_SCOPE,
        "source_commit": source_commit,
        "manifest_file_sha256": _file_hash(DEFAULT_MANIFEST),
        "manifest_content_hash_sha256": manifest["content_hash_sha256"],
        "case_count": len(summaries),
        "cases": summaries,
        "blocked_dataset_backed_families": list(BLOCKED_DATASET_BACKED_FAMILIES),
        "planted_threshold_semantics": manifest["planted_threshold_semantics"],
        "legacy_role_subset_semantics": manifest[
            "legacy_role_subset_semantics"
        ],
        "dataset_or_checkpoint_opened": False,
        "training_run": False,
        "statistical_scope": "deterministic_implementation_correctness_not_inference",
    }
    suite_result["content_hash_sha256"] = _content_hash(suite_result)
    pending_outputs.append(
        (
            output_directory / "SUITE_RESULT.json",
            evaluator.canonical_json_bytes(suite_result) + b"\n",
        )
    )
    for output_path, payload in pending_outputs:
        _require(not output_path.exists(), f"refusing to overwrite output {output_path}")
        output_path.write_bytes(payload)
    return suite_result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Authoritative planted oracle case manifest; alternate paths are rejected.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New or empty directory for canonical planted-oracle artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_suite(
        manifest_path=args.manifest,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
