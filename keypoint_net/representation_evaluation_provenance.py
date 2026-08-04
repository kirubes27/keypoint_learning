"""Fail-closed provenance validation for representation evaluation.

This module deliberately separates version-controlled execution inputs from
external binary/data artifacts:

* committed files are resolved relative to the repository, compared byte for
  byte with the blob at one exact Git commit, and then SHA-256 rehashed;
* external files are never allowed to satisfy a committed role and are
  independently checked by absolute path, byte size, and SHA-256.

The public validator derives its repository root from this module.  The
underscore-prefixed variant exists only so tests can exercise the same logic in
an isolated temporary Git repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = "representation_evaluation_provenance.v1"

CORE_ROLE_PATHS = MappingProxyType(
    {
        "provenance_source": (
            "keypoint_net/representation_evaluation_provenance.py"
        ),
        "evaluator_source": "keypoint_net/eval_representation.py",
        "array_codec_source": (
            "keypoint_net/representation_array_codec.py"
        ),
        "split_adapter_source": (
            "keypoint_net/representation_split_adapter.py"
        ),
        "oracle_harness_source": (
            "keypoint_net/diagnostics/run_representation_oracles.py"
        ),
        "oracle_case_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_cases/"
            "ORACLE_CASE_MANIFEST.json"
        ),
        "numeric_registry": (
            "docs/decisions/2026-07-26/representation_oracle_calibration/"
            "NUMERIC_CALIBRATION_v1_1.json"
        ),
    }
)

PLANTED_PROFILE_PATHS = MappingProxyType(
    {
        "v1": MappingProxyType(
            {
                "oracle_harness_source": (
                    "keypoint_net/diagnostics/run_representation_oracles.py"
                ),
                "oracle_case_manifest": (
                    "docs/decisions/2026-07-26/"
                    "representation_oracle_cases/ORACLE_CASE_MANIFEST.json"
                ),
                "planted_case_definition_prefix": (
                    "docs/decisions/2026-07-26/"
                    "representation_oracle_cases/cases/"
                ),
            }
        ),
        "task20_collapse_v2": MappingProxyType(
            {
                "oracle_harness_source": (
                    "keypoint_net/diagnostics/"
                    "run_task20_collapse_v2_oracles.py"
                ),
                "oracle_case_manifest": (
                    "docs/decisions/2026-07-29/"
                    "representation_oracle_cases_v2/"
                    "TASK20_COLLAPSE_ORACLE_MANIFEST_v2.json"
                ),
                "planted_case_definition_prefix": (
                    "docs/decisions/2026-07-29/"
                    "representation_oracle_cases_v2/cases/"
                ),
            }
        ),
    }
)
_TASK20_COLLAPSE_V2_CASE_PATHS = (
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "separated_nonflat_no_confirmed_flat_dead.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "healthy_separated_plus_confirmed_flat_dead.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "task20_shape_nine_duplicate_plus_confirmed_flat_dead.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "duplicate_cluster_plus_distinct_nonflat_plus_confirmed_flat_dead.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "all_nonflat_duplicate.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "below_minimum_nonflat_void.json",
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/cases/"
    "coordinate_only_missing_logits_retains_all_channels.json",
)

FIXED_ROLE_PATHS = MappingProxyType(
    {
        "calibration_fixture_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_calibration/"
            "CALIBRATION_FIXTURE_MANIFEST.json"
        ),
        "split_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "SPLIT_MANIFEST.json"
        ),
        "split_verifier_report": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "SPLIT_VERIFIER_REPORT.json"
        ),
        "geometry_registry": (
            "docs/decisions/2026-07-26/representation_oracle_geometry/"
            "GEOMETRY_BINDING_REGISTRY_v1.json"
        ),
        "replay_registry": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "REPLAY_REGISTRY_v1.json"
        ),
        "checkpoint_runtime_source_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v3.json"
        ),
        "checkpoint_hash_preflight": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_HASH_PREFLIGHT_RESULT_2026-07-27.json"
        ),
        "replay_dataset_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "inventories/CORPUS_INVENTORY__roll.json"
        ),
        "replay_pair_index": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_PAIR_INDEX_MANIFEST_v1.json"
        ),
        "replay_frame_mask_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_FRAME_MASK_INVENTORY_v1.json"
        ),
        "replay_metadata_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_METADATA_MANIFEST_v1.json"
        ),
        "replay_pair_index": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_PAIR_INDEX_MANIFEST_v1.json"
        ),
        "checkpoint_execution_authorization": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_EXECUTION_AUTHORIZATION_v3.json"
        ),
        "fable_execution_review": (
            "docs/decisions/2026-07-29/"
            "FABLE_5_HIGH_CROSS_BACKEND_TOLERANCE_V3_EXECUTION_REVIEW_2026-07-29.md"
        ),
    }
)

ROLE_PATH_PREFIXES = MappingProxyType(
    {
        "planted_case_definition": (
            "docs/decisions/2026-07-26/representation_oracle_cases/cases/"
        ),
        "evaluation_pair_artifact": (
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
        ),
        "fit_pair_artifact": (
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
        ),
        "corpus_inventory": (
            "docs/decisions/2026-07-26/"
            "representation_oracle_splits/inventories/"
        ),
        "geometry_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
        ),
    }
)

LOADED_SOURCE_ROLES = frozenset(
    {
        "evaluator_source",
        "array_codec_source",
    }
)

_PLANTED_COMMITTED_ROLES = frozenset(
    {
        "calibration_fixture_manifest",
        "planted_case_definition",
    }
)
_DATASET_COMMITTED_ROLES = frozenset(
    {
        "split_manifest",
        "split_verifier_report",
        "evaluation_pair_artifact",
        "corpus_inventory",
        "geometry_registry",
        "geometry_manifest",
    }
)
_DATASET_EXTERNAL_ROLES = frozenset({"prediction_artifact"})
_CHECKPOINT_COMMITTED_ROLES = frozenset(
    {
        "replay_registry",
        "geometry_registry",
        "geometry_manifest",
        "checkpoint_runtime_source_manifest",
        "checkpoint_hash_preflight",
        "replay_dataset_inventory",
        "replay_pair_index",
        "replay_frame_mask_inventory",
        "replay_metadata_manifest",
        "checkpoint_execution_authorization",
        "fable_execution_review",
    }
)
_CHECKPOINT_EXTERNAL_ROLES = frozenset(
    {
        "checkpoint",
        "checkpoint_config",
        "checkpoint_metadata",
    }
)
FRESH_CHECKPOINT_ROLE_PATHS = MappingProxyType(
    {
        "provenance_source": "keypoint_net/representation_evaluation_provenance.py",
        "evaluator_source": "keypoint_net/eval_representation.py",
        "array_codec_source": "keypoint_net/representation_array_codec.py",
        "split_adapter_source": "keypoint_net/representation_split_adapter.py",
        "numeric_registry": (
            "docs/decisions/2026-07-26/representation_oracle_calibration/"
            "NUMERIC_CALIBRATION_v1_1.json"
        ),
        "fresh_authorization_source": (
            "keypoint_net/representation_fresh_checkpoint_authorization.py"
        ),
        "fresh_runtime_source": (
            "keypoint_net/representation_fresh_checkpoint_runtime.py"
        ),
        "model_source": "keypoint_net/model.py",
        "experiment_manifest": (
            "docs/decisions/2026-07-29/roll_head_package_training/"
            "EXPERIMENT_MANIFEST_v1.json"
        ),
        "split_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "SPLIT_MANIFEST.json"
        ),
        "split_verifier_report": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "SPLIT_VERIFIER_REPORT.json"
        ),
        "corpus_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "inventories/CORPUS_INVENTORY__roll.json"
        ),
        "replay_frame_mask_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_FRAME_MASK_INVENTORY_v1.json"
        ),
        "replay_metadata_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_METADATA_MANIFEST_v1.json"
        ),
    }
)
FRESH_CHECKPOINT_EXTERNAL_ROLES = frozenset(
    {"checkpoint", "checkpoint_config", "checkpoint_metadata", "completed_run_receipt"}
)
_CHECKPOINT_PROVENANCE_LOAD_RECEIPT_FIELDS = frozenset(
    {
        "role",
        "absolute_path",
        "file_sha256",
        "size_bytes",
        "task_id",
        "fixture_id",
        "source_commit",
        "same_open_file_descriptor_hash_and_load",
        "weights_only",
    }
)
_FIT_ROLE = "fit_pair_artifact"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})

class ProvenanceContractError(ValueError):
    """Raised when an evaluation provenance claim is not fully verified."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceContractError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(
        not missing and not extra,
        f"{name} keys differ: missing={missing}, extra={extra}",
    )


def capture_loaded_source_digest(
    role: str,
    source_file: str | os.PathLike[str],
) -> dict[str, str]:
    """Capture a source digest at module import time.

    Executable modules should call this once at import and pass the returned
    immutable-by-convention record to :func:`validate_evaluation_provenance`.
    The validator also re-reads the working file, so both the import-time view
    and validation-time bytes must match the committed blob.
    """

    _require(isinstance(role, str) and bool(role), "loaded source role is empty")
    path = Path(source_file)
    _require(path.is_absolute(), f"loaded source {role} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProvenanceContractError(
            f"loaded source {role} cannot be inspected: {path}"
        ) from exc
    _require(
        not stat.S_ISLNK(metadata.st_mode),
        f"loaded source {role} must not be a symlink",
    )
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"loaded source {role} must be a regular file",
    )
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256_bytes(data),
    }


def required_role_profile(
    case_kind: str,
    *,
    fit_from_pairs: bool,
    checkpoint_profile: str = "fixture",
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the exact committed and external role sets for one case."""

    _require(
        case_kind in {"planted", "dataset", "checkpoint"},
        f"unsupported provenance case kind: {case_kind!r}",
    )
    _require(
        isinstance(fit_from_pairs, bool),
        "fit_from_pairs must be a boolean",
    )
    _require(
        checkpoint_profile in {"fixture", "fresh_run"},
        f"unsupported checkpoint profile: {checkpoint_profile!r}",
    )
    if case_kind == "checkpoint" and checkpoint_profile == "fresh_run":
        _require(not fit_from_pairs, "fresh roll checkpoint forbids fit_from_pairs")
        return (
            frozenset(FRESH_CHECKPOINT_ROLE_PATHS),
            FRESH_CHECKPOINT_EXTERNAL_ROLES,
        )
    _require(
        checkpoint_profile == "fixture",
        "fresh_run checkpoint profile requires case_kind=checkpoint",
    )
    committed = set(CORE_ROLE_PATHS)
    external: set[str] = set()
    if case_kind == "planted":
        committed.update(_PLANTED_COMMITTED_ROLES)
    elif case_kind == "dataset":
        committed.update(_DATASET_COMMITTED_ROLES)
        external.update(_DATASET_EXTERNAL_ROLES)
        if fit_from_pairs:
            committed.add(_FIT_ROLE)
    else:
        committed.update(_CHECKPOINT_COMMITTED_ROLES)
        external.update(_CHECKPOINT_EXTERNAL_ROLES)
        if fit_from_pairs:
            committed.add(_FIT_ROLE)
    return frozenset(committed), frozenset(external)


def required_loaded_source_roles(case_kind: str) -> frozenset[str]:
    _require(
        case_kind in {"planted", "dataset", "checkpoint"},
        f"unsupported provenance case kind: {case_kind!r}",
    )
    roles = set(LOADED_SOURCE_ROLES)
    if case_kind == "planted":
        roles.add("oracle_harness_source")
    return frozenset(roles)


def _resolve_planted_profile(
    *,
    case_kind: str,
    committed_paths_by_role: Mapping[str, str],
) -> str:
    """Resolve one exact harness/manifest pair; never accept mixed profiles."""

    harness_path = committed_paths_by_role.get("oracle_harness_source")
    manifest_path = committed_paths_by_role.get("oracle_case_manifest")
    matches = [
        profile
        for profile, paths in PLANTED_PROFILE_PATHS.items()
        if harness_path == paths["oracle_harness_source"]
        and manifest_path == paths["oracle_case_manifest"]
    ]
    _require(
        len(matches) == 1,
        "oracle harness/manifest paths do not form one allowed planted profile",
    )
    profile = matches[0]
    if case_kind != "planted":
        _require(
            profile == "v1",
            "task20_collapse_v2 planted profile is forbidden for "
            f"{case_kind} cases",
        )
    return profile


def _strict_json_document(data: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    def reject_duplicate_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            data.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProvenanceContractError(f"{name} is not strict JSON") from exc
    _require(isinstance(document, Mapping), f"{name} must be a JSON object")
    return document


def _validate_task20_v2_case_manifest_membership(
    *,
    repo_root: Path,
    source_commit: str,
    manifest_path: Path,
    selected_case_relative_path: str,
    selected_case_sha256: str,
) -> None:
    """Bind a v2 planted case to all seven committed manifest members."""

    manifest = _strict_json_document(
        manifest_path.read_bytes(),
        name="Task 20 collapse-v2 oracle manifest",
    )
    _require(
        manifest.get("schema_version")
        == "representation-oracle-case-manifest-v2",
        "Task 20 collapse-v2 oracle manifest schema is invalid",
    )
    claimed_content_hash = manifest.get("content_hash_sha256")
    _require(
        isinstance(claimed_content_hash, str)
        and _SHA256_RE.fullmatch(claimed_content_hash) is not None,
        "Task 20 collapse-v2 oracle manifest content hash is invalid",
    )
    manifest_payload = dict(manifest)
    manifest_payload.pop("content_hash_sha256")
    encoded_payload = json.dumps(
        manifest_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _require(
        _sha256_bytes(encoded_payload) == claimed_content_hash,
        "Task 20 collapse-v2 oracle manifest content hash mismatch",
    )
    entries = manifest.get("ordered_case_definitions")
    _require(
        isinstance(entries, list) and len(entries) == 7,
        "Task 20 collapse-v2 oracle manifest must contain exactly seven cases",
    )
    _require(
        tuple(
            entry.get("relative_path")
            for entry in entries
            if isinstance(entry, Mapping)
        )
        == _TASK20_COLLAPSE_V2_CASE_PATHS,
        "Task 20 collapse-v2 oracle manifest paths/order differ",
    )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    matched_selected = 0
    expected_prefix = PLANTED_PROFILE_PATHS["task20_collapse_v2"][
        "planted_case_definition_prefix"
    ]
    for index, entry in enumerate(entries):
        _require(
            isinstance(entry, Mapping)
            and set(entry)
            == {
                "case_id",
                "relative_path",
                "file_sha256",
                "content_hash_sha256",
                "expected_evaluation_config_sha256",
            },
            f"Task 20 collapse-v2 manifest case {index} keys differ",
        )
        case_id = entry["case_id"]
        relative_path = entry["relative_path"]
        file_sha256 = entry["file_sha256"]
        _require(
            isinstance(case_id, str)
            and bool(case_id)
            and case_id not in seen_ids,
            f"Task 20 collapse-v2 manifest case {index} id is invalid",
        )
        _require(
            isinstance(relative_path, str)
            and relative_path.startswith(expected_prefix)
            and relative_path.endswith(".json")
            and relative_path not in seen_paths,
            f"Task 20 collapse-v2 manifest case {index} path is invalid",
        )
        _require(
            isinstance(file_sha256, str)
            and _SHA256_RE.fullmatch(file_sha256) is not None,
            f"Task 20 collapse-v2 manifest case {index} hash is invalid",
        )
        seen_ids.add(case_id)
        seen_paths.add(relative_path)
        case_path = _safe_committed_path(
            repo_root,
            relative_path,
            role=f"Task 20 collapse-v2 manifest case {index}",
        )
        _blob_oid, blob_bytes = _committed_blob(
            repo_root,
            source_commit,
            relative_path,
            role=f"Task 20 collapse-v2 manifest case {index}",
        )
        current_bytes = case_path.read_bytes()
        _require(
            current_bytes == blob_bytes
            and _sha256_bytes(current_bytes) == file_sha256,
            f"Task 20 collapse-v2 manifest case {index} bytes/hash mismatch",
        )
        if relative_path == selected_case_relative_path:
            matched_selected += 1
            _require(
                file_sha256 == selected_case_sha256,
                "selected Task 20 collapse-v2 case hash differs from manifest",
            )
    _require(
        matched_selected == 1,
        "selected Task 20 collapse-v2 case is not exactly one manifest member",
    )


def _safe_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    return environment


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    context: str,
) -> bytes:
    command = [
        "git",
        "--no-replace-objects",
        "-C",
        str(repo_root),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ProvenanceContractError(f"{context}{suffix}") from exc
    return result.stdout


def _validated_repository_root(repo_root: Path) -> Path:
    try:
        resolved = repo_root.resolve(strict=True)
    except OSError as exc:
        raise ProvenanceContractError(
            f"repository root does not exist: {repo_root}"
        ) from exc
    _require(resolved.is_dir(), f"repository root is not a directory: {resolved}")
    reported = _git(
        resolved,
        ["rev-parse", "--show-toplevel"],
        context="cannot resolve Git repository root",
    ).decode("utf-8", errors="strict").strip()
    try:
        reported_path = Path(reported).resolve(strict=True)
    except OSError as exc:
        raise ProvenanceContractError(
            f"Git reported an invalid repository root: {reported!r}"
        ) from exc
    _require(
        reported_path == resolved,
        f"repository root mismatch: expected {resolved}, Git reported {reported_path}",
    )
    return resolved


def _validate_source_commit(repo_root: Path, source_commit: object) -> str:
    _require(
        isinstance(source_commit, str)
        and _COMMIT_RE.fullmatch(source_commit) is not None,
        "source_commit must be a full lowercase 40-hex commit id",
    )
    _git(
        repo_root,
        ["cat-file", "-e", f"{source_commit}^{{commit}}"],
        context=f"source_commit is not an existing commit: {source_commit}",
    )
    resolved = _git(
        repo_root,
        ["rev-parse", "--verify", f"{source_commit}^{{commit}}"],
        context=f"cannot resolve source_commit: {source_commit}",
    ).decode("ascii", errors="strict").strip()
    _require(
        resolved == source_commit,
        f"source_commit did not resolve to itself: {source_commit} -> {resolved}",
    )
    return source_commit


def _safe_committed_path(repo_root: Path, relative_path: object, *, role: str) -> Path:
    _require(
        isinstance(relative_path, str) and bool(relative_path),
        f"committed role {role} repo_relative_path is empty",
    )
    _require("\\" not in relative_path, f"committed role {role} path contains backslash")
    _require(":" not in relative_path, f"committed role {role} path contains colon")
    pure = PurePosixPath(relative_path)
    _require(
        not pure.is_absolute()
        and bool(pure.parts)
        and "." not in pure.parts
        and ".." not in pure.parts
        and str(pure) == relative_path,
        f"committed role {role} has unsafe repo-relative path: {relative_path}",
    )

    cursor = repo_root
    try:
        for part in pure.parts:
            cursor = cursor / part
            metadata = cursor.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                f"committed role {role} path contains a symlink: {relative_path}",
            )
    except FileNotFoundError as exc:
        raise ProvenanceContractError(
            f"committed role {role} working file is missing: {relative_path}"
        ) from exc
    except OSError as exc:
        raise ProvenanceContractError(
            f"committed role {role} working file cannot be inspected: {relative_path}"
        ) from exc

    _require(
        stat.S_ISREG(metadata.st_mode),
        f"committed role {role} is not a regular file: {relative_path}",
    )
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ProvenanceContractError(
            f"committed role {role} escapes repository root: {relative_path}"
        ) from exc
    return resolved


def _committed_blob(
    repo_root: Path,
    source_commit: str,
    relative_path: str,
    *,
    role: str,
) -> tuple[str, bytes]:
    tree_output = _git(
        repo_root,
        [
            "ls-tree",
            "-z",
            "--full-tree",
            source_commit,
            "--",
            f":(literal){relative_path}",
        ],
        context=(
            f"cannot inspect committed role {role} at "
            f"{source_commit}:{relative_path}"
        ),
    )
    entries = [entry for entry in tree_output.split(b"\0") if entry]
    _require(
        len(entries) == 1,
        f"committed role {role} is absent or ambiguous at "
        f"{source_commit}:{relative_path}",
    )
    try:
        metadata, encoded_path = entries[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        listed_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProvenanceContractError(
            f"invalid Git tree record for committed role {role}"
        ) from exc
    _require(
        listed_path == relative_path,
        f"Git path mismatch for committed role {role}: {listed_path!r}",
    )
    _require(
        mode in _REGULAR_GIT_MODES and object_type == "blob",
        f"committed role {role} is not a regular Git blob: "
        f"mode={mode}, type={object_type}",
    )
    blob = _git(
        repo_root,
        ["cat-file", "blob", object_id],
        context=f"cannot read Git blob for committed role {role}",
    )
    return object_id, blob


def _validate_committed_record_shape(
    record: object,
    *,
    index: int,
) -> tuple[str, str, str]:
    _require(
        isinstance(record, Mapping),
        f"committed_files[{index}] must be a mapping",
    )
    _exact_keys(
        record,
        expected={"role", "repo_relative_path", "file_sha256"},
        name=f"committed_files[{index}]",
    )
    role = record["role"]
    relative_path = record["repo_relative_path"]
    claimed_sha256 = record["file_sha256"]
    _require(
        isinstance(role, str) and bool(role),
        f"committed_files[{index}] role is empty",
    )
    _require(
        isinstance(relative_path, str) and bool(relative_path),
        f"committed role {role} repo_relative_path is empty",
    )
    _require(
        isinstance(claimed_sha256, str)
        and _SHA256_RE.fullmatch(claimed_sha256) is not None,
        f"committed role {role} file_sha256 is invalid",
    )
    return role, relative_path, claimed_sha256


def _validate_external_record_shape(
    record: object,
    *,
    index: int,
) -> tuple[str, str, str, int]:
    _require(
        isinstance(record, Mapping),
        f"external_files[{index}] must be a mapping",
    )
    _exact_keys(
        record,
        expected={"role", "absolute_path", "file_sha256", "size_bytes"},
        name=f"external_files[{index}]",
    )
    role = record["role"]
    absolute_path = record["absolute_path"]
    claimed_sha256 = record["file_sha256"]
    claimed_size = record["size_bytes"]
    _require(
        isinstance(role, str) and bool(role),
        f"external_files[{index}] role is empty",
    )
    _require(
        isinstance(absolute_path, str)
        and bool(absolute_path)
        and Path(absolute_path).is_absolute(),
        f"external role {role} absolute_path must be absolute",
    )
    _require(
        isinstance(claimed_sha256, str)
        and _SHA256_RE.fullmatch(claimed_sha256) is not None,
        f"external role {role} file_sha256 is invalid",
    )
    _require(
        isinstance(claimed_size, int)
        and not isinstance(claimed_size, bool)
        and claimed_size >= 0,
        f"external role {role} size_bytes is invalid",
    )
    return role, absolute_path, claimed_sha256, claimed_size


def _validate_checkpoint_provenance_load_receipt(
    receipt: object,
    *,
    source_commit: str,
    checkpoint_shape: tuple[str, str, str, int],
) -> dict[str, Any]:
    """Cross-bind the one-shot runtime receipt without reopening its file.

    The authorization module has already consumed and capability-checked the
    private runtime state before this mapping reaches provenance validation.
    This boundary accepts only the exact normalized statement it returned and
    binds every checkpoint claim back to the bundle record.
    """

    _require(
        isinstance(receipt, Mapping),
        "checkpoint provenance requires a validated load receipt",
    )
    _exact_keys(
        receipt,
        expected=_CHECKPOINT_PROVENANCE_LOAD_RECEIPT_FIELDS,
        name="checkpoint provenance load receipt",
    )
    role, absolute_path, claimed_sha256, claimed_size = checkpoint_shape
    _require(
        receipt["role"] == role == "checkpoint",
        "checkpoint provenance load receipt role mismatch",
    )
    _require(
        type(receipt["absolute_path"]) is str
        and receipt["absolute_path"] == absolute_path,
        "checkpoint provenance load receipt path mismatch",
    )
    _require(
        type(receipt["file_sha256"]) is str
        and receipt["file_sha256"] == claimed_sha256,
        "checkpoint provenance load receipt SHA-256 mismatch",
    )
    _require(
        type(receipt["size_bytes"]) is int
        and receipt["size_bytes"] == claimed_size,
        "checkpoint provenance load receipt size mismatch",
    )
    _require(
        type(receipt["task_id"]) is int
        and receipt["task_id"] in {20, 55, 80},
        "checkpoint provenance load receipt task_id is invalid",
    )
    _require(
        type(receipt["fixture_id"]) is str and bool(receipt["fixture_id"]),
        "checkpoint provenance load receipt fixture_id is invalid",
    )
    _require(
        type(receipt["source_commit"]) is str
        and receipt["source_commit"] == source_commit,
        "checkpoint provenance load receipt source_commit mismatch",
    )
    _require(
        receipt["same_open_file_descriptor_hash_and_load"] is True,
        "checkpoint provenance load receipt lacks same-FD hash/load proof",
    )
    _require(
        receipt["weights_only"] is True,
        "checkpoint provenance load receipt lacks weights-only proof",
    )
    return {
        "role": "checkpoint",
        "absolute_path": receipt["absolute_path"],
        "file_sha256": receipt["file_sha256"],
        "size_bytes": receipt["size_bytes"],
    }


def _validate_loaded_source_record(
    record: object,
    *,
    role: str,
    expected_path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    _require(
        isinstance(record, Mapping),
        f"loaded source {role} must be a mapping",
    )
    _exact_keys(
        record,
        expected={"absolute_path", "sha256"},
        name=f"loaded source {role}",
    )
    path_value = record["absolute_path"]
    digest = record["sha256"]
    _require(
        isinstance(path_value, str) and Path(path_value).is_absolute(),
        f"loaded source {role} absolute_path must be absolute",
    )
    path = Path(path_value)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProvenanceContractError(
            f"loaded source {role} path does not exist: {path_value}"
        ) from exc
    _require(
        not stat.S_ISLNK(metadata.st_mode),
        f"loaded source {role} must not be a symlink",
    )
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"loaded source {role} must be a regular file",
    )
    _require(
        resolved == expected_path,
        f"loaded source {role} path mismatch: "
        f"expected {expected_path}, got {resolved}",
    )
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        f"loaded source {role} sha256 is invalid",
    )
    _require(
        digest == expected_sha256,
        f"loaded source {role} digest differs from committed blob",
    )
    return {"absolute_path": str(resolved), "sha256": digest}


def _validate_evaluation_provenance_for_repo(
    provenance: Mapping[str, Any],
    *,
    case_kind: str,
    fit_from_pairs: bool,
    loaded_source_digests: Mapping[str, Mapping[str, str]],
    repo_root: Path,
    provenance_loaded_source_digest: Mapping[str, str],
    checkpoint_provenance_load_receipt: Mapping[str, Any] | None = None,
    checkpoint_profile: str = "fixture",
) -> dict[str, Any]:
    """Internal test seam; production callers must use the public wrapper."""

    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    _exact_keys(
        provenance,
        expected={
            "schema_version",
            "source_commit",
            "committed_files",
            "external_files",
        },
        name="provenance",
    )
    _require(
        provenance["schema_version"] == PROVENANCE_SCHEMA_VERSION,
        "unsupported provenance schema_version",
    )
    root = _validated_repository_root(repo_root)
    source_commit = _validate_source_commit(root, provenance["source_commit"])
    required_committed, required_external = required_role_profile(
        case_kind,
        fit_from_pairs=fit_from_pairs,
        checkpoint_profile=checkpoint_profile,
    )

    committed_records = provenance["committed_files"]
    external_records = provenance["external_files"]
    _require(isinstance(committed_records, list), "committed_files must be a list")
    _require(isinstance(external_records, list), "external_files must be a list")

    committed_shapes = [
        _validate_committed_record_shape(record, index=index)
        for index, record in enumerate(committed_records)
    ]
    committed_roles = [shape[0] for shape in committed_shapes]
    _require(
        len(committed_roles) == len(set(committed_roles)),
        "duplicate committed file role",
    )
    actual_committed = set(committed_roles)
    _require(
        actual_committed == set(required_committed),
        "committed role profile mismatch: "
        f"missing={sorted(required_committed - actual_committed)}, "
        f"unknown={sorted(actual_committed - required_committed)}",
    )
    committed_path_by_role = {
        role: relative_path
        for role, relative_path, _claimed_sha256 in committed_shapes
    }
    planted_profile = None
    planted_profile_paths: Mapping[str, str] = {}
    if "oracle_harness_source" in committed_path_by_role:
        planted_profile = _resolve_planted_profile(
            case_kind=case_kind,
            committed_paths_by_role=committed_path_by_role,
        )
        planted_profile_paths = PLANTED_PROFILE_PATHS[planted_profile]

    external_shapes = [
        _validate_external_record_shape(record, index=index)
        for index, record in enumerate(external_records)
    ]
    external_roles = [shape[0] for shape in external_shapes]
    _require(
        len(external_roles) == len(set(external_roles)),
        "duplicate external file role",
    )
    actual_external = set(external_roles)
    _require(
        actual_external == set(required_external),
        "external role profile mismatch: "
        f"missing={sorted(required_external - actual_external)}, "
        f"unknown={sorted(actual_external - required_external)}",
    )
    _require(
        set(committed_roles).isdisjoint(external_roles),
        "a role appears in both committed_files and external_files",
    )
    if case_kind == "checkpoint":
        checkpoint_shape = next(
            shape for shape in external_shapes if shape[0] == "checkpoint"
        )
        normalized_checkpoint_receipt = (
            _validate_checkpoint_provenance_load_receipt(
                checkpoint_provenance_load_receipt,
                source_commit=source_commit,
                checkpoint_shape=checkpoint_shape,
            )
        )
    else:
        _require(
            checkpoint_provenance_load_receipt is None,
            "checkpoint provenance load receipt is forbidden for "
            f"{case_kind} cases",
        )
        normalized_checkpoint_receipt = None

    committed_paths: dict[str, Path] = {}
    for role, relative_path, _claimed_sha256 in committed_shapes:
        committed_paths[role] = _safe_committed_path(
            root,
            relative_path,
            role=role,
        )
        if role in {"oracle_harness_source", "oracle_case_manifest"}:
            expected_path = planted_profile_paths[role]
        else:
            expected_path = CORE_ROLE_PATHS.get(role) or FIXED_ROLE_PATHS.get(
                role
            ) or FRESH_CHECKPOINT_ROLE_PATHS.get(role)
        if expected_path is not None:
            _require(
                relative_path == expected_path,
                f"fixed role {role} must use {expected_path}, "
                f"not {relative_path}",
            )
        if role == "planted_case_definition":
            expected_prefix = planted_profile_paths[
                "planted_case_definition_prefix"
            ]
        else:
            expected_prefix = ROLE_PATH_PREFIXES.get(role)
        if expected_prefix is not None:
            _require(
                relative_path.startswith(expected_prefix)
                and relative_path.endswith(".json")
                and len(relative_path) > len(expected_prefix) + len(".json"),
                f"role {role} must use one JSON file below {expected_prefix}",
            )
    resolved_committed_paths = list(committed_paths.values())
    _require(
        len(resolved_committed_paths) == len(set(resolved_committed_paths)),
        "duplicate committed file path",
    )

    normalized_committed: list[dict[str, Any]] = []
    committed_sha256_by_role: dict[str, str] = {}
    for role, relative_path, claimed_sha256 in committed_shapes:
        current_path = committed_paths[role]
        git_blob_oid, blob_bytes = _committed_blob(
            root,
            source_commit,
            relative_path,
            role=role,
        )
        current_bytes = current_path.read_bytes()
        blob_sha256 = _sha256_bytes(blob_bytes)
        current_sha256 = _sha256_bytes(current_bytes)
        _require(
            current_bytes == blob_bytes,
            f"committed role {role} working bytes differ from Git blob",
        )
        _require(
            current_sha256 == blob_sha256 == claimed_sha256,
            f"committed role {role} SHA-256 mismatch",
        )
        committed_sha256_by_role[role] = current_sha256
        normalized_committed.append(
            {
                "role": role,
                "repo_relative_path": relative_path,
                "absolute_path": str(current_path),
                "file_sha256": current_sha256,
                "git_blob_oid": git_blob_oid,
            }
        )

    if planted_profile == "task20_collapse_v2":
        _validate_task20_v2_case_manifest_membership(
            repo_root=root,
            source_commit=source_commit,
            manifest_path=committed_paths["oracle_case_manifest"],
            selected_case_relative_path=committed_path_by_role[
                "planted_case_definition"
            ],
            selected_case_sha256=committed_sha256_by_role[
                "planted_case_definition"
            ],
        )

    _require(
        isinstance(loaded_source_digests, Mapping),
        "loaded_source_digests must be a mapping",
    )
    actual_loaded_roles = set(loaded_source_digests)
    required_loaded_roles = required_loaded_source_roles(case_kind)
    _require(
        actual_loaded_roles == set(required_loaded_roles),
        "loaded source role profile mismatch: "
        f"missing={sorted(required_loaded_roles - actual_loaded_roles)}, "
        f"unknown={sorted(actual_loaded_roles - required_loaded_roles)}",
    )
    normalized_loaded = {
        role: _validate_loaded_source_record(
            loaded_source_digests[role],
            role=role,
            expected_path=committed_paths[role],
            expected_sha256=committed_sha256_by_role[role],
        )
        for role in sorted(required_loaded_roles)
    }
    normalized_provenance_loaded = _validate_loaded_source_record(
        provenance_loaded_source_digest,
        role="provenance_source",
        expected_path=committed_paths["provenance_source"],
        expected_sha256=committed_sha256_by_role["provenance_source"],
    )

    normalized_external: list[dict[str, Any]] = []
    resolved_external_paths: list[Path] = []
    for role, absolute_path, claimed_sha256, claimed_size in external_shapes:
        if role == "checkpoint":
            _require(
                normalized_checkpoint_receipt is not None,
                "checkpoint provenance load receipt is missing",
            )
            # The checkpoint was hashed and deserialized through one already
            # validated file descriptor.  Reopening, resolving, or even
            # lstat-ing it here would both violate that one-open contract and
            # create a path-replacement race.
            checkpoint_path = Path(
                normalized_checkpoint_receipt["absolute_path"]
            )
            resolved_external_paths.append(checkpoint_path)
            normalized_external.append(
                dict(normalized_checkpoint_receipt)
            )
            continue
        path = Path(absolute_path)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ProvenanceContractError(
                f"external role {role} cannot be inspected: {absolute_path}"
            ) from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"external role {role} must not be a symlink",
        )
        _require(
            stat.S_ISREG(metadata.st_mode),
            f"external role {role} must be a regular file",
        )
        resolved = path.resolve(strict=True)
        actual_sha256, actual_size = _sha256_file(resolved)
        _require(
            actual_size == claimed_size,
            f"external role {role} size mismatch: "
            f"claimed {claimed_size}, actual {actual_size}",
        )
        _require(
            actual_sha256 == claimed_sha256,
            f"external role {role} SHA-256 mismatch",
        )
        resolved_external_paths.append(resolved)
        normalized_external.append(
            {
                "role": role,
                "absolute_path": str(resolved),
                "file_sha256": actual_sha256,
                "size_bytes": actual_size,
            }
        )
    _require(
        len(resolved_external_paths) == len(set(resolved_external_paths)),
        "duplicate external file path",
    )
    _require(
        set(resolved_committed_paths).isdisjoint(resolved_external_paths),
        "a file path appears in both committed_files and external_files",
    )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "source_commit_verified": True,
        "repository_root": str(root),
        "case_kind": case_kind,
        "fit_from_pairs": fit_from_pairs,
        "committed_files": sorted(
            normalized_committed,
            key=lambda record: record["role"],
        ),
        "external_files": sorted(
            normalized_external,
            key=lambda record: record["role"],
        ),
        "loaded_sources": normalized_loaded,
        "provenance_loaded_source": normalized_provenance_loaded,
    }


def validate_evaluation_provenance(
    provenance: Mapping[str, Any],
    *,
    case_kind: str,
    fit_from_pairs: bool,
    loaded_source_digests: Mapping[str, Mapping[str, str]],
    checkpoint_provenance_load_receipt: Mapping[str, Any] | None = None,
    checkpoint_profile: str = "fixture",
) -> dict[str, Any]:
    """Validate production provenance against this repository.

    There is intentionally no public ``repo_root`` argument.  Production code
    cannot redirect provenance checks to a caller-selected Git repository.
    """

    repo_root = Path(__file__).resolve().parents[1]
    return _validate_evaluation_provenance_for_repo(
        provenance,
        case_kind=case_kind,
        fit_from_pairs=fit_from_pairs,
        loaded_source_digests=loaded_source_digests,
        repo_root=repo_root,
        provenance_loaded_source_digest=_LOADED_PROVENANCE_SOURCE_DIGEST,
        checkpoint_provenance_load_receipt=(
            checkpoint_provenance_load_receipt
        ),
        checkpoint_profile=checkpoint_profile,
    )


_LOADED_PROVENANCE_SOURCE_DIGEST = capture_loaded_source_digest(
    "provenance_source",
    Path(__file__).absolute(),
)
__representation_import_sha256__ = _LOADED_PROVENANCE_SOURCE_DIGEST["sha256"]


__all__ = [
    "CORE_ROLE_PATHS",
    "FRESH_CHECKPOINT_EXTERNAL_ROLES",
    "FRESH_CHECKPOINT_ROLE_PATHS",
    "LOADED_SOURCE_ROLES",
    "PLANTED_PROFILE_PATHS",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceContractError",
    "__representation_import_sha256__",
    "capture_loaded_source_digest",
    "required_loaded_source_roles",
    "required_role_profile",
    "validate_evaluation_provenance",
]
