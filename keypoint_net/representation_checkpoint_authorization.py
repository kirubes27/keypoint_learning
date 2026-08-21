"""Fail-closed authorization for read-only saved-checkpoint replay.

This module has two deliberately separate capabilities:

* :class:`CheckpointRuntimeAuthorization` permits one registered checkpoint to
  be opened by the CPU-only runtime.  It is minted only after the committed
  replay registry, opaque hash preflight, reviewed runtime-source manifest, and
  execution-review records agree.
* a process-local, one-shot evaluator context binds the exact evaluation-bundle
  content hash to the checkpoint that the authorized runtime actually loaded.
  A committed manifest by itself therefore cannot authorize fabricated logits.

Neither registry validation nor evaluator authorization opens a checkpoint,
imports a model, reads an image, or reads a mask.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__representation_import_sha256__ = hashlib.sha256(
    Path(__file__).absolute().read_bytes()
).hexdigest()


from keypoint_net import representation_checkpoint_replay as replay_registry


CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v3.json"
)
CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "CHECKPOINT_HASH_PREFLIGHT_RESULT_2026-07-27.json"
)
CHECKPOINT_EXECUTION_AUTHORIZATION_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "CHECKPOINT_EXECUTION_AUTHORIZATION_v3.json"
)
FABLE_EXECUTION_REVIEW_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "FABLE_5_HIGH_CROSS_BACKEND_TOLERANCE_V3_EXECUTION_REVIEW_2026-07-29.md"
)
TASK20_CHECKPOINT_REPLAY_RESULT_V1_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/results/"
    "TASK20_CHECKPOINT_REPLAY_RESULT_v1.json"
)
TASK20_CHECKPOINT_REPLAY_RESULT_V3_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/results/"
    "TASK20_CHECKPOINT_REPLAY_RESULT_v3.json"
)
TASK20_V2_SEMANTIC_LOCK_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "TASK20_STRUCTURAL_COLLAPSE_V2_SEMANTIC_LOCK.md"
)
DECISION_SYNTHESIS_V2_8_AMENDMENT_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "DECISION_SYNTHESIS_v2_8_AMENDMENT_2026-07-29.md"
)
TASK55_CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/"
    "TASK55_COORDINATE_BACKEND_BRIDGE_SEMANTIC_LOCK.md"
)
TASK20_V2_ORACLE_HARNESS_REPO_RELATIVE_PATH = (
    "keypoint_net/diagnostics/run_task20_collapse_v2_oracles.py"
)
TASK20_V2_ORACLE_MANIFEST_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-29/representation_oracle_cases_v2/"
    "TASK20_COLLAPSE_ORACLE_MANIFEST_v2.json"
)
TASK20_V2_ORACLE_CASE_REPO_RELATIVE_PATHS = (
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
TASK20_V2_REVIEWED_FILE_PATHS = (
    TASK20_V2_SEMANTIC_LOCK_REPO_RELATIVE_PATH,
    DECISION_SYNTHESIS_V2_8_AMENDMENT_REPO_RELATIVE_PATH,
    TASK55_CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_REPO_RELATIVE_PATH,
    TASK20_V2_ORACLE_HARNESS_REPO_RELATIVE_PATH,
    TASK20_V2_ORACLE_MANIFEST_REPO_RELATIVE_PATH,
    *TASK20_V2_ORACLE_CASE_REPO_RELATIVE_PATHS,
    CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH,
    CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH,
    TASK20_CHECKPOINT_REPLAY_RESULT_V1_REPO_RELATIVE_PATH,
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

RUNTIME_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "representation_checkpoint_runtime_sources.v3"
)
RUNTIME_SOURCE_MANIFEST_ARTIFACT_TYPE = (
    "representation_checkpoint_runtime_sources"
)
RUNTIME_ENTRYPOINT = (
    "keypoint_net.representation_checkpoint_runtime."
    "run_authorized_checkpoint_replay"
)
V2_COLLAPSE_FIELD = (
    "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
)
TASK20_V1_RESULT_FILE_SHA256 = (
    "e44ec8b839d6b39377a8acf8b5b2997334ad3c518530675cadcc322e696d2675"
)
TASK20_V1_RESULT_CONTENT_HASH_SHA256 = (
    "5087f5538f728647774fc9e88b4b55ac384c162ebcd3f44136e555bb258258fb"
)
TASK20_V1_RESULT_SOURCE_COMMIT = (
    "426d1dbe94a655e6a90b4441b8e368be7338a4ae"
)
FABLE_REVIEW_BINDING_SCHEMA = (
    "cross_backend_tolerance_v3_fable_execution_review_binding.v1"
)
TASK20_V2_ORACLE_RESULT_PREFIX = (
    "docs/decisions/2026-07-29/representation_oracle_results_v2/"
)
TASK20_V2_ORACLE_CASE_IDS = tuple(
    Path(path).stem for path in TASK20_V2_ORACLE_CASE_REPO_RELATIVE_PATHS
)
TASK20_V2_INDEPENDENT_OUTCOME_MATRIX_SHA256 = (
    "15198aa36ec42fc032b2978a6401672df15468323fe52c0f094a1c3545be5477"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VERIFIED_RUNTIME_CAPABILITY = object()
_VERIFIED_LOADED_CHECKPOINT_CAPABILITY = object()
_VERIFIED_EVALUATOR_CAPABILITY = object()

_RUN_ROOT = (
    "/Users/kirubeso.r/Documents/PhD/artifacts/experiments/"
    "2026-06-full360-shared-affine/cluster-results/keypoint_net/runs_hammer_full360_shared"
)
_DATASET_ROOT = (
    "/Users/kirubeso.r/Documents/PhD/keypoint_preoperator_gates/"
    "_tdw_world_z_roll_base_panel_512_v2"
)

_TASK_BINDINGS = MappingProxyType(
    {
        20: {
            "fixture_id": "saved_task20_seed42_collapsed_negative_control",
            "role": "collapsed_negative_control",
            "run_directory": (
                f"{_RUN_ROOT}/phase_a_engineers_hammer_vray_"
                "20260606_050434_151501_seed42_pid2116969"
            ),
            "checkpoint_sha256": (
                "96433168767659ae9144a35a4f7889c3226b65a7a0c5341197d48232a66fe622"
            ),
            "checkpoint_size_bytes": 2994210,
            "config_sha256": (
                "60db767a5a77da28d001ef0ceb4134d063b749053255955629d6780b278d8f71"
            ),
            "config_size_bytes": 1020,
            "history_sha256": (
                "4c5f786f6e531fe7f24b36d7cda61419371f4be8f01dc622377931d41458a199"
            ),
            "history_size_bytes": 63319,
            "learn_inverse_operator": True,
        },
        55: {
            "fixture_id": "saved_task55_seed42_clean_baseline",
            "role": "clean_baseline_replay_only",
            "run_directory": (
                f"{_RUN_ROOT}/phase_a_engineers_hammer_vray_"
                "20260606_131311_378814_seed42_pid606096"
            ),
            "checkpoint_sha256": (
                "942a32082b6bfe83526253cc2dd39e49792b8260d12fc1361a11bae812992418"
            ),
            "checkpoint_size_bytes": 2991906,
            "config_sha256": (
                "9c9bbeb64ba7c79593724700fe6ece92196303d4eb50c05cfaefb0099d5704f8"
            ),
            "config_size_bytes": 1019,
            "history_sha256": (
                "c1e2b1fc2cb369fa1d28d6fcef1a682e58a09fbcdeaf922dd7a945b52d4d48e6"
            ),
            "history_size_bytes": 56044,
            "learn_inverse_operator": False,
        },
        80: {
            "fixture_id": "saved_task80_seed42_assisted_baseline",
            "role": "assisted_baseline_replay_only",
            "run_directory": (
                f"{_RUN_ROOT}/phase_a_engineers_hammer_vray_"
                "20260606_151123_941396_seed42_pid3289780"
            ),
            "checkpoint_sha256": (
                "ccb613c87788a229929b1f6ead002d626819e970237d0e0ad5ca75c9318004f3"
            ),
            "checkpoint_size_bytes": 2994210,
            "config_sha256": (
                "e1fea4fc99c23637f47cb73f904b0f45c330425e15e327ce07a34d59212af9d7"
            ),
            "config_size_bytes": 1020,
            "history_sha256": (
                "9371a65d949cdb1388b8a511142ecc6af1c96ef49d3b7a772a5591f206d19d65"
            ),
            "history_size_bytes": 63233,
            "learn_inverse_operator": True,
        },
    }
)

_EXPECTED_RUNTIME_SOURCES = (
    (
        "authorization_source",
        "keypoint_net/representation_checkpoint_authorization.py",
    ),
    (
        "runtime_source",
        "keypoint_net/representation_checkpoint_runtime.py",
    ),
    (
        "replay_registry_source",
        "keypoint_net/representation_checkpoint_replay.py",
    ),
    (
        "historical_comparison_source",
        "keypoint_net/representation_replay_comparison.py",
    ),
    ("model_source", "keypoint_net/model.py"),
    ("evaluator_source", "keypoint_net/eval_representation.py"),
    (
        "provenance_source",
        "keypoint_net/representation_evaluation_provenance.py",
    ),
    ("array_codec_source", "keypoint_net/representation_array_codec.py"),
    (
        "manifest_builder_source",
        "keypoint_net/diagnostics/build_checkpoint_replay_manifests.py",
    ),
    (
        "corpus_inventory_source",
        "keypoint_net/representation_corpus_inventory.py",
    ),
    (
        "split_validator_source",
        "keypoint_net/representation_splits.py",
    ),
)

_EXPECTED_PREPROCESSING = {
    "image_mode": "RGB",
    "input_size_hw": [512, 512],
    "resize": {
        "mode": "exact_hw",
        "size_hw": [512, 512],
        "interpolation": "bilinear",
        "antialias": True,
    },
    "tensor_conversion": {
        "source_dtype": "uint8",
        "output_dtype": "float32",
        "scale": "divide_by_255",
        "channel_order": "CHW",
    },
    "normalization": {
        "mean_rgb": [0.485, 0.456, 0.406],
        "std_rgb": [0.229, 0.224, 0.225],
    },
    "augmentation": "none",
    "frame_order": "ascending_frame_index_0_through_179",
    "heatmap_logits_preserved": True,
    "coordinate_grid": "normalized_minus1_plus1_endpoint_aligned_xy",
}

_EXPECTED_RUNTIME_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.10.19",
    "packages": [
        {"name": "numpy", "version": "2.2.6"},
        {"name": "pillow", "version": "12.1.0"},
        {"name": "torch", "version": "2.10.0"},
    ],
    "torch_inference_device": "cpu",
    "deterministic_algorithms_required": True,
}

_EXPECTED_EXECUTION_BOUNDARY = {
    "cpu_only": True,
    "weights_only": True,
    "same_open_file_descriptor_hash_and_load": True,
    "strict_state_dict": True,
    "model_eval": True,
    "inference_mode": True,
    "task_ids": [20, 55, 80],
    "task55_and_80_require_immutable_task20_v1_audit": True,
    "task55_and_80_require_committed_task20_v3_stop_gate": True,
    "checkpoint_float32_cross_backend_coordinate_tolerance": 1e-4,
    "training_or_weight_update_authorized": False,
    "selection_use_authorized": False,
}

_EXPECTED_AUTHORIZATION_BOUNDARY = {
    "checkpoint_replay_authorized": True,
    "cpu_only": True,
    "weights_only": True,
    "same_open_file_descriptor_hash_and_load": True,
    "strict_state_dict": True,
    "model_eval": True,
    "inference_mode": True,
    "task55_and_80_require_task20_pass": True,
    "task55_and_80_require_immutable_task20_v1_audit": True,
    "checkpoint_float32_cross_backend_coordinate_tolerance": 1e-4,
    "scientific_quality_threshold_changed": False,
    "training_or_weight_update_authorized": False,
    "selection_use_authorized": False,
}


class CheckpointAuthorizationError(ValueError):
    """Raised when replay or evaluator authorization is not exact."""


@dataclass(frozen=True)
class CheckpointRuntimeAuthorization:
    """Private-capability receipt for one exact task/checkpoint/config triplet."""

    task_id: int
    fixture_id: str
    role: str
    runtime_source_commit: str
    repository_root: str
    checkpoint_absolute_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    config_absolute_path: str
    config_sha256: str
    config_size_bytes: int
    history_absolute_path: str
    history_sha256: str
    history_size_bytes: int
    dataset_root: str
    reviewed_candidate_commit: str
    runtime_source_manifest_file_sha256: str
    runtime_source_manifest_content_hash_sha256: str
    checkpoint_hash_preflight_file_sha256: str
    checkpoint_execution_authorization_file_sha256: str
    fable_execution_review_file_sha256: str
    expected_learn_inverse_operator: bool
    task20_gate_result_file_sha256: str | None = None
    task20_gate_result_content_hash_sha256: str | None = None
    task20_gate_source_commit: str | None = None
    task20_v1_audit_file_sha256: str | None = None
    task20_v1_audit_content_hash_sha256: str | None = None
    expected_num_action_classes: int = 0
    expected_heatmap_res: int = 64
    expected_in_channels: int = 3
    cpu_only: bool = True
    training_or_weight_update_authorized: bool = False
    selection_use_authorized: bool = False
    _capability: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _PendingEvaluatorAuthorization:
    bundle_content_sha256: str
    source_commit: str
    task_id: int
    fixture_id: str
    checkpoint_sha256: str
    runtime_source_manifest_file_sha256: str
    checkpoint_hash_preflight_file_sha256: str
    points_content_sha256: str
    logits_content_sha256: str
    model_state_sha256: str
    loaded_checkpoint_receipt: "_LoadedCheckpointReceipt"
    _capability: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _LoadedCheckpointReceipt:
    source_commit: str
    task_id: int
    fixture_id: str
    checkpoint_absolute_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    _capability: object = field(default=None, repr=False, compare=False)


_PENDING_EVALUATOR_AUTHORIZATION: contextvars.ContextVar[
    _PendingEvaluatorAuthorization | None
] = contextvars.ContextVar(
    "representation_pending_checkpoint_evaluator_authorization",
    default=None,
)
_REGISTERED_LOADED_CHECKPOINT: contextvars.ContextVar[
    _LoadedCheckpointReceipt | None
] = contextvars.ContextVar(
    "representation_registered_loaded_checkpoint",
    default=None,
)
_PENDING_PROVENANCE_LOADED_CHECKPOINT: contextvars.ContextVar[
    _LoadedCheckpointReceipt | None
] = contextvars.ContextVar(
    "representation_pending_provenance_loaded_checkpoint",
    default=None,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointAuthorizationError(message)


def _exact_value(actual: Any, expected: Any, name: str) -> None:
    _require(
        type(actual) is type(expected),
        f"{name} type differs: expected {type(expected).__name__}, "
        f"got {type(actual).__name__}",
    )
    if isinstance(expected, dict):
        _require(
            set(actual) == set(expected),
            f"{name} keys differ: missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}",
        )
        for key, value in expected.items():
            _exact_value(actual[key], value, f"{name}.{key}")
        return
    if isinstance(expected, list):
        _require(len(actual) == len(expected), f"{name} length differs")
        for index, (item, value) in enumerate(zip(actual, expected)):
            _exact_value(item, value, f"{name}[{index}]")
        return
    if isinstance(expected, float):
        _require(math.isfinite(actual), f"{name} must be finite")
    _require(actual == expected, f"{name} differs from the frozen value")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointAuthorizationError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise CheckpointAuthorizationError(
        f"non-standard/non-finite JSON constant: {value}"
    )


def _strict_json(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointAuthorizationError(
            f"{name} is not strict UTF-8 JSON: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{name} top level must be an object")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointAuthorizationError(
            f"value cannot be encoded as canonical JSON: {exc}"
        ) from exc


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


def _git(repo_root: Path, arguments: Sequence[str], *, name: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repo_root),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise CheckpointAuthorizationError(
            f"{name}{': ' + detail if detail else ''}"
        ) from exc
    return result.stdout


def _regular_file_bytes(path: Path, *, name: str) -> tuple[Path, bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointAuthorizationError(
            f"{name} cannot be inspected: {path}"
        ) from exc
    _require(not stat.S_ISLNK(metadata.st_mode), f"{name} must not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{name} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
        data = resolved.read_bytes()
    except OSError as exc:
        raise CheckpointAuthorizationError(
            f"{name} cannot be read: {path}"
        ) from exc
    return resolved, data


def _committed_working_file(
    *,
    repo_root: Path,
    source_commit: str,
    repo_relative_path: str,
    name: str,
) -> tuple[Path, bytes]:
    _require(
        _COMMIT_RE.fullmatch(source_commit) is not None,
        "source_commit must be full lowercase 40-hex",
    )
    relative = Path(repo_relative_path)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts,
        f"{name} path is unsafe",
    )
    working_path, data = _regular_file_bytes(
        repo_root / relative,
        name=name,
    )
    try:
        working_path.relative_to(repo_root)
    except ValueError as exc:
        raise CheckpointAuthorizationError(f"{name} escapes repository") from exc
    committed = _git(
        repo_root,
        ["show", f"{source_commit}:{relative.as_posix()}"],
        name=f"cannot read committed {name}",
    )
    _require(committed == data, f"{name} differs from source_commit")
    return working_path, data


def _require_file_unchanged_since_review(
    *,
    repo_root: Path,
    reviewed_candidate_commit: str,
    repo_relative_path: str,
    current_data: bytes,
    name: str,
) -> None:
    _require(
        _COMMIT_RE.fullmatch(reviewed_candidate_commit) is not None,
        "reviewed_candidate_commit must be full lowercase 40-hex",
    )
    reviewed_data = _git(
        repo_root,
        ["show", f"{reviewed_candidate_commit}:{repo_relative_path}"],
        name=f"cannot read reviewed {name}",
    )
    _require(
        reviewed_data == current_data,
        f"{name} changed after Fable review",
    )


def _validate_fable_review_binding(
    *,
    review_data: bytes,
    reviewed_candidate_commit: str,
    runtime_source_manifest_file_sha256: str,
    runtime_source_manifest_content_hash_sha256: str,
    checkpoint_hash_preflight_file_sha256: str,
    semantic_lock_file_sha256: str,
    decision_amendment_file_sha256: str,
    cross_backend_tolerance_semantic_lock_file_sha256: str,
    task20_v2_oracle_harness_file_sha256: str,
    task20_v2_oracle_manifest_file_sha256: str,
    planted_v2_reviewed_fileset_sha256: str,
) -> None:
    """Require one exact, machine-checkable binding block in Fable's report."""

    try:
        review_text = review_data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CheckpointAuthorizationError(
            "Fable review is not valid UTF-8"
        ) from exc
    expected_markers = (
        ("FABLE_REVIEW_BINDING_SCHEMA", FABLE_REVIEW_BINDING_SCHEMA),
        ("REVIEWED_CANDIDATE_COMMIT", reviewed_candidate_commit),
        (
            "RUNTIME_SOURCE_MANIFEST_FILE_SHA256",
            runtime_source_manifest_file_sha256,
        ),
        (
            "RUNTIME_SOURCE_MANIFEST_CONTENT_SHA256",
            runtime_source_manifest_content_hash_sha256,
        ),
        (
            "CHECKPOINT_HASH_PREFLIGHT_FILE_SHA256",
            checkpoint_hash_preflight_file_sha256,
        ),
        ("TASK20_V2_SEMANTIC_LOCK_FILE_SHA256", semantic_lock_file_sha256),
        (
            "DECISION_SYNTHESIS_V2_8_AMENDMENT_FILE_SHA256",
            decision_amendment_file_sha256,
        ),
        (
            "CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_FILE_SHA256",
            cross_backend_tolerance_semantic_lock_file_sha256,
        ),
        (
            "TASK20_V2_ORACLE_HARNESS_FILE_SHA256",
            task20_v2_oracle_harness_file_sha256,
        ),
        (
            "TASK20_V2_ORACLE_MANIFEST_FILE_SHA256",
            task20_v2_oracle_manifest_file_sha256,
        ),
        (
            "PLANTED_V2_REVIEWED_FILESET_SHA256",
            planted_v2_reviewed_fileset_sha256,
        ),
        ("IMMUTABLE_TASK20_V1_FILE_SHA256", TASK20_V1_RESULT_FILE_SHA256),
        ("MODEL", "fable"),
        ("EFFORT", "high"),
        ("PERMISSION_MODE", "read_only"),
        ("SESSION_PERSISTENCE", "false"),
        ("VERDICT", "PASS_TO_RUN"),
        ("UNRESOLVED_P0_COUNT", "0"),
        ("UNRESOLVED_P1_COUNT", "0"),
    )
    lines = review_text.splitlines()
    for label, expected_value in expected_markers:
        prefix = f"{label}:"
        matching = [line for line in lines if line.startswith(prefix)]
        _require(
            len(matching) == 1,
            f"Fable review must contain exactly one {label} marker",
        )
        _exact_value(
            matching[0],
            f"{label}: {expected_value}",
            f"Fable review marker {label}",
        )


def _validated_reviewed_fileset(
    *,
    repo_root: Path,
    source_commit: str,
    reviewed_candidate_commit: str,
) -> tuple[str, dict[str, bytes]]:
    records: list[dict[str, str]] = []
    data_by_path: dict[str, bytes] = {}
    for repo_relative_path in TASK20_V2_REVIEWED_FILE_PATHS:
        _, data = _committed_working_file(
            repo_root=repo_root,
            source_commit=source_commit,
            repo_relative_path=repo_relative_path,
            name=f"reviewed file {repo_relative_path}",
        )
        _require_file_unchanged_since_review(
            repo_root=repo_root,
            reviewed_candidate_commit=reviewed_candidate_commit,
            repo_relative_path=repo_relative_path,
            current_data=data,
            name=f"reviewed file {repo_relative_path}",
        )
        records.append(
            {
                "repo_relative_path": repo_relative_path,
                "file_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        data_by_path[repo_relative_path] = data
    fileset_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            sorted(records, key=lambda record: record["repo_relative_path"])
        )
    ).hexdigest()
    return fileset_sha256, data_by_path


def _validate_runtime_source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    reviewed_candidate_commit: str | None = None,
) -> dict[str, Any]:
    path, data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH,
        name="checkpoint runtime source manifest",
    )
    document = _strict_json(data, name="checkpoint runtime source manifest")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "entrypoint",
        "sources",
        "preprocessing",
        "runtime_environment",
        "execution_boundary",
        "content_hash_sha256",
    }
    _require(
        set(document) == expected_keys,
        "checkpoint runtime source manifest keys differ",
    )
    _exact_value(
        document["schema_version"],
        RUNTIME_SOURCE_MANIFEST_SCHEMA_VERSION,
        "runtime manifest schema_version",
    )
    _exact_value(
        document["artifact_type"],
        RUNTIME_SOURCE_MANIFEST_ARTIFACT_TYPE,
        "runtime manifest artifact_type",
    )
    _exact_value(document["entrypoint"], RUNTIME_ENTRYPOINT, "runtime entrypoint")
    _exact_value(
        document["preprocessing"],
        _EXPECTED_PREPROCESSING,
        "runtime preprocessing",
    )
    _exact_value(
        document["runtime_environment"],
        _EXPECTED_RUNTIME_ENVIRONMENT,
        "runtime environment",
    )
    _exact_value(
        document["execution_boundary"],
        _EXPECTED_EXECUTION_BOUNDARY,
        "runtime execution boundary",
    )
    claimed_hash = document["content_hash_sha256"]
    _require(
        isinstance(claimed_hash, str)
        and _SHA256_RE.fullmatch(claimed_hash) is not None,
        "runtime manifest content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    computed_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    _require(claimed_hash == computed_hash, "runtime manifest content hash mismatch")

    source_records = document["sources"]
    _require(isinstance(source_records, list), "runtime sources must be a list")
    expected_records = [
        {
            "role": role,
            "repo_relative_path": source_path,
            "file_sha256": record.get("file_sha256")
            if isinstance(record, Mapping)
            else None,
            "import_time_digest_required": True,
        }
        for record, (role, source_path) in zip(
            source_records,
            _EXPECTED_RUNTIME_SOURCES,
        )
    ]
    _require(
        len(source_records) == len(_EXPECTED_RUNTIME_SOURCES),
        "runtime source count differs",
    )
    normalized_sources: dict[str, dict[str, str]] = {}
    for index, (record, expected) in enumerate(
        zip(source_records, expected_records)
    ):
        _require(isinstance(record, Mapping), f"runtime source {index} is invalid")
        _require(
            set(record)
            == {
                "role",
                "repo_relative_path",
                "file_sha256",
                "import_time_digest_required",
            },
            f"runtime source {index} keys differ",
        )
        _exact_value(record["role"], expected["role"], f"runtime source {index}.role")
        _exact_value(
            record["repo_relative_path"],
            expected["repo_relative_path"],
            f"runtime source {index}.repo_relative_path",
        )
        _exact_value(
            record["import_time_digest_required"],
            True,
            f"runtime source {index}.import_time_digest_required",
        )
        digest = record["file_sha256"]
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"runtime source {index} hash is invalid",
        )
        source_path, source_data = _committed_working_file(
            repo_root=repo_root,
            source_commit=source_commit,
            repo_relative_path=record["repo_relative_path"],
            name=f"runtime source {record['role']}",
        )
        _require(
            hashlib.sha256(source_data).hexdigest() == digest,
            f"runtime source {record['role']} hash mismatch",
        )
        if reviewed_candidate_commit is not None:
            _require_file_unchanged_since_review(
                repo_root=repo_root,
                reviewed_candidate_commit=reviewed_candidate_commit,
                repo_relative_path=str(record["repo_relative_path"]),
                current_data=source_data,
                name=f"runtime source {record['role']}",
            )
        normalized_sources[str(record["role"])] = {
            "absolute_path": str(source_path),
            "sha256": digest,
        }

    return {
        "absolute_path": str(path),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "content_hash_sha256": claimed_hash,
        "sources": normalized_sources,
    }


def _validate_actual_runtime_environment() -> None:
    _require(
        platform.python_implementation() == "CPython",
        "official checkpoint replay requires CPython",
    )
    _require(
        platform.python_version() == "3.10.19",
        "official checkpoint replay requires Python 3.10.19",
    )
    for package, version in (
        ("numpy", "2.2.6"),
        ("pillow", "12.1.0"),
        ("torch", "2.10.0"),
    ):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CheckpointAuthorizationError(
                f"official checkpoint replay requires {package}=={version}"
            ) from exc
        _require(
            actual == version,
            f"official checkpoint replay requires {package}=={version}, "
            f"found {actual}",
        )


def _validate_checkpoint_preflight(
    *,
    preflight: Mapping[str, Any],
    candidate: replay_registry.CheckpointCandidate,
) -> dict[str, str]:
    _require(isinstance(preflight, Mapping), "checkpoint preflight must be a mapping")
    required = {
        "source_commit",
        "registry_file_sha256",
        "registry_content_hash_sha256",
        "checkpoint_verification",
        "checkpoint_hash_preflight_passed",
        "checkpoint_load_authorized_by_this_receipt",
        "remaining_load_gate",
        "training_or_weight_update_authorized",
        "selection_use_authorized",
    }
    _require(set(preflight) == required, "checkpoint preflight keys differ")
    _require(
        isinstance(preflight["source_commit"], str)
        and _COMMIT_RE.fullmatch(preflight["source_commit"]) is not None,
        "checkpoint preflight source_commit is invalid",
    )
    _exact_value(
        preflight["checkpoint_hash_preflight_passed"],
        True,
        "checkpoint hash preflight verdict",
    )
    _exact_value(
        preflight["checkpoint_load_authorized_by_this_receipt"],
        False,
        "checkpoint preflight load boundary",
    )
    _exact_value(
        preflight["remaining_load_gate"],
        "separate_fable_reviewed_execution_commit_and_runtime_provenance",
        "checkpoint preflight remaining gate",
    )
    _exact_value(
        preflight["training_or_weight_update_authorized"],
        False,
        "checkpoint preflight training boundary",
    )
    _exact_value(
        preflight["selection_use_authorized"],
        False,
        "checkpoint preflight selection boundary",
    )
    records = preflight["checkpoint_verification"]
    _require(isinstance(records, list), "checkpoint verification must be a list")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("task_id") == candidate.task_id
    ]
    _require(len(matches) == 1, "preflight does not bind exactly one requested task")
    record = matches[0]
    expected = {
        "fixture_id": candidate.fixture_id,
        "task_id": candidate.task_id,
        "role": candidate.role,
        "absolute_path": candidate.absolute_path,
        "size_bytes": candidate.expected_size_bytes,
        "sha256": candidate.expected_sha256,
        "verification_status": "hashed_as_opaque_bytes_not_loaded",
    }
    _exact_value(dict(record), expected, "requested checkpoint preflight record")
    return {
        "checkpoint_sha256": candidate.expected_sha256,
        "preflight_source_commit": str(preflight["source_commit"]),
    }


def _binding_for_candidate(
    candidate: replay_registry.CheckpointCandidate,
) -> Mapping[str, Any]:
    _require(candidate.task_id in _TASK_BINDINGS, "task is not a frozen replay task")
    binding = _TASK_BINDINGS[candidate.task_id]
    _exact_value(candidate.fixture_id, binding["fixture_id"], "candidate fixture_id")
    _exact_value(candidate.role, binding["role"], "candidate role")
    _exact_value(
        candidate.absolute_path,
        f"{binding['run_directory']}/best_model.pt",
        "candidate checkpoint path",
    )
    _exact_value(
        candidate.expected_sha256,
        binding["checkpoint_sha256"],
        "candidate checkpoint hash",
    )
    _exact_value(
        candidate.expected_size_bytes,
        binding["checkpoint_size_bytes"],
        "candidate checkpoint size",
    )
    return binding


def _role_record(
    provenance: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any]:
    records = provenance.get("committed_files")
    _require(isinstance(records, list), "provenance committed_files must be a list")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("role") == role
    ]
    _require(len(matches) == 1, f"provenance must bind exactly one {role}")
    record = matches[0]
    _require(
        set(record) == {"role", "repo_relative_path", "file_sha256"},
        f"provenance role {role} record keys differ",
    )
    _require(
        isinstance(record["file_sha256"], str)
        and _SHA256_RE.fullmatch(record["file_sha256"]) is not None,
        f"provenance role {role} hash is invalid",
    )
    return record


def _validate_committed_task20_v2_planted_suite(
    *,
    repo_root: Path,
    source_commit: str,
    record: Mapping[str, Any],
    reviewed_candidate_commit: str,
    review_data: bytes,
    reviewed_fileset_sha256: str,
    reviewed_files: Mapping[str, bytes],
) -> dict[str, str]:
    _require(
        set(record)
        == {
            "repo_relative_path",
            "file_sha256",
            "content_hash_sha256",
            "case_count",
            "passed",
        },
        "planted v2 oracle-suite authorization record keys differ",
    )
    relative_path = record["repo_relative_path"]
    _require(
        isinstance(relative_path, str),
        "planted v2 oracle-suite path is invalid",
    )
    expected_prefix = (
        TASK20_V2_ORACLE_RESULT_PREFIX
        + reviewed_candidate_commit
        + "/"
    )
    suffix_parts = (
        relative_path[len(expected_prefix) :].split("/")
        if relative_path.startswith(expected_prefix)
        else []
    )
    _require(
        len(suffix_parts) == 2
        and re.fullmatch(r"attempt_[0-9]{4}", suffix_parts[0]) is not None
        and suffix_parts[1] == "SUITE_RESULT.json",
        "planted v2 oracle-suite path is outside one reviewed attempt",
    )
    _, data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=relative_path,
        name="planted v2 oracle suite result",
    )
    _exact_value(
        record["file_sha256"],
        hashlib.sha256(data).hexdigest(),
        "planted v2 oracle-suite file hash",
    )
    suite = _strict_json(data, name="planted v2 oracle suite result")
    _require(
        data == _canonical_json_bytes(suite) + b"\n",
        "planted v2 oracle suite result is not canonical JSON",
    )
    claimed_content_hash = suite.get("content_hash_sha256")
    _require(
        isinstance(claimed_content_hash, str)
        and _SHA256_RE.fullmatch(claimed_content_hash) is not None,
        "planted v2 oracle-suite content hash is invalid",
    )
    payload = dict(suite)
    payload.pop("content_hash_sha256")
    _exact_value(
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        claimed_content_hash,
        "planted v2 oracle-suite content hash",
    )
    _exact_value(
        record["content_hash_sha256"],
        claimed_content_hash,
        "authorized planted v2 oracle-suite content hash",
    )
    _exact_value(record["case_count"], 7, "authorized planted suite case count")
    _exact_value(record["passed"], True, "authorized planted suite pass")
    _exact_value(
        suite.get("schema_version"),
        "representation-oracle-planted-suite-result-v2",
        "planted v2 oracle-suite schema",
    )
    _exact_value(
        suite.get("scope"),
        "task20_structural_negative_control_collapse_v2_only",
        "planted v2 oracle-suite scope",
    )
    _exact_value(
        suite.get("source_commit"),
        reviewed_candidate_commit,
        "planted v2 oracle-suite reviewed source commit",
    )
    _exact_value(
        suite.get("output_attempt"),
        suffix_parts[0],
        "planted v2 oracle-suite output attempt",
    )
    _exact_value(
        suite.get("manifest_file_sha256"),
        hashlib.sha256(
            reviewed_files[TASK20_V2_ORACLE_MANIFEST_REPO_RELATIVE_PATH]
        ).hexdigest(),
        "planted v2 oracle-suite manifest file hash",
    )
    manifest = _strict_json(
        reviewed_files[TASK20_V2_ORACLE_MANIFEST_REPO_RELATIVE_PATH],
        name="reviewed Task 20 v2 oracle manifest",
    )
    _exact_value(
        suite.get("manifest_content_hash_sha256"),
        manifest.get("content_hash_sha256"),
        "planted v2 oracle-suite manifest content hash",
    )
    _exact_value(
        suite.get("execution_environment"),
        {
            "python_implementation": "CPython",
            "python_version": "3.10.19",
            "numpy_version": "2.2.6",
        },
        "planted v2 oracle-suite execution environment",
    )
    _exact_value(
        suite.get("independent_outcome_matrix_sha256"),
        TASK20_V2_INDEPENDENT_OUTCOME_MATRIX_SHA256,
        "planted v2 independent outcome matrix hash",
    )
    _exact_value(suite.get("case_count"), 7, "planted v2 suite case count")
    _exact_value(suite.get("passed"), True, "planted v2 suite pass")
    _exact_value(
        suite.get("completion_status"),
        "complete_suite_result_written_last",
        "planted v2 suite completion status",
    )
    for field in (
        "dataset_or_checkpoint_opened",
        "model_constructed",
        "optimizer_constructed",
        "training_run",
        "v1_artifact_mutation_authorized",
    ):
        _exact_value(suite.get(field), False, f"planted v2 suite {field}")
    review_authority = suite.get("fable_review_authority")
    _exact_value(
        review_authority,
        {
            "reviewed_candidate_commit": reviewed_candidate_commit,
            "execution_commit": suite.get("execution_commit"),
            "review_repo_relative_path": FABLE_EXECUTION_REVIEW_REPO_RELATIVE_PATH,
            "review_file_sha256": hashlib.sha256(review_data).hexdigest(),
            "reviewed_fileset_sha256": reviewed_fileset_sha256,
            "verdict": "PASS_TO_RUN",
        },
        "planted v2 suite Fable authority",
    )
    execution_commit = suite.get("execution_commit")
    _require(
        isinstance(execution_commit, str)
        and _COMMIT_RE.fullmatch(execution_commit) is not None,
        "planted v2 suite execution commit is invalid",
    )
    _git(
        repo_root,
        ["merge-base", "--is-ancestor", execution_commit, source_commit],
        name="planted v2 suite execution commit is not an ancestor",
    )
    cases = suite.get("cases")
    _require(
        isinstance(cases, list)
        and [case.get("case_id") for case in cases if isinstance(case, Mapping)]
        == list(TASK20_V2_ORACLE_CASE_IDS),
        "planted v2 suite cases/order differ",
    )
    for index, case in enumerate(cases):
        _require(
            isinstance(case, Mapping),
            f"planted v2 suite case {index} is invalid",
        )
        _exact_value(
            case.get("bundle_byte_deterministic"),
            True,
            f"planted v2 suite case {index} bundle determinism",
        )
        _exact_value(
            case.get("result_byte_deterministic"),
            True,
            f"planted v2 suite case {index} result determinism",
        )
        config_check = case.get("evaluation_config_hash_recheck")
        _require(
            isinstance(config_check, Mapping)
            and config_check.get("passed") is True
            and config_check.get("expected_sha256")
            == config_check.get("declared_sha256")
            == config_check.get("recomputed_sha256"),
            f"planted v2 suite case {index} config check failed",
        )
        matrix_check = case.get(
            "independent_literal_outcome_matrix_check"
        )
        _require(
            isinstance(matrix_check, Mapping)
            and matrix_check.get("passed") is True
            and matrix_check.get("expected") == matrix_check.get("observed"),
            f"planted v2 suite case {index} independent matrix check failed",
        )
        assertions = case.get("assertions")
        _require(
            isinstance(assertions, list)
            and bool(assertions)
            and all(
                isinstance(row, Mapping)
                and row.get("passed") is True
                and row.get("expected") == row.get("observed")
                for row in assertions
            ),
            f"planted v2 suite case {index} committed assertions failed",
        )
    return {
        "repo_relative_path": relative_path,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "content_hash_sha256": claimed_content_hash,
    }


def _validate_execution_review_roles(
    *,
    provenance: Mapping[str, Any],
    repo_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    execution_record = _role_record(
        provenance,
        role="checkpoint_execution_authorization",
    )
    review_record = _role_record(provenance, role="fable_execution_review")
    source_manifest_record = _role_record(
        provenance,
        role="checkpoint_runtime_source_manifest",
    )
    preflight_provenance_record = _role_record(
        provenance,
        role="checkpoint_hash_preflight",
    )
    _exact_value(
        execution_record["repo_relative_path"],
        CHECKPOINT_EXECUTION_AUTHORIZATION_REPO_RELATIVE_PATH,
        "checkpoint execution authorization path",
    )
    _exact_value(
        review_record["repo_relative_path"],
        FABLE_EXECUTION_REVIEW_REPO_RELATIVE_PATH,
        "Fable execution review path",
    )
    _exact_value(
        source_manifest_record["repo_relative_path"],
        CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH,
        "checkpoint runtime source manifest provenance path",
    )
    _exact_value(
        preflight_provenance_record["repo_relative_path"],
        CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH,
        "checkpoint preflight provenance path",
    )
    _, execution_data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=str(execution_record["repo_relative_path"]),
        name="checkpoint execution authorization",
    )
    _, review_data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=str(review_record["repo_relative_path"]),
        name="Fable execution review",
    )
    _require(
        hashlib.sha256(execution_data).hexdigest()
        == execution_record["file_sha256"],
        "checkpoint execution authorization hash mismatch",
    )
    _require(
        hashlib.sha256(review_data).hexdigest() == review_record["file_sha256"],
        "Fable execution review hash mismatch",
    )
    execution = _strict_json(execution_data, name="checkpoint execution authorization")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "reviewed_candidate_commit",
        "runtime_source_manifest",
        "checkpoint_hash_preflight",
        "fable_review",
        "planted_v2_oracle_suite",
        "superseded_task20_v1_result",
        "semantic_amendments",
        "authorized_task_order",
        "task20_stop_gate",
        "execution_boundary",
        "content_hash_sha256",
    }
    _require(
        set(execution) == expected_keys,
        "checkpoint execution authorization keys differ",
    )
    claimed_hash = execution.get("content_hash_sha256")
    _require(
        isinstance(claimed_hash, str)
        and _SHA256_RE.fullmatch(claimed_hash) is not None,
        "checkpoint execution authorization content hash is invalid",
    )
    payload = dict(execution)
    payload.pop("content_hash_sha256")
    _require(
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        == claimed_hash,
        "checkpoint execution authorization content hash mismatch",
    )
    _exact_value(
        execution["schema_version"],
        "representation_checkpoint_execution_authorization.v3",
        "checkpoint execution authorization schema",
    )
    _exact_value(
        execution["artifact_type"],
        "representation_checkpoint_execution_authorization",
        "checkpoint execution authorization artifact type",
    )
    reviewed_commit = execution["reviewed_candidate_commit"]
    _require(
        isinstance(reviewed_commit, str)
        and _COMMIT_RE.fullmatch(reviewed_commit) is not None,
        "reviewed_candidate_commit is invalid",
    )
    _git(
        repo_root,
        ["merge-base", "--is-ancestor", reviewed_commit, source_commit],
        name="reviewed candidate is not an ancestor of runtime source_commit",
    )
    runtime_manifest = execution["runtime_source_manifest"]
    _require(
        isinstance(runtime_manifest, Mapping)
        and set(runtime_manifest)
        == {"repo_relative_path", "file_sha256", "content_hash_sha256"},
        "runtime_source_manifest authorization record keys differ",
    )
    _exact_value(
        runtime_manifest["repo_relative_path"],
        CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH,
        "authorized runtime source manifest path",
    )
    source_manifest = _validate_runtime_source_manifest(
        repo_root=repo_root,
        source_commit=source_commit,
        reviewed_candidate_commit=reviewed_commit,
    )
    _exact_value(
        runtime_manifest["file_sha256"],
        source_manifest["file_sha256"],
        "authorized runtime source manifest file hash",
    )
    _exact_value(
        runtime_manifest["content_hash_sha256"],
        source_manifest["content_hash_sha256"],
        "authorized runtime source manifest content hash",
    )
    _exact_value(
        source_manifest_record["file_sha256"],
        source_manifest["file_sha256"],
        "provenance runtime source manifest file hash",
    )
    preflight_record = execution["checkpoint_hash_preflight"]
    _require(
        isinstance(preflight_record, Mapping)
        and set(preflight_record) == {"repo_relative_path", "file_sha256"},
        "checkpoint_hash_preflight authorization record keys differ",
    )
    _exact_value(
        preflight_record["repo_relative_path"],
        CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH,
        "authorized checkpoint preflight path",
    )
    _, preflight_data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH,
        name="checkpoint hash preflight result",
    )
    _exact_value(
        preflight_record["file_sha256"],
        hashlib.sha256(preflight_data).hexdigest(),
        "authorized checkpoint preflight file hash",
    )
    _exact_value(
        preflight_provenance_record["file_sha256"],
        hashlib.sha256(preflight_data).hexdigest(),
        "provenance checkpoint preflight file hash",
    )
    fable = execution["fable_review"]
    _require(
        isinstance(fable, Mapping)
        and set(fable)
        == {
            "repo_relative_path",
            "file_sha256",
            "model",
            "effort",
            "permission_mode",
            "session_persistence",
            "verdict",
            "unresolved_p0_count",
            "unresolved_p1_count",
        },
        "Fable authorization record keys differ",
    )
    _exact_value(
        fable,
        {
            "repo_relative_path": FABLE_EXECUTION_REVIEW_REPO_RELATIVE_PATH,
            "file_sha256": hashlib.sha256(review_data).hexdigest(),
            "model": "fable",
            "effort": "high",
            "permission_mode": "read_only",
            "session_persistence": False,
            "verdict": "PASS_TO_RUN",
            "unresolved_p0_count": 0,
            "unresolved_p1_count": 0,
        },
        "Fable authorization",
    )
    v1_record = execution["superseded_task20_v1_result"]
    _require(
        isinstance(v1_record, Mapping)
        and set(v1_record)
        == {
            "repo_relative_path",
            "file_sha256",
            "content_hash_sha256",
            "source_commit",
            "gate_passed",
        },
        "superseded Task 20 v1 result authorization record keys differ",
    )
    _exact_value(
        v1_record,
        {
            "repo_relative_path": (
                TASK20_CHECKPOINT_REPLAY_RESULT_V1_REPO_RELATIVE_PATH
            ),
            "file_sha256": TASK20_V1_RESULT_FILE_SHA256,
            "content_hash_sha256": TASK20_V1_RESULT_CONTENT_HASH_SHA256,
            "source_commit": TASK20_V1_RESULT_SOURCE_COMMIT,
            "gate_passed": False,
        },
        "superseded Task 20 v1 result authorization",
    )
    _validate_immutable_task20_v1_result(
        repo_root=repo_root,
        source_commit=source_commit,
    )
    amendment_records = execution["semantic_amendments"]
    _require(
        isinstance(amendment_records, list)
        and len(amendment_records) == 3,
        "semantic amendment authorization records differ",
    )
    expected_amendment_paths = [
        TASK20_V2_SEMANTIC_LOCK_REPO_RELATIVE_PATH,
        DECISION_SYNTHESIS_V2_8_AMENDMENT_REPO_RELATIVE_PATH,
        TASK55_CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_REPO_RELATIVE_PATH,
    ]
    amendment_file_sha256_by_path: dict[str, str] = {}
    for index, (record, expected_path) in enumerate(
        zip(amendment_records, expected_amendment_paths)
    ):
        _require(
            isinstance(record, Mapping)
            and set(record) == {"repo_relative_path", "file_sha256"},
            f"semantic amendment {index} record keys differ",
        )
        _exact_value(
            record["repo_relative_path"],
            expected_path,
            f"semantic amendment {index} path",
        )
        _, amendment_data = _committed_working_file(
            repo_root=repo_root,
            source_commit=source_commit,
            repo_relative_path=expected_path,
            name=f"semantic amendment {index}",
        )
        _exact_value(
            record["file_sha256"],
            hashlib.sha256(amendment_data).hexdigest(),
            f"semantic amendment {index} file hash",
        )
        amendment_file_sha256_by_path[expected_path] = hashlib.sha256(
            amendment_data
        ).hexdigest()
        _require_file_unchanged_since_review(
            repo_root=repo_root,
            reviewed_candidate_commit=reviewed_commit,
            repo_relative_path=expected_path,
            current_data=amendment_data,
            name=f"semantic amendment {index}",
        )
    _exact_value(
        execution["authorized_task_order"],
        [20, 55, 80],
        "authorized task order",
    )
    _exact_value(
        execution["task20_stop_gate"],
        {
            "required_before_tasks": [55, 80],
            "required_classification": V2_COLLAPSE_FIELD,
            "required_value": True,
            "legacy_v1_classification": (
                "structural_negative_control_collapse"
            ),
            "legacy_v1_required_recorded_value": False,
            "historical_comparison_resolved_value_count": 245,
            "failure_action": "stop_without_loading_task55_or_task80",
        },
        "Task 20 stop gate",
    )
    _exact_value(
        execution["execution_boundary"],
        _EXPECTED_AUTHORIZATION_BOUNDARY,
        "checkpoint execution authorization boundary",
    )
    reviewed_fileset_sha256, reviewed_v2_files = (
        _validated_reviewed_fileset(
            repo_root=repo_root,
            source_commit=source_commit,
            reviewed_candidate_commit=reviewed_commit,
        )
    )
    _validate_fable_review_binding(
        review_data=review_data,
        reviewed_candidate_commit=reviewed_commit,
        runtime_source_manifest_file_sha256=source_manifest["file_sha256"],
        runtime_source_manifest_content_hash_sha256=source_manifest[
            "content_hash_sha256"
        ],
        checkpoint_hash_preflight_file_sha256=hashlib.sha256(
            preflight_data
        ).hexdigest(),
        semantic_lock_file_sha256=amendment_file_sha256_by_path[
            TASK20_V2_SEMANTIC_LOCK_REPO_RELATIVE_PATH
        ],
        decision_amendment_file_sha256=amendment_file_sha256_by_path[
            DECISION_SYNTHESIS_V2_8_AMENDMENT_REPO_RELATIVE_PATH
        ],
        cross_backend_tolerance_semantic_lock_file_sha256=(
            amendment_file_sha256_by_path[
                TASK55_CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_REPO_RELATIVE_PATH
            ]
        ),
        task20_v2_oracle_harness_file_sha256=hashlib.sha256(
            reviewed_v2_files[
                TASK20_V2_ORACLE_HARNESS_REPO_RELATIVE_PATH
            ]
        ).hexdigest(),
        task20_v2_oracle_manifest_file_sha256=hashlib.sha256(
            reviewed_v2_files[
                TASK20_V2_ORACLE_MANIFEST_REPO_RELATIVE_PATH
            ]
        ).hexdigest(),
        planted_v2_reviewed_fileset_sha256=reviewed_fileset_sha256,
    )
    planted_suite_receipt = _validate_committed_task20_v2_planted_suite(
        repo_root=repo_root,
        source_commit=source_commit,
        record=execution["planted_v2_oracle_suite"],
        reviewed_candidate_commit=reviewed_commit,
        review_data=review_data,
        reviewed_fileset_sha256=reviewed_fileset_sha256,
        reviewed_files=reviewed_v2_files,
    )
    return {
        "reviewed_candidate_commit": reviewed_commit,
        "runtime_source_manifest": source_manifest,
        "runtime_source_manifest_file_sha256": source_manifest[
            "file_sha256"
        ],
        "runtime_source_manifest_content_hash_sha256": source_manifest[
            "content_hash_sha256"
        ],
        "checkpoint_hash_preflight_file_sha256": hashlib.sha256(
            preflight_data
        ).hexdigest(),
        "checkpoint_hash_preflight": _strict_json(
            preflight_data,
            name="checkpoint hash preflight result",
        ),
        "checkpoint_execution_authorization_file_sha256": hashlib.sha256(
            execution_data
        ).hexdigest(),
        "fable_execution_review_file_sha256": hashlib.sha256(
            review_data
        ).hexdigest(),
        "planted_v2_oracle_suite": planted_suite_receipt,
    }


def _validate_inference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable evidence emitted around one inference pass."""

    _require(isinstance(record, Mapping), "inference record must be a mapping")
    expected_keys = {
        "frame_count",
        "channel_count",
        "batch_size",
        "device",
        "model_training_flag",
        "gradients_enabled_inside_inference",
        "state_sha256_before",
        "state_sha256_after",
        "state_unchanged",
    }
    _require(set(record) == expected_keys, "inference record keys differ")
    _exact_value(record["frame_count"], 180, "inference frame count")
    _exact_value(record["channel_count"], 10, "inference channel count")
    _require(
        isinstance(record["batch_size"], int)
        and not isinstance(record["batch_size"], bool)
        and record["batch_size"] > 0,
        "inference batch size must be a positive integer",
    )
    _exact_value(record["device"], "cpu", "inference device")
    _exact_value(
        record["model_training_flag"],
        False,
        "model training flag during inference",
    )
    _exact_value(
        record["gradients_enabled_inside_inference"],
        False,
        "gradient flag during inference",
    )
    before = record["state_sha256_before"]
    after = record["state_sha256_after"]
    _require(
        isinstance(before, str)
        and _SHA256_RE.fullmatch(before) is not None
        and isinstance(after, str)
        and _SHA256_RE.fullmatch(after) is not None,
        "inference model-state hash is invalid",
    )
    _exact_value(after, before, "model state after inference")
    _exact_value(record["state_unchanged"], True, "inference state verdict")
    return dict(record)


def _validate_checkpoint_load_record(
    record: Mapping[str, Any],
    *,
    authorization: CheckpointRuntimeAuthorization,
) -> None:
    """Cross-bind the runtime record from the sole same-descriptor load."""

    _require(isinstance(record, Mapping), "checkpoint load record must be a mapping")
    expected_keys = {
        "checkpoint_epoch",
        "checkpoint_loss",
        "state_key_count",
        "embedded_config_verified",
        "checkpoint_absolute_path",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "load_device",
        "weights_only",
        "same_open_file_descriptor_hash_and_load",
        "unsafe_fallback_used",
    }
    _require(set(record) == expected_keys, "checkpoint load record keys differ")
    _require(
        isinstance(record["checkpoint_epoch"], int)
        and not isinstance(record["checkpoint_epoch"], bool)
        and record["checkpoint_epoch"] >= 0,
        "checkpoint epoch is invalid",
    )
    _require(
        isinstance(record["checkpoint_loss"], (int, float))
        and not isinstance(record["checkpoint_loss"], bool)
        and math.isfinite(float(record["checkpoint_loss"])),
        "checkpoint loss is invalid",
    )
    _require(
        isinstance(record["state_key_count"], int)
        and not isinstance(record["state_key_count"], bool)
        and record["state_key_count"] > 0,
        "checkpoint state-key count is invalid",
    )
    _exact_value(
        record["embedded_config_verified"],
        True,
        "checkpoint embedded-config verdict",
    )
    _exact_value(
        record["checkpoint_absolute_path"],
        authorization.checkpoint_absolute_path,
        "loaded checkpoint path",
    )
    _exact_value(
        record["checkpoint_size_bytes"],
        authorization.checkpoint_size_bytes,
        "loaded checkpoint size",
    )
    _exact_value(
        record["checkpoint_sha256"],
        authorization.checkpoint_sha256,
        "loaded checkpoint hash",
    )
    _exact_value(record["load_device"], "cpu", "checkpoint load device")
    _exact_value(record["weights_only"], True, "weights-only load flag")
    _exact_value(
        record["same_open_file_descriptor_hash_and_load"],
        True,
        "same-descriptor hash/load flag",
    )
    _exact_value(
        record["unsafe_fallback_used"],
        False,
        "unsafe checkpoint fallback flag",
    )


def _validate_immutable_task20_v1_result(
    *,
    repo_root: Path,
    source_commit: str,
) -> dict[str, str]:
    """Verify the exact committed v1 false-negative audit artifact."""

    _, data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=TASK20_CHECKPOINT_REPLAY_RESULT_V1_REPO_RELATIVE_PATH,
        name="immutable Task 20 v1 checkpoint replay result",
    )
    _exact_value(
        hashlib.sha256(data).hexdigest(),
        TASK20_V1_RESULT_FILE_SHA256,
        "immutable Task 20 v1 result file hash",
    )
    result = _strict_json(data, name="immutable Task 20 v1 checkpoint replay result")
    _require(
        data == _canonical_json_bytes(result) + b"\n",
        "immutable Task 20 v1 result is not canonical JSON",
    )
    _exact_value(
        result.get("schema_version"),
        "representation_checkpoint_replay_result.v1",
        "immutable Task 20 v1 result schema",
    )
    _exact_value(
        result.get("content_hash_sha256"),
        TASK20_V1_RESULT_CONTENT_HASH_SHA256,
        "immutable Task 20 v1 result content hash",
    )
    _exact_value(
        result.get("source_commit"),
        TASK20_V1_RESULT_SOURCE_COMMIT,
        "immutable Task 20 v1 result source commit",
    )
    _exact_value(
        result.get("gate"),
        {
            "expected_structural_negative_control_collapse": True,
            "observed_structural_negative_control_collapse": False,
            "classification_passed": False,
            "historical_245_record_comparison_passed": True,
            "passed": False,
            "next_task_authorized": False,
        },
        "immutable Task 20 v1 gate",
    )
    return {
        "file_sha256": TASK20_V1_RESULT_FILE_SHA256,
        "content_hash_sha256": TASK20_V1_RESULT_CONTENT_HASH_SHA256,
        "source_commit": TASK20_V1_RESULT_SOURCE_COMMIT,
    }


def _validate_committed_task20_v3_result(
    *,
    repo_root: Path,
    source_commit: str,
    runtime_source_manifest: Mapping[str, Any],
    checkpoint_hash_preflight_file_sha256: str,
) -> dict[str, str]:
    """Require the exact committed v3 Task 20 stop-gate result."""

    _, data = _committed_working_file(
        repo_root=repo_root,
        source_commit=source_commit,
        repo_relative_path=TASK20_CHECKPOINT_REPLAY_RESULT_V3_REPO_RELATIVE_PATH,
        name="Task 20 v3 checkpoint replay result",
    )
    result = _strict_json(data, name="Task 20 v3 checkpoint replay result")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "task_id",
        "fixture_id",
        "fixture_role",
        "source_commit",
        "candidate_eligibility",
        "execution_boundary",
        "environment",
        "architecture",
        "checkpoint",
        "live_inputs",
        "inference",
        "loaded_source_digests",
        "bundle_content_sha256",
        "evaluator_result",
        "historical_comparison",
        "gate",
        "content_hash_sha256",
    }
    _require(set(result) == expected_keys, "Task 20 result keys differ")
    _require(
        data == _canonical_json_bytes(result) + b"\n",
        "Task 20 result is not canonical JSON with one trailing newline",
    )
    _exact_value(
        result["schema_version"],
        "representation_checkpoint_replay_result.v3",
        "Task 20 v3 result schema",
    )
    _exact_value(
        result["artifact_type"],
        "representation_checkpoint_replay_result",
        "Task 20 result artifact type",
    )
    _exact_value(result["task_id"], 20, "Task 20 result task_id")
    _exact_value(
        result["fixture_id"],
        _TASK_BINDINGS[20]["fixture_id"],
        "Task 20 result fixture_id",
    )
    _exact_value(
        result["fixture_role"],
        _TASK_BINDINGS[20]["role"],
        "Task 20 result role",
    )
    _exact_value(
        result["candidate_eligibility"],
        "forbidden",
        "Task 20 candidate eligibility",
    )
    claimed_content_hash = result["content_hash_sha256"]
    _require(
        isinstance(claimed_content_hash, str)
        and _SHA256_RE.fullmatch(claimed_content_hash) is not None,
        "Task 20 result content hash is invalid",
    )
    result_payload = dict(result)
    result_payload.pop("content_hash_sha256")
    _exact_value(
        hashlib.sha256(_canonical_json_bytes(result_payload)).hexdigest(),
        claimed_content_hash,
        "Task 20 result content hash",
    )
    result_source_commit = result["source_commit"]
    _require(
        isinstance(result_source_commit, str)
        and _COMMIT_RE.fullmatch(result_source_commit) is not None,
        "Task 20 result source_commit is invalid",
    )
    _git(
        repo_root,
        ["merge-base", "--is-ancestor", result_source_commit, source_commit],
        name="Task 20 result source_commit is not an ancestor of current source_commit",
    )
    execution_boundary = result["execution_boundary"]
    _exact_value(
        execution_boundary,
        {
            "checkpoint_loaded": True,
            "training_performed": False,
            "optimizer_constructed": False,
            "weight_update_performed": False,
            "device": "cpu",
            "weights_only": True,
            "selection_use_authorized": False,
        },
        "Task 20 result execution boundary",
    )
    checkpoint = result["checkpoint"]
    provisional = CheckpointRuntimeAuthorization(
        task_id=20,
        fixture_id=str(_TASK_BINDINGS[20]["fixture_id"]),
        role=str(_TASK_BINDINGS[20]["role"]),
        runtime_source_commit=result_source_commit,
        repository_root=str(repo_root),
        checkpoint_absolute_path=(
            f"{_TASK_BINDINGS[20]['run_directory']}/best_model.pt"
        ),
        checkpoint_sha256=str(_TASK_BINDINGS[20]["checkpoint_sha256"]),
        checkpoint_size_bytes=int(_TASK_BINDINGS[20]["checkpoint_size_bytes"]),
        config_absolute_path=f"{_TASK_BINDINGS[20]['run_directory']}/config.json",
        config_sha256=str(_TASK_BINDINGS[20]["config_sha256"]),
        config_size_bytes=int(_TASK_BINDINGS[20]["config_size_bytes"]),
        history_absolute_path=f"{_TASK_BINDINGS[20]['run_directory']}/history.json",
        history_sha256=str(_TASK_BINDINGS[20]["history_sha256"]),
        history_size_bytes=int(_TASK_BINDINGS[20]["history_size_bytes"]),
        dataset_root=_DATASET_ROOT,
        reviewed_candidate_commit=result_source_commit,
        runtime_source_manifest_file_sha256=str(
            runtime_source_manifest["file_sha256"]
        ),
        runtime_source_manifest_content_hash_sha256=str(
            runtime_source_manifest["content_hash_sha256"]
        ),
        checkpoint_hash_preflight_file_sha256=(
            checkpoint_hash_preflight_file_sha256
        ),
        checkpoint_execution_authorization_file_sha256="0" * 64,
        fable_execution_review_file_sha256="0" * 64,
        expected_learn_inverse_operator=True,
    )
    _validate_checkpoint_load_record(checkpoint, authorization=provisional)
    _validate_inference_record(result["inference"])

    loaded_sources = result["loaded_source_digests"]
    _require(
        isinstance(loaded_sources, Mapping),
        "Task 20 loaded source digests must be a mapping",
    )
    manifest_sources = runtime_source_manifest["sources"]
    _require(
        set(loaded_sources) == set(manifest_sources),
        "Task 20 loaded source role closure differs",
    )
    expected_source_paths = dict(_EXPECTED_RUNTIME_SOURCES)
    for role, source_record in manifest_sources.items():
        loaded = loaded_sources[role]
        _require(
            isinstance(loaded, Mapping)
            and set(loaded) == {
                "module",
                "absolute_path",
                "import_time_sha256",
            },
            f"Task 20 loaded source record {role} keys differ",
        )
        _exact_value(
            loaded["import_time_sha256"],
            source_record["sha256"],
            f"Task 20 loaded source hash {role}",
        )
        _exact_value(
            loaded["absolute_path"],
            source_record["absolute_path"],
            f"Task 20 loaded source absolute path {role}",
        )
        _exact_value(
            loaded["module"],
            ".".join(
                Path(expected_source_paths[role]).with_suffix("").parts
            ),
            f"Task 20 loaded source module {role}",
        )
        result_source_bytes = _git(
            repo_root,
            [
                "show",
                f"{result_source_commit}:{expected_source_paths[role]}",
            ],
            name=f"cannot read Task 20 runtime source {role}",
        )
        _exact_value(
            hashlib.sha256(result_source_bytes).hexdigest(),
            source_record["sha256"],
            f"Task 20 runtime source blob {role}",
        )

    bundle_hash = result["bundle_content_sha256"]
    _require(
        isinstance(bundle_hash, str)
        and _SHA256_RE.fullmatch(bundle_hash) is not None,
        "Task 20 bundle hash is invalid",
    )
    evaluator_result = result["evaluator_result"]
    _require(
        isinstance(evaluator_result, Mapping),
        "Task 20 evaluator result must be a mapping",
    )
    evaluator_hash = evaluator_result.get("result_content_sha256")
    _require(
        isinstance(evaluator_hash, str)
        and _SHA256_RE.fullmatch(evaluator_hash) is not None,
        "Task 20 evaluator result content hash is invalid",
    )
    evaluator_payload = dict(evaluator_result)
    evaluator_payload.pop("result_content_sha256")
    _exact_value(
        hashlib.sha256(_canonical_json_bytes(evaluator_payload)).hexdigest(),
        evaluator_hash,
        "Task 20 evaluator result content hash",
    )
    _exact_value(
        evaluator_result.get("schema_version"),
        "representation-evaluation-result-v2",
        "Task 20 evaluator result schema",
    )
    _exact_value(
        evaluator_result.get("case_id"),
        _TASK_BINDINGS[20]["fixture_id"],
        "Task 20 evaluator case_id",
    )
    _exact_value(
        evaluator_result.get("case_kind"),
        "checkpoint",
        "Task 20 evaluator case_kind",
    )
    _exact_value(
        evaluator_result.get("bundle_content_sha256"),
        result["bundle_content_sha256"],
        "Task 20 evaluator/outer bundle content hash",
    )
    evaluator_checkpoint_authorization = evaluator_result.get(
        "checkpoint_authorization"
    )
    _require(
        isinstance(evaluator_checkpoint_authorization, Mapping),
        "Task 20 evaluator checkpoint authorization is missing",
    )
    preflight_hash = evaluator_checkpoint_authorization.get(
        "checkpoint_hash_preflight_file_sha256"
    )
    _require(
        isinstance(preflight_hash, str)
        and _SHA256_RE.fullmatch(preflight_hash) is not None,
        "Task 20 evaluator checkpoint preflight hash is invalid",
    )
    _exact_value(
        preflight_hash,
        checkpoint_hash_preflight_file_sha256,
        "Task 20 evaluator checkpoint preflight hash",
    )
    _exact_value(
        evaluator_checkpoint_authorization,
        {
            "checkpoint_evaluation_authorized": True,
            "source_commit": result_source_commit,
            "task_id": 20,
            "fixture_id": _TASK_BINDINGS[20]["fixture_id"],
            "checkpoint_sha256": _TASK_BINDINGS[20]["checkpoint_sha256"],
            "runtime_source_manifest_file_sha256": runtime_source_manifest[
                "file_sha256"
            ],
            "checkpoint_hash_preflight_file_sha256": preflight_hash,
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": False,
        },
        "Task 20 evaluator checkpoint authorization",
    )
    _exact_value(
        evaluator_result.get("validated_checkpoint"),
        {
            "absolute_path": checkpoint["checkpoint_absolute_path"],
            "sha256": checkpoint["checkpoint_sha256"],
            "size_bytes": checkpoint["checkpoint_size_bytes"],
        },
        "Task 20 evaluator validated checkpoint",
    )
    evaluator_provenance = evaluator_result.get("provenance")
    _require(
        isinstance(evaluator_provenance, Mapping),
        "Task 20 evaluator provenance is missing",
    )
    _exact_value(
        {
            "source_commit": evaluator_provenance.get("source_commit"),
            "case_kind": evaluator_provenance.get("case_kind"),
        },
        {
            "source_commit": result_source_commit,
            "case_kind": "checkpoint",
        },
        "Task 20 evaluator provenance identity",
    )
    _exact_value(
        evaluator_result.get("stratum"),
        {
            "object_id": "engineers_hammer_vray",
            "seed": 42,
            "partition": "full_corpus",
            "transform_family": "roll",
            "direction": "forward",
            "stride": 3,
        },
        "Task 20 evaluator stratum",
    )
    health = evaluator_result.get("channel_health")
    _require(
        isinstance(health, Mapping),
        "Task 20 v2 channel health is missing",
    )
    channel_rows = health.get("channels")
    _require(
        isinstance(channel_rows, list) and len(channel_rows) == 10,
        "Task 20 v2 channel-health row count differs",
    )
    health_by_channel: dict[int, Mapping[str, Any]] = {}
    for row in channel_rows:
        _require(
            isinstance(row, Mapping),
            "Task 20 v2 channel-health row must be a mapping",
        )
        channel = row.get("channel")
        _require(
            isinstance(channel, int)
            and not isinstance(channel, bool)
            and 0 <= channel < 10
            and channel not in health_by_channel,
            "Task 20 v2 channel-health indices differ",
        )
        flat_dead_status = row.get("heatmap_flat_dead")
        _require(
            flat_dead_status is True
            or flat_dead_status is False
            or flat_dead_status is None,
            "Task 20 v2 flat/dead status is invalid",
        )
        health_by_channel[channel] = row
    _exact_value(
        sorted(health_by_channel),
        list(range(10)),
        "Task 20 v2 channel-health index closure",
    )
    confirmed_flat_dead = [
        channel
        for channel, row in sorted(health_by_channel.items())
        if row.get("heatmap_flat_dead") is True
    ]
    retained_channels = [
        channel
        for channel, row in sorted(health_by_channel.items())
        if row.get("heatmap_flat_dead") is not True
    ]
    _exact_value(
        confirmed_flat_dead,
        [8],
        "Task 20 v2 raw confirmed flat-dead channels",
    )
    _exact_value(
        retained_channels,
        [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "Task 20 v2 raw retained structural channels",
    )
    collapse = evaluator_result.get("collapse_evidence")
    _require(isinstance(collapse, Mapping), "Task 20 collapse evidence is missing")
    _exact_value(
        collapse.get("structural_negative_control_collapse"),
        False,
        "Task 20 legacy v1 all-channel structural collapse flag",
    )
    _exact_value(
        collapse.get(V2_COLLAPSE_FIELD),
        True,
        "Task 20 v2 structural collapse flag",
    )
    _exact_value(
        collapse.get(
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead"
        ),
        "available",
        "Task 20 v2 structural collapse status",
    )
    _exact_value(
        collapse.get("confirmed_flat_dead_channel_indices"),
        confirmed_flat_dead,
        "Task 20 v2 confirmed flat-dead channels",
    )
    _exact_value(
        collapse.get("channels_not_confirmed_flat_dead_indices"),
        retained_channels,
        "Task 20 v2 retained structural channels",
    )
    _exact_value(
        collapse.get("channels_not_confirmed_flat_dead_count"),
        len(retained_channels),
        "Task 20 v2 retained structural channel count",
    )
    _exact_value(
        collapse.get("minimum_channels_not_confirmed_flat_dead"),
        2,
        "Task 20 v2 minimum retained structural channel count",
    )
    separation = evaluator_result.get("trajectory_separation")
    _require(
        isinstance(separation, Mapping),
        "Task 20 v2 trajectory separation is missing",
    )
    _exact_value(
        {
            "channels_not_confirmed_flat_dead_count": separation.get(
                "channels_not_confirmed_flat_dead_count"
            ),
            "channels_not_confirmed_flat_dead_indices": separation.get(
                "channels_not_confirmed_flat_dead_indices"
            ),
            "confirmed_flat_dead_channel_indices": separation.get(
                "confirmed_flat_dead_channel_indices"
            ),
            "channels_not_confirmed_flat_dead_metric_void": separation.get(
                "channels_not_confirmed_flat_dead_metric_void"
            ),
        },
        {
            "channels_not_confirmed_flat_dead_count": len(retained_channels),
            "channels_not_confirmed_flat_dead_indices": retained_channels,
            "confirmed_flat_dead_channel_indices": confirmed_flat_dead,
            "channels_not_confirmed_flat_dead_metric_void": False,
        },
        "Task 20 v2 trajectory-separation summary",
    )
    retained_metric = separation.get("channels_not_confirmed_flat_dead")
    _require(
        isinstance(retained_metric, Mapping),
        "Task 20 v2 retained-channel metric is missing",
    )
    _exact_value(
        {
            "channel_indices": retained_metric.get("channel_indices"),
            "pair_count": retained_metric.get("pair_count"),
            "evaluable_pair_count": retained_metric.get(
                "evaluable_pair_count"
            ),
            "void_pair_count": retained_metric.get("void_pair_count"),
            "all_pairs_persistent_duplicate": retained_metric.get(
                "all_pairs_persistent_duplicate"
            ),
            "persistent_duplicate_count": (
                retained_metric.get("category_counts", {}).get(
                    "persistent_duplicate"
                )
                if isinstance(retained_metric.get("category_counts"), Mapping)
                else None
            ),
        },
        {
            "channel_indices": [0, 1, 2, 3, 4, 5, 6, 7, 9],
            "pair_count": 36,
            "evaluable_pair_count": 36,
            "void_pair_count": 0,
            "all_pairs_persistent_duplicate": True,
            "persistent_duplicate_count": 36,
        },
        "Task 20 v2 retained-channel collapse evidence",
    )
    _exact_value(
        retained_metric.get("category_counts"),
        {
            "persistent_duplicate": 36,
            "recurrent_close_pair": 0,
            "transient_crossing": 0,
            "clustered_pair": 0,
            "separate_pair": 0,
            "void_no_joint_visibility": 0,
        },
        "Task 20 v2 retained-channel category counts",
    )
    comparison = result["historical_comparison"]
    _require(
        isinstance(comparison, Mapping),
        "Task 20 historical comparison must be a mapping",
    )
    _exact_value(
        comparison.get("frozen_record_count"),
        245,
        "Task 20 historical comparison record count",
    )
    _exact_value(
        comparison.get("failed_record_count"),
        0,
        "Task 20 historical comparison failure count",
    )
    _exact_value(
        comparison.get("passed_record_count"),
        245,
        "Task 20 historical comparison passed count",
    )
    _exact_value(
        comparison.get("all_definition_identical_fields_passed"),
        True,
        "Task 20 historical comparison verdict",
    )
    _exact_value(
        result["gate"],
        {
            "classification_field": V2_COLLAPSE_FIELD,
            "expected_structural_negative_control_collapse_v2": True,
            "observed_structural_negative_control_collapse_v2": True,
            "observed_legacy_v1_all_channel_structural_negative_control_collapse": (
                False
            ),
            "classification_passed": True,
            "historical_245_record_comparison_passed": True,
            "passed": True,
            "next_task_authorized": True,
        },
        "Task 20 stop gate result",
    )
    return {
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "content_hash_sha256": claimed_content_hash,
        "source_commit": result_source_commit,
    }


def authorize_checkpoint_runtime(
    *,
    registry: replay_registry.ValidatedReplayRegistry,
    provenance: replay_registry.VerifiedRegistryProvenanceReceipt,
    checkpoint_hash_preflight: Mapping[str, Any],
    task_id: int,
    runtime_source_commit: str,
    evaluation_provenance: Mapping[str, Any],
) -> CheckpointRuntimeAuthorization:
    """Mint one private runtime capability without touching a checkpoint."""

    _require(
        isinstance(registry, replay_registry.ValidatedReplayRegistry),
        "registry receipt has wrong type",
    )
    replay_registry._require_preflight_authorization(registry, provenance)
    _require(
        runtime_source_commit == provenance.source_commit,
        "runtime source commit differs from registry provenance commit",
    )
    _require(
        isinstance(task_id, int) and not isinstance(task_id, bool),
        "task_id must be an integer",
    )
    matches = [
        candidate
        for candidate in registry.checkpoint_candidates
        if candidate.task_id == task_id
    ]
    _require(len(matches) == 1, "registry does not bind exactly one requested task")
    candidate = matches[0]
    binding = _binding_for_candidate(candidate)
    repo_root = Path(__file__).resolve().parents[1]
    review_authorization = _validate_execution_review_roles(
        provenance=evaluation_provenance,
        repo_root=repo_root,
        source_commit=runtime_source_commit,
    )
    committed_preflight = review_authorization["checkpoint_hash_preflight"]
    _require(
        dict(checkpoint_hash_preflight) == committed_preflight,
        "caller checkpoint preflight differs from committed authorization",
    )
    _validate_checkpoint_preflight(
        preflight=committed_preflight,
        candidate=candidate,
    )
    task20_gate: dict[str, str] | None = None
    task20_v1_audit: dict[str, str] | None = None
    if task_id in {55, 80}:
        task20_v1_audit = _validate_immutable_task20_v1_result(
            repo_root=repo_root,
            source_commit=runtime_source_commit,
        )
        task20_gate = _validate_committed_task20_v3_result(
            repo_root=repo_root,
            source_commit=runtime_source_commit,
            runtime_source_manifest=review_authorization[
                "runtime_source_manifest"
            ],
            checkpoint_hash_preflight_file_sha256=str(
                review_authorization[
                    "checkpoint_hash_preflight_file_sha256"
                ]
            ),
        )
    _validate_actual_runtime_environment()

    run_directory = str(binding["run_directory"])
    return CheckpointRuntimeAuthorization(
        task_id=task_id,
        fixture_id=str(binding["fixture_id"]),
        role=str(binding["role"]),
        runtime_source_commit=runtime_source_commit,
        repository_root=str(repo_root),
        checkpoint_absolute_path=candidate.absolute_path,
        checkpoint_sha256=candidate.expected_sha256,
        checkpoint_size_bytes=candidate.expected_size_bytes,
        config_absolute_path=f"{run_directory}/config.json",
        config_sha256=str(binding["config_sha256"]),
        config_size_bytes=int(binding["config_size_bytes"]),
        history_absolute_path=f"{run_directory}/history.json",
        history_sha256=str(binding["history_sha256"]),
        history_size_bytes=int(binding["history_size_bytes"]),
        dataset_root=_DATASET_ROOT,
        reviewed_candidate_commit=str(
            review_authorization["reviewed_candidate_commit"]
        ),
        runtime_source_manifest_file_sha256=str(
            review_authorization["runtime_source_manifest_file_sha256"]
        ),
        runtime_source_manifest_content_hash_sha256=str(
            review_authorization[
                "runtime_source_manifest_content_hash_sha256"
            ]
        ),
        checkpoint_hash_preflight_file_sha256=str(
            review_authorization["checkpoint_hash_preflight_file_sha256"]
        ),
        checkpoint_execution_authorization_file_sha256=str(
            review_authorization[
                "checkpoint_execution_authorization_file_sha256"
            ]
        ),
        fable_execution_review_file_sha256=str(
            review_authorization["fable_execution_review_file_sha256"]
        ),
        expected_learn_inverse_operator=bool(
            binding["learn_inverse_operator"]
        ),
        task20_gate_result_file_sha256=(
            task20_gate["file_sha256"] if task20_gate is not None else None
        ),
        task20_gate_result_content_hash_sha256=(
            task20_gate["content_hash_sha256"]
            if task20_gate is not None
            else None
        ),
        task20_gate_source_commit=(
            task20_gate["source_commit"] if task20_gate is not None else None
        ),
        task20_v1_audit_file_sha256=(
            task20_v1_audit["file_sha256"]
            if task20_v1_audit is not None
            else None
        ),
        task20_v1_audit_content_hash_sha256=(
            task20_v1_audit["content_hash_sha256"]
            if task20_v1_audit is not None
            else None
        ),
        _capability=_VERIFIED_RUNTIME_CAPABILITY,
    )


def require_checkpoint_runtime_authorization(
    authorization: CheckpointRuntimeAuthorization,
    *,
    task_id: int | None = None,
) -> CheckpointRuntimeAuthorization:
    """Validate the private capability and immutable no-training boundary."""

    _require(
        isinstance(authorization, CheckpointRuntimeAuthorization)
        and authorization._capability is _VERIFIED_RUNTIME_CAPABILITY,
        "a live checkpoint runtime capability is required",
    )
    if task_id is not None:
        _require(authorization.task_id == task_id, "capability binds another task")
    _require(authorization.cpu_only is True, "runtime capability is not CPU-only")
    _require(
        authorization.training_or_weight_update_authorized is False,
        "runtime capability must forbid training/weight updates",
    )
    _require(
        authorization.selection_use_authorized is False,
        "runtime capability must forbid selection use",
    )
    if authorization.task_id == 20:
        _require(
            authorization.task20_gate_result_file_sha256 is None
            and authorization.task20_gate_result_content_hash_sha256 is None
            and authorization.task20_gate_source_commit is None,
            "Task 20 capability must not claim its own stop-gate result",
        )
        _require(
            authorization.task20_v1_audit_file_sha256 is None
            and authorization.task20_v1_audit_content_hash_sha256 is None,
            "Task 20 capability must not claim the downstream v1 audit binding",
        )
    else:
        _require(
            authorization.task_id in {55, 80}
            and isinstance(
                authorization.task20_gate_result_file_sha256,
                str,
            )
            and _SHA256_RE.fullmatch(
                authorization.task20_gate_result_file_sha256
            )
            is not None
            and isinstance(
                authorization.task20_gate_result_content_hash_sha256,
                str,
            )
            and _SHA256_RE.fullmatch(
                authorization.task20_gate_result_content_hash_sha256
            )
            is not None
            and isinstance(authorization.task20_gate_source_commit, str)
            and _COMMIT_RE.fullmatch(authorization.task20_gate_source_commit)
            is not None,
            "Tasks 55/80 capability lacks the committed Task 20 stop-gate binding",
        )
        _exact_value(
            authorization.task20_v1_audit_file_sha256,
            TASK20_V1_RESULT_FILE_SHA256,
            "Tasks 55/80 capability Task 20 v1 audit file hash",
        )
        _exact_value(
            authorization.task20_v1_audit_content_hash_sha256,
            TASK20_V1_RESULT_CONTENT_HASH_SHA256,
            "Tasks 55/80 capability Task 20 v1 audit content hash",
        )
    return authorization


def _checkpoint_record_from_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    return _external_role_from_bundle(bundle, role="checkpoint")


def _external_role_from_bundle(
    bundle: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any]:
    provenance = bundle.get("provenance")
    _require(isinstance(provenance, Mapping), "bundle provenance is missing")
    records = provenance.get("external_files")
    _require(isinstance(records, list), "provenance external_files must be a list")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("role") == role
    ]
    _require(len(matches) == 1, f"bundle must bind exactly one {role}")
    record = matches[0]
    _require(
        set(record)
        == {"role", "absolute_path", "file_sha256", "size_bytes"},
        f"bundle external role {role} keys differ",
    )
    return record


def _compact_float32_output_hash(
    bundle: Mapping[str, Any],
    *,
    key: str,
    shape: list[int],
) -> str:
    evaluation = bundle.get("evaluation")
    _require(isinstance(evaluation, Mapping), "bundle evaluation is missing")
    record = evaluation.get(key)
    _require(
        isinstance(record, Mapping),
        f"bundle evaluation {key} must be a compact-array record",
    )
    expected_keys = {
        "schema_version",
        "encoding",
        "dtype",
        "shape",
        "order",
        "payload_base64",
        "payload_nbytes",
        "payload_sha256",
        "content_sha256",
    }
    _require(
        set(record) == expected_keys,
        f"bundle evaluation {key} compact-array keys differ",
    )
    _exact_value(
        record["schema_version"],
        "representation_compact_array.v1",
        f"bundle evaluation {key} compact-array schema",
    )
    _exact_value(
        record["encoding"],
        "raw_float32_le_base64",
        f"bundle evaluation {key} compact-array encoding",
    )
    _exact_value(
        record["dtype"],
        "float32",
        f"bundle evaluation {key} compact-array dtype",
    )
    _exact_value(record["shape"], shape, f"bundle evaluation {key} shape")
    _exact_value(
        record["order"],
        "C",
        f"bundle evaluation {key} compact-array order",
    )
    expected_nbytes = math.prod(shape) * 4
    _exact_value(
        record["payload_nbytes"],
        expected_nbytes,
        f"bundle evaluation {key} payload size",
    )
    _require(
        isinstance(record["payload_base64"], str)
        and bool(record["payload_base64"]),
        f"bundle evaluation {key} payload is missing",
    )
    _require(
        isinstance(record["payload_sha256"], str)
        and _SHA256_RE.fullmatch(record["payload_sha256"]) is not None,
        f"bundle evaluation {key} payload hash is invalid",
    )
    content_hash = record["content_sha256"]
    _require(
        isinstance(content_hash, str)
        and _SHA256_RE.fullmatch(content_hash) is not None,
        f"bundle evaluation {key} content hash is invalid",
    )
    record_payload = dict(record)
    record_payload.pop("content_sha256")
    _exact_value(
        hashlib.sha256(_canonical_json_bytes(record_payload)).hexdigest(),
        content_hash,
        f"bundle evaluation {key} content hash",
    )
    return content_hash


def _validate_bundle_against_runtime_authorization(
    bundle: Mapping[str, Any],
    *,
    authorization: CheckpointRuntimeAuthorization,
) -> tuple[str, str, str]:
    _require(isinstance(bundle, Mapping), "checkpoint bundle must be a mapping")
    _exact_value(bundle.get("case_kind"), "checkpoint", "bundle case kind")
    _exact_value(bundle.get("case_id"), authorization.fixture_id, "bundle case_id")
    _require(
        _PENDING_EVALUATOR_AUTHORIZATION.get() is None,
        "an evaluator authorization is already pending",
    )
    claimed_hash = bundle.get("bundle_content_sha256")
    _require(
        isinstance(claimed_hash, str) and _SHA256_RE.fullmatch(claimed_hash),
        "bundle_content_sha256 is invalid",
    )
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        == claimed_hash,
        "cannot arm evaluator for a bundle with a wrong content hash",
    )
    checkpoint_record = _checkpoint_record_from_bundle(bundle)
    _exact_value(
        checkpoint_record.get("file_sha256"),
        authorization.checkpoint_sha256,
        "bundle checkpoint hash",
    )
    _exact_value(
        checkpoint_record.get("absolute_path"),
        authorization.checkpoint_absolute_path,
        "bundle checkpoint path",
    )
    _exact_value(
        checkpoint_record.get("size_bytes"),
        authorization.checkpoint_size_bytes,
        "bundle checkpoint size",
    )
    for role, path, digest, size in (
        (
            "checkpoint_config",
            authorization.config_absolute_path,
            authorization.config_sha256,
            authorization.config_size_bytes,
        ),
        (
            "checkpoint_metadata",
            authorization.history_absolute_path,
            authorization.history_sha256,
            authorization.history_size_bytes,
        ),
    ):
        record = _external_role_from_bundle(bundle, role=role)
        _exact_value(record["absolute_path"], path, f"bundle {role} path")
        _exact_value(record["file_sha256"], digest, f"bundle {role} hash")
        _exact_value(record["size_bytes"], size, f"bundle {role} size")
    provenance = bundle["provenance"]
    _exact_value(
        provenance.get("source_commit"),
        authorization.runtime_source_commit,
        "bundle source_commit",
    )
    points_hash = _compact_float32_output_hash(
        bundle,
        key="points",
        shape=[180, 10, 2],
    )
    logits_hash = _compact_float32_output_hash(
        bundle,
        key="logits",
        shape=[180, 10, 64, 64],
    )
    return claimed_hash, points_hash, logits_hash


def _register_loaded_checkpoint_after_same_descriptor_load(
    *,
    runtime_authorization: CheckpointRuntimeAuthorization,
    checkpoint_record: Mapping[str, Any],
) -> None:
    """Register the sole successful same-descriptor hash/load operation.

    The runtime calls this immediately after ``torch.load`` has returned and
    the strict payload/model reconstruction has succeeded.  No file is opened
    here; the exact runtime record is only cross-bound to the private token.
    """

    authorization = require_checkpoint_runtime_authorization(
        runtime_authorization
    )
    _require(
        _REGISTERED_LOADED_CHECKPOINT.get() is None,
        "a loaded checkpoint receipt is already registered",
    )
    _require(
        _PENDING_EVALUATOR_AUTHORIZATION.get() is None
        and _PENDING_PROVENANCE_LOADED_CHECKPOINT.get() is None,
        "a prior checkpoint receipt is still pending",
    )
    _validate_checkpoint_load_record(
        checkpoint_record,
        authorization=authorization,
    )
    _REGISTERED_LOADED_CHECKPOINT.set(
        _LoadedCheckpointReceipt(
            source_commit=authorization.runtime_source_commit,
            task_id=authorization.task_id,
            fixture_id=authorization.fixture_id,
            checkpoint_absolute_path=authorization.checkpoint_absolute_path,
            checkpoint_sha256=authorization.checkpoint_sha256,
            checkpoint_size_bytes=authorization.checkpoint_size_bytes,
            _capability=_VERIFIED_LOADED_CHECKPOINT_CAPABILITY,
        )
    )


def _mint_checkpoint_evaluator_context_after_inference(
    bundle: Mapping[str, Any],
    *,
    runtime_authorization: CheckpointRuntimeAuthorization,
    inference_record: Mapping[str, Any],
) -> None:
    """Arm evaluator only after a registered load and unchanged inference."""

    loaded = _REGISTERED_LOADED_CHECKPOINT.get()
    _REGISTERED_LOADED_CHECKPOINT.set(None)
    _require(
        isinstance(loaded, _LoadedCheckpointReceipt)
        and loaded._capability is _VERIFIED_LOADED_CHECKPOINT_CAPABILITY,
        "evaluator authorization requires a registered same-descriptor checkpoint load",
    )
    authorization = require_checkpoint_runtime_authorization(
        runtime_authorization
    )
    _exact_value(
        loaded.source_commit,
        authorization.runtime_source_commit,
        "loaded receipt source_commit",
    )
    _exact_value(
        loaded.task_id,
        authorization.task_id,
        "loaded receipt task_id",
    )
    _exact_value(
        loaded.fixture_id,
        authorization.fixture_id,
        "loaded receipt fixture_id",
    )
    _exact_value(
        loaded.checkpoint_absolute_path,
        authorization.checkpoint_absolute_path,
        "loaded receipt checkpoint path",
    )
    _exact_value(
        loaded.checkpoint_sha256,
        authorization.checkpoint_sha256,
        "loaded receipt checkpoint hash",
    )
    _exact_value(
        loaded.checkpoint_size_bytes,
        authorization.checkpoint_size_bytes,
        "loaded receipt checkpoint size",
    )
    validated_inference = _validate_inference_record(inference_record)
    claimed_hash, points_hash, logits_hash = (
        _validate_bundle_against_runtime_authorization(
            bundle,
            authorization=authorization,
        )
    )
    _PENDING_EVALUATOR_AUTHORIZATION.set(
        _PendingEvaluatorAuthorization(
            bundle_content_sha256=claimed_hash,
            source_commit=authorization.runtime_source_commit,
            task_id=authorization.task_id,
            fixture_id=authorization.fixture_id,
            checkpoint_sha256=authorization.checkpoint_sha256,
            runtime_source_manifest_file_sha256=(
                authorization.runtime_source_manifest_file_sha256
            ),
            checkpoint_hash_preflight_file_sha256=(
                authorization.checkpoint_hash_preflight_file_sha256
            ),
            points_content_sha256=points_hash,
            logits_content_sha256=logits_hash,
            model_state_sha256=validated_inference[
                "state_sha256_after"
            ],
            loaded_checkpoint_receipt=loaded,
            _capability=_VERIFIED_EVALUATOR_CAPABILITY,
        )
    )


def validate_checkpoint_evaluator_authorization(
    bundle: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Consume the live one-shot runtime context for exactly ``bundle``.

    The pending context is cleared before validation, so a failed attempt
    cannot be retried with a modified bundle.
    """

    pending = _PENDING_EVALUATOR_AUTHORIZATION.get()
    _PENDING_EVALUATOR_AUTHORIZATION.set(None)
    _require(
        isinstance(pending, _PendingEvaluatorAuthorization)
        and pending._capability is _VERIFIED_EVALUATOR_CAPABILITY,
        "checkpoint evaluation requires a live one-shot runtime context",
    )
    _require(bundle.get("case_kind") == "checkpoint", "bundle is not a checkpoint case")
    claimed_hash = bundle.get("bundle_content_sha256")
    _exact_value(
        claimed_hash,
        pending.bundle_content_sha256,
        "one-shot bundle content hash",
    )
    payload = dict(bundle)
    payload.pop("bundle_content_sha256", None)
    _require(
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        == claimed_hash,
        "bundle content changed after runtime authorization",
    )
    checkpoint = _checkpoint_record_from_bundle(bundle)
    _exact_value(
        checkpoint.get("file_sha256"),
        pending.checkpoint_sha256,
        "one-shot checkpoint hash",
    )
    _exact_value(
        checkpoint.get("absolute_path"),
        pending.loaded_checkpoint_receipt.checkpoint_absolute_path,
        "one-shot checkpoint path",
    )
    _exact_value(
        checkpoint.get("size_bytes"),
        pending.loaded_checkpoint_receipt.checkpoint_size_bytes,
        "one-shot checkpoint size",
    )
    provenance = bundle.get("provenance")
    _require(isinstance(provenance, Mapping), "bundle provenance is missing")
    _exact_value(
        provenance.get("source_commit"),
        pending.source_commit,
        "one-shot source_commit",
    )
    _exact_value(
        _compact_float32_output_hash(
            bundle,
            key="points",
            shape=[180, 10, 2],
        ),
        pending.points_content_sha256,
        "one-shot points content hash",
    )
    _exact_value(
        _compact_float32_output_hash(
            bundle,
            key="logits",
            shape=[180, 10, 64, 64],
        ),
        pending.logits_content_sha256,
        "one-shot logits content hash",
    )
    _require(
        _PENDING_PROVENANCE_LOADED_CHECKPOINT.get() is None,
        "a provenance checkpoint receipt is already pending",
    )
    _PENDING_PROVENANCE_LOADED_CHECKPOINT.set(
        pending.loaded_checkpoint_receipt
    )
    return MappingProxyType(
        {
            "checkpoint_evaluation_authorized": True,
            "source_commit": pending.source_commit,
            "task_id": pending.task_id,
            "fixture_id": pending.fixture_id,
            "checkpoint_sha256": pending.checkpoint_sha256,
            "runtime_source_manifest_file_sha256": (
                pending.runtime_source_manifest_file_sha256
            ),
            "checkpoint_hash_preflight_file_sha256": (
                pending.checkpoint_hash_preflight_file_sha256
            ),
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": False,
        }
    )


def consume_checkpoint_provenance_load_receipt(
    *,
    source_commit: str,
    checkpoint_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Consume the loaded-byte receipt so provenance does not reopen checkpoint.

    The receipt is cleared before validation.  A wrong caller record therefore
    burns the only receipt and cannot be retried with different claims.
    """

    loaded = _PENDING_PROVENANCE_LOADED_CHECKPOINT.get()
    _PENDING_PROVENANCE_LOADED_CHECKPOINT.set(None)
    _require(
        isinstance(loaded, _LoadedCheckpointReceipt)
        and loaded._capability is _VERIFIED_LOADED_CHECKPOINT_CAPABILITY,
        "checkpoint provenance requires a live one-shot load receipt",
    )
    _exact_value(source_commit, loaded.source_commit, "provenance source_commit")
    _require(
        isinstance(checkpoint_record, Mapping)
        and set(checkpoint_record)
        == {"role", "absolute_path", "file_sha256", "size_bytes"},
        "provenance checkpoint record keys differ",
    )
    _exact_value(
        checkpoint_record["role"],
        "checkpoint",
        "provenance checkpoint role",
    )
    _exact_value(
        checkpoint_record["absolute_path"],
        loaded.checkpoint_absolute_path,
        "provenance checkpoint path",
    )
    _exact_value(
        checkpoint_record["file_sha256"],
        loaded.checkpoint_sha256,
        "provenance checkpoint hash",
    )
    _exact_value(
        checkpoint_record["size_bytes"],
        loaded.checkpoint_size_bytes,
        "provenance checkpoint size",
    )
    return MappingProxyType(
        {
            "role": "checkpoint",
            "absolute_path": loaded.checkpoint_absolute_path,
            "file_sha256": loaded.checkpoint_sha256,
            "size_bytes": loaded.checkpoint_size_bytes,
            "task_id": loaded.task_id,
            "fixture_id": loaded.fixture_id,
            "source_commit": loaded.source_commit,
            "same_open_file_descriptor_hash_and_load": True,
            "weights_only": True,
        }
    )


__all__ = [
    "CHECKPOINT_EXECUTION_AUTHORIZATION_REPO_RELATIVE_PATH",
    "CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH",
    "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_REPO_RELATIVE_PATH",
    "CheckpointAuthorizationError",
    "CheckpointRuntimeAuthorization",
    "FABLE_EXECUTION_REVIEW_REPO_RELATIVE_PATH",
    "RUNTIME_ENTRYPOINT",
    "TASK20_CHECKPOINT_REPLAY_RESULT_V1_REPO_RELATIVE_PATH",
    "TASK20_CHECKPOINT_REPLAY_RESULT_V3_REPO_RELATIVE_PATH",
    "V2_COLLAPSE_FIELD",
    "authorize_checkpoint_runtime",
    "consume_checkpoint_provenance_load_receipt",
    "require_checkpoint_runtime_authorization",
    "validate_checkpoint_evaluator_authorization",
]
