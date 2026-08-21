"""Fail-closed registry gate for saved representation-checkpoint replays.

The registry validator in this module is deliberately checkpoint-blind.  It
parses and validates only the committed JSON registry; it does not stat, open,
hash, import, or load any path named by a checkpoint record.

Checkpoint bytes may be hashed only by :func:`preflight_checkpoint_candidates`.
That API requires a capability-bearing provenance receipt minted by
:func:`verify_committed_registry`.  The receipt proves that the exact registry
bytes were already committed and still match the working tree.  Neither API
imports a model runtime or deserializes weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


__representation_import_sha256__ = hashlib.sha256(
    Path(__file__).absolute().read_bytes()
).hexdigest()


REGISTRY_SCHEMA_VERSION = "representation-checkpoint-replay-registry-v1"
REGISTRY_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/"
    "REPLAY_REGISTRY_v1.json"
)
GEOMETRY_REGISTRY_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_geometry/"
    "GEOMETRY_BINDING_REGISTRY_v1.json"
)
HAMMER_ROLL_GEOMETRY_BINDING_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
    "engineers_hammer_vray__roll__world_z__v1.json"
)

# This is the SHA-256 of canonical sorted compact JSON after removing the
# top-level content_hash_sha256 field.  It locks every value in the registry,
# including fields not otherwise repeated as readable constants below.
EXPECTED_REGISTRY_CONTENT_HASH_SHA256 = (
    "02612a4dd10863d0dc78a2b2e078689aa588d91b99e667d452fa0c9b2ac45c75"
)
EXPECTED_GEOMETRY_REGISTRY_FILE_SHA256 = (
    "22b55959199f76a8056897ba1e291b388fa754e80cca407dc172b8d6a70f880e"
)
EXPECTED_GEOMETRY_REGISTRY_CONTENT_HASH_SHA256 = (
    "27353548998ccbe5346eaa84ba92b30423d955e6d643bb6b53b1995715343fbb"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})
_VERIFIED_PROVENANCE_CAPABILITY = object()

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "date",
        "status",
        "purpose",
        "execution_boundary",
        "replay_rule",
        "dataset_binding",
        "definition_identical_fields",
        "fixtures",
        "content_hash_sha256",
    }
)

_EXPECTED_EXECUTION_BOUNDARY = {
    "registry_must_be_committed_before_checkpoint_hash_or_load": True,
    "checkpoint_candidate_path_is_not_confirmed_until_post_commit_hash_preflight": True,
    "hash_mismatch_action": "stop_without_loading_checkpoint",
    "training_or_weight_update_authorized": False,
    "selection_use_authorized": False,
    "official_replay_requires_fable_reviewed_execution_commit": True,
}

_EXPECTED_GEOMETRY_EXECUTION_BOUNDARY = {
    "planted_synthetic_geometry_authorized": True,
    "dataset_backed_geometry_authorized": True,
    "saved_checkpoint_hash_preflight_authorized": True,
    "saved_checkpoint_replay_authorized": False,
    "training_authorized": False,
    "unregistered_geometry_rejected": True,
}

_EXPECTED_HAMMER_ROLL_INVENTORY_CONTENT_HASH_SHA256 = (
    "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
)
_EXPECTED_HAMMER_ROLL_SEMANTIC_LOCK_SHA256 = (
    "c625cc590e8de42b6d8162b044ff6398c0f2e1c04cd6b5ce0cea6f6bccb152aa"
)
_EXPECTED_HAMMER_ROLL_EVIDENCE_ROLES = [
    "roll_corpus_inventory",
    "dataset_semantic_lock",
    "verified_generator_snapshot_manifest",
    "verified_roll_render_driver",
    "verified_roll_operator_and_recentering_source",
    "render_camera_and_object_transform_metadata",
    "mask_pivot_sign_checker_source",
    "rendered_mask_pivot_and_sign_report",
]

_EXPECTED_REPLAY_RULE = {
    "scope": "definition_identical_historical_fields_only",
    "predicate": (
        "absolute_difference <= 2e-6 OR relative_difference <= 5e-4"
    ),
    "absolute_difference_max": 0.000002,
    "relative_difference_max": 0.0005,
    "zero_reference_relative_difference": "not_used",
    "new_metric_policy": (
        "New all-channel collapse, absolute-activity, attachment, switching, "
        "and revised canonical-normalization fields are classified by their "
        "frozen new definitions and are not numerically forced to equal "
        "historical eligible-only fields."
    ),
}

_EXPECTED_DATASET_BINDING = {
    "object_id": "engineers_hammer_vray",
    "replay_root_absolute_path": (
        "/Users/kirubeso.r/Documents/PhD/keypoint_preoperator_gates/"
        "_tdw_world_z_roll_base_panel_512_v2"
    ),
    "training_config_declared_data_root": (
        "./_tdw_world_z_roll_base_panel_512_v2"
    ),
    "training_config_declared_pair_index": (
        "_tdw_world_z_roll_base_panel_512_v2/indices/"
        "pairs_skip3_cyclic.json"
    ),
    "corpus_inventory": {
        "repo_relative_path": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "inventories/CORPUS_INVENTORY__roll.json"
        ),
        "artifact_commit": "78071297ba46bb22b637a00e1c141eb6e9a8f2de",
        "file_sha256": (
            "d1b989ac78866be12c06687d9141180845494895f7f03c8d1a3b1438779a8703"
        ),
        "content_hash_sha256": (
            "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
        ),
    },
    "historical_pair_index": {
        "relative_path_under_replay_root": "indices/pairs_skip3_cyclic.json",
        "sha256": (
            "87f51500a080fa964fced92f6382c5d421d74f707c926686d565cabaf9a39fc3"
        ),
        "size_bytes": 640562,
        "split_semantics": (
            "historical full cyclic plus-3-frame index; not the new train/test "
            "split"
        ),
    },
    "semantic_lock": {
        "relative_path_under_replay_root": "semantic_lock.json",
        "sha256": (
            "c625cc590e8de42b6d8162b044ff6398c0f2e1c04cd6b5ce0cea6f6bccb152aa"
        ),
        "size_bytes": 1478,
    },
    "live_content_requirement": (
        "The committed corpus inventory and historical pair-index hash must "
        "validate against the replay root before checkpoint loading."
    ),
}

_MIGRATED_PREOPERATOR_ROOT = Path(
    "/Users/kirubeso.r/Documents/PhD/code/worktrees/active/preoperator-gates"
)
_LEGACY_PREOPERATOR_ROOT = Path(
    "/Users/kirubeso.r/Documents/PhD/keypoint_preoperator_gates"
)


def _resolve_migrated_operational_path(path: Path) -> Path:
    """Resolve one retired live prefix without rewriting frozen provenance."""
    try:
        relative = path.relative_to(_LEGACY_PREOPERATOR_ROOT)
    except ValueError:
        return path
    return _MIGRATED_PREOPERATOR_ROOT / relative

_EXPECTED_DEFINITION_IDENTICAL_FIELDS = {
    "operator": [
        {
            "historical_json": "rollout/rollout_metrics.json",
            "historical_pointer": "/operator_spectrum/shared_affine/A",
            "production_pointer": "/operator/learned_A",
            "comparison": "elementwise_replay_rule",
        },
        {
            "historical_json": "rollout/rollout_metrics.json",
            "historical_pointer": "/operator_spectrum/shared_affine/bias",
            "production_pointer": "/operator/learned_b",
            "comparison": "elementwise_replay_rule",
        },
        {
            "historical_json": "rollout/rollout_metrics.json",
            "historical_pointer": "/operator_spectrum/shared_affine/det_A",
            "production_pointer": "/operator/determinant_A",
            "comparison": "replay_rule",
        },
        {
            "historical_json": "rollout/rollout_metrics.json",
            "historical_pointer": (
                "/operator_spectrum/shared_affine/singular_values"
            ),
            "production_pointer": "/operator/singular_values",
            "comparison": "elementwise_replay_rule",
        },
        {
            "historical_json": "rollout/rollout_metrics.json",
            "historical_pointer": (
                "/operator_spectrum/shared_affine/"
                "closest_rotation_angle_deg"
            ),
            "production_pointer": "/operator/proper_rotation_angle_deg",
            "comparison": "replay_rule",
        },
    ],
    "cyclic_rollout": {
        "historical_json": "rollout/rollout_metrics.json",
        "historical_pointer_pattern": (
            "/cyclic_compositionality/errors/{k}/{field}"
        ),
        "production_location": {
            "intermediate_horizons": {
                "list_pointer": (
                    "/rollout/full_corpus_identity_normalized_auc/horizons"
                ),
                "selector_field": "k",
                "first": 1,
                "last": 59,
            },
            "closure": {
                "record_pointer": "/rollout/closure",
                "k": 60,
            },
        },
        "production_metric_subpointer_pattern": (
            "/model_mse/{mapped_field}"
        ),
        "k_values": {"first": 1, "last": 60, "closure_k": 60},
        "field_mapping": {
            "mean": "mean",
            "min": "minimum",
            "max": "maximum",
            "n_samples": "n",
        },
        "comparison": "numeric_fields_replay_rule_and_n_exact",
        "forbidden_historical_field": "sem",
    },
}

_RUN_ROOT = (
    "/Users/kirubeso.r/Documents/PhD/cluster_downloads/"
    "hammer_full360_shared_complete/keypoint_net/runs_hammer_full360_shared"
)

_EXPECTED_FIXTURES = (
    {
        "fixture_id": "saved_task20_seed42_collapsed_negative_control",
        "task_id": 20,
        "role": "collapsed_negative_control",
        "expected_new_classification": {
            "structural_negative_control_collapse": True,
            "candidate_eligibility": "forbidden",
        },
        "run_directory": (
            f"{_RUN_ROOT}/"
            "phase_a_engineers_hammer_vray_20260606_050434_151501_seed42_"
            "pid2116969"
        ),
        "checkpoint_sha256": (
            "96433168767659ae9144a35a4f7889c3226b65a7a0c5341197d48232a66fe622"
        ),
        "checkpoint_size_bytes": 2994210,
        "config_sha256": (
            "60db767a5a77da28d001ef0ceb4134d063b749053255955629d6780b278d8f71"
        ),
        "config_size_bytes": 1020,
        "metadata_sha256": (
            "4c5f786f6e531fe7f24b36d7cda61419371f4be8f01dc622377931d41458a199"
        ),
        "metadata_size_bytes": 63319,
        "historical_outputs": (
            (
                "rollout/rollout_metrics.json",
                "d2a9bf8255faf824d1069144cd3493cdbc24e66e5fd4df768e9affbd1e7308fe",
                35212,
            ),
            (
                "visualizations/operator_metrics.json",
                "d7705b7503fceeeb762b8028c09c8d3290dc6b85e8acaf71f7272c656d631b3b",
                4142,
            ),
            (
                "compositionality/compositionality_results.json",
                "4e44d4c4e9411a5399e9462d8c34d109238e1de1d47e7d4888d3e414d665dc2e",
                2358,
            ),
            (
                "ablations/ablation_results.json",
                "05502fc1e2a64c276356c3f737f55180b40dc90bb1c4495736a8990ea2f1db59",
                212,
            ),
        ),
    },
    {
        "fixture_id": "saved_task55_seed42_clean_baseline",
        "task_id": 55,
        "role": "clean_baseline_replay_only",
        "expected_new_classification": {
            "structural_negative_control_collapse": False,
            "candidate_eligibility": "replay_only_not_selection",
        },
        "run_directory": (
            f"{_RUN_ROOT}/"
            "phase_a_engineers_hammer_vray_20260606_131311_378814_seed42_"
            "pid606096"
        ),
        "checkpoint_sha256": (
            "942a32082b6bfe83526253cc2dd39e49792b8260d12fc1361a11bae812992418"
        ),
        "checkpoint_size_bytes": 2991906,
        "config_sha256": (
            "9c9bbeb64ba7c79593724700fe6ece92196303d4eb50c05cfaefb0099d5704f8"
        ),
        "config_size_bytes": 1019,
        "metadata_sha256": (
            "c1e2b1fc2cb369fa1d28d6fcef1a682e58a09fbcdeaf922dd7a945b52d4d48e6"
        ),
        "metadata_size_bytes": 56044,
        "historical_outputs": (
            (
                "rollout/rollout_metrics.json",
                "e8ebda12dc8dc04443c77ed272ef5861bbd01dfeba9936e6caefb9ea6d25a2c0",
                35225,
            ),
            (
                "visualizations/operator_metrics.json",
                "8da7a68443096c708b5b5f5050dd9f74fb34f3a417f2a0231c8a8bdcb4b5b056",
                4169,
            ),
            (
                "compositionality/compositionality_results.json",
                "791720ebe2efd18961db40cd9d667b91c3e1e78c60017ca47e5267a56ebf5d14",
                2357,
            ),
            (
                "ablations/ablation_results.json",
                "bb7bbc7e56252d36e90c536e9b0a4e06b561c764d47cb24ba2cd0a0123aab96c",
                210,
            ),
        ),
    },
    {
        "fixture_id": "saved_task80_seed42_assisted_baseline",
        "task_id": 80,
        "role": "assisted_baseline_replay_only",
        "expected_new_classification": {
            "structural_negative_control_collapse": False,
            "candidate_eligibility": "replay_only_not_selection",
        },
        "run_directory": (
            f"{_RUN_ROOT}/"
            "phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_"
            "pid3289780"
        ),
        "checkpoint_sha256": (
            "ccb613c87788a229929b1f6ead002d626819e970237d0e0ad5ca75c9318004f3"
        ),
        "checkpoint_size_bytes": 2994210,
        "config_sha256": (
            "e1fea4fc99c23637f47cb73f904b0f45c330425e15e327ce07a34d59212af9d7"
        ),
        "config_size_bytes": 1020,
        "metadata_sha256": (
            "9371a65d949cdb1388b8a511142ecc6af1c96ef49d3b7a772a5591f206d19d65"
        ),
        "metadata_size_bytes": 63233,
        "historical_outputs": (
            (
                "rollout/rollout_metrics.json",
                "580fcbca702b54091d6ffb20c9302140fec00bcde80039f2ec618330de4d2b86",
                35169,
            ),
            (
                "visualizations/operator_metrics.json",
                "3ca1ba900d80026625fefea9375de12c5486a413c7afa4ccb0f903ce66b3880c",
                4161,
            ),
            (
                "compositionality/compositionality_results.json",
                "da7258c126a92bf44650f068e16ab30cc989b8e67994bc5998d090ed3ad98f84",
                2357,
            ),
            (
                "ablations/ablation_results.json",
                "a62455e91a5a2e1509d6a97eb46823d46cc787f839c1105ea795e0749f91e433",
                212,
            ),
        ),
    },
)


class ReplayRegistryError(ValueError):
    """Raised when the saved-checkpoint replay boundary is not satisfied."""


@dataclass(frozen=True)
class CheckpointCandidate:
    """Registry metadata only; constructing this object never touches its path."""

    fixture_id: str
    task_id: int
    role: str
    absolute_path: str
    relative_path: str
    expected_sha256: str
    expected_size_bytes: int


@dataclass(frozen=True)
class ValidatedReplayRegistry:
    """Receipt proving strict validation of registry bytes, not checkpoint bytes."""

    registry_absolute_path: str | None
    registry_file_sha256: str
    content_hash_sha256: str
    schema_version: str
    checkpoint_candidates: tuple[CheckpointCandidate, ...]


@dataclass(frozen=True)
class VerifiedRegistryProvenanceReceipt:
    """Capability receipt binding one validated registry to one Git commit."""

    source_commit: str
    repository_root: str
    registry_absolute_path: str
    registry_file_sha256: str
    registry_content_hash_sha256: str
    git_blob_oid: str
    source_commit_verified: bool = True
    _capability: object = field(
        default=None,
        repr=False,
        compare=False,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayRegistryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical byte encoding used for content hashes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayRegistryError(
            f"value cannot be canonicalized as strict JSON: {exc}"
        ) from exc


def replay_registry_content_hash(document: Mapping[str, Any]) -> str:
    """Hash a registry mapping with its top-level content hash omitted."""

    _require(isinstance(document, Mapping), "registry must be a mapping")
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayRegistryError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ReplayRegistryError(f"non-standard/non-finite JSON constant: {value}")


def _decode_strict_json(data: bytes) -> dict[str, Any]:
    _require(isinstance(data, bytes), "registry input must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayRegistryError(
            f"registry is not strict UTF-8 JSON: {exc}"
        ) from exc
    _require(isinstance(value, dict), "registry top level must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    context: str,
) -> None:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    actual = set(value)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    _require(
        not missing and not extra,
        f"{context} keys differ: missing={missing}, extra={extra}",
    )


def _exact_value(actual: Any, expected: Any, context: str) -> None:
    """Compare recursively without Python's bool/int or int/float coercions."""

    _require(
        type(actual) is type(expected),
        f"{context} has wrong JSON type: expected "
        f"{type(expected).__name__}, got {type(actual).__name__}",
    )
    if isinstance(expected, dict):
        _exact_keys(actual, set(expected), context)
        for key, expected_value in expected.items():
            _exact_value(actual[key], expected_value, f"{context}.{key}")
        return
    if isinstance(expected, list):
        _require(
            len(actual) == len(expected),
            f"{context} length differs: expected {len(expected)}, got {len(actual)}",
        )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _exact_value(actual_item, expected_item, f"{context}[{index}]")
        return
    if isinstance(expected, float):
        _require(math.isfinite(actual), f"{context} must be finite")
    _require(actual == expected, f"{context} differs from the frozen value")


def _validate_fixture(
    fixture: Any,
    expected: Mapping[str, Any],
    *,
    index: int,
) -> CheckpointCandidate:
    context = f"fixtures[{index}]"
    _exact_keys(
        fixture,
        {
            "fixture_id",
            "task_id",
            "role",
            "expected_new_classification",
            "run_directory",
            "checkpoint",
            "config",
            "metadata",
            "historical_outputs",
        },
        context,
    )
    for field_name in (
        "fixture_id",
        "task_id",
        "role",
        "expected_new_classification",
        "run_directory",
    ):
        _exact_value(
            fixture[field_name],
            expected[field_name],
            f"{context}.{field_name}",
        )

    checkpoint = fixture["checkpoint"]
    _exact_keys(
        checkpoint,
        {
            "relative_path",
            "candidate_absolute_path",
            "expected_sha256",
            "candidate_size_bytes",
            "verification_status",
        },
        f"{context}.checkpoint",
    )
    expected_checkpoint_path = (
        f"{expected['run_directory']}/best_model.pt"
    )
    _exact_value(
        checkpoint["relative_path"],
        "best_model.pt",
        f"{context}.checkpoint.relative_path",
    )
    _exact_value(
        checkpoint["candidate_absolute_path"],
        expected_checkpoint_path,
        f"{context}.checkpoint.candidate_absolute_path",
    )
    _exact_value(
        checkpoint["expected_sha256"],
        expected["checkpoint_sha256"],
        f"{context}.checkpoint.expected_sha256",
    )
    _exact_value(
        checkpoint["candidate_size_bytes"],
        expected["checkpoint_size_bytes"],
        f"{context}.checkpoint.candidate_size_bytes",
    )
    _exact_value(
        checkpoint["verification_status"],
        "not_hashed_or_loaded_in_this_registry_draft",
        f"{context}.checkpoint.verification_status",
    )

    for record_name, relative_path, sha_field, size_field in (
        ("config", "config.json", "config_sha256", "config_size_bytes"),
        ("metadata", "history.json", "metadata_sha256", "metadata_size_bytes"),
    ):
        record = fixture[record_name]
        _exact_keys(
            record,
            {"relative_path", "sha256", "size_bytes"},
            f"{context}.{record_name}",
        )
        _exact_value(
            record,
            {
                "relative_path": relative_path,
                "sha256": expected[sha_field],
                "size_bytes": expected[size_field],
            },
            f"{context}.{record_name}",
        )

    historical_outputs = fixture["historical_outputs"]
    _require(
        isinstance(historical_outputs, list),
        f"{context}.historical_outputs must be a list",
    )
    expected_outputs = expected["historical_outputs"]
    _require(
        len(historical_outputs) == len(expected_outputs),
        f"{context}.historical_outputs must contain exactly "
        f"{len(expected_outputs)} records",
    )
    for output_index, (record, expected_record) in enumerate(
        zip(historical_outputs, expected_outputs)
    ):
        output_context = f"{context}.historical_outputs[{output_index}]"
        _exact_value(
            record,
            {
                "relative_path": expected_record[0],
                "sha256": expected_record[1],
                "size_bytes": expected_record[2],
            },
            output_context,
        )

    return CheckpointCandidate(
        fixture_id=fixture["fixture_id"],
        task_id=fixture["task_id"],
        role=fixture["role"],
        absolute_path=checkpoint["candidate_absolute_path"],
        relative_path=checkpoint["relative_path"],
        expected_sha256=checkpoint["expected_sha256"],
        expected_size_bytes=checkpoint["candidate_size_bytes"],
    )


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON pointer without fallback or coercion."""

    _require(isinstance(pointer, str), "JSON pointer must be a string")
    if pointer == "":
        return document
    _require(pointer.startswith("/"), f"invalid JSON pointer: {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        _require(
            re.search(r"~(?:[^01]|$)", raw_token) is None,
            f"invalid JSON pointer escape in {pointer!r}",
        )
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            _require(
                token in current,
                f"JSON pointer does not resolve: {pointer!r}",
            )
            current = current[token]
        elif isinstance(current, list):
            _require(
                token == "0" or (token.isdigit() and not token.startswith("0")),
                f"JSON pointer list index is invalid: {pointer!r}",
            )
            index = int(token)
            _require(
                0 <= index < len(current),
                f"JSON pointer list index is out of range: {pointer!r}",
            )
            current = current[index]
        else:
            raise ReplayRegistryError(
                f"JSON pointer traverses a scalar: {pointer!r}"
            )
    return current


def _require_finite_number(value: Any, context: str) -> None:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must resolve to a finite number",
    )


def validate_production_pointer_contract(
    evaluator_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every frozen production pointer resolves against evaluator output.

    This check is intended for a deterministic synthetic evaluator result.  It
    reads only the supplied mapping and never accesses registry or checkpoint
    paths.
    """

    _require(
        isinstance(evaluator_result, Mapping),
        "evaluator result must be a mapping",
    )
    resolved_count = 0
    resolved_operator_pointers: list[str] = []
    for record in _EXPECTED_DEFINITION_IDENTICAL_FIELDS["operator"]:
        pointer = record["production_pointer"]
        value = resolve_json_pointer(evaluator_result, pointer)
        if pointer.endswith("/learned_A"):
            _require(
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(row, list) and len(row) == 2 for row in value),
                f"{pointer} must resolve to a 2x2 matrix",
            )
            for row in value:
                for item in row:
                    _require_finite_number(item, pointer)
        elif pointer.endswith("/learned_b") or pointer.endswith(
            "/singular_values"
        ):
            _require(
                isinstance(value, list) and len(value) == 2,
                f"{pointer} must resolve to a length-two vector",
            )
            for item in value:
                _require_finite_number(item, pointer)
        else:
            _require_finite_number(value, pointer)
        resolved_operator_pointers.append(pointer)
        resolved_count += 1

    rollout_contract = _EXPECTED_DEFINITION_IDENTICAL_FIELDS["cyclic_rollout"]
    location = rollout_contract["production_location"]
    intermediate = location["intermediate_horizons"]
    horizons = resolve_json_pointer(
        evaluator_result,
        intermediate["list_pointer"],
    )
    _require(
        isinstance(horizons, list),
        "intermediate rollout pointer must resolve to a list",
    )
    selector = intermediate["selector_field"]
    records_by_k: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(horizons):
        _require(
            isinstance(record, Mapping),
            f"intermediate rollout record {index} must be an object",
        )
        k = record.get(selector)
        _require(
            isinstance(k, int) and not isinstance(k, bool),
            f"intermediate rollout record {index} has invalid {selector}",
        )
        _require(k not in records_by_k, f"duplicate intermediate rollout k={k}")
        records_by_k[k] = record
    expected_intermediate = set(
        range(int(intermediate["first"]), int(intermediate["last"]) + 1)
    )
    _require(
        set(records_by_k) == expected_intermediate,
        "intermediate rollout horizons differ from frozen k=1..59",
    )

    closure_contract = location["closure"]
    closure = resolve_json_pointer(
        evaluator_result,
        closure_contract["record_pointer"],
    )
    _require(isinstance(closure, Mapping), "closure pointer must resolve to an object")
    _require(
        closure.get(selector) == closure_contract["k"],
        "closure record must have frozen k=60",
    )

    field_mapping = rollout_contract["field_mapping"]
    metric_pattern = rollout_contract["production_metric_subpointer_pattern"]
    resolved_rollout_fields: list[str] = []
    for k in range(
        int(rollout_contract["k_values"]["first"]),
        int(rollout_contract["k_values"]["last"]) + 1,
    ):
        record = (
            records_by_k[k]
            if k != int(rollout_contract["k_values"]["closure_k"])
            else closure
        )
        for historical_field, mapped_field in field_mapping.items():
            subpointer = metric_pattern.format(mapped_field=mapped_field)
            value = resolve_json_pointer(record, subpointer)
            context = f"k={k}{subpointer}"
            if historical_field == "n_samples":
                _require(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0,
                    f"{context} must resolve to a positive integer",
                )
            else:
                _require_finite_number(value, context)
            resolved_rollout_fields.append(context)
            resolved_count += 1

    return {
        "operator_pointers": tuple(resolved_operator_pointers),
        "rollout_fields": tuple(resolved_rollout_fields),
        "resolved_value_count": resolved_count,
        "intermediate_horizon_count": len(records_by_k),
        "closure_k": closure_contract["k"],
        "status": "all_frozen_production_pointers_resolved",
    }


def validate_replay_registry_bytes(
    data: bytes,
    *,
    registry_absolute_path: str | None = None,
) -> ValidatedReplayRegistry:
    """Validate registry bytes while remaining completely checkpoint-blind."""

    document = _decode_strict_json(data)
    _exact_keys(document, _TOP_LEVEL_KEYS, "registry")

    claimed_hash = document["content_hash_sha256"]
    _require(
        isinstance(claimed_hash, str)
        and _SHA256_RE.fullmatch(claimed_hash) is not None,
        "registry.content_hash_sha256 must be lowercase 64-hex",
    )
    recomputed_hash = replay_registry_content_hash(document)
    _require(
        claimed_hash == recomputed_hash,
        "registry content hash mismatch: "
        f"claimed {claimed_hash}, recomputed {recomputed_hash}",
    )
    _require(
        recomputed_hash == EXPECTED_REGISTRY_CONTENT_HASH_SHA256,
        "registry payload does not match the frozen replay registry",
    )

    fixed_scalars = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": "representation_checkpoint_replay_registry",
        "date": "2026-07-27",
        "status": "preregistered_candidate_paths_checkpoint_bytes_not_opened",
        "purpose": (
            "Bind the saved Task 20, 55, and 80 replay inputs before any "
            "checkpoint is opened by the production replay path."
        ),
    }
    for field_name, expected in fixed_scalars.items():
        _exact_value(
            document[field_name],
            expected,
            f"registry.{field_name}",
        )
    _exact_value(
        document["execution_boundary"],
        _EXPECTED_EXECUTION_BOUNDARY,
        "registry.execution_boundary",
    )
    _exact_value(
        document["replay_rule"],
        _EXPECTED_REPLAY_RULE,
        "registry.replay_rule",
    )
    _exact_value(
        document["dataset_binding"],
        _EXPECTED_DATASET_BINDING,
        "registry.dataset_binding",
    )
    _exact_value(
        document["definition_identical_fields"],
        _EXPECTED_DEFINITION_IDENTICAL_FIELDS,
        "registry.definition_identical_fields",
    )

    fixtures = document["fixtures"]
    _require(isinstance(fixtures, list), "registry.fixtures must be a list")
    _require(
        len(fixtures) == len(_EXPECTED_FIXTURES),
        "registry.fixtures must contain exactly Task 20, Task 55, and Task 80",
    )
    candidates = tuple(
        _validate_fixture(fixture, expected, index=index)
        for index, (fixture, expected) in enumerate(
            zip(fixtures, _EXPECTED_FIXTURES)
        )
    )
    _require(
        tuple(candidate.task_id for candidate in candidates) == (20, 55, 80),
        "registry fixture task order must be exactly 20, 55, 80",
    )
    _require(
        len({candidate.absolute_path for candidate in candidates})
        == len(candidates),
        "registry checkpoint candidate paths must be distinct",
    )
    _require(
        len({candidate.expected_sha256 for candidate in candidates})
        == len(candidates),
        "registry checkpoint hashes must be distinct",
    )

    normalized_path: str | None = None
    if registry_absolute_path is not None:
        path = Path(registry_absolute_path)
        _require(path.is_absolute(), "registry path must be absolute")
        normalized_path = str(path)

    return ValidatedReplayRegistry(
        registry_absolute_path=normalized_path,
        registry_file_sha256=hashlib.sha256(data).hexdigest(),
        content_hash_sha256=recomputed_hash,
        schema_version=REGISTRY_SCHEMA_VERSION,
        checkpoint_candidates=candidates,
    )


def _regular_file_bytes(path: Path, *, context: str) -> tuple[Path, bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReplayRegistryError(f"{context} cannot be inspected: {path}") from exc
    _require(not stat.S_ISLNK(metadata.st_mode), f"{context} must not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{context} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
        data = resolved.read_bytes()
    except OSError as exc:
        raise ReplayRegistryError(f"{context} cannot be read: {path}") from exc
    return resolved, data


def load_and_validate_replay_registry(
    path: str | os.PathLike[str] | None = None,
) -> ValidatedReplayRegistry:
    """Read and validate only the replay registry, never its candidate paths."""

    if path is None:
        path = Path(__file__).resolve().parents[1] / REGISTRY_REPO_RELATIVE_PATH
    resolved, data = _regular_file_bytes(Path(path), context="replay registry")
    return validate_replay_registry_bytes(
        data,
        registry_absolute_path=str(resolved),
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
        suffix = f": {detail}" if detail else ""
        raise ReplayRegistryError(f"{context}{suffix}") from exc
    return result.stdout


def _validated_repo_root(repo_root: Path) -> Path:
    try:
        resolved = repo_root.resolve(strict=True)
    except OSError as exc:
        raise ReplayRegistryError(
            f"repository root does not exist: {repo_root}"
        ) from exc
    _require(resolved.is_dir(), "repository root must be a directory")
    reported = _git(
        resolved,
        ["rev-parse", "--show-toplevel"],
        context="cannot resolve Git repository root",
    ).decode("utf-8").strip()
    _require(
        Path(reported).resolve(strict=True) == resolved,
        "repository root is not the Git top level",
    )
    return resolved


def _verify_committed_registry_for_repo(
    registry: ValidatedReplayRegistry,
    *,
    source_commit: str,
    repo_root: Path,
) -> VerifiedRegistryProvenanceReceipt:
    """Test seam for the public post-commit provenance verifier."""

    _require(
        isinstance(registry, ValidatedReplayRegistry),
        "registry must be a validated replay-registry receipt",
    )
    _require(
        isinstance(source_commit, str)
        and _COMMIT_RE.fullmatch(source_commit) is not None,
        "source_commit must be full lowercase 40-hex",
    )
    root = _validated_repo_root(repo_root)
    _git(
        root,
        ["cat-file", "-e", f"{source_commit}^{{commit}}"],
        context="source_commit is not an existing commit",
    )

    expected_path = (root / REGISTRY_REPO_RELATIVE_PATH).resolve(strict=False)
    _require(
        registry.registry_absolute_path == str(expected_path),
        "validated registry does not bind the repository replay-registry path",
    )
    current_registry = load_and_validate_replay_registry(expected_path)
    _require(
        current_registry == registry,
        "registry bytes changed after validation",
    )

    tree_line = _git(
        root,
        ["ls-tree", source_commit, "--", REGISTRY_REPO_RELATIVE_PATH],
        context="cannot inspect committed registry blob",
    ).decode("utf-8").strip()
    _require(tree_line != "", "registry is absent from source_commit")
    try:
        metadata, reported_path = tree_line.split("\t", 1)
        mode, object_type, blob_oid = metadata.split(" ", 2)
    except ValueError as exc:
        raise ReplayRegistryError("unexpected git ls-tree output") from exc
    _require(
        reported_path == REGISTRY_REPO_RELATIVE_PATH,
        "Git reported the wrong registry path",
    )
    _require(mode in _REGULAR_GIT_MODES, "committed registry is not a regular file")
    _require(object_type == "blob", "committed registry is not a Git blob")
    _require(
        _COMMIT_RE.fullmatch(blob_oid) is not None,
        "committed registry blob id is malformed",
    )
    blob = _git(
        root,
        ["show", f"{source_commit}:{REGISTRY_REPO_RELATIVE_PATH}"],
        context="cannot read committed registry blob",
    )
    _require(
        hashlib.sha256(blob).hexdigest() == registry.registry_file_sha256,
        "committed registry blob SHA-256 differs from validated registry",
    )
    current_path, current_bytes = _regular_file_bytes(
        expected_path,
        context="working replay registry",
    )
    _require(
        blob == current_bytes,
        "working replay registry bytes differ from source_commit",
    )

    return VerifiedRegistryProvenanceReceipt(
        source_commit=source_commit,
        repository_root=str(root),
        registry_absolute_path=str(current_path),
        registry_file_sha256=registry.registry_file_sha256,
        registry_content_hash_sha256=registry.content_hash_sha256,
        git_blob_oid=blob_oid,
        source_commit_verified=True,
        _capability=_VERIFIED_PROVENANCE_CAPABILITY,
    )


def verify_committed_registry(
    registry: ValidatedReplayRegistry,
    *,
    source_commit: str,
) -> VerifiedRegistryProvenanceReceipt:
    """Mint the provenance receipt required by checkpoint hash preflight.

    This function verifies only Git and registry bytes.  It does not inspect
    any checkpoint candidate.  Production callers cannot redirect it to a
    caller-selected repository.
    """

    repo_root = Path(__file__).resolve().parents[1]
    return _verify_committed_registry_for_repo(
        registry,
        source_commit=source_commit,
        repo_root=repo_root,
    )


def _require_preflight_authorization(
    registry: ValidatedReplayRegistry,
    provenance: VerifiedRegistryProvenanceReceipt,
) -> None:
    _require(
        isinstance(provenance, VerifiedRegistryProvenanceReceipt)
        and provenance._capability is _VERIFIED_PROVENANCE_CAPABILITY,
        "checkpoint preflight requires a verified registry provenance receipt",
    )
    _require(
        provenance.source_commit_verified is True,
        "registry provenance receipt is not verified",
    )
    _require(
        registry.registry_absolute_path == provenance.registry_absolute_path,
        "provenance receipt binds a different registry path",
    )
    _require(
        registry.registry_file_sha256 == provenance.registry_file_sha256,
        "provenance receipt binds different registry bytes",
    )
    _require(
        registry.content_hash_sha256
        == provenance.registry_content_hash_sha256,
        "provenance receipt binds a different registry payload",
    )


def _hash_checkpoint_candidate(candidate: CheckpointCandidate) -> dict[str, Any]:
    """Hash one checkpoint as opaque bytes; never import or deserialize it."""

    path = Path(candidate.absolute_path)
    _require(path.is_absolute(), "checkpoint candidate path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReplayRegistryError(
            f"checkpoint candidate cannot be inspected: {path}"
        ) from exc
    _require(
        not stat.S_ISLNK(metadata.st_mode),
        f"checkpoint candidate must not be a symlink: {path}",
    )
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"checkpoint candidate must be a regular file: {path}",
    )
    _require(
        metadata.st_size == candidate.expected_size_bytes,
        f"checkpoint size mismatch for Task {candidate.task_id}; "
        "stop without loading checkpoint",
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise ReplayRegistryError(
            f"checkpoint candidate cannot be hashed: {path}"
        ) from exc
    actual_sha256 = digest.hexdigest()
    _require(
        size == candidate.expected_size_bytes,
        f"checkpoint size changed while hashing Task {candidate.task_id}; "
        "stop without loading checkpoint",
    )
    _require(
        actual_sha256 == candidate.expected_sha256,
        f"checkpoint hash mismatch for Task {candidate.task_id}; "
        "stop without loading checkpoint",
    )
    return {
        "fixture_id": candidate.fixture_id,
        "task_id": candidate.task_id,
        "role": candidate.role,
        "absolute_path": candidate.absolute_path,
        "size_bytes": size,
        "sha256": actual_sha256,
        "verification_status": "hashed_as_opaque_bytes_not_loaded",
    }


def _require_saved_checkpoint_replay_geometry_authorization(
    *,
    source_commit: str,
) -> None:
    """Reject checkpoint touches until exact reviewed roll geometry permits hashing.

    This gate reads only the small committed geometry registry.  In particular,
    it runs before any checkpoint path is inspected, statted, opened, or hashed.
    The frozen registry authorizes opaque checkpoint hashing only.  It
    deliberately keeps checkpoint loading/replay and training blocked.
    """

    _require(
        isinstance(source_commit, str)
        and _COMMIT_RE.fullmatch(source_commit) is not None,
        "geometry authorization requires a full source commit",
    )
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = repo_root / GEOMETRY_REGISTRY_REPO_RELATIVE_PATH
    resolved, data = _regular_file_bytes(
        registry_path,
        context="geometry registry",
    )
    _require(
        resolved == registry_path.resolve(strict=True),
        "geometry registry path differs from the frozen repository path",
    )
    _require(
        hashlib.sha256(data).hexdigest()
        == EXPECTED_GEOMETRY_REGISTRY_FILE_SHA256,
        "geometry registry file hash mismatch",
    )
    document = _decode_strict_json(data)
    claimed_content_hash = document.get("content_hash_sha256")
    _require(
        isinstance(claimed_content_hash, str)
        and _SHA256_RE.fullmatch(claimed_content_hash) is not None,
        "geometry registry content hash is invalid",
    )
    content_payload = dict(document)
    content_payload.pop("content_hash_sha256")
    computed_content_hash = hashlib.sha256(
        canonical_json_bytes(content_payload)
    ).hexdigest()
    _require(
        claimed_content_hash == computed_content_hash
        == EXPECTED_GEOMETRY_REGISTRY_CONTENT_HASH_SHA256,
        "geometry registry content hash mismatch",
    )
    boundary = document.get("execution_boundary")
    _exact_value(
        boundary,
        _EXPECTED_GEOMETRY_EXECUTION_BOUNDARY,
        "geometry registry execution_boundary",
    )
    committed_bytes = _git(
        repo_root,
        [
            "show",
            f"{source_commit}:{GEOMETRY_REGISTRY_REPO_RELATIVE_PATH}",
        ],
        context="cannot read geometry registry from source_commit",
    )
    _require(
        committed_bytes == data,
        "working geometry registry differs from source_commit",
    )

    _require(
        boundary["saved_checkpoint_hash_preflight_authorized"] is True,
        "checkpoint hashing is blocked by the reviewed geometry registry",
    )
    _require(
        boundary["saved_checkpoint_replay_authorized"] is False,
        "geometry registry must not authorize checkpoint loading or replay",
    )
    _require(
        boundary["training_authorized"] is False,
        "geometry registry must not authorize training",
    )
    _require(
        boundary["dataset_backed_geometry_authorized"] is True,
        "checkpoint hashing is blocked: dataset-backed roll geometry is not authorized",
    )
    bindings = document.get("registered_bindings")
    _require(
        isinstance(bindings, list),
        "geometry registry registered_bindings must be a list",
    )
    matching_roll_bindings = [
        record
        for record in bindings
        if isinstance(record, Mapping)
        and record.get("transform_family") == "roll"
        and record.get("object_id") == "engineers_hammer_vray"
    ]
    _require(
        len(matching_roll_bindings) == 1,
        "checkpoint hashing requires one registered hammer roll geometry binding",
    )
    record = matching_roll_bindings[0]
    _exact_keys(
        record,
        {
            "transform_family",
            "object_id",
            "inventory_content_hash_sha256",
            "repo_relative_path",
            "file_sha256",
            "content_hash_sha256",
            "derivation_method",
            "evidence_roles",
        },
        "registered hammer roll geometry binding",
    )
    _exact_value(
        record["inventory_content_hash_sha256"],
        _EXPECTED_HAMMER_ROLL_INVENTORY_CONTENT_HASH_SHA256,
        "registered hammer roll inventory hash",
    )
    _exact_value(
        record["repo_relative_path"],
        HAMMER_ROLL_GEOMETRY_BINDING_REPO_RELATIVE_PATH,
        "registered hammer roll binding path",
    )
    _require(
        isinstance(record["file_sha256"], str)
        and _SHA256_RE.fullmatch(record["file_sha256"]) is not None,
        "registered hammer roll binding file hash is invalid",
    )
    _require(
        isinstance(record["content_hash_sha256"], str)
        and _SHA256_RE.fullmatch(record["content_hash_sha256"]) is not None,
        "registered hammer roll binding content hash is invalid",
    )
    _require(
        isinstance(record["derivation_method"], str)
        and bool(record["derivation_method"]),
        "registered hammer roll derivation method is empty",
    )
    _exact_value(
        record["evidence_roles"],
        _EXPECTED_HAMMER_ROLL_EVIDENCE_ROLES,
        "registered hammer roll evidence roles",
    )

    binding_path = repo_root / HAMMER_ROLL_GEOMETRY_BINDING_REPO_RELATIVE_PATH
    binding_resolved, binding_data = _regular_file_bytes(
        binding_path,
        context="registered hammer roll geometry binding",
    )
    _require(
        binding_resolved == binding_path.resolve(strict=True),
        "hammer roll geometry binding path differs from registered path",
    )
    _require(
        hashlib.sha256(binding_data).hexdigest() == record["file_sha256"],
        "registered hammer roll geometry binding file hash mismatch",
    )
    committed_binding = _git(
        repo_root,
        [
            "show",
            f"{source_commit}:{HAMMER_ROLL_GEOMETRY_BINDING_REPO_RELATIVE_PATH}",
        ],
        context="cannot read hammer roll geometry binding from source_commit",
    )
    _require(
        committed_binding == binding_data,
        "working hammer roll geometry binding differs from source_commit",
    )

    binding = _decode_strict_json(binding_data)
    _exact_keys(
        binding,
        {
            "schema_version",
            "artifact_type",
            "object_id",
            "transform_family",
            "inventory_content_hash_sha256",
            "dataset_semantic_lock_sha256",
            "estimator_geometry",
            "parameters",
            "evidence_files",
            "content_hash_sha256",
        },
        "hammer roll geometry binding",
    )
    content_payload = dict(binding)
    claimed_binding_content_hash = content_payload.pop(
        "content_hash_sha256",
        None,
    )
    computed_binding_content_hash = hashlib.sha256(
        canonical_json_bytes(content_payload)
    ).hexdigest()
    _require(
        claimed_binding_content_hash
        == computed_binding_content_hash
        == record["content_hash_sha256"],
        "registered hammer roll geometry binding content hash mismatch",
    )
    _exact_value(
        binding["schema_version"],
        "representation_geometry_binding.v1",
        "hammer roll geometry schema",
    )
    _exact_value(
        binding["artifact_type"],
        "representation_geometry_binding",
        "hammer roll geometry artifact type",
    )
    _exact_value(
        binding["object_id"],
        "engineers_hammer_vray",
        "hammer roll geometry object",
    )
    _exact_value(
        binding["transform_family"],
        "roll",
        "hammer roll geometry family",
    )
    _exact_value(
        binding["inventory_content_hash_sha256"],
        _EXPECTED_HAMMER_ROLL_INVENTORY_CONTENT_HASH_SHA256,
        "hammer roll geometry inventory hash",
    )
    _exact_value(
        binding["dataset_semantic_lock_sha256"],
        _EXPECTED_HAMMER_ROLL_SEMANTIC_LOCK_SHA256,
        "hammer roll geometry semantic-lock hash",
    )
    _exact_value(
        binding["estimator_geometry"],
        {
            "coordinate_system": (
                "normalized_minus1_plus1_endpoint_aligned_xy"
            ),
            "image_y_direction": "down",
            "crop": {"mode": "none"},
            "resize": [512, 512],
            "align_corners": None,
        },
        "hammer roll estimator geometry",
    )
    _exact_value(
        binding["parameters"],
        {"projected_centre_xy": [0.0, 0.0]},
        "hammer roll projected centre",
    )
    evidence_files = binding["evidence_files"]
    _require(
        isinstance(evidence_files, list),
        "hammer roll geometry evidence_files must be a list",
    )
    _exact_value(
        [item.get("role") if isinstance(item, Mapping) else None for item in evidence_files],
        _EXPECTED_HAMMER_ROLL_EVIDENCE_ROLES,
        "hammer roll geometry evidence-file roles",
    )
    for index, evidence in enumerate(evidence_files):
        _exact_keys(
            evidence,
            {"role", "absolute_path", "sha256"},
            f"hammer roll geometry evidence {index}",
        )
        evidence_path_value = evidence["absolute_path"]
        _require(
            isinstance(evidence_path_value, str)
            and Path(evidence_path_value).is_absolute(),
            f"hammer roll geometry evidence {index} path is invalid",
        )
        evidence_resolved, evidence_data = _regular_file_bytes(
            _resolve_migrated_operational_path(Path(evidence_path_value)),
            context=f"hammer roll geometry evidence {index}",
        )
        try:
            evidence_relative = evidence_resolved.relative_to(repo_root)
        except ValueError as exc:
            raise ReplayRegistryError(
                f"hammer roll geometry evidence {index} lies outside repository"
            ) from exc
        _require(
            isinstance(evidence["sha256"], str)
            and _SHA256_RE.fullmatch(evidence["sha256"]) is not None
            and hashlib.sha256(evidence_data).hexdigest() == evidence["sha256"],
            f"hammer roll geometry evidence {index} hash mismatch",
        )
        committed_evidence = _git(
            repo_root,
            ["show", f"{source_commit}:{evidence_relative.as_posix()}"],
            context=(
                f"cannot read hammer roll geometry evidence {index} "
                "from source_commit"
            ),
        )
        _require(
            committed_evidence == evidence_data,
            f"hammer roll geometry evidence {index} differs from source_commit",
        )


def preflight_checkpoint_candidates(
    registry: ValidatedReplayRegistry,
    *,
    provenance: VerifiedRegistryProvenanceReceipt,
) -> dict[str, Any]:
    """Post-commit opaque-byte preflight; this is the first checkpoint touch.

    Callers must first load the registry, commit it, and mint ``provenance`` via
    :func:`verify_committed_registry`.  That receipt is necessary but not
    sufficient: the exact same source commit must also contain a reviewed
    geometry registry that authorizes saved hammer-roll replay.  A missing,
    stale, forged, or geometry-blocked receipt is rejected before any checkpoint
    path is inspected.
    """

    _require(
        isinstance(registry, ValidatedReplayRegistry),
        "registry must be a validated replay-registry receipt",
    )
    _require_preflight_authorization(registry, provenance)

    # Revalidate the committed registry immediately before the first checkpoint
    # path is touched, closing the validate-then-edit gap.
    current = load_and_validate_replay_registry(registry.registry_absolute_path)
    _require(current == registry, "registry changed after provenance verification")
    _require_saved_checkpoint_replay_geometry_authorization(
        source_commit=provenance.source_commit,
    )
    results = tuple(
        _hash_checkpoint_candidate(candidate)
        for candidate in registry.checkpoint_candidates
    )
    return {
        "source_commit": provenance.source_commit,
        "registry_file_sha256": registry.registry_file_sha256,
        "registry_content_hash_sha256": registry.content_hash_sha256,
        "checkpoint_verification": results,
        "checkpoint_hash_preflight_passed": True,
        "checkpoint_load_authorized_by_this_receipt": False,
        "remaining_load_gate": (
            "separate_fable_reviewed_execution_commit_and_runtime_provenance"
        ),
        "training_or_weight_update_authorized": False,
        "selection_use_authorized": False,
    }


__all__ = [
    "CheckpointCandidate",
    "EXPECTED_GEOMETRY_REGISTRY_CONTENT_HASH_SHA256",
    "EXPECTED_GEOMETRY_REGISTRY_FILE_SHA256",
    "EXPECTED_REGISTRY_CONTENT_HASH_SHA256",
    "GEOMETRY_REGISTRY_REPO_RELATIVE_PATH",
    "REGISTRY_REPO_RELATIVE_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "ReplayRegistryError",
    "ValidatedReplayRegistry",
    "VerifiedRegistryProvenanceReceipt",
    "canonical_json_bytes",
    "load_and_validate_replay_registry",
    "preflight_checkpoint_candidates",
    "replay_registry_content_hash",
    "resolve_json_pointer",
    "validate_replay_registry_bytes",
    "validate_production_pointer_contract",
    "verify_committed_registry",
]
