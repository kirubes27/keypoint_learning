"""Build the committed, checkpoint-blind inputs for hammer-roll replay.

This builder is intentionally standard-library-only apart from importing the
repository's own standard-library corpus and pair validators.  It validates
the complete rendered corpus, the historical cyclic pair index, and every
selected hammer frame/mask/metadata record.  It never resolves, stats, opens,
hashes, imports, or deserializes a checkpoint or any run-directory artifact.

Four purpose-specific manifests are emitted.  The complete roll corpus
inventory is reused directly as the ``replay_dataset_inventory`` provenance
role and is therefore not copied into the replay directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


__representation_import_sha256__ = hashlib.sha256(
    Path(__file__).absolute().read_bytes()
).hexdigest()


from keypoint_net import representation_corpus_inventory as _corpus_inventory
from keypoint_net import representation_splits as _split_validator


def _capture_legacy_loaded_source_digest(module: Any) -> str:
    """Capture unchanged legacy source bytes immediately after import.

    These two modules are already byte-bound by the frozen split manifest, so
    adding a self-hash marker inside either file would invalidate that earlier
    evidence.  The reviewed replay builder records their loaded source paths
    instead and attaches the digest before any replay-manifest work runs.
    """

    source_path = Path(str(module.__file__)).absolute()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    setattr(module, "__representation_import_sha256__", digest)
    return digest


_capture_legacy_loaded_source_digest(_corpus_inventory)
_capture_legacy_loaded_source_digest(_split_validator)

ValidatedCorpusInventory = _corpus_inventory.ValidatedCorpusInventory
validate_corpus_inventory = _corpus_inventory.validate_corpus_inventory
PRIMARY_GROUPS = _split_validator.PRIMARY_GROUPS
validate_source_pair_document = _split_validator.validate_source_pair_document


RUNTIME_SCHEMA_VERSION = "representation_checkpoint_runtime_sources.v2"
PAIR_SCHEMA_VERSION = "representation_checkpoint_replay_pair_index.v1"
FRAME_MASK_SCHEMA_VERSION = (
    "representation_checkpoint_replay_frame_mask_inventory.v1"
)
METADATA_SCHEMA_VERSION = "representation_checkpoint_replay_metadata.v1"

REPLAY_DIRECTORY_RELPATH = PurePosixPath(
    "docs/decisions/2026-07-26/representation_oracle_replay"
)
DATASET_ROOT_RELPATH = PurePosixPath(
    "_tdw_world_z_roll_base_panel_512_v2"
)
CORPUS_INVENTORY_RELPATH = PurePosixPath(
    "docs/decisions/2026-07-26/representation_oracle_splits/"
    "inventories/CORPUS_INVENTORY__roll.json"
)
PAIR_INDEX_RELPATH = PurePosixPath("indices/pairs_skip3_cyclic.json")
OBJECT_ID = "engineers_hammer_vray"
FRAME_COUNT = 180

MANIFEST_FILENAMES = {
    "runtime_source": "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v2.json",
    "pair_index": "HAMMER_ROLL_PAIR_INDEX_MANIFEST_v1.json",
    "frame_mask": "HAMMER_ROLL_FRAME_MASK_INVENTORY_v1.json",
    "metadata": "HAMMER_ROLL_METADATA_MANIFEST_v1.json",
}

EXPECTED_CORPUS_INVENTORY_FILE_SHA256 = (
    "d1b989ac78866be12c06687d9141180845494895f7f03c8d1a3b1438779a8703"
)
EXPECTED_CORPUS_INVENTORY_CONTENT_SHA256 = (
    "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
)
EXPECTED_PAIR_INDEX_SHA256 = (
    "87f51500a080fa964fced92f6382c5d421d74f707c926686d565cabaf9a39fc3"
)
EXPECTED_PAIR_INDEX_SIZE_BYTES = 640562
EXPECTED_DATASET_INDEX_SHA256 = (
    "719645aee0b092d647cbc29a4cd807d7cd7ca4da1ec2608baacf1c2a12224c6b"
)
EXPECTED_DATASET_INDEX_SIZE_BYTES = 9580
EXPECTED_SEMANTIC_LOCK_SHA256 = (
    "c625cc590e8de42b6d8162b044ff6398c0f2e1c04cd6b5ce0cea6f6bccb152aa"
)
EXPECTED_SEMANTIC_LOCK_SIZE_BYTES = 1478
EXPECTED_OBJECT_METADATA_SHA256 = (
    "eb3aed5fd94711e14f63a429f75c32e60ff02364ed3d9881c2723faf6354bf23"
)
EXPECTED_OBJECT_METADATA_SIZE_BYTES = 273632
EXPECTED_OPERATOR_REFERENCE_SHA256 = (
    "668c79a1a1f7ce789293b4355e3130517c567cd40181fbfd8e02237a151ca856"
)
EXPECTED_OPERATOR_REFERENCE_SIZE_BYTES = 504

RUNTIME_ENTRYPOINT = (
    "keypoint_net.representation_checkpoint_runtime."
    "run_authorized_checkpoint_replay"
)
RUNTIME_SOURCE_PATHS = (
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

RUNTIME_PREPROCESSING = {
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

RUNTIME_EXECUTION_BOUNDARY = {
    "cpu_only": True,
    "weights_only": True,
    "same_open_file_descriptor_hash_and_load": True,
    "strict_state_dict": True,
    "model_eval": True,
    "inference_mode": True,
    "task_ids": [20, 55, 80],
    "task55_and_80_require_immutable_task20_v1_audit": True,
    "training_or_weight_update_authorized": False,
    "selection_use_authorized": False,
}

FROZEN_RUNTIME_ENVIRONMENT = {
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "entrypoint",
        "sources",
        "preprocessing",
        "runtime_environment",
        "execution_boundary",
        "content_hash_sha256",
    }
)
_PAIR_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_binding",
        "object_id",
        "pair_index",
        "semantics",
        "content_hash_sha256",
    }
)
_FRAME_MASK_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_binding",
        "object_id",
        "frame_count",
        "records",
        "content_hash_sha256",
    }
)
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_binding",
        "object_id",
        "metadata_files",
        "frame_count",
        "frame_records",
        "semantic_invariants",
        "content_hash_sha256",
    }
)


class CheckpointReplayManifestError(ValueError):
    """Raised when a replay manifest input or output is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointReplayManifestError(message)


def canonical_json_bytes(
    value: Any,
    *,
    trailing_newline: bool = False,
) -> bytes:
    """Return strict sorted compact UTF-8 JSON."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointReplayManifestError(
            f"value cannot be encoded as strict canonical JSON: {exc}"
        ) from exc
    return encoded + (b"\n" if trailing_newline else b"")


def manifest_content_hash(document: Mapping[str, Any]) -> str:
    """Hash a manifest after omitting its top-level content hash."""

    _require(isinstance(document, Mapping), "manifest must be a mapping")
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _with_content_hash(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    _require(
        "content_hash_sha256" not in result,
        "builder input already contains content_hash_sha256",
    )
    result["content_hash_sha256"] = manifest_content_hash(result)
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointReplayManifestError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise CheckpointReplayManifestError(
            f"non-finite JSON number: {token}"
        )
    return value


def _reject_constant(token: str) -> None:
    raise CheckpointReplayManifestError(
        f"non-standard JSON constant: {token}"
    )


def decode_strict_json(data: bytes, *, context: str) -> dict[str, Any]:
    """Decode one strict JSON object, rejecting duplicates and non-finite data."""

    _require(isinstance(data, bytes), f"{context}: input must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointReplayManifestError(
            f"{context}: invalid strict UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"{context}: top level must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    *,
    context: str,
) -> None:
    _require(isinstance(value, Mapping), f"{context}: expected an object")
    actual = set(value)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    _require(
        not missing and not extra,
        f"{context}: keys differ; missing={missing}, extra={extra}",
    )


def _safe_relative_path(value: object, *, context: str) -> PurePosixPath:
    _require(
        isinstance(value, str) and bool(value),
        f"{context}: relative path is empty",
    )
    _require("\\" not in value, f"{context}: backslashes are forbidden")
    _require(":" not in value, f"{context}: colons are forbidden")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and bool(pure.parts)
        and "." not in pure.parts
        and ".." not in pure.parts
        and pure.as_posix() == value,
        f"{context}: unsafe relative path {value!r}",
    )
    return pure


def _regular_file(
    root: Path,
    relative_path: str | PurePosixPath,
    *,
    context: str,
) -> Path:
    pure = _safe_relative_path(
        PurePosixPath(relative_path).as_posix(),
        context=context,
    )
    root_resolved = root.resolve(strict=True)
    cursor = root_resolved
    try:
        for part in pure.parts:
            cursor = cursor / part
            metadata = cursor.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                f"{context}: path contains a symlink",
            )
    except OSError as exc:
        raise CheckpointReplayManifestError(
            f"{context}: path cannot be inspected"
        ) from exc
    _require(
        stat.S_ISREG(metadata.st_mode),
        f"{context}: path is not a regular file",
    )
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise CheckpointReplayManifestError(
            f"{context}: path escapes its root"
        ) from exc
    return resolved


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _file_record(
    root: Path,
    relative_path: str | PurePosixPath,
    *,
    context: str,
) -> dict[str, Any]:
    pure = PurePosixPath(relative_path)
    path = _regular_file(root, pure, context=context)
    digest, size = _sha256_file(path)
    return {
        "relative_path": pure.as_posix(),
        "size_bytes": size,
        "sha256": digest,
    }


def _runtime_environment() -> dict[str, Any]:
    """Return the reviewed target environment without importing any package."""

    return {
        "python_implementation": FROZEN_RUNTIME_ENVIRONMENT[
            "python_implementation"
        ],
        "python_version": FROZEN_RUNTIME_ENVIRONMENT["python_version"],
        "packages": [
            dict(record)
            for record in FROZEN_RUNTIME_ENVIRONMENT["packages"]
        ],
        "torch_inference_device": FROZEN_RUNTIME_ENVIRONMENT[
            "torch_inference_device"
        ],
        "deterministic_algorithms_required": FROZEN_RUNTIME_ENVIRONMENT[
            "deterministic_algorithms_required"
        ],
    }


def _dataset_binding(
    *,
    repo_root: Path,
    dataset_root: Path,
    inventory_bytes: bytes,
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    inventory_file_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    _require(
        inventory_file_sha256 == EXPECTED_CORPUS_INVENTORY_FILE_SHA256,
        "roll corpus inventory file hash differs from the frozen value",
    )
    _require(
        inventory.content_hash_sha256
        == EXPECTED_CORPUS_INVENTORY_CONTENT_SHA256,
        "roll corpus inventory content hash differs from the frozen value",
    )
    _require(
        inventory.root == dataset_root.resolve(strict=True),
        "validated roll inventory binds a different dataset root",
    )
    return {
        "dataset_root_absolute_path": str(dataset_root.resolve(strict=True)),
        "dataset_basename": DATASET_ROOT_RELPATH.name,
        "corpus_inventory": {
            "repo_relative_path": CORPUS_INVENTORY_RELPATH.as_posix(),
            "file_sha256": inventory_file_sha256,
            "content_hash_sha256": inventory.content_hash_sha256,
        },
    }


def _load_validated_inputs(
    repo_root: Path,
    dataset_root: Path,
) -> tuple[
    bytes,
    ValidatedCorpusInventory,
    dict[str, Any],
    list[dict[str, Any]],
]:
    inventory_path = _regular_file(
        repo_root,
        CORPUS_INVENTORY_RELPATH,
        context="roll corpus inventory",
    )
    inventory_bytes = inventory_path.read_bytes()
    try:
        inventory = validate_corpus_inventory(
            inventory_bytes,
            "roll",
            dataset_root,
        )
    except Exception as exc:
        if isinstance(exc, CheckpointReplayManifestError):
            raise
        raise CheckpointReplayManifestError(
            f"roll corpus inventory validation failed: {exc}"
        ) from exc

    pair_path = _regular_file(
        dataset_root,
        PAIR_INDEX_RELPATH,
        context="historical hammer-roll pair index",
    )
    pair_bytes = pair_path.read_bytes()
    pair_digest = hashlib.sha256(pair_bytes).hexdigest()
    _require(
        len(pair_bytes) == EXPECTED_PAIR_INDEX_SIZE_BYTES,
        "historical pair-index size differs from the frozen value",
    )
    _require(
        pair_digest == EXPECTED_PAIR_INDEX_SHA256,
        "historical pair-index hash differs from the frozen value",
    )
    pair_document = decode_strict_json(
        pair_bytes,
        context="historical hammer-roll pair index",
    )
    roll_groups = [
        group
        for group in PRIMARY_GROUPS
        if group.dataset_key == "roll"
        and group.physical_axis == "world_z"
        and group.direction == "forward"
    ]
    _require(
        len(roll_groups) == 1,
        "repository does not define one frozen forward world-Z roll group",
    )
    try:
        validated_pairs = validate_source_pair_document(
            pair_document,
            roll_groups[0],
            dataset_root,
            inventory=inventory,
        )
    except Exception as exc:
        raise CheckpointReplayManifestError(
            f"historical pair-index semantic validation failed: {exc}"
        ) from exc
    return inventory_bytes, inventory, pair_document, validated_pairs


def _build_runtime_source_manifest(repo_root: Path) -> dict[str, Any]:
    sources = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for role, relative_path in RUNTIME_SOURCE_PATHS:
        _require(role not in seen_roles, f"duplicate runtime source role: {role}")
        _require(
            relative_path not in seen_paths,
            f"duplicate runtime source path: {relative_path}",
        )
        record = _file_record(
            repo_root,
            relative_path,
            context=f"runtime source {role}",
        )
        sources.append(
            {
                "role": role,
                "repo_relative_path": record["relative_path"],
                "file_sha256": record["sha256"],
                "import_time_digest_required": True,
            }
        )
        seen_roles.add(role)
        seen_paths.add(relative_path)
    return _with_content_hash(
        {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "artifact_type": "representation_checkpoint_runtime_sources",
            "entrypoint": RUNTIME_ENTRYPOINT,
            "sources": sources,
            "preprocessing": RUNTIME_PREPROCESSING,
            "runtime_environment": _runtime_environment(),
            "execution_boundary": RUNTIME_EXECUTION_BOUNDARY,
        }
    )


def _build_pair_manifest(
    *,
    dataset_binding: Mapping[str, Any],
    pair_document: Mapping[str, Any],
    validated_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        dict(row)
        for row in validated_pairs
        if row.get("model_name") == OBJECT_ID
    ]
    _require(
        len(validated_pairs) == 1080,
        "historical pair index must contain exactly 1080 rows",
    )
    _require(
        len(selected) == FRAME_COUNT,
        "historical pair index must contain exactly 180 hammer rows",
    )
    selected.sort(key=lambda row: int(row["src_frame_index"]))
    _require(
        [row["src_frame_index"] for row in selected]
        == list(range(FRAME_COUNT)),
        "hammer pair sources must be the ordered range 0..179",
    )
    _require(
        all(
            row["dst_frame_index"]
            == (row["src_frame_index"] + 3) % FRAME_COUNT
            for row in selected
        ),
        "hammer pair targets must be source+3 modulo 180",
    )
    _require(
        pair_document.get("operator_name") == "tdw_world_z_roll"
        and pair_document.get("skip_frames") == 3
        and pair_document.get("delta_theta_deg") == 6.0
        and pair_document.get("cyclic") is True,
        "historical pair-index header has wrong roll semantics",
    )
    return _with_content_hash(
        {
            "schema_version": PAIR_SCHEMA_VERSION,
            "artifact_type": "representation_checkpoint_replay_pair_index",
            "dataset_binding": dict(dataset_binding),
            "object_id": OBJECT_ID,
            "pair_index": {
                "relative_path": PAIR_INDEX_RELPATH.as_posix(),
                "size_bytes": EXPECTED_PAIR_INDEX_SIZE_BYTES,
                "sha256": EXPECTED_PAIR_INDEX_SHA256,
            },
            "semantics": {
                "transform_family": "roll",
                "physical_axis": "world_z",
                "operator_name": "tdw_world_z_roll",
                "skip_frames": 3,
                "delta_theta_deg": 6.0,
                "cyclic": True,
                "total_pair_count": 1080,
                "selected_object_pair_count": FRAME_COUNT,
                "source_frame_range": {
                    "first": 0,
                    "last": FRAME_COUNT - 1,
                    "count": FRAME_COUNT,
                    "step": 1,
                },
                "target_frame_formula": (
                    "(source_frame_index + 3) modulo 180"
                ),
                "selected_object_pairs_sha256": hashlib.sha256(
                    canonical_json_bytes(selected)
                ).hexdigest(),
            },
        }
    )


def _inventory_file_record(
    inventory: ValidatedCorpusInventory,
    relative_path: str,
    *,
    context: str,
) -> dict[str, Any]:
    try:
        record = inventory.files_by_relpath[relative_path]
    except KeyError as exc:
        raise CheckpointReplayManifestError(
            f"{context}: path is absent from corpus inventory"
        ) from exc
    _exact_keys(
        record,
        {"relative_path", "size_bytes", "sha256"},
        context=context,
    )
    return {
        "relative_path": record["relative_path"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def _build_frame_mask_manifest(
    *,
    dataset_binding: Mapping[str, Any],
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        frame = inventory.frame(OBJECT_ID, frame_index)
        records.append(
            {
                "frame_index": frame_index,
                "image": _inventory_file_record(
                    inventory,
                    frame["image_relpath"],
                    context=f"hammer frame {frame_index} image",
                ),
                "mask": _inventory_file_record(
                    inventory,
                    frame["mask_relpath"],
                    context=f"hammer frame {frame_index} mask",
                ),
            }
        )
    return _with_content_hash(
        {
            "schema_version": FRAME_MASK_SCHEMA_VERSION,
            "artifact_type": (
                "representation_checkpoint_replay_frame_mask_inventory"
            ),
            "dataset_binding": dict(dataset_binding),
            "object_id": OBJECT_ID,
            "frame_count": FRAME_COUNT,
            "records": records,
        }
    )


def _read_jsonl(path: Path, *, context: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CheckpointReplayManifestError(
            f"{context}: cannot read UTF-8 JSONL"
        ) from exc
    lines = text.splitlines()
    _require(lines, f"{context}: metadata file is empty")
    records = []
    for line_number, line in enumerate(lines, start=1):
        _require(line.strip() != "", f"{context}:{line_number}: blank line")
        records.append(
            decode_strict_json(
                line.encode("utf-8"),
                context=f"{context}:{line_number}",
            )
        )
    return records


def _require_metadata_semantics(
    metadata_rows: Sequence[Mapping[str, Any]],
    inventory: ValidatedCorpusInventory,
) -> None:
    _require(
        len(metadata_rows) == FRAME_COUNT,
        "hammer metadata must contain exactly 180 rows",
    )
    for frame_index, row in enumerate(metadata_rows):
        context = f"hammer metadata row {frame_index + 1}"
        expected_scalars = {
            "frame_index": frame_index,
            "model_name": OBJECT_ID,
            "operator_name": "tdw_world_z_roll",
            "theta_deg": float(2 * frame_index),
            "theta_step_deg": 2.0,
            "rotation_is_tdw_rerendered": True,
            "tdw_axis": "roll",
            "is_world": True,
            "use_centroid": True,
            "valid": True,
        }
        for key, expected in expected_scalars.items():
            _require(
                type(row.get(key)) is type(expected)
                and row.get(key) == expected,
                f"{context}: {key} differs from frozen semantics",
            )
        _require(
            row.get("camera_position") == [0.0, 1.0, 0.56],
            f"{context}: camera_position differs",
        )
        _require(
            row.get("look_at") == [0.0, 1.0, 0.0],
            f"{context}: look_at differs",
        )
        inventory_frame = inventory.frame(OBJECT_ID, frame_index)
        line_hash = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        _require(
            inventory_frame["meta_line_number"] == frame_index + 1
            and inventory_frame["meta_record_sha256"] == line_hash,
            f"{context}: row hash differs from corpus inventory",
        )


def _build_metadata_manifest(
    *,
    repo_root: Path,
    dataset_root: Path,
    dataset_binding: Mapping[str, Any],
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    dataset_index = _file_record(
        dataset_root,
        "dataset_index.json",
        context="roll dataset index",
    )
    semantic_lock = _file_record(
        dataset_root,
        "semantic_lock.json",
        context="roll semantic lock",
    )
    object_metadata = _file_record(
        dataset_root,
        f"train/{OBJECT_ID}/meta.jsonl",
        context="hammer roll metadata",
    )
    operator_reference = _file_record(
        dataset_root,
        "operator_reference.json",
        context="roll operator reference",
    )
    expected_records = (
        (
            dataset_index,
            EXPECTED_DATASET_INDEX_SIZE_BYTES,
            EXPECTED_DATASET_INDEX_SHA256,
            "dataset index",
        ),
        (
            semantic_lock,
            EXPECTED_SEMANTIC_LOCK_SIZE_BYTES,
            EXPECTED_SEMANTIC_LOCK_SHA256,
            "semantic lock",
        ),
        (
            object_metadata,
            EXPECTED_OBJECT_METADATA_SIZE_BYTES,
            EXPECTED_OBJECT_METADATA_SHA256,
            "object metadata",
        ),
        (
            operator_reference,
            EXPECTED_OPERATOR_REFERENCE_SIZE_BYTES,
            EXPECTED_OPERATOR_REFERENCE_SHA256,
            "operator reference",
        ),
    )
    for record, size, digest, name in expected_records:
        _require(
            record["size_bytes"] == size and record["sha256"] == digest,
            f"{name} differs from the frozen size/hash",
        )

    dataset_index_document = decode_strict_json(
        _regular_file(
            dataset_root,
            "dataset_index.json",
            context="roll dataset index",
        ).read_bytes(),
        context="roll dataset index",
    )
    _require(
        dataset_index_document.get("dataset_name")
        == "_tdw_world_z_roll_base_panel_512_v2"
        and dataset_index_document.get("operator_name")
        == "tdw_world_z_roll"
        and dataset_index_document.get("img_size") == 512
        and dataset_index_document.get("frames_per_object") == FRAME_COUNT
        and dataset_index_document.get("n_objects") == 6,
        "roll dataset index has wrong dataset semantics",
    )

    metadata_path = _regular_file(
        dataset_root,
        f"train/{OBJECT_ID}/meta.jsonl",
        context="hammer roll metadata",
    )
    metadata_rows = _read_jsonl(
        metadata_path,
        context="hammer roll metadata",
    )
    _require_metadata_semantics(metadata_rows, inventory)
    frame_records = []
    for frame_index in range(FRAME_COUNT):
        frame = inventory.frame(OBJECT_ID, frame_index)
        frame_records.append(
            {
                "frame_index": frame_index,
                "meta_line_number": frame["meta_line_number"],
                "meta_record_sha256": frame["meta_record_sha256"],
                "physical_state": frame["physical_state"],
            }
        )

    # Keep repo_root in the signature: callers may not redirect the inventory
    # binding to an unrelated repository while retaining a live dataset path.
    expected_inventory_path = (
        repo_root / Path(*CORPUS_INVENTORY_RELPATH.parts)
    ).resolve(strict=True)
    _require(
        dataset_binding["corpus_inventory"]["repo_relative_path"]
        == CORPUS_INVENTORY_RELPATH.as_posix()
        and expected_inventory_path.is_file(),
        "metadata manifest corpus-inventory repository binding differs",
    )
    return _with_content_hash(
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "artifact_type": "representation_checkpoint_replay_metadata",
            "dataset_binding": dict(dataset_binding),
            "object_id": OBJECT_ID,
            "metadata_files": {
                "dataset_index": dataset_index,
                "semantic_lock": semantic_lock,
                "object_metadata": object_metadata,
                "operator_reference": operator_reference,
            },
            "frame_count": FRAME_COUNT,
            "frame_records": frame_records,
            "semantic_invariants": {
                "source_partition": "train",
                "frame_index_range": {
                    "first": 0,
                    "last": FRAME_COUNT - 1,
                    "count": FRAME_COUNT,
                    "step": 1,
                },
                "transform_family": "roll",
                "physical_axis": "world_z",
                "operator_name": "tdw_world_z_roll",
                "theta_degrees": {
                    "first": 0.0,
                    "last": 358.0,
                    "step": 2.0,
                },
                "rotation_is_tdw_rerendered": True,
                "is_world": True,
                "use_centroid": True,
                "all_frames_valid": True,
                "camera_position": [0.0, 1.0, 0.56],
                "look_at": [0.0, 1.0, 0.0],
                "input_size_hw": [512, 512],
            },
        }
    )


def _validate_common_manifest(
    document: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
    expected_schema: str,
    expected_artifact_type: str,
    context: str,
) -> None:
    _exact_keys(document, expected_keys, context=context)
    _require(
        document["schema_version"] == expected_schema,
        f"{context}: wrong schema_version",
    )
    _require(
        document["artifact_type"] == expected_artifact_type,
        f"{context}: wrong artifact_type",
    )
    claimed = document["content_hash_sha256"]
    _require(
        isinstance(claimed, str)
        and _SHA256_RE.fullmatch(claimed) is not None,
        f"{context}: invalid content_hash_sha256",
    )
    _require(
        claimed == manifest_content_hash(document),
        f"{context}: content hash mismatch",
    )


def _validate_dataset_binding_shape(
    value: object,
    *,
    context: str,
) -> None:
    _require(isinstance(value, Mapping), f"{context}: expected an object")
    _exact_keys(
        value,
        {
            "dataset_root_absolute_path",
            "dataset_basename",
            "corpus_inventory",
        },
        context=context,
    )
    expected_root = str(
        (
            Path(__file__).resolve().parents[2]
            / Path(*DATASET_ROOT_RELPATH.parts)
        ).resolve(strict=True)
    )
    _require(
        value["dataset_root_absolute_path"] == expected_root
        and value["dataset_basename"] == DATASET_ROOT_RELPATH.name,
        f"{context}: dataset root or basename differs",
    )
    inventory = value["corpus_inventory"]
    _require(
        isinstance(inventory, Mapping),
        f"{context}: corpus_inventory must be an object",
    )
    _exact_keys(
        inventory,
        {
            "repo_relative_path",
            "file_sha256",
            "content_hash_sha256",
        },
        context=f"{context}.corpus_inventory",
    )
    _require(
        inventory["repo_relative_path"]
        == CORPUS_INVENTORY_RELPATH.as_posix()
        and inventory["file_sha256"]
        == EXPECTED_CORPUS_INVENTORY_FILE_SHA256
        and inventory["content_hash_sha256"]
        == EXPECTED_CORPUS_INVENTORY_CONTENT_SHA256,
        f"{context}: corpus inventory binding differs",
    )


def validate_manifest_shapes(
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate exact output schemas without reading any external artifact."""

    _exact_keys(
        documents,
        set(MANIFEST_FILENAMES),
        context="replay manifest document set",
    )
    runtime = documents["runtime_source"]
    _validate_common_manifest(
        runtime,
        expected_keys=_RUNTIME_KEYS,
        expected_schema=RUNTIME_SCHEMA_VERSION,
        expected_artifact_type="representation_checkpoint_runtime_sources",
        context="runtime source manifest",
    )
    _require(
        runtime["entrypoint"] == RUNTIME_ENTRYPOINT,
        "runtime source manifest: entrypoint differs",
    )
    _require(
        runtime["preprocessing"] == RUNTIME_PREPROCESSING,
        "runtime source manifest: preprocessing differs",
    )
    _require(
        runtime["execution_boundary"] == RUNTIME_EXECUTION_BOUNDARY,
        "runtime source manifest: execution boundary differs",
    )
    _require(
        runtime["runtime_environment"] == _runtime_environment(),
        "runtime source manifest: runtime environment differs",
    )
    sources = runtime["sources"]
    _require(
        isinstance(sources, list)
        and len(sources) == len(RUNTIME_SOURCE_PATHS),
        "runtime source manifest: wrong source count",
    )
    for index, (record, expected) in enumerate(
        zip(sources, RUNTIME_SOURCE_PATHS)
    ):
        _exact_keys(
            record,
            {
                "role",
                "repo_relative_path",
                "file_sha256",
                "import_time_digest_required",
            },
            context=f"runtime source manifest source {index}",
        )
        _require(
            record["role"] == expected[0]
            and record["repo_relative_path"] == expected[1]
            and record["import_time_digest_required"] is True
            and isinstance(record["file_sha256"], str)
            and _SHA256_RE.fullmatch(record["file_sha256"]) is not None,
            f"runtime source manifest source {index}: frozen fields differ",
        )

    pair = documents["pair_index"]
    _validate_common_manifest(
        pair,
        expected_keys=_PAIR_KEYS,
        expected_schema=PAIR_SCHEMA_VERSION,
        expected_artifact_type=(
            "representation_checkpoint_replay_pair_index"
        ),
        context="pair-index manifest",
    )
    _require(
        pair["object_id"] == OBJECT_ID,
        "pair-index manifest: object differs",
    )
    _validate_dataset_binding_shape(
        pair["dataset_binding"],
        context="pair-index manifest dataset binding",
    )
    _require(
        pair["pair_index"]
        == {
            "relative_path": PAIR_INDEX_RELPATH.as_posix(),
            "size_bytes": EXPECTED_PAIR_INDEX_SIZE_BYTES,
            "sha256": EXPECTED_PAIR_INDEX_SHA256,
        },
        "pair-index manifest: file binding differs",
    )
    semantics = pair["semantics"]
    _require(
        isinstance(semantics, Mapping),
        "pair-index manifest: semantics must be an object",
    )
    _exact_keys(
        semantics,
        {
            "transform_family",
            "physical_axis",
            "operator_name",
            "skip_frames",
            "delta_theta_deg",
            "cyclic",
            "total_pair_count",
            "selected_object_pair_count",
            "source_frame_range",
            "target_frame_formula",
            "selected_object_pairs_sha256",
        },
        context="pair-index manifest semantics",
    )
    selected_hash = semantics["selected_object_pairs_sha256"]
    expected_semantics = {
        "transform_family": "roll",
        "physical_axis": "world_z",
        "operator_name": "tdw_world_z_roll",
        "skip_frames": 3,
        "delta_theta_deg": 6.0,
        "cyclic": True,
        "total_pair_count": 1080,
        "selected_object_pair_count": FRAME_COUNT,
        "source_frame_range": {
            "first": 0,
            "last": FRAME_COUNT - 1,
            "count": FRAME_COUNT,
            "step": 1,
        },
        "target_frame_formula": "(source_frame_index + 3) modulo 180",
        "selected_object_pairs_sha256": selected_hash,
    }
    _require(
        semantics == expected_semantics
        and isinstance(selected_hash, str)
        and _SHA256_RE.fullmatch(selected_hash) is not None,
        "pair-index manifest: frozen semantics differ",
    )

    frame_mask = documents["frame_mask"]
    _validate_common_manifest(
        frame_mask,
        expected_keys=_FRAME_MASK_KEYS,
        expected_schema=FRAME_MASK_SCHEMA_VERSION,
        expected_artifact_type=(
            "representation_checkpoint_replay_frame_mask_inventory"
        ),
        context="frame-mask manifest",
    )
    _require(
        frame_mask["object_id"] == OBJECT_ID
        and frame_mask["frame_count"] == FRAME_COUNT
        and isinstance(frame_mask["records"], list)
        and len(frame_mask["records"]) == FRAME_COUNT,
        "frame-mask manifest: object/count differs",
    )
    _validate_dataset_binding_shape(
        frame_mask["dataset_binding"],
        context="frame-mask manifest dataset binding",
    )
    for frame_index, record in enumerate(frame_mask["records"]):
        _exact_keys(
            record,
            {"frame_index", "image", "mask"},
            context=f"frame-mask manifest record {frame_index}",
        )
        _require(
            record["frame_index"] == frame_index,
            f"frame-mask manifest record {frame_index}: wrong frame index",
        )
        for kind in ("image", "mask"):
            _exact_keys(
                record[kind],
                {"relative_path", "size_bytes", "sha256"},
                context=(
                    f"frame-mask manifest record {frame_index} {kind}"
                ),
            )
            _safe_relative_path(
                record[kind]["relative_path"],
                context=(
                    f"frame-mask manifest record {frame_index} {kind}"
                ),
            )
            _require(
                isinstance(record[kind]["size_bytes"], int)
                and not isinstance(record[kind]["size_bytes"], bool)
                and record[kind]["size_bytes"] >= 0
                and isinstance(record[kind]["sha256"], str)
                and _SHA256_RE.fullmatch(record[kind]["sha256"]) is not None,
                f"frame-mask manifest record {frame_index}: invalid file record",
            )
            expected_path = (
                f"train/{OBJECT_ID}/frames/a/img_{frame_index:04d}.png"
                if kind == "image"
                else f"train/{OBJECT_ID}/masks/a/mask_{frame_index:04d}.png"
            )
            _require(
                record[kind]["relative_path"] == expected_path,
                f"frame-mask manifest record {frame_index}: wrong {kind} path",
            )

    metadata = documents["metadata"]
    _validate_common_manifest(
        metadata,
        expected_keys=_METADATA_KEYS,
        expected_schema=METADATA_SCHEMA_VERSION,
        expected_artifact_type="representation_checkpoint_replay_metadata",
        context="metadata manifest",
    )
    _require(
        metadata["object_id"] == OBJECT_ID
        and metadata["frame_count"] == FRAME_COUNT
        and isinstance(metadata["frame_records"], list)
        and len(metadata["frame_records"]) == FRAME_COUNT,
        "metadata manifest: object/count differs",
    )
    _validate_dataset_binding_shape(
        metadata["dataset_binding"],
        context="metadata manifest dataset binding",
    )
    metadata_files = metadata["metadata_files"]
    _require(
        isinstance(metadata_files, Mapping),
        "metadata manifest: metadata_files must be an object",
    )
    _exact_keys(
        metadata_files,
        {
            "dataset_index",
            "semantic_lock",
            "object_metadata",
            "operator_reference",
        },
        context="metadata manifest metadata_files",
    )
    expected_metadata_files = {
        "dataset_index": (
            "dataset_index.json",
            EXPECTED_DATASET_INDEX_SIZE_BYTES,
            EXPECTED_DATASET_INDEX_SHA256,
        ),
        "semantic_lock": (
            "semantic_lock.json",
            EXPECTED_SEMANTIC_LOCK_SIZE_BYTES,
            EXPECTED_SEMANTIC_LOCK_SHA256,
        ),
        "object_metadata": (
            f"train/{OBJECT_ID}/meta.jsonl",
            EXPECTED_OBJECT_METADATA_SIZE_BYTES,
            EXPECTED_OBJECT_METADATA_SHA256,
        ),
        "operator_reference": (
            "operator_reference.json",
            EXPECTED_OPERATOR_REFERENCE_SIZE_BYTES,
            EXPECTED_OPERATOR_REFERENCE_SHA256,
        ),
    }
    for role, expected in expected_metadata_files.items():
        _exact_keys(
            metadata_files[role],
            {"relative_path", "size_bytes", "sha256"},
            context=f"metadata manifest {role}",
        )
        _require(
            metadata_files[role]
            == {
                "relative_path": expected[0],
                "size_bytes": expected[1],
                "sha256": expected[2],
            },
            f"metadata manifest: {role} binding differs",
        )
    for frame_index, record in enumerate(metadata["frame_records"]):
        _exact_keys(
            record,
            {
                "frame_index",
                "meta_line_number",
                "meta_record_sha256",
                "physical_state",
            },
            context=f"metadata manifest frame record {frame_index}",
        )
        _require(
            record["frame_index"] == frame_index
            and record["meta_line_number"] == frame_index + 1
            and isinstance(record["meta_record_sha256"], str)
            and _SHA256_RE.fullmatch(record["meta_record_sha256"]) is not None,
            f"metadata manifest frame record {frame_index}: identity differs",
        )
    _require(
        metadata["semantic_invariants"]
        == {
            "source_partition": "train",
            "frame_index_range": {
                "first": 0,
                "last": FRAME_COUNT - 1,
                "count": FRAME_COUNT,
                "step": 1,
            },
            "transform_family": "roll",
            "physical_axis": "world_z",
            "operator_name": "tdw_world_z_roll",
            "theta_degrees": {
                "first": 0.0,
                "last": 358.0,
                "step": 2.0,
            },
            "rotation_is_tdw_rerendered": True,
            "is_world": True,
            "use_centroid": True,
            "all_frames_valid": True,
            "camera_position": [0.0, 1.0, 0.56],
            "look_at": [0.0, 1.0, 0.0],
            "input_size_hw": [512, 512],
        },
        "metadata manifest: semantic invariants differ",
    )

    dataset_bindings = [
        documents[key]["dataset_binding"]
        for key in ("pair_index", "frame_mask", "metadata")
    ]
    _require(
        dataset_bindings[0] == dataset_bindings[1] == dataset_bindings[2],
        "purpose-specific manifests bind different datasets",
    )


def build_checkpoint_replay_manifest_documents(
    *,
    repo_root: str | os.PathLike[str] | None = None,
    dataset_root: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and fully validate the four checkpoint-blind replay manifests."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReplayManifestError(
            f"repository root does not exist: {root}"
        ) from exc
    _require(root.is_dir(), "repository root must be a directory")
    expected_root = Path(__file__).resolve().parents[2]
    _require(
        root == expected_root,
        "production builder cannot be redirected to another repository",
    )
    resolved_dataset_root = (
        Path(dataset_root)
        if dataset_root is not None
        else root / Path(*DATASET_ROOT_RELPATH.parts)
    )
    try:
        resolved_dataset_root = resolved_dataset_root.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReplayManifestError(
            f"dataset root does not exist: {resolved_dataset_root}"
        ) from exc
    expected_dataset_root = (
        root / Path(*DATASET_ROOT_RELPATH.parts)
    ).resolve(strict=True)
    _require(
        resolved_dataset_root == expected_dataset_root,
        "production builder cannot be redirected to another dataset",
    )

    (
        inventory_bytes,
        inventory,
        pair_document,
        validated_pairs,
    ) = _load_validated_inputs(root, resolved_dataset_root)
    dataset_binding = _dataset_binding(
        repo_root=root,
        dataset_root=resolved_dataset_root,
        inventory_bytes=inventory_bytes,
        inventory=inventory,
    )
    documents = {
        "runtime_source": _build_runtime_source_manifest(root),
        "pair_index": _build_pair_manifest(
            dataset_binding=dataset_binding,
            pair_document=pair_document,
            validated_pairs=validated_pairs,
        ),
        "frame_mask": _build_frame_mask_manifest(
            dataset_binding=dataset_binding,
            inventory=inventory,
        ),
        "metadata": _build_metadata_manifest(
            repo_root=root,
            dataset_root=resolved_dataset_root,
            dataset_binding=dataset_binding,
            inventory=inventory,
        ),
    }
    validate_manifest_shapes(documents)
    return documents


def encoded_manifest_documents(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    """Encode a validated document set as canonical JSON plus one newline."""

    validate_manifest_shapes(documents)
    return {
        MANIFEST_FILENAMES[key]: canonical_json_bytes(
            documents[key],
            trailing_newline=True,
        )
        for key in MANIFEST_FILENAMES
    }


def write_checkpoint_replay_manifests(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Create absent manifests exclusively and byte-check existing artifacts."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    ).resolve(strict=True)
    _require(
        root == Path(__file__).resolve().parents[2],
        "production writer cannot be redirected to another repository",
    )
    output_directory = (
        root / Path(*REPLAY_DIRECTORY_RELPATH.parts)
    ).resolve(strict=True)
    encoded = encoded_manifest_documents(documents)
    return _create_absent_or_verify_existing_manifests(
        output_directory=output_directory,
        encoded=encoded,
    )


def _create_absent_or_verify_existing_manifests(
    *,
    output_directory: Path,
    encoded: Mapping[str, bytes],
) -> dict[str, str]:
    """Never rewrite an existing manifest, even when its bytes are identical."""

    verified: dict[str, str] = {}
    for filename in sorted(encoded):
        path = output_directory / filename
        expected = encoded[filename]
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o644)
            except OSError as exc:
                raise CheckpointReplayManifestError(
                    f"cannot exclusively create replay manifest: {filename}"
                ) from exc
            try:
                written = 0
                while written < len(expected):
                    written += os.write(descriptor, expected[written:])
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise CheckpointReplayManifestError(
                f"cannot inspect existing replay manifest: {filename}"
            ) from exc
        else:
            _require(
                stat.S_ISREG(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode),
                "existing replay manifest is not a regular non-symlink file: "
                f"{filename}",
            )
            actual = path.read_bytes()
            _require(
                actual == expected,
                f"refusing to overwrite differing replay manifest: {filename}",
            )
        verified[filename] = hashlib.sha256(expected).hexdigest()
    return verified


def check_checked_in_manifests(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Require checked-in bytes to equal a fresh deterministic build."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    ).resolve(strict=True)
    _require(
        root == Path(__file__).resolve().parents[2],
        "production checker cannot be redirected to another repository",
    )
    output_directory = (
        root / Path(*REPLAY_DIRECTORY_RELPATH.parts)
    ).resolve(strict=True)
    encoded = encoded_manifest_documents(documents)
    verified: dict[str, str] = {}
    for filename in sorted(encoded):
        path = _regular_file(
            output_directory,
            filename,
            context=f"checked-in replay manifest {filename}",
        )
        actual = path.read_bytes()
        _require(
            actual == encoded[filename],
            f"checked-in replay manifest differs from fresh build: {filename}",
        )
        verified[filename] = hashlib.sha256(actual).hexdigest()
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or check the four checkpoint-blind hammer-roll replay "
            "manifests."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    documents = build_checkpoint_replay_manifest_documents()
    result = (
        write_checkpoint_replay_manifests(documents)
        if arguments.write
        else check_checked_in_manifests(documents)
    )
    print(
        json.dumps(
            {
                "status": (
                    "checkpoint_blind_replay_manifests_created_or_verified_"
                    "without_overwrite"
                    if arguments.write
                    else "checkpoint_blind_replay_manifests_verified"
                ),
                "checkpoint_or_run_artifact_touched": False,
                "manifest_file_sha256": result,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
