"""
PyTorch Dataset for Phase A training.

Loads consecutive frame pairs (x_t, x_{t+1}) from the TDW-generated dataset.
"""

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


_STRICT_INDEX_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_basename",
        "dataset_binding_sha256",
        "dataset_semantic_lock_sha256",
        "source_pair_index_relpath",
        "source_pair_index_sha256",
        "unversioned_source",
        "generator",
        "split",
        "object_roles",
        "included_objects",
        "transform",
        "frame_partition",
        "pair_count",
        "pair_counts_by_object",
        "pairs",
        "content_hash_sha256",
    }
)

_STRICT_INDEX_TRANSFORM_KEYS = frozenset(
    {
        "family",
        "physical_axis",
        "direction",
        "signed_generator",
        "generator_units",
        "stride",
        "stride_units",
        "cyclic",
        "expected_2d_family",
    }
)

_STRICT_INDEX_GENERATOR_KEYS = frozenset({"name", "commit", "config_sha256"})
_STRICT_INDEX_FRAME_PARTITION_KEYS = frozenset(
    {
        "train_frame_indices",
        "holdout_frame_indices",
        "guard_frame_indices",
    }
)

_STRICT_INDEX_PAIR_KEYS = frozenset(
    {
        "pair_id",
        "model_name",
        "object_role",
        "split",
        "transform_family",
        "physical_axis",
        "direction",
        "signed_generator",
        "generator_units",
        "stride",
        "stride_units",
        "cyclic",
        "src_frame_index",
        "dst_frame_index",
        "src_state",
        "dst_state",
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    }
)

_INDEX_SPLITS = frozenset({"train", "validation", "test"})
_SHA256_HEX_LENGTH = 64
_FROZEN_OBJECT_ROLES = {
    "engineers_hammer_vray": "development",
    "b03_banana_01_high": "confirmation",
    "kettle": "confirmation",
    "dewalt_compact_drill_vray": "final_test",
    "b03_trumpet_vray": "final_test",
    "toy_monkey_medium": "final_test",
}
_FROZEN_INCLUDED_OBJECTS_BY_SPLIT = {
    "train": tuple(_FROZEN_OBJECT_ROLES),
    "validation": ("engineers_hammer_vray",),
    "test": tuple(
        name
        for name, role in _FROZEN_OBJECT_ROLES.items()
        if role in {"confirmation", "final_test"}
    ),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field)
    if (
        len(text) != _SHA256_HEX_LENGTH
        or text != text.lower()
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return text


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be a finite real scalar")
    return float(value)


def _require_finite_json(value: Any, field: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            _require_finite_json(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} contains unsupported JSON value {type(value).__name__}")


def _resolve_index_path(
    data_root: Path,
    index_path: str,
    *,
    strict_metadata: bool,
) -> Path:
    requested = Path(index_path).expanduser()
    if requested.is_absolute():
        candidate = requested
    elif strict_metadata:
        candidate = data_root / requested
    elif requested.exists():
        candidate = requested
    else:
        candidate = data_root / requested
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Pair index not found: '{index_path}' (resolved to {candidate})"
        )
    return candidate.resolve()


def _resolve_dataset_file(
    data_root: Path,
    relpath: Any,
    field: str,
    *,
    require_exists: bool,
) -> Path:
    text = _require_nonempty_string(relpath, field)
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to data_root, got: {text}")
    resolved_root = data_root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes data_root: {text}") from exc
    if require_exists and not resolved.is_file():
        raise FileNotFoundError(f"{field} does not resolve to a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class IndexPairManifest:
    """Validated pair-index metadata without opening any image or mask."""

    data_root: Path
    index_path: Path
    index_sha256: str
    content_hash_sha256: Optional[str]
    metadata: dict[str, Any]
    pairs: tuple[dict[str, Any], ...]
    source_endpoint_ids: frozenset[str]
    target_endpoint_ids: frozenset[str]
    endpoint_ids: frozenset[str]
    strict_metadata: bool

    @property
    def split(self) -> Optional[str]:
        value = self.metadata.get("split")
        return value if isinstance(value, str) else None

    @property
    def transform(self) -> dict[str, Any]:
        value = self.metadata.get("transform", {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    @property
    def dataset_binding_sha256(self) -> Optional[str]:
        value = self.metadata.get("dataset_binding_sha256")
        return value if isinstance(value, str) else None


def inspect_index_pair_manifest(
    data_root: str,
    index_path: str,
    *,
    expected_split: Optional[str] = None,
    object_name: Optional[str] = None,
    strict_metadata: bool = False,
    expected_index_sha256: Optional[str] = None,
    expected_dataset_binding_sha256: Optional[str] = None,
    validate_paths: bool = True,
) -> IndexPairManifest:
    """
    Validate a pair index and expose transform-complete metadata.

    This function deliberately does not construct a Dataset and does not open
    image or mask contents. It is therefore safe for fixed-final preflight,
    where the test Dataset must remain unopened until training is complete.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root is not a directory: {root}")
    resolved_index = _resolve_index_path(
        root, index_path, strict_metadata=strict_metadata
    )
    index_bytes = resolved_index.read_bytes()
    index_sha256 = _sha256_bytes(index_bytes)
    if expected_index_sha256 is not None:
        expected = _require_sha256(
            expected_index_sha256, "expected_index_sha256"
        )
        if index_sha256 != expected:
            raise ValueError(
                f"Pair-index SHA-256 changed for {resolved_index}: "
                f"expected {expected}, found {index_sha256}"
            )
    try:
        index_value = json.loads(index_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pair index is not valid UTF-8 JSON: {resolved_index}") from exc

    if strict_metadata:
        if not isinstance(index_value, dict):
            raise ValueError("Strict pair index must be a top-level JSON object")
        actual_keys = frozenset(index_value)
        if actual_keys != _STRICT_INDEX_TOP_LEVEL_KEYS:
            missing = sorted(_STRICT_INDEX_TOP_LEVEL_KEYS - actual_keys)
            extra = sorted(actual_keys - _STRICT_INDEX_TOP_LEVEL_KEYS)
            raise ValueError(
                f"Strict pair-index top-level keys mismatch; missing={missing}, "
                f"extra={extra}"
            )
        if index_value["schema_version"] != "representation_pair_index.v1":
            raise ValueError(
                "schema_version must be 'representation_pair_index.v1'"
            )
        if index_value["artifact_type"] != "representation_pair_index":
            raise ValueError(
                "artifact_type must be 'representation_pair_index'"
            )
        dataset_basename = _require_nonempty_string(
            index_value["dataset_basename"], "dataset_basename"
        )
        if dataset_basename != root.name:
            raise ValueError(
                f"dataset_basename={dataset_basename!r} does not match "
                f"data_root basename {root.name!r}"
            )
        dataset_binding = _require_sha256(
            index_value["dataset_binding_sha256"], "dataset_binding_sha256"
        )
        _require_sha256(
            index_value["dataset_semantic_lock_sha256"],
            "dataset_semantic_lock_sha256",
        )
        source_index_sha256 = _require_sha256(
            index_value["source_pair_index_sha256"],
            "source_pair_index_sha256",
        )
        if expected_dataset_binding_sha256 is None:
            raise ValueError(
                "Strict indexed mode requires an independently supplied "
                "expected_dataset_binding_sha256"
            )
        expected_binding = _require_sha256(
            expected_dataset_binding_sha256,
            "expected_dataset_binding_sha256",
        )
        if dataset_binding != expected_binding:
            raise ValueError(
                "dataset_binding_sha256 does not match the independently "
                "supplied binding"
            )
        if not isinstance(index_value["unversioned_source"], bool):
            raise ValueError("unversioned_source must be a boolean")
        generator = index_value["generator"]
        if not isinstance(generator, dict) or frozenset(
            generator
        ) != _STRICT_INDEX_GENERATOR_KEYS:
            raise ValueError(
                "generator keys must be exactly "
                f"{sorted(_STRICT_INDEX_GENERATOR_KEYS)}"
            )
        _require_nonempty_string(generator["name"], "generator.name")
        generator_commit = _require_nonempty_string(
            generator["commit"], "generator.commit"
        )
        if (
            len(generator_commit) != 40
            or generator_commit != generator_commit.lower()
            or any(
                char not in "0123456789abcdef" for char in generator_commit
            )
        ):
            raise ValueError("generator.commit must be a full lowercase Git SHA")
        _require_sha256(generator["config_sha256"], "generator.config_sha256")

        frame_partition = index_value["frame_partition"]
        if not isinstance(frame_partition, dict) or frozenset(
            frame_partition
        ) != _STRICT_INDEX_FRAME_PARTITION_KEYS:
            raise ValueError(
                "frame_partition keys must be exactly "
                f"{sorted(_STRICT_INDEX_FRAME_PARTITION_KEYS)}"
            )
        partition_sets: dict[str, set[int]] = {}
        for key, values in frame_partition.items():
            if not isinstance(values, list):
                raise ValueError(f"frame_partition.{key} must be a list")
            parsed = {
                _require_nonnegative_int(
                    value, f"frame_partition.{key}[{index}]"
                )
                for index, value in enumerate(values)
            }
            if len(parsed) != len(values):
                raise ValueError(
                    f"frame_partition.{key} contains duplicate frame indices"
                )
            partition_sets[key] = parsed
        partition_names = sorted(partition_sets)
        for left_index, left_name in enumerate(partition_names):
            for right_name in partition_names[left_index + 1:]:
                overlap = partition_sets[left_name] & partition_sets[right_name]
                if overlap:
                    raise ValueError(
                        f"frame_partition.{left_name} overlaps "
                        f"frame_partition.{right_name}: {sorted(overlap)[:5]}"
                    )

        split = index_value["split"]
        if split not in _INDEX_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_INDEX_SPLITS)}, got {split!r}"
            )
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"Expected split {expected_split!r}, found {split!r} "
                f"in {resolved_index}"
            )

        transform = index_value["transform"]
        if not isinstance(transform, dict):
            raise ValueError("transform must be an object")
        transform_keys = frozenset(transform)
        if transform_keys != _STRICT_INDEX_TRANSFORM_KEYS:
            missing = sorted(_STRICT_INDEX_TRANSFORM_KEYS - transform_keys)
            extra = sorted(transform_keys - _STRICT_INDEX_TRANSFORM_KEYS)
            raise ValueError(
                f"Strict transform keys mismatch; missing={missing}, extra={extra}"
            )
        for key in (
            "family",
            "physical_axis",
            "direction",
            "generator_units",
            "stride_units",
            "expected_2d_family",
        ):
            _require_nonempty_string(transform[key], f"transform.{key}")
        _require_finite_number(
            transform["signed_generator"], "transform.signed_generator"
        )
        _require_positive_int(transform["stride"], "transform.stride")
        if not isinstance(transform["cyclic"], bool):
            raise ValueError("transform.cyclic must be a boolean")

        included_objects = index_value["included_objects"]
        if (
            not isinstance(included_objects, list)
            or not included_objects
            or any(
                not isinstance(value, str) or not value.strip()
                for value in included_objects
            )
            or len(included_objects) != len(set(included_objects))
        ):
            raise ValueError(
                "included_objects must be a non-empty list of unique strings"
            )
        object_roles = index_value["object_roles"]
        if not isinstance(object_roles, dict) or object_roles != _FROZEN_OBJECT_ROLES:
            raise ValueError(
                "object_roles must exactly match the frozen global six-object "
                "role lock"
            )
        for name, role in object_roles.items():
            _require_nonempty_string(role, f"object_roles.{name}")
            if role not in {"development", "confirmation", "final_test"}:
                raise ValueError(
                    f"object_roles.{name} has unknown frozen role {role!r}"
                )
        expected_included = _FROZEN_INCLUDED_OBJECTS_BY_SPLIT[split]
        if tuple(included_objects) != expected_included:
            raise ValueError(
                f"included_objects for split={split!r} must be exactly "
                f"{list(expected_included)} in that order"
            )

        raw_pairs = index_value["pairs"]
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("pairs must be a non-empty list")
        pair_count = _require_positive_int(
            index_value["pair_count"], "pair_count"
        )
        if pair_count != len(raw_pairs):
            raise ValueError(
                f"pair_count={pair_count} does not match len(pairs)={len(raw_pairs)}"
            )
        declared_counts = index_value["pair_counts_by_object"]
        if not isinstance(declared_counts, dict):
            raise ValueError("pair_counts_by_object must be an object")
        for name, count in declared_counts.items():
            _require_nonempty_string(name, "pair_counts_by_object key")
            _require_nonnegative_int(count, f"pair_counts_by_object.{name}")

        content_hash = _require_sha256(
            index_value["content_hash_sha256"], "content_hash_sha256"
        )
        hash_payload = copy.deepcopy(index_value)
        del hash_payload["content_hash_sha256"]
        actual_content_hash = _sha256_bytes(_canonical_json_bytes(hash_payload))
        if actual_content_hash != content_hash:
            raise ValueError(
                f"content_hash_sha256 mismatch for {resolved_index}: "
                f"declared {content_hash}, computed {actual_content_hash}"
            )

        source_relpath = _require_nonempty_string(
            index_value["source_pair_index_relpath"],
            "source_pair_index_relpath",
        )
        source_path = _resolve_dataset_file(
            root,
            source_relpath,
            "source_pair_index_relpath",
            require_exists=True,
        )
        source_bytes = source_path.read_bytes()
        actual_source_sha256 = _sha256_bytes(source_bytes)
        if actual_source_sha256 != source_index_sha256:
            raise ValueError(
                f"source_pair_index_sha256 mismatch for {source_path}: "
                f"declared {source_index_sha256}, computed {actual_source_sha256}"
            )
        try:
            source_document = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Bound source pair index is not valid UTF-8 JSON: {source_path}"
            ) from exc
        if not isinstance(source_document, dict) or not isinstance(
            source_document.get("pairs"), list
        ):
            raise ValueError(
                "Bound source pair index must be an object with a pairs list"
            )
        source_pairs_by_identity: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        for source_row_index, source_pair in enumerate(source_document["pairs"]):
            source_label = f"source pairs[{source_row_index}]"
            if not isinstance(source_pair, dict):
                raise ValueError(f"{source_label} must be an object")
            source_model = _require_nonempty_string(
                source_pair.get("model_name"), f"{source_label}.model_name"
            )
            source_src = _require_nonnegative_int(
                source_pair.get("src_frame_index"),
                f"{source_label}.src_frame_index",
            )
            source_dst = _require_nonnegative_int(
                source_pair.get("dst_frame_index"),
                f"{source_label}.dst_frame_index",
            )
            for path_key in (
                "src_image_relpath",
                "dst_image_relpath",
                "src_mask_relpath",
                "dst_mask_relpath",
            ):
                _require_nonempty_string(
                    source_pair.get(path_key), f"{source_label}.{path_key}"
                )
            source_identity = (source_model, source_src, source_dst)
            if source_identity in source_pairs_by_identity:
                raise ValueError(
                    "Bound source pair index contains duplicate directed "
                    f"identity {source_identity}"
                )
            source_pairs_by_identity[source_identity] = source_pair
    else:
        if isinstance(index_value, dict):
            raw_pairs = index_value.get("pairs")
            if raw_pairs is None:
                raise ValueError("Pair-index object is missing 'pairs'")
        elif isinstance(index_value, list):
            raw_pairs = index_value
            index_value = {"pairs": raw_pairs}
        else:
            raise ValueError("Pair index must be a JSON object or list")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("Pair index must contain a non-empty pairs list")
        content_hash = None
        split = index_value.get("split")
        transform = index_value.get("transform", {})
        source_pairs_by_identity = {}

    selected_pairs: list[dict[str, Any]] = []
    source_endpoints: set[str] = set()
    target_endpoints: set[str] = set()
    seen_pair_ids: set[str] = set()
    seen_endpoint_pairs: set[tuple[str, str]] = set()
    observed_counts: Counter[str] = Counter()

    for pair_index, pair in enumerate(raw_pairs):
        label = f"pairs[{pair_index}]"
        if not isinstance(pair, dict):
            raise ValueError(f"{label} must be an object")
        if strict_metadata:
            pair_keys = frozenset(pair)
            if pair_keys != _STRICT_INDEX_PAIR_KEYS:
                missing = sorted(_STRICT_INDEX_PAIR_KEYS - pair_keys)
                extra = sorted(pair_keys - _STRICT_INDEX_PAIR_KEYS)
                raise ValueError(
                    f"{label} keys mismatch; missing={missing}, extra={extra}"
                )
        for key in (
            "model_name",
            "src_image_relpath",
            "dst_image_relpath",
        ):
            _require_nonempty_string(pair.get(key), f"{label}.{key}")
        _require_nonnegative_int(
            pair.get("src_frame_index"), f"{label}.src_frame_index"
        )
        _require_nonnegative_int(
            pair.get("dst_frame_index"), f"{label}.dst_frame_index"
        )
        model_name = pair["model_name"]

        if strict_metadata:
            pair_id = _require_nonempty_string(pair["pair_id"], f"{label}.pair_id")
            if pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id in index: {pair_id}")
            seen_pair_ids.add(pair_id)
            if pair["split"] != split:
                raise ValueError(f"{label}.split disagrees with top-level split")
            if pair["object_role"] != index_value["object_roles"].get(model_name):
                raise ValueError(
                    f"{label}.object_role disagrees with object_roles"
                )
            if model_name not in index_value["included_objects"]:
                raise ValueError(
                    f"{label}.model_name is absent from included_objects"
                )
            transform_pairs = {
                "transform_family": "family",
                "physical_axis": "physical_axis",
                "direction": "direction",
                "signed_generator": "signed_generator",
                "generator_units": "generator_units",
                "stride": "stride",
                "stride_units": "stride_units",
                "cyclic": "cyclic",
            }
            for pair_key, transform_key in transform_pairs.items():
                if pair[pair_key] != transform[transform_key]:
                    raise ValueError(
                        f"{label}.{pair_key} disagrees with "
                        f"transform.{transform_key}"
                    )
            if not isinstance(pair["src_state"], dict) or not isinstance(
                pair["dst_state"], dict
            ):
                raise ValueError(f"{label} src_state/dst_state must be objects")
            _require_finite_json(pair["src_state"], f"{label}.src_state")
            _require_finite_json(pair["dst_state"], f"{label}.dst_state")
            allowed_partition = (
                partition_sets["train_frame_indices"]
                if split == "train"
                else partition_sets["holdout_frame_indices"]
            )
            for endpoint_key in ("src_frame_index", "dst_frame_index"):
                if pair[endpoint_key] not in allowed_partition:
                    raise ValueError(
                        f"{label}.{endpoint_key}={pair[endpoint_key]} is not "
                        f"in the declared {split} frame partition"
                    )
            for mask_key in ("src_mask_relpath", "dst_mask_relpath"):
                _resolve_dataset_file(
                    root,
                    pair[mask_key],
                    f"{label}.{mask_key}",
                    require_exists=validate_paths,
                )
            source_identity = (
                model_name,
                pair["src_frame_index"],
                pair["dst_frame_index"],
            )
            source_pair = source_pairs_by_identity.get(source_identity)
            if source_pair is None:
                raise ValueError(
                    f"{label} is absent from the bound source pair index: "
                    f"{source_identity}"
                )
            for path_key in (
                "src_image_relpath",
                "dst_image_relpath",
                "src_mask_relpath",
                "dst_mask_relpath",
            ):
                if pair[path_key] != source_pair[path_key]:
                    raise ValueError(
                        f"{label}.{path_key} does not match the bound source "
                        "pair-index row"
                    )

        source_path = _resolve_dataset_file(
            root,
            pair["src_image_relpath"],
            f"{label}.src_image_relpath",
            require_exists=validate_paths,
        )
        target_path = _resolve_dataset_file(
            root,
            pair["dst_image_relpath"],
            f"{label}.dst_image_relpath",
            require_exists=validate_paths,
        )
        endpoint_pair = (str(source_path), str(target_path))
        if strict_metadata and endpoint_pair in seen_endpoint_pairs:
            raise ValueError(
                f"Duplicate directed endpoint pair in index: {endpoint_pair}"
            )
        seen_endpoint_pairs.add(endpoint_pair)
        observed_counts[model_name] += 1

        if object_name is None or model_name == object_name:
            selected_pairs.append(copy.deepcopy(pair))
            source_endpoints.add(str(source_path))
            target_endpoints.add(str(target_path))

    if strict_metadata:
        if dict(observed_counts) != index_value["pair_counts_by_object"]:
            raise ValueError(
                "pair_counts_by_object does not match observed pair counts"
            )
        if set(observed_counts) != set(index_value["included_objects"]):
            raise ValueError(
                "included_objects does not match objects represented by pairs"
            )
    if object_name is not None and not selected_pairs:
        raise ValueError(
            f"No pairs for object_name={object_name!r} in {resolved_index.name}"
        )

    metadata = {
        key: copy.deepcopy(value)
        for key, value in index_value.items()
        if key != "pairs"
    }
    return IndexPairManifest(
        data_root=root,
        index_path=resolved_index,
        index_sha256=index_sha256,
        content_hash_sha256=content_hash,
        metadata=metadata,
        pairs=tuple(selected_pairs),
        source_endpoint_ids=frozenset(source_endpoints),
        target_endpoint_ids=frozenset(target_endpoints),
        endpoint_ids=frozenset(source_endpoints | target_endpoints),
        strict_metadata=strict_metadata,
    )


class PoseSequenceDataset(Dataset):
    """
    Dataset that loads frame pairs (x_t, x_{t+delta}) for training.
    
    Expected directory structure (from create_dataset.py):
        data_root/
            dataset_index.json
            triples.json
            train/
                model_name/
                    frames/a/img_0000.png, img_0001.png, ...
                    meta.jsonl
            test/
                model_name/
                    frames/a/img_0000.png, ...
    
    From Training Protocol (10.4):
    - Use a single fixed step size (Angela suggests 6° or 4°, don't mix)
    - frame_skip controls angular step: skip=1 → consecutive, skip=3 → every 3rd frame
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        object_name: Optional[str] = None,
        img_size: int = 256,
        frame_skip: int = 1,
        center_crop: Optional[int] = None,
        augment: bool = False,
        include_backward: bool = False,
    ):
        """
        Args:
            data_root: Path to dataset root (contains dataset_index.json)
            split: "train" or "test"
            object_name: If specified, only load this object. Otherwise load all.
            img_size: Resize images to this size
            frame_skip: Number of frames to skip between pairs.
                        If yaw_step=2° in dataset: skip=1→2°, skip=2→4°, skip=3→6°
            center_crop: If specified, center crop to this size before resize
            augment: Whether to apply data augmentation (not recommended for Phase A)
            include_backward: If True, add reversed pairs for action classification
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.include_backward = include_backward
        
        # Load dataset index
        index_path = self.data_root / "dataset_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")
        
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Determine which objects to load
        if object_name:
            self.objects = [object_name]
        else:
            self.objects = self.index["splits"].get(split, [])
        
        if not self.objects:
            raise ValueError(f"No objects found for split '{split}'")
        
        # Build list of all frame pairs with frame_skip
        self.pairs = []
        for obj in self.objects:
            obj_dir = self.data_root / split / obj / "frames" / "a"
            if not obj_dir.exists():
                print(f"Warning: Object directory not found: {obj_dir}")
                continue
            
            # Get sorted list of frame files
            frames = sorted(obj_dir.glob("img_*.png"))
            
            # Create pairs with frame_skip
            for i in range(len(frames) - frame_skip):
                self.pairs.append({
                    'x_t_path': frames[i],
                    'x_t1_path': frames[i + frame_skip],
                    'object': obj,
                    't': i,
                    't1': i + frame_skip,
                    'action_label': 0,  # forward (yaw+)
                })
                if include_backward:
                    self.pairs.append({
                        'x_t_path': frames[i + frame_skip],
                        'x_t1_path': frames[i],
                        'object': obj,
                        't': i + frame_skip,
                        't1': i,
                        'action_label': 1,  # backward (yaw-)
                    })
        
        if not self.pairs:
            raise ValueError(
                f"Dataset produced 0 pairs for split='{split}', objects={self.objects}, "
                f"frame_skip={frame_skip}. Check that frame directories exist and contain "
                f"enough frames for the requested frame_skip."
            )
        
        print(
            f"Loaded {len(self.pairs)} pairs from {len(self.objects)} objects "
            f"({split}, frame_skip={frame_skip}, include_backward={include_backward})"
        )
        
        # Image transforms
        transform_list = []
        if center_crop is not None:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        
        # Load images
        x_t = Image.open(pair['x_t_path']).convert('RGB')
        x_t1 = Image.open(pair['x_t1_path']).convert('RGB')
        
        # Apply transforms
        x_t = self.transform(x_t)
        x_t1 = self.transform(x_t1)
        
        return {
            'x_t': x_t,
            'x_t1': x_t1,
            'object': pair['object'],
            't': pair['t'],
            't1': pair['t1'],
            'action_label': pair['action_label'],
        }


class SingleObjectDataset(Dataset):
    """
    Simplified dataset for single-object training (recommended for Phase A debugging).
    
    Directly loads frames from one object's directory.
    
    From Training Protocol (10.4):
    - Use a single fixed step size (Angela suggests 6° or 4°, don't mix)
    - frame_skip controls the angular step: skip=1 → consecutive, skip=3 → every 3rd frame
    """
    
    def __init__(
        self,
        frames_dir: str,
        img_size: int = 256,
        frame_skip: int = 1,
        center_crop: Optional[int] = None,
        include_backward: bool = False,
    ):
        """
        Args:
            frames_dir: Path to frames directory (e.g., data/train/coffeemug/frames/a/)
            img_size: Resize images to this size
            frame_skip: Number of frames to skip between pairs.
                        If yaw_step=2° in dataset: skip=1→2°, skip=2→4°, skip=3→6°
            center_crop: If specified, center crop to this size before resize (reduces background)
            include_backward: If True, add reversed pairs for action classification
        """
        self.frames_dir = Path(frames_dir)
        self.frame_skip = frame_skip
        self.include_backward = include_backward
        
        # Get sorted list of frame files
        self.frames = sorted(self.frames_dir.glob("img_*.png"))
        if not self.frames:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        
        n_pairs = len(self.frames) - frame_skip
        if n_pairs <= 0:
            raise ValueError(
                f"Dataset produced 0 pairs: {len(self.frames)} frames with "
                f"frame_skip={frame_skip}. Need at least {frame_skip + 1} frames."
            )
        self.n_forward = n_pairs
        n_total = n_pairs * (2 if include_backward else 1)
        print(
            f"Loaded {len(self.frames)} frames, frame_skip={frame_skip} -> "
            f"{self.n_forward} forward pairs ({n_total} total, include_backward={include_backward})"
        )
        
        # Image transforms
        transform_list = []
        if center_crop is not None:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        if self.include_backward:
            return self.n_forward * 2
        return self.n_forward

    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_forward:
            t = idx
            t1 = idx + self.frame_skip
            action_label = 0
        else:
            rev_idx = idx - self.n_forward
            t = rev_idx + self.frame_skip
            t1 = rev_idx
            action_label = 1

        # Load frames with skip (forward or backward)
        x_t = Image.open(self.frames[t]).convert('RGB')
        x_t1 = Image.open(self.frames[t1]).convert('RGB')
        
        # Apply transforms
        x_t = self.transform(x_t)
        x_t1 = self.transform(x_t1)
        
        return {
            'x_t': x_t,
            'x_t1': x_t1,
            't': t,
            't1': t1,
            'action_label': action_label,
        }


class IndexPairDataset(Dataset):
    """
    Loads explicit transform-complete frame pairs without inferring geometry.

    Strict mode accepts the frozen roll/yaw/pitch/scale/translation schema and
    exposes the full top-level transform metadata plus canonical per-pair JSON.
    No angle, axis, direction, stride, or cyclicity is inferred from a filename
    or frame-index ordering. Legacy non-strict indices remain readable for
    historical tools, but the training entry point cannot use one without an
    explicit safe train/holdout mode.

    If include_backward=True, a reversed loader copy is appended with
    action_label 1 and ``loader_reversed=True``; the immutable pair metadata
    still records the declared source direction.
    """

    def __init__(
        self,
        data_root: str,
        index_path: str,
        img_size: int = 256,
        center_crop: Optional[int] = None,
        include_backward: bool = False,
        object_name: Optional[str] = None,
        *,
        strict_metadata: bool = False,
        expected_split: Optional[str] = None,
        expected_index_sha256: Optional[str] = None,
        expected_dataset_binding_sha256: Optional[str] = None,
        manifest: Optional[IndexPairManifest] = None,
    ):
        """
        Args:
            data_root: dataset folder the pair relpaths are relative to
                       (e.g. .../_tdw_world_z_roll_base_panel_512_v2).
            index_path: path to the pair-index JSON. May be absolute or
                        relative to data_root (e.g. "indices/pairs_skip3_cyclic.json").
            img_size: resize images to this size.
            center_crop: optional center crop before resize.
            include_backward: add reversed (-delta_theta) pairs for L_act.
            object_name: if set, keep only pairs for this object (single-object
                         training). If None, use all objects in the index.
            strict_metadata: require and validate the frozen representation
                             pair-index schema.
            expected_split: required split role in strict mode.
            expected_index_sha256: optional preflight hash that must still
                                   match when the Dataset is constructed.
            expected_dataset_binding_sha256: independently verified corpus
                                             binding required in strict mode.
            manifest: optional already-validated manifest. Supplying it avoids
                      reopening the index but never opens image contents.
        """
        self.data_root = Path(data_root).expanduser().resolve()
        if manifest is None:
            manifest = inspect_index_pair_manifest(
                data_root=str(self.data_root),
                index_path=index_path,
                expected_split=expected_split,
                object_name=object_name,
                strict_metadata=strict_metadata,
                expected_index_sha256=expected_index_sha256,
                expected_dataset_binding_sha256=expected_dataset_binding_sha256,
                validate_paths=True,
            )
        else:
            if manifest.data_root != self.data_root:
                raise ValueError("Prevalidated manifest data_root mismatch")
            if expected_split is not None and manifest.split != expected_split:
                raise ValueError(
                    f"Expected split {expected_split!r}, found {manifest.split!r}"
                )
            if expected_index_sha256 is not None and (
                manifest.index_sha256 != expected_index_sha256
            ):
                raise ValueError("Prevalidated manifest index SHA-256 mismatch")
            if strict_metadata and not manifest.strict_metadata:
                raise ValueError("Strict Dataset requires a strict manifest")
            if expected_dataset_binding_sha256 is not None and (
                manifest.dataset_binding_sha256
                != expected_dataset_binding_sha256
            ):
                raise ValueError(
                    "Prevalidated manifest dataset binding mismatch"
                )

        self.manifest = manifest
        self.resolved_index_path = manifest.index_path
        self.index_sha256 = manifest.index_sha256
        self.content_hash_sha256 = manifest.content_hash_sha256
        self.dataset_binding_sha256 = manifest.dataset_binding_sha256
        self.index_metadata = copy.deepcopy(manifest.metadata)
        self.transform_metadata = manifest.transform
        self.split_role = manifest.split
        self.source_endpoint_ids = manifest.source_endpoint_ids
        self.target_endpoint_ids = manifest.target_endpoint_ids
        self.endpoint_ids = manifest.endpoint_ids

        raw_pairs = manifest.pairs
        self.cyclic = bool(
            self.transform_metadata.get(
                "cyclic", self.index_metadata.get("cyclic", False)
            )
        )
        self.delta_theta_deg = self.index_metadata.get("delta_theta_deg")
        self.transform_family = self.transform_metadata.get("family")
        self.signed_generator = self.transform_metadata.get("signed_generator")

        self.samples = []
        for pair_number, p in enumerate(raw_pairs):
            pair_id = str(p.get("pair_id", pair_number))
            pair_metadata_json = _canonical_json_bytes(p).decode("utf-8")
            source_endpoint_id = str(
                _resolve_dataset_file(
                    self.data_root,
                    p["src_image_relpath"],
                    f"pairs[{pair_number}].src_image_relpath",
                    require_exists=True,
                )
            )
            target_endpoint_id = str(
                _resolve_dataset_file(
                    self.data_root,
                    p["dst_image_relpath"],
                    f"pairs[{pair_number}].dst_image_relpath",
                    require_exists=True,
                )
            )
            self.samples.append({
                "x_t_rel": p["src_image_relpath"],
                "x_t1_rel": p["dst_image_relpath"],
                "object": p["model_name"],
                "t": p["src_frame_index"],
                "t1": p["dst_frame_index"],
                "action_label": 0,
                "pair_id": pair_id,
                "pair_metadata_json": pair_metadata_json,
                "source_endpoint_id": source_endpoint_id,
                "target_endpoint_id": target_endpoint_id,
                "loader_reversed": False,
            })
            if include_backward:
                self.samples.append({
                    "x_t_rel": p["dst_image_relpath"],
                    "x_t1_rel": p["src_image_relpath"],
                    "object": p["model_name"],
                    "t": p["dst_frame_index"],
                    "t1": p["src_frame_index"],
                    "action_label": 1,
                    "pair_id": pair_id,
                    "pair_metadata_json": pair_metadata_json,
                    "source_endpoint_id": target_endpoint_id,
                    "target_endpoint_id": source_endpoint_id,
                    "loader_reversed": True,
                })

        if not self.samples:
            raise ValueError(
                f"Pair index produced 0 samples: {manifest.index_path}"
            )

        n_obj = len({s["object"] for s in self.samples})
        print(
            f"Loaded {len(self.samples)} pairs from {manifest.index_path.name} "
            f"(objects={n_obj}, split={self.split_role}, "
            f"transform_family={self.transform_family}, cyclic={self.cyclic}, "
            f"signed_generator={self.signed_generator}, "
            f"include_backward={include_backward})"
        )

        transform_list = []
        if center_crop is not None:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        x_t = Image.open(self.data_root / s["x_t_rel"]).convert("RGB")
        x_t1 = Image.open(self.data_root / s["x_t1_rel"]).convert("RGB")
        return {
            "x_t": self.transform(x_t),
            "x_t1": self.transform(x_t1),
            "object": s["object"],
            "t": s["t"],
            "t1": s["t1"],
            "action_label": s["action_label"],
            "pair_id": s["pair_id"],
            "pair_metadata_json": s["pair_metadata_json"],
            "source_endpoint_id": s["source_endpoint_id"],
            "target_endpoint_id": s["target_endpoint_id"],
            "loader_reversed": s["loader_reversed"],
        }


def get_inverse_transform():
    """Get transform to convert normalized tensor back to displayable image."""
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
    ])


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root>")
        print("Example: python dataset.py ./phase_a_yaw_only")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    # Test full dataset
    try:
        dataset = PoseSequenceDataset(data_root, split="train")
        sample = dataset[0]
        print(f"Sample x_t shape: {sample['x_t'].shape}")
        print(f"Sample x_t1 shape: {sample['x_t1'].shape}")
        print(f"Sample object: {sample['object']}")
        print(f"Sample t: {sample['t']}")
        print(f"Sample action_label: {sample['action_label']}")
    except Exception as e:
        print(f"Could not load full dataset: {e}")
    
    # Test single object dataset (if path provided)
    if len(sys.argv) > 2:
        frames_dir = sys.argv[2]
        dataset = SingleObjectDataset(frames_dir)
        sample = dataset[0]
        print(f"Single object - x_t shape: {sample['x_t'].shape}")
        print(f"Single object - action_label: {sample['action_label']}")
