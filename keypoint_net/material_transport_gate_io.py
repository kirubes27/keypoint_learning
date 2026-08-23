"""Source binding and sanitized-manifest helpers for the transport gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class MaterialTransportIOError(ValueError):
    """Raised when a transport artifact violates its source contract."""


SANITIZED_SCHEMA = "sanitized_cyclic_rgb_field_manifest.v1"
EXPECTED_FRAMES = 180


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterialTransportIOError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"file missing: {resolved}")
    record: dict[str, Any] = {
        "sha256": sha256_path(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if include_path:
        record["absolute_path"] = str(resolved)
    return record


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def validate_sanitized_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if the raw-stage manifest contains non-RGB information."""

    allowed_root = {
        "schema_version",
        "artifact_type",
        "implementation_head",
        "implementation_sources",
        "semantic_lock",
        "upstream_manifest_digest",
        "field_config",
        "dataset",
        "statistical_scope",
        "information_boundary",
    }
    require(set(manifest) == allowed_root, "sanitized manifest root keys differ")
    require(manifest["schema_version"] == SANITIZED_SCHEMA, "sanitized schema differs")
    require(manifest["artifact_type"] == "sanitized_cyclic_rgb_field_manifest", "artifact type differs")
    dataset = manifest["dataset"]
    require(isinstance(dataset, Mapping), "dataset entry is not an object")
    allowed_dataset = {
        "object_root_at_build_time",
        "frame_count",
        "frames",
        "portable_rebinding_rule",
        "edge_rule",
    }
    require(set(dataset) == allowed_dataset, "sanitized dataset keys differ")
    require(int(dataset["frame_count"]) == EXPECTED_FRAMES, "frame count differs")
    require(dataset["edge_rule"] == "cyclic_f_to_f_plus_1_mod_180", "edge rule differs")
    frames = dataset["frames"]
    require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "frame rows differ")
    allowed_frame = {"frame_index", "image_relpath", "image_sha256", "size_bytes"}
    for expected, row in enumerate(frames):
        require(isinstance(row, Mapping) and set(row) == allowed_frame, f"frame {expected} keys differ")
        require(int(row["frame_index"]) == expected, f"frame {expected} index differs")
        require(Path(str(row["image_relpath"])).as_posix() == str(row["image_relpath"]), f"frame {expected} path is not normalized")
        require(len(str(row["image_sha256"])) == 64, f"frame {expected} hash differs")
        require(int(row["size_bytes"]) > 0, f"frame {expected} size differs")

    serialized = json.dumps(manifest, sort_keys=True).lower()
    forbidden_tokens = (
        '"mask',
        '"theta',
        '"pivot',
        '"witness',
        '"checkpoint',
        '"operator',
        '"detector',
        '"target_coordinate',
        '"material_coordinate',
    )
    for token in forbidden_tokens:
        require(token not in serialized, f"sanitized manifest contains forbidden token {token}")


def resolve_rgb_paths(
    manifest: Mapping[str, Any],
    *,
    object_root_override: Path | None = None,
) -> list[Path]:
    validate_sanitized_manifest(manifest)
    dataset = manifest["dataset"]
    root = (
        object_root_override.resolve()
        if object_root_override is not None
        else Path(str(dataset["object_root_at_build_time"])).resolve()
    )
    require(root.is_dir(), f"RGB object root missing: {root}")
    output: list[Path] = []
    for row in dataset["frames"]:
        path = (root / str(row["image_relpath"])).resolve()
        record = file_record(path, include_path=False)
        require(record["sha256"] == row["image_sha256"], f"RGB hash differs: {path}")
        require(record["size_bytes"] == int(row["size_bytes"]), f"RGB size differs: {path}")
        output.append(path)
    return output


__all__ = [
    "EXPECTED_FRAMES",
    "MaterialTransportIOError",
    "SANITIZED_SCHEMA",
    "file_record",
    "load_json",
    "require",
    "resolve_rgb_paths",
    "sha256_path",
    "validate_sanitized_manifest",
    "write_json",
]
