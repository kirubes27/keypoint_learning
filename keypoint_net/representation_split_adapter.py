"""Hash-bound adapter from the frozen split bundle to evaluator row strata.

This module does not read model outputs.  It validates the complete committed
split bundle, selects one object and one evaluation artifact, and keeps
evaluation and optional fitting frames/rows structurally separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from keypoint_net.representation_split_bundle import validate_primary_split_bundle
from keypoint_net.representation_splits import canonical_json_bytes


ADAPTER_SCHEMA_VERSION = "representation_split_adapter_rows.v1"
GEOMETRY_SCHEMA_VERSION = "representation_geometry_binding.v1"
GEOMETRY_REGISTRY_SCHEMA_VERSION = "representation-geometry-binding-registry-v1"
FROZEN_BUNDLE_RELPATH = PurePosixPath(
    "docs/decisions/2026-07-26/representation_oracle_splits"
)
FROZEN_GEOMETRY_REGISTRY_RELPATH = PurePosixPath(
    "docs/decisions/2026-07-26/representation_oracle_geometry/"
    "GEOMETRY_BINDING_REGISTRY_v1.json"
)
FROZEN_GEOMETRY_REGISTRY_FILE_SHA256 = (
    "55453188e3c49d7f8ffdf7b2128167f721ba76468ba79ec1d10f49444f2ece4f"
)
FROZEN_GEOMETRY_REGISTRY_CONTENT_SHA256 = (
    "4bcbc6541ea5069cddc17f83ea35fea919bf1883ecccefaeec16b40823a6e31a"
)
FROZEN_ARTIFACT_COMMIT = "78071297ba46bb22b637a00e1c141eb6e9a8f2de"
FROZEN_GENERATOR_COMMIT = "d325cdb9bdf07dd3d215b1f9e3153012e2065f5b"
FROZEN_MANIFEST_FILE_SHA256 = (
    "9b70db2b2f30e151c1322d7839715167b9dbf750f4caa0b6986a31218ac7ed55"
)
FROZEN_MANIFEST_CONTENT_SHA256 = (
    "aae48879593e6bb7e4c9fdd4706a71f09785b8ebee9d1a7662c0fce62d640a7d"
)
FROZEN_VERIFIER_FILE_SHA256 = (
    "7c809cf0817b50c3a40c1e85b74cc11aa359110475f52e419164f0a5bb0bba22"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FAMILY_TO_INVENTORY = {
    "roll": "roll",
    "yaw": "yaw",
    "pitch": "pitch",
    "scale": "scale",
    "translation": "translation",
}
_PAIR_HEADER_MATCH_FIELDS = (
    "dataset_basename",
    "dataset_binding_sha256",
    "dataset_semantic_lock_sha256",
    "source_pair_index_relpath",
    "source_pair_index_sha256",
    "generator",
    "transform",
    "object_roles",
)


class SplitAdapterError(ValueError):
    """Raised when frozen split meaning cannot be preserved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitAdapterError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SplitAdapterError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise SplitAdapterError(f"non-finite JSON number {token!r}")
    return value


def _reject_constant(token: str) -> None:
    raise SplitAdapterError(f"non-finite JSON constant {token!r}")


def _decode_json(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitAdapterError(f"{name}: invalid JSON") from exc
    _require(isinstance(value, dict), f"{name}: top level must be an object")
    return value


def _canonical_content_hash(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    claimed = payload.pop("content_hash_sha256", None)
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        "document content_hash_sha256 is invalid",
    )
    computed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _require(claimed == computed, "document canonical content hash mismatch")
    return computed


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(value)
    missing = required - actual
    extra = actual - required - optional
    _require(not missing, f"{name}: missing keys {sorted(missing)}")
    _require(not extra, f"{name}: unexpected keys {sorted(extra)}")


def _finite_vector2(value: object, *, name: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == 2,
        f"{name} must be a two-element list",
    )
    result: list[float] = []
    for component in value:
        _require(
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and math.isfinite(float(component)),
            f"{name} must contain finite numbers",
        )
        result.append(float(component))
    return result


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _authorize_registered_dataset_geometry(
    *,
    geometry_binding_path: str | Path,
    family: str,
    object_id: str,
    inventory_content_hash: str,
) -> None:
    """Fail closed unless the exact dataset geometry binding is registered.

    The current registry deliberately authorizes no dataset-backed geometry.
    Keeping this check before ``_load_geometry`` prevents a caller-created,
    self-hashed JSON document from becoming semantic evidence merely because
    its fields are internally consistent.
    """

    registry_path = (
        _repository_root() / FROZEN_GEOMETRY_REGISTRY_RELPATH
    ).resolve(strict=True)
    _require(
        registry_path.is_file() and not registry_path.is_symlink(),
        "geometry registry must be a regular non-symlink file",
    )
    raw = registry_path.read_bytes()
    _require(
        _sha256_bytes(raw) == FROZEN_GEOMETRY_REGISTRY_FILE_SHA256,
        "geometry registry file hash mismatch",
    )
    registry = _decode_json(raw, name="geometry binding registry")
    _require(
        _canonical_content_hash(registry)
        == FROZEN_GEOMETRY_REGISTRY_CONTENT_SHA256,
        "geometry registry content hash mismatch",
    )
    _exact_keys(
        registry,
        required={
            "schema_version",
            "artifact_type",
            "date",
            "status",
            "purpose",
            "execution_boundary",
            "required_evidence",
            "registered_bindings",
            "stop_condition",
            "content_hash_sha256",
        },
        name="geometry binding registry",
    )
    _require(
        registry["schema_version"] == GEOMETRY_REGISTRY_SCHEMA_VERSION,
        "geometry registry schema mismatch",
    )
    boundary = registry["execution_boundary"]
    _require(
        isinstance(boundary, Mapping),
        "geometry registry execution boundary is invalid",
    )
    _require(
        boundary.get("unregistered_geometry_rejected") is True,
        "geometry registry does not reject unregistered geometry",
    )
    bindings = registry["registered_bindings"]
    _require(
        isinstance(bindings, list),
        "geometry registry registered_bindings must be a list",
    )
    requested_path = Path(geometry_binding_path)
    _require(
        requested_path.is_absolute(),
        "dataset geometry binding path must be absolute",
    )
    matching = [
        record
        for record in bindings
        if isinstance(record, Mapping)
        and record.get("transform_family") == family
        and record.get("object_id") == object_id
        and record.get("inventory_content_hash_sha256")
        == inventory_content_hash
    ]
    _require(
        boundary.get("dataset_backed_geometry_authorized") is True,
        (
            "dataset-backed geometry is blocked: the frozen registry has no "
            "reviewed semantic geometry evidence"
        ),
    )
    _require(
        len(matching) == 1,
        "dataset geometry has no unique frozen registry entry",
    )
    record = matching[0]
    _exact_keys(
        record,
        required={
            "transform_family",
            "object_id",
            "inventory_content_hash_sha256",
            "repo_relative_path",
            "file_sha256",
            "content_hash_sha256",
            "derivation_method",
            "evidence_roles",
        },
        name="registered geometry binding",
    )
    expected_path = (
        _repository_root() / PurePosixPath(record["repo_relative_path"])
    ).resolve(strict=True)
    _require(
        requested_path.resolve(strict=True) == expected_path,
        "geometry binding path differs from frozen registry entry",
    )
    _require(
        _sha256_file(expected_path) == record["file_sha256"],
        "registered geometry binding file hash mismatch",
    )


def _verify_frozen_git_blobs(
    bundle: Mapping[str, bytes],
    *,
    repo_root: Path | None = None,
) -> None:
    repository = (repo_root or _repository_root()).resolve()
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{FROZEN_ARTIFACT_COMMIT}^{{commit}}"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SplitAdapterError("frozen split artifact commit is unavailable") from exc
    for relative, current in sorted(bundle.items()):
        repository_relative = (FROZEN_BUNDLE_RELPATH / relative).as_posix()
        try:
            committed = subprocess.run(
                ["git", "show", f"{FROZEN_ARTIFACT_COMMIT}:{repository_relative}"],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SplitAdapterError(
                f"{relative}: absent from frozen artifact commit"
            ) from exc
        _require(
            current == committed,
            f"{relative}: working bytes differ from frozen artifact commit",
        )


def _load_frozen_bundle(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    repo_root = _repository_root()
    expected_root = (repo_root / FROZEN_BUNDLE_RELPATH).resolve(strict=True)
    supplied_root = Path(root).resolve(strict=True)
    _require(
        supplied_root == expected_root,
        "split bundle root is not the frozen committed bundle path",
    )
    bundle: dict[str, bytes] = {}
    for directory, directory_names, filenames in os.walk(
        supplied_root, followlinks=False
    ):
        directory_path = Path(directory)
        for name in directory_names:
            candidate = directory_path / name
            _require(not candidate.is_symlink(), f"bundle directory is a symlink: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            _require(not candidate.is_symlink(), f"bundle file is a symlink: {candidate}")
            _require(candidate.is_file(), f"bundle entry is not a regular file: {candidate}")
            relative = candidate.relative_to(supplied_root).as_posix()
            _require(
                not PurePosixPath(relative).is_absolute()
                and ".." not in PurePosixPath(relative).parts,
                f"unsafe bundle path {relative!r}",
            )
            bundle[relative] = candidate.read_bytes()

    try:
        manifest = validate_primary_split_bundle(
            bundle,
            verify_corpus_contents=True,
        )
    except Exception as exc:
        raise SplitAdapterError(f"full split-bundle validation failed: {exc}") from exc

    _require(
        _sha256_bytes(bundle["SPLIT_MANIFEST.json"])
        == FROZEN_MANIFEST_FILE_SHA256,
        "frozen split manifest file hash mismatch",
    )
    _require(
        manifest.get("content_hash_sha256") == FROZEN_MANIFEST_CONTENT_SHA256,
        "frozen split manifest content hash mismatch",
    )
    _require(
        _sha256_bytes(bundle["SPLIT_VERIFIER_REPORT.json"])
        == FROZEN_VERIFIER_FILE_SHA256,
        "frozen split verifier file hash mismatch",
    )
    _require(
        manifest.get("split_generator", {}).get("commit")
        == FROZEN_GENERATOR_COMMIT,
        "split generator commit mismatch",
    )

    _verify_frozen_git_blobs(bundle, repo_root=repo_root)
    return bundle, manifest


def _manifest_entry(
    manifest: Mapping[str, Any],
    relative_path: str,
) -> Mapping[str, Any]:
    entries = manifest.get("pair_artifacts")
    _require(isinstance(entries, list), "split manifest pair_artifacts is invalid")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("relative_path") == relative_path
    ]
    _require(len(matches) == 1, f"{relative_path}: not uniquely bound by split manifest")
    return matches[0]


def _load_pair(
    bundle: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    relative_path: str,
) -> dict[str, Any]:
    path = PurePosixPath(relative_path)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == 2
        and path.parts[0] == "pairs"
        and path.suffix == ".json",
        "pair artifact path must be an exact safe pairs/*.json path",
    )
    _require(relative_path in bundle, f"pair artifact is absent: {relative_path}")
    raw = bundle[relative_path]
    document = _decode_json(raw, name=relative_path)
    content_hash = _canonical_content_hash(document)
    entry = _manifest_entry(manifest, relative_path)
    _require(
        entry.get("file_sha256") == _sha256_bytes(raw)
        and entry.get("content_hash_sha256") == content_hash,
        f"{relative_path}: manifest hash binding mismatch",
    )
    return document


def _load_geometry(
    path_value: str | Path,
    *,
    family: str,
    object_id: str,
    inventory_content_hash: str,
    dataset_semantic_lock_hash: str,
    required_frame_ids: set[int],
) -> dict[str, Any]:
    """Validate structure only; public semantic authority comes from the registry.

    The public adapter calls ``_authorize_registered_dataset_geometry`` before
    this parser.  Therefore an internally consistent caller-created document
    cannot authorize dataset-backed evaluation.
    """

    path = Path(path_value)
    _require(path.is_absolute(), "geometry binding path must be absolute")
    _require(path.exists() and path.is_file(), "geometry binding must be an existing file")
    _require(not path.is_symlink(), "geometry binding must not be a symlink")
    raw = path.read_bytes()
    document = _decode_json(raw, name="geometry binding")
    required = {
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
    }
    _exact_keys(document, required=required, name="geometry binding")
    _require(
        document["schema_version"] == GEOMETRY_SCHEMA_VERSION,
        "geometry binding schema mismatch",
    )
    _require(
        document["artifact_type"] == "representation_geometry_binding",
        "geometry binding artifact type mismatch",
    )
    _require(document["object_id"] == object_id, "geometry binding object mismatch")
    _require(
        document["transform_family"] == family,
        "geometry binding transform-family mismatch",
    )
    _require(
        document["inventory_content_hash_sha256"] == inventory_content_hash,
        "geometry binding inventory hash mismatch",
    )
    _require(
        document["dataset_semantic_lock_sha256"] == dataset_semantic_lock_hash,
        "geometry binding semantic-lock hash mismatch",
    )
    content_hash = _canonical_content_hash(document)

    estimator_geometry = document["estimator_geometry"]
    _require(isinstance(estimator_geometry, Mapping), "estimator_geometry must be an object")
    _exact_keys(
        estimator_geometry,
        required={
            "coordinate_system",
            "image_y_direction",
            "crop",
            "resize",
            "align_corners",
        },
        name="estimator_geometry",
    )
    _require(
        estimator_geometry["coordinate_system"]
        == "normalized_minus1_plus1_endpoint_aligned_xy",
        "unsupported geometry coordinate system",
    )
    _require(
        estimator_geometry["image_y_direction"] == "down",
        "geometry must bind downward-positive image y",
    )
    _require(
        isinstance(estimator_geometry["resize"], list)
        and len(estimator_geometry["resize"]) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 1
            for item in estimator_geometry["resize"]
        ),
        "geometry resize must be [height,width]",
    )
    _require(
        isinstance(estimator_geometry["align_corners"], (bool, type(None))),
        "geometry align_corners must be bool or null",
    )

    evidence_files = document["evidence_files"]
    _require(
        isinstance(evidence_files, list) and bool(evidence_files),
        "geometry binding requires at least one evidence file",
    )
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    normalized_evidence = []
    for index, record in enumerate(evidence_files):
        _require(isinstance(record, Mapping), f"geometry evidence {index} is invalid")
        _exact_keys(
            record,
            required={"role", "absolute_path", "sha256"},
            name=f"geometry evidence {index}",
        )
        role = record["role"]
        _require(isinstance(role, str) and role, f"geometry evidence {index} role is empty")
        _require(
            isinstance(record["absolute_path"], str) and record["absolute_path"],
            "geometry evidence path must be a non-empty string",
        )
        evidence_path = Path(record["absolute_path"])
        _require(evidence_path.is_absolute(), "geometry evidence path must be absolute")
        _require(
            evidence_path.exists() and evidence_path.is_file() and not evidence_path.is_symlink(),
            "geometry evidence must be an existing regular non-symlink file",
        )
        claimed_hash = record["sha256"]
        _require(
            isinstance(claimed_hash, str) and _SHA256_RE.fullmatch(claimed_hash) is not None,
            "geometry evidence hash is invalid",
        )
        _require(_sha256_file(evidence_path) == claimed_hash, "geometry evidence hash mismatch")
        resolved = str(evidence_path.resolve())
        _require(role not in seen_roles, "duplicate geometry evidence role")
        _require(resolved not in seen_paths, "duplicate geometry evidence path")
        seen_roles.add(role)
        seen_paths.add(resolved)
        normalized_evidence.append(
            {"role": role, "absolute_path": resolved, "sha256": claimed_hash}
        )

    parameters = document["parameters"]
    _require(isinstance(parameters, Mapping), "geometry parameters must be an object")
    normalized_parameters: dict[str, Any]
    if family in {"roll", "scale"}:
        _exact_keys(
            parameters,
            required={"projected_centre_xy"},
            name=f"{family} geometry parameters",
        )
        centre = _finite_vector2(
            parameters["projected_centre_xy"],
            name="projected_centre_xy",
        )
        _require(
            all(-1.0 <= component <= 1.0 for component in centre),
            "projected centre lies outside normalized image coordinates",
        )
        normalized_parameters = {"projected_centre_xy": centre}
    elif family == "translation":
        _exact_keys(
            parameters,
            required={"calibration_method", "frame_calibrated_offsets"},
            name="translation geometry parameters",
        )
        _require(
            isinstance(parameters["calibration_method"], str)
            and bool(parameters["calibration_method"]),
            "translation calibration method is empty",
        )
        records = parameters["frame_calibrated_offsets"]
        _require(isinstance(records, list), "translation calibrated offsets must be a list")
        offsets: dict[int, list[float]] = {}
        for index, record in enumerate(records):
            _require(isinstance(record, Mapping), f"translation offset {index} is invalid")
            _exact_keys(
                record,
                required={"frame_id", "calibrated_image_offset_xy"},
                name=f"translation offset {index}",
            )
            frame_id = record["frame_id"]
            _require(
                isinstance(frame_id, int) and not isinstance(frame_id, bool),
                "translation offset frame_id must be an integer",
            )
            _require(frame_id not in offsets, "duplicate translation calibrated frame")
            offsets[frame_id] = _finite_vector2(
                record["calibrated_image_offset_xy"],
                name=f"translation calibrated offset {frame_id}",
            )
        _require(
            required_frame_ids <= set(offsets),
            "translation calibration does not cover all selected frames",
        )
        normalized_parameters = {
            "calibration_method": parameters["calibration_method"],
            "frame_calibrated_offsets": [
                {
                    "frame_id": frame_id,
                    "calibrated_image_offset_xy": offsets[frame_id],
                }
                for frame_id in sorted(offsets)
            ],
        }
    else:
        _exact_keys(
            parameters,
            required={"frame_projection_geometry"},
            name=f"{family} geometry parameters",
        )
        records = parameters["frame_projection_geometry"]
        _require(isinstance(records, list), f"{family} frame geometry must be a list")
        projection: dict[int, dict[str, str]] = {}
        for index, record in enumerate(records):
            _require(isinstance(record, Mapping), f"{family} frame geometry {index} is invalid")
            _exact_keys(
                record,
                required={"frame_id", "projection_model", "depth_configuration"},
                name=f"{family} frame geometry {index}",
            )
            frame_id = record["frame_id"]
            _require(
                isinstance(frame_id, int) and not isinstance(frame_id, bool),
                f"{family} frame_id must be an integer",
            )
            _require(frame_id not in projection, f"duplicate {family} geometry frame")
            for key in ("projection_model", "depth_configuration"):
                _require(
                    isinstance(record[key], str) and bool(record[key]),
                    f"{family} {key} must be a non-empty string",
                )
            projection[frame_id] = {
                "projection_model": record["projection_model"],
                "depth_configuration": record["depth_configuration"],
            }
        _require(
            required_frame_ids <= set(projection),
            f"{family} geometry does not cover all selected frames",
        )
        normalized_parameters = {
            "frame_projection_geometry": [
                {"frame_id": frame_id, **projection[frame_id]}
                for frame_id in sorted(projection)
            ]
        }

    return {
        "absolute_path": str(path.resolve()),
        "file_sha256": _sha256_bytes(raw),
        "content_hash_sha256": content_hash,
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "estimator_geometry": dict(estimator_geometry),
        "estimator_geometry_sha256": hashlib.sha256(
            canonical_json_bytes(estimator_geometry)
        ).hexdigest(),
        "parameters": normalized_parameters,
        "evidence_files": normalized_evidence,
    }


def _selected_pairs(document: Mapping[str, Any], object_id: str) -> list[dict[str, Any]]:
    included = document.get("included_objects")
    _require(
        isinstance(included, list) and object_id in included,
        f"object {object_id!r} is absent from selected pair artifact",
    )
    rows = document.get("pairs")
    _require(isinstance(rows, list), "pair artifact pairs field is invalid")
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("model_name") == object_id
    ]
    expected_counts = document.get("pair_counts_by_object")
    _require(
        isinstance(expected_counts, Mapping)
        and expected_counts.get(object_id) == len(selected)
        and bool(selected),
        "selected object pair count disagrees with artifact summary",
    )
    return selected


def _frame_records(
    rows: list[Mapping[str, Any]],
    *,
    family: str,
    geometry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = geometry["parameters"]
    offsets = {
        record["frame_id"]: record["calibrated_image_offset_xy"]
        for record in parameters.get("frame_calibrated_offsets", [])
    }
    projection = {
        record["frame_id"]: record
        for record in parameters.get("frame_projection_geometry", [])
    }
    by_frame: dict[int, dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_endpoints: set[tuple[int, int]] = set()
    for row_number, row in enumerate(rows):
        pair_id = row.get("pair_id")
        _require(isinstance(pair_id, str) and pair_id, f"pair row {row_number} has no pair_id")
        _require(pair_id not in seen_pair_ids, "duplicate pair_id")
        seen_pair_ids.add(pair_id)
        source = row.get("src_frame_index")
        target = row.get("dst_frame_index")
        _require(
            isinstance(source, int)
            and not isinstance(source, bool)
            and isinstance(target, int)
            and not isinstance(target, bool)
            and source != target,
            f"pair row {row_number} has invalid endpoints",
        )
        endpoints = (source, target)
        _require(endpoints not in seen_endpoints, "duplicate pair endpoints")
        seen_endpoints.add(endpoints)
        for prefix, frame_id in (("src", source), ("dst", target)):
            state = dict(row[f"{prefix}_state"])
            sources = {key: "frozen_split_pair_artifact" for key in state}
            if family == "translation":
                _require(frame_id in offsets, "selected translation frame lacks calibration")
                state["calibrated_image_offset_xy"] = offsets[frame_id]
                sources["calibrated_image_offset_xy"] = "geometry_binding"
            elif family in {"yaw", "pitch"}:
                _require(frame_id in projection, f"selected {family} frame lacks geometry")
                state["projection_model"] = projection[frame_id]["projection_model"]
                state["depth_configuration"] = projection[frame_id]["depth_configuration"]
                sources["projection_model"] = "geometry_binding"
                sources["depth_configuration"] = "geometry_binding"
            record = {
                "frame_id": frame_id,
                "image_relpath": row[f"{prefix}_image_relpath"],
                "mask_relpath": row[f"{prefix}_mask_relpath"],
                "physical_state": state,
                "physical_state_field_sources": sources,
            }
            if frame_id in by_frame:
                _require(
                    by_frame[frame_id] == record,
                    f"frame {frame_id}: repeated path or physical state is inconsistent",
                )
            else:
                by_frame[frame_id] = record
        normalized_rows.append(
            {
                "pair_id": pair_id,
                "source_frame": source,
                "target_frame": target,
                "direction": row["direction"],
                "stride": row["stride"],
                "partition": row["split"],
            }
        )
    return [by_frame[frame_id] for frame_id in sorted(by_frame)], normalized_rows


def _pair_binding(
    relative_path: str,
    raw: bytes,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "file_sha256": _sha256_bytes(raw),
        "content_hash_sha256": document["content_hash_sha256"],
        "schema_version": document["schema_version"],
        "split": document["split"],
    }


def _derive_evaluator_transform(
    *,
    transform: Mapping[str, Any],
    family: str,
    geometry: Mapping[str, Any],
    evaluation_frames: list[Mapping[str, Any]],
    evaluation_rows: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive evaluator vocabulary without changing frozen physical fields."""

    evaluator_transform = dict(transform)
    transform_derivations: dict[str, Any] = {
        "geometry_binding_sha256": geometry["content_hash_sha256"],
    }
    if family in {"roll", "scale"}:
        evaluator_transform["projected_centre_xy"] = geometry["parameters"][
            "projected_centre_xy"
        ]
    if family == "scale":
        transform_derivations["image_ratio"] = math.exp(
            float(transform["signed_generator"])
        )
    elif family == "translation":
        frame_offsets = {
            record["frame_id"]: record["physical_state"][
                "calibrated_image_offset_xy"
            ]
            for record in evaluation_frames
        }
        deltas = [
            [
                float(frame_offsets[row["target_frame"]][axis])
                - float(frame_offsets[row["source_frame"]][axis])
                for axis in (0, 1)
            ]
            for row in evaluation_rows
        ]
        _require(bool(deltas), "translation evaluation has no calibrated pair delta")
        calibrated_delta = deltas[0]
        _require(
            all(delta == calibrated_delta for delta in deltas),
            "translation calibration does not yield one exact shared image delta",
        )
        component = 0 if transform["physical_axis"] == "world_x" else 1
        other = 1 - component
        _require(
            calibrated_delta[other] == 0.0,
            "translation calibrated target has orthogonal image leakage",
        )
        _require(
            calibrated_delta[component] != 0.0,
            "translation calibrated target has zero commanded-axis displacement",
        )
        generator = float(transform["signed_generator"])
        _require(
            calibrated_delta[component] * generator < 0.0,
            (
                "translation calibration violates the frozen camera sign: "
                "positive world motion must decrease the matching image coordinate"
            ),
        )
        evaluator_transform["translation_source"] = "rendered_world_calibrated"
        evaluator_transform["expected_image_component"] = component
        evaluator_transform["expected_image_sign"] = -1
        evaluator_transform["calibrated_image_delta_xy"] = calibrated_delta
        transform_derivations["calibration_binding_sha256"] = geometry[
            "content_hash_sha256"
        ]
        transform_derivations["calibrated_image_delta_xy"] = calibrated_delta
        transform_derivations["expected_image_sign"] = -1
    elif family in {"yaw", "pitch"}:
        evaluator_transform["reference_transform"] = "fitted_reference"
    return evaluator_transform, transform_derivations


def build_split_adapter_rows(
    *,
    split_bundle_root: str | Path,
    evaluation_pair_relpath: str,
    object_id: str,
    fit_from_pairs: bool,
    geometry_binding_path: str | Path,
    fit_pair_relpath: str | None = None,
) -> dict[str, Any]:
    """Build one hash-bound split stratum without reading model outputs."""

    _require(isinstance(object_id, str) and object_id, "object_id is empty")
    _require(isinstance(fit_from_pairs, bool), "fit_from_pairs must be boolean")
    if fit_from_pairs:
        _require(fit_pair_relpath is not None, "fit_from_pairs requires a train artifact")
    else:
        _require(fit_pair_relpath is None, "supplied-operator adapter must not include fit rows")

    bundle, manifest = _load_frozen_bundle(split_bundle_root)
    evaluation_document = _load_pair(
        bundle, manifest, evaluation_pair_relpath
    )
    _require(
        evaluation_document.get("split") in {"validation", "test"},
        "evaluation artifact must be a held-out validation or test artifact",
    )
    evaluation_rows_raw = _selected_pairs(evaluation_document, object_id)
    family = evaluation_document.get("transform", {}).get("family")
    _require(family in _FAMILY_TO_INVENTORY, "selected pair transform family is invalid")

    fit_document = None
    fit_rows_raw: list[dict[str, Any]] = []
    if fit_from_pairs:
        assert fit_pair_relpath is not None
        fit_document = _load_pair(bundle, manifest, fit_pair_relpath)
        _require(fit_document.get("split") == "train", "fit artifact must use split=train")
        for field in _PAIR_HEADER_MATCH_FIELDS:
            _require(
                fit_document.get(field) == evaluation_document.get(field),
                f"fit/evaluation artifact mismatch at {field}",
            )
        fit_rows_raw = _selected_pairs(fit_document, object_id)

    inventory_key = _FAMILY_TO_INVENTORY[str(family)]
    inventory_relpath = f"inventories/CORPUS_INVENTORY__{inventory_key}.json"
    inventory_raw = bundle[inventory_relpath]
    inventory = _decode_json(inventory_raw, name=inventory_relpath)
    inventory_content_hash = _canonical_content_hash(inventory)
    manifest_inventory = manifest["corpus_inventories"][inventory_key]
    _require(
        manifest_inventory["file_sha256"] == _sha256_bytes(inventory_raw)
        and manifest_inventory["content_hash_sha256"] == inventory_content_hash,
        "selected inventory disagrees with split manifest",
    )
    _require(
        evaluation_document["dataset_binding_sha256"]
        == inventory_content_hash,
        "pair artifact dataset binding disagrees with inventory",
    )
    _require(
        evaluation_document["dataset_semantic_lock_sha256"]
        == inventory["semantic_lock"]["sha256"],
        "pair artifact semantic lock disagrees with inventory",
    )

    required_frame_ids = {
        int(row[key])
        for row in evaluation_rows_raw + fit_rows_raw
        for key in ("src_frame_index", "dst_frame_index")
    }
    _authorize_registered_dataset_geometry(
        geometry_binding_path=geometry_binding_path,
        family=str(family),
        object_id=object_id,
        inventory_content_hash=inventory_content_hash,
    )
    geometry = _load_geometry(
        geometry_binding_path,
        family=str(family),
        object_id=object_id,
        inventory_content_hash=inventory_content_hash,
        dataset_semantic_lock_hash=evaluation_document[
            "dataset_semantic_lock_sha256"
        ],
        required_frame_ids=required_frame_ids,
    )
    evaluation_frames, evaluation_rows = _frame_records(
        evaluation_rows_raw,
        family=str(family),
        geometry=geometry,
    )
    fit_frames = None
    fit_rows = None
    if fit_document is not None:
        fit_frames, fit_rows = _frame_records(
            fit_rows_raw,
            family=str(family),
            geometry=geometry,
        )
        evaluation_endpoints = {
            row[key]
            for row in evaluation_rows
            for key in ("source_frame", "target_frame")
        }
        fit_endpoints = {
            row[key]
            for row in fit_rows
            for key in ("source_frame", "target_frame")
        }
        _require(
            evaluation_endpoints.isdisjoint(fit_endpoints),
            "fit and evaluation endpoint frame IDs overlap",
        )
        _require(
            all(row["partition"] == "train" for row in fit_rows),
            "fit rows contain a non-train partition",
        )
    _require(
        all(
            row["partition"] == evaluation_document["split"]
            for row in evaluation_rows
        ),
        "evaluation rows contain a foreign partition",
    )

    transform = dict(evaluation_document["transform"])
    evaluator_transform, transform_derivations = _derive_evaluator_transform(
        transform=transform,
        family=str(family),
        geometry=geometry,
        evaluation_frames=evaluation_frames,
        evaluation_rows=evaluation_rows,
    )

    object_roles = evaluation_document["object_roles"]
    result: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_binding": {
            "source_relative_path": "keypoint_net/representation_split_adapter.py",
            "source_file_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "split_binding": {
            "artifact_commit": FROZEN_ARTIFACT_COMMIT,
            "manifest": {
                "relative_path": "SPLIT_MANIFEST.json",
                "file_sha256": FROZEN_MANIFEST_FILE_SHA256,
                "content_hash_sha256": FROZEN_MANIFEST_CONTENT_SHA256,
                "schema_version": manifest["schema_version"],
            },
            "verifier_report": {
                "relative_path": "SPLIT_VERIFIER_REPORT.json",
                "file_sha256": FROZEN_VERIFIER_FILE_SHA256,
            },
            "inventory": {
                "relative_path": inventory_relpath,
                "file_sha256": _sha256_bytes(inventory_raw),
                "content_hash_sha256": inventory_content_hash,
                "schema_version": inventory["schema_version"],
            },
            "evaluation_pair_artifact": _pair_binding(
                evaluation_pair_relpath,
                bundle[evaluation_pair_relpath],
                evaluation_document,
            ),
            "fit_pair_artifact": (
                _pair_binding(
                    str(fit_pair_relpath),
                    bundle[str(fit_pair_relpath)],
                    fit_document,
                )
                if fit_document is not None and fit_pair_relpath is not None
                else None
            ),
            "dataset_basename": evaluation_document["dataset_basename"],
            "dataset_binding_sha256": evaluation_document[
                "dataset_binding_sha256"
            ],
            "dataset_semantic_lock_sha256": evaluation_document[
                "dataset_semantic_lock_sha256"
            ],
            "source_pair_index_relpath": evaluation_document[
                "source_pair_index_relpath"
            ],
            "source_pair_index_sha256": evaluation_document[
                "source_pair_index_sha256"
            ],
            "generator": evaluation_document["generator"],
        },
        "stratum": {
            "object_id": object_id,
            "object_role": object_roles[object_id],
            "transform_family": family,
            "physical_axis": transform["physical_axis"],
            "direction": transform["direction"],
            "stride": transform["stride"],
            "stride_units": transform["stride_units"],
            "signed_generator": transform["signed_generator"],
            "generator_units": transform["generator_units"],
            "evaluation_partition": evaluation_document["split"],
            "fit_partition": "train" if fit_document is not None else None,
        },
        "split_transform": transform,
        "evaluator_transform": evaluator_transform,
        "transform_derivations": transform_derivations,
        "evaluation": {
            "partition": evaluation_document["split"],
            "frames": evaluation_frames,
            "rows": evaluation_rows,
        },
        "fit": (
            {"partition": "train", "frames": fit_frames, "rows": fit_rows}
            if fit_document is not None
            else None
        ),
        "geometry_binding": geometry,
    }
    result["adapter_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "FROZEN_ARTIFACT_COMMIT",
    "FROZEN_BUNDLE_RELPATH",
    "FROZEN_GENERATOR_COMMIT",
    "GEOMETRY_SCHEMA_VERSION",
    "SplitAdapterError",
    "build_split_adapter_rows",
]
