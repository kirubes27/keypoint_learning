"""Content-derived inventories for immutable representation corpora.

The inventory is deliberately standard-library-only.  It binds every regular
file below a dataset root (apart from explicitly ignored macOS ``.DS_Store``
metadata), validates the per-object metadata ledgers, and records the exact
physical state associated with every frame.

An inventory digest is evidence only after :func:`validate_corpus_inventory`
has recomputed the complete file set, sizes, hashes, and metadata against the
current immutable corpus.  Callers must never accept a digest string by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


INVENTORY_SCHEMA_VERSION = "representation_corpus_inventory.v1"
INVENTORY_ARTIFACT_TYPE = "representation_corpus_inventory"
IGNORED_BASENAMES = (".DS_Store",)
UNVERSIONED_WARNING = (
    "Unversioned rendered corpus: scientific identity is bound by this complete "
    "content inventory and the separately versioned generator-provenance snapshot."
)

OBJECT_ORDER = (
    "engineers_hammer_vray",
    "b03_banana_01_high",
    "kettle",
    "dewalt_compact_drill_vray",
    "b03_trumpet_vray",
    "toy_monkey_medium",
)


@dataclass(frozen=True)
class CorpusDefinition:
    dataset_key: str
    dataset_basename: str
    family: str
    train_frame_count: int
    auxiliary_partitions: tuple[tuple[str, int], ...] = ()


CORPUS_DEFINITIONS: dict[str, CorpusDefinition] = {
    "roll": CorpusDefinition(
        "roll",
        "_tdw_world_z_roll_base_panel_512_v2",
        "roll",
        180,
    ),
    "yaw": CorpusDefinition(
        "yaw",
        "_tdw_world_y_yaw_arc60_step1_base_panel_512_v1",
        "yaw",
        121,
    ),
    "pitch": CorpusDefinition(
        "pitch",
        "_tdw_world_x_pitch_arc60_step1_base_panel_512_v1",
        "pitch",
        121,
    ),
    "scale": CorpusDefinition(
        "scale",
        "_tdw_uniform_scale_loghalf_to_one_base_panel_512_v1",
        "scale",
        121,
        (("eval_abs_holdout", 31),),
    ),
    "translation": CorpusDefinition(
        "translation",
        "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
        "translation",
        121,
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_key",
        "dataset_basename",
        "source_root_provenance",
        "unversioned_source",
        "unversioned_warning",
        "ignored_basenames",
        "semantic_lock",
        "objects",
        "file_count",
        "total_bytes",
        "files",
        "source_pair_indices",
        "frame_records",
        "content_hash_sha256",
    }
)
_FILE_KEYS = frozenset({"relative_path", "size_bytes", "sha256"})
_PAIR_INDEX_KEYS = _FILE_KEYS
_FRAME_KEYS = frozenset(
    {
        "model_name",
        "source_partition",
        "frame_index",
        "image_relpath",
        "mask_relpath",
        "id_pass_relpath",
        "meta_jsonl_relpath",
        "meta_line_number",
        "meta_record_sha256",
        "physical_state",
    }
)


class CorpusInventoryError(ValueError):
    """Raised when an immutable corpus or its inventory is inconsistent."""


@dataclass(frozen=True)
class ValidatedCorpusInventory:
    """Validated inventory plus lookup maps used by split generation."""

    dataset_key: str
    root: Path
    document: dict[str, Any]
    files_by_relpath: dict[str, dict[str, Any]]
    frames_by_identity: dict[tuple[str, str, int], dict[str, Any]]

    @property
    def content_hash_sha256(self) -> str:
        return str(self.document["content_hash_sha256"])

    @property
    def semantic_lock_sha256(self) -> str:
        return str(self.document["semantic_lock"]["sha256"])

    def frame(
        self,
        model_name: str,
        frame_index: int,
        *,
        source_partition: str = "train",
    ) -> dict[str, Any]:
        try:
            return self.frames_by_identity[(source_partition, model_name, frame_index)]
        except KeyError as exc:
            raise CorpusInventoryError(
                f"{self.dataset_key}: inventory has no frame "
                f"{source_partition}/{model_name}/{frame_index}"
            ) from exc


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


def inventory_content_hash(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusInventoryError(message)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    _require(
        actual == expected,
        f"{context}: schema mismatch; missing={sorted(expected-actual)}, "
        f"extra={sorted(actual-expected)}",
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(actual: Any, expected: float, context: str, *, tolerance: float = 1e-12) -> None:
    _require(_is_number(actual), f"{context}: expected a finite number")
    _require(
        math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{context}: expected {expected!r}, got {actual!r}",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusInventoryError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise CorpusInventoryError(f"non-standard/non-finite JSON constant: {value}")


def _decode_json_bytes(data: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusInventoryError(f"{context}: invalid strict UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{context}: top-level JSON must be an object")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        return _decode_json_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise CorpusInventoryError(f"cannot read {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CorpusInventoryError(f"cannot read UTF-8 metadata {path}: {exc}") from exc
    lines = text.splitlines()
    _require(lines, f"{path}: metadata ledger is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(line.strip() != "", f"{path}:{line_number}: blank metadata row")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise CorpusInventoryError(
                f"{path}:{line_number}: invalid strict JSON: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _scan_regular_files(root: Path) -> list[dict[str, Any]]:
    """Recursively bind every regular file and reject unsafe filesystem entries."""

    _require(root.exists(), f"dataset root does not exist: {root}")
    _require(not root.is_symlink(), f"dataset root may not be a symlink: {root}")
    _require(root.is_dir(), f"dataset root is not a directory: {root}")
    root_resolved = root.resolve(strict=True)
    collected: list[dict[str, Any]] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise CorpusInventoryError(f"cannot scan {directory}: {exc}") from exc
        for entry in entries:
            rel_parts = relative_parts + (entry.name,)
            relpath = PurePosixPath(*rel_parts).as_posix()
            if entry.is_symlink():
                raise CorpusInventoryError(f"{relpath}: symlinks are forbidden")
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CorpusInventoryError(f"cannot stat {relpath}: {exc}") from exc
            entry_path = Path(entry.path)
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(entry_path, rel_parts)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise CorpusInventoryError(f"{relpath}: non-regular filesystem entry")
            if entry.name in IGNORED_BASENAMES:
                continue
            resolved = entry_path.resolve(strict=True)
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise CorpusInventoryError(f"{relpath}: path escapes dataset root") from exc
            collected.append(
                {
                    "relative_path": relpath,
                    "size_bytes": entry_stat.st_size,
                    "sha256": _sha256_file(resolved),
                }
            )

    visit(root_resolved, ())
    collected.sort(key=lambda item: item["relative_path"])
    _require(
        len({item["relative_path"] for item in collected}) == len(collected),
        f"{root}: duplicate canonical relative paths",
    )
    return collected


def _canonical_frame_paths(partition: str, model: str, index: int) -> tuple[str, str, str]:
    prefix = f"{partition}/{model}"
    return (
        f"{prefix}/frames/a/img_{index:04d}.png",
        f"{prefix}/masks/a/mask_{index:04d}.png",
        f"{prefix}/id_passes/a/id_{index:04d}.png",
    )


def _validate_common_metadata(
    row: Mapping[str, Any],
    *,
    definition: CorpusDefinition,
    partition: str,
    model: str,
    index: int,
    context: str,
) -> None:
    _require(
        _is_int(row.get("frame_index")) and row["frame_index"] == index,
        f"{context}: frame_index must equal {index}",
    )
    _require(row.get("model_name") == model, f"{context}: wrong model_name")
    if definition.family != "roll":
        _require(row.get("split") == partition, f"{context}: wrong source partition")
        _require(row.get("gate_failure") is None, f"{context}: frame failed render gate")


def _physical_state(
    row: Mapping[str, Any],
    *,
    definition: CorpusDefinition,
    partition: str,
    index: int,
    context: str,
) -> dict[str, Any]:
    family = definition.family
    if family == "roll":
        _close(row.get("theta_deg"), 2.0 * index, f"{context}.theta_deg")
        _close(row.get("theta_step_deg"), 2.0, f"{context}.theta_step_deg")
        _require(row.get("operator_name") == "tdw_world_z_roll", f"{context}: wrong operator")
        _require(row.get("tdw_axis") == "roll", f"{context}: wrong TDW axis")
        _require(row.get("is_world") is True, f"{context}: roll must use world axis")
        _require(row.get("use_centroid") is True, f"{context}: roll must use centroid")
        _require(
            row.get("rotation_is_tdw_rerendered") is True,
            f"{context}: roll frame is not marked as a TDW rerender",
        )
        _require(row.get("valid") is True, f"{context}: roll frame is invalid")
        _require(row.get("invalid_reason") == "valid", f"{context}: invalid roll reason")
        _require(row.get("t") == index, f"{context}: t must equal frame_index")
        _require(row.get("image_index") == index, f"{context}: image_index mismatch")
        return {
            "theta_deg": float(row["theta_deg"]),
            "theta_step_deg": float(row["theta_step_deg"]),
            "operator_name": row["operator_name"],
            "tdw_axis": row["tdw_axis"],
            "is_world": True,
            "use_centroid": True,
            "rotation_is_tdw_rerendered": True,
            "valid": True,
        }
    if family in {"yaw", "pitch"}:
        expected_operator = (
            "tdw_world_y_yaw_by_axis" if family == "yaw" else "tdw_world_x_pitch"
        )
        _close(row.get("theta_deg"), -60.0 + index, f"{context}.theta_deg")
        _require(row.get("operator_name") == expected_operator, f"{context}: wrong operator")
        _require(
            _is_number(row.get("scale_abs")) and float(row["scale_abs"]) > 0.0,
            f"{context}: invalid scale_abs",
        )
        _require(
            _is_number(row.get("render_scale_rel"))
            and float(row["render_scale_rel"]) > 0.0,
            f"{context}: invalid render_scale_rel",
        )
        return {
            "theta_deg": float(row["theta_deg"]),
            "operator_name": row["operator_name"],
            "scale_abs": float(row["scale_abs"]),
            "render_scale_rel": float(row["render_scale_rel"]),
        }
    if family == "scale":
        expected_ladder = "train" if partition == "train" else "eval_abs_holdout"
        exponent = index if partition == "train" else (4.0 * index + 0.5)
        expected_scale = 0.5 * (2.0 ** (exponent / 120.0))
        _require(row.get("ladder") == expected_ladder, f"{context}: wrong scale ladder")
        _require(row.get("ladder_index") == index, f"{context}: wrong ladder_index")
        _close(row.get("s_rel"), expected_scale, f"{context}.s_rel")
        _require(row.get("operator_name") == "tdw_uniform_scale", f"{context}: wrong operator")
        _require(
            _is_number(row.get("scale_abs")) and float(row["scale_abs"]) > 0.0,
            f"{context}: invalid scale_abs",
        )
        return {
            "s_rel": float(row["s_rel"]),
            "scale_abs": float(row["scale_abs"]),
            "ladder": row["ladder"],
            "ladder_index": index,
            "operator_name": row["operator_name"],
        }
    if family == "translation":
        grid_x = index % 11
        grid_y = index // 11
        _require(row.get("grid_x") == grid_x, f"{context}: wrong grid_x")
        _require(row.get("grid_y") == grid_y, f"{context}: wrong grid_y")
        _close(row.get("dx_world"), -0.08 + 0.016 * grid_x, f"{context}.dx_world")
        _close(row.get("dy_world"), -0.08 + 0.016 * grid_y, f"{context}.dy_world")
        _close(row.get("scale_rel"), 0.60, f"{context}.scale_rel")
        _require(
            row.get("operator_name") == "tdw_camera_plane_translation",
            f"{context}: wrong operator",
        )
        _require(
            _is_number(row.get("scale_abs")) and float(row["scale_abs"]) > 0.0,
            f"{context}: invalid scale_abs",
        )
        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "dx_world": float(row["dx_world"]),
            "dy_world": float(row["dy_world"]),
            "scale_rel": float(row["scale_rel"]),
            "scale_abs": float(row["scale_abs"]),
            "operator_name": row["operator_name"],
        }
    raise CorpusInventoryError(f"{context}: unknown family {family!r}")


def _frame_records(
    root: Path,
    definition: CorpusDefinition,
    files_by_relpath: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    partitions = (("train", definition.train_frame_count),) + definition.auxiliary_partitions

    for partition, expected_count in partitions:
        partition_dir = root / partition
        _require(partition_dir.is_dir(), f"{definition.dataset_key}: missing {partition}/")
        actual_objects = sorted(
            entry.name
            for entry in partition_dir.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        )
        _require(
            actual_objects == sorted(OBJECT_ORDER),
            f"{definition.dataset_key}/{partition}: expected exact object directories "
            f"{sorted(OBJECT_ORDER)}, got {actual_objects}",
        )
        for model in OBJECT_ORDER:
            meta_relpath = f"{partition}/{model}/meta.jsonl"
            _require(
                meta_relpath in files_by_relpath,
                f"{definition.dataset_key}: missing {meta_relpath}",
            )
            rows = _read_jsonl(root / Path(*PurePosixPath(meta_relpath).parts))
            _require(
                len(rows) == expected_count,
                f"{meta_relpath}: expected {expected_count} rows, got {len(rows)}",
            )
            observed_indices = [row.get("frame_index") for row in rows]
            _require(
                observed_indices == list(range(expected_count)),
                f"{meta_relpath}: frame indices must be the exact ordered range "
                f"0..{expected_count-1}",
            )
            for index, row in enumerate(rows):
                context = f"{meta_relpath}:{index+1}"
                _validate_common_metadata(
                    row,
                    definition=definition,
                    partition=partition,
                    model=model,
                    index=index,
                    context=context,
                )
                image_relpath, mask_relpath, id_relpath = _canonical_frame_paths(
                    partition, model, index
                )
                for kind, relpath in (
                    ("image", image_relpath),
                    ("mask", mask_relpath),
                    ("id pass", id_relpath),
                ):
                    _require(
                        relpath in files_by_relpath,
                        f"{context}: missing canonical {kind} {relpath}",
                    )
                if definition.family == "roll":
                    _require(
                        row.get("image_relpath") == f"frames/a/img_{index:04d}.png",
                        f"{context}: metadata image_relpath does not match frame_index",
                    )
                    _require(
                        row.get("mask_relpath") == f"masks/a/mask_{index:04d}.png",
                        f"{context}: metadata mask_relpath does not match frame_index",
                    )
                    _require(
                        row.get("id_relpath") == f"id_passes/a/id_{index:04d}.png",
                        f"{context}: metadata id_relpath does not match frame_index",
                    )
                records.append(
                    {
                        "model_name": model,
                        "source_partition": partition,
                        "frame_index": index,
                        "image_relpath": image_relpath,
                        "mask_relpath": mask_relpath,
                        "id_pass_relpath": id_relpath,
                        "meta_jsonl_relpath": meta_relpath,
                        "meta_line_number": index + 1,
                        "meta_record_sha256": hashlib.sha256(
                            canonical_json_bytes(row)
                        ).hexdigest(),
                        "physical_state": _physical_state(
                            row,
                            definition=definition,
                            partition=partition,
                            index=index,
                            context=context,
                        ),
                    }
                )
    records.sort(
        key=lambda record: (
            record["source_partition"],
            OBJECT_ORDER.index(record["model_name"]),
            record["frame_index"],
        )
    )
    return records


def build_corpus_inventory(dataset_key: str, dataset_root: str | Path) -> bytes:
    """Build canonical inventory bytes from the complete current corpus."""

    _require(dataset_key in CORPUS_DEFINITIONS, f"unknown dataset_key {dataset_key!r}")
    definition = CORPUS_DEFINITIONS[dataset_key]
    root = Path(dataset_root)
    _require(not root.is_symlink(), f"dataset root may not be a symlink: {root}")
    root_resolved = root.resolve(strict=True)
    _require(root_resolved.is_dir(), f"dataset root is not a directory: {root}")
    _require(
        root_resolved.name == definition.dataset_basename,
        f"{dataset_key}: expected basename {definition.dataset_basename!r}, "
        f"got {root_resolved.name!r}",
    )

    files = _scan_regular_files(root_resolved)
    files_by_relpath = {item["relative_path"]: item for item in files}
    _require("semantic_lock.json" in files_by_relpath, f"{dataset_key}: missing semantic_lock.json")
    _read_json_object(root_resolved / "semantic_lock.json")
    frames = _frame_records(root_resolved, definition, files_by_relpath)
    pair_indices = [
        dict(item)
        for item in files
        if PurePosixPath(item["relative_path"]).parts[:1] == ("indices",)
        and item["relative_path"].endswith(".json")
    ]
    _require(pair_indices, f"{dataset_key}: no JSON source pair indices found")

    document: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "artifact_type": INVENTORY_ARTIFACT_TYPE,
        "dataset_key": dataset_key,
        "dataset_basename": definition.dataset_basename,
        "source_root_provenance": str(root_resolved),
        "unversioned_source": True,
        "unversioned_warning": UNVERSIONED_WARNING,
        "ignored_basenames": list(IGNORED_BASENAMES),
        "semantic_lock": {
            "relative_path": "semantic_lock.json",
            "size_bytes": files_by_relpath["semantic_lock.json"]["size_bytes"],
            "sha256": files_by_relpath["semantic_lock.json"]["sha256"],
        },
        "objects": list(OBJECT_ORDER),
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "source_pair_indices": pair_indices,
        "frame_records": frames,
    }
    document["content_hash_sha256"] = inventory_content_hash(document)
    _exact_keys(document, _INVENTORY_KEYS, f"{dataset_key} inventory")
    return canonical_json_bytes(document, trailing_newline=True)


def _validated_view(
    dataset_key: str,
    root: Path,
    document: dict[str, Any],
) -> ValidatedCorpusInventory:
    files_by_relpath: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(document["files"]):
        _require(isinstance(item, dict), f"{dataset_key}.files[{index}]: expected object")
        _exact_keys(item, _FILE_KEYS, f"{dataset_key}.files[{index}]")
        relpath = item["relative_path"]
        _require(isinstance(relpath, str) and relpath != "", f"{dataset_key}: bad file path")
        relative = PurePosixPath(relpath)
        _require(not relative.is_absolute() and ".." not in relative.parts, f"{relpath}: unsafe path")
        _require(_is_int(item["size_bytes"]) and item["size_bytes"] >= 0, f"{relpath}: bad size")
        _require(
            isinstance(item["sha256"], str) and _SHA256_RE.fullmatch(item["sha256"]) is not None,
            f"{relpath}: bad SHA-256",
        )
        _require(relpath not in files_by_relpath, f"{relpath}: duplicate inventory path")
        files_by_relpath[relpath] = item

    frames_by_identity: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, record in enumerate(document["frame_records"]):
        _require(isinstance(record, dict), f"{dataset_key}.frame_records[{index}]: expected object")
        _exact_keys(record, _FRAME_KEYS, f"{dataset_key}.frame_records[{index}]")
        identity = (
            record["source_partition"],
            record["model_name"],
            record["frame_index"],
        )
        _require(identity not in frames_by_identity, f"{dataset_key}: duplicate frame {identity}")
        frames_by_identity[identity] = record
    return ValidatedCorpusInventory(
        dataset_key=dataset_key,
        root=root,
        document=document,
        files_by_relpath=files_by_relpath,
        frames_by_identity=frames_by_identity,
    )


def validate_corpus_inventory(
    inventory_bytes: bytes,
    dataset_key: str,
    dataset_root: str | Path,
) -> ValidatedCorpusInventory:
    """Recompute the current corpus and require exact canonical inventory equality."""

    _require(isinstance(inventory_bytes, bytes), "inventory must be canonical JSON bytes")
    context = f"{dataset_key} corpus inventory"
    document = _decode_json_bytes(inventory_bytes, context)
    _require(
        inventory_bytes == canonical_json_bytes(document, trailing_newline=True),
        f"{context}: bytes are not canonical JSON plus one newline",
    )
    _exact_keys(document, _INVENTORY_KEYS, context)
    _require(document["schema_version"] == INVENTORY_SCHEMA_VERSION, f"{context}: wrong schema")
    _require(document["artifact_type"] == INVENTORY_ARTIFACT_TYPE, f"{context}: wrong type")
    _require(document["dataset_key"] == dataset_key, f"{context}: wrong dataset_key")
    claimed_hash = document["content_hash_sha256"]
    _require(
        isinstance(claimed_hash, str) and _SHA256_RE.fullmatch(claimed_hash) is not None,
        f"{context}: invalid content hash",
    )
    _require(
        claimed_hash == inventory_content_hash(document),
        f"{context}: content hash mismatch",
    )

    fresh_bytes = build_corpus_inventory(dataset_key, dataset_root)
    fresh_document = _decode_json_bytes(fresh_bytes, f"fresh {context}")
    _require(
        document == fresh_document,
        f"{context}: inventory does not match the current exact corpus contents/metadata",
    )
    return _validated_view(dataset_key, Path(dataset_root).resolve(strict=True), document)


def build_all_corpus_inventories(
    dataset_roots: Mapping[str, str | Path],
) -> dict[str, bytes]:
    expected = set(CORPUS_DEFINITIONS)
    _require(
        set(dataset_roots) == expected,
        f"dataset_roots: expected exact keys {sorted(expected)}, got {sorted(dataset_roots)}",
    )
    return {
        key: build_corpus_inventory(key, dataset_roots[key])
        for key in sorted(expected)
    }


__all__ = [
    "CORPUS_DEFINITIONS",
    "CorpusInventoryError",
    "IGNORED_BASENAMES",
    "INVENTORY_ARTIFACT_TYPE",
    "INVENTORY_SCHEMA_VERSION",
    "OBJECT_ORDER",
    "UNVERSIONED_WARNING",
    "ValidatedCorpusInventory",
    "build_all_corpus_inventories",
    "build_corpus_inventory",
    "canonical_json_bytes",
    "inventory_content_hash",
    "validate_corpus_inventory",
]
