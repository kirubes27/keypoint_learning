"""Independent verifier for frozen primary representation split artifacts.

This module intentionally does not import :mod:`representation_splits`.
Frozen constants and predicates are restated here so a defect in generation is
not accepted merely because verification called the same implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from keypoint_net.representation_corpus_inventory import (
    ValidatedCorpusInventory,
    validate_corpus_inventory,
)


_SCHEMA_VERSION = "representation_pair_index.v1"
_ARTIFACT_TYPE = "representation_pair_index"
_GENERATOR_NAME = "keypoint_net.representation_splits.generate_primary_split_artifacts"
_ROLES = {
    "engineers_hammer_vray": "development",
    "b03_banana_01_high": "confirmation",
    "kettle": "confirmation",
    "dewalt_compact_drill_vray": "final_test",
    "b03_trumpet_vray": "final_test",
    "toy_monkey_medium": "final_test",
}
_OBJECTS = tuple(_ROLES)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_TOP_KEYS = frozenset(
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
_GENERATOR_KEYS = frozenset({"name", "commit", "config_sha256"})
_TRANSFORM_KEYS = frozenset(
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
_PARTITION_KEYS = frozenset(
    {"train_frame_indices", "holdout_frame_indices", "guard_frame_indices"}
)
_PAIR_KEYS = frozenset(
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
_ROLL_TOP_KEYS = frozenset(
    {"operator_name", "skip_frames", "delta_theta_deg", "cyclic", "pairs"}
)
_ROLL_PAIR_KEYS = frozenset(
    {
        "model_name",
        "operator_name",
        "skip_frames",
        "delta_theta_deg",
        "cyclic",
        "src_frame_index",
        "dst_frame_index",
        "src_theta_deg",
        "dst_theta_deg",
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    }
)
_ANGLE_TOP_KEYS = frozenset(
    {"skip_frames", "delta_theta_deg", "cyclic", "direction", "pairs"}
)
_ANGLE_PAIR_KEYS = frozenset(
    {
        "model_name",
        "skip_frames",
        "delta_theta_deg",
        "src_frame_index",
        "dst_frame_index",
        "src_theta_deg",
        "dst_theta_deg",
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    }
)
_SCALE_TOP_KEYS = frozenset(
    {"skip_frames", "scale_ratio", "cyclic", "direction", "pairs"}
)
_SCALE_PAIR_KEYS = frozenset(
    {
        "model_name",
        "skip_frames",
        "scale_ratio",
        "delta_log_scale",
        "src_frame_index",
        "dst_frame_index",
        "src_s_rel",
        "dst_s_rel",
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    }
)
_TRANSLATION_TOP_KEYS = frozenset(
    {"direction", "skip_steps", "delta_world", "grid", "world_step", "cyclic", "pairs"}
)
_TRANSLATION_PAIR_KEYS = frozenset(
    {
        "model_name",
        "direction",
        "skip_steps",
        "delta_world",
        "src_frame_index",
        "dst_frame_index",
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    }
)


@dataclass(frozen=True)
class _Group:
    dataset_key: str
    dataset_basename: str
    family: str
    physical_axis: str
    direction: str
    source_relpath: str
    stride: int
    stride_units: str
    cyclic: bool
    expected_2d_family: str
    frame_count: int
    signed_generator: float
    generator_units: str

    @property
    def stem(self) -> str:
        return f"{self.family}__{self.physical_axis}__{self.direction}"


_GROUPS = (
    _Group(
        "roll",
        "_tdw_world_z_roll_base_panel_512_v2",
        "roll",
        "world_z",
        "forward",
        "indices/pairs_skip3_cyclic.json",
        3,
        "frames",
        True,
        "planar_rotation_about_projected_center",
        180,
        6.0,
        "degrees",
    ),
    _Group(
        "yaw",
        "_tdw_world_y_yaw_arc60_step1_base_panel_512_v1",
        "yaw",
        "world_y",
        "forward",
        "indices/pairs_skip6_forward_nocycle.json",
        6,
        "frames",
        False,
        "local_affine_approximation",
        121,
        6.0,
        "degrees",
    ),
    _Group(
        "yaw",
        "_tdw_world_y_yaw_arc60_step1_base_panel_512_v1",
        "yaw",
        "world_y",
        "reverse",
        "indices/pairs_skip6_reverse_nocycle.json",
        6,
        "frames",
        False,
        "local_affine_approximation",
        121,
        -6.0,
        "degrees",
    ),
    _Group(
        "pitch",
        "_tdw_world_x_pitch_arc60_step1_base_panel_512_v1",
        "pitch",
        "world_x",
        "forward",
        "indices/pairs_skip6_forward_nocycle.json",
        6,
        "frames",
        False,
        "local_affine_approximation",
        121,
        6.0,
        "degrees",
    ),
    _Group(
        "pitch",
        "_tdw_world_x_pitch_arc60_step1_base_panel_512_v1",
        "pitch",
        "world_x",
        "reverse",
        "indices/pairs_skip6_reverse_nocycle.json",
        6,
        "frames",
        False,
        "local_affine_approximation",
        121,
        -6.0,
        "degrees",
    ),
    _Group(
        "scale",
        "_tdw_uniform_scale_loghalf_to_one_base_panel_512_v1",
        "scale",
        "uniform",
        "forward",
        "indices/pairs_skip4_forward_nocycle.json",
        4,
        "frames",
        False,
        "uniform_scale_about_projected_center",
        121,
        0.02310490601866489,
        "log_scale",
    ),
    _Group(
        "scale",
        "_tdw_uniform_scale_loghalf_to_one_base_panel_512_v1",
        "scale",
        "uniform",
        "reverse",
        "indices/pairs_skip4_reverse_nocycle.json",
        4,
        "frames",
        False,
        "uniform_scale_about_projected_center",
        121,
        -0.023104906018664894,
        "log_scale",
    ),
    _Group(
        "translation",
        "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
        "translation",
        "world_x",
        "forward",
        "indices/pairs_dx3_forward_nocycle.json",
        3,
        "steps",
        False,
        "image_plane_translation_after_calibration",
        121,
        0.048,
        "world_units",
    ),
    _Group(
        "translation",
        "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
        "translation",
        "world_x",
        "reverse",
        "indices/pairs_dx3_reverse_nocycle.json",
        3,
        "steps",
        False,
        "image_plane_translation_after_calibration",
        121,
        -0.048,
        "world_units",
    ),
    _Group(
        "translation",
        "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
        "translation",
        "world_y",
        "forward",
        "indices/pairs_dy3_forward_nocycle.json",
        3,
        "steps",
        False,
        "image_plane_translation_after_calibration",
        121,
        0.048,
        "world_units",
    ),
    _Group(
        "translation",
        "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
        "translation",
        "world_y",
        "reverse",
        "indices/pairs_dy3_reverse_nocycle.json",
        3,
        "steps",
        False,
        "image_plane_translation_after_calibration",
        121,
        -0.048,
        "world_units",
    ),
)


class SplitVerificationError(ValueError):
    """Raised when an artifact fails any frozen semantic predicate."""


def _fail(message: str) -> None:
    raise SplitVerificationError(message)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _decode(data: bytes, context: str) -> dict[str, Any]:
    _check(isinstance(data, bytes), f"{context}: artifact must be bytes")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitVerificationError(f"{context}: invalid UTF-8 JSON: {exc}") from exc
    _check(isinstance(value, dict), f"{context}: top-level value must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _decode(path.read_bytes(), str(path))
    except OSError as exc:
        raise SplitVerificationError(f"cannot read {path}: {exc}") from exc


def _keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    _check(
        actual == expected,
        f"{context}: schema mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}",
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _near(actual: Any, expected: float, context: str) -> None:
    _check(_finite(actual), f"{context}: expected finite number")
    _check(
        math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{context}: expected {expected!r}, got {actual!r}",
    )


def _partition(group: _Group) -> dict[str, list[int]]:
    if group.family == "roll":
        train = list(range(27, 177))
        holdout = list(range(24))
        guard = list(range(24, 27)) + list(range(177, 180))
    elif group.family in {"yaw", "pitch"}:
        train = list(range(47)) + list(range(74, 121))
        holdout = list(range(53, 68))
        guard = list(range(47, 53)) + list(range(68, 74))
    elif group.family == "scale":
        train = list(range(49)) + list(range(72, 121))
        holdout = list(range(53, 68))
        guard = list(range(49, 53)) + list(range(68, 72))
    elif group.physical_axis == "world_x":
        train = [gy * 11 + gx for gy in range(7) for gx in range(11)]
        guard = [gy * 11 + gx for gy in range(7, 9) for gx in range(11)]
        holdout = [gy * 11 + gx for gy in range(9, 11) for gx in range(11)]
    else:
        train = [gy * 11 + gx for gy in range(11) for gx in range(7)]
        guard = [gy * 11 + gx for gy in range(11) for gx in range(7, 9)]
        holdout = [gy * 11 + gx for gy in range(11) for gx in range(9, 11)]
    _check(
        sorted(train + holdout + guard) == list(range(group.frame_count)),
        f"{group.stem}: independent partition coverage failure",
    )
    return {
        "train_frame_indices": train,
        "holdout_frame_indices": holdout,
        "guard_frame_indices": guard,
    }


def _config_payload() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "object_roles": _ROLES,
        "primary_groups": [
            {
                "dataset_key": group.dataset_key,
                "dataset_basename": group.dataset_basename,
                "family": group.family,
                "physical_axis": group.physical_axis,
                "direction": group.direction,
                "source_pair_index_relpath": group.source_relpath,
                "stride": group.stride,
                "stride_units": group.stride_units,
                "cyclic": group.cyclic,
                "expected_2d_family": group.expected_2d_family,
                "frame_count": group.frame_count,
                "signed_generator": group.signed_generator,
                "generator_units": group.generator_units,
                "train_pairs_per_object": _per_object_count(group, "train"),
                "holdout_pairs_per_object": _per_object_count(group, "validation"),
                **_partition(group),
            }
            for group in _GROUPS
        ],
    }


def _source_edge_set(group: _Group) -> set[tuple[int, int]]:
    if group.family == "roll":
        return {(src, (src + 3) % 180) for src in range(180)}
    if group.family in {"yaw", "pitch"}:
        if group.direction == "forward":
            return {(src, src + 6) for src in range(115)}
        return {(src, src - 6) for src in range(6, 121)}
    if group.family == "scale":
        if group.direction == "forward":
            return {(src, src + 4) for src in range(117)}
        return {(src, src - 4) for src in range(4, 121)}
    edges = set()
    if group.physical_axis == "world_x":
        for gy in range(11):
            for gx in (range(8) if group.direction == "forward" else range(3, 11)):
                src = gy * 11 + gx
                edges.add((src, src + 3 if group.direction == "forward" else src - 3))
    else:
        for gy in (range(8) if group.direction == "forward" else range(3, 11)):
            for gx in range(11):
                src = gy * 11 + gx
                edges.add((src, src + 33 if group.direction == "forward" else src - 33))
    return edges


def _source_schema(group: _Group) -> tuple[frozenset[str], frozenset[str]]:
    if group.family == "roll":
        return _ROLL_TOP_KEYS, _ROLL_PAIR_KEYS
    if group.family in {"yaw", "pitch"}:
        return _ANGLE_TOP_KEYS, _ANGLE_PAIR_KEYS
    if group.family == "scale":
        return _SCALE_TOP_KEYS, _SCALE_PAIR_KEYS
    return _TRANSLATION_TOP_KEYS, _TRANSLATION_PAIR_KEYS


def _check_source_header(source: Mapping[str, Any], group: _Group, context: str) -> None:
    _check(source.get("cyclic") is group.cyclic, f"{context}: source cyclic mismatch")
    _check(isinstance(source.get("pairs"), list), f"{context}.pairs must be a list")
    if group.family == "roll":
        _check(source.get("operator_name") == "tdw_world_z_roll", f"{context}: wrong operator")
        _check(
            _integer(source.get("skip_frames")) and source["skip_frames"] == 3,
            f"{context}: wrong source stride",
        )
        _near(source.get("delta_theta_deg"), 6.0, f"{context}.delta_theta_deg")
    elif group.family in {"yaw", "pitch"}:
        _check(source.get("direction") == group.direction, f"{context}: wrong source direction")
        _check(
            _integer(source.get("skip_frames")) and source["skip_frames"] == 6,
            f"{context}: wrong source stride",
        )
        _near(source.get("delta_theta_deg"), group.signed_generator, f"{context}.delta_theta_deg")
    elif group.family == "scale":
        ratio = 1.023373891996775 if group.direction == "forward" else 0.9771599684342459
        _check(source.get("direction") == group.direction, f"{context}: wrong source direction")
        _check(
            _integer(source.get("skip_frames")) and source["skip_frames"] == 4,
            f"{context}: wrong source stride",
        )
        _near(source.get("scale_ratio"), ratio, f"{context}.scale_ratio")
    else:
        axis = "x" if group.physical_axis == "world_x" else "y"
        _check(source.get("direction") == axis, f"{context}: wrong source axis")
        _check(
            _integer(source.get("skip_steps")) and source["skip_steps"] == 3,
            f"{context}: wrong source stride",
        )
        _check(
            _integer(source.get("grid")) and source["grid"] == 11,
            f"{context}: wrong source grid",
        )
        _near(source.get("world_step"), 0.016, f"{context}.world_step")
        _near(source.get("delta_world"), group.signed_generator, f"{context}.delta_world")


def _check_corpus_path(
    root: Path,
    model: str,
    value: Any,
    context: str,
    path_cache: set[str],
) -> None:
    _check(isinstance(value, str) and value != "", f"{context}: path must be non-empty")
    relative = PurePosixPath(value)
    _check(not relative.is_absolute(), f"{context}: absolute path forbidden")
    _check(".." not in relative.parts, f"{context}: parent traversal forbidden")
    _check(
        len(relative.parts) >= 3
        and relative.parts[0] == "train"
        and relative.parts[1] == model,
        f"{context}: wrong object path prefix",
    )
    if value not in path_cache:
        try:
            root_resolved = root.resolve(strict=True)
            resolved = (root / Path(*relative.parts)).resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise SplitVerificationError(f"{context}: unresolved/escaping path {value!r}") from exc
        _check(resolved.is_file(), f"{context}: path is not a file")
        path_cache.add(value)


def _check_source_semantics(
    pair: Mapping[str, Any],
    group: _Group,
    src: int,
    dst: int,
    src_record: Mapping[str, Any],
    dst_record: Mapping[str, Any],
    context: str,
) -> None:
    src_state = src_record["physical_state"]
    dst_state = dst_record["physical_state"]
    if group.family == "roll":
        _check(pair["operator_name"] == "tdw_world_z_roll", f"{context}: wrong operator")
        _check(
            _integer(pair["skip_frames"])
            and pair["skip_frames"] == 3
            and pair["cyclic"] is True,
            f"{context}: wrong roll flags",
        )
        _near(pair["delta_theta_deg"], 6.0, f"{context}.delta_theta_deg")
        _near(pair["src_theta_deg"], 2.0 * src, f"{context}.src_theta_deg")
        _near(pair["dst_theta_deg"], 2.0 * dst, f"{context}.dst_theta_deg")
        _near(pair["src_theta_deg"], src_state["theta_deg"], f"{context}: source metadata")
        _near(pair["dst_theta_deg"], dst_state["theta_deg"], f"{context}: target metadata")
    elif group.family in {"yaw", "pitch"}:
        _check(
            _integer(pair["skip_frames"]) and pair["skip_frames"] == 6,
            f"{context}: wrong angular stride",
        )
        _near(pair["delta_theta_deg"], group.signed_generator, f"{context}.delta_theta_deg")
        _near(pair["src_theta_deg"], -60.0 + src, f"{context}.src_theta_deg")
        _near(pair["dst_theta_deg"], -60.0 + dst, f"{context}.dst_theta_deg")
        _near(pair["src_theta_deg"], src_state["theta_deg"], f"{context}: source metadata")
        _near(pair["dst_theta_deg"], dst_state["theta_deg"], f"{context}: target metadata")
    elif group.family == "scale":
        ratio = 1.023373891996775 if group.direction == "forward" else 0.9771599684342459
        _check(
            _integer(pair["skip_frames"]) and pair["skip_frames"] == 4,
            f"{context}: wrong scale stride",
        )
        _near(pair["scale_ratio"], ratio, f"{context}.scale_ratio")
        _near(pair["delta_log_scale"], group.signed_generator, f"{context}.delta_log_scale")
        _check(
            _finite(pair["src_s_rel"])
            and _finite(pair["dst_s_rel"])
            and float(pair["src_s_rel"]) > 0,
            f"{context}: invalid scale state",
        )
        _near(
            float(pair["dst_s_rel"]) / float(pair["src_s_rel"]),
            ratio,
            f"{context}: endpoint scale ratio",
        )
        _near(pair["src_s_rel"], src_state["s_rel"], f"{context}: source metadata")
        _near(pair["dst_s_rel"], dst_state["s_rel"], f"{context}: target metadata")
    else:
        axis = "x" if group.physical_axis == "world_x" else "y"
        _check(pair["direction"] == axis, f"{context}: wrong translation axis")
        _check(
            _integer(pair["skip_steps"]) and pair["skip_steps"] == 3,
            f"{context}: wrong translation stride",
        )
        _near(pair["delta_world"], group.signed_generator, f"{context}.delta_world")
        if group.physical_axis == "world_x":
            _near(
                float(dst_state["dx_world"]) - float(src_state["dx_world"]),
                group.signed_generator,
                f"{context}: metadata x displacement",
            )
            _near(
                float(dst_state["dy_world"]) - float(src_state["dy_world"]),
                0.0,
                f"{context}: orthogonal metadata y displacement",
            )
        else:
            _near(
                float(dst_state["dy_world"]) - float(src_state["dy_world"]),
                group.signed_generator,
                f"{context}: metadata y displacement",
            )
            _near(
                float(dst_state["dx_world"]) - float(src_state["dx_world"]),
                0.0,
                f"{context}: orthogonal metadata x displacement",
            )


def _load_and_check_source(
    root: Path,
    group: _Group,
    inventory: ValidatedCorpusInventory,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], str]:
    context = str(root / group.source_relpath)
    _check(root.is_dir(), f"{context}: dataset root missing")
    _check(root.name == group.dataset_basename, f"{context}: dataset basename mismatch")
    _check(
        inventory.dataset_key == group.dataset_key and inventory.root == root.resolve(strict=True),
        f"{context}: validated inventory does not bind this corpus root",
    )
    source_path = root / group.source_relpath
    source = _read_json(source_path)
    top_schema, row_schema = _source_schema(group)
    _keys(source, top_schema, context)
    _check_source_header(source, group, context)

    expected_edges = _source_edge_set(group)
    expected_total = len(_ROLES) * len(expected_edges)
    _check(len(source["pairs"]) == expected_total, f"{context}: wrong source row count")
    by_identity: dict[tuple[str, int, int], dict[str, Any]] = {}
    observed = {model: set() for model in _ROLES}
    path_cache: set[str] = set()
    for index, raw in enumerate(source["pairs"]):
        row_context = f"{context}: pair[{index}]"
        _check(isinstance(raw, dict), f"{row_context}: row must be an object")
        _keys(raw, row_schema, row_context)
        model = raw["model_name"]
        src = raw["src_frame_index"]
        dst = raw["dst_frame_index"]
        _check(
            isinstance(model, str) and model in _ROLES,
            f"{row_context}: unknown model",
        )
        _check(_integer(src) and _integer(dst), f"{row_context}: non-integer frame index")
        _check((src, dst) in expected_edges, f"{row_context}: non-primary edge")
        identity = (model, src, dst)
        _check(identity not in by_identity, f"{row_context}: duplicate edge")
        src_record = inventory.frame(model, src)
        dst_record = inventory.frame(model, dst)
        expected_paths = {
            "src_image_relpath": src_record["image_relpath"],
            "dst_image_relpath": dst_record["image_relpath"],
            "src_mask_relpath": src_record["mask_relpath"],
            "dst_mask_relpath": dst_record["mask_relpath"],
        }
        for key, expected_path in expected_paths.items():
            _check(
                raw[key] == expected_path,
                f"{row_context}.{key}: does not match metadata-bound frame path",
            )
            _check_corpus_path(root, model, raw[key], f"{row_context}.{key}", path_cache)
        _check_source_semantics(
            raw,
            group,
            src,
            dst,
            src_record,
            dst_record,
            row_context,
        )
        by_identity[identity] = raw
        observed[model].add((src, dst))
    for model, edges in observed.items():
        _check(edges == expected_edges, f"{context}: incomplete source edge set for {model}")
    return by_identity, _sha_file(source_path)


def _expected_objects(split: str) -> list[str]:
    if split == "train":
        return list(_OBJECTS)
    if split == "validation":
        return ["engineers_hammer_vray"]
    return [model for model in _OBJECTS if model != "engineers_hammer_vray"]


def _per_object_count(group: _Group, split: str) -> int:
    holdout = split != "train"
    if group.family == "roll":
        return 21 if holdout else 147
    if group.family in {"yaw", "pitch"}:
        return 9 if holdout else 82
    if group.family == "scale":
        return 11 if holdout else 90
    return 16 if holdout else 56


_CONFIG_SHA = hashlib.sha256(_canonical(_config_payload())).hexdigest()


def _expected_state(
    group: _Group,
    source: Mapping[str, Any],
    endpoint: str,
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    model = str(source["model_name"])
    index = int(source[f"{endpoint}_frame_index"])
    state = inventory.frame(model, index)["physical_state"]
    if group.family in {"roll", "yaw", "pitch"}:
        return {"theta_deg": state["theta_deg"]}
    if group.family == "scale":
        return {"s_rel": state["s_rel"]}
    return {
        "grid_x": state["grid_x"],
        "grid_y": state["grid_y"],
        "dx_world": state["dx_world"],
        "dy_world": state["dy_world"],
    }


def _expected_transform(group: _Group) -> dict[str, Any]:
    return {
        "family": group.family,
        "physical_axis": group.physical_axis,
        "direction": group.direction,
        "signed_generator": group.signed_generator,
        "generator_units": group.generator_units,
        "stride": group.stride,
        "stride_units": group.stride_units,
        "cyclic": group.cyclic,
        "expected_2d_family": group.expected_2d_family,
    }


def _check_partition_distance(group: _Group, partition: Mapping[str, list[int]]) -> None:
    train = partition["train_frame_indices"]
    holdout = partition["holdout_frame_indices"]
    if group.family == "translation":
        coordinate = (
            (lambda index: index // 11)
            if group.physical_axis == "world_x"
            else (lambda index: index % 11)
        )
        distance = min(
            abs(coordinate(left) - coordinate(right)) for left in train for right in holdout
        )
    elif group.cyclic:
        distance = min(
            min(abs(left - right), group.frame_count - abs(left - right))
            for left in train
            for right in holdout
        )
    else:
        distance = min(abs(left - right) for left in train for right in holdout)
    _check(
        distance >= group.stride,
        f"{group.stem}: nearest train/holdout context {distance} is below stride {group.stride}",
    )


def _check_pair(
    pair: Mapping[str, Any],
    source: Mapping[str, Any],
    group: _Group,
    split: str,
    inventory: ValidatedCorpusInventory,
    context: str,
) -> None:
    _keys(pair, _PAIR_KEYS, context)
    _check(_integer(pair["stride"]), f"{context}.stride must be an integer")
    _check(isinstance(pair["cyclic"], bool), f"{context}.cyclic must be boolean")
    _check(_finite(pair["signed_generator"]), f"{context}.signed_generator must be finite")
    model = source["model_name"]
    src = source["src_frame_index"]
    dst = source["dst_frame_index"]
    expected_id = (
        f"{group.family}:{group.physical_axis}:{group.direction}:"
        f"{model}:{split}:{src:04d}:{dst:04d}"
    )
    expected_scalar_fields = {
        "pair_id": expected_id,
        "model_name": model,
        "object_role": _ROLES[model],
        "split": split,
        "transform_family": group.family,
        "physical_axis": group.physical_axis,
        "direction": group.direction,
        "signed_generator": group.signed_generator,
        "generator_units": group.generator_units,
        "stride": group.stride,
        "stride_units": group.stride_units,
        "cyclic": group.cyclic,
        "src_frame_index": src,
        "dst_frame_index": dst,
    }
    for key, expected in expected_scalar_fields.items():
        _check(pair[key] == expected, f"{context}.{key}: expected {expected!r}, got {pair[key]!r}")
    _check(
        pair["src_state"] == _expected_state(group, source, "src", inventory),
        f"{context}: bad src_state",
    )
    _check(
        pair["dst_state"] == _expected_state(group, source, "dst", inventory),
        f"{context}: bad dst_state",
    )
    for key in (
        "src_image_relpath",
        "dst_image_relpath",
        "src_mask_relpath",
        "dst_mask_relpath",
    ):
        _check(pair[key] == source[key], f"{context}.{key}: does not match bound source")
        _check("eval_abs_holdout" not in PurePosixPath(pair[key]).parts, f"{context}: scale holdout leak")


def verify_primary_split_artifacts(
    artifacts: Mapping[str, bytes],
    dataset_roots: Mapping[str, str | Path],
    *,
    corpus_inventories: Mapping[str, bytes],
    expected_generator_commit: str,
    regenerated_artifacts: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Independently verify all primary split artifacts and return a report.

    If ``regenerated_artifacts`` is supplied, every filename and byte must be
    identical.  This makes the byte-identical regeneration predicate explicit
    rather than inferring it from valid JSON or matching hashes.
    """

    dataset_keys = {group.dataset_key for group in _GROUPS}
    for label, mapping in (
        ("dataset_roots", dataset_roots),
        ("corpus_inventories", corpus_inventories),
    ):
        _check(set(mapping) == dataset_keys, f"{label}: wrong dataset keys")
    _check(
        isinstance(expected_generator_commit, str)
        and _COMMIT_RE.fullmatch(expected_generator_commit) is not None,
        "expected_generator_commit must be lowercase 40-hex",
    )
    for key in dataset_keys:
        _check(isinstance(corpus_inventories[key], bytes), f"invalid inventory bytes for {key}")

    expected_names = {
        f"{group.stem}__{split}.json"
        for group in _GROUPS
        for split in ("train", "validation", "test")
    }
    _check(
        set(artifacts) == expected_names,
        f"artifact filename set mismatch; missing={sorted(expected_names-set(artifacts))}, "
        f"extra={sorted(set(artifacts)-expected_names)}",
    )
    if regenerated_artifacts is not None:
        _check(set(regenerated_artifacts) == expected_names, "regeneration filename set mismatch")
        for filename in sorted(expected_names):
            _check(
                artifacts[filename] == regenerated_artifacts[filename],
                f"{filename}: regeneration is not byte-identical",
            )

    roots = {key: Path(value) for key, value in dataset_roots.items()}
    validated_inventories = {
        key: validate_corpus_inventory(corpus_inventories[key], key, roots[key])
        for key in sorted(roots)
    }
    directed_sets: dict[tuple[str, str, str, str], set[tuple[str, int, int]]] = {}
    content_hashes: dict[str, str] = {}
    pair_counts: dict[str, int] = {}

    for group in _GROUPS:
        root = roots[group.dataset_key]
        inventory = validated_inventories[group.dataset_key]
        _check(root.name == group.dataset_basename, f"{group.dataset_key}: wrong dataset basename")
        semantic_lock = root / "semantic_lock.json"
        _check(semantic_lock.is_file(), f"{group.dataset_key}: missing semantic_lock.json")
        semantic_sha = _sha_file(semantic_lock)
        source_by_identity, source_sha = _load_and_check_source(root, group, inventory)
        _check(
            source_sha
            == inventory.files_by_relpath[group.source_relpath]["sha256"],
            f"{group.stem}: source pair index differs from validated inventory",
        )
        partition = _partition(group)
        _keys(partition, _PARTITION_KEYS, f"{group.stem}.partition")
        _check_partition_distance(group, partition)
        train_set = set(partition["train_frame_indices"])
        holdout_set = set(partition["holdout_frame_indices"])
        guard_set = set(partition["guard_frame_indices"])

        for split in ("train", "validation", "test"):
            filename = f"{group.stem}__{split}.json"
            artifact = _decode(artifacts[filename], filename)
            _check(
                artifacts[filename] == _canonical(artifact, newline=True),
                f"{filename}: bytes are not canonical JSON plus one newline",
            )
            _keys(artifact, _TOP_KEYS, filename)
            claimed_hash = artifact["content_hash_sha256"]
            _check(
                isinstance(claimed_hash, str) and _SHA_RE.fullmatch(claimed_hash) is not None,
                f"{filename}: invalid content hash syntax",
            )
            hash_payload = dict(artifact)
            del hash_payload["content_hash_sha256"]
            recomputed_hash = hashlib.sha256(_canonical(hash_payload)).hexdigest()
            _check(claimed_hash == recomputed_hash, f"{filename}: content hash mismatch")
            content_hashes[filename] = claimed_hash

            expected_objects = _expected_objects(split)
            expected_top = {
                "schema_version": _SCHEMA_VERSION,
                "artifact_type": _ARTIFACT_TYPE,
                "dataset_basename": group.dataset_basename,
                "dataset_binding_sha256": inventory.content_hash_sha256,
                "dataset_semantic_lock_sha256": semantic_sha,
                "source_pair_index_relpath": group.source_relpath,
                "source_pair_index_sha256": source_sha,
                "unversioned_source": inventory.document["unversioned_source"],
                "split": split,
                "object_roles": _ROLES,
                "included_objects": expected_objects,
                "frame_partition": partition,
            }
            for key, expected in expected_top.items():
                _check(
                    artifact[key] == expected,
                    f"{filename}.{key}: expected {expected!r}, got {artifact[key]!r}",
                )
            _check(
                isinstance(artifact["frame_partition"], dict),
                f"{filename}.frame_partition must be an object",
            )
            _keys(
                artifact["frame_partition"],
                _PARTITION_KEYS,
                f"{filename}.frame_partition",
            )
            for partition_name, indices in artifact["frame_partition"].items():
                _check(
                    isinstance(indices, list)
                    and all(_integer(index) for index in indices)
                    and len(indices) == len(set(indices)),
                    f"{filename}.frame_partition.{partition_name} must contain unique integers",
                )
            _check(
                isinstance(artifact["generator"], dict),
                f"{filename}.generator must be an object",
            )
            _keys(artifact["generator"], _GENERATOR_KEYS, f"{filename}.generator")
            _check(artifact["generator"]["name"] == _GENERATOR_NAME, f"{filename}: wrong generator")
            _check(
                artifact["generator"]["commit"] == expected_generator_commit,
                f"{filename}: wrong generator commit",
            )
            _check(
                artifact["generator"]["config_sha256"] == _CONFIG_SHA,
                f"{filename}: wrong generator config hash",
            )
            _check(
                isinstance(artifact["transform"], dict),
                f"{filename}.transform must be an object",
            )
            _keys(artifact["transform"], _TRANSFORM_KEYS, f"{filename}.transform")
            _check(
                _integer(artifact["transform"]["stride"]),
                f"{filename}.transform.stride must be an integer",
            )
            _check(
                isinstance(artifact["transform"]["cyclic"], bool),
                f"{filename}.transform.cyclic must be boolean",
            )
            _check(
                _finite(artifact["transform"]["signed_generator"]),
                f"{filename}.transform.signed_generator must be finite",
            )
            _check(
                artifact["transform"] == _expected_transform(group),
                f"{filename}: transform metadata mismatch",
            )

            expected_per_object = _per_object_count(group, split)
            expected_counts = {model: expected_per_object for model in expected_objects}
            _check(
                isinstance(artifact["pair_counts_by_object"], dict)
                and all(
                    isinstance(model, str) and _integer(count)
                    for model, count in artifact["pair_counts_by_object"].items()
                ),
                f"{filename}: pair_counts_by_object must map strings to integers",
            )
            _check(
                artifact["pair_counts_by_object"] == expected_counts,
                f"{filename}: per-object count mapping mismatch",
            )
            expected_total = expected_per_object * len(expected_objects)
            _check(
                _integer(artifact["pair_count"]) and artifact["pair_count"] == expected_total,
                f"{filename}: wrong pair_count",
            )
            _check(
                isinstance(artifact["pairs"], list) and len(artifact["pairs"]) == expected_total,
                f"{filename}: pairs length mismatch",
            )

            active_set = train_set if split == "train" else holdout_set
            expected_identities = {
                identity
                for identity in source_by_identity
                if identity[0] in expected_objects
                and identity[1] in active_set
                and identity[2] in active_set
            }
            actual_identities: list[tuple[str, int, int]] = []
            pair_ids: set[str] = set()
            for index, pair in enumerate(artifact["pairs"]):
                context = f"{filename}.pairs[{index}]"
                _check(isinstance(pair, dict), f"{context}: pair must be an object")
                _check(
                    isinstance(pair.get("model_name"), str)
                    and _integer(pair.get("src_frame_index"))
                    and _integer(pair.get("dst_frame_index")),
                    f"{context}: model and frame-index types are invalid",
                )
                identity = (
                    pair.get("model_name"),
                    pair.get("src_frame_index"),
                    pair.get("dst_frame_index"),
                )
                _check(identity in source_by_identity, f"{context}: row absent from bound source")
                _check(
                    identity[1] in active_set and identity[2] in active_set,
                    f"{context}: endpoint outside named split",
                )
                _check(
                    identity[1] not in guard_set and identity[2] not in guard_set,
                    f"{context}: guard endpoint retained",
                )
                _check_pair(
                    pair,
                    source_by_identity[identity],
                    group,
                    split,
                    inventory,
                    context,
                )
                _check(pair["pair_id"] not in pair_ids, f"{context}: duplicate pair_id")
                pair_ids.add(pair["pair_id"])
                actual_identities.append(identity)

            object_position = {model: i for i, model in enumerate(expected_objects)}
            expected_order = sorted(
                expected_identities,
                key=lambda identity: (
                    object_position[identity[0]],
                    identity[1],
                    identity[2],
                ),
            )
            _check(actual_identities == expected_order, f"{filename}: pair order/set mismatch")
            pair_counts[filename] = len(actual_identities)
            directed_sets[(group.family, group.physical_axis, split, group.direction)] = set(
                actual_identities
            )

    for family, axis in (
        ("yaw", "world_y"),
        ("pitch", "world_x"),
        ("scale", "uniform"),
        ("translation", "world_x"),
        ("translation", "world_y"),
    ):
        for split in ("train", "validation", "test"):
            forward = directed_sets[(family, axis, split, "forward")]
            reverse = directed_sets[(family, axis, split, "reverse")]
            reversed_forward = {(model, dst, src) for model, src, dst in forward}
            _check(
                reverse == reversed_forward,
                f"{family}/{axis}/{split}: reverse pair file changes endpoint partition",
            )

    return {
        "schema_version": "representation_split_verification_report.v1",
        "artifact_type": "representation_split_verification_report",
        "structurally_valid": True,
        "validated_corpus_inventories": True,
        "artifact_count": len(artifacts),
        "content_hashes": dict(sorted(content_hashes.items())),
        "pair_counts": dict(sorted(pair_counts.items())),
        "scale_abs_holdout_nonmembership": True,
        "byte_identical_regeneration": regenerated_artifacts is not None,
        "gate_pass": regenerated_artifacts is not None,
        "generator_config_sha256": _CONFIG_SHA,
    }


__all__ = ["SplitVerificationError", "verify_primary_split_artifacts"]
