"""Deterministic primary representation split generation.

This module is deliberately standard-library-only.  It reads the immutable
source pair indices, validates their transformation semantics, and returns
canonical JSON bytes.  It never writes, copies, opens, or rerenders an image.

The caller must supply the SHA-256 binding of each complete dataset inventory.
That binding is intentionally not inferred from the pair index: the frozen
specification requires a separate content manifest covering frames, masks,
metadata, and source indices.
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


SCHEMA_VERSION = "representation_pair_index.v1"
ARTIFACT_TYPE = "representation_pair_index"
GENERATOR_NAME = "keypoint_net.representation_splits.generate_primary_split_artifacts"

OBJECT_ROLES: dict[str, str] = {
    "engineers_hammer_vray": "development",
    "b03_banana_01_high": "confirmation",
    "kettle": "confirmation",
    "dewalt_compact_drill_vray": "final_test",
    "b03_trumpet_vray": "final_test",
    "toy_monkey_medium": "final_test",
}
OBJECT_ORDER = tuple(OBJECT_ROLES)

TOP_LEVEL_KEYS = frozenset(
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
TRANSFORM_KEYS = frozenset(
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
PAIR_KEYS = frozenset(
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

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
class PrimaryGroup:
    dataset_key: str
    dataset_basename: str
    family: str
    physical_axis: str
    direction: str
    source_pair_index_relpath: str
    stride: int
    stride_units: str
    cyclic: bool
    expected_2d_family: str
    frame_count: int

    @property
    def artifact_stem(self) -> str:
        return f"{self.family}__{self.physical_axis}__{self.direction}"


PRIMARY_GROUPS = (
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
    PrimaryGroup(
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
    ),
)


class SplitGenerationError(ValueError):
    """Raised when a source corpus violates the frozen split contract."""


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Return the frozen canonical JSON encoding.

    Content hashes use ``trailing_newline=False``.  Artifact serialization uses
    one trailing newline for normal text-file behavior.
    """

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


def content_hash_sha256(artifact_without_or_with_hash: Mapping[str, Any]) -> str:
    """Hash the full artifact after omitting only its top-level content hash."""

    payload = dict(artifact_without_or_with_hash)
    payload.pop("content_hash_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SplitGenerationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_no_duplicate_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitGenerationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitGenerationError(f"{path}: top-level JSON value must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SplitGenerationError(f"{context}: schema mismatch; missing={missing}, extra={extra}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitGenerationError(message)


def _close(actual: Any, expected: float, context: str, *, tolerance: float = 1e-12) -> None:
    _require(_is_number(actual), f"{context}: expected a finite number, got {actual!r}")
    _require(
        math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{context}: expected {expected!r}, got {actual!r}",
    )


def _partition(group: PrimaryGroup) -> dict[str, list[int]]:
    if group.family == "roll":
        train = list(range(27, 177))
        holdout = list(range(0, 24))
        guard = list(range(24, 27)) + list(range(177, 180))
    elif group.family in {"yaw", "pitch"}:
        train = list(range(0, 47)) + list(range(74, 121))
        holdout = list(range(53, 68))
        guard = list(range(47, 53)) + list(range(68, 74))
    elif group.family == "scale":
        train = list(range(0, 49)) + list(range(72, 121))
        holdout = list(range(53, 68))
        guard = list(range(49, 53)) + list(range(68, 72))
    elif group.family == "translation":
        axis = group.physical_axis
        if axis == "world_x":
            train = [gy * 11 + gx for gy in range(0, 7) for gx in range(11)]
            guard = [gy * 11 + gx for gy in range(7, 9) for gx in range(11)]
            holdout = [gy * 11 + gx for gy in range(9, 11) for gx in range(11)]
        elif axis == "world_y":
            train = [gy * 11 + gx for gy in range(11) for gx in range(0, 7)]
            guard = [gy * 11 + gx for gy in range(11) for gx in range(7, 9)]
            holdout = [gy * 11 + gx for gy in range(11) for gx in range(9, 11)]
        else:  # pragma: no cover - constant table construction protects this
            raise SplitGenerationError(f"unknown translation axis: {axis}")
    else:  # pragma: no cover - constant table construction protects this
        raise SplitGenerationError(f"unknown family: {group.family}")

    _require(
        set(train).isdisjoint(holdout)
        and set(train).isdisjoint(guard)
        and set(holdout).isdisjoint(guard),
        f"{group.artifact_stem}: frozen frame partitions overlap",
    )
    _require(
        sorted(train + holdout + guard) == list(range(group.frame_count)),
        f"{group.artifact_stem}: frozen frame partitions do not cover the corpus",
    )
    return {
        "train_frame_indices": train,
        "holdout_frame_indices": holdout,
        "guard_frame_indices": guard,
    }


def _configuration_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "object_roles": OBJECT_ROLES,
        "primary_groups": [
            {
                "dataset_key": group.dataset_key,
                "dataset_basename": group.dataset_basename,
                "family": group.family,
                "physical_axis": group.physical_axis,
                "direction": group.direction,
                "source_pair_index_relpath": group.source_pair_index_relpath,
                "stride": group.stride,
                "stride_units": group.stride_units,
                "cyclic": group.cyclic,
                "expected_2d_family": group.expected_2d_family,
                "frame_count": group.frame_count,
                "signed_generator": _frozen_signed_generator(group)[0],
                "generator_units": _frozen_signed_generator(group)[1],
                "train_pairs_per_object": _expected_pair_count(group, "train"),
                "holdout_pairs_per_object": _expected_pair_count(group, "validation"),
                **_partition(group),
            }
            for group in PRIMARY_GROUPS
        ],
    }


def _expected_source_pairs(group: PrimaryGroup) -> set[tuple[int, int]]:
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
    if group.family == "translation":
        expected: set[tuple[int, int]] = set()
        if group.physical_axis == "world_x":
            for gy in range(11):
                starts = range(0, 8) if group.direction == "forward" else range(3, 11)
                for gx in starts:
                    src = gy * 11 + gx
                    dst = src + 3 if group.direction == "forward" else src - 3
                    expected.add((src, dst))
        else:
            starts = range(0, 8) if group.direction == "forward" else range(3, 11)
            for gy in starts:
                for gx in range(11):
                    src = gy * 11 + gx
                    dst = src + 33 if group.direction == "forward" else src - 33
                    expected.add((src, dst))
        return expected
    raise SplitGenerationError(f"unknown family: {group.family}")


def _validate_relative_corpus_path(
    root: Path,
    model_name: str,
    value: Any,
    context: str,
    checked_paths: set[str],
) -> str:
    _require(isinstance(value, str) and value != "", f"{context}: expected a non-empty string")
    relative = PurePosixPath(value)
    _require(not relative.is_absolute(), f"{context}: absolute paths are forbidden")
    _require(".." not in relative.parts, f"{context}: parent traversal is forbidden")
    _require(
        len(relative.parts) >= 3
        and relative.parts[0] == "train"
        and relative.parts[1] == model_name,
        f"{context}: path must begin with train/{model_name}/",
    )
    if value not in checked_paths:
        root_resolved = root.resolve(strict=True)
        candidate = (root / Path(*relative.parts)).resolve(strict=True)
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise SplitGenerationError(f"{context}: path escapes dataset root") from exc
        _require(candidate.is_file(), f"{context}: path is not a file: {value}")
        checked_paths.add(value)
    return value


def _validate_common_pair(
    pair: Mapping[str, Any],
    group: PrimaryGroup,
    root: Path,
    inventory: ValidatedCorpusInventory,
    row_index: int,
    checked_paths: set[str],
) -> tuple[str, int, int, dict[str, Any], dict[str, Any]]:
    context = f"{group.source_pair_index_relpath}: pair[{row_index}]"
    model = pair.get("model_name")
    _require(
        isinstance(model, str) and model in OBJECT_ROLES,
        f"{context}: unknown model_name {model!r}",
    )
    src = pair.get("src_frame_index")
    dst = pair.get("dst_frame_index")
    _require(_is_int(src), f"{context}: src_frame_index must be an integer")
    _require(_is_int(dst), f"{context}: dst_frame_index must be an integer")
    _require(0 <= src < group.frame_count, f"{context}: source frame out of range")
    _require(0 <= dst < group.frame_count, f"{context}: destination frame out of range")
    src_record = inventory.frame(model, src)
    dst_record = inventory.frame(model, dst)
    expected_paths = {
        "src_image_relpath": src_record["image_relpath"],
        "dst_image_relpath": dst_record["image_relpath"],
        "src_mask_relpath": src_record["mask_relpath"],
        "dst_mask_relpath": dst_record["mask_relpath"],
    }
    for key, expected_path in expected_paths.items():
        _require(
            pair.get(key) == expected_path,
            f"{context}.{key}: expected canonical path {expected_path!r} for frame index",
        )
        _validate_relative_corpus_path(root, model, pair.get(key), f"{context}.{key}", checked_paths)
    return model, src, dst, src_record, dst_record


def _source_schema(group: PrimaryGroup) -> tuple[frozenset[str], frozenset[str]]:
    if group.family == "roll":
        return _ROLL_TOP_KEYS, _ROLL_PAIR_KEYS
    if group.family in {"yaw", "pitch"}:
        return _ANGLE_TOP_KEYS, _ANGLE_PAIR_KEYS
    if group.family == "scale":
        return _SCALE_TOP_KEYS, _SCALE_PAIR_KEYS
    if group.family == "translation":
        return _TRANSLATION_TOP_KEYS, _TRANSLATION_PAIR_KEYS
    raise SplitGenerationError(f"unknown family: {group.family}")


def _validate_source_header(document: Mapping[str, Any], group: PrimaryGroup, context: str) -> None:
    _require(isinstance(document.get("pairs"), list), f"{context}.pairs must be a list")
    _require(document.get("cyclic") is group.cyclic, f"{context}: wrong cyclic flag")
    if group.family == "roll":
        _require(document.get("operator_name") == "tdw_world_z_roll", f"{context}: wrong operator")
        _require(
            _is_int(document.get("skip_frames")) and document["skip_frames"] == 3,
            f"{context}: wrong skip_frames",
        )
        _close(document.get("delta_theta_deg"), 6.0, f"{context}.delta_theta_deg")
    elif group.family in {"yaw", "pitch"}:
        signed = 6.0 if group.direction == "forward" else -6.0
        _require(document.get("direction") == group.direction, f"{context}: wrong direction")
        _require(
            _is_int(document.get("skip_frames")) and document["skip_frames"] == 6,
            f"{context}: wrong skip_frames",
        )
        _close(document.get("delta_theta_deg"), signed, f"{context}.delta_theta_deg")
    elif group.family == "scale":
        expected_ratio = (
            1.023373891996775 if group.direction == "forward" else 0.9771599684342459
        )
        _require(document.get("direction") == group.direction, f"{context}: wrong direction")
        _require(
            _is_int(document.get("skip_frames")) and document["skip_frames"] == 4,
            f"{context}: wrong skip_frames",
        )
        _close(document.get("scale_ratio"), expected_ratio, f"{context}.scale_ratio")
    elif group.family == "translation":
        axis = "x" if group.physical_axis == "world_x" else "y"
        signed = 0.048 if group.direction == "forward" else -0.048
        _require(document.get("direction") == axis, f"{context}: wrong physical axis")
        _require(
            _is_int(document.get("skip_steps")) and document["skip_steps"] == 3,
            f"{context}: wrong skip_steps",
        )
        _require(
            _is_int(document.get("grid")) and document["grid"] == 11,
            f"{context}: wrong grid size",
        )
        _close(document.get("world_step"), 0.016, f"{context}.world_step")
        _close(document.get("delta_world"), signed, f"{context}.delta_world")


def _validate_source_pair_semantics(
    pair: Mapping[str, Any],
    group: PrimaryGroup,
    src: int,
    dst: int,
    src_record: Mapping[str, Any],
    dst_record: Mapping[str, Any],
    context: str,
) -> None:
    src_state = src_record["physical_state"]
    dst_state = dst_record["physical_state"]
    if group.family == "roll":
        _require(pair["operator_name"] == "tdw_world_z_roll", f"{context}: wrong operator")
        _require(
            _is_int(pair["skip_frames"]) and pair["skip_frames"] == 3,
            f"{context}: wrong pair skip",
        )
        _require(pair["cyclic"] is True, f"{context}: wrong pair cyclic flag")
        _close(pair["delta_theta_deg"], 6.0, f"{context}.delta_theta_deg")
        _close(pair["src_theta_deg"], 2.0 * src, f"{context}.src_theta_deg")
        _close(pair["dst_theta_deg"], 2.0 * dst, f"{context}.dst_theta_deg")
        _close(pair["src_theta_deg"], src_state["theta_deg"], f"{context}: source metadata")
        _close(pair["dst_theta_deg"], dst_state["theta_deg"], f"{context}: target metadata")
    elif group.family in {"yaw", "pitch"}:
        signed = 6.0 if group.direction == "forward" else -6.0
        _require(
            _is_int(pair["skip_frames"]) and pair["skip_frames"] == 6,
            f"{context}: wrong pair skip",
        )
        _close(pair["delta_theta_deg"], signed, f"{context}.delta_theta_deg")
        _close(pair["src_theta_deg"], -60.0 + src, f"{context}.src_theta_deg")
        _close(pair["dst_theta_deg"], -60.0 + dst, f"{context}.dst_theta_deg")
        _close(pair["src_theta_deg"], src_state["theta_deg"], f"{context}: source metadata")
        _close(pair["dst_theta_deg"], dst_state["theta_deg"], f"{context}: target metadata")
    elif group.family == "scale":
        ratio = 1.023373891996775 if group.direction == "forward" else 0.9771599684342459
        delta = 0.02310490601866489 if group.direction == "forward" else -0.023104906018664894
        _require(
            _is_int(pair["skip_frames"]) and pair["skip_frames"] == 4,
            f"{context}: wrong pair skip",
        )
        _close(pair["scale_ratio"], ratio, f"{context}.scale_ratio")
        _close(pair["delta_log_scale"], delta, f"{context}.delta_log_scale")
        _require(
            _is_number(pair["src_s_rel"]) and float(pair["src_s_rel"]) > 0,
            f"{context}.src_s_rel must be positive and finite",
        )
        _require(
            _is_number(pair["dst_s_rel"]) and float(pair["dst_s_rel"]) > 0,
            f"{context}.dst_s_rel must be positive and finite",
        )
        _close(
            float(pair["dst_s_rel"]) / float(pair["src_s_rel"]),
            ratio,
            f"{context}: endpoint scale ratio",
        )
        _close(pair["src_s_rel"], src_state["s_rel"], f"{context}: source metadata")
        _close(pair["dst_s_rel"], dst_state["s_rel"], f"{context}: target metadata")
    elif group.family == "translation":
        axis = "x" if group.physical_axis == "world_x" else "y"
        signed = 0.048 if group.direction == "forward" else -0.048
        _require(pair["direction"] == axis, f"{context}: wrong pair axis")
        _require(
            _is_int(pair["skip_steps"]) and pair["skip_steps"] == 3,
            f"{context}: wrong pair skip",
        )
        _close(pair["delta_world"], signed, f"{context}.delta_world")
        if group.physical_axis == "world_x":
            _close(
                float(dst_state["dx_world"]) - float(src_state["dx_world"]),
                signed,
                f"{context}: metadata x displacement",
            )
            _close(
                float(dst_state["dy_world"]) - float(src_state["dy_world"]),
                0.0,
                f"{context}: orthogonal metadata y displacement",
            )
        else:
            _close(
                float(dst_state["dy_world"]) - float(src_state["dy_world"]),
                signed,
                f"{context}: metadata y displacement",
            )
            _close(
                float(dst_state["dx_world"]) - float(src_state["dx_world"]),
                0.0,
                f"{context}: orthogonal metadata x displacement",
            )


def validate_source_pair_document(
    document: Mapping[str, Any],
    group: PrimaryGroup,
    dataset_root: str | Path,
    *,
    inventory: ValidatedCorpusInventory,
) -> list[dict[str, Any]]:
    """Strictly validate one immutable primary source-pair document."""

    root = Path(dataset_root)
    context = str(root / group.source_pair_index_relpath)
    _require(root.is_dir(), f"dataset root is not a directory: {root}")
    _require(root.name == group.dataset_basename, f"{context}: wrong dataset basename {root.name!r}")
    _require(
        inventory.dataset_key == group.dataset_key and inventory.root == root.resolve(strict=True),
        f"{context}: validated inventory does not bind this corpus root",
    )
    top_keys, pair_keys = _source_schema(group)
    _exact_keys(document, top_keys, context)
    _validate_source_header(document, group, context)

    rows = document["pairs"]
    expected_pairs = _expected_source_pairs(group)
    expected_row_count = len(OBJECT_ROLES) * len(expected_pairs)
    _require(
        len(rows) == expected_row_count,
        f"{context}: expected {expected_row_count} rows, got {len(rows)}",
    )
    checked_paths: set[str] = set()
    observed: dict[str, set[tuple[int, int]]] = {name: set() for name in OBJECT_ROLES}
    validated: list[dict[str, Any]] = []
    for row_index, raw_pair in enumerate(rows):
        row_context = f"{context}: pair[{row_index}]"
        _require(isinstance(raw_pair, dict), f"{row_context}: pair must be an object")
        _exact_keys(raw_pair, pair_keys, row_context)
        model, src, dst, src_record, dst_record = _validate_common_pair(
            raw_pair, group, root, inventory, row_index, checked_paths
        )
        _require(
            (src, dst) in expected_pairs,
            f"{row_context}: ({src}, {dst}) is not a frozen primary edge",
        )
        _require(
            (src, dst) not in observed[model],
            f"{row_context}: duplicate directed pair for {model}",
        )
        _validate_source_pair_semantics(
            raw_pair,
            group,
            src,
            dst,
            src_record,
            dst_record,
            row_context,
        )
        observed[model].add((src, dst))
        validated.append(dict(raw_pair))

    for model, directed_pairs in observed.items():
        _require(
            directed_pairs == expected_pairs,
            f"{context}: {model} does not contain the exact primary edge set",
        )
    return validated


def _frozen_signed_generator(group: PrimaryGroup) -> tuple[float, str]:
    sign = 1.0 if group.direction == "forward" else -1.0
    if group.family in {"roll", "yaw", "pitch"}:
        return (6.0 if group.family == "roll" else sign * 6.0), "degrees"
    if group.family == "scale":
        value = 0.02310490601866489 if group.direction == "forward" else -0.023104906018664894
        return value, "log_scale"
    if group.family == "translation":
        return sign * 0.048, "world_units"
    raise SplitGenerationError(f"unknown family: {group.family}")


def _signed_generator(
    group: PrimaryGroup, source_document: Mapping[str, Any]
) -> tuple[float, str]:
    expected, units = _frozen_signed_generator(group)
    observed_key = (
        "delta_theta_deg"
        if group.family in {"roll", "yaw", "pitch"}
        else "delta_world"
        if group.family == "translation"
        else None
    )
    if observed_key is not None:
        _close(source_document[observed_key], expected, f"{group.artifact_stem}.{observed_key}")
    return expected, units


def _state(
    group: PrimaryGroup,
    source_pair: Mapping[str, Any],
    endpoint: str,
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    model = str(source_pair["model_name"])
    frame_index = int(source_pair[f"{endpoint}_frame_index"])
    state = inventory.frame(model, frame_index)["physical_state"]
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


def _expected_pair_count(group: PrimaryGroup, split: str) -> int:
    if group.family == "roll":
        return 147 if split == "train" else 21
    if group.family in {"yaw", "pitch"}:
        return 82 if split == "train" else 9
    if group.family == "scale":
        return 90 if split == "train" else 11
    if group.family == "translation":
        return 56 if split == "train" else 16
    raise SplitGenerationError(f"unknown family: {group.family}")


GENERATOR_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(_configuration_payload())
).hexdigest()


def _included_objects(split: str) -> list[str]:
    if split == "train":
        return list(OBJECT_ORDER)
    if split == "validation":
        return [name for name in OBJECT_ORDER if OBJECT_ROLES[name] == "development"]
    if split == "test":
        return [name for name in OBJECT_ORDER if OBJECT_ROLES[name] != "development"]
    raise SplitGenerationError(f"unknown output split: {split}")


def artifact_filename(group: PrimaryGroup, split: str) -> str:
    """Return the deterministic relative filename for one split artifact."""

    _require(split in {"train", "validation", "test"}, f"unknown output split: {split}")
    return f"{group.artifact_stem}__{split}.json"


def _enrich_pair(
    source_pair: Mapping[str, Any],
    group: PrimaryGroup,
    split: str,
    signed_generator: float,
    generator_units: str,
    inventory: ValidatedCorpusInventory,
) -> dict[str, Any]:
    model = source_pair["model_name"]
    src = source_pair["src_frame_index"]
    dst = source_pair["dst_frame_index"]
    pair = {
        "pair_id": (
            f"{group.family}:{group.physical_axis}:{group.direction}:"
            f"{model}:{split}:{src:04d}:{dst:04d}"
        ),
        "model_name": model,
        "object_role": OBJECT_ROLES[model],
        "split": split,
        "transform_family": group.family,
        "physical_axis": group.physical_axis,
        "direction": group.direction,
        "signed_generator": signed_generator,
        "generator_units": generator_units,
        "stride": group.stride,
        "stride_units": group.stride_units,
        "cyclic": group.cyclic,
        "src_frame_index": src,
        "dst_frame_index": dst,
        "src_state": _state(group, source_pair, "src", inventory),
        "dst_state": _state(group, source_pair, "dst", inventory),
        "src_image_relpath": source_pair["src_image_relpath"],
        "dst_image_relpath": source_pair["dst_image_relpath"],
        "src_mask_relpath": source_pair["src_mask_relpath"],
        "dst_mask_relpath": source_pair["dst_mask_relpath"],
    }
    _exact_keys(pair, PAIR_KEYS, pair["pair_id"])
    return pair


def _validate_input_mappings(
    dataset_roots: Mapping[str, str | Path],
    corpus_inventories: Mapping[str, bytes],
    generator_commit: str,
) -> None:
    expected = {group.dataset_key for group in PRIMARY_GROUPS}
    for label, mapping in (
        ("dataset_roots", dataset_roots),
        ("corpus_inventories", corpus_inventories),
    ):
        _require(
            set(mapping) == expected,
            f"{label}: expected exact keys {sorted(expected)}, got {sorted(mapping)}",
        )
    _require(
        isinstance(generator_commit, str) and _COMMIT_RE.fullmatch(generator_commit) is not None,
        "generator_commit must be a lowercase 40-hex commit",
    )
    for key in sorted(expected):
        _require(
            isinstance(corpus_inventories[key], bytes),
            f"corpus_inventories[{key!r}] must contain canonical JSON bytes",
        )


def generate_primary_split_artifacts(
    dataset_roots: Mapping[str, str | Path],
    *,
    corpus_inventories: Mapping[str, bytes],
    generator_commit: str,
) -> dict[str, bytes]:
    """Generate all 33 primary pair-index artifacts as canonical JSON bytes.

    The function performs no writes.  It validates each source pair index and
    every referenced frame/mask path, filters only same-block endpoints, and
    verifies the frozen per-object counts before returning.
    """

    _validate_input_mappings(
        dataset_roots,
        corpus_inventories,
        generator_commit,
    )
    roots = {key: Path(value) for key, value in dataset_roots.items()}
    validated_inventories = {
        key: validate_corpus_inventory(corpus_inventories[key], key, roots[key])
        for key in sorted(roots)
    }
    artifacts: dict[str, bytes] = {}

    for group in PRIMARY_GROUPS:
        root = roots[group.dataset_key]
        inventory = validated_inventories[group.dataset_key]
        _require(root.is_dir(), f"dataset root is not a directory: {root}")
        _require(
            root.name == group.dataset_basename,
            f"{group.dataset_key}: expected basename {group.dataset_basename!r}, got {root.name!r}",
        )
        semantic_lock_path = root / "semantic_lock.json"
        _require(
            semantic_lock_path.is_file(),
            f"{group.dataset_key}: missing semantic_lock.json",
        )
        source_path = root / group.source_pair_index_relpath
        _require(source_path.is_file(), f"missing source pair index: {source_path}")
        source_document = _load_json(source_path)
        source_pairs = validate_source_pair_document(
            source_document,
            group,
            root,
            inventory=inventory,
        )
        source_sha = _sha256_file(source_path)
        semantic_lock_sha = _sha256_file(semantic_lock_path)
        _require(
            source_sha
            == inventory.files_by_relpath[group.source_pair_index_relpath]["sha256"],
            f"{group.artifact_stem}: source pair index differs from validated inventory",
        )
        _require(
            semantic_lock_sha == inventory.semantic_lock_sha256,
            f"{group.artifact_stem}: semantic lock differs from validated inventory",
        )
        signed_generator, generator_units = _signed_generator(group, source_document)
        partition = _partition(group)
        train_set = set(partition["train_frame_indices"])
        holdout_set = set(partition["holdout_frame_indices"])

        for split in ("train", "validation", "test"):
            included_objects = _included_objects(split)
            endpoint_set = train_set if split == "train" else holdout_set
            selected_source_pairs = [
                row
                for row in source_pairs
                if row["model_name"] in included_objects
                and row["src_frame_index"] in endpoint_set
                and row["dst_frame_index"] in endpoint_set
            ]
            object_position = {name: index for index, name in enumerate(included_objects)}
            selected_source_pairs.sort(
                key=lambda row: (
                    object_position[row["model_name"]],
                    row["src_frame_index"],
                    row["dst_frame_index"],
                )
            )
            pairs = [
                _enrich_pair(
                    row,
                    group,
                    split,
                    signed_generator,
                    generator_units,
                    inventory,
                )
                for row in selected_source_pairs
            ]
            expected_per_object = _expected_pair_count(group, split)
            counts = {
                model: sum(pair["model_name"] == model for pair in pairs)
                for model in included_objects
            }
            _require(
                all(count == expected_per_object for count in counts.values()),
                f"{group.artifact_stem}/{split}: wrong per-object counts {counts}",
            )
            _require(
                len({pair["pair_id"] for pair in pairs}) == len(pairs),
                f"{group.artifact_stem}/{split}: duplicate pair_id",
            )

            artifact: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "dataset_basename": group.dataset_basename,
                "dataset_binding_sha256": inventory.content_hash_sha256,
                "dataset_semantic_lock_sha256": semantic_lock_sha,
                "source_pair_index_relpath": group.source_pair_index_relpath,
                "source_pair_index_sha256": source_sha,
                "unversioned_source": inventory.document["unversioned_source"],
                "generator": {
                    "name": GENERATOR_NAME,
                    "commit": generator_commit,
                    "config_sha256": GENERATOR_CONFIG_SHA256,
                },
                "split": split,
                "object_roles": OBJECT_ROLES,
                "included_objects": included_objects,
                "transform": {
                    "family": group.family,
                    "physical_axis": group.physical_axis,
                    "direction": group.direction,
                    "signed_generator": signed_generator,
                    "generator_units": generator_units,
                    "stride": group.stride,
                    "stride_units": group.stride_units,
                    "cyclic": group.cyclic,
                    "expected_2d_family": group.expected_2d_family,
                },
                "frame_partition": partition,
                "pair_count": len(pairs),
                "pair_counts_by_object": counts,
                "pairs": pairs,
            }
            _exact_keys(
                {**artifact, "content_hash_sha256": ""},
                TOP_LEVEL_KEYS,
                f"{group.artifact_stem}/{split}",
            )
            artifact["content_hash_sha256"] = content_hash_sha256(artifact)
            filename = artifact_filename(group, split)
            _require(filename not in artifacts, f"duplicate artifact filename: {filename}")
            artifacts[filename] = canonical_json_bytes(artifact, trailing_newline=True)

    _require(len(artifacts) == 33, f"expected 33 primary artifacts, got {len(artifacts)}")
    return dict(sorted(artifacts.items()))


__all__ = [
    "ARTIFACT_TYPE",
    "GENERATOR_CONFIG_SHA256",
    "GENERATOR_NAME",
    "OBJECT_ROLES",
    "PAIR_KEYS",
    "PRIMARY_GROUPS",
    "PrimaryGroup",
    "SCHEMA_VERSION",
    "SplitGenerationError",
    "TOP_LEVEL_KEYS",
    "TRANSFORM_KEYS",
    "artifact_filename",
    "canonical_json_bytes",
    "content_hash_sha256",
    "generate_primary_split_artifacts",
    "validate_source_pair_document",
]
