"""Run geometry-blind adjacent RGB correspondence for one frozen role."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

try:
    from .rgb_material_observability import (
        PATCH_SIZES,
        RGBObservabilityConfig,
        decode_rgb_edge,
        flatten_decode,
    )
except ImportError:
    from rgb_material_observability import (  # type: ignore
        PATCH_SIZES,
        RGBObservabilityConfig,
        decode_rgb_edge,
        flatten_decode,
    )


SCHEMA_VERSION = "rgb_material_observability_raw_receipt.v1"
EXPECTED_MANIFEST_SCHEMA = "rgb_material_observability_manifest.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
SCOPES = ("global", "local")


class RGBRawError(ValueError):
    """Raised when the geometry-blind raw gate differs from its lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBRawError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {"absolute_path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "manifest schema differs")
    _require(value.get("training_or_weight_update_performed") is False, "manifest trained")
    implementation = value.get("implementation", {}).get("implementation_sources")
    _require(isinstance(implementation, Mapping), "implementation binding missing")
    for source_record in implementation.values():
        _require(_file_record(str(source_record["absolute_path"])) == dict(source_record), "implementation source differs")
    return value, record


def _role(manifest: Mapping[str, Any], role_key: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("roles", []) if row.get("role_key") == role_key]
    _require(len(matches) == 1, f"expected one role {role_key}, got {len(matches)}")
    return dict(matches[0])


def _load_coordinates(role: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    record = role.get("source_detector_coordinates")
    _require(isinstance(record, Mapping), "source-coordinate binding missing")
    observed = _file_record(str(record["absolute_path"]))
    _require(observed == dict(record), "source-coordinate artifact changed")
    with np.load(observed["absolute_path"], allow_pickle=False) as archive:
        normalized = np.asarray(archive["source_soft_coordinate_normalized"], dtype=np.float64)
        pixel = np.asarray(archive["source_soft_coordinate_px"], dtype=np.float64)
        frame = np.asarray(archive["frame_index"], dtype=np.int64)
    _require(normalized.shape == (EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), "normalized coordinate shape differs")
    _require(pixel.shape == normalized.shape and np.isfinite(pixel).all(), "pixel coordinate shape differs")
    _require(np.isfinite(normalized).all(), "normalized coordinate contains non-finite values")
    _require(np.array_equal(frame, np.arange(EXPECTED_FRAMES)), "frame index differs")
    return normalized, pixel, observed


def _load_images(manifest: Mapping[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    corpus = manifest.get("corpus")
    _require(isinstance(corpus, Mapping), "corpus binding missing")
    root = Path(str(corpus["rgb_object_root"])).resolve(strict=True)
    frames = corpus.get("frames")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "RGB frame count differs")
    images: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frame, row in enumerate(frames):
        _require(int(row.get("frame_index", -1)) == frame, "RGB frame index differs")
        path = (root / str(row["image_relpath"])).resolve(strict=True)
        record = _file_record(path)
        _require(record["sha256"] == row.get("image_sha256"), f"RGB hash differs at frame {frame}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        _require(rgb.shape == (512, 512, 3), f"RGB shape differs at frame {frame}")
        images.append(np.ascontiguousarray(rgb))
        records.append(record)
    return images, records


def _allocate() -> dict[str, np.ndarray]:
    shape = (len(PATCH_SIZES), EXPECTED_FRAMES, EXPECTED_CHANNELS)
    arrays: dict[str, np.ndarray] = {
        "source_valid": np.zeros(shape, dtype=bool),
        "source_patch_rms": np.zeros(shape, dtype=np.float64),
    }
    for scope in SCOPES:
        arrays[f"{scope}_valid"] = np.zeros(shape, dtype=bool)
        arrays[f"{scope}_top1_coordinate_px"] = np.full(shape + (2,), -1.0, dtype=np.float64)
        arrays[f"{scope}_top1_score"] = np.full(shape, -2.0, dtype=np.float64)
        arrays[f"{scope}_top2_coordinate_px"] = np.full(shape + (2,), -1.0, dtype=np.float64)
        arrays[f"{scope}_top2_score"] = np.full(shape, -2.0, dtype=np.float64)
        arrays[f"{scope}_margin"] = np.zeros(shape, dtype=np.float64)
        arrays[f"{scope}_candidate_count"] = np.zeros(shape, dtype=np.int64)
    return arrays


def _decode_order(
    images: Sequence[np.ndarray],
    source_px: np.ndarray,
    order: Sequence[int],
    *,
    config: RGBObservabilityConfig,
) -> dict[str, np.ndarray]:
    result = _allocate()
    for frame in order:
        target_frame = (int(frame) + 1) % EXPECTED_FRAMES
        for scale_index, patch_size in enumerate(PATCH_SIZES):
            for channel in range(EXPECTED_CHANNELS):
                flat = flatten_decode(
                    decode_rgb_edge(
                        images[int(frame)],
                        images[target_frame],
                        source_px[int(frame), channel],
                        patch_size,
                        config=config,
                    )
                )
                for name, value in flat.items():
                    result[name][scale_index, int(frame), channel] = value
    return result


def _arrays_exact(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> bool:
    return set(left) == set(right) and all(np.array_equal(left[name], right[name]) for name in left)


def run_raw(manifest_path: Path, role_key: str, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    manifest, manifest_record = _load_manifest(manifest_path)
    role = _role(manifest, role_key)
    source_normalized, source_px, coordinate_record = _load_coordinates(role)
    images, image_records = _load_images(manifest)
    config = RGBObservabilityConfig()
    _require(config.as_dict() == manifest.get("config"), "runtime config differs from manifest")
    cv2.setNumThreads(1)

    output_dir.mkdir(parents=True, exist_ok=False)
    forward = _decode_order(images, source_px, range(EXPECTED_FRAMES), config=config)
    reverse = _decode_order(images, source_px, range(EXPECTED_FRAMES - 1, -1, -1), config=config)
    order_exact = _arrays_exact(forward, reverse)
    _require(order_exact, "reverse edge order changed an indexed prediction")

    arrays_path = output_dir / "raw_rgb_observability_arrays.npz"
    np.savez_compressed(
        arrays_path,
        patch_size=np.asarray(PATCH_SIZES, dtype=np.int64),
        frame_index=np.arange(EXPECTED_FRAMES, dtype=np.int64),
        target_frame_index=np.roll(np.arange(EXPECTED_FRAMES, dtype=np.int64), -1),
        channel_index=np.arange(EXPECTED_CHANNELS, dtype=np.int64),
        source_coordinate_normalized=source_normalized,
        source_coordinate_px=source_px,
        **forward,
    )
    arrays_record = _file_record(arrays_path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "geometry_blind_adjacent_rgb_material_observability_raw",
        "role": role,
        "manifest": manifest_record,
        "implementation_head": head,
        "config": config.as_dict(),
        "raw_arrays": arrays_record,
        "source_coordinate_artifact": coordinate_record,
        "rgb_frame_records": image_records,
        "frame_order_reversal_exact": order_exact,
        "masks_or_geometry_opened": False,
        "forbidden_inputs_opened": [],
        "target_detector_coordinates_opened": False,
        "learned_features_or_operator_opened": False,
        "temporal_propagation_used": False,
        "optimizer_gradient_or_weight_update_used": False,
        "source_valid_count_by_scale": {
            str(size): int(np.sum(forward["source_valid"][index]))
            for index, size in enumerate(PATCH_SIZES)
        },
        "expected_rows_per_scale": EXPECTED_FRAMES * EXPECTED_CHANNELS,
        "training_or_weight_update_performed": False,
    }
    receipt_path = output_dir / "raw_rgb_observability_receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"receipt": _file_record(receipt_path), "raw_arrays": arrays_record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory exists; use a fresh attempt")
    print(
        json.dumps(
            run_raw(
                args.manifest.resolve(strict=True),
                args.role_key,
                args.output_dir,
                args.repo_root.resolve(strict=True),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
