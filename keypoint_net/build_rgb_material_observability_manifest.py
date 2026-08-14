"""Build the source-bound manifest for adjacent RGB material observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

try:
    from .rgb_material_observability import RGBObservabilityConfig, normalized_to_pixel
except ImportError:
    from rgb_material_observability import RGBObservabilityConfig, normalized_to_pixel  # type: ignore


SCHEMA_VERSION = "rgb_material_observability_manifest.v1"
EXPECTED_SOURCE_SCHEMA = "frozen_feature_decode_manifest.v1"
EXPECTED_RAW_SCHEMA = "frozen_feature_decode_raw_receipt.v1"
EXPECTED_ROLES = 24
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
IMPLEMENTATION_SOURCES = (
    "keypoint_net/rgb_material_observability.py",
    "keypoint_net/build_rgb_material_observability_manifest.py",
    "keypoint_net/run_rgb_material_observability_raw.py",
    "keypoint_net/evaluate_rgb_material_observability.py",
    "keypoint_net/run_rgb_material_observability_matrix.py",
    "keypoint_net/summarize_rgb_material_observability.py",
    "tests/test_rgb_material_observability.py",
)


class RGBManifestError(ValueError):
    """Raised when an upstream binding cannot support the frozen gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_bound_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    observed = _file_record(str(record["absolute_path"]))
    _require(observed == dict(record), f"{name} binding differs")
    value = json.loads(Path(observed["absolute_path"]).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    missing = [relative for relative in IMPLEMENTATION_SOURCES if not (repo_root / relative).is_file()]
    _require(not missing, f"implementation sources are missing: {missing}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *IMPLEMENTATION_SOURCES],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(status == "", "RGB observability implementation is not clean at HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "implementation_head": head,
        "implementation_sources": {
            relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def build_manifest(
    source_manifest_path: Path,
    source_matrix_root: Path,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_manifest_record = _file_record(source_manifest_path)
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _require(source.get("schema_version") == EXPECTED_SOURCE_SCHEMA, "source manifest schema differs")
    _require(source.get("training_or_weight_update_performed") is False, "source manifest trained")
    source_roles = source.get("roles")
    _require(isinstance(source_roles, list) and len(source_roles) == EXPECTED_ROLES, "source role count differs")
    corpus = source.get("corpus")
    _require(isinstance(corpus, Mapping), "source corpus binding is missing")
    frames = corpus.get("frames")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "source frame count differs")
    rgb_root = Path(str(corpus.get("rgb_object_root"))).resolve(strict=True)
    for frame, row in enumerate(frames):
        _require(int(row.get("frame_index", -1)) == frame, "source frame index differs")
        path = (rgb_root / str(row["image_relpath"])).resolve(strict=True)
        _require(_sha256(path) == row.get("image_sha256"), f"RGB hash differs at frame {frame}")

    config = RGBObservabilityConfig()
    config.validate()
    output_root.mkdir(parents=True, exist_ok=False)
    coordinate_root = output_root / "source_coordinates"
    coordinate_root.mkdir()
    roles: list[dict[str, Any]] = []
    for source_role in source_roles:
        role_key = str(source_role["role_key"])
        role_dir = (source_matrix_root / role_key).resolve(strict=True)
        raw_receipt_path = role_dir / "raw_feature_decode_receipt.json"
        raw_receipt_record = _file_record(raw_receipt_path)
        raw_receipt = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
        _require(raw_receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, f"{role_key}: source raw schema differs")
        _require(raw_receipt.get("role", {}).get("role_key") == role_key, f"{role_key}: source raw role differs")
        _require(raw_receipt.get("frame_order_reversal_exact") is True, f"{role_key}: source order proof failed")
        _require(raw_receipt.get("masks_or_geometry_opened") is False, f"{role_key}: source raw opened geometry")
        raw_arrays_record = raw_receipt.get("raw_arrays")
        _require(isinstance(raw_arrays_record, Mapping), f"{role_key}: source raw arrays missing")
        observed_arrays = _file_record(str(raw_arrays_record["absolute_path"]))
        _require(observed_arrays == dict(raw_arrays_record), f"{role_key}: source raw arrays changed")
        with np.load(observed_arrays["absolute_path"], allow_pickle=False) as archive:
            detector = np.asarray(archive["detector_coordinate"], dtype=np.float64)
            frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        _require(detector.shape == (2, EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), f"{role_key}: detector shape differs")
        _require(np.isfinite(detector).all(), f"{role_key}: detector contains non-finite values")
        _require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), f"{role_key}: frame index differs")
        soft = detector[1]
        coordinate_path = coordinate_root / f"{role_key}.npz"
        np.savez_compressed(
            coordinate_path,
            source_soft_coordinate_normalized=soft,
            source_soft_coordinate_px=normalized_to_pixel(soft),
            frame_index=frame_index,
            channel_index=np.arange(EXPECTED_CHANNELS, dtype=np.int64),
        )
        report_record = raw_receipt.get("role", {}).get("source_forensic_report")
        _require(isinstance(report_record, Mapping), f"{role_key}: source forensic report missing")
        source_report = _load_bound_json(report_record, name=f"{role_key} source forensic report")
        _require(source_report.get("cell_id") == source_role.get("cell_id"), f"{role_key}: cell ID differs")
        roles.append(
            {
                "role_key": role_key,
                "task": str(source_role["task"]),
                "arm": str(source_role["arm"]),
                "seed": int(source_role["seed"]),
                "checkpoint_role": str(source_role["checkpoint_role"]),
                "checkpoint_epoch": int(source_role["checkpoint_epoch"]),
                "cell_id": str(source_role["cell_id"]),
                "source_detector_coordinates": _file_record(coordinate_path),
                "source_feature_decode_raw_receipt": raw_receipt_record,
                "source_feature_decode_raw_arrays": observed_arrays,
                "source_forensic_report": dict(report_record),
            }
        )
    roles.sort(key=lambda row: row["role_key"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "source_bound_adjacent_rgb_material_observability_manifest",
        "source_feature_decode_manifest": source_manifest_record,
        "source_feature_decode_matrix_root": str(source_matrix_root.resolve(strict=True)),
        "implementation": _implementation_binding(repo_root),
        "config": config.as_dict(),
        "pairing": {
            "source_frames": list(range(EXPECTED_FRAMES)),
            "target_frames": [(frame + 1) % EXPECTED_FRAMES for frame in range(EXPECTED_FRAMES)],
            "cyclic": True,
            "runtime_geometry_used": False,
        },
        "raw_information_lock": {
            "allowed": ["bound RGB", "frame index", "adjacent pairing", "source soft detector coordinate", "fixed matcher configuration"],
            "forbidden": ["mask", "theta", "pivot", "material target", "target detector coordinate", "encoder feature", "heatmap", "operator", "OCR", "tracker state"],
        },
        "scale_derivation": {
            "local_patch_px": 35,
            "local_patch_reason": "exact nominal receptive field of current four-convolution encoder",
            "context_patch_px": 105,
            "context_patch_reason": "one predeclared three-times-receptive-field sensitivity; not a sweep",
            "local_search_radius_px": 32,
            "search_reason": "four legacy r64 cells and above the 25.2-pixel pivot-free in-frame +2-degree displacement bound",
        },
        "corpus": {
            "rgb_object_root": str(rgb_root),
            "frame_count": EXPECTED_FRAMES,
            "frames": frames,
        },
        "roles": roles,
        "training_or_weight_update_performed": False,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": _file_record(manifest_path), "role_count": len(roles)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-matrix-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_root.exists(), "output root exists; use a fresh attempt")
    print(
        json.dumps(
            build_manifest(
                args.source_manifest.resolve(strict=True),
                args.source_matrix_root.resolve(strict=True),
                args.repo_root.resolve(strict=True),
                args.output_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
