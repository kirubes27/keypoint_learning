"""Build a geometry-blind manifest for frozen full-orbit feature decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "frozen_feature_decode_manifest.v1"
EXPECTED_MATRIX_SHA256 = "f74d15e46f9d165a542867c07c1257ea28f41b12cdd8af1313a570527aa4d2b0"
EXPECTED_ROLES = 24
EXPECTED_FRAMES = 180
IMPLEMENTATION_SOURCES = (
    "keypoint_net/build_frozen_feature_decode_manifest.py",
    "keypoint_net/frozen_feature_decode.py",
    "keypoint_net/run_frozen_feature_decode_raw.py",
    "keypoint_net/model.py",
    "keypoint_net/run_frozen_wobble_forensics.py",
)


class DecodeManifestError(ValueError):
    """Raised when the source-bound matrix cannot yield a safe raw manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeManifestError(message)


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


def _load_bound_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    observed = _file_record(str(record["absolute_path"]))
    _require(observed == dict(record), f"{name} file binding differs")
    value = json.loads(Path(observed["absolute_path"]).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCES], cwd=repo_root, check=False
    )
    _require(clean.returncode == 0, "decode implementation differs from recorded HEAD")
    return {
        "implementation_head": head,
        "implementation_sources": {
            relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES
        },
    }


def build_manifest(matrix_summary_path: Path, repo_root: Path) -> dict[str, Any]:
    matrix_record = _file_record(matrix_summary_path)
    _require(matrix_record["sha256"] == EXPECTED_MATRIX_SHA256, "matrix summary SHA-256 differs")
    matrix = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
    _require(matrix.get("schema_version") == "frozen_wobble_complete_matrix_summary.v1", "matrix schema differs")
    _require(matrix.get("inventory", {}).get("matrix_complete") is True, "source matrix is incomplete")
    runs = matrix.get("per_run")
    _require(isinstance(runs, list) and len(runs) == EXPECTED_ROLES, "source matrix role count differs")

    roles: list[dict[str, Any]] = []
    common_images: list[dict[str, Any]] | None = None
    common_data_root: str | None = None
    seen: set[str] = set()
    for run in runs:
        _require(isinstance(run, Mapping), "matrix run is invalid")
        role_key = run.get("role_key")
        _require(isinstance(role_key, str) and role_key not in seen, "role key is invalid or duplicated")
        seen.add(role_key)
        report_record = run.get("bindings", {}).get("forensic_report")
        _require(isinstance(report_record, Mapping), f"{role_key}: report binding missing")
        report = _load_bound_json(report_record, name=f"{role_key} report")
        _require(report.get("training_or_weight_update_performed") is False, f"{role_key}: source report trained")
        _require(report.get("checkpoint_role") == run.get("checkpoint_role"), f"{role_key}: role differs")
        _require(report.get("checkpoint_epoch") == run.get("checkpoint_epoch"), f"{role_key}: epoch differs")
        checkpoint = report.get("bindings", {}).get("checkpoint")
        corpus = report.get("bindings", {}).get("corpus")
        _require(isinstance(checkpoint, Mapping) and isinstance(corpus, Mapping), f"{role_key}: input binding missing")
        observed_checkpoint = _file_record(str(checkpoint["absolute_path"]))
        _require(
            observed_checkpoint["sha256"] == checkpoint.get("sha256")
            and observed_checkpoint["size_bytes"] == checkpoint.get("size_bytes"),
            f"{role_key}: checkpoint binding differs",
        )
        frames = corpus.get("frame_records")
        _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, f"{role_key}: frame count differs")
        images = []
        for index, frame in enumerate(frames):
            _require(int(frame.get("frame_index", -1)) == index, f"{role_key}: frame index differs")
            images.append(
                {
                    "frame_index": index,
                    "image_relpath": str(frame["image_relpath"]),
                    "image_sha256": str(frame["image_sha256"]),
                }
            )
        data_root = str(Path(str(corpus["object_root"])).resolve(strict=True))
        if common_images is None:
            for image in images:
                image_path = (Path(data_root) / image["image_relpath"]).resolve(strict=True)
                _require(
                    _sha256(image_path) == image["image_sha256"],
                    f"{role_key}: RGB object-root binding differs at frame {image['frame_index']}",
                )
            common_images = images
            common_data_root = data_root
        else:
            _require(images == common_images and data_root == common_data_root, "roles do not share one RGB corpus")
        roles.append(
            {
                "role_key": role_key,
                "task": str(run["task"]),
                "arm": str(run["arm"]),
                "seed": int(run["seed"]),
                "cell_id": str(report["cell_id"]),
                "checkpoint_role": str(run["checkpoint_role"]),
                "checkpoint_epoch": int(run["checkpoint_epoch"]),
                "checkpoint": observed_checkpoint,
                "source_forensic_report": dict(report_record),
            }
        )
    _require(len(seen) == EXPECTED_ROLES, "role inventory differs")
    roles.sort(key=lambda row: row["role_key"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "geometry_blind_frozen_feature_decode_manifest",
        "source_matrix_summary": matrix_record,
        "implementation": _implementation_binding(repo_root),
        "prediction_information_lock": {
            "allowed": ["checkpoint", "RGB frames", "frame indices", "model configuration embedded in checkpoint"],
            "forbidden": ["mask", "theta", "pivot", "physical track", "operator prediction", "prior forensic arrays", "OCR match"],
        },
        "corpus": {
            "rgb_object_root": common_data_root,
            "frame_count": EXPECTED_FRAMES,
            "frames": common_images,
        },
        "anchor_frame_index": 27,
        "anchor_bases": ["hard_peak", "soft_coordinate"],
        "feature_field": {"channels": 128, "height": 64, "width": 64},
        "separated_second_peak_radius_cells": 4.0,
        "roles": roles,
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.matrix_summary.resolve(strict=True), args.repo_root.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "role_count": len(manifest["roles"])}, sort_keys=True))


if __name__ == "__main__":
    main()
