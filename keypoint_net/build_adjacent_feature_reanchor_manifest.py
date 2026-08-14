"""Build a geometry-blind manifest for adjacent learned-feature re-anchoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "adjacent_feature_reanchor_manifest.v1"
EXPECTED_SOURCE_SCHEMA = "frozen_feature_decode_manifest.v1"
EXPECTED_SOURCE_SHA256 = "16e3af9e7afbe736370ca0cd0b0531fed0a622194741f478d7164e9ca61f60ec"
EXPECTED_ROLES = 24
EXPECTED_FRAMES = 180
IMPLEMENTATION_SOURCES = (
    "keypoint_net/adjacent_feature_reanchor.py",
    "keypoint_net/build_adjacent_feature_reanchor_manifest.py",
    "keypoint_net/run_adjacent_feature_reanchor_raw.py",
    "keypoint_net/run_frozen_feature_decode_raw.py",
    "keypoint_net/frozen_feature_decode.py",
    "keypoint_net/model.py",
    "keypoint_net/run_frozen_wobble_forensics.py",
    "keypoint_net/descriptor_attachment.py",
)


class AdjacentFeatureManifestError(ValueError):
    """Raised when a source-bound adjacent manifest cannot be constructed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjacentFeatureManifestError(message)


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


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCES], cwd=repo_root, check=False
    )
    # New files are intentionally uncommitted during the production-path smoke;
    # every byte is still bound below. Tracked dependencies must remain clean.
    tracked = tuple(source for source in IMPLEMENTATION_SOURCES if (repo_root / source).is_file())
    _require(tracked, "implementation source inventory is empty")
    return {
        "implementation_head": head,
        "tracked_sources_clean_at_head": clean.returncode == 0,
        "implementation_sources": {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES},
    }


def build_manifest(source_manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    source_record = _file_record(source_manifest_path)
    _require(source_record["sha256"] == EXPECTED_SOURCE_SHA256, "source manifest SHA-256 differs")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(source, Mapping), "source manifest is not an object")
    _require(source.get("schema_version") == EXPECTED_SOURCE_SCHEMA, "source manifest schema differs")
    _require(source.get("artifact_type") == "geometry_blind_frozen_feature_decode_manifest", "source manifest type differs")
    roles = source.get("roles")
    corpus = source.get("corpus")
    _require(isinstance(roles, list) and len(roles) == EXPECTED_ROLES, "source role inventory differs")
    _require(isinstance(corpus, Mapping) and corpus.get("frame_count") == EXPECTED_FRAMES, "source corpus differs")
    frames = corpus.get("frames")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "source RGB inventory differs")
    _require([int(row.get("frame_index", -1)) for row in frames] == list(range(EXPECTED_FRAMES)), "frame order differs")
    role_keys = [str(role.get("role_key")) for role in roles]
    _require(len(set(role_keys)) == EXPECTED_ROLES, "role keys are duplicated")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "geometry_blind_adjacent_feature_reanchor_manifest",
        "source_full_orbit_manifest": source_record,
        "implementation": _implementation_binding(repo_root),
        "prediction_information_lock": {
            "allowed": ["checkpoint", "RGB frames", "frame indices", "model configuration embedded in checkpoint"],
            "forbidden": [
                "mask",
                "theta",
                "pivot",
                "physical track",
                "operator prediction",
                "prior forensic arrays",
                "OCR match",
                "previous decoded coordinate",
            ],
        },
        "corpus": dict(corpus),
        "feature_field": {"channels": 128, "height": 64, "width": 64},
        "pairing": {
            "source_frame_indices": list(range(EXPECTED_FRAMES)),
            "target_rule": "(source_frame_index + 1) mod 180",
            "cyclic_edge_count": EXPECTED_FRAMES,
            "physical_step_deg_not_opened_by_raw_stage": 2.0,
        },
        "anchor_basis": "hard detector cell re-anchored independently at every source frame",
        "search": "full-image pointwise cosine over endpoint-aligned 64x64 encoder field",
        "separated_second_peak_radius_cells": 4.0,
        "roles": [dict(role) for role in roles],
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "output already exists; use a fresh attempt")
    manifest = build_manifest(args.source_manifest.resolve(strict=True), args.repo_root.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "role_count": len(manifest["roles"])}, sort_keys=True))


if __name__ == "__main__":
    main()
