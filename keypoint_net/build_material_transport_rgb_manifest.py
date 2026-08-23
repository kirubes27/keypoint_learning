"""Build the RGB-only manifest consumed by the free-logit raw stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

try:
    from .material_transport_free_logits import MaterialTransportConfig
    from .material_transport_gate_io import (
        EXPECTED_FRAMES,
        SANITIZED_SCHEMA,
        file_record,
        load_json,
        require,
        validate_sanitized_manifest,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_free_logits import MaterialTransportConfig
    from material_transport_gate_io import (
        EXPECTED_FRAMES,
        SANITIZED_SCHEMA,
        file_record,
        load_json,
        require,
        validate_sanitized_manifest,
        write_json,
    )


EXPECTED_UPSTREAM_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_LOCK_SHA256 = "472c4b451fd59a3c8b88d92df501caf73e3fd230f5a379f3138f7f1ab358060e"


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(args.repo_root.is_dir(), "repository root missing")
    require(_git(args.repo_root, "status", "--porcelain") == "", "implementation worktree must be clean")
    upstream_record = file_record(args.upstream_manifest, include_path=False)
    lock_record = file_record(args.semantic_lock)
    require(upstream_record["sha256"] == EXPECTED_UPSTREAM_SHA256, "upstream manifest SHA-256 differs")
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic lock SHA-256 differs")
    upstream = load_json(args.upstream_manifest)
    dataset = upstream.get("dataset")
    require(isinstance(dataset, dict), "upstream dataset missing")
    source_frames = dataset.get("frames")
    require(isinstance(source_frames, list) and len(source_frames) == EXPECTED_FRAMES, "upstream frame rows differ")
    object_root = args.data_object_root.resolve()
    require(object_root.is_dir(), "data object root missing")

    frames: list[dict[str, Any]] = []
    for expected, row in enumerate(source_frames):
        require(int(row["frame_index"]) == expected, f"upstream frame {expected} order differs")
        image_relpath = Path(str(row["image_relpath"]))
        require(not image_relpath.is_absolute() and ".." not in image_relpath.parts, f"frame {expected} path is unsafe")
        record = file_record(object_root / image_relpath, include_path=False)
        require(record["sha256"] == row["image_sha256"], f"frame {expected} RGB SHA-256 differs")
        frames.append(
            {
                "frame_index": expected,
                "image_relpath": image_relpath.as_posix(),
                "image_sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )

    implementation_names = (
        "keypoint_net/material_transport_free_logits.py",
        "keypoint_net/material_transport_gate_io.py",
        "keypoint_net/build_material_transport_rgb_manifest.py",
        "keypoint_net/run_material_transport_rgb_field.py",
    )
    implementation = {
        name: file_record(args.repo_root / name, include_path=False)
        for name in implementation_names
    }
    manifest = {
        "schema_version": SANITIZED_SCHEMA,
        "artifact_type": "sanitized_cyclic_rgb_field_manifest",
        "implementation_head": _git(args.repo_root, "rev-parse", "HEAD"),
        "implementation_sources": implementation,
        "semantic_lock": lock_record,
        "upstream_manifest_digest": upstream_record,
        "field_config": MaterialTransportConfig().as_dict(),
        "dataset": {
            "object_root_at_build_time": str(object_root),
            "frame_count": EXPECTED_FRAMES,
            "frames": frames,
            "portable_rebinding_rule": "an alternate root is allowed only after all 180 RGB hashes and sizes match",
            "edge_rule": "cyclic_f_to_f_plus_1_mod_180",
        },
        "statistical_scope": "one rendered hammer orbit; all frame summaries are descriptive",
        "information_boundary": "raw stage receives this RGB-only manifest and no privileged evaluation files",
    }
    validate_sanitized_manifest(manifest)
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "SANITIZED_RGB_FIELD_MANIFEST.json"
    write_json(manifest_path, manifest)
    receipt = {
        "schema_version": "sanitized_cyclic_rgb_field_manifest_receipt.v1",
        "manifest": file_record(manifest_path),
        "implementation_head": manifest["implementation_head"],
        "frame_count": EXPECTED_FRAMES,
        "all_rgb_hashes_rechecked": True,
        "privileged_fields_emitted": False,
    }
    write_json(args.output_dir / "MANIFEST_RECEIPT.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-manifest", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--data-object-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    receipt = build(parse_args())
    print(receipt)


if __name__ == "__main__":
    main()
