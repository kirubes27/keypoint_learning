"""Build the frame-zero-only initialization for the recursive RGB teacher gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

try:
    from .material_transport_gate_io import file_record, load_json, require, write_json
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_gate_io import file_record, load_json, require, write_json


EXPECTED_AMENDMENT_SHA256 = "913398162f3795c062793f1423d548edcd7109b62d7d95417d1097ec02d41854"
EXPECTED_CAPABILITY_MANIFEST_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def build(args: argparse.Namespace) -> dict[str, object]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    capability_record = file_record(args.capability_manifest)
    require(capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256, "capability SHA-256 differs")
    capability = load_json(args.capability_manifest)
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == capability["portable_tracks"]["sha256"], "track SHA-256 differs")
    with np.load(args.tracks) as archive:
        frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        tracks = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "track frame order differs")
    require(tracks.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "track shape differs")

    args.output_dir.mkdir(parents=True)
    initials_path = args.output_dir / "RECURSIVE_INITIALS.npz"
    np.savez_compressed(
        initials_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(0, dtype=np.int64),
        initial_coordinate_px=tracks[0],
    )
    receipt = {
        "schema_version": "recursive_continuous_teacher_initials.v1",
        "artifact_type": "privileged_frame_zero_only_initialization",
        "initial_frame_index": 0,
        "witness_count": EXPECTED_WITNESSES,
        "initials": file_record(initials_path),
        "sources": {
            "amendment": amendment_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
        },
        "implementation_head": _git(args.repo_root, "rev-parse", "HEAD"),
        "implementation_source": file_record(Path(__file__)),
        "non_frame_zero_coordinates_written": False,
        "training_or_weight_update_performed": False,
    }
    receipt_path = args.output_dir / "RECURSIVE_INITIALS_RECEIPT.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
