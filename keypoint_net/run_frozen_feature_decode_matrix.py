"""Run the frozen feature decoder and post-hash evaluator over all 24 roles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class DecodeMatrixError(ValueError):
    """Raised when a role is incomplete or an output path is ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeMatrixError(message)


def _complete_evaluation(role_dir: Path, role_key: str) -> Path | None:
    candidates = sorted(role_dir.glob("evaluation_v*/feature_decode_evaluation.json"))
    complete = []
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("role", {}).get("role_key") == role_key:
            complete.append(path)
    _require(len(complete) <= 1, f"{role_key}: multiple complete evaluations")
    return complete[0] if complete else None


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roles = manifest.get("roles")
    _require(isinstance(roles, list) and len(roles) == 24, "manifest must contain 24 roles")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.repo_root)
    environment["MPLCONFIGDIR"] = str(args.mplconfigdir)
    for ordinal, role in enumerate(roles, start=1):
        role_key = str(role["role_key"])
        role_dir = args.matrix_root / role_key
        receipt = role_dir / "raw_feature_decode_receipt.json"
        evaluation = _complete_evaluation(role_dir, role_key) if role_dir.exists() else None
        if evaluation is not None:
            _require(receipt.is_file(), f"{role_key}: evaluation exists without raw receipt")
            print(json.dumps({"event": "already_complete", "ordinal": ordinal, "role_key": role_key}), flush=True)
            continue
        _require(not role_dir.exists(), f"{role_key}: incomplete role directory exists; use a fresh attempt")
        print(json.dumps({"event": "raw_start", "ordinal": ordinal, "role_key": role_key}), flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "keypoint_net.run_frozen_feature_decode_raw",
                "--manifest",
                str(args.manifest),
                "--role-key",
                role_key,
                "--output-dir",
                str(role_dir),
                "--batch-size",
                str(args.batch_size),
                "--torch-threads",
                str(args.torch_threads),
            ],
            cwd=args.repo_root,
            env=environment,
            check=True,
        )
        print(json.dumps({"event": "evaluation_start", "ordinal": ordinal, "role_key": role_key}), flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "keypoint_net.evaluate_frozen_feature_decode",
                "--raw-receipt",
                str(receipt),
                "--output-dir",
                str(role_dir / "evaluation_v1"),
                "--repo-root",
                str(args.repo_root),
            ],
            cwd=args.repo_root,
            env=environment,
            check=True,
        )
        print(json.dumps({"event": "role_complete", "ordinal": ordinal, "role_key": role_key}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mplconfigdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve(strict=True)
    args.matrix_root = args.matrix_root.resolve(strict=True)
    args.repo_root = args.repo_root.resolve(strict=True)
    args.mplconfigdir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
