"""Run the complete 24-role adjacent frozen-feature matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST_SCHEMA = "adjacent_feature_reanchor_manifest.v1"
EXPECTED_ROLES = 24
IMPLEMENTATION_SOURCES = (
    "keypoint_net/run_adjacent_feature_reanchor_matrix.py",
    "keypoint_net/run_adjacent_feature_reanchor_raw.py",
    "keypoint_net/evaluate_adjacent_feature_reanchor.py",
)


class AdjacentMatrixError(ValueError):
    """Raised when the complete adjacent matrix cannot be executed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjacentMatrixError(message)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--mplconfigdir", type=Path, default=Path("/private/tmp/mpl-adjacent-feature"))
    args = parser.parse_args()
    _require(not args.output_root.exists(), "output root already exists; use a fresh attempt")
    manifest_path = args.manifest.resolve(strict=True)
    manifest_record = _file_record(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "manifest schema differs")
    roles = manifest.get("roles")
    _require(isinstance(roles, list) and len(roles) == EXPECTED_ROLES, "role inventory differs")
    repo_root = args.repo_root.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    args.mplconfigdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    environment["MPLCONFIGDIR"] = str(args.mplconfigdir.resolve())
    completed: list[dict[str, Any]] = []
    for ordinal, role in enumerate(roles, start=1):
        role_key = str(role["role_key"])
        role_root = args.output_root / role_key
        raw_dir = role_root / "raw"
        evaluation_dir = role_root / "evaluation_v1"
        print(f"[{ordinal:02d}/{EXPECTED_ROLES}] raw {role_key}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "keypoint_net/run_adjacent_feature_reanchor_raw.py"),
                "--manifest",
                str(manifest_path),
                "--role-key",
                role_key,
                "--output-dir",
                str(raw_dir),
                "--batch-size",
                str(args.batch_size),
                "--torch-threads",
                str(args.torch_threads),
            ],
            cwd=repo_root,
            env=environment,
            check=True,
        )
        receipt_path = raw_dir / "raw_adjacent_feature_receipt.json"
        print(f"[{ordinal:02d}/{EXPECTED_ROLES}] evaluate {role_key}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "keypoint_net/evaluate_adjacent_feature_reanchor.py"),
                "--raw-receipt",
                str(receipt_path),
                "--output-dir",
                str(evaluation_dir),
            ],
            cwd=repo_root,
            env=environment,
            check=True,
        )
        evaluation_path = evaluation_dir / "adjacent_feature_evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _require(evaluation.get("role", {}).get("role_key") == role_key, f"role output differs: {role_key}")
        _require(evaluation.get("training_or_weight_update_performed") is False, f"training flag differs: {role_key}")
        completed.append(
            {
                "ordinal": ordinal,
                "role_key": role_key,
                "raw_receipt": _file_record(receipt_path),
                "evaluation": _file_record(evaluation_path),
                "strict_pass_count": int(evaluation["report"]["strict_pass_count"]),
            }
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    receipt = {
        "schema_version": "adjacent_feature_reanchor_matrix_run.v1",
        "artifact_type": "complete_adjacent_feature_reanchor_matrix_run",
        "manifest": manifest_record,
        "implementation_head": head,
        "implementation_sources": {relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES},
        "role_count": len(completed),
        "roles": completed,
        "execution": "sequential; stop on first failed role; fresh per-role raw and evaluation directories",
        "training_or_weight_update_performed": False,
    }
    output = args.output_root / "MATRIX_RUN_RECEIPT.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "role_count": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
