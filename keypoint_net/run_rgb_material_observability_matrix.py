"""Run the complete 24-role adjacent RGB observability matrix in parallel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .evaluate_rgb_material_observability import evaluate
    from .run_rgb_material_observability_raw import run_raw
except ImportError:
    from evaluate_rgb_material_observability import evaluate  # type: ignore
    from run_rgb_material_observability_raw import run_raw  # type: ignore


EXPECTED_MANIFEST_SCHEMA = "rgb_material_observability_manifest.v1"
EXPECTED_ROLES = 24


class RGBMatrixError(ValueError):
    """Raised when the complete matrix cannot run without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGBMatrixError(message)


def _run_one(manifest_path: str, role_key: str, role_root: str, repo_root: str) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve(strict=True)
    root = Path(role_root)
    repo = Path(repo_root).resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    raw = run_raw(manifest, role_key, root / "raw", repo)
    evaluation = evaluate(
        Path(raw["receipt"]["absolute_path"]).resolve(strict=True),
        root / "evaluation",
        repo,
    )
    return {"role_key": role_key, "raw": raw, "evaluation": evaluation}


def run_matrix(manifest_path: Path, repo_root: Path, output_root: Path, max_workers: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "manifest schema differs")
    roles = manifest.get("roles")
    _require(isinstance(roles, list) and len(roles) == EXPECTED_ROLES, "manifest role count differs")
    _require(max_workers >= 1, "max workers must be positive")
    output_root.mkdir(parents=True, exist_ok=False)
    futures = {}
    completed = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for role in roles:
            role_key = str(role["role_key"])
            future = executor.submit(
                _run_one,
                str(manifest_path),
                role_key,
                str(output_root / role_key),
                str(repo_root),
            )
            futures[future] = role_key
        for future in as_completed(futures):
            role_key = futures[future]
            try:
                completed.append(future.result())
                print(json.dumps({"completed": role_key, "count": len(completed), "expected": EXPECTED_ROLES}), flush=True)
            except Exception as exc:
                print(json.dumps({"failed": role_key, "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
                raise
    completed.sort(key=lambda row: row["role_key"])
    receipt = {
        "schema_version": "rgb_material_observability_matrix_run.v1",
        "artifact_type": "complete_adjacent_rgb_observability_matrix_run",
        "manifest_path": str(manifest_path),
        "role_count": len(completed),
        "roles": completed,
        "max_workers": max_workers,
        "training_or_weight_update_performed": False,
    }
    receipt_path = output_root / "matrix_run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"matrix_run_receipt": str(receipt_path.resolve()), "role_count": len(completed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    _require(not args.output_root.exists(), "output root exists; use a fresh attempt")
    print(
        json.dumps(
            run_matrix(
                args.manifest.resolve(strict=True),
                args.repo_root.resolve(strict=True),
                args.output_root,
                args.max_workers,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
