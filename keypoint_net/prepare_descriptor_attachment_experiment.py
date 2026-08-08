"""Create the exact 12-cell control/attachment manifest after CPU calibration.

This command performs no training and refuses to overwrite an existing file.
It converts the retained 64-head base recipes into paired zero/attachment arms
for seeds 42--44 while binding every scientific train.py argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .descriptor_attachment import LOSS_SPEC_SHA256, canonical_json_bytes, canonical_sha256
    from .descriptor_attachment_authorization import (
        BOUND_TRAINING_ARGUMENTS,
        EXPECTED_BRANCH,
        EXPERIMENT_ARTIFACT_TYPE,
        EXPERIMENT_SCHEMA_VERSION,
        validate_weight_receipt,
    )
    from .descriptor_attachment_calibration import (
        CALIBRATION_RECIPES,
        CALIBRATION_SEEDS,
        CHECKPOINT_SELECTOR,
    )
    from . import representation_fresh_checkpoint_authorization as fresh
except ImportError:  # Historical direct-script execution.
    from descriptor_attachment import (  # type: ignore
        LOSS_SPEC_SHA256,
        canonical_json_bytes,
        canonical_sha256,
    )
    from descriptor_attachment_authorization import (  # type: ignore
        BOUND_TRAINING_ARGUMENTS,
        EXPECTED_BRANCH,
        EXPERIMENT_ARTIFACT_TYPE,
        EXPERIMENT_SCHEMA_VERSION,
        validate_weight_receipt,
    )
    from descriptor_attachment_calibration import (  # type: ignore
        CALIBRATION_RECIPES,
        CALIBRATION_SEEDS,
        CHECKPOINT_SELECTOR,
    )
    import representation_fresh_checkpoint_authorization as fresh  # type: ignore


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state(repo_root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise ValueError(f"wrong source branch {branch!r}")
    if tracked:
        raise ValueError("tracked source is modified")
    return commit, branch


def build_experiment_manifest(
    *,
    repo_root: Path,
    data_root: Path,
    output_root: Path,
    weight_receipt_path: Path,
    weight_receipt_file_sha256: str,
) -> dict[str, Any]:
    source_commit, source_branch = _source_state(repo_root)
    receipt, receipt_file_hash, receipt_content_hash = validate_weight_receipt(
        repo_root,
        weight_receipt_path,
        expected_file_sha256=weight_receipt_file_sha256,
    )
    if receipt["source"]["commit"] != source_commit:
        raise ValueError("weight receipt and experiment manifest source commits differ")
    cells = []
    for recipe in CALIBRATION_RECIPES:
        for seed in CALIBRATION_SEEDS:
            base_cell_id = f"{recipe}__r64__seed{seed}"
            base_arguments = fresh.training_arguments(
                repo_root,
                base_cell_id,
                data_root=data_root,
                output_root=output_root,
            )
            base_arguments.pop("fresh_cell_id")
            for arm, lambda_attach in (
                ("control", 0.0),
                ("attachment", float(receipt["lambda_attach"])),
            ):
                arguments = {
                    **base_arguments,
                    "lambda_attach": lambda_attach,
                }
                if set(arguments) != BOUND_TRAINING_ARGUMENTS:
                    raise RuntimeError("generated cell does not bind every training argument")
                cells.append({
                    "cell_id": f"{recipe}__{arm}__seed{seed}",
                    "arm": arm,
                    "training_arguments": arguments,
                })
    if len(cells) != 12 or len({cell["cell_id"] for cell in cells}) != 12:
        raise RuntimeError("descriptor experiment must contain exactly 12 unique cells")
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": EXPERIMENT_ARTIFACT_TYPE,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "loss_spec_sha256": LOSS_SPEC_SHA256,
        "checkpoint_selector": CHECKPOINT_SELECTOR,
        "weight_receipt": {
            "absolute_path": str(weight_receipt_path.resolve(strict=True)),
            "file_sha256": receipt_file_hash,
            "content_hash_sha256": receipt_content_hash,
        },
        "cells": cells,
    }
    manifest["content_hash_sha256"] = canonical_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the exact descriptor control/attachment experiment manifest"
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--weight_receipt", type=Path, required=True)
    parser.add_argument("--weight_receipt_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_path = args.output.expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment manifest: {output_path}")
    manifest = build_experiment_manifest(
        repo_root=repo_root,
        data_root=args.data_root.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().absolute(),
        weight_receipt_path=args.weight_receipt.expanduser().resolve(strict=True),
        weight_receipt_file_sha256=args.weight_receipt_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(json.dumps({
        "output": str(output_path),
        "file_sha256": _sha256_file(output_path),
        "content_hash_sha256": manifest["content_hash_sha256"],
        "cell_count": len(manifest["cells"]),
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_experiment_manifest"]
