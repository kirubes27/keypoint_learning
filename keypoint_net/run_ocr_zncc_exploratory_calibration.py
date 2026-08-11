"""Exploratory total-gradient calibration after a failed coverage gate.

This command performs no optimizer step and does not authorize training.  It
exists to measure a reproducible coefficient for a later, separately reviewed
matched experiment without pretending that the frozen 50% Stage-0 gate passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from keypoint_net.dataset import IndexPairDataset
from keypoint_net.ocr_zncc_transport import OCRZNCCConfig
from keypoint_net.run_ocr_zncc_stage0 import (
    DATASET_BINDING_SHA256,
    OBJECT_NAME,
    RECIPES,
    SEEDS,
    TRAIN_INDEX_RELPATH,
    TRAIN_INDEX_SHA256,
    _calibration_cell,
    _calibration_receipt,
    _file_record,
    _load_json,
    _require,
    _sha256_file,
    _write_exclusive,
    canonical_sha256,
    validate_audit_checkout,
)


def run(
    *,
    repo_root: Path,
    data_root: Path,
    direction_summary: Path,
    output_path: Path,
    batch_size: int,
    peak_margin_exclusion_radius_cells: int,
) -> dict[str, Any]:
    _require(not output_path.exists(), f"refusing to overwrite {output_path}")
    checkout = validate_audit_checkout(repo_root)
    summary, summary_record = _load_json(direction_summary)
    _require(summary.get("both_recipe_gates_pass") is False, "direction gate status differs")
    _require(summary.get("calibration_authorized_by_stage0_rule") is False,
             "Stage-0 calibration status differs")

    config = OCRZNCCConfig(
        peak_margin_exclusion_radius_cells=peak_margin_exclusion_radius_cells,
    )
    for record in summary.get("direction_cell_artifacts", []):
        cell, _ = _load_json(Path(record["absolute_path"]))
        _require(cell.get("ocr_zncc_config") == config.as_dict(),
                 "direction-cell OCR-ZNCC config differs")

    train_index = repo_root / TRAIN_INDEX_RELPATH
    _require(_sha256_file(train_index) == TRAIN_INDEX_SHA256, "train index differs")
    dataset = IndexPairDataset(
        data_root=str(data_root),
        index_path=str(train_index),
        img_size=512,
        center_crop=None,
        include_backward=False,
        object_name=OBJECT_NAME,
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=TRAIN_INDEX_SHA256,
        expected_dataset_binding_sha256=DATASET_BINDING_SHA256,
    )
    _require(len(dataset) == 147, "training pair count differs")
    cells = [
        _calibration_cell(
            recipe=recipe,
            seed=seed,
            dataset=dataset,
            batch_size=batch_size,
            config=config,
        )
        for recipe in RECIPES
        for seed in SEEDS
    ]
    calibration = _calibration_receipt(cells)
    document = {
        "schema_version": "ocr_zncc_exploratory_total_gradient_calibration.v1",
        "artifact_type": "exploratory_total_gradient_calibration_after_failed_coverage_gate",
        "status": "coefficient_measured_not_training_authorization",
        "checkout": checkout,
        "source_files": {
            "calibration_runner": _file_record(Path(__file__)),
            "stage0_runner": _file_record(repo_root / "keypoint_net/run_ocr_zncc_stage0.py"),
            "ocr_zncc_transport": _file_record(repo_root / "keypoint_net/ocr_zncc_transport.py"),
            "model": _file_record(repo_root / "keypoint_net/model.py"),
            "dataset": _file_record(repo_root / "keypoint_net/dataset.py"),
            "train_index": _file_record(train_index),
        },
        "direction_summary": summary_record,
        "failed_gate_binding": {
            "both_recipe_gates_pass": False,
            "calibration_authorized_by_stage0_rule": False,
            "coverage_threshold": 0.50,
            "interpretation": "engineering gate remains failed; it was not changed",
        },
        "ocr_zncc_config": config.as_dict(),
        "dataset": {
            "absolute_path": str(data_root.resolve(strict=True)),
            "split": "train",
            "pair_count": len(dataset),
            "dataset_binding_sha256": DATASET_BINDING_SHA256,
            "train_index_sha256": TRAIN_INDEX_SHA256,
        },
        "calibration": calibration,
        "authorization_boundary": {
            "optimizer_steps": 0,
            "training_performed": False,
            "gpu_job_submitted": False,
            "training_authorized_by_this_artifact": False,
        },
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    _write_exclusive(output_path, document)
    return document


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--direction-summary", type=Path, required=True)
    result.add_argument("--output-path", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument(
        "--peak-margin-exclusion-radius-cells",
        type=int,
        choices=(0, 1),
        required=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    torch.set_num_threads(1)
    document = run(
        repo_root=args.repo_root,
        data_root=args.data_root,
        direction_summary=args.direction_summary,
        output_path=args.output_path,
        batch_size=args.batch_size,
        peak_margin_exclusion_radius_cells=args.peak_margin_exclusion_radius_cells,
    )
    print(json.dumps({
        "status": document["status"],
        "coefficient": document["calibration"]["coefficient"],
        "scaled_median_contribution": document["calibration"][
            "scaled_median_contribution_all_six_cells"
        ],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
