"""Audit OCR/base extractor gradients at the exact training minibatch size.

This CPU-only command performs zero optimizer and scheduler steps.  For each
Task-55/Task-80 initialization it compares the gradient of the complete recipe
base loss with the OCR-ZNCC auxiliary gradient over the extractor parameters,
batch by batch.  It never decomposes the base loss into components.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from keypoint_net.dataset import IndexPairDataset
from keypoint_net.model import PhaseAModel, compute_losses
from keypoint_net.ocr_zncc_transport import (
    OCRZNCCConfig,
    normalized_images_to_retained_rgb,
    ocr_zncc_transport_loss,
    patch_centres_inside,
)
from keypoint_net.run_ocr_zncc_stage0 import (
    DATASET_BINDING_SHA256,
    OBJECT_NAME,
    RECIPES,
    SEEDS,
    TRAIN_INDEX_RELPATH,
    TRAIN_INDEX_SHA256,
    _expected_recipe_training,
    _file_record,
    _parameter_items,
    _require,
    _seed_everything,
    _sha256_file,
    _state_sha256,
    _write_exclusive,
    canonical_sha256,
    validate_audit_checkout,
)


SCHEMA_VERSION = "ocr_zncc_training_minibatch_total_gradient_audit.v2"
COEFFICIENT_CAP = 0.5
TARGET_MEDIAN_CONTRIBUTION = 0.10
NORM_FLOOR = 1e-15


def _gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    vector = torch.cat(
        [item.detach().cpu().to(dtype=torch.float64).reshape(-1) for item in gradients]
    )
    _require(bool(torch.isfinite(vector).all()), "minibatch gradient is non-finite")
    return vector


def _norm(vector: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(vector).item())
    _require(math.isfinite(value), "minibatch gradient norm is non-finite")
    return value


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    _require(bool(values), "cannot summarize an empty minibatch statistic")
    _require(all(math.isfinite(value) for value in values), "minibatch statistic is non-finite")
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _cell(
    *,
    recipe: str,
    seed: int,
    dataset: IndexPairDataset,
    batch_size: int,
    config: OCRZNCCConfig,
) -> dict[str, Any]:
    _seed_everything(seed)
    training = _expected_recipe_training(recipe)
    model = PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(training["learn_inverse_operator"]),
        heatmap_res=64,
    )
    model.train()
    parameter_items = _parameter_items(model)
    parameters = [parameter for _, parameter in parameter_items]
    before_state = _state_sha256(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    sample_offset = 0
    for batch_index, batch in enumerate(loader):
        x_t = batch["x_t"].cpu()
        x_t1 = batch["x_t1"].cpu()
        action_labels = batch["action_label"].cpu().long()
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=float(training["lambda_smooth"]),
            lambda_disp=float(training["lambda_disp"]),
            lambda_ent=float(training["lambda_ent"]),
            lambda_act=float(training["lambda_act"]),
            lambda_loc=float(training["lambda_loc"]),
            lambda_inv=float(training["lambda_inv"]),
            lambda_cycle=float(training["lambda_cycle"]),
            lambda_attach=0.0,
            sigma=0.1,
            num_keypoints=10,
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=30.0,
        )
        source = outputs["p_t"].reshape(x_t.shape[0], 10, 2)
        transport = ocr_zncc_transport_loss(
            normalized_images_to_retained_rgb(x_t),
            normalized_images_to_retained_rgb(x_t1),
            source,
            outputs["p_hat_t1"].reshape(x_t.shape[0], 10, 2),
            outputs["p_t1"].reshape(x_t.shape[0], 10, 2),
            config=config,
        )
        base = _gradient_vector(losses["base_loss"], parameters, retain_graph=True)
        auxiliary = _gradient_vector(transport.loss, parameters, retain_graph=False)
        base_norm = _norm(base)
        auxiliary_norm = _norm(auxiliary)
        _require(base_norm > NORM_FLOOR, "base gradient norm is zero")
        accepted = transport.accepted_count
        if accepted == 0:
            _require(auxiliary_norm <= NORM_FLOOR, "zero-accept batch has nonzero auxiliary gradient")
            ratio = None
            cosine = None
        else:
            _require(auxiliary_norm > NORM_FLOOR, "accepted batch has zero auxiliary gradient")
            ratio = auxiliary_norm / base_norm
            cosine = float(torch.dot(base, auxiliary).item() / (base_norm * auxiliary_norm))
            _require(math.isfinite(ratio) and math.isfinite(cosine), "gradient comparison is invalid")
        source_inside = patch_centres_inside(
            source.detach(), patch_size=config.patch_size, grid_size=config.grid_size
        )
        accepted_source_outside = int((transport.accepted & ~source_inside).sum().item())
        _require(accepted_source_outside == 0, "accepted source patch crosses image boundary")
        rows.append(
            {
                "batch_index": batch_index,
                "sample_start": sample_offset,
                "sample_count": int(x_t.shape[0]),
                "accepted_match_count": accepted,
                "possible_match_count": int(x_t.shape[0]) * 10,
                "accepted_source_patch_outside_image_count": accepted_source_outside,
                "base_gradient_norm": base_norm,
                "auxiliary_gradient_norm": auxiliary_norm,
                "unscaled_auxiliary_to_base_norm_ratio": ratio,
                "cap_scaled_auxiliary_to_base_norm_ratio": (
                    COEFFICIENT_CAP * ratio if ratio is not None else None
                ),
                "base_auxiliary_gradient_cosine": cosine,
            }
        )
        sample_offset += int(x_t.shape[0])
        _require(all(parameter.grad is None for parameter in model.parameters()),
                 "autograd populated a parameter .grad field")
    _require(sample_offset == len(dataset), "minibatch audit did not consume full split")
    nonzero = [row for row in rows if row["accepted_match_count"] > 0]
    unscaled_ratios = [
        float(row["unscaled_auxiliary_to_base_norm_ratio"]) for row in nonzero
    ]
    cap_scaled_ratios = [
        float(row["cap_scaled_auxiliary_to_base_norm_ratio"]) for row in nonzero
    ]
    cosines = [float(row["base_auxiliary_gradient_cosine"]) for row in nonzero]
    return {
        "cell_id": f"{recipe}__r64__seed{seed}",
        "recipe": recipe,
        "seed": seed,
        "model_initialization_seed": seed,
        "parameter_names": [name for name, _ in parameter_items],
        "parameter_set": "all_trainable_extractor.encoder_and_extractor.heatmap_head",
        "rows": rows,
        "summary": {
            "batch_count": len(rows),
            "nonzero_accept_batch_count": len(nonzero),
            "zero_accept_batch_count": len(rows) - len(nonzero),
            "zero_accept_batch_fraction": (len(rows) - len(nonzero)) / len(rows),
            "accepted_match_count": sum(row["accepted_match_count"] for row in rows),
            "possible_match_count": len(dataset) * 10,
            "accepted_match_coverage": sum(row["accepted_match_count"] for row in rows)
            / (len(dataset) * 10),
            "unscaled_auxiliary_to_base_norm_ratio_nonzero_batches": _describe(
                unscaled_ratios
            ),
            "cap_scaled_auxiliary_to_base_norm_ratio_nonzero_batches": _describe(
                cap_scaled_ratios
            ),
            "base_auxiliary_gradient_cosine_nonzero_batches": _describe(cosines),
            "accepted_source_patch_outside_image_count": 0,
        },
        "execution": {
            "device": "cpu",
            "model_mode": "train",
            "batch_size": batch_size,
            "shuffle": False,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "model_state_sha256_before": before_state,
            "model_state_sha256_after": _state_sha256(model),
            "batchnorm_state_change_expected": True,
        },
    }


def run(
    *, repo_root: Path, data_root: Path, output_path: Path, batch_size: int
) -> dict[str, Any]:
    _require(not output_path.exists(), f"refusing to overwrite {output_path}")
    checkout = validate_audit_checkout(repo_root)
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
    config = OCRZNCCConfig(peak_margin_exclusion_radius_cells=1)
    cells = [
        _cell(recipe=recipe, seed=seed, dataset=dataset, batch_size=batch_size, config=config)
        for recipe in RECIPES
        for seed in SEEDS
    ]
    unscaled_cell_medians = [
        float(cell["summary"]["unscaled_auxiliary_to_base_norm_ratio_nonzero_batches"]["median"])
        for cell in cells
    ]
    median_unscaled = float(statistics.median(unscaled_cell_medians))
    coefficient = min(
        COEFFICIENT_CAP,
        TARGET_MEDIAN_CONTRIBUTION / median_unscaled,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "zero_step_training_minibatch_total_gradient_audit",
        "checkout": checkout,
        "source_files": {
            "runner": _file_record(Path(__file__)),
            "ocr_zncc_transport": _file_record(repo_root / "keypoint_net/ocr_zncc_transport.py"),
            "stage0_runner": _file_record(repo_root / "keypoint_net/run_ocr_zncc_stage0.py"),
            "model": _file_record(repo_root / "keypoint_net/model.py"),
            "loss_implementation": _file_record(repo_root / "keypoint_net/model.py"),
            "train_index": _file_record(train_index),
        },
        "dataset": {
            "absolute_path": str(data_root.resolve(strict=True)),
            "split": "train",
            "pair_count": len(dataset),
            "dataset_binding_sha256": DATASET_BINDING_SHA256,
            "train_index_sha256": TRAIN_INDEX_SHA256,
        },
        "ocr_zncc_config": config.as_dict(),
        "coefficient_rule": {
            "formula": "min(0.5, 0.10 / median_of_six_cell_median_unscaled_batch_ratios)",
            "target_median_contribution": TARGET_MEDIAN_CONTRIBUTION,
            "coefficient_safety_cap": COEFFICIENT_CAP,
            "raw_lambda_sweep_performed": False,
        },
        "recommended_coefficient": coefficient,
        "cells": cells,
        "combined_summary": {
            "median_of_six_cell_median_unscaled_batch_ratios": median_unscaled,
            "minimum_cell_median_unscaled_batch_ratio": min(unscaled_cell_medians),
            "maximum_cell_median_unscaled_batch_ratio": max(unscaled_cell_medians),
            "recommended_scaled_median_contribution": coefficient * median_unscaled,
            "all_finite": True,
            "all_base_gradients_nonzero": True,
            "all_accepted_source_patches_inside_image": True,
        },
        "interpretation": {
            "measures": (
                "per-training-minibatch L2 norm and cosine of the OCR auxiliary gradient "
                "versus the entire recipe base-loss gradient over extractor parameters"
            ),
            "does_not_measure": [
                "individual_base_loss_component_gradients",
                "optimizer_update_after_momentum_or_adam_state",
                "training_outcome",
            ],
            "batch_partition_note": (
                "The complete split is partitioned into deterministic contiguous batches; "
                "training shuffles batches, so per-step ratios remain descriptive rather "
                "than an exact forecast of every optimizer step."
            ),
        },
        "authorization_boundary": {
            "optimizer_steps": 0,
            "training_performed": False,
            "gpu_jobs_submitted": 0,
            "training_authorized_by_this_artifact_alone": False,
        },
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    _write_exclusive(output_path, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    _require(args.batch_size == 16, "audit requires the frozen training batch size 16")
    torch.set_num_threads(1)
    document = run(
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(json.dumps(document["combined_summary"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
