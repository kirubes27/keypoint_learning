"""Audit the exact OCR-ZNCC coordinate-loss descent direction.

Stage 0 checked the displacement from the current target keypoint to the hard
ZNCC match.  Training follows the derivative of a component-wise SmoothL1
loss, whose clipping can rotate that displacement.  This CPU-only audit checks
the actual negative gradient with respect to the target coordinates at both
the six frozen Task-55/Task-80 controls and their exact random initializations.

No parameter ``.grad`` field, optimizer, scheduler, checkpoint, split, or
selection state is changed.  The material rotation is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from keypoint_net.dataset import IndexPairDataset
from keypoint_net.model import PhaseAModel
from keypoint_net.ocr_zncc_transport import (
    OCRZNCCConfig,
    grid_step,
    normalized_images_to_retained_rgb,
    ocr_zncc_transport_loss,
)
from keypoint_net.run_ocr_zncc_stage0 import (
    ANGLE_DEGREES,
    CELL_IDS,
    DATASET_BINDING_SHA256,
    OBJECT_NAME,
    ON_OBJECT_RATE_MINIMUM,
    RECIPES,
    SEEDS,
    TRAIN_INDEX_RELPATH,
    TRAIN_INDEX_SHA256,
    _descriptive,
    _expected_recipe_training,
    _file_record,
    _load_target_masks,
    _require,
    _rotation,
    _sample_masks,
    _sha256_file,
    _state_sha256,
    _validate_control_cell,
    _write_exclusive,
    canonical_sha256,
    validate_audit_checkout,
)


SCHEMA_VERSION = "ocr_zncc_exact_coordinate_gradient_direction.v2"
NORM_EPSILON = 1e-12


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _initial_model(recipe: str, seed: int) -> PhaseAModel:
    _seed_everything(seed)
    training = _expected_recipe_training(recipe)
    return PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(training["learn_inverse_operator"]),
        heatmap_res=64,
    )


def _summary(rows: Sequence[Mapping[str, Any]], channels: Sequence[int]) -> dict[str, Any]:
    _require(bool(rows), "exact-gradient cell has no rows")
    accepted = [row for row in rows if bool(row["accepted_match"])]
    usable = [row for row in rows if bool(row["usable_direction"])]
    cosine = (
        _descriptive(
            (float(row["direction_cosine"]) for row in usable),
            sample_unit="pair_channel_exact_smoothl1_descent_cosine",
        )
        if usable
        else None
    )
    delta = (
        _descriptive(
            (float(row["one_cell_material_error_delta_objdiag"]) for row in usable),
            sample_unit="pair_channel_exact_smoothl1_one_cell_error_delta_objdiag",
        )
        if usable
        else None
    )
    per_channel = []
    for channel in channels:
        local = [row for row in rows if int(row["channel"]) == int(channel)]
        _require(bool(local), f"channel {channel} has no exact-gradient rows")
        before_rate = float(np.mean([bool(row["on_object_before"]) for row in local]))
        after_rate = float(np.mean([bool(row["on_object_after"]) for row in local]))
        per_channel.append({
            "channel": int(channel),
            "sample_count": len(local),
            "on_object_rate_before": before_rate,
            "on_object_rate_after": after_rate,
            "on_object_before": before_rate >= ON_OBJECT_RATE_MINIMUM,
            "on_object_after": after_rate >= ON_OBJECT_RATE_MINIMUM,
        })
    before_count = sum(bool(row["on_object_before"]) for row in per_channel)
    after_count = sum(bool(row["on_object_after"]) for row in per_channel)
    positive_fraction = (
        float(np.mean([float(row["direction_cosine"]) > 0.0 for row in usable]))
        if usable else None
    )
    infinitesimal_gate = bool(
        cosine is not None
        and cosine["median"] > 0.0
        and cosine["mean"] > 0.0
        and positive_fraction is not None
        and positive_fraction > 0.5
        and after_count >= before_count
    )
    one_cell_gate = bool(
        infinitesimal_gate
        and delta is not None
        and delta["median"] < 0.0
    )
    return {
        "row_count": len(rows),
        "accepted_match_count": len(accepted),
        "accepted_match_coverage": len(accepted) / len(rows),
        "accepted_source_patch_outside_image_count": sum(
            bool(row["accepted_match"]) and not bool(row["source_patch_inside"])
            for row in rows
        ),
        "usable_direction_count": len(usable),
        "usable_direction_coverage": len(usable) / len(rows),
        "direction_cosine": cosine,
        "positive_infinitesimal_material_direction_fraction": positive_fraction,
        "one_cell_material_error_delta_objdiag": delta,
        "active_on_object": {
            "rate_threshold": ON_OBJECT_RATE_MINIMUM,
            "before_count": before_count,
            "after_count": after_count,
            "change": after_count - before_count,
            "per_channel": per_channel,
        },
        "infinitesimal_direction_gate": infinitesimal_gate,
        "infinitesimal_direction_gate_rule": {
            "median_exact_gradient_cosine_strictly_positive": True,
            "mean_exact_gradient_cosine_strictly_positive": True,
            "positive_cosine_fraction_strictly_above_half": True,
            "active_on_object_count_nonworse": True,
            "coverage_threshold_applied": False,
            "coverage_note": (
                "The frozen 50% engineering gate remains failed and is not "
                "retrospectively changed; this audit isolates gradient direction."
            ),
        },
        "one_cell_finite_step_gate": one_cell_gate,
        "one_cell_finite_step_gate_rule": {
            "infinitesimal_direction_gate_passes": True,
            "median_exact_gradient_one_cell_error_delta_strictly_negative": True,
            "interpretation": (
                "This is a finite-step overshoot diagnostic, not the sign of the "
                "infinitesimal training gradient."
            ),
        },
    }


def _cell(
    *,
    cell_id: str,
    stage: str,
    model: PhaseAModel,
    channels: Sequence[int],
    dataset: IndexPairDataset,
    target_masks: Sequence[np.ndarray],
    target_diagonals: Sequence[float],
    config: OCRZNCCConfig,
    batch_size: int,
) -> dict[str, Any]:
    _require(stage in {"frozen_control", "random_initialization"}, "invalid audit stage")
    if stage == "frozen_control":
        model.eval()
    else:
        model.train()
    before_state = _state_sha256(model)
    rows: list[dict[str, Any]] = []
    offset = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        x_t = batch["x_t"].cpu()
        x_t1 = batch["x_t1"].cpu()
        outputs = model(x_t, x_t1)
        source = outputs["p_t"].reshape(x_t.shape[0], 10, 2).detach()
        search = outputs["p_hat_t1"].reshape(x_t.shape[0], 10, 2).detach()
        candidate = (
            outputs["p_t1"].reshape(x_t.shape[0], 10, 2).detach().clone()
            .requires_grad_(True)
        )
        transport = ocr_zncc_transport_loss(
            normalized_images_to_retained_rgb(x_t),
            normalized_images_to_retained_rgb(x_t1),
            source,
            search,
            candidate,
            config=config,
        )
        gradient = torch.autograd.grad(
            transport.loss,
            candidate,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        _require(bool(torch.isfinite(gradient).all()), "coordinate gradient is non-finite")
        _require(all(parameter.grad is None for parameter in model.parameters()),
                 "exact-gradient audit populated parameter .grad")

        source_np = source.numpy().astype(np.float64)
        candidate_np = candidate.detach().numpy().astype(np.float64)
        descent = -gradient.detach().numpy().astype(np.float64)
        accepted = transport.accepted.numpy().astype(bool)
        source_inside = transport.source_patch_inside.numpy().astype(bool)
        _require(not bool(np.any(accepted & ~source_inside)),
                 "accepted source patch crosses the image boundary")
        oracle = _rotation(source_np, ANGLE_DEGREES)
        correction = oracle - candidate_np
        descent_norm = np.linalg.norm(descent, axis=-1)
        correction_norm = np.linalg.norm(correction, axis=-1)
        usable = accepted & (descent_norm > NORM_EPSILON) & (
            correction_norm > NORM_EPSILON
        )
        cosine = np.full(descent_norm.shape, np.nan, dtype=np.float64)
        cosine[usable] = (
            np.sum(descent * correction, axis=-1)[usable]
            / (descent_norm[usable] * correction_norm[usable])
        )
        unit = np.divide(
            descent,
            descent_norm[..., None],
            out=np.zeros_like(descent),
            where=descent_norm[..., None] > NORM_EPSILON,
        )
        stepped = candidate_np.copy()
        stepped[usable] = np.clip(
            candidate_np[usable] + grid_step(config.grid_size) * unit[usable],
            -1.0,
            1.0,
        )
        local_masks = target_masks[offset : offset + x_t.shape[0]]
        diagonals = np.asarray(
            target_diagonals[offset : offset + x_t.shape[0]], dtype=np.float64
        )[:, None]
        on_before = _sample_masks(candidate_np, local_masks)
        on_after = _sample_masks(stepped, local_masks)
        error_before = np.linalg.norm(oracle - candidate_np, axis=-1) / 2.0 / diagonals
        error_after = np.linalg.norm(oracle - stepped, axis=-1) / 2.0 / diagonals
        error_delta = error_after - error_before
        pair_ids = [str(item) for item in batch["pair_id"]]
        for local_index, pair_id in enumerate(pair_ids):
            for channel in channels:
                is_usable = bool(usable[local_index, channel])
                rows.append({
                    "pair_id": pair_id,
                    "channel": int(channel),
                    "accepted_match": bool(accepted[local_index, channel]),
                    "source_patch_inside": bool(
                        transport.source_patch_inside[local_index, channel]
                    ),
                    "usable_direction": is_usable,
                    "direction_cosine": (
                        float(cosine[local_index, channel]) if is_usable else None
                    ),
                    "one_cell_material_error_delta_objdiag": (
                        float(error_delta[local_index, channel]) if is_usable else None
                    ),
                    "on_object_before": bool(on_before[local_index, channel]),
                    "on_object_after": bool(on_after[local_index, channel]),
                    "source_coordinate": source_np[local_index, channel].tolist(),
                    "candidate_coordinate": candidate_np[local_index, channel].tolist(),
                    "match_coordinate": transport.match_coordinates[
                        local_index, channel
                    ].numpy().astype(np.float64).tolist(),
                    "exact_coordinate_descent": descent[local_index, channel].tolist(),
                })
        offset += int(x_t.shape[0])
    _require(offset == len(dataset), "exact-gradient loader did not consume full split")
    after_state = _state_sha256(model)
    if stage == "frozen_control":
        _require(before_state == after_state, "frozen-control model state changed")
    return {
        "cell_id": cell_id,
        "stage": stage,
        "ocr_zncc_config": config.as_dict(),
        "channel_indices": [int(item) for item in channels],
        "execution": {
            "device": "cpu",
            "model_mode": "eval" if stage == "frozen_control" else "train",
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "parameter_grad_fields_unchanged": True,
            "model_state_sha256_before": before_state,
            "model_state_sha256_after": after_state,
            "batchnorm_state_change_expected": stage == "random_initialization",
        },
        "rows": rows,
        "summary": _summary(rows, channels),
    }


def run(
    *,
    repo_root: Path,
    data_root: Path,
    matrix_root: Path,
    output_path: Path,
    batch_size: int,
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
    _require(len(dataset) == 147, "training split pair count differs")
    target_masks, target_diagonals = _load_target_masks(dataset)
    controls = {
        cell_id: _validate_control_cell(
            repo_root=repo_root,
            matrix_root=matrix_root,
            cell_id=cell_id,
        )
        for cell_id in CELL_IDS
    }
    cells = []
    for exclusion_radius in (0, 1):
        config = OCRZNCCConfig(
            peak_margin_exclusion_radius_cells=exclusion_radius
        )
        for recipe in RECIPES:
            for seed in SEEDS:
                cell_id = f"{recipe}__r64__seed{seed}"
                control = controls[cell_id]
                cells.append(_cell(
                    cell_id=cell_id,
                    stage="frozen_control",
                    model=control["model"],
                    channels=control["eligible_channels"],
                    dataset=dataset,
                    target_masks=target_masks,
                    target_diagonals=target_diagonals,
                    config=config,
                    batch_size=batch_size,
                ))
                cells.append(_cell(
                    cell_id=cell_id,
                    stage="random_initialization",
                    model=_initial_model(recipe, seed),
                    channels=tuple(range(10)),
                    dataset=dataset,
                    target_masks=target_masks,
                    target_diagonals=target_diagonals,
                    config=config,
                    batch_size=batch_size,
                ))
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ocr_zncc_exact_coordinate_gradient_direction_audit",
        "checkout": checkout,
        "inputs": {
            "audit_source": _file_record(Path(__file__)),
            "ocr_zncc_source": _file_record(repo_root / "keypoint_net/ocr_zncc_transport.py"),
            "stage0_source": _file_record(repo_root / "keypoint_net/run_ocr_zncc_stage0.py"),
            "train_index": _file_record(train_index),
            "matrix_root": str(matrix_root.resolve(strict=True)),
            "data_root": str(data_root.resolve(strict=True)),
            "dataset_binding_sha256": DATASET_BINDING_SHA256,
        },
        "semantic_lock": {
            "question": (
                "Does the exact component-wise SmoothL1 coordinate descent point "
                "toward the known +6 degree hammer material correction?"
            ),
            "must_not_infer": [
                "training_outcome",
                "parameter-space_update_semantics",
                "retrospective_stage0_pass",
            ],
            "material_oracle_use": "evaluation_only",
        },
        "cells": cells,
        "decision_summary": {
            "all_frozen_infinitesimal_direction_gates": all(
                cell["summary"]["infinitesimal_direction_gate"]
                for cell in cells if cell["stage"] == "frozen_control"
            ),
            "all_initial_infinitesimal_direction_gates": all(
                cell["summary"]["infinitesimal_direction_gate"]
                for cell in cells if cell["stage"] == "random_initialization"
            ),
            "all_frozen_one_cell_finite_step_gates": all(
                cell["summary"]["one_cell_finite_step_gate"]
                for cell in cells if cell["stage"] == "frozen_control"
            ),
            "all_initial_one_cell_finite_step_gates": all(
                cell["summary"]["one_cell_finite_step_gate"]
                for cell in cells if cell["stage"] == "random_initialization"
            ),
            "frozen_50_percent_coverage_gate_status": "remains_failed_unchanged",
            "training_authorized_by_artifact": False,
        },
        "execution": {
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "gpu_jobs_submitted": 0,
        },
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    _write_exclusive(output_path, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    _require(args.batch_size > 0, "batch size must be positive")
    torch.set_num_threads(1)
    document = run(
        repo_root=args.repo_root,
        data_root=args.data_root,
        matrix_root=args.matrix_root,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(json.dumps(document["decision_summary"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
