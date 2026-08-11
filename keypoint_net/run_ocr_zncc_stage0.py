"""CPU-only OCR-ZNCC Stage-0 direction gate and total-gradient calibration.

This entry point performs no optimizer or scheduler step and never submits a
job.  It audits frozen Task-55 and Task-80 controls on the complete hammer
training split.  Only if both three-seed recipe gates pass does it run one
combined six-cell initialization calibration and choose one shared coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from keypoint_net.dataset import IndexPairDataset
from keypoint_net.model import PhaseAModel, compute_losses
from keypoint_net.ocr_zncc_transport import (
    OCRZNCCConfig,
    grid_step,
    normalized_images_to_retained_rgb,
    ocr_zncc_transport_loss,
)


SCHEMA_VERSION = "ocr_zncc_stage0.v2_distinct_competitor"
EXPECTED_BRANCH = "agent/ocr-zncc-training-20260811"
AUTHORIZED_BASE_COMMIT = "4521538621b410421b1058603d090e2b09ec4178"
CONTROL_SOURCE_COMMIT = "196245fdc8b6d65a5348d1addebcfc3c58ddb3d6"
CONTROL_SOURCE_BRANCH = "agent/representation-oracles-20260726"
TRAIN_INDEX_RELPATH = (
    "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
    "roll__world_z__forward__train.json"
)
TRAIN_INDEX_SHA256 = "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
TRAIN_INDEX_CONTENT_SHA256 = "40a70495d144fcf838146178bbdb2756251a413fcea65bc2626a87a5f75c5432"
DATASET_BINDING_SHA256 = "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
OBJECT_NAME = "engineers_hammer_vray"
RECIPES = ("task55_clean", "task80_assisted")
SEEDS = (42, 43, 44)
CELL_IDS = tuple(f"{recipe}__r64__seed{seed}" for recipe in RECIPES for seed in SEEDS)
ANGLE_DEGREES = 6.0
ON_OBJECT_RATE_MINIMUM = 0.75
USABLE_COVERAGE_MINIMUM = 0.50
RECIPE_PASS_COUNT_MINIMUM = 2
GRADIENT_NORM_FLOOR = 1e-12
MEDIAN_TARGET_CONTRIBUTION = 0.10
COEFFICIENT_CAP = 0.50


class Stage0Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"invalid file: {resolved}")
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, record


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "exclusive write made no progress")
            offset += written
    finally:
        os.close(descriptor)
    return _file_record(path)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_audit_checkout(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve(strict=True)
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    _require(branch == EXPECTED_BRANCH, "Stage-0 audit branch differs")
    _require(tracked_status == "", "Stage-0 audit checkout has tracked changes")
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", AUTHORIZED_BASE_COMMIT, commit],
        check=True,
        capture_output=True,
    )
    return {
        "absolute_path": str(root),
        "branch": branch,
        "commit": commit,
        "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
        "authorized_base_is_ancestor": True,
        "tracked_status": tracked_status,
    }


def _expected_recipe_training(recipe: str) -> dict[str, Any]:
    common = {
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "operator_type": "shared_affine",
        "heatmap_res": 64,
        "num_keypoints": 10,
        "base_channels": 32,
        "img_size": 512,
    }
    if recipe == "task55_clean":
        return {
            **common,
            "lambda_smooth": 0.0,
            "lambda_inv": 0.0,
            "lambda_cycle": 0.0,
            "learn_inverse_operator": False,
        }
    if recipe == "task80_assisted":
        return {
            **common,
            "lambda_smooth": 0.001,
            "lambda_inv": 0.5,
            "lambda_cycle": 0.5,
            "learn_inverse_operator": True,
        }
    raise Stage0Error(f"unknown recipe: {recipe}")


def _validate_control_cell(
    *,
    repo_root: Path,
    matrix_root: Path,
    cell_id: str,
) -> dict[str, Any]:
    recipe, _, seed_text = cell_id.partition("__r64__seed")
    seed = int(seed_text)
    run_root = matrix_root / "runs" / cell_id
    evaluation_path = matrix_root / "evaluations" / f"{cell_id}.json"
    receipt, receipt_record = _load_json(run_root / "COMPLETED_RUN_RECEIPT.json")
    config, config_record = _load_json(run_root / "config.json")
    history_record = _file_record(run_root / "history.json")
    checkpoint_record = _file_record(run_root / "best_model.pt")
    evaluation, evaluation_record = _load_json(evaluation_path)

    _require(receipt.get("cell_id") == cell_id, "receipt cell differs")
    _require(receipt.get("source_branch") == CONTROL_SOURCE_BRANCH, "control source branch differs")
    _require(receipt.get("source_commit") == CONTROL_SOURCE_COMMIT, "control source commit differs")
    execution = receipt.get("execution")
    _require(isinstance(execution, dict), "receipt execution block missing")
    _require(execution.get("training_completed") is True, "control training is incomplete")
    _require(execution.get("authoritative_checkpoint_name") == "best_model.pt", "checkpoint name differs")
    _require(execution.get("selection_use_authorized") is True, "checkpoint selection was unauthorized")
    embedded = receipt.get("expected_embedded_config")
    _require(isinstance(embedded, dict), "embedded control config missing")
    _require(embedded.get("recipe") == recipe and embedded.get("seed") == seed, "recipe/seed differs")
    _require(embedded.get("head_package") == "64", "control head package differs")
    _require(embedded.get("object", {}).get("name") == OBJECT_NAME, "control object differs")
    training = embedded.get("training")
    _require(isinstance(training, dict), "embedded training config missing")
    for key, expected in _expected_recipe_training(recipe).items():
        _require(training.get(key) == expected, f"{cell_id} training.{key} differs")
    _require(
        training.get("checkpoint_policy", {}).get("selection") == "minimum_total_validation_loss",
        "historical control selector differs from its frozen receipt",
    )
    train_split = embedded.get("split", {}).get("train", {})
    _require(train_split.get("file_sha256") == TRAIN_INDEX_SHA256, "control train split hash differs")
    _require(train_split.get("content_hash_sha256") == TRAIN_INDEX_CONTENT_SHA256, "train content hash differs")
    _require(train_split.get("pair_count_after_object_filter") == 147, "control train pair count differs")
    _require(embedded.get("dataset", {}).get("binding_sha256") == DATASET_BINDING_SHA256, "dataset binding differs")

    receipt_files = receipt.get("files")
    _require(isinstance(receipt_files, dict), "receipt file records missing")
    for role, observed in (
        ("checkpoint", checkpoint_record),
        ("config", config_record),
        ("history", history_record),
    ):
        expected = receipt_files.get(role)
        _require(isinstance(expected, dict), f"receipt {role} record missing")
        _require(expected.get("sha256") == observed["sha256"], f"{role} SHA-256 differs")
        _require(expected.get("size_bytes") == observed["size_bytes"], f"{role} size differs")
    _require(config.get("cell_id") == cell_id and config.get("seed") == seed, "run config identity differs")
    for key, expected in _expected_recipe_training(recipe).items():
        if key == "learn_inverse_operator":
            observed = config.get("learn_inverse_operator_effective", config.get(key))
        else:
            observed = config.get(key)
        _require(observed == expected, f"run config {key} differs")

    _require(evaluation.get("case_id") == cell_id, "evaluator case differs")
    _require(evaluation.get("critical_failures") == [], "evaluator has critical failures")
    validated_checkpoint = evaluation.get("validated_checkpoint")
    _require(isinstance(validated_checkpoint, dict), "validated checkpoint missing")
    _require(validated_checkpoint.get("sha256") == checkpoint_record["sha256"], "evaluator checkpoint differs")
    _require(validated_checkpoint.get("size_bytes") == checkpoint_record["size_bytes"], "evaluator checkpoint size differs")
    _require(evaluation.get("provenance", {}).get("source_commit") == CONTROL_SOURCE_COMMIT, "evaluator source differs")
    health = evaluation.get("channel_health")
    _require(isinstance(health, dict), "channel health missing")
    eligible = health.get("eligible_indices")
    _require(isinstance(eligible, list) and len(eligible) >= 2, "eligible channels are insufficient")
    _require(len(set(int(item) for item in eligible)) == len(eligible), "eligible channels repeat")
    _require(all(0 <= int(item) < 10 for item in eligible), "eligible channel out of range")

    checkpoint = torch.load(checkpoint_record["absolute_path"], map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict), "checkpoint state missing")
    model = PhaseAModel(
        num_keypoints=int(training["num_keypoints"]),
        base_channels=int(training["base_channels"]),
        temperature=float(training["temperature"]),
        padding_mode=str(training["padding_mode"]),
        operator_type=str(training["operator_type"]),
        learn_inverse_operator=bool(training["learn_inverse_operator"]),
        heatmap_res=int(training["heatmap_res"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "cell_id": cell_id,
        "recipe": recipe,
        "seed": seed,
        "training": training,
        "eligible_channels": [int(item) for item in eligible],
        "historical_checkpoint_selector": "minimum_total_validation_loss",
        "future_auxiliary_checkpoint_selector": "minimum_base_validation_loss",
        "receipt": receipt_record,
        "config": config_record,
        "history": history_record,
        "checkpoint": checkpoint_record,
        "evaluation": evaluation_record,
        "evaluation_result_content_sha256": evaluation.get("result_content_sha256"),
        "strict_state_dict_load": True,
        "model": model,
    }


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rotation(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    radians = math.radians(float(angle_degrees))
    matrix = np.asarray(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    return np.matmul(np.asarray(points, dtype=np.float64), matrix.T)


def _load_target_masks(dataset: IndexPairDataset) -> tuple[list[np.ndarray], list[float]]:
    masks: list[np.ndarray] = []
    diagonals: list[float] = []
    for index, sample in enumerate(dataset.samples):
        metadata = json.loads(sample["pair_metadata_json"])
        path = dataset.data_root / metadata["dst_mask_relpath"]
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        _require(mask.ndim == 2 and bool(mask.any()), f"target mask {index} is invalid")
        rows, columns = np.nonzero(mask)
        height, width = mask.shape
        diagonal = math.hypot(
            float(columns.max() - columns.min()) / float(width - 1),
            float(rows.max() - rows.min()) / float(height - 1),
        )
        _require(diagonal > 0.0 and math.isfinite(diagonal), f"target mask {index} diagonal invalid")
        masks.append(mask)
        diagonals.append(diagonal)
    return masks, diagonals


def _sample_masks(points: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[-1] == 2, "mask point shape differs")
    _require(values.shape[0] == len(masks), "mask count differs")
    output = np.zeros(values.shape[:-1], dtype=bool)
    for row, mask in enumerate(masks):
        height, width = mask.shape
        x = np.rint((values[row, :, 0] + 1.0) * (width - 1) / 2.0).astype(np.int64)
        y = np.rint((values[row, :, 1] + 1.0) * (height - 1) / 2.0).astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        output[row, valid] = mask[y[valid], x[valid]]
    return output


def _descriptive(values: Iterable[float], *, sample_unit: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, f"{sample_unit} has no values")
    _require(bool(np.isfinite(array).all()), f"{sample_unit} has non-finite values")
    return {
        "sample_unit": sample_unit,
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "uncertainty_bar": None,
        "inference": "descriptive_correlated_training_pair_channel_values",
    }


def _summarize_direction_rows(
    rows: Sequence[Mapping[str, Any]],
    eligible_channels: Sequence[int],
) -> dict[str, Any]:
    total = len(rows)
    _require(total > 0, "direction cell has no rows")
    accepted = [row for row in rows if bool(row["accepted_match"])]
    usable = [row for row in rows if bool(row["usable_direction"])]
    raw_coverage = len(accepted) / total
    usable_coverage = len(usable) / total
    if usable:
        cosine = _descriptive(
            (float(row["direction_cosine"]) for row in usable),
            sample_unit="eligible_training_pair_channel_usable_direction_cosine",
        )
        error_delta = _descriptive(
            (float(row["one_cell_material_error_delta_objdiag"]) for row in usable),
            sample_unit="eligible_training_pair_channel_usable_one_cell_error_delta_objdiag",
        )
    else:
        cosine = None
        error_delta = None

    channel_rows = []
    for channel in eligible_channels:
        local = [row for row in rows if int(row["channel"]) == int(channel)]
        _require(local, f"eligible channel {channel} has no rows")
        before_rate = float(np.mean([bool(row["on_object_before"]) for row in local]))
        after_rate = float(np.mean([bool(row["on_object_after"]) for row in local]))
        channel_rows.append(
            {
                "channel": int(channel),
                "sample_count": len(local),
                "on_object_rate_before": before_rate,
                "on_object_rate_after": after_rate,
                "on_object_before": before_rate >= ON_OBJECT_RATE_MINIMUM,
                "on_object_after": after_rate >= ON_OBJECT_RATE_MINIMUM,
            }
        )
    before_count = sum(bool(row["on_object_before"]) for row in channel_rows)
    after_count = sum(bool(row["on_object_after"]) for row in channel_rows)
    passed = bool(
        usable_coverage >= USABLE_COVERAGE_MINIMUM
        and cosine is not None
        and cosine["median"] > 0.0
        and error_delta is not None
        and error_delta["median"] < 0.0
        and after_count >= before_count
    )
    return {
        "eligible_row_count": total,
        "accepted_match_count": len(accepted),
        "accepted_match_coverage": raw_coverage,
        "usable_direction_count": len(usable),
        "usable_direction_coverage": usable_coverage,
        "direction_cosine": cosine,
        "one_cell_material_error_delta_objdiag": error_delta,
        "active_on_object": {
            "definition": f"originally evaluator-eligible channel with target-mask rate >= {ON_OBJECT_RATE_MINIMUM}",
            "before_count": before_count,
            "after_count": after_count,
            "change": after_count - before_count,
            "per_channel": channel_rows,
        },
        "cell_pass": passed,
        "preregistered_engineering_rule": {
            "usable_direction_coverage_at_least": USABLE_COVERAGE_MINIMUM,
            "median_direction_cosine_strictly_greater_than": 0.0,
            "median_one_cell_material_error_delta_strictly_less_than": 0.0,
            "active_on_object_after_at_least_before": True,
            "threshold_status": "preregistered_engineering_thresholds_not_established_evidence",
        },
    }


def _direction_cell(
    *,
    control: Mapping[str, Any],
    dataset: IndexPairDataset,
    target_masks: Sequence[np.ndarray],
    target_diagonals: Sequence[float],
    batch_size: int,
    config: OCRZNCCConfig,
) -> dict[str, Any]:
    model = control["model"]
    _require(isinstance(model, PhaseAModel), "control model missing")
    model.eval()
    before_state = _state_sha256(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    eligible = [int(item) for item in control["eligible_channels"]]
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x_t = batch["x_t"].cpu()
            x_t1 = batch["x_t1"].cpu()
            outputs = model(x_t, x_t1)
            source = outputs["p_t"].reshape(x_t.shape[0], 10, 2)
            candidate = outputs["p_t1"].reshape(x_t.shape[0], 10, 2)
            search = outputs["p_hat_t1"].reshape(x_t.shape[0], 10, 2)
            transport = ocr_zncc_transport_loss(
                normalized_images_to_retained_rgb(x_t),
                normalized_images_to_retained_rgb(x_t1),
                source,
                search,
                candidate,
                config=config,
            )
            source_np = source.numpy().astype(np.float64)
            candidate_np = candidate.numpy().astype(np.float64)
            matches_np = transport.match_coordinates.numpy().astype(np.float64)
            accepted_np = transport.accepted.numpy().astype(bool)
            descent = matches_np - candidate_np
            oracle = _rotation(source_np, ANGLE_DEGREES)
            correction = oracle - candidate_np
            descent_norm = np.linalg.norm(descent, axis=-1)
            correction_norm = np.linalg.norm(correction, axis=-1)
            usable = accepted_np & (descent_norm > 1e-12) & (correction_norm > 1e-12)
            cosine = np.full(descent_norm.shape, np.nan, dtype=np.float64)
            cosine[usable] = (
                np.sum(descent * correction, axis=-1)[usable]
                / (descent_norm[usable] * correction_norm[usable])
            )
            unit = np.divide(
                descent,
                descent_norm[..., None],
                out=np.zeros_like(descent),
                where=descent_norm[..., None] > 1e-12,
            )
            stepped = candidate_np.copy()
            stepped[usable] = np.clip(
                candidate_np[usable] + grid_step(config.grid_size) * unit[usable],
                -1.0,
                1.0,
            )
            local_masks = target_masks[offset : offset + x_t.shape[0]]
            local_diagonals = np.asarray(
                target_diagonals[offset : offset + x_t.shape[0]], dtype=np.float64
            )[:, None]
            on_before = _sample_masks(candidate_np, local_masks)
            on_after = _sample_masks(stepped, local_masks)
            error_before = np.linalg.norm(oracle - candidate_np, axis=-1) / 2.0 / local_diagonals
            error_after = np.linalg.norm(oracle - stepped, axis=-1) / 2.0 / local_diagonals
            error_delta = error_after - error_before
            pair_ids = [str(item) for item in batch["pair_id"]]
            for batch_index, pair_id in enumerate(pair_ids):
                for channel in eligible:
                    is_usable = bool(usable[batch_index, channel])
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "channel": channel,
                            "eligible_channel": True,
                            "accepted_match": bool(accepted_np[batch_index, channel]),
                            "usable_direction": is_usable,
                            "direction_cosine": float(cosine[batch_index, channel]) if is_usable else None,
                            "one_cell_material_error_before_objdiag": float(error_before[batch_index, channel]),
                            "one_cell_material_error_after_objdiag": float(error_after[batch_index, channel]),
                            "one_cell_material_error_delta_objdiag": float(error_delta[batch_index, channel]) if is_usable else None,
                            "on_object_before": bool(on_before[batch_index, channel]),
                            "on_object_after": bool(on_after[batch_index, channel]),
                            "source_coordinate": [float(item) for item in source_np[batch_index, channel]],
                            "operator_search_centre": [float(item) for item in search[batch_index, channel].numpy()],
                            "candidate_coordinate": [float(item) for item in candidate_np[batch_index, channel]],
                            "match_coordinate": [float(item) for item in matches_np[batch_index, channel]],
                            "confidence": float(transport.confidence[batch_index, channel]),
                            "source_patch_rms": float(transport.source_patch_rms[batch_index, channel]),
                            "matched_target_patch_rms": float(transport.matched_target_patch_rms[batch_index, channel]),
                            "target_peak_zncc": float(transport.target_peak_zncc[batch_index, channel]),
                            "target_peak_margin": float(transport.target_peak_margin[batch_index, channel]),
                            "reciprocal_peak_zncc": float(transport.reciprocal_peak_zncc[batch_index, channel]),
                            "reciprocal_peak_margin": float(transport.reciprocal_peak_margin[batch_index, channel]),
                            "reciprocal_error_cells": float(transport.reciprocal_error_cells[batch_index, channel]),
                            "target_competitor_distance_cells": float(
                                transport.target_competitor_distance_cells[
                                    batch_index, channel
                                ]
                            ),
                            "reciprocal_competitor_distance_cells": float(
                                transport.reciprocal_competitor_distance_cells[
                                    batch_index, channel
                                ]
                            ),
                        }
                    )
            offset += int(x_t.shape[0])
    after_state = _state_sha256(model)
    _require(offset == len(dataset), "direction loader did not consume full split")
    _require(before_state == after_state, "frozen control state changed")
    _require(len(rows) == len(dataset) * len(eligible), "direction row count differs")
    document = {
        "schema_version": f"{SCHEMA_VERSION}.direction_cell",
        "artifact_type": "ocr_zncc_direction_cell",
        "cell_id": control["cell_id"],
        "recipe": control["recipe"],
        "seed": control["seed"],
        "control_provenance": {
            key: value
            for key, value in control.items()
            if key not in {"model", "training", "eligible_channels"}
        },
        "eligible_channels": eligible,
        "dataset": {
            "split": "train",
            "sample_count": len(dataset),
            "sample_unit": "directed_hammer_world_z_roll_training_pair",
            "index_sha256": dataset.index_sha256,
            "index_content_sha256": dataset.content_hash_sha256,
            "dataset_binding_sha256": dataset.dataset_binding_sha256,
            "material_oracle_use": "evaluation_only_known_positive_6_degree_world_z_roll",
        },
        "execution": {
            "device": "cpu",
            "model_mode": "eval",
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_performed": False,
            "state_sha256_before": before_state,
            "state_sha256_after": after_state,
            "state_unchanged": True,
        },
        "ocr_zncc_config": config.as_dict(),
        "rows": rows,
        "summary": _summarize_direction_rows(rows, eligible),
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    return document


def _parameter_items(model: PhaseAModel) -> list[tuple[str, torch.nn.Parameter]]:
    items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("extractor.encoder.") or name.startswith("extractor.heatmap_head."))
    ]
    _require(bool(items), "extractor parameter set is empty")
    return items


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
    pieces = []
    for gradient in gradients:
        piece = gradient.detach().cpu().to(dtype=torch.float64).reshape(-1)
        _require(bool(torch.isfinite(piece).all()), "calibration gradient is non-finite")
        pieces.append(piece)
    _require(bool(pieces), "calibration gradient vector is empty")
    return torch.cat(pieces)


def _weighted_mean(vectors: Sequence[torch.Tensor], weights: Sequence[int]) -> torch.Tensor:
    _require(len(vectors) == len(weights) and bool(vectors), "gradient aggregation inputs differ")
    _require(sum(weights) > 0, "gradient aggregation has zero total weight")
    shape = vectors[0].shape
    _require(all(vector.shape == shape for vector in vectors), "gradient vector shapes differ")
    total = torch.zeros_like(vectors[0])
    for vector, weight in zip(vectors, weights):
        _require(weight >= 0, "gradient aggregation weight is negative")
        total.add_(vector, alpha=float(weight))
    return total / float(sum(weights))


def _strict_norm(vector: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(vector).item())
    _require(math.isfinite(value) and value > GRADIENT_NORM_FLOOR, "gradient norm is invalid")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _calibration_cell(
    *,
    recipe: str,
    seed: int,
    dataset: IndexPairDataset,
    batch_size: int,
    config: OCRZNCCConfig,
) -> dict[str, Any]:
    _seed_everything(seed)
    recipe_training = _expected_recipe_training(recipe)
    model = PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(recipe_training["learn_inverse_operator"]),
        heatmap_res=64,
    )
    model.train()
    parameter_items = _parameter_items(model)
    parameter_names = [name for name, _ in parameter_items]
    parameters = [parameter for _, parameter in parameter_items]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    base_vectors: list[torch.Tensor] = []
    auxiliary_vectors: list[torch.Tensor] = []
    sample_counts: list[int] = []
    accepted_counts: list[int] = []
    before_state = _state_sha256(model)
    for batch in loader:
        x_t = batch["x_t"].cpu()
        x_t1 = batch["x_t1"].cpu()
        action_labels = batch["action_label"].cpu().long()
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=float(recipe_training["lambda_smooth"]),
            lambda_disp=float(recipe_training["lambda_disp"]),
            lambda_ent=float(recipe_training["lambda_ent"]),
            lambda_act=float(recipe_training["lambda_act"]),
            lambda_loc=float(recipe_training["lambda_loc"]),
            lambda_inv=float(recipe_training["lambda_inv"]),
            lambda_cycle=float(recipe_training["lambda_cycle"]),
            lambda_attach=0.0,
            sigma=0.1,
            num_keypoints=10,
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=30.0,
        )
        transport = ocr_zncc_transport_loss(
            normalized_images_to_retained_rgb(x_t),
            normalized_images_to_retained_rgb(x_t1),
            outputs["p_t"].reshape(x_t.shape[0], 10, 2),
            outputs["p_hat_t1"].reshape(x_t.shape[0], 10, 2),
            outputs["p_t1"].reshape(x_t.shape[0], 10, 2),
            config=config,
        )
        base_vectors.append(_gradient_vector(losses["base_loss"], parameters, retain_graph=True))
        auxiliary_vectors.append(_gradient_vector(transport.loss, parameters, retain_graph=False))
        sample_counts.append(int(x_t.shape[0]))
        accepted_counts.append(transport.accepted_count)
        _require(all(parameter.grad is None for parameter in model.parameters()), "autograd populated .grad")
    _require(sum(sample_counts) == len(dataset), "calibration did not consume full split")
    _require(sum(accepted_counts) > 0, "calibration has no accepted OCR-ZNCC matches")
    base_mean = _weighted_mean(base_vectors, sample_counts)
    auxiliary_mean = _weighted_mean(auxiliary_vectors, accepted_counts)
    base_norm = _strict_norm(base_mean)
    auxiliary_norm = _strict_norm(auxiliary_mean)
    ratio = auxiliary_norm / base_norm
    _require(math.isfinite(ratio) and ratio > 0.0, "auxiliary/base gradient ratio is invalid")
    return {
        "cell_id": f"{recipe}__r64__seed{seed}",
        "recipe": recipe,
        "seed": seed,
        "model_initialization_seed": seed,
        "sample_count": len(dataset),
        "accepted_match_count": sum(accepted_counts),
        "accepted_match_coverage_all_channels": sum(accepted_counts) / (len(dataset) * 10),
        "base_gradient_norm": base_norm,
        "auxiliary_gradient_norm": auxiliary_norm,
        "auxiliary_to_base_gradient_ratio": ratio,
        "parameter_names": parameter_names,
        "parameter_set": "all_trainable_extractor.encoder_and_extractor.heatmap_head",
        "base_gradient_reduction": "sample_weighted_full_training_split_mean_then_global_float64_l2",
        "auxiliary_gradient_reduction": "accepted_match_weighted_full_training_split_mean_then_global_float64_l2",
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": _state_sha256(model),
        "batchnorm_state_change_expected_from_train_mode_for_ephemeral_model": True,
        "ephemeral_model_discarded_after_cell": True,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
    }


def _calibration_receipt(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(cells) == 6, "combined calibration requires exactly six cells")
    ratios = [float(cell["auxiliary_to_base_gradient_ratio"]) for cell in cells]
    _require(all(math.isfinite(value) and value > 0.0 for value in ratios), "calibration ratios invalid")
    median_ratio = float(statistics.median(ratios))
    coefficient = min(COEFFICIENT_CAP, MEDIAN_TARGET_CONTRIBUTION / median_ratio)
    _require(math.isfinite(coefficient) and coefficient > 0.0, "calibrated coefficient invalid")
    recipe_summaries = {}
    for recipe in RECIPES:
        local = [float(cell["auxiliary_to_base_gradient_ratio"]) for cell in cells if cell["recipe"] == recipe]
        _require(len(local) == 3, f"{recipe} calibration cell count differs")
        recipe_summaries[recipe] = {
            "ratios_by_seed": local,
            "median_unscaled_auxiliary_to_base_ratio": float(statistics.median(local)),
            "median_scaled_auxiliary_to_base_contribution": coefficient * float(statistics.median(local)),
        }
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}.calibration",
        "artifact_type": "ocr_zncc_combined_task55_task80_gradient_calibration",
        "status": "calibrated" if coefficient < COEFFICIENT_CAP else "calibrated_at_safety_cap",
        "cells": list(cells),
        "recipe_summaries": recipe_summaries,
        "median_unscaled_auxiliary_to_base_ratio_all_six_cells": median_ratio,
        "coefficient": coefficient,
        "scaled_median_contribution_all_six_cells": coefficient * median_ratio,
        "coefficient_rule": {
            "formula": "min(0.5, 0.10 / median_six_cell_auxiliary_to_base_gradient_ratio)",
            "target_median_contribution": MEDIAN_TARGET_CONTRIBUTION,
            "coefficient_safety_cap": COEFFICIENT_CAP,
            "one_shared_coefficient_across_task55_and_task80": True,
            "raw_lambda_sweep_performed": False,
        },
        "interpretation": {
            "measures": "global_l2_norm_of_full_split_auxiliary_gradient_relative_to_entire_recipe_base_loss_gradient_over_extractor_parameters",
            "does_not_measure": [
                "componentwise_base_loss_gradients",
                "gradient_direction_agreement_between_base_and_auxiliary",
                "training_outcome",
            ],
        },
        "checkpoint_selector_for_any_future_auxiliary_training": "minimum_base_validation_loss",
        "optimizer_steps": 0,
        "training_performed": False,
    }
    receipt["content_hash_sha256"] = canonical_sha256(receipt)
    return receipt


def run_stage0(
    *,
    repo_root: Path,
    data_root: Path,
    matrix_root: Path,
    output_dir: Path,
    direction_batch_size: int,
    calibration_batch_size: int,
    peak_margin_exclusion_radius_cells: int,
) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkout = validate_audit_checkout(repo_root)
    config = OCRZNCCConfig(
        peak_margin_exclusion_radius_cells=peak_margin_exclusion_radius_cells,
    )
    config.validate()
    train_index = repo_root / TRAIN_INDEX_RELPATH
    _require(_sha256_file(train_index) == TRAIN_INDEX_SHA256, "local train index SHA-256 differs")
    _require(data_root.name == "_tdw_world_z_roll_base_panel_512_v2", "dataset basename differs")
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
    _require(len(dataset) == 147, "training split object-filtered count differs")
    target_masks, target_diagonals = _load_target_masks(dataset)
    controls = [
        _validate_control_cell(repo_root=repo_root, matrix_root=matrix_root, cell_id=cell_id)
        for cell_id in CELL_IDS
    ]
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}.provenance",
        "artifact_type": "ocr_zncc_stage0_provenance",
        "checkout": checkout,
        "source_files": {
            "ocr_zncc_transport": _file_record(repo_root / "keypoint_net/ocr_zncc_transport.py"),
            "stage0_runner": _file_record(repo_root / "keypoint_net/run_ocr_zncc_stage0.py"),
            "model": _file_record(repo_root / "keypoint_net/model.py"),
            "dataset": _file_record(repo_root / "keypoint_net/dataset.py"),
            "train_index": _file_record(train_index),
        },
        "dataset_root": str(data_root.resolve(strict=True)),
        "matrix_root": str(matrix_root.resolve(strict=True)),
        "control_cells": [
            {key: value for key, value in control.items() if key not in {"model", "training", "eligible_channels"}}
            for control in controls
        ],
        "selector_resolution": {
            "historical_frozen_controls": "minimum_total_validation_loss",
            "future_ocr_zncc_auxiliary_experiments": "minimum_base_validation_loss",
            "historical_evidence_rewritten": False,
            "ambiguity_status": "resolved_before_stage0_execution",
        },
        "authorization_boundary": {
            "cpu_only": True,
            "optimizer_steps": 0,
            "gpu_jobs_submitted": False,
            "training_authorized": False,
            "task55_and_task80_direction_diagnostics_authorized": True,
            "combined_task55_task80_gradient_calibration_authorized_only_after_both_direction_gates_pass": True,
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": str(np.__version__),
            "pytorch_version": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    provenance["content_hash_sha256"] = canonical_sha256(provenance)
    provenance_record = _write_exclusive(output_dir / "STAGE0_PROVENANCE.json", provenance)

    direction_documents = []
    direction_records = []
    for control in controls:
        print(f"[OCR-ZNCC direction] {control['cell_id']}", flush=True)
        document = _direction_cell(
            control=control,
            dataset=dataset,
            target_masks=target_masks,
            target_diagonals=target_diagonals,
            batch_size=direction_batch_size,
            config=config,
        )
        record = _write_exclusive(
            output_dir / f"DIRECTION__{control['cell_id']}.json",
            document,
        )
        direction_documents.append(document)
        direction_records.append(record)

    recipe_direction = {}
    both_pass = True
    for recipe in RECIPES:
        local = [document for document in direction_documents if document["recipe"] == recipe]
        pass_count = sum(bool(document["summary"]["cell_pass"]) for document in local)
        recipe_pass = pass_count >= RECIPE_PASS_COUNT_MINIMUM
        both_pass = both_pass and recipe_pass
        recipe_direction[recipe] = {
            "cell_pass_count": pass_count,
            "cell_count": len(local),
            "recipe_pass": recipe_pass,
            "preregistered_engineering_rule": f"at_least_{RECIPE_PASS_COUNT_MINIMUM}_of_3_cells_pass",
            "cells": [
                {
                    "cell_id": document["cell_id"],
                    "summary": document["summary"],
                }
                for document in local
            ],
        }
    direction_summary = {
        "schema_version": f"{SCHEMA_VERSION}.direction_summary",
        "artifact_type": "ocr_zncc_task55_task80_direction_summary",
        "recipe_results": recipe_direction,
        "both_recipe_gates_pass": both_pass,
        "calibration_authorized_by_stage0_rule": both_pass,
        "direction_cell_artifacts": direction_records,
    }
    direction_summary["content_hash_sha256"] = canonical_sha256(direction_summary)
    direction_summary_record = _write_exclusive(
        output_dir / "DIRECTION_SUMMARY.json", direction_summary
    )

    calibration = None
    calibration_record = None
    if both_pass:
        calibration_cells = []
        for recipe in RECIPES:
            for seed in SEEDS:
                print(f"[OCR-ZNCC calibration] {recipe} seed {seed}", flush=True)
                calibration_cells.append(
                    _calibration_cell(
                        recipe=recipe,
                        seed=seed,
                        dataset=dataset,
                        batch_size=calibration_batch_size,
                        config=config,
                    )
                )
        calibration = _calibration_receipt(calibration_cells)
        calibration_record = _write_exclusive(
            output_dir / "GRADIENT_CALIBRATION.json", calibration
        )

    final = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ocr_zncc_stage0_result",
        "status": "stage0_passed_with_shared_coefficient" if calibration is not None else "blocked_by_direction_gate",
        "decision": {
            "ocr_zncc_semantically_capable_in_both_recipes": both_pass,
            "task55_direction_gate_pass": bool(recipe_direction["task55_clean"]["recipe_pass"]),
            "task80_direction_gate_pass": bool(recipe_direction["task80_assisted"]["recipe_pass"]),
            "shared_auxiliary_coefficient": calibration["coefficient"] if calibration is not None else None,
            "gpu_training_authorized": False,
        },
        "provenance_artifact": provenance_record,
        "direction_summary_artifact": direction_summary_record,
        "calibration_artifact": calibration_record,
        "selector_for_any_future_auxiliary_training": "minimum_base_validation_loss",
        "authorization_boundary": {
            "training_performed": False,
            "optimizer_steps": 0,
            "gpu_jobs_submitted": False,
            "task80_training_started": False,
            "held_out_objects_opened": False,
            "other_transform_families_opened": False,
        },
    }
    final["content_hash_sha256"] = canonical_sha256(final)
    _write_exclusive(output_dir / "STAGE0_RESULT.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-batch-size", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument(
        "--peak-margin-exclusion-radius-cells",
        type=int,
        choices=(0, 1),
        required=True,
        help="0 reproduces the legacy top-two margin; 1 compares spatially distinct peaks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.direction_batch_size <= 0 or args.calibration_batch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    torch.set_num_threads(1)
    result = run_stage0(
        repo_root=args.repo_root,
        data_root=args.data_root,
        matrix_root=args.matrix_root,
        output_dir=args.output_dir,
        direction_batch_size=args.direction_batch_size,
        calibration_batch_size=args.calibration_batch_size,
        peak_margin_exclusion_radius_cells=args.peak_margin_exclusion_radius_cells,
    )
    print(json.dumps(result["decision"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_calibration_receipt",
    "_summarize_direction_rows",
    "run_stage0",
]
