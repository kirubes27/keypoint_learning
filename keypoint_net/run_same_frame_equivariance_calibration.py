"""Full-split initial-state gradient calibration for same-frame equivariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .model import PhaseAModel, compute_losses
    from .run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess
    from .same_frame_equivariance import add_locked_equivariance_to_pair_loss
except ImportError:  # pragma: no cover - direct script compatibility
    from model import PhaseAModel, compute_losses  # type: ignore
    from run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess  # type: ignore
    from same_frame_equivariance import add_locked_equivariance_to_pair_loss  # type: ignore


SCHEMA_VERSION = "same_frame_equivariance_gradient_calibration.v1"
TRAIN_INDEX_SHA256 = "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
DATASET_BINDING_SHA256 = "acfa835813e128b6f3336fe1f51bc14ac6e4cb4cf1b285afe418d4dbdf598d93"
OBJECT_NAME = "engineers_hammer_vray"
RECIPES = ("task55_clean", "task80_assisted")
SEEDS = (42, 43, 44)
TARGET_CONTRIBUTION = 0.10
COEFFICIENT_CAP = 0.5
GRADIENT_NORM_FLOOR = 1e-12


class EquivarianceCalibrationError(ValueError):
    """Raised when calibration differs from its exact full-split lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EquivarianceCalibrationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _training_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _recipe_configuration(recipe: str) -> dict[str, Any]:
    if recipe == "task55_clean":
        return {
            "lambda_smooth": 0.0,
            "lambda_disp": 0.1,
            "lambda_ent": 0.01,
            "lambda_inv": 0.0,
            "lambda_cycle": 0.0,
            "learn_inverse_operator": False,
        }
    if recipe == "task80_assisted":
        return {
            "lambda_smooth": 0.001,
            "lambda_disp": 0.1,
            "lambda_ent": 0.01,
            "lambda_inv": 0.5,
            "lambda_cycle": 0.5,
            "learn_inverse_operator": True,
        }
    raise EquivarianceCalibrationError(f"unknown recipe: {recipe}")


def _load_train_rows(train_index: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(_sha256(train_index) == TRAIN_INDEX_SHA256, "train index hash differs")
    payload = json.loads(train_index.read_text(encoding="utf-8"))
    _require(payload.get("dataset_binding_sha256") == DATASET_BINDING_SHA256, "dataset binding differs")
    _require(payload.get("split") == "train", "train index split differs")
    rows = [row for row in payload.get("pairs", []) if row.get("model_name") == OBJECT_NAME]
    _require(len(rows) == 147, "hammer train-pair count differs")
    _require(all(row.get("dst_frame_index") - row.get("src_frame_index") in {3, -177} for row in rows), "pair stride differs")
    return rows, payload


def _expected_initial_hash(outcome_root: Path, recipe: str, seed: int) -> tuple[str, dict[str, Any]]:
    path = outcome_root / f"{recipe}__control__seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("cell_id") == f"{recipe}__control__seed{seed}", "outcome cell differs")
    value = payload.get("lineage", {}).get("matched_pairing_evidence", {}).get(
        "initial_model_state_sha256"
    )
    _require(isinstance(value, str) and len(value) == 64, "initial-state receipt is absent")
    return value, _file_record(path)


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
    return torch.cat(pieces)


def _strict_norm(vector: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(vector).item())
    _require(math.isfinite(value) and value > GRADIENT_NORM_FLOOR, "gradient norm is invalid")
    return value


def _cell(
    *,
    recipe: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    frame_paths: Sequence[Path],
    outcome_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    config = _recipe_configuration(recipe)
    _seed_everything(seed)
    model = PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(config["learn_inverse_operator"]),
        heatmap_res=64,
    )
    expected_initial, outcome_record = _expected_initial_hash(outcome_root, recipe, seed)
    observed_initial = _training_state_sha256(model)
    _require(observed_initial == expected_initial, f"{recipe} seed{seed} initial state differs")
    model.train()
    parameter_items = _parameter_items(model)
    parameter_names = [name for name, _ in parameter_items]
    parameters = [parameter for _, parameter in parameter_items]
    vector_size = sum(parameter.numel() for parameter in parameters)
    base_sum = torch.zeros(vector_size, dtype=torch.float64)
    auxiliary_sum = torch.zeros(vector_size, dtype=torch.float64)
    per_transform_sums = {name: 0.0 for name in (
        "translation_plus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
    )}
    base_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    sample_count = 0
    transformed_image_count = 0
    opened_paths: list[str] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        source_paths = [frame_paths[int(row["src_frame_index"])] for row in batch]
        target_paths = [frame_paths[int(row["dst_frame_index"])] for row in batch]
        opened_paths.extend(str(path) for path in source_paths)
        opened_paths.extend(str(path) for path in target_paths)
        x_t = _preprocess(source_paths)
        x_t1 = _preprocess(target_paths)
        action_labels = torch.zeros(len(batch), dtype=torch.long)
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=float(config["lambda_smooth"]),
            lambda_disp=float(config["lambda_disp"]),
            lambda_ent=float(config["lambda_ent"]),
            lambda_act=0.0,
            lambda_loc=0.0,
            lambda_inv=float(config["lambda_inv"]),
            lambda_cycle=float(config["lambda_cycle"]),
            lambda_attach=0.0,
            sigma=0.1,
            num_keypoints=10,
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=30.0,
        )
        paired = add_locked_equivariance_to_pair_loss(
            model.extractor,
            x_t,
            x_t1,
            outputs["heatmaps_t"],
            outputs["heatmaps_t1"],
            losses["base_loss"],
            weight=0.0,
        )
        _require(paired.optimization_loss is losses["base_loss"], "zero arm does not alias base loss")
        base_vector = _gradient_vector(losses["base_loss"], parameters, retain_graph=True)
        auxiliary_vector = _gradient_vector(
            paired.equivariance.loss,
            parameters,
            retain_graph=False,
        )
        count = len(batch)
        base_sum.add_(base_vector, alpha=float(count))
        auxiliary_sum.add_(auxiliary_vector, alpha=float(count))
        base_loss_sum += float(losses["base_loss"].detach()) * count
        auxiliary_loss_sum += float(paired.equivariance.loss.detach()) * count
        for name, value in paired.equivariance.per_transform_loss.items():
            per_transform_sums[name] += float(value.detach()) * count
        sample_count += count
        transformed_image_count += paired.equivariance.transformed_image_count
        _require(all(parameter.grad is None for parameter in model.parameters()), "autograd populated .grad")

    _require(sample_count == len(rows) == 147, "full split was not consumed")
    base_mean = base_sum / sample_count
    auxiliary_mean = auxiliary_sum / sample_count
    base_norm = _strict_norm(base_mean)
    auxiliary_norm = _strict_norm(auxiliary_mean)
    ratio = auxiliary_norm / base_norm
    cosine = float(torch.dot(base_mean, auxiliary_mean) / (base_norm * auxiliary_norm))
    _require(math.isfinite(ratio) and ratio > 0.0, "gradient ratio is invalid")
    _require(math.isfinite(cosine) and -1.000001 <= cosine <= 1.000001, "gradient cosine is invalid")
    return {
        "cell_id": f"{recipe}__r64__seed{seed}",
        "recipe": recipe,
        "seed": seed,
        "recipe_configuration": config,
        "sample_count": sample_count,
        "batch_size": batch_size,
        "batch_count": math.ceil(sample_count / batch_size),
        "transformed_image_count": transformed_image_count,
        "base_loss_full_split_sample_mean": base_loss_sum / sample_count,
        "auxiliary_loss_full_split_sample_mean": auxiliary_loss_sum / sample_count,
        "per_transform_auxiliary_loss_full_split_sample_mean": {
            name: value / sample_count for name, value in per_transform_sums.items()
        },
        "base_gradient_norm": base_norm,
        "auxiliary_gradient_norm": auxiliary_norm,
        "auxiliary_to_base_gradient_ratio": ratio,
        "base_auxiliary_gradient_cosine": cosine,
        "gradient_reduction": "sample_weighted_full_training_split_mean_then_global_float64_l2",
        "parameter_names": parameter_names,
        "parameter_count": vector_size,
        "parameter_set": "all_trainable_extractor.encoder_and_extractor.heatmap_head",
        "historical_initial_state_sha256": expected_initial,
        "reconstructed_initial_state_sha256": observed_initial,
        "model_state_sha256_after_train_mode_forwards": _training_state_sha256(model),
        "batchnorm_state_change_expected": True,
        "historical_outcome_receipt": outcome_record,
        "opened_rgb_count_with_repetition": len(opened_paths),
        "opened_unique_rgb_count": len(set(opened_paths)),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "training_or_weight_update_performed": False,
        "ephemeral_model_discarded_after_cell": True,
    }


def _coefficient_receipt(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(cells) == 6, "calibration requires six cells")
    ratios = [float(cell["auxiliary_to_base_gradient_ratio"]) for cell in cells]
    median_ratio = float(statistics.median(ratios))
    coefficient = min(COEFFICIENT_CAP, TARGET_CONTRIBUTION / median_ratio)
    recipe_summaries = {}
    for recipe in RECIPES:
        local = [float(cell["auxiliary_to_base_gradient_ratio"]) for cell in cells if cell["recipe"] == recipe]
        _require(len(local) == 3, f"{recipe} cell count differs")
        local_median = float(statistics.median(local))
        recipe_summaries[recipe] = {
            "ratios_by_seed": local,
            "median_unscaled_auxiliary_to_base_ratio": local_median,
            "median_scaled_auxiliary_to_base_contribution": coefficient * local_median,
        }
    result = {
        "cells": list(cells),
        "median_unscaled_auxiliary_to_base_ratio_all_six_cells": median_ratio,
        "coefficient": coefficient,
        "scaled_median_contribution_all_six_cells": coefficient * median_ratio,
        "status": "calibrated_at_safety_cap" if coefficient == COEFFICIENT_CAP else "calibrated",
        "coefficient_rule": {
            "formula": "min(0.5, 0.10 / median_six_cell_auxiliary_to_base_gradient_ratio)",
            "target_median_contribution": TARGET_CONTRIBUTION,
            "coefficient_safety_cap": COEFFICIENT_CAP,
            "one_shared_coefficient_across_task55_and_task80": True,
            "raw_lambda_sweep_performed": False,
        },
        "recipe_summaries": recipe_summaries,
        "checkpoint_selector_for_future_training": "minimum_base_validation_loss",
        "interpretation": {
            "measures": "full-split auxiliary versus complete-recipe base extractor-gradient magnitude",
            "does_not_measure": [
                "training outcome",
                "material identity",
                "on-objectness or distinctness",
                "population-level generalisation",
            ],
        },
        "optimizer_steps": 0,
        "training_or_weight_update_performed": False,
    }
    return result


def run(
    *,
    repo_root: Path,
    manifest_path: Path,
    train_index: Path,
    outcome_root: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    _require(not output_path.exists(), "output path already exists")
    _require(batch_size == 4, "locked calibration batch size is four")
    rows, index_payload = _load_train_rows(train_index)
    manifest, manifest_record = _load_manifest(manifest_path)
    frame_paths = _image_paths(manifest)
    _require(len(frame_paths) == 180, "full-orbit frame count differs")
    cells = [
        _cell(
            recipe=recipe,
            seed=seed,
            rows=rows,
            frame_paths=frame_paths,
            outcome_root=outcome_root,
            batch_size=batch_size,
        )
        for recipe in RECIPES
        for seed in SEEDS
    ]
    calibration = _coefficient_receipt(cells)
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "same_frame_equivariance_full_split_gradient_calibration",
        "status": "coefficient_measured_not_training_authorization",
        "calibration": calibration,
        "dataset": {
            "object": OBJECT_NAME,
            "split": "train",
            "pair_count": len(rows),
            "dataset_binding_sha256": index_payload["dataset_binding_sha256"],
            "train_index": _file_record(train_index),
            "full_orbit_manifest": manifest_record,
            "opened_frame_records": [_file_record(path) for path in frame_paths],
        },
        "source_files": {
            relative: _file_record(repo_root / relative)
            for relative in (
                "keypoint_net/run_same_frame_equivariance_calibration.py",
                "keypoint_net/same_frame_equivariance.py",
                "keypoint_net/frozen_nuisance_sensitivity.py",
                "keypoint_net/model.py",
                "keypoint_net/run_frozen_feature_decode_raw.py",
            )
        },
        "runtime": {
            "python_version": __import__("platform").python_version(),
            "pytorch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "authorization_boundary": {
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_or_weight_update_performed": False,
            "gpu_job_submitted": False,
            "training_authorized_by_this_artifact": False,
        },
    }
    document["content_hash_sha256"] = _canonical_sha256(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--outcome-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    _require(args.torch_threads >= 1, "torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run(
        repo_root=args.repo_root.resolve(strict=True),
        manifest_path=args.manifest.resolve(strict=True),
        train_index=args.train_index.resolve(strict=True),
        outcome_root=args.outcome_root.resolve(strict=True),
        output_path=args.output_path.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": result["status"],
        "coefficient": result["calibration"]["coefficient"],
        "scaled_median_contribution": result["calibration"]["scaled_median_contribution_all_six_cells"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
