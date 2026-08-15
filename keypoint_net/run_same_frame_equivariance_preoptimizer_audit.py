"""Audit the exact paired production path before constructing an optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import IndexPairDataset
    from .model import PhaseAModel, compute_losses
    from .same_frame_equivariance import add_locked_equivariance_to_pair_loss
    from .same_frame_equivariance_smoke_authorization import (
        DATASET_BINDING_SHA256,
        TASK_RECIPES,
        TRAIN_INDEX_SHA256,
        canonical_sha256,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import IndexPairDataset  # type: ignore
    from model import PhaseAModel, compute_losses  # type: ignore
    from same_frame_equivariance import add_locked_equivariance_to_pair_loss  # type: ignore
    from same_frame_equivariance_smoke_authorization import (  # type: ignore
        DATASET_BINDING_SHA256,
        TASK_RECIPES,
        TRAIN_INDEX_SHA256,
        canonical_sha256,
    )


SCHEMA_VERSION = "same_frame_equivariance_preoptimizer_audit.v1"
OBJECT_NAME = "engineers_hammer_vray"
RECIPES = ("task55_clean", "task80_assisted")
ARMS = ("control", "self_equivariance")
SEED = 42
COEFFICIENT = 0.5
SAMPLER_SEED_OFFSET = 10_000_019
GRADIENT_NORM_FLOOR = 1e-12
GRADIENT_RECONSTRUCTION_ATOL = 1e-6


class PreoptimizerAuditError(ValueError):
    """Raised when the exact no-update audit cannot be executed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreoptimizerAuditError(message)


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_sha256(
    model: torch.nn.Module, *, names: Iterable[str] | None = None
) -> str:
    allowed = set(names) if names is not None else None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if allowed is not None and name not in allowed:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bn_buffer_names(model: torch.nn.Module) -> list[str]:
    names = [
        name
        for name, _ in model.named_buffers()
        if name.endswith("running_mean")
        or name.endswith("running_var")
        or name.endswith("num_batches_tracked")
    ]
    _require(bool(names), "model has no BatchNorm buffers")
    return sorted(names)


def _pair_ids(batch: Mapping[str, Any]) -> list[str]:
    values = batch.get("pair_id")
    _require(values is not None, "batch lacks pair IDs")
    return [str(value) for value in values]


def _gradient_vector(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
    allow_unused: bool,
) -> tuple[torch.Tensor, list[bool]]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=allow_unused,
    )
    pieces: list[torch.Tensor] = []
    unused: list[bool] = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        unused.append(gradient is None)
        if gradient is None:
            piece = torch.zeros(parameter.numel(), dtype=torch.float64)
        else:
            piece = gradient.detach().cpu().to(dtype=torch.float64).reshape(-1)
            _require(bool(torch.isfinite(piece).all()), "gradient is non-finite")
        pieces.append(piece)
    return torch.cat(pieces), unused


def _norm(vector: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(vector).item())
    _require(math.isfinite(value), "gradient norm is non-finite")
    return value


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = _norm(first) * _norm(second)
    _require(denominator > GRADIENT_NORM_FLOOR, "cosine has a zero gradient")
    return float(torch.dot(first, second).item() / denominator)


def _losses(
    model: PhaseAModel,
    batch: Mapping[str, Any],
    *,
    recipe: str,
    weight: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any]]:
    config = TASK_RECIPES[recipe]
    x_t = batch["x_t"].to(device)
    x_t1 = batch["x_t1"].to(device)
    action_labels = batch["action_label"].to(device).long()
    outputs = model(x_t, x_t1, return_descriptor_features=False)
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
        losses["loss"],
        weight=weight,
    )
    counts = {
        "pair_batch_size": int(x_t.shape[0]),
        "untransformed_auxiliary_image_count": (
            paired.equivariance.untransformed_image_count
        ),
        "transformed_auxiliary_image_count": (
            paired.equivariance.transformed_image_count
        ),
        "total_auxiliary_forward_image_count": (
            paired.equivariance.auxiliary_forward_image_count
        ),
    }
    return losses, paired, counts


def _arm(
    *,
    recipe: str,
    arm: str,
    dataset: IndexPairDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(SEED)
    config = TASK_RECIPES[recipe]
    model = PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(config["learn_inverse_operator"]),
        heatmap_res=64,
    ).to(device)
    model.train()
    initial_state_sha256 = _state_sha256(model)
    bn_names = _bn_buffer_names(model)
    generator = torch.Generator()
    generator.manual_seed(SEED + SAMPLER_SEED_OFFSET)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    diagnostic_batch = next(iter(loader))
    weight = 0.0 if arm == "control" else COEFFICIENT
    with torch.no_grad():
        _, diagnostic_paired, diagnostic_counts = _losses(
            model,
            diagnostic_batch,
            recipe=recipe,
            weight=weight,
            device=device,
        )
    post_diagnostic_state_sha256 = _state_sha256(model)
    post_diagnostic_bn_sha256 = _state_sha256(model, names=bn_names)

    training_batch = next(iter(loader))
    losses, paired, training_counts = _losses(
        model,
        training_batch,
        recipe=recipe,
        weight=weight,
        device=device,
    )
    post_forward_state_sha256 = _state_sha256(model)
    post_forward_bn_sha256 = _state_sha256(model, names=bn_names)

    extractor_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith("extractor.encoder.")
            or name.startswith("extractor.heatmap_head.")
        )
    ]
    operator_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith("operator.")
            or name.startswith("inverse_operator.")
            or name.startswith("action_classifier.")
        )
    ]
    _require(bool(extractor_items), "extractor parameter set is empty")
    _require(bool(operator_items), "operator parameter set is empty")
    extractor_parameters = [parameter for _, parameter in extractor_items]
    operator_parameters = [parameter for _, parameter in operator_items]
    base_vector, base_unused = _gradient_vector(
        losses["loss"],
        extractor_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    auxiliary_vector, auxiliary_unused = _gradient_vector(
        paired.equivariance.loss,
        extractor_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    combined_vector, combined_unused = _gradient_vector(
        paired.optimization_loss,
        extractor_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    operator_auxiliary_vector, operator_auxiliary_unused = _gradient_vector(
        paired.equivariance.loss,
        operator_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    head_weight_gradient = torch.autograd.grad(
        paired.equivariance.loss,
        model.extractor.heatmap_head.weight,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    channel_norms = [
        float(torch.linalg.vector_norm(row.detach().cpu().to(torch.float64)).item())
        for row in head_weight_gradient
    ]
    base_norm = _norm(base_vector)
    auxiliary_norm = _norm(auxiliary_vector)
    combined_norm = _norm(combined_vector)
    operator_auxiliary_norm = _norm(operator_auxiliary_vector)
    zero_weight_aliases_base = (
        paired.optimization_loss is losses["loss"] if arm == "control" else None
    )
    expected_combined = (
        base_vector
        if arm == "control"
        else base_vector + COEFFICIENT * auxiliary_vector
    )
    reconstruction_error = _norm(combined_vector - expected_combined)

    checks = {
        "losses_finite": all(
            math.isfinite(float(value.detach().cpu().item()))
            for value in (
                losses["loss"],
                paired.equivariance.loss,
                paired.optimization_loss,
                *paired.equivariance.per_transform_loss.values(),
            )
        ),
        "all_extractor_gradients_used": not any(
            base_unused + auxiliary_unused + combined_unused
        ),
        "base_extractor_gradient_nonzero": base_norm > GRADIENT_NORM_FLOOR,
        "auxiliary_extractor_gradient_nonzero": (
            auxiliary_norm > GRADIENT_NORM_FLOOR
        ),
        "all_ten_heatmap_output_rows_nonzero": (
            len(channel_norms) == 10
            and all(value > GRADIENT_NORM_FLOOR for value in channel_norms)
        ),
        "operator_auxiliary_gradients_all_unused": all(operator_auxiliary_unused),
        "operator_auxiliary_gradient_exact_zero": operator_auxiliary_norm == 0.0,
        "combined_gradient_reconstructs": (
            reconstruction_error <= GRADIENT_RECONSTRUCTION_ATOL
        ),
        "zero_weight_aliases_exact_base": (
            zero_weight_aliases_base is True if arm == "control" else True
        ),
        "diagnostic_common_batch_count_is_6B": (
            diagnostic_counts["total_auxiliary_forward_image_count"]
            == 6 * diagnostic_counts["pair_batch_size"]
        ),
        "training_common_batch_count_is_6B": (
            training_counts["total_auxiliary_forward_image_count"]
            == 6 * training_counts["pair_batch_size"]
        ),
        "no_parameter_update_during_audit": (
            post_forward_state_sha256 != initial_state_sha256
        ),
    }
    # The full state changes because train-mode BatchNorm buffers update.  A
    # parameter-only check is recorded separately below and must remain exact.
    parameter_names = [name for name, _ in model.named_parameters()]
    parameter_state_sha256 = _state_sha256(model, names=parameter_names)
    _seed_everything(SEED)
    reference = PhaseAModel(
        num_keypoints=10,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        operator_type="shared_affine",
        learn_inverse_operator=bool(config["learn_inverse_operator"]),
        heatmap_res=64,
    ).to(device)
    reference_parameter_sha256 = _state_sha256(reference, names=parameter_names)
    checks["no_parameter_update_during_audit"] = (
        parameter_state_sha256 == reference_parameter_sha256
    )

    return {
        "cell_id": f"{recipe}__{arm}__seed42__preoptimizer",
        "recipe": recipe,
        "arm": arm,
        "seed": SEED,
        "weight": weight,
        "initial_model_state_sha256": initial_state_sha256,
        "diagnostic_pair_ids": _pair_ids(diagnostic_batch),
        "first_training_pair_ids": _pair_ids(training_batch),
        "post_diagnostic_model_state_sha256": post_diagnostic_state_sha256,
        "post_diagnostic_bn_buffer_sha256": post_diagnostic_bn_sha256,
        "post_first_training_forward_model_state_sha256": post_forward_state_sha256,
        "post_first_training_forward_bn_buffer_sha256": post_forward_bn_sha256,
        "parameter_state_sha256_after_gradients": parameter_state_sha256,
        "reference_initial_parameter_state_sha256": reference_parameter_sha256,
        "diagnostic_counts": diagnostic_counts,
        "training_counts": training_counts,
        "losses": {
            "base_optimization_loss": float(losses["loss"].detach().cpu().item()),
            "self_equivariance_loss": float(
                paired.equivariance.loss.detach().cpu().item()
            ),
            "combined_optimization_loss": float(
                paired.optimization_loss.detach().cpu().item()
            ),
            "per_transform": {
                name: float(value.detach().cpu().item())
                for name, value in paired.equivariance.per_transform_loss.items()
            },
            "diagnostic_self_equivariance_loss": float(
                diagnostic_paired.equivariance.loss.detach().cpu().item()
            ),
        },
        "gradients": {
            "extractor_parameter_names": [name for name, _ in extractor_items],
            "operator_parameter_names": [name for name, _ in operator_items],
            "base_extractor_norm": base_norm,
            "auxiliary_extractor_norm": auxiliary_norm,
            "combined_extractor_norm": combined_norm,
            "auxiliary_to_base_norm_ratio": auxiliary_norm / base_norm,
            "base_auxiliary_cosine": _cosine(base_vector, auxiliary_vector),
            "operator_auxiliary_norm": operator_auxiliary_norm,
            "operator_auxiliary_unused": operator_auxiliary_unused,
            "heatmap_head_output_row_norms": channel_norms,
            "combined_gradient_reconstruction_error": reconstruction_error,
            "combined_gradient_reconstruction_absolute_tolerance": (
                GRADIENT_RECONSTRUCTION_ATOL
            ),
            "combined_gradient_reconstruction_relative_error": (
                reconstruction_error / combined_norm
            ),
        },
        "zero_weight_optimization_loss_is_base_object": zero_weight_aliases_base,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    train_index = Path(args.train_index).expanduser().resolve(strict=True)
    _require(_sha256(train_index) == TRAIN_INDEX_SHA256, "train index hash differs")
    _require(1 <= args.batch_size <= 16, "batch size must be within [1, 16]")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    torch.set_num_threads(args.torch_threads)
    dataset = IndexPairDataset(
        str(data_root),
        str(train_index),
        img_size=512,
        center_crop=None,
        include_backward=False,
        object_name=OBJECT_NAME,
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=TRAIN_INDEX_SHA256,
        expected_dataset_binding_sha256=DATASET_BINDING_SHA256,
    )
    _require(len(dataset) == 147, "hammer train pair count differs")

    cells = [
        _arm(
            recipe=recipe,
            arm=arm,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
        )
        for recipe in RECIPES
        for arm in ARMS
    ]
    pair_checks: dict[str, Any] = {}
    for recipe in RECIPES:
        control = next(
            cell for cell in cells if cell["recipe"] == recipe and cell["arm"] == "control"
        )
        candidate = next(
            cell
            for cell in cells
            if cell["recipe"] == recipe and cell["arm"] == "self_equivariance"
        )
        checks = {
            "initial_model_state_identical": (
                control["initial_model_state_sha256"]
                == candidate["initial_model_state_sha256"]
            ),
            "diagnostic_pair_order_identical": (
                control["diagnostic_pair_ids"] == candidate["diagnostic_pair_ids"]
            ),
            "first_training_pair_order_identical": (
                control["first_training_pair_ids"]
                == candidate["first_training_pair_ids"]
            ),
            "post_diagnostic_bn_identical": (
                control["post_diagnostic_bn_buffer_sha256"]
                == candidate["post_diagnostic_bn_buffer_sha256"]
            ),
            "post_training_forward_bn_identical": (
                control["post_first_training_forward_bn_buffer_sha256"]
                == candidate["post_first_training_forward_bn_buffer_sha256"]
            ),
            "base_loss_identical_before_weight": math.isclose(
                control["losses"]["base_optimization_loss"],
                candidate["losses"]["base_optimization_loss"],
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "auxiliary_loss_identical_before_weight": math.isclose(
                control["losses"]["self_equivariance_loss"],
                candidate["losses"]["self_equivariance_loss"],
                rel_tol=0.0,
                abs_tol=0.0,
            ),
        }
        pair_checks[recipe] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "paired_same_frame_equivariance_preoptimizer_production_audit",
        "status": "pass" if (
            all(cell["all_checks_passed"] for cell in cells)
            and all(value["all_checks_passed"] for value in pair_checks.values())
        ) else "fail",
        "semantic_lock": {
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "parameter_updates": 0,
            "both_arms_execute_common_batch_path": True,
            "only_arm_difference": "loss multiplier 0.0 versus 0.5",
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "torch_threads": args.torch_threads,
            "pytorch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
        },
        "source_files": {
            relative: _file_record(repo_root / relative)
            for relative in (
                "keypoint_net/model.py",
                "keypoint_net/train.py",
                "keypoint_net/frozen_nuisance_sensitivity.py",
                "keypoint_net/same_frame_equivariance.py",
                "keypoint_net/run_same_frame_equivariance_preoptimizer_audit.py",
            )
        },
        "dataset": {
            "root": str(data_root),
            "binding_sha256": DATASET_BINDING_SHA256,
            "train_index": _file_record(train_index),
            "object": OBJECT_NAME,
            "pair_count": len(dataset),
        },
        "cells": cells,
        "paired_arm_checks": pair_checks,
    }
    result["content_hash_sha256"] = canonical_sha256(result)
    output_path = Path(args.output_path).expanduser().absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "cells": len(result["cells"]),
        "output_path": str(Path(args.output_path).expanduser().absolute()),
        "content_hash_sha256": result["content_hash_sha256"],
    }, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
