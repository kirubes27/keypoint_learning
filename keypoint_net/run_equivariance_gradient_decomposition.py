"""Decompose the full-split base/auxiliary gradient conflict without training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    from .model import PhaseAModel, compute_losses
    from .run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess
    from .run_same_frame_equivariance_calibration import (
        RECIPES,
        SEEDS,
        _expected_initial_hash,
        _file_record,
        _load_train_rows,
        _parameter_items,
        _recipe_configuration,
        _seed_everything,
        _training_state_sha256,
    )
    from .same_frame_equivariance import add_locked_equivariance_to_pair_loss
except ImportError:  # pragma: no cover
    from model import PhaseAModel, compute_losses  # type: ignore
    from run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess  # type: ignore
    from run_same_frame_equivariance_calibration import (  # type: ignore
        RECIPES,
        SEEDS,
        _expected_initial_hash,
        _file_record,
        _load_train_rows,
        _parameter_items,
        _recipe_configuration,
        _seed_everything,
        _training_state_sha256,
    )
    from same_frame_equivariance import add_locked_equivariance_to_pair_loss  # type: ignore


SCHEMA_VERSION = "same_frame_equivariance_gradient_decomposition.v1"
RELATIVE_VECTOR_TOLERANCE = 5e-4
NORM_FLOOR = 1e-12


class GradientDecompositionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GradientDecompositionError(message)


def _vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    result = torch.cat([g.detach().cpu().to(torch.float64).reshape(-1) for g in gradients])
    _require(bool(torch.isfinite(result).all()), "gradient contains non-finite values")
    return result


def _groups(items: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, slice]:
    offset = 0
    spans: dict[str, list[int]] = {"extractor": [0, 0], "encoder": [0, 0], "heatmap_head": [0, 0]}
    for name, parameter in items:
        start, stop = offset, offset + parameter.numel()
        offset = stop
        spans["extractor"][1] = stop
        group = "encoder" if name.startswith("extractor.encoder.") else "heatmap_head"
        if spans[group][1] == 0:
            spans[group][0] = start
        spans[group][1] = stop
    _require(offset > 0 and spans["encoder"][1] > 0 and spans["heatmap_head"][1] > 0, "parameter groups differ")
    return {name: slice(values[0], values[1]) for name, values in spans.items()}


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first_norm = float(torch.linalg.vector_norm(first))
    second_norm = float(torch.linalg.vector_norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return None
    return float(torch.dot(first, second) / (first_norm * second_norm))


def _relative_reconstruction_error(observed: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float]:
    absolute = float(torch.linalg.vector_norm(reconstructed - observed))
    denominator = max(float(torch.linalg.vector_norm(observed)), NORM_FLOOR)
    return absolute, absolute / denominator


def _summary(
    base: torch.Tensor,
    auxiliary: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    aux_components: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    base_norm = float(torch.linalg.vector_norm(base))
    aux_norm = float(torch.linalg.vector_norm(auxiliary))
    total_dot = float(torch.dot(base, auxiliary))
    _require(base_norm > 0.0 and aux_norm > 0.0, "total gradient norm is zero")
    component_rows = {}
    for name, value in components.items():
        dot = float(torch.dot(value, auxiliary))
        component_rows[name] = {
            "gradient_norm": float(torch.linalg.vector_norm(value)),
            "cosine_with_auxiliary": _cosine(value, auxiliary),
            "dot_with_auxiliary": dot,
            "fraction_of_total_base_auxiliary_dot": dot / total_dot if total_dot != 0.0 else None,
        }
    aux_rows = {
        name: {
            "gradient_norm": float(torch.linalg.vector_norm(value)),
            "cosine_with_base": _cosine(value, base),
            "dot_with_base": float(torch.dot(value, base)),
        }
        for name, value in aux_components.items()
    }
    return {
        "base_gradient_norm": base_norm,
        "auxiliary_gradient_norm": aux_norm,
        "base_auxiliary_cosine": _cosine(base, auxiliary),
        "base_auxiliary_dot": total_dot,
        "base_components": component_rows,
        "auxiliary_components": aux_rows,
    }


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
    items = _parameter_items(model)
    parameters = [parameter for _, parameter in items]
    groups = _groups(items)
    vector_size = sum(parameter.numel() for parameter in parameters)
    term_names = ("prediction", "smoothness", "dispersion", "entropy", "inverse", "cycle")
    sums = {name: torch.zeros(vector_size, dtype=torch.float64) for name in ("base", "auxiliary", *term_names)}
    aux_sums = {
        "translation": torch.zeros(vector_size, dtype=torch.float64),
        "rotation": torch.zeros(vector_size, dtype=torch.float64),
    }
    sample_count = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        x_t = _preprocess([frame_paths[int(row["src_frame_index"])] for row in batch])
        x_t1 = _preprocess([frame_paths[int(row["dst_frame_index"])] for row in batch])
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
            action_labels=torch.zeros(len(batch), dtype=torch.long),
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
        weighted = {
            "prediction": losses["l_pred"],
            "smoothness": float(config["lambda_smooth"]) * losses["l_smooth"],
            "dispersion": float(config["lambda_disp"]) * losses["l_disp"],
            "entropy": float(config["lambda_ent"]) * losses["l_ent"],
            "inverse": float(config["lambda_inv"]) * losses["l_inv"],
            "cycle": float(config["lambda_cycle"]) * losses["l_cycle"],
        }
        count = len(batch)
        sums["base"].add_(_vector(losses["base_loss"], parameters, retain_graph=True), alpha=count)
        for name in term_names:
            sums[name].add_(_vector(weighted[name], parameters, retain_graph=True), alpha=count)
        component_values = list(paired.equivariance.per_transform_loss.values())
        _require(len(component_values) == 2, "auxiliary transform count differs")
        aux_sums["translation"].add_(_vector(component_values[0], parameters, retain_graph=True), alpha=count)
        aux_sums["rotation"].add_(_vector(component_values[1], parameters, retain_graph=True), alpha=count)
        sums["auxiliary"].add_(_vector(paired.equivariance.loss, parameters, retain_graph=False), alpha=count)
        sample_count += count
        _require(all(parameter.grad is None for parameter in model.parameters()), ".grad was populated")

    _require(sample_count == 147, "full split was not consumed")
    means = {name: value / sample_count for name, value in sums.items()}
    aux_means = {name: value / sample_count for name, value in aux_sums.items()}
    base_reconstructed = sum((means[name] for name in term_names), torch.zeros_like(means["base"]))
    aux_reconstructed = 0.5 * (aux_means["translation"] + aux_means["rotation"])
    base_error, base_relative_error = _relative_reconstruction_error(means["base"], base_reconstructed)
    aux_error, aux_relative_error = _relative_reconstruction_error(means["auxiliary"], aux_reconstructed)
    _require(
        base_relative_error <= RELATIVE_VECTOR_TOLERANCE,
        f"base vector reconstruction failed: absolute={base_error}, relative={base_relative_error}",
    )
    _require(
        aux_relative_error <= RELATIVE_VECTOR_TOLERANCE,
        f"auxiliary vector reconstruction failed: absolute={aux_error}, relative={aux_relative_error}",
    )

    return {
        "cell_id": f"{recipe}__seed{seed}",
        "recipe": recipe,
        "seed": seed,
        "sample_count": sample_count,
        "batch_size": batch_size,
        "historical_initial_state_sha256": expected_initial,
        "reconstructed_initial_state_sha256": observed_initial,
        "historical_outcome_receipt": outcome_record,
        "vector_reconstruction": {
            "relative_l2_tolerance": RELATIVE_VECTOR_TOLERANCE,
            "base_l2_error": base_error,
            "base_relative_l2_error": base_relative_error,
            "auxiliary_l2_error": aux_error,
            "auxiliary_relative_l2_error": aux_relative_error,
        },
        "parameter_groups": {
            group: _summary(
                means["base"][span],
                means["auxiliary"][span],
                {name: means[name][span] for name in term_names},
                {name: value[span] for name, value in aux_means.items()},
            )
            for group, span in groups.items()
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_or_weight_update_performed": False,
    }


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
    _require(batch_size == 4, "locked batch size is four")
    rows, index_payload = _load_train_rows(train_index)
    manifest, manifest_record = _load_manifest(manifest_path)
    frame_paths = _image_paths(manifest)
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
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "same_frame_equivariance_full_split_gradient_conflict_decomposition",
        "status": "diagnostic_not_training_authorization",
        "cells": cells,
        "dataset": {
            "object": "engineers_hammer_vray",
            "pair_count": len(rows),
            "dataset_binding_sha256": index_payload["dataset_binding_sha256"],
            "train_index": _file_record(train_index),
            "manifest": manifest_record,
        },
        "source_files": {
            relative: _file_record(repo_root / relative)
            for relative in (
                "keypoint_net/run_equivariance_gradient_decomposition.py",
                "keypoint_net/run_same_frame_equivariance_calibration.py",
                "keypoint_net/same_frame_equivariance.py",
                "keypoint_net/model.py",
            )
        },
        "runtime": {
            "python_version": __import__("platform").python_version(),
            "pytorch_version": str(torch.__version__),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "authorization_boundary": {
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_or_weight_update_performed": False,
            "gpu_job_submitted": False,
        },
    }
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
        "cell_count": len(result["cells"]),
        "extractor_cosines": {
            cell["cell_id"]: cell["parameter_groups"]["extractor"]["base_auxiliary_cosine"]
            for cell in result["cells"]
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
