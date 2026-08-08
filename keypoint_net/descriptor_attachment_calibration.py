"""Outcome-blind six-cell gradient calibration for descriptor attachment.

The command in this module performs no optimizer or scheduler step and opens
only the frozen training split.  Each recipe/seed cell gets an ephemeral model.
Base and attachment gradients are taken from the same initialized forward
results with ``torch.autograd.grad`` and aggregated as one sample-weighted mean
gradient vector before a float64 global L2 norm is calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import IndexPairDataset
    from .descriptor_attachment import (
        LOSS_SPEC_SHA256,
        canonical_json_bytes,
        canonical_sha256,
    )
    from .model import PhaseAModel, compute_losses
    from . import representation_fresh_checkpoint_authorization as fresh
except ImportError:  # Historical direct-script execution.
    from dataset import IndexPairDataset  # type: ignore
    from descriptor_attachment import (  # type: ignore
        LOSS_SPEC_SHA256,
        canonical_json_bytes,
        canonical_sha256,
    )
    from model import PhaseAModel, compute_losses  # type: ignore
    import representation_fresh_checkpoint_authorization as fresh  # type: ignore


SCHEMA_VERSION = "descriptor_attachment_weight_receipt.v1"
CALIBRATION_RECIPES = ("task55_clean", "task80_assisted")
CALIBRATION_SEEDS = (42, 43, 44)
CALIBRATION_CELL_IDS = tuple(
    f"{recipe}__r64__seed{seed}"
    for recipe in CALIBRATION_RECIPES
    for seed in CALIBRATION_SEEDS
)
GRADIENT_NORM_FLOOR = 1e-12
GRADIENT_DENOMINATOR_EPSILON = 1e-12
MEDIAN_TARGET = 0.10
MAXIMUM_CAP = 0.50
MINIMUM_MEDIAN_CONTRIBUTION = 0.02
CHECKPOINT_SELECTOR = "minimum_base_validation_loss"


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
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or not branch:
        raise RuntimeError("calibration requires an exact non-detached Git source")
    return commit, branch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parameter_items(model: PhaseAModel) -> list[tuple[str, torch.nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            name.startswith("extractor.encoder.")
            or name.startswith("extractor.heatmap_head.")
        )
    ]
    if not selected:
        raise RuntimeError("calibration parameter set is empty")
    selected_names = {name for name, _ in selected}
    expected_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            name.startswith("extractor.encoder.")
            or name.startswith("extractor.heatmap_head.")
        )
    }
    if selected_names != expected_names:
        raise RuntimeError("calibration parameter selection is incomplete")
    forbidden = (
        "operator.",
        "inverse_operator.",
        "action_classifier.",
    )
    if any(name.startswith(forbidden) for name, _ in selected):
        raise RuntimeError("operator, inverse, or action parameters entered calibration")
    return selected


def gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    """Return a strict, finite float64 CPU gradient vector."""

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    pieces = []
    for index, gradient in enumerate(gradients):
        if gradient is None:
            raise RuntimeError(f"required calibration gradient {index} is missing")
        piece = gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if not torch.isfinite(piece).all():
            raise RuntimeError(f"required calibration gradient {index} is non-finite")
        pieces.append(piece)
    if not pieces:
        raise RuntimeError("calibration produced no gradient components")
    return torch.cat(pieces)


def sample_weighted_mean_gradient(
    gradient_vectors: Iterable[torch.Tensor],
    sample_counts: Iterable[int],
) -> torch.Tensor:
    """Aggregate batch-mean gradients into one all-sample mean vector."""

    vectors = list(gradient_vectors)
    counts = list(sample_counts)
    if not vectors or len(vectors) != len(counts):
        raise ValueError("gradient vectors and sample counts must be non-empty and aligned")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts):
        raise ValueError("all calibration sample counts must be positive integers")
    expected_shape = vectors[0].shape
    if any(vector.dtype != torch.float64 or vector.device.type != "cpu" for vector in vectors):
        raise ValueError("calibration gradient vectors must be float64 CPU tensors")
    if any(vector.shape != expected_shape for vector in vectors):
        raise ValueError("calibration gradient vector shapes differ")
    total = torch.zeros_like(vectors[0], dtype=torch.float64, device="cpu")
    sample_total = 0
    for vector, count in zip(vectors, counts):
        if not torch.isfinite(vector).all():
            raise ValueError("calibration gradient vector is non-finite")
        total.add_(vector, alpha=float(count))
        sample_total += count
    return total / float(sample_total)


def strict_gradient_norm(vector: torch.Tensor) -> float:
    if vector.dtype != torch.float64 or vector.device.type != "cpu":
        raise ValueError("gradient norm requires a float64 CPU vector")
    if not torch.isfinite(vector).all():
        raise ValueError("gradient norm input is non-finite")
    norm = float(torch.linalg.vector_norm(vector, ord=2).item())
    if not math.isfinite(norm) or norm <= GRADIENT_NORM_FLOOR:
        raise ValueError(
            "gradient norm is missing, non-finite, or at/below the frozen floor"
        )
    return norm


def choose_global_lambda(cell_norms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen median-10%, max-50%, median-2% rule exactly."""

    if len(cell_norms) != len(CALIBRATION_CELL_IDS):
        raise ValueError("lambda calibration requires exactly six cells")
    observed_ids = [record.get("cell_id") for record in cell_norms]
    if observed_ids != list(CALIBRATION_CELL_IDS):
        raise ValueError("calibration cell IDs or order differ from the frozen six")
    ratios = []
    normalized_records = []
    for record in cell_norms:
        base_norm = float(record["base_gradient_norm"])
        attach_norm = float(record["attachment_gradient_norm"])
        if (
            not math.isfinite(base_norm)
            or not math.isfinite(attach_norm)
            or base_norm <= GRADIENT_NORM_FLOOR
            or attach_norm <= GRADIENT_NORM_FLOOR
        ):
            raise ValueError("calibration norms must be finite and above the floor")
        ratio = attach_norm / (base_norm + GRADIENT_DENOMINATOR_EPSILON)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("calibration ratio is invalid")
        ratios.append(ratio)
        normalized_records.append({
            **record,
            "gradient_ratio": ratio,
        })
    # ``statistics.median`` averages the two middle values for six cells;
    # torch.median would silently select only the lower middle value.
    median_ratio = float(statistics.median(ratios))
    maximum_ratio = float(max(ratios))
    lambda_attach = min(
        MEDIAN_TARGET / median_ratio,
        MAXIMUM_CAP / maximum_ratio,
    )
    median_scaled = lambda_attach * median_ratio
    maximum_scaled = lambda_attach * maximum_ratio
    status = (
        "authorized"
        if median_scaled >= MINIMUM_MEDIAN_CONTRIBUTION
        else "stop_below_minimum_median_contribution"
    )
    return {
        "status": status,
        "cell_norms": normalized_records,
        "median_gradient_ratio": median_ratio,
        "maximum_gradient_ratio": maximum_ratio,
        "lambda_attach": lambda_attach,
        "scaled_median_contribution": median_scaled,
        "scaled_maximum_contribution": maximum_scaled,
        "caps": {
            "median_target": MEDIAN_TARGET,
            "maximum_cap": MAXIMUM_CAP,
            "minimum_median_contribution": MINIMUM_MEDIAN_CONTRIBUTION,
        },
    }


def _batch_boundaries(sample_ids: Sequence[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        list(sample_ids[start:start + batch_size])
        for start in range(0, len(sample_ids), batch_size)
    ]


def _module_hashes(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "keypoint_net/descriptor_attachment.py",
        "keypoint_net/descriptor_attachment_calibration.py",
        "keypoint_net/model.py",
        "keypoint_net/dataset.py",
    )
    records = {
        relative: _sha256_file(repo_root / relative)
        for relative in relative_paths
    }
    records["aggregate_sha256"] = canonical_sha256(records)
    return records


def _batchnorm_state_hash(model: PhaseAModel) -> str:
    state = {
        name: tensor.detach().cpu().tolist()
        for name, tensor in model.state_dict().items()
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name
    }
    return canonical_sha256(state)


def calibrate_cell(
    *,
    repo_root: Path,
    data_root: Path,
    cell_id: str,
    batch_size: int,
) -> dict[str, Any]:
    cell = fresh.resolve_fresh_cell(repo_root, cell_id)
    if cell.head_package != "64" or cell.seed not in CALIBRATION_SEEDS:
        raise ValueError("calibration cell is outside the retained primary six")
    training = cell.expected_config["training"]
    _seed_everything(cell.seed)
    model = PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=0,
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=training["learn_inverse_operator"],
        heatmap_res=training["heatmap_res"],
    ).cpu()
    model.train()
    parameter_items = _parameter_items(model)
    parameter_names = [name for name, _ in parameter_items]
    parameters = [parameter for _, parameter in parameter_items]

    dataset = IndexPairDataset(
        data_root=str(data_root),
        index_path=cell.train_pairs_path,
        img_size=training["img_size"],
        center_crop=training["center_crop"],
        include_backward=False,
        object_name=cell.expected_config["object"]["name"],
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=cell.expected_config["split"]["train"]["file_sha256"],
        expected_dataset_binding_sha256=cell.expected_config["dataset"]["binding_sha256"],
    )
    sample_ids = [str(sample["pair_id"]) for sample in dataset.samples]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    base_vectors = []
    attachment_vectors = []
    sample_counts = []
    bn_before = _batchnorm_state_hash(model)
    for batch in loader:
        x_t = batch["x_t"].cpu()
        x_t1 = batch["x_t1"].cpu()
        action_labels = batch["action_label"].cpu().long()
        outputs = model(x_t, x_t1, return_descriptor_features=True)
        losses = compute_losses(
            outputs,
            lambda_smooth=training["lambda_smooth"],
            lambda_disp=training["lambda_disp"],
            lambda_ent=training["lambda_ent"],
            lambda_act=training["lambda_act"],
            lambda_loc=training["lambda_loc"],
            lambda_inv=training["lambda_inv"],
            lambda_cycle=training["lambda_cycle"],
            lambda_attach=1.0,
            sigma=training["sigma"],
            num_keypoints=training["num_keypoints"],
            action_labels=action_labels,
            x_t=x_t,
            x_t1=x_t1,
            loc_bg_threshold=training["loc_bg_threshold"],
        )
        base_vectors.append(
            gradient_vector(losses["base_loss"], parameters, retain_graph=True)
        )
        attachment_vectors.append(
            gradient_vector(losses["attachment_loss"], parameters, retain_graph=False)
        )
        sample_counts.append(int(x_t.shape[0]))
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("torch.autograd.grad unexpectedly populated .grad state")
    base_mean = sample_weighted_mean_gradient(base_vectors, sample_counts)
    attachment_mean = sample_weighted_mean_gradient(
        attachment_vectors,
        sample_counts,
    )
    if sum(sample_counts) != len(sample_ids):
        raise RuntimeError("calibration loader did not consume the frozen sample list once")
    return {
        "cell_id": cell_id,
        "recipe": cell.recipe,
        "seed": cell.seed,
        "model_initialization_seed": cell.seed,
        "sample_count": len(sample_ids),
        "base_gradient_norm": strict_gradient_norm(base_mean),
        "attachment_gradient_norm": strict_gradient_norm(attachment_mean),
        "batchnorm_state_before_sha256": bn_before,
        "batchnorm_state_after_sha256": _batchnorm_state_hash(model),
        "ephemeral_model_discarded_after_cell": True,
        "parameter_names": parameter_names,
    }


def run_calibration(
    *,
    repo_root: Path,
    data_root: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration receipt: {output_path}")
    if data_root.name != "_tdw_world_z_roll_base_panel_512_v2":
        raise ValueError("calibration data_root basename differs from the frozen corpus")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_commit, source_branch = _source_state(repo_root)
    reference_cell = fresh.resolve_fresh_cell(repo_root, CALIBRATION_CELL_IDS[0])
    training = reference_cell.expected_config["training"]
    reference_dataset = IndexPairDataset(
        data_root=str(data_root),
        index_path=reference_cell.train_pairs_path,
        img_size=training["img_size"],
        center_crop=training["center_crop"],
        include_backward=False,
        object_name=reference_cell.expected_config["object"]["name"],
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=reference_cell.expected_config["split"]["train"]["file_sha256"],
        expected_dataset_binding_sha256=reference_cell.expected_config["dataset"]["binding_sha256"],
    )
    sample_ids = [str(sample["pair_id"]) for sample in reference_dataset.samples]
    boundaries = _batch_boundaries(sample_ids, batch_size)
    preprocessing = {
        "img_size": training["img_size"],
        "center_crop": training["center_crop"],
        "resize_shape": [training["img_size"], training["img_size"]],
        "to_tensor": True,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "dataset_module_sha256": _sha256_file(repo_root / "keypoint_net/dataset.py"),
        "torch_version": str(torch.__version__),
        "torchvision_version": importlib.metadata.version("torchvision"),
    }
    cell_norms = []
    for cell_id in CALIBRATION_CELL_IDS:
        print(f"[descriptor calibration] {cell_id}", flush=True)
        cell_norms.append(
            calibrate_cell(
                repo_root=repo_root,
                data_root=data_root,
                cell_id=cell_id,
                batch_size=batch_size,
            )
        )
    decision = choose_global_lambda(cell_norms)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_weight_receipt",
        "status": decision["status"],
        "source": {
            "commit": source_commit,
            "branch": source_branch,
            "module_hashes": _module_hashes(repo_root),
        },
        "loss_spec_sha256": LOSS_SPEC_SHA256,
        "checkpoint_selector": CHECKPOINT_SELECTOR,
        "calibration_protocol": {
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "autograd_api": "torch.autograd.grad",
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "model_mode": "train",
            "batchnorm_policy": "one_ephemeral_model_per_cell",
            "parameter_set": (
                "all_trainable_extractor.encoder_and_extractor.heatmap_head"
            ),
            "excluded_parameter_prefixes": [
                "operator",
                "inverse_operator",
                "action_classifier",
            ],
            "batch_size": batch_size,
            "batch_reduction": "sample_weighted_mean_gradient_vector_then_global_l2",
            "norm_arithmetic": "float64",
            "gradient_norm_floor": GRADIENT_NORM_FLOOR,
            "gradient_ratio_denominator_epsilon": GRADIENT_DENOMINATOR_EPSILON,
            "base_and_attachment_share_forward_results": True,
        },
        "dataset": {
            "basename": data_root.name,
            "binding_sha256": reference_cell.expected_config["dataset"]["binding_sha256"],
            "train_index_absolute_path": reference_cell.train_pairs_path,
            "train_index_file_sha256": reference_cell.expected_config["split"]["train"]["file_sha256"],
            "train_index_content_hash_sha256": reference_cell.expected_config["split"]["train"]["content_hash_sha256"],
            "sample_ids": sample_ids,
            "sample_ids_sha256": canonical_sha256(sample_ids),
            "batch_boundaries": boundaries,
            "batch_boundaries_sha256": canonical_sha256(boundaries),
            "preprocessing": preprocessing,
            "preprocessing_sha256": canonical_sha256(preprocessing),
            "validation_or_outcome_data_opened": False,
        },
        **decision,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": str(np.__version__),
            "pytorch_version": str(torch.__version__),
        },
    }
    receipt["content_hash_sha256"] = canonical_sha256(receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate one global descriptor-attachment weight on CPU"
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    # Freeze CPU reduction behavior for reproducible calibration arithmetic.
    torch.set_num_threads(1)
    receipt = run_calibration(
        repo_root=repo_root,
        data_root=args.data_root.expanduser().resolve(strict=True),
        output_path=args.output.expanduser().absolute(),
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": receipt["status"],
        "lambda_attach": receipt["lambda_attach"],
        "content_hash_sha256": receipt["content_hash_sha256"],
        "output": str(args.output.expanduser().absolute()),
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CALIBRATION_CELL_IDS",
    "CALIBRATION_RECIPES",
    "CALIBRATION_SEEDS",
    "CHECKPOINT_SELECTOR",
    "GRADIENT_NORM_FLOOR",
    "SCHEMA_VERSION",
    "choose_global_lambda",
    "gradient_vector",
    "run_calibration",
    "sample_weighted_mean_gradient",
    "strict_gradient_norm",
]
