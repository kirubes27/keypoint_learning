"""One-batch generic-CUDA smoke gate for descriptor attachment.

This is not a scientific result.  It proves that the frozen control and
attachment arms can execute on the assigned CUDA device, start from identical
weights, preserve the zero-arm forward semantics, route attachment gradients
only through the keypoint extractor, and take one finite optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from keypoint_net import descriptor_attachment as attachment
from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net.dataset import IndexPairDataset
from keypoint_net.model import PhaseAModel, compute_losses


SCHEMA_VERSION = "descriptor_attachment_gpu_smoke.v1"
SMOKE_CONTROL_CELL = "task55_clean__control__seed42"
SMOKE_ATTACHMENT_CELL = "task55_clean__attachment__seed42"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state(repo_root: Path, expected_commit: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(commit == expected_commit, "smoke source commit differs")
    _require(branch == authorization.EXPECTED_BRANCH, "smoke source branch differs")
    _require(not tracked, "smoke tracked source is modified")


def _read_json(path: Path, *, expected_sha256: str, name: str) -> Mapping[str, Any]:
    _require(path.is_absolute() and path.is_file() and not path.is_symlink(),
             f"{name} is not an absolute regular file")
    _require(_sha256_file(path) == expected_sha256, f"{name} hash differs")
    value = json.loads(path.read_bytes())
    _require(isinstance(value, Mapping), f"{name} is not an object")
    return value


def _namespace_for_cell(
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
) -> argparse.Namespace:
    cells = manifest.get("cells")
    _require(isinstance(cells, list), "smoke manifest cells are invalid")
    matches = [cell for cell in cells if cell.get("cell_id") == cell_id]
    _require(len(matches) == 1, f"smoke cell {cell_id} is missing or duplicated")
    values = dict(matches[0]["training_arguments"])
    values.update({
        "fresh_cell_id": None,
        "descriptor_cell_id": cell_id,
        "descriptor_experiment_manifest": str(manifest_path),
        "descriptor_experiment_manifest_sha256": manifest_sha256,
        "descriptor_weight_receipt": str(receipt_path),
        "descriptor_weight_receipt_sha256": receipt_sha256,
        "descriptor_loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
    })
    return argparse.Namespace(**values)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model(args: argparse.Namespace, device: torch.device) -> PhaseAModel:
    return PhaseAModel(
        num_keypoints=args.num_keypoints,
        base_channels=args.base_channels,
        temperature=args.temperature,
        num_action_classes=args.num_action_classes,
        padding_mode=args.padding_mode,
        operator_type=args.operator_type,
        learn_inverse_operator=args.learn_inverse_operator,
        heatmap_res=args.heatmap_res,
    ).to(device)


def _losses(
    model: PhaseAModel,
    args: argparse.Namespace,
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_arm: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = model(x_t, x_t1, return_descriptor_features=positive_arm)
    losses = compute_losses(
        outputs,
        lambda_smooth=args.lambda_smooth,
        lambda_disp=args.lambda_disp,
        lambda_ent=args.lambda_ent,
        lambda_act=args.lambda_act,
        lambda_loc=args.lambda_loc,
        lambda_inv=args.lambda_inv,
        lambda_cycle=args.lambda_cycle,
        lambda_attach=args.lambda_attach,
        sigma=args.sigma,
        num_keypoints=args.num_keypoints,
        action_labels=labels,
        x_t=x_t,
        x_t1=x_t1,
        loc_bg_threshold=args.loc_bg_threshold,
    )
    return outputs, losses


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [parameter.grad.detach().double().reshape(-1)
              for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values)).item()) if values else 0.0


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _source_state(repo_root, args.expected_commit)
    manifest_path = args.experiment_manifest.resolve(strict=True)
    receipt_path = args.weight_receipt.resolve(strict=True)
    manifest = _read_json(
        manifest_path, expected_sha256=args.experiment_manifest_sha256,
        name="descriptor experiment manifest",
    )
    _read_json(
        receipt_path, expected_sha256=args.weight_receipt_sha256,
        name="descriptor weight receipt",
    )
    control_args = _namespace_for_cell(
        manifest, cell_id=SMOKE_CONTROL_CELL,
        manifest_path=manifest_path, manifest_sha256=args.experiment_manifest_sha256,
        receipt_path=receipt_path, receipt_sha256=args.weight_receipt_sha256,
    )
    attachment_args = _namespace_for_cell(
        manifest, cell_id=SMOKE_ATTACHMENT_CELL,
        manifest_path=manifest_path, manifest_sha256=args.experiment_manifest_sha256,
        receipt_path=receipt_path, receipt_sha256=args.weight_receipt_sha256,
    )
    authorization.bind_training_namespace(repo_root, control_args)
    authorization.bind_training_namespace(repo_root, attachment_args)

    _require(torch.cuda.is_available(), "generic GPU smoke did not receive CUDA")
    device = torch.device("cuda")
    _seed(int(control_args.seed))
    # The split file hash is frozen in the underlying base-cell contract; use
    # it directly rather than conflating it with the corpus binding hash.
    from keypoint_net import representation_fresh_checkpoint_authorization as fresh
    base_cell = fresh.resolve_fresh_cell(repo_root, "task55_clean__r64__seed42")
    dataset = IndexPairDataset(
        data_root=control_args.data_root,
        index_path=control_args.train_pairs_index,
        img_size=control_args.img_size,
        center_crop=control_args.center_crop,
        include_backward=False,
        object_name=control_args.object,
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=base_cell.expected_config["split"]["train"]["file_sha256"],
        expected_dataset_binding_sha256=control_args.dataset_binding_sha256,
    )
    batch = next(iter(DataLoader(dataset, batch_size=control_args.batch_size,
                                 shuffle=False, num_workers=0)))
    x_t = batch["x_t"].to(device)
    x_t1 = batch["x_t1"].to(device)
    labels = batch["action_label"].to(device).long()
    pair_ids = [str(value) for value in batch["pair_id"]]

    _seed(int(control_args.seed))
    control = _model(control_args, device).train()
    attachment_model = _model(attachment_args, device).train()
    attachment_model.load_state_dict(control.state_dict(), strict=True)
    initial_state_equal = all(
        torch.equal(value, attachment_model.state_dict()[name])
        for name, value in control.state_dict().items()
    )
    control_outputs, control_losses = _losses(
        control, control_args, x_t, x_t1, labels, positive_arm=False
    )
    attachment_outputs, attachment_losses = _losses(
        attachment_model, attachment_args, x_t, x_t1, labels, positive_arm=True
    )
    shared_keys = ("p_t", "p_t1", "p_hat_t1", "heatmaps_t", "heatmaps_t1")
    zero_arm_forward_equal = all(
        torch.equal(control_outputs[key], attachment_outputs[key]) for key in shared_keys
    )
    base_loss_equal = torch.equal(
        control_losses["base_loss"], attachment_losses["base_loss"]
    )
    _require(initial_state_equal, "smoke models did not start identically")
    _require(zero_arm_forward_equal and base_loss_equal,
             "descriptor feature return changed the base forward")
    _require(tuple(attachment_outputs["descriptor_features_t"].shape[1:]) == (128, 64, 64),
             "descriptor feature shape differs")

    attachment_model.zero_grad(set_to_none=True)
    attachment_losses["attachment_loss"].backward()
    encoder_parameters = [parameter for name, parameter in attachment_model.named_parameters()
                          if name.startswith("extractor.encoder.")]
    head_parameters = [parameter for name, parameter in attachment_model.named_parameters()
                       if name.startswith("extractor.heatmap_head.")]
    forbidden_parameters = [parameter for name, parameter in attachment_model.named_parameters()
                            if name.startswith(("operator.", "inverse_operator.",
                                                "action_classifier."))]
    encoder_grad_norm = _grad_norm(encoder_parameters)
    head_grad_norm = _grad_norm(head_parameters)
    forbidden_grad_norm = _grad_norm(forbidden_parameters)
    _require(encoder_grad_norm > 0.0 and head_grad_norm > 0.0,
             "attachment gradient did not reach extractor encoder/head")
    _require(forbidden_grad_norm == 0.0,
             "attachment gradient reached a forbidden operator/action parameter")

    # Recreate both arms from the same initial state before the real optimizer
    # smoke so the earlier gradient probe cannot influence the step.
    initial_state = {name: value.detach().clone() for name, value in control.state_dict().items()}
    stepped = {}
    for arm_name, arm_args, positive in (
        ("control", control_args, False),
        ("attachment", attachment_args, True),
    ):
        model = _model(arm_args, device).train()
        model.load_state_dict(initial_state, strict=True)
        optimizer = Adam(model.parameters(), lr=arm_args.lr,
                         weight_decay=arm_args.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        _, losses = _losses(model, arm_args, x_t, x_t1, labels, positive_arm=positive)
        losses["train_loss"].backward()
        optimizer.step()
        finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
        _require(finite and math.isfinite(float(losses["train_loss"].detach().cpu())),
                 f"{arm_name} one-step smoke is non-finite")
        stepped[arm_name] = {
            "base_loss": float(losses["base_loss"].detach().cpu()),
            "attachment_loss": float(losses["attachment_loss"].detach().cpu()),
            "train_loss": float(losses["train_loss"].detach().cpu()),
            "optimizer_steps": 1,
            "model_state_finite_after_step": finite,
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_gpu_smoke_receipt",
        "status": "pass",
        "scientific_result_authorized": False,
        "source_commit": args.expected_commit,
        "source_branch": authorization.EXPECTED_BRANCH,
        "loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
        "experiment_manifest": {
            "absolute_path": str(manifest_path),
            "file_sha256": args.experiment_manifest_sha256,
        },
        "weight_receipt": {
            "absolute_path": str(receipt_path),
            "file_sha256": args.weight_receipt_sha256,
        },
        "cuda": {
            "available": True,
            "device_name": torch.cuda.get_device_name(device),
            "device_count_visible": torch.cuda.device_count(),
            "pytorch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "batch": {"pair_ids": pair_ids, "sample_count": len(pair_ids)},
        "checks": {
            "identical_initial_state": initial_state_equal,
            "zero_arm_forward_equal": zero_arm_forward_equal,
            "base_loss_equal": base_loss_equal,
            "descriptor_feature_shape": list(
                attachment_outputs["descriptor_features_t"].shape
            ),
            "attachment_encoder_gradient_norm": encoder_grad_norm,
            "attachment_head_gradient_norm": head_grad_norm,
            "forbidden_gradient_norm": forbidden_grad_norm,
        },
        "one_step": stepped,
    }
    result["content_hash_sha256"] = attachment.canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--experiment-manifest-sha256", required=True)
    parser.add_argument("--weight-receipt", type=Path, required=True)
    parser.add_argument("--weight-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GPU smoke receipt: {output}")
    result = run_smoke(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o444)
    try:
        data = attachment.canonical_json_bytes(result) + b"\n"
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)
    print(json.dumps({"output": str(output), "status": result["status"],
                      "file_sha256": _sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
