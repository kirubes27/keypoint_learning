"""Decompose wobble-repair gradient conflict at mature Task80 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    from .model import compute_losses
    from .run_equivariance_gradient_decomposition import (
        RELATIVE_VECTOR_TOLERANCE,
        _groups,
        _relative_reconstruction_error,
        _summary,
        _vector,
    )
    from .run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess
    from .run_frozen_wobble_forensics import (
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from .run_same_frame_equivariance_calibration import (
        _file_record,
        _load_train_rows,
        _parameter_items,
    )
    from .same_frame_equivariance import add_locked_equivariance_to_pair_loss
except ImportError:  # pragma: no cover
    from model import compute_losses  # type: ignore
    from run_equivariance_gradient_decomposition import (  # type: ignore
        RELATIVE_VECTOR_TOLERANCE,
        _groups,
        _relative_reconstruction_error,
        _summary,
        _vector,
    )
    from run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess  # type: ignore
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from run_same_frame_equivariance_calibration import (  # type: ignore
        _file_record,
        _load_train_rows,
        _parameter_items,
    )
    from same_frame_equivariance import add_locked_equivariance_to_pair_loss  # type: ignore


SCHEMA_VERSION = "mature_same_frame_equivariance_gradient_decomposition.v1"
EXPECTED_ROLES = (
    "task80_assisted__control__seed42__best_model",
    "task80_assisted__ocr_zncc__seed42__best_model",
)
TASK80_WEIGHTS = {
    "lambda_smooth": 0.001,
    "lambda_disp": 0.1,
    "lambda_ent": 0.01,
    "lambda_inv": 0.5,
    "lambda_cycle": 0.5,
}


class MatureGradientDecompositionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatureGradientDecompositionError(message)


def _role(
    *,
    manifest: Mapping[str, Any],
    role_key: str,
    rows: Sequence[Mapping[str, Any]],
    frame_paths: Sequence[Path],
    batch_size: int,
) -> dict[str, Any]:
    matches = [item for item in manifest["roles"] if item.get("role_key") == role_key]
    _require(len(matches) == 1, "role key is absent or duplicated")
    role = matches[0]
    _require(role.get("task") == "task80_assisted", "role is not Task80")
    _require(role.get("seed") == 42 and role.get("checkpoint_epoch") == 210, "role seed/epoch differs")
    checkpoint = role["checkpoint"]
    payload, checkpoint_record = _same_fd_checkpoint_load(
        checkpoint["absolute_path"],
        expected_sha256=checkpoint["sha256"],
        expected_size=checkpoint["size_bytes"],
    )
    model, config = _construct_frozen_model(
        payload,
        cell_id=role["cell_id"],
        checkpoint_role=role["checkpoint_role"],
        expected_epoch=role["checkpoint_epoch"],
    )
    loaded_state = _state_sha256(model)
    model.requires_grad_(True)
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
            lambda_smooth=TASK80_WEIGHTS["lambda_smooth"],
            lambda_disp=TASK80_WEIGHTS["lambda_disp"],
            lambda_ent=TASK80_WEIGHTS["lambda_ent"],
            lambda_act=0.0,
            lambda_loc=0.0,
            lambda_inv=TASK80_WEIGHTS["lambda_inv"],
            lambda_cycle=TASK80_WEIGHTS["lambda_cycle"],
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
            "smoothness": TASK80_WEIGHTS["lambda_smooth"] * losses["l_smooth"],
            "dispersion": TASK80_WEIGHTS["lambda_disp"] * losses["l_disp"],
            "entropy": TASK80_WEIGHTS["lambda_ent"] * losses["l_ent"],
            "inverse": TASK80_WEIGHTS["lambda_inv"] * losses["l_inv"],
            "cycle": TASK80_WEIGHTS["lambda_cycle"] * losses["l_cycle"],
        }
        count = len(batch)
        sums["base"].add_(_vector(losses["base_loss"], parameters, retain_graph=True), alpha=count)
        for name in term_names:
            sums[name].add_(_vector(weighted[name], parameters, retain_graph=True), alpha=count)
        aux_parts = list(paired.equivariance.per_transform_loss.values())
        _require(len(aux_parts) == 2, "auxiliary transform count differs")
        aux_sums["translation"].add_(_vector(aux_parts[0], parameters, retain_graph=True), alpha=count)
        aux_sums["rotation"].add_(_vector(aux_parts[1], parameters, retain_graph=True), alpha=count)
        sums["auxiliary"].add_(_vector(paired.equivariance.loss, parameters, retain_graph=False), alpha=count)
        sample_count += count
        _require(all(parameter.grad is None for parameter in model.parameters()), ".grad was populated")

    _require(sample_count == 147, "full split was not consumed")
    means = {name: value / sample_count for name, value in sums.items()}
    aux_means = {name: value / sample_count for name, value in aux_sums.items()}
    base_reconstructed = sum((means[name] for name in term_names), torch.zeros_like(means["base"]))
    aux_reconstructed = 0.5 * (aux_means["translation"] + aux_means["rotation"])
    base_error, base_relative = _relative_reconstruction_error(means["base"], base_reconstructed)
    aux_error, aux_relative = _relative_reconstruction_error(means["auxiliary"], aux_reconstructed)
    _require(base_relative <= RELATIVE_VECTOR_TOLERANCE, "base reconstruction differs")
    _require(aux_relative <= RELATIVE_VECTOR_TOLERANCE, "auxiliary reconstruction differs")
    return {
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "loaded_eval_model_state_sha256": loaded_state,
        "sample_count": sample_count,
        "batch_size": batch_size,
        "base_weights": dict(TASK80_WEIGHTS),
        "ocr_transport_gradient_included": False,
        "vector_reconstruction": {
            "relative_l2_tolerance": RELATIVE_VECTOR_TOLERANCE,
            "base_l2_error": base_error,
            "base_relative_l2_error": base_relative,
            "auxiliary_l2_error": aux_error,
            "auxiliary_relative_l2_error": aux_relative,
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
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    _require(not output_path.exists(), "output path already exists")
    _require(batch_size == 4, "locked batch size is four")
    rows, index_payload = _load_train_rows(train_index)
    manifest, manifest_record = _load_manifest(manifest_path)
    frame_paths = _image_paths(manifest)
    roles = [
        _role(
            manifest=manifest,
            role_key=role_key,
            rows=rows,
            frame_paths=frame_paths,
            batch_size=batch_size,
        )
        for role_key in EXPECTED_ROLES
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "mature_task80_same_frame_equivariance_gradient_conflict_decomposition",
        "status": "diagnostic_not_training_authorization",
        "roles": roles,
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
                "keypoint_net/run_mature_equivariance_gradient_decomposition.py",
                "keypoint_net/run_equivariance_gradient_decomposition.py",
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
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    result = run(
        repo_root=args.repo_root.resolve(strict=True),
        manifest_path=args.manifest.resolve(strict=True),
        train_index=args.train_index.resolve(strict=True),
        output_path=args.output_path.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": result["status"],
        "roles": {
            item["role"]["role_key"]: item["parameter_groups"]["extractor"]["base_auxiliary_cosine"]
            for item in result["roles"]
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
