"""CUDA execution smoke for the paired OCR-ZNCC experiment.

This is not scientific evidence.  It first checks identical initial
weights/base forwards, OCR gradient routing, boundary validity, and one finite
manual Adam step.  It then executes both dedicated one-epoch ``__smoke`` cells
through the exact production ``train.py`` entrypoint and validates their
completed-run receipts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import run_ocr_zncc_paired_training as paired_training
from keypoint_net.dataset import IndexPairDataset
from keypoint_net.model import PhaseAModel, compute_losses
from keypoint_net.ocr_zncc_transport import (
    OCRZNCCConfig,
    combine_with_base_loss,
    normalized_images_to_retained_rgb,
    ocr_zncc_transport_loss,
)


SCHEMA_VERSION = "ocr_zncc_gpu_smoke.v2"
TRAIN_INDEX_SHA256 = "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
SMOKE_RECIPES = ("task55_clean", "task80_assisted")
CUDA_MEMORY_POLL_INTERVAL_SECONDS = 0.05


def _smoke_cells(recipe: str) -> tuple[str, str]:
    _require(recipe in SMOKE_RECIPES, "GPU smoke recipe is not authorized")
    return (
        f"{recipe}__control__seed42__smoke",
        f"{recipe}__ocr_zncc__seed42__smoke",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, name: str) -> dict[str, Any]:
    expanded = path.expanduser()
    _require(expanded.is_absolute() and not expanded.is_symlink(),
             f"{name} is not an absolute regular file")
    absolute = expanded.resolve(strict=True)
    _require(absolute.is_file(), f"{name} is not a regular file")
    return {
        "absolute_path": str(absolute),
        "sha256": _sha256(absolute),
        "size_bytes": absolute.stat().st_size,
    }


def _load_bound_json(
    path: Path, *, expected_sha256: str, name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, name=name)
    _require(record["sha256"] == expected_sha256, f"{name} hash differs")
    try:
        value = json.loads(Path(record["absolute_path"]).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{name} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{name} is not an object")
    return value, record


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> str:
    claimed = value.get("content_hash_sha256")
    _require(isinstance(claimed, str), f"{name} content hash is missing")
    payload = dict(value)
    payload.pop("content_hash_sha256", None)
    _require(authorization.canonical_sha256(payload) == claimed,
             f"{name} content hash differs")
    return claimed


def _load_smoke_launch_lock(
    *, source_commit: str, source_branch: str, manifest_path: Path,
    manifest_sha256: str, manifest_content_sha256: str, recipe: str,
    smoke_cells: tuple[str, str],
) -> dict[str, Any]:
    lock_path_value = os.environ.get("OCR_STAGE_LAUNCH_LOCK")
    lock_sha256 = os.environ.get("OCR_STAGE_LAUNCH_LOCK_SHA256")
    _require(bool(lock_path_value) and bool(lock_sha256),
             "smoke launch-lock environment binding is missing")
    lock, record = _load_bound_json(
        Path(str(lock_path_value)), expected_sha256=str(lock_sha256),
        name="OCR GPU smoke launch lock",
    )
    content_hash = _validate_content_hash(lock, name="OCR GPU smoke launch lock")
    _require(lock.get("schema_version") == "ocr_zncc_gpu_smoke_launch_lock.v1"
             and lock.get("artifact_type") == "ocr_zncc_gpu_smoke_launch_lock",
             "OCR GPU smoke launch-lock schema differs")
    _require(lock.get("source_commit") == source_commit
             and lock.get("source_branch") == source_branch,
             "OCR GPU smoke launch-lock source differs")
    _require(lock.get("cells") == sorted(smoke_cells)
             and lock.get("scientific_result_authorized") is False,
             "OCR GPU smoke launch-lock semantics differ")
    manifest_record = lock.get("manifest")
    _require(isinstance(manifest_record, dict)
             and manifest_record.get("absolute_path") == str(manifest_path)
             and manifest_record.get("sha256") == manifest_sha256
             and manifest_record.get("content_hash_sha256")
             == manifest_content_sha256,
             "OCR GPU smoke launch-lock manifest differs")
    _require(lock.get("recipe") == recipe,
             "OCR GPU smoke launch-lock recipe differs")
    expected_runtime = {
        "environment_lock": (
            os.environ.get("OCR_ENV_LOCK"),
            os.environ.get("OCR_ENV_LOCK_SHA256"),
        ),
        "slurm_script": (
            os.environ.get("OCR_SLURM_SCRIPT"),
            os.environ.get("OCR_SLURM_SCRIPT_SHA256"),
        ),
    }
    for field, (live_path, live_hash) in expected_runtime.items():
        value = lock.get(field)
        _require(isinstance(value, dict) and bool(live_path) and bool(live_hash),
                 f"OCR GPU smoke {field} binding is missing")
        live_record = _file_record(Path(str(live_path)), name=field)
        _require(live_record["sha256"] == live_hash and value == live_record,
                 f"OCR GPU smoke {field} differs from launch lock")
    return {
        **record,
        "content_hash_sha256": content_hash,
        "schema_version": lock["schema_version"],
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _namespace(
    manifest: Mapping[str, Any], *, cell_id: str, path: Path, sha256: str,
) -> argparse.Namespace:
    matches = [cell for cell in manifest["cells"] if cell["cell_id"] == cell_id]
    _require(len(matches) == 1, f"smoke cell {cell_id} is absent or duplicated")
    values = dict(matches[0]["training_arguments"])
    values.update({
        "ocr_cell_id": cell_id,
        "ocr_experiment_manifest": str(path),
        "ocr_experiment_manifest_sha256": sha256,
    })
    return argparse.Namespace(**values)


def _model(args: argparse.Namespace, device: torch.device) -> PhaseAModel:
    return PhaseAModel(
        num_keypoints=args.num_keypoints,
        base_channels=args.base_channels,
        temperature=args.temperature,
        num_action_classes=0,
        padding_mode=args.padding_mode,
        operator_type=args.operator_type,
        learn_inverse_operator=args.learn_inverse_operator,
        heatmap_res=args.heatmap_res,
    ).to(device)


def _base(
    model: PhaseAModel, args: argparse.Namespace, x_t: torch.Tensor,
    x_t1: torch.Tensor, labels: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = model(x_t, x_t1)
    losses = compute_losses(
        outputs,
        lambda_smooth=args.lambda_smooth,
        lambda_disp=args.lambda_disp,
        lambda_ent=args.lambda_ent,
        lambda_act=args.lambda_act,
        lambda_loc=args.lambda_loc,
        lambda_inv=args.lambda_inv,
        lambda_cycle=args.lambda_cycle,
        lambda_attach=0.0,
        sigma=args.sigma,
        num_keypoints=args.num_keypoints,
        action_labels=labels,
        x_t=x_t,
        x_t1=x_t1,
        loc_bg_threshold=args.loc_bg_threshold,
    )
    return outputs, losses


def _transport(
    outputs: Mapping[str, torch.Tensor], x_t: torch.Tensor, x_t1: torch.Tensor,
    config: OCRZNCCConfig,
):
    batch = x_t.shape[0]
    return ocr_zncc_transport_loss(
        normalized_images_to_retained_rgb(x_t),
        normalized_images_to_retained_rgb(x_t1),
        outputs["p_t"].reshape(batch, 10, 2),
        outputs["p_hat_t1"].reshape(batch, 10, 2),
        outputs["p_t1"].reshape(batch, 10, 2),
        config=config,
    )


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    pieces = [parameter.grad.detach().double().reshape(-1)
              for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(pieces)).item()) if pieces else 0.0


def _manual_cuda_checks(
    *, control_args: argparse.Namespace, positive_args: argparse.Namespace,
    batch: Mapping[str, Any], device: torch.device,
) -> dict[str, Any]:
    """Run meaning-first CUDA checks without producing scientific results."""
    torch.cuda.reset_peak_memory_stats(device)
    x_t = batch["x_t"].to(device)
    x_t1 = batch["x_t1"].to(device)
    labels = batch["action_label"].to(device).long()
    config = OCRZNCCConfig(peak_margin_exclusion_radius_cells=1)

    _seed(42)
    control = _model(control_args, device).train()
    positive = _model(positive_args, device).train()
    positive.load_state_dict(control.state_dict(), strict=True)
    initial_state_equal = all(
        torch.equal(value, positive.state_dict()[name])
        for name, value in control.state_dict().items()
    )
    _require(initial_state_equal, "smoke arms do not have identical initial state")
    initial = {
        name: value.detach().clone()
        for name, value in control.state_dict().items()
    }
    control_outputs, control_losses = _base(
        control, control_args, x_t, x_t1, labels,
    )
    positive_outputs, positive_losses = _base(
        positive, positive_args, x_t, x_t1, labels,
    )
    _require(torch.equal(control_losses["base_loss"], positive_losses["base_loss"]),
             "identical arms have different base loss")
    for key in ("p_t", "p_t1", "p_hat_t1", "heatmaps_t", "heatmaps_t1"):
        _require(torch.equal(control_outputs[key], positive_outputs[key]),
                 f"identical arms differ at {key}")
    transport = _transport(positive_outputs, x_t, x_t1, config)
    _require(transport.accepted_count > 0, "smoke batch has no accepted match")
    _require(bool(transport.source_patch_inside[transport.accepted].all()),
             "smoke accepted a boundary-invalid source patch")

    positive.zero_grad(set_to_none=True)
    transport.loss.backward()
    extractor = [
        parameter for name, parameter in positive.named_parameters()
        if name.startswith(("extractor.encoder.", "extractor.heatmap_head."))
    ]
    forbidden = [
        parameter for name, parameter in positive.named_parameters()
        if name.startswith(("operator.", "inverse_operator.",
                            "action_classifier."))
    ]
    extractor_norm = _grad_norm(extractor)
    forbidden_norm = _grad_norm(forbidden)
    _require(extractor_norm > 0.0, "OCR gradient missed extractor")
    _require(forbidden_norm == 0.0, "OCR gradient directly reached operator")

    stepped = {}
    for arm, arm_args in (("control", control_args), ("ocr_zncc", positive_args)):
        model = _model(arm_args, device).train()
        model.load_state_dict(initial, strict=True)
        optimizer = Adam(
            model.parameters(), lr=arm_args.lr,
            weight_decay=arm_args.weight_decay,
        )
        outputs, losses = _base(model, arm_args, x_t, x_t1, labels)
        if arm == "ocr_zncc":
            auxiliary = _transport(outputs, x_t, x_t1, config)
            optimization_loss = combine_with_base_loss(
                losses["loss"], auxiliary.loss,
                weight=arm_args.lambda_ocr_zncc,
            )
            auxiliary_value = float(auxiliary.loss.detach().cpu())
            accepted = auxiliary.accepted_count
        else:
            optimization_loss = losses["loss"]
            auxiliary_value = 0.0
            accepted = None
        optimizer.zero_grad(set_to_none=True)
        optimization_loss.backward()
        optimizer.step()
        finite = all(
            bool(torch.isfinite(value).all())
            for value in model.state_dict().values()
        )
        _require(finite and math.isfinite(float(optimization_loss.detach().cpu())),
                 f"{arm} one-step result is non-finite")
        stepped[arm] = {
            "base_loss": float(losses["base_loss"].detach().cpu()),
            "ocr_zncc_loss": auxiliary_value,
            "optimization_loss": float(optimization_loss.detach().cpu()),
            "accepted_match_count": accepted,
            "optimizer_steps": 1,
            "model_state_finite": finite,
        }

    return {
        "batch": {
            "sample_count": len(batch["pair_id"]),
            "pair_ids": [str(value) for value in batch["pair_id"]],
        },
        "checks": {
            "identical_initial_state": initial_state_equal,
            "identical_base_forward": True,
            "accepted_match_count": transport.accepted_count,
            "accepted_source_patches_inside": True,
            "auxiliary_extractor_gradient_norm": extractor_norm,
            "auxiliary_forbidden_gradient_norm": forbidden_norm,
        },
        "one_step": stepped,
        "cuda_memory": {
            "measurement_scope": "manual_equality_gradient_and_one_step_checks",
            "allocator": "torch.cuda",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def _pid_cuda_memory_mib(pid: int) -> float | None:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 2:
            continue
        try:
            live_pid = int(fields[0])
            memory_mib = float(fields[1])
        except ValueError:
            continue
        if live_pid == pid:
            return memory_mib
    return None


def _run_production_command(
    command: list[str], *, environment: Mapping[str, str]
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        start_new_session=True,
        env=dict(environment),
    )
    observed_peak_mib: float | None = None
    observed_samples = 0
    poll_count = 0
    try:
        while True:
            memory_mib = _pid_cuda_memory_mib(process.pid)
            poll_count += 1
            if memory_mib is not None:
                observed_samples += 1
                observed_peak_mib = (
                    memory_mib if observed_peak_mib is None
                    else max(observed_peak_mib, memory_mib)
                )
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(CUDA_MEMORY_POLL_INTERVAL_SECONDS)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    elapsed = time.monotonic() - started
    _require(returncode == 0,
             f"production smoke command exited with status {returncode}")
    _require(observed_peak_mib is not None,
             "production smoke CUDA memory could not be observed")
    return {
        "elapsed_seconds": elapsed,
        "cuda_memory": {
            "measurement": "sampled_process_device_memory_via_nvidia_smi",
            "sleep_between_polls_seconds": CUDA_MEMORY_POLL_INTERVAL_SECONDS,
            "poll_count": poll_count,
            "samples_with_process": observed_samples,
            "observed_peak_mib": observed_peak_mib,
            "note": "Observed sampled device memory, not an exact allocator maximum.",
        },
    }


def _cell(manifest: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    matches = [
        cell for cell in manifest["cells"]
        if cell.get("cell_id") == cell_id
    ]
    _require(len(matches) == 1,
             f"smoke cell {cell_id} is absent or duplicated")
    return matches[0]


def _validate_completion_receipt(
    *, cell: Mapping[str, Any], command: list[str], source_commit: str,
    source_branch: str, manifest_path: Path, manifest_sha256: str,
    smoke_launch_lock: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    run_directory = (
        Path(str(cell["training_arguments"]["output_dir"])) / cell_id
    ).resolve(strict=True)
    receipt_path = run_directory / authorization.COMPLETED_RUN_RECEIPT_NAME
    receipt, record = _load_bound_json(
        receipt_path, expected_sha256=_sha256(receipt_path),
        name=f"{cell_id} completed-run receipt",
    )
    content_hash = _validate_content_hash(
        receipt, name=f"{cell_id} completed-run receipt",
    )
    _require(receipt.get("schema_version")
             == authorization.COMPLETED_RUN_SCHEMA_VERSION,
             f"{cell_id} completion schema differs")
    _require(receipt.get("cell_id") == cell_id
             and receipt.get("recipe") == cell.get("recipe")
             and receipt.get("arm") == cell.get("arm"),
             f"{cell_id} completion identity differs")
    _require(receipt.get("source_commit") == source_commit
             and receipt.get("source_branch") == source_branch,
             f"{cell_id} completion source differs")
    _require(receipt.get("manifest", {}).get("absolute_path")
             == str(manifest_path)
             and receipt.get("manifest", {}).get("file_sha256")
             == manifest_sha256,
             f"{cell_id} completion manifest differs")
    _require(receipt.get("expected_training_arguments")
             == cell.get("training_arguments"),
             f"{cell_id} completion training arguments differ")
    _require(receipt.get("run_directory") == str(run_directory),
             f"{cell_id} completion run directory differs")
    execution = receipt.get("execution")
    _require(isinstance(execution, dict)
             and execution.get("optimizer_step_count") == 10
             and execution.get("scheduler_step_count") == 1
             and execution.get("full_command") == command,
             f"{cell_id} completion execution differs")
    runtime = execution.get("runtime_environment")
    _require(isinstance(runtime, dict),
             f"{cell_id} runtime environment is absent")
    expected_runtime = {
        "ocr_stage_launch_lock": {
            key: smoke_launch_lock[key]
            for key in ("absolute_path", "sha256", "size_bytes")
        },
        "environment_lock": _file_record(
            Path(str(os.environ["OCR_ENV_LOCK"])), name="environment lock",
        ),
        "slurm_job_script": _file_record(
            Path(str(os.environ["OCR_SLURM_SCRIPT"])), name="Slurm script",
        ),
    }
    for key, value in expected_runtime.items():
        _require(runtime.get(key) == value,
                 f"{cell_id} runtime {key} binding differs")
    runtime_pair = runtime.get("ocr_pair_execution")
    expected_position = (
        "1_control" if cell.get("arm") == "control" else "2_ocr_zncc"
    )
    _require(
        isinstance(runtime_pair, dict)
        and runtime_pair.get("pair_id")
        == f"{cell.get('recipe')}__seed42__smoke"
        and runtime_pair.get("position") == expected_position
        and runtime_pair.get("slurm_job_id")
        == runtime.get("slurm_job_id")
        and runtime_pair.get("allocated_gpu_count") == 1
        and isinstance(runtime_pair.get("wrapper_pid"), int)
        and runtime_pair["wrapper_pid"] > 0
        and isinstance(runtime_pair.get("cuda_visible_devices"), str)
        and bool(runtime_pair["cuda_visible_devices"])
        and "," not in runtime_pair["cuda_visible_devices"],
        f"{cell_id} runtime pair-wrapper binding differs",
    )
    _require(receipt.get("checkpoint_selection")
             == "minimum_base_validation_loss"
             and receipt.get("checkpoint_validation", {}).get("final_epoch") == 1,
             f"{cell_id} checkpoint semantics differ")
    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, dict), f"{cell_id} artifact records are absent")
    config_path = run_directory / "config.json"
    sampler_path = run_directory / "sampler_order.json"
    config_record = _file_record(config_path, name=f"{cell_id} config")
    sampler_record = _file_record(sampler_path, name=f"{cell_id} sampler order")
    _require(artifacts.get("config.json") == config_record,
             f"{cell_id} live config differs from receipt")
    _require(artifacts.get("sampler_order.json") == sampler_record,
             f"{cell_id} live sampler order differs from receipt")
    config = json.loads(config_path.read_bytes())
    sampler = json.loads(sampler_path.read_bytes())
    pairing = {
        "initial_model_state_sha256": config.get("initial_model_state_sha256"),
        "post_diagnostic_model_state_sha256": config.get(
            "post_diagnostic_model_state_sha256"
        ),
        "diagnostic_pair_order_sha256": config.get(
            "diagnostic_pair_order_sha256"
        ),
        "diagnostic_pair_order_sample_count": config.get(
            "diagnostic_pair_order_sample_count"
        ),
        "train_sampler_seed": sampler.get("generator_seed"),
        "epoch_pair_orders": sampler.get("epochs"),
        "device": execution.get("device"),
        "pair_execution_common": {
            key: runtime_pair[key]
            for key in (
                "pair_id",
                "wrapper_pid",
                "slurm_job_id",
                "cuda_visible_devices",
                "allocated_gpu_count",
            )
        },
    }
    _require(
        all(
            isinstance(pairing[field], str)
            and len(pairing[field]) == 64
            for field in (
                "initial_model_state_sha256",
                "post_diagnostic_model_state_sha256",
                "diagnostic_pair_order_sha256",
            )
        )
        and pairing["diagnostic_pair_order_sample_count"] == 16
        and pairing["train_sampler_seed"]
        == int(cell["training_arguments"]["seed"]) + 10_000_019
        and isinstance(pairing["epoch_pair_orders"], list)
        and len(pairing["epoch_pair_orders"]) == 1
        and pairing["epoch_pair_orders"][0].get("sample_count") == 147
        and pairing["device"] == "cuda",
        f"{cell_id} production pairing evidence differs",
    )
    return {
        "cell_id": cell_id,
        "run_directory": str(run_directory),
        "completed_run_receipt": {
            **record,
            "content_hash_sha256": content_hash,
        },
        "optimizer_step_count": execution["optimizer_step_count"],
        "best_epoch": receipt["checkpoint_validation"]["best_epoch"],
        "final_epoch": receipt["checkpoint_validation"]["final_epoch"],
        "runtime_environment": runtime,
        "matched_pairing_evidence": pairing,
        "pair_position": runtime_pair["position"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[2]
    commit, branch = authorization._source_state(repo_root)
    _require(commit == args.expected_commit, "smoke source commit differs")
    manifest_path = args.experiment_manifest.resolve(strict=True)
    manifest, _ = _load_bound_json(
        manifest_path, expected_sha256=args.experiment_manifest_sha256,
        name="OCR experiment manifest",
    )
    manifest_content_sha256 = _validate_content_hash(
        manifest, name="OCR experiment manifest",
    )
    recipe = manifest.get("decision_lock", {}).get("authorized_recipe")
    _require(recipe in SMOKE_RECIPES, "smoke manifest recipe is invalid")
    smoke_cells = _smoke_cells(str(recipe))
    smoke_launch_lock = _load_smoke_launch_lock(
        source_commit=commit, source_branch=branch,
        manifest_path=manifest_path,
        manifest_sha256=args.experiment_manifest_sha256,
        manifest_content_sha256=manifest_content_sha256,
        recipe=str(recipe), smoke_cells=smoke_cells,
    )
    control_args = _namespace(
        manifest, cell_id=smoke_cells[0],
        path=manifest_path, sha256=args.experiment_manifest_sha256,
    )
    positive_args = _namespace(
        manifest, cell_id=smoke_cells[1],
        path=manifest_path, sha256=args.experiment_manifest_sha256,
    )
    authorization.bind_training_namespace(repo_root, control_args)
    authorization.bind_training_namespace(repo_root, positive_args)
    _require(bool(os.environ.get("SLURM_JOB_ID")),
             "GPU smoke requires a Slurm job identity")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device("cuda")
    dataset = IndexPairDataset(
        data_root=control_args.data_root,
        index_path=control_args.train_pairs_index,
        img_size=control_args.img_size,
        center_crop=control_args.center_crop,
        include_backward=False,
        object_name=control_args.object,
        strict_metadata=True,
        expected_split="train",
        expected_index_sha256=TRAIN_INDEX_SHA256,
        expected_dataset_binding_sha256=control_args.dataset_binding_sha256,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=control_args.batch_size, shuffle=False, num_workers=0,
    )))
    manual_started = time.monotonic()
    manual = _manual_cuda_checks(
        control_args=control_args, positive_args=positive_args,
        batch=batch, device=device,
    )
    manual_elapsed = time.monotonic() - manual_started
    gc.collect()
    torch.cuda.empty_cache()

    production_started = time.monotonic()
    production_cells: list[dict[str, Any]] = []
    pair_id = f"{recipe}__seed42__smoke"
    for position, cell_id in enumerate(smoke_cells, start=1):
        cell = _cell(manifest, cell_id)
        command = paired_training._cell_command(
            cell=cell, manifest_path=manifest_path,
            manifest_sha256=args.experiment_manifest_sha256,
        )
        child_environment = os.environ.copy()
        child_environment.update({
            "OCR_PAIR_ID": pair_id,
            "OCR_PAIR_POSITION": (
                "1_control" if position == 1 else "2_ocr_zncc"
            ),
            "OCR_PAIR_WRAPPER_PID": str(os.getpid()),
        })
        execution = _run_production_command(
            command,
            environment=child_environment,
        )
        receipt = _validate_completion_receipt(
            cell=cell, command=command, source_commit=commit,
            source_branch=branch, manifest_path=manifest_path,
            manifest_sha256=args.experiment_manifest_sha256,
            smoke_launch_lock=smoke_launch_lock,
        )
        production_cells.append({**receipt, **execution})
    production_elapsed = time.monotonic() - production_started

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ocr_zncc_gpu_execution_smoke",
        "status": "pass",
        "scientific_result_authorized": False,
        "source_commit": commit,
        "source_branch": branch,
        "recipe": recipe,
        "manifest": {
            "absolute_path": str(manifest_path),
            "file_sha256": args.experiment_manifest_sha256,
            "content_hash_sha256": manifest["content_hash_sha256"],
        },
        "runtime_provenance": {
            "smoke_launch_lock": smoke_launch_lock,
            "environment_lock": _file_record(
                Path(str(os.environ["OCR_ENV_LOCK"])), name="environment lock",
            ),
            "slurm_script": _file_record(
                Path(str(os.environ["OCR_SLURM_SCRIPT"])), name="Slurm script",
            ),
        },
        "cuda": {
            "available": True,
            "device_name": torch.cuda.get_device_name(device),
            "device_count_visible": torch.cuda.device_count(),
            "pytorch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "manual_checks_memory": manual["cuda_memory"],
        },
        "batch": manual["batch"],
        "checks": manual["checks"],
        "one_step": manual["one_step"],
        "elapsed_time": {
            "clock": "time.monotonic",
            "manual_checks_seconds": manual_elapsed,
            "production_cells_seconds": production_elapsed,
            "total_run_seconds": time.monotonic() - run_started,
        },
        "production_path": {
            "entrypoint": str(
                (repo_root / "keypoint_net/train.py").resolve(strict=True)
            ),
            "execution_order": list(smoke_cells),
            "cell_count": len(production_cells),
            "elapsed_seconds": production_elapsed,
            "cells": production_cells,
            "semantics": (
                "exact one-epoch manifest-bound train.py executions; their "
                "outputs are execution evidence only and are not scientific runs"
            ),
        },
    }
    result["content_hash_sha256"] = authorization.canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--experiment-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"], "output": str(args.output),
        "file_sha256": _sha256(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
