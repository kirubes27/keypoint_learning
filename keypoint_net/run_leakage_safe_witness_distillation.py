"""Train the detector with training-witness truth only, on local CPU."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    CapabilityContractError,
    dense_heatmap_cross_entropy,
    evaluation_score,
    file_record,
    model_state_sha256,
    require,
    sha256_file,
)
from leakage_safe_distillation_contract import (
    EXPECTED_INITIAL_MODEL_STATE_SHA256,
    FULL_TRACKS_SHA256,
    SEMANTIC_LOCK_SHA256,
    TRAIN_FRAMES,
    compact_report,
    load_bound_training_dataset,
    load_json,
    predict_model_readouts,
    require_local_cpu_only,
    verify_clean_repository,
    verify_record,
    verify_semantic_lock,
    write_json,
)
from model import KeypointExtractor
from run_certified_witness_capability import _save_worst_montage
from run_certified_witness_local_confirmation import _local_semantic_controls


def _source_hashes(args: argparse.Namespace, train_targets: Path, train_manifest: Path) -> dict[str, str]:
    paths = {
        "semantic_lock": args.semantic_lock,
        "train_input_receipt": args.train_input_receipt,
        "train_targets": train_targets,
        "train_frame_manifest": train_manifest,
        "runner": Path(__file__),
        "contract": args.repo_root
        / "keypoint_net"
        / "leakage_safe_distillation_contract.py",
        "capability_contract": args.repo_root
        / "keypoint_net"
        / "certified_witness_capability.py",
        "local_readout": args.repo_root
        / "keypoint_net"
        / "certified_witness_local_readout.py",
        "local_readout_controls": args.repo_root
        / "keypoint_net"
        / "run_certified_witness_local_confirmation.py",
        "model": args.repo_root / "keypoint_net" / "model.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _row(
    update: int,
    epoch: int,
    loss: torch.Tensor,
    global_report: dict[str, Any],
    local_report: dict[str, Any],
    update_seconds: float,
    evaluation_seconds: float,
) -> dict[str, Any]:
    gv = global_report["violations"]
    lv = local_report["violations"]
    return {
        "update": update,
        "epoch": epoch,
        "train_batch_loss": float(loss.detach().cpu()),
        "update_seconds": update_seconds,
        "evaluation_seconds": evaluation_seconds,
        "global_strict_pass": bool(global_report["strict_capability_pass"]),
        "global_outside_half_cell_count": int(gv["outside_half_cell_count"]),
        "global_wrong_identity_count": int(gv["wrong_identity_count"]),
        "global_collapsed_pair_count": int(gv["collapsed_pair_count"]),
        "global_off_object_count": int(gv["off_object_count"]),
        "global_median_error_px": float(global_report["material_error_px"]["median"]),
        "global_maximum_error_px": float(global_report["material_error_px"]["maximum"]),
        "local_strict_pass": bool(local_report["strict_capability_pass"]),
        "local_outside_half_cell_count": int(lv["outside_half_cell_count"]),
        "local_wrong_identity_count": int(lv["wrong_identity_count"]),
        "local_collapsed_pair_count": int(lv["collapsed_pair_count"]),
        "local_off_object_count": int(lv["off_object_count"]),
        "local_median_error_px": float(local_report["material_error_px"]["median"]),
        "local_maximum_error_px": float(local_report["material_error_px"]["maximum"]),
    }


def _timing_projection(
    update_times: list[float], evaluation_times: list[float]
) -> dict[str, Any]:
    require(len(update_times) >= 5, "timing smoke has too few optimizer updates")
    require(len(evaluation_times) >= 2, "timing smoke has too few evaluations")
    steady = np.asarray(update_times[-5:], dtype=np.float64)
    update_estimate = float(max(steady.mean(), np.quantile(steady, 0.9)))
    evaluation_estimate = float(max(evaluation_times))
    evaluation_count = 51  # update 1 and updates 100..5000 inclusive
    projected = 1.25 * (
        5000.0 * update_estimate + evaluation_count * evaluation_estimate
    )
    return {
        "update_seconds_estimate": update_estimate,
        "evaluation_seconds_estimate": evaluation_estimate,
        "full_update_count": 5000,
        "full_evaluation_count": evaluation_count,
        "safety_factor": 1.25,
        "projected_full_seconds": projected,
        "maximum_authorized_seconds": 7200.0,
        "full_run_authorized": bool(projected <= 7200.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    require(args.seed == 42, "only seed 42 is frozen")
    require(args.device == "cpu", "only CPU is frozen")
    if args.run_kind == "smoke":
        require(args.max_updates == 10, "smoke requires ten updates")
        require(args.eval_every == 10, "smoke requires evaluation at update ten")
    else:
        require(args.run_kind == "full", "unknown run kind")
        require(args.max_updates == 5000, "full run requires 5000 updates")
        require(args.eval_every == 100, "full run requires evaluation every 100 updates")
    require(args.batch_size == 16, "batch size differs from frozen recipe")
    require("SEALED_VALIDATION_TRUTH" not in " ".join(sys.argv), "validation truth entered training command")

    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)
    verify_semantic_lock(args.semantic_lock)
    receipt_text = args.train_input_receipt.read_text()
    require("SEALED_VALIDATION_TRUTH" not in receipt_text, "training receipt names validation truth")
    require(FULL_TRACKS_SHA256 not in receipt_text, "training receipt contains full-track binding")
    receipt = load_json(args.train_input_receipt)
    require(
        receipt.get("schema_version")
        == "leakage_safe_distillation_train_input_receipt.v1",
        "training receipt schema differs",
    )
    require(receipt.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256, "training receipt lock differs")
    require(receipt.get("repository_head") == repository_head, "training receipt repository head differs")
    require(receipt.get("validation_truth_received_by_training_stage") is False, "training receipt boundary differs")
    require(receipt.get("full_track_archive_received_by_training_stage") is False, "training receipt full-track boundary differs")
    train_targets = verify_record(receipt["train_targets"], "train targets")
    train_manifest = verify_record(receipt["train_frame_manifest"], "train manifest")
    require(np.array_equal(np.asarray(receipt["training_frame_indices"], dtype=np.int64), TRAIN_FRAMES), "training receipt frames differ")
    environment = require_local_cpu_only()

    dataset, target_px, masks, original_frames, data_controls = load_bound_training_dataset(
        train_manifest, train_targets, args.object_root
    )
    local_controls = _local_semantic_controls(dataset, target_px, masks)
    source_hashes_before = _source_hashes(args, train_targets, train_manifest)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    initial_state_hash = model_state_sha256(model)
    require(
        initial_state_hash == EXPECTED_INITIAL_MODEL_STATE_SHA256,
        "seed-42 initial model state differs",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    config = {
        "schema_version": "leakage_safe_witness_distillation_training_config.v1",
        "run_kind": args.run_kind,
        "command_argv": list(sys.argv),
        "repository_head": repository_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "contract": file_record(
                args.repo_root
                / "keypoint_net"
                / "leakage_safe_distillation_contract.py"
            ),
            "capability_contract": file_record(
                args.repo_root / "keypoint_net" / "certified_witness_capability.py"
            ),
            "local_readout": file_record(
                args.repo_root
                / "keypoint_net"
                / "certified_witness_local_readout.py"
            ),
            "local_controls": file_record(
                args.repo_root
                / "keypoint_net"
                / "run_certified_witness_local_confirmation.py"
            ),
            "model": file_record(args.repo_root / "keypoint_net" / "model.py"),
        },
        "semantic_lock": file_record(args.semantic_lock),
        "train_input_receipt": file_record(args.train_input_receipt),
        "seed": args.seed,
        "device": "cpu",
        "training_frame_indices": original_frames.tolist(),
        "validation_truth_path_received": False,
        "max_updates": args.max_updates,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "model": {
            "type": "KeypointExtractor",
            "num_keypoints": 10,
            "base_channels": 32,
            "temperature": 1.0,
            "padding_mode": "reflect",
            "heatmap_resolution": 64,
            "true_quarter_resolution": False,
        },
        "loss": "gaussian_target_distribution_cross_entropy_only",
        "sigma_input_px": 8.0,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "augmentation": "none",
        "checkpoint_selection": "training_only_fixed_local_3x3_lexicographic_score",
        "concurrent_control": "native_global_soft_argmax",
        "environment": environment,
    }
    config_path = args.output_dir / "config.json"
    write_json(config_path, config)
    controls = {
        "data_controls": data_controls,
        "local_readout_controls": local_controls,
        "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_STATE_SHA256,
        "actual_initial_model_state_sha256": initial_state_hash,
        "source_hashes_before_first_update": source_hashes_before,
        "validation_truth_received_or_opened": False,
        "full_track_archive_received_or_opened": False,
    }
    controls_path = args.output_dir / "semantic_controls.json"
    write_json(controls_path, controls)

    history: list[dict[str, Any]] = []
    best_score: tuple[int, int, int, int, float, float] | None = None
    best_checkpoint_path: Path | None = None
    parameter_change_proved = False
    update_times: list[float] = []
    evaluation_times: list[float] = []
    update = 0
    epoch = 0
    start = time.perf_counter()
    while update < args.max_updates:
        epoch += 1
        model.train()
        for batch in loader:
            update_start = time.perf_counter()
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            _, logits = model(images)
            loss = dense_heatmap_cross_entropy(logits, targets, sigma_input_px=8.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update += 1
            update_seconds = time.perf_counter() - update_start
            update_times.append(update_seconds)
            if update == 1:
                parameter_change_proved = model_state_sha256(model) != initial_state_hash
                require(parameter_change_proved, "optimizer did not change model state")
                require(
                    _source_hashes(args, train_targets, train_manifest)
                    == source_hashes_before,
                    "a bound source changed through first update",
                )
            should_evaluate = (
                update == 1
                or update % args.eval_every == 0
                or update == args.max_updates
            )
            if should_evaluate:
                evaluation_start = time.perf_counter()
                arrays, global_report, local_report = predict_model_readouts(
                    model,
                    dataset,
                    device,
                    args.batch_size,
                    target_px=target_px,
                    masks=masks,
                )
                del arrays
                require(global_report is not None and local_report is not None, "training evaluation missing")
                evaluation_seconds = time.perf_counter() - evaluation_start
                evaluation_times.append(evaluation_seconds)
                score = evaluation_score(local_report)
                row = _row(
                    update,
                    epoch,
                    loss,
                    global_report,
                    local_report,
                    update_seconds,
                    evaluation_seconds,
                )
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                if best_score is None or score < best_score:
                    best_score = score
                    state_hash = model_state_sha256(model)
                    checkpoint_path = checkpoint_dir / f"update_{update:06d}.pt"
                    torch.save(
                        {
                            "schema_version": "leakage_safe_witness_distillation_checkpoint.v1",
                            "extractor_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "update": update,
                            "epoch": epoch,
                            "seed": args.seed,
                            "training_local_score": list(score),
                            "training_global_score": list(evaluation_score(global_report)),
                            "model_state_sha256": state_hash,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    best_checkpoint_path = checkpoint_path
                model.train()
            if update >= args.max_updates:
                break

    require(best_checkpoint_path is not None and best_score is not None, "no checkpoint selected")
    best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["extractor_state_dict"], strict=True)
    loaded_state_hash = model_state_sha256(model)
    require(loaded_state_hash == best_payload["model_state_sha256"], "checkpoint state hash differs")
    arrays, global_report, local_report = predict_model_readouts(
        model,
        dataset,
        device,
        args.batch_size,
        target_px=target_px,
        masks=masks,
    )
    require(global_report is not None and local_report is not None, "selected checkpoint evaluation missing")
    require(evaluation_score(local_report) == best_score, "selected local score replay differs")
    replay_arrays, replay_global, replay_local = predict_model_readouts(
        model,
        dataset,
        device,
        args.batch_size,
        target_px=target_px,
        masks=masks,
    )
    require(replay_global == global_report, "selected global report replay differs")
    require(replay_local == local_report, "selected local report replay differs")
    for name in (
        "native_heatmap_logits",
        "global_soft_prediction_px",
        "local_3x3_prediction_px",
        "hard_cell_x",
        "hard_cell_y",
    ):
        require(np.array_equal(replay_arrays[name], arrays[name]), f"{name} replay differs")

    best_copy = args.output_dir / "best_model.pt"
    shutil.copy2(best_checkpoint_path, best_copy)
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    arrays_path = args.output_dir / "TRAINING_SELECTED_PREDICTIONS.npz"
    arrays["frame_index"] = original_frames
    arrays["target_coordinate_px"] = target_px
    np.savez_compressed(arrays_path, **arrays)
    montage_path = args.output_dir / "TRAINING_WORST_LOCAL_EVENTS.png"
    local_error = np.linalg.norm(arrays["local_3x3_prediction_px"] - target_px, axis=-1)
    _save_worst_montage(
        dataset.images,
        arrays["local_3x3_prediction_px"],
        target_px,
        local_error,
        montage_path,
    )
    pip_freeze_path = args.output_dir / "PIP_FREEZE.txt"
    pip_freeze_path.write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    timing = _timing_projection(update_times, evaluation_times)
    if args.run_kind == "full":
        timing["full_run_authorized"] = True
        timing["projection_role"] = "postrun_diagnostic_only"
    decision_branch = (
        "authorize_full_training"
        if args.run_kind == "smoke" and timing["full_run_authorized"]
        else "reject_full_training_runtime_or_control"
        if args.run_kind == "smoke"
        else "freeze_checkpoint_and_run_truth_free_inference"
    )
    result_path = args.output_dir / "DISTILLATION_TRAINING_RESULT.json"
    result = {
        "schema_version": "leakage_safe_witness_distillation_training_result.v1",
        "artifact_type": "training_truth_only_supervised_detector",
        "run_kind": args.run_kind,
        "repository_head": repository_head,
        "runtime_seconds": time.perf_counter() - start,
        "completed_updates": update,
        "selected_update": int(best_payload["update"]),
        "initial_model_state_sha256": initial_state_hash,
        "selected_model_state_sha256": loaded_state_hash,
        "parameter_change_proved": parameter_change_proved,
        "checkpoint_round_trip_exact": True,
        "selected_prediction_replay_exact": True,
        "source_hashes_unchanged_through_first_update": True,
        "validation_truth_received_or_opened": False,
        "full_track_archive_received_or_opened": False,
        "training_global_control": compact_report(global_report),
        "training_local_candidate": compact_report(local_report),
        "timing_projection": timing,
        "decision_branch": decision_branch,
        "statistical_scope": {
            "inference": "descriptive_only",
            "object_count": 1,
            "orbit_count": 1,
            "optimization_seed_count": 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "TRAINING_RUN_RECEIPT.json"
    receipt = {
        "schema_version": "leakage_safe_witness_distillation_training_receipt.v1",
        "result": file_record(result_path),
        "config": file_record(config_path),
        "semantic_controls": file_record(controls_path),
        "selected_model": file_record(best_copy),
        "selected_checkpoint": file_record(best_checkpoint_path),
        "history": file_record(history_path),
        "training_predictions": file_record(arrays_path),
        "training_worst_visual": file_record(montage_path),
        "pip_freeze": file_record(pip_freeze_path),
        "run_kind": args.run_kind,
        "decision_branch": decision_branch,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--train-input-receipt", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"DISTILLATION TRAINING CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
