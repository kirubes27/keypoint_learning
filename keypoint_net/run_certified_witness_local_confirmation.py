"""Run the predeclared seed-42 local-readout capability confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    bilinear_planted_logits,
    dense_heatmap_cross_entropy,
    evaluate_predictions,
    evaluation_score,
    file_record,
    model_state_sha256,
    normalized_to_pixel,
    require,
    sha256_file,
)
from certified_witness_local_readout import (
    classify_localization_failures,
    compact_information_boundary,
    grid_to_pixel,
    readout_arrays,
)
from model import KeypointExtractor
from run_certified_witness_capability import (
    BoundCapabilityDataset,
    _load_bound_inputs,
    _save_worst_montage,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _summary(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "invalid summary vector")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "q90": float(np.quantile(vector, 0.9)),
        "maximum": float(vector.max()),
    }


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_capability_pass": report["strict_capability_pass"],
        "violations": report["violations"],
        "material_error_px": report["material_error_px"],
        "within_half_cell_rate": report["within_half_cell_rate"],
        "on_object_rate": report["on_object_rate"],
        "identity_assignment_rate": report["identity_assignment_rate"],
        "minimum_predicted_pair_distance_px": report[
            "minimum_predicted_pair_distance_px"
        ],
        "minimum_predicted_to_physical_pair_ratio": report[
            "minimum_predicted_to_physical_pair_ratio"
        ],
    }


def _verify_repository(args: argparse.Namespace) -> str:
    head = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == args.expected_repo_head, "repository HEAD differs from command lock")
    status = subprocess.run(
        ["git", "-C", str(args.repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "repository is not clean")
    return head


def _verify_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    require(
        sha256_file(args.audit_lock) == args.expected_audit_lock_sha256,
        "audit-lock SHA-256 differs",
    )
    require(
        sha256_file(args.phase_a_receipt) == args.expected_phase_a_receipt_sha256,
        "Phase-A receipt SHA-256 differs",
    )
    receipt = _load_json(args.phase_a_receipt)
    require(receipt.get("phase_a_gate_pass") is True, "Phase-A receipt does not pass")
    require(
        receipt.get("phase_b_confirmation_authorized") is True,
        "Phase-A receipt does not authorize confirmation",
    )
    result_record = receipt.get("result")
    require(isinstance(result_record, dict), "Phase-A result record missing")
    result_path = Path(str(result_record.get("absolute_path", "")))
    require(result_path.is_file(), "Phase-A result missing")
    require(
        sha256_file(result_path) == result_record.get("sha256"),
        "Phase-A result SHA-256 differs from receipt",
    )
    result = _load_json(result_path)
    require(result.get("phase_a_gate_pass") is True, "Phase-A result does not pass")
    require(
        result.get("previous_readout_replay_exact_all_seeds") is True,
        "Phase-A result did not replay the prior readout exactly",
    )
    return receipt


def _local_semantic_controls(
    dataset: BoundCapabilityDataset,
    target_px: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    planted_logits = bilinear_planted_logits(dataset.targets).numpy()
    planted = readout_arrays(planted_logits, target_px)
    planted_report, _ = evaluate_predictions(
        planted["local_3x3_prediction_px"], target_px, masks
    )
    require(
        planted_report["strict_capability_pass"] is True,
        "planted local-readout positive control failed",
    )

    remote_logits = np.full(
        (1, EXPECTED_WITNESSES, 64, 64), -80.0, dtype=np.float64
    )
    remote_logits[:, :, 40, 41] = 9.0
    remote_logits[:, :, 10, 11] = 8.0
    remote_target = np.broadcast_to(
        grid_to_pixel(11.0, 10.0), (1, EXPECTED_WITNESSES, 2)
    ).copy()
    remote = readout_arrays(remote_logits, remote_target)
    remote_error = np.linalg.norm(
        remote["local_3x3_prediction_px"] - remote_target, axis=-1
    )
    within = remote_error <= planted_report["half_cell_diagonal_px"] + 1e-12
    category, category_counts = classify_localization_failures(remote, within)
    require(bool(np.all(remote["target_nearest_cell_rank"] == 2)), "remote alias rank differs")
    require(bool(np.all(category == 2)), "remote alias was not rejected as wrong coarse mode")

    return {
        "planted_local_readout_full_contract": planted_report,
        "remote_alias_negative": {
            "hard_argmax_cell_xy": [41, 40],
            "target_cell_xy": [11, 10],
            "target_cell_rank": 2,
            "expected_category": "wrong_coarse_mode_target_top10",
            "category_counts": category_counts,
            "passed": True,
        },
        "border_clipping_tie_temperature_and_renormalization": {
            "status": "covered_by_committed_unit_tests",
            "temperature": 1.0,
            "window": "clipped_3x3",
            "tie_rule": "numpy_row_major_first_argmax",
            "renormalized_inside_window": True,
        },
    }


@torch.no_grad()
def _evaluate_pair(
    model: KeypointExtractor,
    dataset: BoundCapabilityDataset,
    target_px: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    global_normalized_values: list[np.ndarray] = []
    logit_values: list[np.ndarray] = []
    frame_values: list[np.ndarray] = []
    for batch in loader:
        flat, logits = model(batch["image"].to(device))
        global_normalized_values.append(
            flat.view(-1, EXPECTED_WITNESSES, 2).cpu().numpy()
        )
        logit_values.append(logits.cpu().numpy())
        frame_values.append(batch["frame"].numpy())

    frame_index = np.concatenate(frame_values)
    order = np.argsort(frame_index)
    require(
        np.array_equal(frame_index[order], np.arange(len(dataset))),
        "evaluation frame order incomplete",
    )
    global_normalized = np.concatenate(global_normalized_values)[order]
    logits = np.concatenate(logit_values)[order]
    global_px = normalized_to_pixel(global_normalized)
    readouts = readout_arrays(logits, target_px)
    global_report, global_derived = evaluate_predictions(global_px, target_px, masks)
    local_report, local_derived = evaluate_predictions(
        readouts["local_3x3_prediction_px"], target_px, masks
    )
    global_report["heatmap_entropy"] = _summary(readouts["heatmap_entropy"])
    global_report["heatmap_peak_probability"] = _summary(readouts["top1_probability"])
    local_report["heatmap_entropy"] = _summary(readouts["heatmap_entropy"])
    local_report["heatmap_peak_probability"] = _summary(readouts["top1_probability"])

    derived = {
        "frame_index": np.arange(len(dataset), dtype=np.int64),
        "target_coordinate_px": target_px,
        "native_heatmap_logits": logits,
        "global_soft_prediction_normalized": global_normalized,
        "global_soft_prediction_px": global_px,
        "global_material_error_px": global_derived["material_error_px"],
        "global_within_half_cell": global_derived["within_half_cell"],
        "global_on_object": global_derived["on_object"],
        "global_identity_correct": global_derived["identity_correct"],
        "global_distinct_pair": global_derived["distinct_pair"],
        "hard_cell_x": readouts["hard_cell_x"],
        "hard_cell_y": readouts["hard_cell_y"],
        "hard_prediction_px": readouts["hard_prediction_px"],
        "local_3x3_prediction_px": readouts["local_3x3_prediction_px"],
        "local_material_error_px": local_derived["material_error_px"],
        "local_within_half_cell": local_derived["within_half_cell"],
        "local_on_object": local_derived["on_object"],
        "local_identity_correct": local_derived["identity_correct"],
        "local_distinct_pair": local_derived["distinct_pair"],
        "target_cell_x": readouts["target_cell_x"],
        "target_cell_y": readouts["target_cell_y"],
        "target_nearest_cell_rank": readouts["target_nearest_cell_rank"],
        "target_cell_inside_local_window": readouts[
            "target_cell_inside_local_window"
        ],
        "inside_window_probability_mass": readouts[
            "inside_window_probability_mass"
        ],
        "outside_window_probability_mass": readouts[
            "outside_window_probability_mass"
        ],
        "top1_probability": readouts["top1_probability"],
        "top2_probability": readouts["top2_probability"],
        "top1_top2_probability_margin": readouts[
            "top1_top2_probability_margin"
        ],
        "heatmap_entropy": readouts["heatmap_entropy"],
    }
    return global_report, local_report, derived


def _source_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "manifest": args.manifest,
        "tracks": args.tracks,
        "audit_lock": args.audit_lock,
        "phase_a_receipt": args.phase_a_receipt,
        "runner": Path(__file__),
        "local_readout": args.repo_root
        / "keypoint_net"
        / "certified_witness_local_readout.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _row(
    update: int,
    epoch: int,
    loss: torch.Tensor,
    global_report: dict[str, Any],
    local_report: dict[str, Any],
) -> dict[str, Any]:
    global_violations = global_report["violations"]
    local_violations = local_report["violations"]
    return {
        "update": update,
        "epoch": epoch,
        "train_batch_loss": float(loss.detach().cpu()),
        "global_strict_capability_pass": global_report["strict_capability_pass"],
        "global_outside_half_cell_count": global_violations[
            "outside_half_cell_count"
        ],
        "global_wrong_identity_count": global_violations["wrong_identity_count"],
        "global_collapsed_pair_count": global_violations["collapsed_pair_count"],
        "global_off_object_count": global_violations["off_object_count"],
        "global_median_material_error_px": global_report["material_error_px"][
            "median"
        ],
        "global_maximum_material_error_px": global_report["material_error_px"][
            "maximum"
        ],
        "local_strict_capability_pass": local_report["strict_capability_pass"],
        "local_outside_half_cell_count": local_violations[
            "outside_half_cell_count"
        ],
        "local_wrong_identity_count": local_violations["wrong_identity_count"],
        "local_collapsed_pair_count": local_violations["collapsed_pair_count"],
        "local_off_object_count": local_violations["off_object_count"],
        "local_median_material_error_px": local_report["material_error_px"][
            "median"
        ],
        "local_maximum_material_error_px": local_report["material_error_px"][
            "maximum"
        ],
    }


def _event_list(mask: np.ndarray) -> list[dict[str, int]]:
    return [
        {
            "frame": int(frame),
            "channel": int(channel),
            "witness_id": int(EXPECTED_WITNESS_IDS[int(channel)]),
        }
        for frame, channel in np.argwhere(mask)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh run")
    require(args.seed == 42, "the confirmation lock permits only seed 42")
    require(args.frame_limit == EXPECTED_FRAMES, "the confirmation requires all 180 frames")
    require(args.batch_size == 16, "the confirmation requires batch size 16")
    require(args.device == "cpu", "the matched seed-42 confirmation requires CPU")
    if args.run_kind == "smoke":
        require(args.max_updates == 1, "one-update smoke must use max_updates=1")
        require(args.eval_every == 1, "one-update smoke must evaluate update 1")
    else:
        require(args.max_updates == 5000, "confirmation requires a 5000-update ceiling")
        require(args.eval_every == 100, "confirmation requires evaluation every 100 updates")

    repository_head = _verify_repository(args)
    phase_a_receipt = _verify_phase_a(args)
    source_hashes_before = _source_hashes(args)
    manifest, dataset, target_px, masks, inherited_controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        args.frame_limit,
    )
    local_controls = _local_semantic_controls(dataset, target_px, masks)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    initial_state_hash = model_state_sha256(model)
    require(
        initial_state_hash == args.expected_initial_model_state_sha256,
        "seed-42 initial model state differs from the matched control",
    )
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
        "schema_version": "certified_witness_local_confirmation_config.v1",
        "run_kind": args.run_kind,
        "repository_head": repository_head,
        "runner": file_record(Path(__file__)),
        "local_readout_source": file_record(
            args.repo_root / "keypoint_net" / "certified_witness_local_readout.py"
        ),
        "audit_lock": file_record(args.audit_lock),
        "phase_a_receipt": file_record(args.phase_a_receipt),
        "phase_a_result": phase_a_receipt["result"],
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "manifest_implementation_head": manifest["implementation_head"],
        "seed": args.seed,
        "device": str(device),
        "frame_limit": args.frame_limit,
        "max_updates": args.max_updates,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "loss": "gaussian_target_distribution_cross_entropy_only",
        "sigma_input_px": 8.0,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "augmentation": "none",
        "preservation_loss_or_intervention": "none",
        "checkpoint_selection": "frozen_evaluation_score_applied_to_local_3x3_report",
        "concurrent_control": "native_global_soft_argmax_from_same_model_state",
        "information_boundary": compact_information_boundary(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    controls = {
        "inherited_manifest_controls": inherited_controls,
        "local_readout_controls": local_controls,
        "expected_initial_model_state_sha256": args.expected_initial_model_state_sha256,
        "actual_initial_model_state_sha256": initial_state_hash,
        "source_hashes_before_first_update": source_hashes_before,
    }
    controls_path = args.output_dir / "semantic_controls.json"
    controls_path.write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n")

    history: list[dict[str, Any]] = []
    best_score: tuple[int, int, int, int, float, float] | None = None
    best_checkpoint_path: Path | None = None
    parameter_change_proved = False
    update = 0
    epoch = 0
    start = time.perf_counter()
    stop = False
    while update < args.max_updates and not stop:
        epoch += 1
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            _, logits = model(images)
            loss = dense_heatmap_cross_entropy(logits, targets, sigma_input_px=8.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update += 1
            if update == 1:
                parameter_change_proved = model_state_sha256(model) != initial_state_hash
                require(parameter_change_proved, "optimizer step did not change model state")
                require(
                    _source_hashes(args) == source_hashes_before,
                    "a bound input or source changed during the first optimizer update",
                )
            should_evaluate = (
                update == 1
                or update % args.eval_every == 0
                or update == args.max_updates
            )
            if should_evaluate:
                global_report, local_report, _ = _evaluate_pair(
                    model, dataset, target_px, masks, device, args.batch_size
                )
                score = evaluation_score(local_report)
                row = _row(update, epoch, loss, global_report, local_report)
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                if best_score is None or score < best_score:
                    best_score = score
                    checkpoint_path = checkpoint_dir / f"update_{update:06d}.pt"
                    state_hash = model_state_sha256(model)
                    torch.save(
                        {
                            "schema_version": "certified_witness_local_confirmation_checkpoint.v1",
                            "extractor_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "update": update,
                            "epoch": epoch,
                            "seed": args.seed,
                            "local_score": list(score),
                            "global_score": list(evaluation_score(global_report)),
                            "model_state_sha256": state_hash,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    best_checkpoint_path = checkpoint_path
                if local_report["strict_capability_pass"] and args.stop_on_pass:
                    stop = True
                if not stop:
                    model.train()
            if update >= args.max_updates or stop:
                break

    require(best_checkpoint_path is not None and best_score is not None, "no checkpoint selected")
    best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["extractor_state_dict"], strict=True)
    loaded_state_hash = model_state_sha256(model)
    require(
        loaded_state_hash == best_payload["model_state_sha256"],
        "checkpoint round-trip state hash differs",
    )
    global_report, local_report, derived = _evaluate_pair(
        model, dataset, target_px, masks, device, args.batch_size
    )
    require(evaluation_score(local_report) == best_score, "reloaded local score differs")
    require(
        list(evaluation_score(global_report)) == best_payload["global_score"],
        "reloaded global score differs",
    )
    replay_global, replay_local, replay_derived = _evaluate_pair(
        model, dataset, target_px, masks, device, args.batch_size
    )
    require(replay_global == global_report, "global report replay differs")
    require(replay_local == local_report, "local report replay differs")
    for name in (
        "native_heatmap_logits",
        "global_soft_prediction_px",
        "local_3x3_prediction_px",
        "hard_prediction_px",
    ):
        require(
            np.array_equal(replay_derived[name], derived[name]),
            f"selected checkpoint {name} replay differs",
        )

    category, category_counts = classify_localization_failures(
        readout_arrays(derived["native_heatmap_logits"], target_px),
        derived["local_within_half_cell"],
    )
    derived["localization_category_code"] = category
    residual_mask = (
        np.logical_not(derived["local_within_half_cell"])
        | np.logical_not(derived["local_on_object"])
        | np.logical_not(derived["local_identity_correct"])
    )

    best_copy = args.output_dir / "best_model.pt"
    shutil.copy2(best_checkpoint_path, best_copy)
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    predictions_path = args.output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **derived)
    with np.load(predictions_path) as saved:
        require(
            np.array_equal(saved["global_soft_prediction_px"], derived["global_soft_prediction_px"]),
            "saved global coordinate replay differs",
        )
        require(
            np.array_equal(saved["local_3x3_prediction_px"], derived["local_3x3_prediction_px"]),
            "saved local coordinate replay differs",
        )

    global_montage_path = args.output_dir / "global_worst_events.png"
    local_montage_path = args.output_dir / "local_worst_events.png"
    _save_worst_montage(
        dataset.images,
        derived["global_soft_prediction_px"],
        target_px,
        derived["global_material_error_px"],
        global_montage_path,
    )
    _save_worst_montage(
        dataset.images,
        derived["local_3x3_prediction_px"],
        target_px,
        derived["local_material_error_px"],
        local_montage_path,
    )

    is_confirmation = args.run_kind == "confirmation"
    strict_local_pass = bool(local_report["strict_capability_pass"])
    result = {
        "schema_version": "certified_witness_local_confirmation_result.v1",
        "artifact_type": "source_bound_supervised_local_readout_confirmation",
        "run_kind": args.run_kind,
        "scientific_full_confirmation": is_confirmation,
        "seed": args.seed,
        "best_update": int(best_payload["update"]),
        "completed_updates": update,
        "runtime_seconds": time.perf_counter() - start,
        "device": str(device),
        "initial_model_state_sha256": initial_state_hash,
        "best_model_state_sha256": loaded_state_hash,
        "optimizer_parameter_change_proved": parameter_change_proved,
        "source_hashes_unchanged_through_first_update": True,
        "checkpoint_round_trip_exact": True,
        "global_and_local_prediction_replay_exact": True,
        "checkpoint_selection_readout": "local_3x3",
        "concurrent_global_control": _compact_report(global_report),
        "local_3x3_candidate": _compact_report(local_report),
        "localization_category_counts": category_counts,
        "local_residual_event_count": int(residual_mask.sum()),
        "local_residual_events": _event_list(residual_mask),
        "strict_local_capability_pass": strict_local_pass,
        "decision_branch": (
            "bounded_one_update_smoke_only"
            if not is_confirmation
            else "strict_seed42_local_capability_pass_run_seeds43_44"
            if strict_local_pass
            else "strict_seed42_local_capability_not_reached_inspect_residual_geometry"
        ),
        "unsupervised_discovery_established": False,
        "preservation_phase_authorized_by_this_result": False,
        "statistical_scope": {
            "inference": "descriptive_only",
            "optimization_seed_count": 1,
            "object_count": 1,
            "orbit_count": 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    result_path = args.output_dir / "LOCAL_READOUT_CAPABILITY_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_local_confirmation_receipt.v1",
        "result": file_record(result_path),
        "config": file_record(config_path),
        "semantic_controls": file_record(controls_path),
        "best_model": file_record(best_copy),
        "selected_checkpoint": file_record(best_checkpoint_path),
        "history": file_record(history_path),
        "predictions": file_record(predictions_path),
        "global_worst_events_visual": file_record(global_montage_path),
        "local_worst_events_visual": file_record(local_montage_path),
        "run_kind": args.run_kind,
        "strict_local_capability_pass": strict_local_pass,
        "scientific_full_confirmation": is_confirmation,
        "preservation_phase_authorized": False,
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--audit-lock", type=Path, required=True)
    parser.add_argument("--expected-audit-lock-sha256", required=True)
    parser.add_argument("--phase-a-receipt", type=Path, required=True)
    parser.add_argument("--expected-phase-a-receipt-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--expected-initial-model-state-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--frame-limit", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stop-on-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"LOCAL CONFIRMATION FAILURE: {error}") from error


if __name__ == "__main__":
    main()
