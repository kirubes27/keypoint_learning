"""Probe BatchNorm state on a frozen certified-witness capability checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    CapabilityContractError,
    evaluate_predictions,
    evaluation_score,
    file_record,
    model_state_sha256,
    normalized_to_pixel,
    require,
    sha256_file,
)
from model import KeypointExtractor
from run_certified_witness_capability import (
    BoundCapabilityDataset,
    _evaluate_model,
    _load_bound_inputs,
    _summary,
)


def _parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _new_model(state_dict: dict[str, torch.Tensor], device: torch.device) -> KeypointExtractor:
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def _evaluate_with_batch_statistics(
    model: KeypointExtractor,
    dataset: BoundCapabilityDataset,
    target_px: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
    indices: Iterable[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate using per-batch BN statistics; diagnostic only, not deployment."""
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.train()
    predictions: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    peak_probability: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for batch in loader:
        flat, logits = model(batch["image"].to(device))
        coordinates = flat.view(-1, EXPECTED_WITNESSES, 2)
        probability = torch.softmax(logits.flatten(-2), dim=-1)
        predictions.append(coordinates.cpu().numpy())
        entropy.append((-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)).cpu().numpy())
        peak_probability.append(probability.max(dim=-1).values.cpu().numpy())
        frames.append(batch["frame"].numpy())
    prediction_normalized = np.concatenate(predictions)
    frame_index = np.concatenate(frames)
    order = np.argsort(frame_index)
    require(np.array_equal(frame_index[order], np.arange(len(dataset))), "diagnostic frame order incomplete")
    prediction_normalized = prediction_normalized[order]
    entropy_array = np.concatenate(entropy)[order]
    peak_array = np.concatenate(peak_probability)[order]
    prediction_px = normalized_to_pixel(prediction_normalized)
    report, derived = evaluate_predictions(prediction_px, target_px, masks)
    report["heatmap_entropy"] = _summary(entropy_array)
    report["heatmap_peak_probability"] = _summary(peak_array)
    derived.update(
        {
            "frame_index": np.arange(len(dataset), dtype=np.int64),
            "prediction_coordinate_normalized": prediction_normalized,
            "prediction_coordinate_px": prediction_px,
            "target_coordinate_px": target_px,
            "heatmap_entropy": entropy_array,
            "heatmap_peak_probability": peak_array,
        }
    )
    return report, derived


@torch.no_grad()
def _recalibrate_running_statistics(
    model: KeypointExtractor,
    dataset: BoundCapabilityDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Reset and cumulatively estimate BN buffers from one fixed image pass."""
    batch_norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    require(bool(batch_norms), "model has no BatchNorm modules")
    original_momenta = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = 0
    for batch in loader:
        model(batch["image"].to(device))
        batches += 1
    for module, momentum in zip(batch_norms, original_momenta, strict=True):
        module.momentum = momentum
    return {
        "method": "reset_buffers_then_one_fixed_forward_pass_with_cumulative_batch_average",
        "label_information_used": False,
        "batch_norm_layer_count": len(batch_norms),
        "batch_count": batches,
        "batch_size": batch_size,
    }


def _save_arrays(path: Path, derived: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **derived)


def _material_effect(report: dict[str, Any], baseline: dict[str, Any]) -> bool:
    outside = int(report["violations"]["outside_half_cell_count"])
    baseline_outside = int(baseline["violations"]["outside_half_cell_count"])
    median = float(report["material_error_px"]["median"])
    baseline_median = float(baseline["material_error_px"]["median"])
    return outside <= int(np.floor(0.75 * baseline_outside)) and median <= baseline_median


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh diagnostic path")
    require(args.frame_limit == EXPECTED_FRAMES, "diagnostic requires all 180 frames")
    require(args.batch_size >= 2, "BatchNorm diagnostic batch size must be at least two")
    require(sha256_file(args.checkpoint) == args.expected_checkpoint_sha256, "checkpoint SHA-256 differs")
    require(sha256_file(args.reference_result) == args.expected_reference_result_sha256, "reference result SHA-256 differs")
    require(
        sha256_file(args.reference_predictions) == args.expected_reference_predictions_sha256,
        "reference predictions SHA-256 differs",
    )
    manifest, dataset, target_px, masks, controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        args.frame_limit,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    require(checkpoint["schema_version"] == "certified_witness_supervised_capability_checkpoint.v1", "checkpoint schema differs")
    base_model = _new_model(checkpoint["extractor_state_dict"], device)
    require(model_state_sha256(base_model) == checkpoint["model_state_sha256"], "loaded model state hash differs")
    frozen_parameter_hash = _parameter_sha256(base_model)

    start = time.perf_counter()
    baseline_report, baseline_derived = _evaluate_model(
        base_model, dataset, target_px, masks, device, args.batch_size
    )
    reference_result = json.loads(args.reference_result.read_text())
    require(
        evaluation_score(baseline_report) == evaluation_score(reference_result["evaluation"]),
        "baseline evaluation score does not replay reference result",
    )
    with np.load(args.reference_predictions) as reference_arrays:
        require(
            np.array_equal(
                baseline_derived["prediction_coordinate_px"],
                np.asarray(reference_arrays["prediction_coordinate_px"]),
            ),
            "baseline coordinates do not exactly replay saved predictions",
        )

    forward_model = _new_model(checkpoint["extractor_state_dict"], device)
    forward_report, forward_derived = _evaluate_with_batch_statistics(
        forward_model,
        dataset,
        target_px,
        masks,
        device,
        args.batch_size,
        range(len(dataset)),
    )
    require(_parameter_sha256(forward_model) == frozen_parameter_hash, "batch-stat probe changed parameters")

    reverse_model = _new_model(checkpoint["extractor_state_dict"], device)
    reverse_report, reverse_derived = _evaluate_with_batch_statistics(
        reverse_model,
        dataset,
        target_px,
        masks,
        device,
        args.batch_size,
        reversed(range(len(dataset))),
    )
    require(_parameter_sha256(reverse_model) == frozen_parameter_hash, "reverse batch-stat probe changed parameters")

    recalibrated_model = _new_model(checkpoint["extractor_state_dict"], device)
    recalibration = _recalibrate_running_statistics(
        recalibrated_model, dataset, device, args.batch_size
    )
    require(_parameter_sha256(recalibrated_model) == frozen_parameter_hash, "BN recalibration changed parameters")
    recalibrated_state_hash = model_state_sha256(recalibrated_model)
    recalibrated_report, recalibrated_derived = _evaluate_model(
        recalibrated_model, dataset, target_px, masks, device, args.batch_size
    )

    forward_reverse_delta = np.linalg.norm(
        forward_derived["prediction_coordinate_px"] - reverse_derived["prediction_coordinate_px"],
        axis=-1,
    )
    reports = {
        "saved_eval_replay": baseline_report,
        "forward_batch_statistics": forward_report,
        "reverse_batch_statistics": reverse_report,
        "recalibrated_eval": recalibrated_report,
    }
    mechanism_flags = {
        name: _material_effect(report, baseline_report)
        for name, report in reports.items()
        if name != "saved_eval_replay"
    }
    result = {
        "schema_version": "certified_witness_bn_state_diagnostic.v1",
        "artifact_type": "frozen_checkpoint_posthoc_mechanism_diagnostic",
        "seed": int(checkpoint["seed"]),
        "selected_update": int(checkpoint["update"]),
        "device": str(device),
        "runtime_seconds": time.perf_counter() - start,
        "weights_optimized": False,
        "architecture_changed": False,
        "frozen_parameter_sha256": frozen_parameter_hash,
        "saved_model_state_sha256": checkpoint["model_state_sha256"],
        "recalibrated_model_state_sha256": recalibrated_state_hash,
        "baseline_exact_prediction_replay": True,
        "material_effect_rule": {
            "outside_half_cell_reduction_at_least_fraction": 0.25,
            "outside_half_cell_count_at_most": int(np.floor(0.75 * baseline_report["violations"]["outside_half_cell_count"])),
            "median_material_error_must_not_increase": True,
            "interpretation": "mechanism flag only; never a capability pass",
        },
        "material_normalization_effect": mechanism_flags,
        "batch_stat_forward_reverse_coordinate_delta_px": _summary(forward_reverse_delta),
        "batch_stat_interpretation": "diagnostic only; each prediction depends on other images in its batch",
        "recalibration": recalibration,
        "reports": reports,
        "semantic_controls": controls,
        "decision_branch": (
            "normalization_is_active_mechanism_uncertainty_stop_before_replica_seeds"
            if any(mechanism_flags.values())
            else "no_material_normalization_effect_proceed_to_replica_seeds"
        ),
        "capability_pass_changed": False,
        "preservation_phase_authorized": False,
        "statistical_scope": {
            "inference": "descriptive_only",
            "object_count": 1,
            "orbit_count": 1,
            "optimization_seed_count": 1,
            "frame_values_independent": False,
        },
    }

    args.output_dir.mkdir(parents=True)
    arrays = {
        "saved_eval_replay": baseline_derived,
        "forward_batch_statistics": forward_derived,
        "reverse_batch_statistics": reverse_derived,
        "recalibrated_eval": recalibrated_derived,
    }
    array_records: dict[str, Any] = {}
    for name, derived in arrays.items():
        path = args.output_dir / f"{name}.npz"
        _save_arrays(path, derived)
        array_records[name] = file_record(path)
    result_path = args.output_dir / "BN_STATE_DIAGNOSTIC_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_bn_state_diagnostic_config.v1",
        "repo_root": str(args.repo_root.resolve()),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "checkpoint": file_record(args.checkpoint),
        "reference_result": file_record(args.reference_result),
        "reference_predictions": file_record(args.reference_predictions),
        "manifest_implementation_head": manifest["implementation_head"],
        "diagnostic_source": file_record(Path(__file__).resolve()),
        "frame_limit": args.frame_limit,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_bn_state_diagnostic_receipt.v1",
        "result": file_record(result_path),
        "config": file_record(config_path),
        "arrays": array_records,
        "decision_branch": result["decision_branch"],
        "preservation_phase_authorized": False,
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--expected-reference-result-sha256", required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--expected-reference-predictions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frame-limit", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"BN DIAGNOSTIC CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
