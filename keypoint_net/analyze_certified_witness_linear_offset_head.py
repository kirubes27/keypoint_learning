"""Fit and audit the frozen-feature affine per-cell offset head."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    HALF_CELL_DIAGONAL_PX,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    model_state_sha256,
    normalized_to_pixel,
    require,
    sha256_file,
)
from certified_witness_linear_offset_head import (
    FEATURE_CHANNELS,
    compact_information_boundary,
    fit_linear_offset_heads,
    predict_linear_offset_head,
    solve_affine_offset,
)
from certified_witness_local_readout import (
    category_name,
    classify_localization_failures,
    grid_to_pixel,
    readout_arrays,
)
from model import KeypointExtractor
from run_certified_witness_capability import _load_bound_inputs, _save_worst_montage


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


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


def _verify_record(record: dict[str, Any], expected_path: Path, label: str) -> None:
    require(
        Path(str(record.get("absolute_path", ""))).resolve() == expected_path.resolve(),
        f"{label} path differs from receipt",
    )
    require(expected_path.is_file(), f"{label} missing")
    require(
        sha256_file(expected_path) == record.get("sha256"),
        f"{label} SHA-256 differs from receipt",
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _infer_with_features(
    checkpoint_path: Path,
    dataset: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, str]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require(
        payload.get("schema_version")
        == "certified_witness_local_confirmation_checkpoint.v1",
        "checkpoint schema differs",
    )
    require(int(payload.get("update", -1)) == 4200, "checkpoint update differs")
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    model.load_state_dict(payload["extractor_state_dict"], strict=True)
    state_hash = model_state_sha256(model)
    require(state_hash == payload["model_state_sha256"], "model state hash differs")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    global_values: list[np.ndarray] = []
    logit_values: list[np.ndarray] = []
    feature_values: list[np.ndarray] = []
    frame_values: list[np.ndarray] = []
    for batch in loader:
        flat, logits, features = model(
            batch["image"].to(device), return_descriptor_features=True
        )
        global_values.append(flat.view(-1, EXPECTED_WITNESSES, 2).cpu().numpy())
        logit_values.append(logits.cpu().numpy())
        feature_values.append(features.cpu().numpy())
        frame_values.append(batch["frame"].numpy())
    frame_index = np.concatenate(frame_values)
    order = np.argsort(frame_index)
    require(
        np.array_equal(frame_index[order], np.arange(EXPECTED_FRAMES)),
        "inference frame order differs",
    )
    global_px = normalized_to_pixel(np.concatenate(global_values)[order])
    logits = np.concatenate(logit_values)[order]
    features = np.concatenate(feature_values)[order]
    require(model_state_sha256(model) == state_hash, "frozen model state changed during inference")
    return payload, global_px, logits, features, state_hash


def _runtime_controls(
    features: np.ndarray,
    logits: np.ndarray,
    baseline: dict[str, np.ndarray],
) -> dict[str, Any]:
    design = np.eye(FEATURE_CHANNELS + 1, dtype=np.float64)
    planted_coefficient = np.stack(
        [
            np.linspace(-0.25, 0.25, FEATURE_CHANNELS + 1),
            np.linspace(0.40, -0.40, FEATURE_CHANNELS + 1),
        ],
        axis=-1,
    )
    labels = design @ planted_coefficient
    solved, solved_report = solve_affine_offset(design, labels)
    planted_max_abs_error = float(np.max(np.abs(solved - planted_coefficient)))
    require(
        planted_max_abs_error <= 1e-12,
        "planted full-rank affine offset solver did not replay",
    )

    zero = np.zeros(
        (EXPECTED_WITNESSES, FEATURE_CHANNELS + 1, 2), dtype=np.float64
    )
    zero_prediction = predict_linear_offset_head(features, logits, zero)
    require(
        np.array_equal(zero_prediction["prediction_px"], baseline["hard_prediction_px"]),
        "zero offset head did not replay hard-cell centers",
    )

    remote_features = np.zeros(
        (1, FEATURE_CHANNELS, 64, 64), dtype=np.float64
    )
    remote_logits = np.full(
        (1, EXPECTED_WITNESSES, 64, 64), -10.0, dtype=np.float64
    )
    remote_logits[:, :, 40, 41] = 10.0
    remote_coefficient = zero.copy()
    remote_coefficient[:, 0, :] = 100.0
    remote = predict_linear_offset_head(
        remote_features, remote_logits, remote_coefficient
    )
    remote_target = np.broadcast_to(
        grid_to_pixel(11.0, 10.0), (1, EXPECTED_WITNESSES, 2)
    ).copy()
    remote_error = np.linalg.norm(remote["prediction_px"] - remote_target, axis=-1)
    require(
        bool(np.all(remote_error > HALF_CELL_DIAGONAL_PX)),
        "remote alias escaped its bounded local support",
    )
    return {
        "planted_full_rank_affine_solver": {
            "passed": True,
            "maximum_abs_coefficient_error": planted_max_abs_error,
            "solver_report": solved_report,
        },
        "zero_coefficient_head_replays_hard_cells_exactly": True,
        "remote_alias_cannot_escape_bounded_support": {
            "passed": True,
            "selected_hard_cell_xy": [41, 40],
            "maximum_grid_xy": [42.5, 41.5],
            "target_grid_xy": [11.0, 10.0],
        },
    }


def _residual_events(
    prediction: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
    target_px: np.ndarray,
    category: np.ndarray,
) -> list[dict[str, Any]]:
    residual = (
        np.logical_not(derived["within_half_cell"])
        | np.logical_not(derived["on_object"])
        | np.logical_not(derived["identity_correct"])
    )
    events: list[dict[str, Any]] = []
    for frame_value, channel_value in np.argwhere(residual):
        frame = int(frame_value)
        channel = int(channel_value)
        code = int(category[frame, channel])
        events.append(
            {
                "frame": frame,
                "channel": channel,
                "witness_id": int(EXPECTED_WITNESS_IDS[channel]),
                "localization_failed": bool(not derived["within_half_cell"][frame, channel]),
                "localization_category": category_name(code) if code else None,
                "off_object": bool(not derived["on_object"][frame, channel]),
                "wrong_identity": bool(not derived["identity_correct"][frame, channel]),
                "material_error_px": float(derived["material_error_px"][frame, channel]),
                "target_coordinate_px": target_px[frame, channel].tolist(),
                "prediction_coordinate_px": prediction["prediction_px"][frame, channel].tolist(),
                "local_expectation_coordinate_px": baseline["local_3x3_prediction_px"][frame, channel].tolist(),
                "hard_cell_xy": [
                    int(baseline["hard_cell_x"][frame, channel]),
                    int(baseline["hard_cell_y"][frame, channel]),
                ],
                "target_nearest_cell_xy": [
                    int(baseline["target_cell_x"][frame, channel]),
                    int(baseline["target_cell_y"][frame, channel]),
                ],
                "target_nearest_cell_rank": int(
                    baseline["target_nearest_cell_rank"][frame, channel]
                ),
                "target_cell_inside_hard_3x3": bool(
                    baseline["target_cell_inside_local_window"][frame, channel]
                ),
                "raw_offset_grid": prediction["raw_offset_grid"][frame, channel].tolist(),
                "bounded_offset_grid": prediction["bounded_offset_grid"][frame, channel].tolist(),
                "offset_clamp_applied": bool(
                    prediction["offset_clamp_applied"][frame, channel]
                ),
            }
        )
    return events


def _collapsed_pair_events(distinct_pair: np.ndarray) -> list[dict[str, int]]:
    pair_index = list(zip(*np.triu_indices(EXPECTED_WITNESSES, k=1)))
    events: list[dict[str, int]] = []
    for frame_value, pair_value in np.argwhere(np.logical_not(distinct_pair)):
        frame = int(frame_value)
        first, second = pair_index[int(pair_value)]
        events.append(
            {
                "frame": frame,
                "first_channel": int(first),
                "first_witness_id": int(EXPECTED_WITNESS_IDS[first]),
                "second_channel": int(second),
                "second_witness_id": int(EXPECTED_WITNESS_IDS[second]),
            }
        )
    return events


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh path")
    repository_head = _verify_repository(args)
    require(
        sha256_file(args.analysis_lock) == args.expected_analysis_lock_sha256,
        "analysis-lock SHA-256 differs",
    )
    require(
        sha256_file(args.confirmation_receipt)
        == args.expected_confirmation_receipt_sha256,
        "confirmation receipt SHA-256 differs",
    )
    require(
        sha256_file(args.gaussian_receipt) == args.expected_gaussian_receipt_sha256,
        "Gaussian-readout receipt SHA-256 differs",
    )
    confirmation_receipt = _load_json(args.confirmation_receipt)
    gaussian_receipt = _load_json(args.gaussian_receipt)
    require(
        gaussian_receipt.get("decision_branch") == "reject_analytic_readout_new_escape",
        "Gaussian diagnostic branch differs",
    )
    _verify_record(
        confirmation_receipt["selected_checkpoint"], args.checkpoint, "checkpoint"
    )
    _verify_record(
        confirmation_receipt["predictions"], args.confirmation_predictions, "predictions"
    )
    _, dataset, target_px, masks, _ = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        EXPECTED_FRAMES,
    )
    device = torch.device(args.device)
    start = time.perf_counter()
    _, global_px, logits, features, state_hash = _infer_with_features(
        args.checkpoint, dataset, device, args.batch_size
    )
    with np.load(args.confirmation_predictions) as saved:
        require(
            np.array_equal(
                global_px,
                np.asarray(saved["global_soft_prediction_px"], dtype=np.float64),
            ),
            "saved global replay differs",
        )
        require(
            np.array_equal(logits, np.asarray(saved["native_heatmap_logits"])),
            "saved heatmap-logit replay differs",
        )
        saved_local = np.asarray(saved["local_3x3_prediction_px"], dtype=np.float64)
    baseline = readout_arrays(logits, target_px)
    require(
        np.array_equal(baseline["local_3x3_prediction_px"], saved_local),
        "saved local-expectation replay differs",
    )
    require(
        bool(np.all(baseline["target_cell_inside_local_window"])),
        "a target-nearest cell lies outside the hard-centered 3x3 window",
    )
    baseline_report, _ = evaluate_predictions(saved_local, target_px, masks)
    controls = _runtime_controls(features, logits, baseline)

    coefficients, fit_reports, metadata = fit_linear_offset_heads(features, target_px)
    prediction = predict_linear_offset_head(features, logits, coefficients)
    replay = predict_linear_offset_head(features, logits, coefficients)
    for name in prediction:
        require(
            np.array_equal(prediction[name], replay[name]),
            f"linear offset prediction replay differs: {name}",
        )
    require(
        np.array_equal(prediction["hard_cell_x"], baseline["hard_cell_x"])
        and np.array_equal(prediction["hard_cell_y"], baseline["hard_cell_y"]),
        "linear head hard cells differ from baseline",
    )
    candidate_report, derived = evaluate_predictions(
        prediction["prediction_px"], target_px, masks
    )
    category, category_counts = classify_localization_failures(
        baseline, derived["within_half_cell"]
    )
    wrong_coarse_count = int(
        category_counts["wrong_coarse_mode_target_top10"]
        + category_counts["wrong_coarse_mode_target_below_top10"]
    )
    baseline_violations = baseline_report["violations"]
    candidate_violations = candidate_report["violations"]
    new_escape = any(
        int(candidate_violations[name]) > int(baseline_violations[name])
        for name in (
            "wrong_identity_count",
            "collapsed_pair_count",
            "off_object_count",
        )
    )
    strict_pass = bool(candidate_report["strict_capability_pass"])
    decision_branch = (
        "strict_supervised_linear_offset_capability_requires_fresh_confirmation"
        if strict_pass
        else "wrong_coarse_residual_stop_offset_work"
        if wrong_coarse_count
        else "reject_linear_offset_head_new_escape"
        if new_escape
        else "linear_offset_near_fail_inspect_before_nonlinear_head"
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "LINEAR_OFFSET_HEAD_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        coefficients=coefficients,
        target_coordinate_px=target_px,
        global_soft_prediction_px=global_px,
        hard_cell_x=prediction["hard_cell_x"],
        hard_cell_y=prediction["hard_cell_y"],
        local_expectation_prediction_px=saved_local,
        linear_offset_prediction_px=prediction["prediction_px"],
        raw_offset_grid=prediction["raw_offset_grid"],
        bounded_offset_grid=prediction["bounded_offset_grid"],
        offset_clamp_applied=prediction["offset_clamp_applied"],
        image_clamp_applied=prediction["image_clamp_applied"],
        material_error_px=derived["material_error_px"],
        within_half_cell=derived["within_half_cell"],
        on_object=derived["on_object"],
        identity_correct=derived["identity_correct"],
        distinct_pair=derived["distinct_pair"],
        localization_category_code=category,
    )
    with np.load(arrays_path) as saved_arrays:
        require(
            np.array_equal(saved_arrays["coefficients"], coefficients),
            "saved coefficient replay differs",
        )
        require(
            np.array_equal(
                saved_arrays["linear_offset_prediction_px"],
                prediction["prediction_px"],
            ),
            "saved prediction replay differs",
        )
    visual_path = args.output_dir / "LINEAR_OFFSET_HEAD_WORST.png"
    _save_worst_montage(
        dataset.images,
        prediction["prediction_px"],
        target_px,
        derived["material_error_px"],
        visual_path,
    )
    per_witness = [
        {
            "channel": channel,
            "witness_id": int(EXPECTED_WITNESS_IDS[channel]),
            "fit": fit_reports[channel],
            "supervised_anchor_example_count": int(metadata[channel]["frame"].size),
            "localization_failure_count": int(
                np.logical_not(derived["within_half_cell"][:, channel]).sum()
            ),
            "off_object_count": int(
                np.logical_not(derived["on_object"][:, channel]).sum()
            ),
            "wrong_identity_count": int(
                np.logical_not(derived["identity_correct"][:, channel]).sum()
            ),
        }
        for channel in range(EXPECTED_WITNESSES)
    ]
    result = {
        "schema_version": "certified_witness_linear_offset_head_diagnostic.v1",
        "artifact_type": "source_bound_frozen_feature_supervised_linear_offset_head",
        "extractor_weights_optimized": False,
        "extractor_batchnorm_state_changed": False,
        "linear_readout_coefficients_fitted": True,
        "gradient_optimizer_used": False,
        "information_boundary": compact_information_boundary(),
        "model_state_sha256_before_and_after": state_hash,
        "encoder_features_sha256": _array_sha256(features),
        "global_heatmap_and_local_expectation_replay_exact": True,
        "all_target_cells_inside_hard_centered_3x3": True,
        "target_cell_inside_count": int(
            baseline["target_cell_inside_local_window"].sum()
        ),
        "controls": controls,
        "baseline_local_expectation": _compact_report(baseline_report),
        "linear_offset_candidate": _compact_report(candidate_report),
        "localization_category_counts": category_counts,
        "offset_clamp_applied_count": int(prediction["offset_clamp_applied"].sum()),
        "image_clamp_applied_count": int(prediction["image_clamp_applied"].sum()),
        "per_witness": per_witness,
        "residual_events": _residual_events(
            prediction, baseline, derived, target_px, category
        ),
        "collapsed_pair_events": _collapsed_pair_events(derived["distinct_pair"]),
        "strict_supervised_linear_offset_capability_pass": strict_pass,
        "wrong_coarse_residual_count": wrong_coarse_count,
        "new_identity_collapse_or_offobject_escape": new_escape,
        "decision_branch": decision_branch,
        "fresh_confirmation_required_for_capability_claim": strict_pass,
        "unsupervised_discovery_established": False,
        "runtime_seconds": time.perf_counter() - start,
        "statistical_scope": {
            "inference": "descriptive_only",
            "fit_and_evaluation_frames_are_the_same": True,
            "optimization_seed_count": 1,
            "object_count": 1,
            "orbit_count": 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    result_path = args.output_dir / "LINEAR_OFFSET_HEAD_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_linear_offset_head_config.v1",
        "repository_head": repository_head,
        "analysis_source": file_record(Path(__file__)),
        "offset_head_source": file_record(
            args.repo_root / "keypoint_net" / "certified_witness_linear_offset_head.py"
        ),
        "analysis_lock": file_record(args.analysis_lock),
        "confirmation_receipt": file_record(args.confirmation_receipt),
        "gaussian_receipt": file_record(args.gaussian_receipt),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "checkpoint": file_record(args.checkpoint),
        "confirmation_predictions": file_record(args.confirmation_predictions),
        "device": str(device),
        "batch_size": args.batch_size,
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_linear_offset_head_receipt.v1",
        "result": file_record(result_path),
        "arrays_and_coefficients": file_record(arrays_path),
        "config": file_record(config_path),
        "visual": file_record(visual_path),
        "strict_supervised_linear_offset_capability_pass": strict_pass,
        "decision_branch": decision_branch,
        "extractor_weights_optimized": False,
        "linear_readout_coefficients_fitted": True,
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--expected-analysis-lock-sha256", required=True)
    parser.add_argument("--confirmation-receipt", type=Path, required=True)
    parser.add_argument("--expected-confirmation-receipt-sha256", required=True)
    parser.add_argument("--gaussian-receipt", type=Path, required=True)
    parser.add_argument("--expected-gaussian-receipt-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"LINEAR OFFSET HEAD FAILURE: {error}") from error


if __name__ == "__main__":
    main()
