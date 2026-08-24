"""Compare detector-initialized tracks with the certified-anchor control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from material_transport_gate_io import file_record, require, write_json
from run_frame27_anchored_tapnextpp import EXPECTED_WITNESS_IDS


EXPECTED_LOCK_SHA256 = (
    "8b9a9e0cb0871fb37d706fe923de0b55a726c4399500ed92936edfe5cb6f1432"
)
EXPECTED_DETECTOR_SOURCE_SHA256 = (
    "ab5f3fb46ff0d7187a88fe06b522d55a3f560ac2927e877f081d17062a270301"
)
EXPECTED_CERTIFIED_ARRAYS_SHA256 = (
    "28a709f2e4694bf3a73969eac9abad6838c87ec5f2a58884f17d74e68295b3c5"
)
EXPECTED_DETECTOR_ARRAYS_SHA256 = (
    "703be9111822d102b4e40425acaf19c916d7b3a6c63ee8948404225121be5d04"
)
ANCHOR_FRAME = 27
HELDOUT_FRAMES = np.arange(24, dtype=np.int64)


def descriptive_linear_slope(x: Any, y: Any) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    require(
        x_array.ndim == y_array.ndim == 1 and x_array.size == y_array.size,
        "slope arrays differ",
    )
    require(x_array.size >= 2, "slope requires at least two values")
    require(bool(np.isfinite(x_array).all() and np.isfinite(y_array).all()), "slope input invalid")
    require(float(np.ptp(x_array)) > 0.0, "slope x values are constant")
    return float(np.polyfit(x_array, y_array, 1)[0])


def compute_audit(
    *,
    frame_index: Any,
    witness_id: Any,
    detector_initial_prediction: Any,
    detector_initial_target: Any,
    certified_prediction: Any,
    detector_prediction: Any,
    certified_material_error: Any,
    detector_material_error: Any,
) -> dict[str, Any]:
    frames = np.asarray(frame_index, dtype=np.int64)
    witnesses = np.asarray(witness_id, dtype=np.int64)
    initial_prediction = np.asarray(detector_initial_prediction, dtype=np.float64)
    initial_target = np.asarray(detector_initial_target, dtype=np.float64)
    certified = np.asarray(certified_prediction, dtype=np.float64)
    detector = np.asarray(detector_prediction, dtype=np.float64)
    certified_error = np.asarray(certified_material_error, dtype=np.float64)
    detector_error = np.asarray(detector_material_error, dtype=np.float64)
    require(np.array_equal(frames, HELDOUT_FRAMES), "held-out frame order differs")
    require(tuple(witnesses.tolist()) == tuple(EXPECTED_WITNESS_IDS), "witness order differs")
    require(initial_prediction.shape == initial_target.shape == (10, 2), "initialization shape differs")
    require(certified.shape == detector.shape == (24, 10, 2), "track shape differs")
    require(certified_error.shape == detector_error.shape == (24, 10), "error shape differs")
    values = (
        initial_prediction,
        initial_target,
        certified,
        detector,
        certified_error,
        detector_error,
    )
    require(all(bool(np.isfinite(value).all()) for value in values), "audit input is non-finite")

    initialization_error = np.linalg.norm(initial_prediction - initial_target, axis=-1)
    inter_arm_distance = np.linalg.norm(detector - certified, axis=-1)
    per_witness_inter_arm_mean = inter_arm_distance.mean(axis=0)
    require(float(np.std(initialization_error)) > 0.0, "initialization errors are constant")
    require(float(np.std(per_witness_inter_arm_mean)) > 0.0, "inter-arm means are constant")
    correlation = float(
        np.corrcoef(initialization_error, per_witness_inter_arm_mean)[0, 1]
    )
    backward_distance = ANCHOR_FRAME - frames
    frame_inter_arm_mean = inter_arm_distance.mean(axis=1)
    frame_detector_error_mean = detector_error.mean(axis=1)
    frame_certified_error_mean = certified_error.mean(axis=1)

    return {
        "schema_version": "frame27_detector_initialization_vs_drift_audit.v1",
        "artifact_type": "posthash_descriptive_mechanism_audit",
        "sample_scope": {
            "fixed_witness_count": 10,
            "correlated_frame_count": 24,
            "event_count": 240,
            "statistics": "descriptive only; no hypothesis test, SEM, confidence interval, or population claim",
        },
        "initialization_error_px": {
            "per_witness": initialization_error.tolist(),
            "mean": float(initialization_error.mean()),
            "maximum": float(initialization_error.max()),
        },
        "inter_arm_distance_px": {
            "per_witness_mean_over_frames": per_witness_inter_arm_mean.tolist(),
            "per_frame_mean_over_witnesses": frame_inter_arm_mean.tolist(),
            "overall_mean": float(inter_arm_distance.mean()),
            "frame0_mean": float(frame_inter_arm_mean[0]),
            "frame23_mean": float(frame_inter_arm_mean[23]),
            "slope_per_backward_frame": descriptive_linear_slope(
                backward_distance, frame_inter_arm_mean
            ),
        },
        "detector_initialized_material_error_px": {
            "per_witness_mean": detector_error.mean(axis=0).tolist(),
            "per_frame_mean": frame_detector_error_mean.tolist(),
            "overall_mean": float(detector_error.mean()),
            "maximum": float(detector_error.max()),
            "slope_per_backward_frame": descriptive_linear_slope(
                backward_distance, frame_detector_error_mean
            ),
        },
        "certified_anchor_material_error_px": {
            "per_witness_mean": certified_error.mean(axis=0).tolist(),
            "per_frame_mean": frame_certified_error_mean.tolist(),
            "overall_mean": float(certified_error.mean()),
            "maximum": float(certified_error.max()),
            "slope_per_backward_frame": descriptive_linear_slope(
                backward_distance, frame_certified_error_mean
            ),
        },
        "initialization_error_vs_mean_inter_arm_distance": {
            "statistic": "Pearson correlation across ten fixed witnesses",
            "n": 10,
            "value": correlation,
            "inferential": False,
        },
        "frame_index": frames.tolist(),
        "backward_temporal_distance": backward_distance.tolist(),
        "witness_id": witnesses.tolist(),
        "interpretation_limits": {
            "proves_no_tracker_drift": False,
            "changes_historical_gate_branch": False,
            "authorizes_cross_object_or_full_orbit_claim": False,
            "authorizes_operator_or_training": False,
        },
    }


def _render(result: dict[str, Any], output_path: Path) -> None:
    initial = np.asarray(result["initialization_error_px"]["per_witness"])
    inter_witness = np.asarray(
        result["inter_arm_distance_px"]["per_witness_mean_over_frames"]
    )
    backward = np.asarray(result["backward_temporal_distance"])
    inter_frame = np.asarray(
        result["inter_arm_distance_px"]["per_frame_mean_over_witnesses"]
    )
    detector_frame = np.asarray(
        result["detector_initialized_material_error_px"]["per_frame_mean"]
    )
    certified_frame = np.asarray(
        result["certified_anchor_material_error_px"]["per_frame_mean"]
    )
    witness_id = result["witness_id"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(initial, inter_witness, color="#6a3d9a", s=55)
    limit = float(max(initial.max(), inter_witness.max()) * 1.08)
    axes[0].plot([0.0, limit], [0.0, limit], "--", color="0.5", label="equal magnitude")
    for x_value, y_value, label in zip(initial, inter_witness, witness_id):
        axes[0].annotate(str(label), (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[0].set(xlim=(0.0, limit), ylim=(0.0, limit))
    axes[0].set_xlabel("frame-27 detector initialization error (px)")
    axes[0].set_ylabel("mean detector-vs-certified track distance (px)")
    axes[0].set_title(
        "Initialization magnitude predicts carried offset\n"
        f"descriptive r={result['initialization_error_vs_mean_inter_arm_distance']['value']:.3f}, n=10"
    )
    axes[0].legend(frameon=False)

    order = np.argsort(backward)
    axes[1].plot(backward[order], inter_frame[order], "o-", label="inter-arm distance")
    axes[1].plot(backward[order], detector_frame[order], "o-", label="detector-init error")
    axes[1].plot(backward[order], certified_frame[order], "o-", label="certified-init error")
    axes[1].set_xlabel("backward temporal distance from frame 27")
    axes[1].set_ylabel("mean over ten fixed witnesses (px)")
    axes[1].set_title(
        "No material growth in the arm separation\n"
        f"inter-arm slope={result['inter_arm_distance_px']['slope_per_backward_frame']:.4f} px/frame"
    )
    axes[1].legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    lock_record = file_record(args.audit_lock)
    detector_source_record = file_record(args.detector_training_predictions)
    certified_record = file_record(args.certified_evaluation_arrays)
    detector_record = file_record(args.detector_evaluation_arrays)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "audit lock differs")
    require(detector_source_record["sha256"] == EXPECTED_DETECTOR_SOURCE_SHA256, "detector source differs")
    require(certified_record["sha256"] == EXPECTED_CERTIFIED_ARRAYS_SHA256, "certified arrays differ")
    require(detector_record["sha256"] == EXPECTED_DETECTOR_ARRAYS_SHA256, "detector arrays differ")

    with np.load(args.detector_training_predictions, allow_pickle=False) as source:
        source_frames = np.asarray(source["frame_index"], dtype=np.int64)
        selected = np.flatnonzero(source_frames == ANCHOR_FRAME)
        require(selected.size == 1, "source frame 27 is not unique")
        source_index = int(selected[0])
        initial_prediction = np.asarray(
            source["local_3x3_prediction_px"][source_index], dtype=np.float64
        )
        initial_target = np.asarray(
            source["target_coordinate_px"][source_index], dtype=np.float64
        )
    with np.load(args.certified_evaluation_arrays, allow_pickle=False) as source:
        certified = {name: np.asarray(source[name]) for name in source.files}
    with np.load(args.detector_evaluation_arrays, allow_pickle=False) as source:
        detector = {name: np.asarray(source[name]) for name in source.files}
    require(np.array_equal(certified["frame_index"], detector["frame_index"]), "arm frame order differs")
    require(np.array_equal(certified["witness_id"], detector["witness_id"]), "arm witness order differs")

    result = compute_audit(
        frame_index=detector["frame_index"],
        witness_id=detector["witness_id"],
        detector_initial_prediction=initial_prediction,
        detector_initial_target=initial_target,
        certified_prediction=certified["prediction_xy"],
        detector_prediction=detector["prediction_xy"],
        certified_material_error=certified["material_error_px"],
        detector_material_error=detector["material_error_px"],
    )
    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "01_INITIALIZATION_VS_TEMPORAL_OFFSET.png"
    _render(result, figure_path)
    result["bindings"] = {
        "audit_lock": lock_record,
        "detector_training_predictions": detector_source_record,
        "certified_evaluation_arrays": certified_record,
        "detector_evaluation_arrays": detector_record,
        "figure": file_record(figure_path),
    }
    result["implementation_source"] = file_record(Path(__file__))
    result_path = args.output_dir / "INITIALIZATION_VS_DRIFT_AUDIT_RESULT.json"
    write_json(result_path, result)
    receipt_path = args.output_dir / "INITIALIZATION_VS_DRIFT_AUDIT_RECEIPT.json"
    receipt = {
        "schema_version": "frame27_detector_initialization_vs_drift_audit_receipt.v1",
        "result": file_record(result_path),
        "figure": file_record(figure_path),
        "descriptive_pearson": result["initialization_error_vs_mean_inter_arm_distance"]["value"],
        "inter_arm_slope_px_per_backward_frame": result["inter_arm_distance_px"]["slope_per_backward_frame"],
        "historical_branch_changed": False,
        "operator_or_training_authorized": False,
        "command_argv": list(sys.argv),
    }
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-lock", required=True, type=Path)
    parser.add_argument("--detector-training-predictions", required=True, type=Path)
    parser.add_argument("--certified-evaluation-arrays", required=True, type=Path)
    parser.add_argument("--detector-evaluation-arrays", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
