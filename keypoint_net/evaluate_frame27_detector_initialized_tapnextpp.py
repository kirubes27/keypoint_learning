"""Evaluate detector-once, TAPNext++-thereafter tracks after raw hashing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from certified_witness_capability import evaluate_predictions
from evaluate_frame27_anchored_tapnextpp import (
    EXPECTED_FRAMES,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MASK_MANIFEST_SHA256,
    EXPECTED_VALIDATION_TRUTH_SHA256,
    EXPECTED_WITNESSES,
    HALF_CELL_DIAGONAL_PX,
    _git,
    _load_masks,
    _load_targets,
    _render_worst,
    visibility_diagnostic,
)
from material_transport_gate_io import (
    file_record,
    load_json,
    require,
    resolve_rgb_paths,
    write_json,
)
from run_frame27_detector_initialized_tapnextpp import (
    ANCHOR_FRAME,
    EXPECTED_DETECTOR_CHECKPOINT_SHA256,
    EXPECTED_DETECTOR_PREDICTIONS_SHA256,
    EXPECTED_DETECTOR_RECEIPT_SHA256,
    EXPECTED_WITNESS_IDS,
    select_detector_anchor,
)


EXPECTED_LOCK_SHA256 = (
    "92db01963af3502cee29395221b418e65fc8e4b86ed901d6e2159b402607dd12"
)
EXPECTED_RAW_SCHEMA = (
    "raw_frame27_detector_initialized_tapnextpp_512_support64.v1"
)
RASTER_BOUNDARY_MAXIMUM_DISTANCE_PX = 1.0


def distance_to_object_mask_px(
    prediction_px: Any, masks: Any
) -> np.ndarray:
    prediction = np.asarray(prediction_px, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=bool)
    require(
        prediction.ndim == 3 and prediction.shape[1:] == (EXPECTED_WITNESSES, 2),
        "prediction shape differs for mask distance",
    )
    require(
        mask_array.shape == (prediction.shape[0], 512, 512),
        "mask shape differs for mask distance",
    )
    rounded = np.rint(prediction).astype(np.int64)
    in_image = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < 512)
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < 512)
    )
    clipped = np.clip(rounded, 0, 511)
    distance = np.full(in_image.shape, np.inf, dtype=np.float64)
    for frame in range(prediction.shape[0]):
        frame_distance = distance_transform_edt(~mask_array[frame])
        distance[frame] = frame_distance[
            clipped[frame, :, 1], clipped[frame, :, 0]
        ]
    distance[~in_image] = np.inf
    return distance


def select_bridge_branch(
    report: dict[str, Any], mask_distance_px: Any
) -> str:
    if bool(report["strict_capability_pass"]):
        return "strict_practical_hammer_bridge"
    violations = report["violations"]
    distance = np.asarray(mask_distance_px, dtype=np.float64)
    qualified = (
        int(violations["outside_half_cell_count"]) == 0
        and int(violations["wrong_identity_count"]) == 0
        and int(violations["collapsed_pair_count"]) == 0
        and bool(np.isfinite(distance).all())
        and float(distance.max()) <= RASTER_BOUNDARY_MAXIMUM_DISTANCE_PX + 1e-12
    )
    if qualified:
        return "raster_boundary_qualified_material_bridge"
    return "detector_initialization_failure_distil_selector"


def _detector_initialization_evidence(
    source_path: Path, raw_anchor: np.ndarray
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as archive:
        require(
            {
                "frame_index",
                "local_3x3_prediction_px",
                "target_coordinate_px",
            }.issubset(archive.files),
            "detector source omits post-hash initialization evidence",
        )
        frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        local_prediction = np.asarray(
            archive["local_3x3_prediction_px"], dtype=np.float64
        )
        target = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    selected = np.flatnonzero(frame_index == ANCHOR_FRAME)
    require(selected.size == 1, "detector source frame 27 is not unique")
    anchor = select_detector_anchor(frame_index, local_prediction)
    require(np.array_equal(anchor, raw_anchor), "raw detector anchor differs from source")
    target_anchor = target[int(selected[0])]
    require(target_anchor.shape == anchor.shape, "detector target anchor shape differs")
    error = np.linalg.norm(anchor.astype(np.float64) - target_anchor, axis=-1)
    return {
        "frame_index": ANCHOR_FRAME,
        "error_px": error.tolist(),
        "mean_error_px": float(error.mean()),
        "maximum_error_px": float(error.max()),
        "all_within_half_cell": bool((error <= HALF_CELL_DIAGONAL_PX + 1e-12).all()),
        "training_target_opened_only_after_raw_prediction_hash": True,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")

    # Bind and load the raw tracks before any target or mask array is opened.
    raw_receipt_record = file_record(args.raw_receipt)
    raw_receipt = load_json(args.raw_receipt)
    require(raw_receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw schema differs")
    require(
        raw_receipt.get("privileged_evaluation_authorized") is True,
        "raw stage blocks evaluation",
    )
    raw_arrays_path = Path(str(raw_receipt["raw_predictions"]["absolute_path"]))
    require(
        file_record(raw_arrays_path) == raw_receipt["raw_predictions"],
        "raw prediction binding differs",
    )
    with np.load(raw_arrays_path, allow_pickle=False) as loaded:
        raw = {name: np.asarray(loaded[name]) for name in loaded.files}
    raw_loaded_before_privileged_inputs = True

    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(implementation_head == args.expected_repo_head, "implementation HEAD differs")
    raw_head = str(raw_receipt.get("implementation_head", ""))
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", raw_head, implementation_head],
        cwd=args.repo_root,
        check=False,
    )
    require(ancestry.returncode == 0, "raw implementation is not an evaluator ancestor")

    lock_record = file_record(args.semantic_lock)
    manifest_record = file_record(args.manifest)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic lock differs")
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "RGB manifest differs")
    require(raw_receipt["sources"]["semantic_lock"] == lock_record, "raw lock binding differs")
    require(
        raw_receipt["sources"]["sanitized_manifest"] == manifest_record,
        "raw manifest binding differs",
    )
    detector_prediction_record = raw_receipt["sources"]["detector_training_predictions"]
    detector_receipt_record = raw_receipt["sources"]["detector_training_receipt"]
    detector_checkpoint_record = raw_receipt["sources"]["detector_checkpoint"]
    require(
        detector_prediction_record["sha256"] == EXPECTED_DETECTOR_PREDICTIONS_SHA256,
        "raw detector prediction hash differs",
    )
    require(
        detector_receipt_record["sha256"] == EXPECTED_DETECTOR_RECEIPT_SHA256,
        "raw detector receipt hash differs",
    )
    require(
        detector_checkpoint_record["sha256"] == EXPECTED_DETECTOR_CHECKPOINT_SHA256,
        "raw detector checkpoint hash differs",
    )
    for label, record in raw_receipt["implementation_sources"].items():
        require(
            file_record(Path(str(record["absolute_path"]))) == record,
            f"raw source differs: {label}",
        )
    controls = raw_receipt["controls"]
    require(controls["frame0_to_23_truth_opened"] is False, "raw stage opened validation truth")
    require(controls["supplied_masks_opened"] is False, "raw stage opened masks")
    require(
        controls["detector_target_coordinate_array_read"] is False,
        "raw stage read detector target coordinates",
    )
    require(controls["detector_logits_or_hard_peaks_read"] is False, "raw stage read forbidden detector arrays")
    require(controls["local_laptop_gpu_used"] is False, "raw stage used laptop GPU")
    require(controls["training_or_weight_update_performed"] is False, "raw stage trained")
    require(raw_receipt["traversal"]["order"] == list(range(27, -1, -1)), "raw traversal differs")
    require(raw_receipt["traversal"]["rgb_file_open_order"] == list(range(27, -1, -1)), "RGB open order differs")
    require(raw_receipt["traversal"]["continuous_without_reset"] is True, "raw traversal reset")

    require(np.array_equal(raw["frame_index"], np.arange(28)), "raw frame index differs")
    require(tuple(raw["witness_id"].tolist()) == tuple(EXPECTED_WITNESS_IDS), "raw witness order differs")
    raw_anchor = np.asarray(raw["detector_anchor_coordinate_px"], dtype=np.float32)
    require(raw_anchor.shape == (EXPECTED_WITNESSES, 2), "raw anchor shape differs")

    detector_source_path = Path(str(detector_prediction_record["absolute_path"]))
    require(file_record(detector_source_path) == detector_prediction_record, "detector source changed")
    initialization = _detector_initialization_evidence(detector_source_path, raw_anchor)
    require(initialization["all_within_half_cell"], "locked detector initialization evidence differs")

    truth_record = file_record(args.validation_truth)
    mask_record = file_record(args.mask_manifest)
    require(truth_record["sha256"] == EXPECTED_VALIDATION_TRUTH_SHA256, "validation truth differs")
    require(mask_record["sha256"] == EXPECTED_MASK_MANIFEST_SHA256, "mask manifest differs")
    frame_index, witness_id, target_px = _load_targets(args.validation_truth)
    require(np.array_equal(frame_index, EXPECTED_FRAMES), "validation frames differ")
    object_root = args.object_root.resolve(strict=True)
    masks = _load_masks(args.mask_manifest, object_root, frame_index)
    rgb_paths = resolve_rgb_paths(load_json(args.manifest), object_root_override=object_root)

    prediction_px = np.asarray(raw["prediction_xy"][:24], dtype=np.float64)
    visible = np.asarray(raw["visible"][:24], dtype=bool)
    require(prediction_px.shape == target_px.shape, "prediction shape differs")
    require(visible.shape == prediction_px.shape[:2], "visibility shape differs")

    report, derived = evaluate_predictions(prediction_px, target_px, masks)
    report["statistical_scope"] = {
        "inference": "descriptive_only",
        "sample_unit": "fixed_witness_event_over_one_24_frame_correlated_heldout_wedge",
        "frame_values_independent": False,
        "sem_or_confidence_interval_computed": False,
    }
    mask_distance = distance_to_object_mask_px(prediction_px, masks)
    off_object = ~derived["on_object"]
    off_distance = mask_distance[off_object]
    mask_boundary = {
        "off_object_event_count": int(off_object.sum()),
        "maximum_off_object_distance_to_mask_px": float(off_distance.max())
        if off_distance.size
        else 0.0,
        "all_off_object_events_within_one_pixel": bool(
            (off_distance <= RASTER_BOUNDARY_MAXIMUM_DISTANCE_PX + 1e-12).all()
        ),
        "maximum_qualifying_distance_px": RASTER_BOUNDARY_MAXIMUM_DISTANCE_PX,
        "distance_statistic": "Euclidean distance between rounded prediction pixel and nearest true binary-mask pixel",
    }
    branch = select_bridge_branch(report, mask_distance)
    visibility = visibility_diagnostic(
        visible, derived["within_half_cell"], derived["material_error_px"]
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "FRAME27_DETECTOR_INITIALIZED_TAPNEXTPP_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        target_coordinate_px=target_px,
        prediction_xy=prediction_px,
        visible=visible,
        material_error_px=derived["material_error_px"],
        within_half_cell=derived["within_half_cell"],
        identity_correct=derived["identity_correct"],
        assigned_identity=derived["assigned_identity"],
        on_object=derived["on_object"],
        distance_to_object_mask_px=mask_distance,
        distinct_pair=derived["distinct_pair"],
        prediction_pair_distance_px=derived["prediction_pair_distance_px"],
        target_pair_distance_px=derived["target_pair_distance_px"],
    )
    worst_path = args.output_dir / "01_WORST_DETECTOR_INITIALIZED_EVENTS.png"
    worst_events = _render_worst(
        rgb_paths=rgb_paths,
        truth=target_px,
        prediction=prediction_px,
        visible=visible,
        error=derived["material_error_px"],
        witness_id=witness_id,
        output_path=worst_path,
    )
    result_path = args.output_dir / "FRAME27_DETECTOR_INITIALIZED_TAPNEXTPP_RESULT.json"
    result = {
        "schema_version": "frame27_detector_initialized_tapnextpp_evaluation.v1",
        "artifact_type": "privileged_posthash_detector_initialized_track_evaluation",
        "implementation_head": implementation_head,
        "raw_implementation_head": raw_head,
        "raw_loaded_and_hash_verified_before_truth_masks_or_training_target": raw_loaded_before_privileged_inputs,
        "sample_scope": {
            "object_count": 1,
            "heldout_frame_count": 24,
            "witness_count": 10,
            "event_count": 240,
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
            "cross_object_or_full_orbit_generalization_authorized": False,
        },
        "detector_initialization": initialization,
        "material_result": report,
        "mask_boundary_diagnostic": mask_boundary,
        "visibility_diagnostic": visibility,
        "decision": {
            "branch": branch,
            "strict_practical_hammer_bridge": branch == "strict_practical_hammer_bridge",
            "raster_boundary_qualified_material_bridge": branch
            == "raster_boundary_qualified_material_bridge",
            "operator_authorized": False,
            "training_authorized": False,
            "gpu_run_authorized": False,
            "next_if_bridge": "run one longer-horizon and second-object falsifier before broad operator use",
            "next_if_initialization_failure": "distil tracker trajectories into the selector; do not first adapt the tracker",
        },
        "thresholds": {
            "strict_half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "raster_boundary_qualification_px": RASTER_BOUNDARY_MAXIMUM_DISTANCE_PX,
        },
        "worst_visual_events": worst_events,
        "controls": {
            "continuous_raw_coordinates_evaluated_without_snapping": True,
            "visibility_used_to_hide_predictions": False,
            "training_or_weight_update_performed": False,
            "laptop_gpu_used": False,
            "validation_truth_opened_only_after_raw_prediction_hash": True,
            "detector_training_target_opened_only_after_raw_prediction_hash": True,
        },
        "bindings": {
            "semantic_lock": lock_record,
            "raw_receipt": raw_receipt_record,
            "raw_predictions": file_record(raw_arrays_path),
            "detector_training_predictions": detector_prediction_record,
            "validation_truth": truth_record,
            "mask_manifest": mask_record,
            "rgb_manifest": manifest_record,
            "evaluation_arrays": file_record(arrays_path),
            "worst_figure": file_record(worst_path),
        },
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "shared_evaluation_helpers": file_record(
                Path(__file__).with_name("evaluate_frame27_anchored_tapnextpp.py")
            ),
            "runner_helpers": file_record(
                Path(__file__).with_name("run_frame27_detector_initialized_tapnextpp.py")
            ),
            "capability_contract": file_record(
                Path(__file__).with_name("certified_witness_capability.py")
            ),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "FRAME27_DETECTOR_INITIALIZED_TAPNEXTPP_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "frame27_detector_initialized_tapnextpp_evaluation_receipt.v1",
        "implementation_head": implementation_head,
        "raw_implementation_head": raw_head,
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "worst_figure": file_record(worst_path),
        "decision_branch": branch,
        "strict_practical_hammer_bridge": branch == "strict_practical_hammer_bridge",
        "raster_boundary_qualified_material_bridge": branch
        == "raster_boundary_qualified_material_bridge",
        "operator_authorized": False,
        "training_authorized": False,
        "command_argv": list(sys.argv),
    }
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-truth", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
