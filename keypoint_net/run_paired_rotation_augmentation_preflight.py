"""Fail-closed geometry, leakage, and prior-grounding preflight."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from certified_witness_capability import (
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    require,
    sha256_file,
)
from leakage_safe_distillation_contract import (
    SEMANTIC_LOCK_SHA256,
    TRAIN_FRAMES,
    load_bound_training_dataset,
    load_json,
    require_local_cpu_only,
    verify_clean_repository,
    verify_exact_file,
    verify_record,
    verify_semantic_lock,
    write_json,
)
from paired_rotation_augmentation import (
    BACKGROUND_RGB_UINT8,
    MEAN,
    ROTATION_CENTER_PX,
    STD,
    apply_arm_transform,
    pixel_to_normalized_torch,
    transform_points_px,
    warp_masks,
    warp_normalized_images_and_targets,
)


PREVIOUS_TRAINING_PREDICTIONS_SHA256 = (
    "e57d672a49fa7f988fe8a0ec7437f84c5d00889dc849ea0f5d72cc86bea2ef8f"
)
PREVIOUS_EVALUATION_RESULT_SHA256 = (
    "0ee5d54090a410b6a15998d8d629ba36756c58f4a150e5c44a804d13410c76e9"
)
EXPECTED_OFF_OBJECT_BY_CHANNEL = {0: 18, 2: 5, 5: 3, 6: 2}
TEST_ANGLES_DEG = (-137.5, -90.0, -37.25, 37.25, 90.0, 137.5)


def _summary(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "invalid summary")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    require(union > 0, "mask IoU has empty union")
    return float(np.count_nonzero(left & right) / union)


def _dataset_batch(dataset: Any, indices: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [dataset[index] for index in indices]
    return (
        torch.stack([row["image"] for row in rows]),
        torch.stack([row["target"] for row in rows]),
    )


def _off_object_audit(
    prediction_path: Path,
    target_px: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    with np.load(prediction_path, allow_pickle=False) as loaded:
        frame_index = np.asarray(loaded["frame_index"], dtype=np.int64)
        prediction = np.asarray(loaded["local_3x3_prediction_px"], dtype=np.float64)
        recorded_target = np.asarray(loaded["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(frame_index, TRAIN_FRAMES), "prior prediction frames differ")
    require(np.array_equal(recorded_target, target_px), "prior prediction targets differ")
    report, derived = evaluate_predictions(prediction, target_px, masks)
    require(
        int(report["violations"]["off_object_count"]) == 28,
        "prior off-object count did not replay",
    )
    off = np.logical_not(np.asarray(derived["on_object"], dtype=bool))
    events: list[dict[str, Any]] = []
    for row, channel in np.argwhere(off).tolist():
        xy = np.argwhere(masks[row])[:, ::-1]
        distance, nearest = cKDTree(xy).query(prediction[row, channel])
        events.append(
            {
                "frame_index": int(TRAIN_FRAMES[row]),
                "channel": int(channel),
                "material_error_px": float(
                    np.linalg.norm(prediction[row, channel] - target_px[row, channel])
                ),
                "nearest_foreground_pixel_center_distance_px": float(distance),
                "nearest_foreground_xy": xy[int(nearest)].tolist(),
            }
        )
    by_channel = {
        channel: sum(event["channel"] == channel for event in events)
        for channel in sorted({event["channel"] for event in events})
    }
    require(by_channel == EXPECTED_OFF_OBJECT_BY_CHANNEL, "prior off-object channels differ")
    distances = np.asarray(
        [event["nearest_foreground_pixel_center_distance_px"] for event in events]
    )
    errors = np.asarray([event["material_error_px"] for event in events])
    return {
        "count": len(events),
        "by_channel": {str(key): value for key, value in by_channel.items()},
        "nearest_foreground_pixel_center_distance_px": _summary(distances),
        "material_error_px": _summary(errors),
        "all_within_1_5_px_of_foreground": bool((distances <= 1.5).all()),
        "events": events,
    }


def _radial_and_background_audit(dataset: Any, target_px: np.ndarray) -> dict[str, Any]:
    center = np.asarray(ROTATION_CENTER_PX, dtype=np.float64)
    target_radius = np.linalg.norm(target_px - center, axis=-1)
    mask_maximum = 0.0
    for mask in dataset.masks:
        foreground_xy = np.argwhere(mask)[:, ::-1]
        mask_maximum = max(
            mask_maximum,
            float(np.linalg.norm(foreground_xy - center, axis=-1).max()),
        )
    corners = np.concatenate(
        (
            dataset.images[:, 0, 0],
            dataset.images[:, 0, -1],
            dataset.images[:, -1, 0],
            dataset.images[:, -1, -1],
        ),
        axis=0,
    )
    expected_background = np.asarray(BACKGROUND_RGB_UINT8, dtype=np.uint8)
    return {
        "maximum_target_radius_px": float(target_radius.max()),
        "target_radial_clearance_px": float(255.5 - target_radius.max()),
        "maximum_foreground_radius_px": mask_maximum,
        "foreground_radial_clearance_px": float(255.5 - mask_maximum),
        "all_targets_inside_every_rotation": bool(target_radius.max() <= 199.550746),
        "all_foreground_inside_every_rotation": bool(mask_maximum <= 213.012911),
        "all_corners_fixed_background": bool(
            np.all(corners == expected_background[None])
        ),
        "corner_rgb_uint8": expected_background.tolist(),
    }


def _analytic_point_audit(target_px: np.ndarray) -> dict[str, Any]:
    right = torch.tensor([[[300.0, 255.5]]], dtype=torch.float64)
    plus_ninety = transform_points_px(
        right, torch.tensor([90.0], dtype=torch.float32)
    )[0, 0]
    expected = torch.tensor((255.5, 300.0), dtype=torch.float64)
    sign_error = float(torch.max(torch.abs(plus_ninety - expected)))
    source = torch.from_numpy(target_px.astype(np.float64, copy=False))
    roundtrip_errors: list[float] = []
    for angle in TEST_ANGLES_DEG:
        angles = torch.full((source.shape[0],), angle, dtype=torch.float32)
        rotated = transform_points_px(source, angles)
        recovered = transform_points_px(rotated, -angles)
        roundtrip_errors.append(float(torch.max(torch.abs(recovered - source))))
    return {
        "plus_90_image_down_max_abs_error_px": sign_error,
        "roundtrip_angles_deg": list(TEST_ANGLES_DEG),
        "roundtrip_max_abs_error_px": max(roundtrip_errors),
        "pass": sign_error <= 1e-8 and max(roundtrip_errors) <= 1e-5,
    }


def _zero_angle_audit(dataset: Any) -> dict[str, Any]:
    images, targets = _dataset_batch(dataset, (0, 49, 99, 149))
    proposed = torch.tensor((-93.0, 14.0, 121.0, -179.0), dtype=torch.float32)
    bypass_images, bypass_targets, _, effective = apply_arm_transform(
        images,
        targets,
        proposed,
        global_exposure_start=0,
        arm="control",
    )
    bypass_exact = bool(
        torch.equal(images, bypass_images)
        and torch.equal(targets, bypass_targets)
        and torch.count_nonzero(effective).item() == 0
    )
    zeros = torch.zeros((images.shape[0],), dtype=torch.float32)
    warped_images, warped_targets = warp_normalized_images_and_targets(
        images, targets, zeros
    )
    mean = MEAN.to(dtype=images.dtype)
    std = STD.to(dtype=images.dtype)
    original_raw = images * std + mean
    warped_raw = warped_images * std + mean
    raw_error = float(torch.max(torch.abs(warped_raw - original_raw)))
    target_error = float(
        torch.max(
            torch.abs(
                (warped_targets - targets) * (511.0 * 0.5)
            )
        )
    )
    masks = torch.from_numpy(dataset.masks[[0, 49, 99, 149]])
    warped_masks = warp_masks(masks, zeros)
    mask_exact = bool(torch.equal(masks, warped_masks))
    return {
        "control_bypass_bitwise_exact": bypass_exact,
        "zero_core_raw_rgb_max_abs_error_0_to_1": raw_error,
        "zero_core_target_max_abs_error_px": target_error,
        "zero_core_mask_bitwise_exact": mask_exact,
        "pass": bool(
            bypass_exact
            and raw_error <= 1e-4
            and target_error <= 1e-4
            and mask_exact
        ),
    }


def _impulse_audit() -> dict[str, Any]:
    batch_size = len(TEST_ANGLES_DEG)
    background = torch.tensor(BACKGROUND_RGB_UINT8, dtype=torch.float32).view(
        1, 3, 1, 1
    ) / 255.0
    raw = background.expand(batch_size, 3, 512, 512).clone()
    source_xy = torch.tensor((330.0, 255.0), dtype=torch.float32)
    raw[:, :, int(source_xy[1]), int(source_xy[0])] = 1.0
    images = (raw - MEAN) / STD
    target_px = source_xy.view(1, 1, 2).expand(batch_size, 1, 2).clone()
    targets = pixel_to_normalized_torch(target_px)
    angles = torch.tensor(TEST_ANGLES_DEG, dtype=torch.float32)
    warped, warped_targets = warp_normalized_images_and_targets(
        images, targets, angles
    )
    warped_raw = warped * STD + MEAN
    deviation = torch.abs(warped_raw - background).sum(dim=1)
    peak_index = deviation.flatten(1).argmax(dim=1)
    peak_xy = torch.stack(
        (peak_index.remainder(512), torch.div(peak_index, 512, rounding_mode="floor")),
        dim=-1,
    ).float()
    expected_px = (warped_targets[:, 0] + 1.0) * 255.5
    errors = torch.linalg.vector_norm(peak_xy - expected_px, dim=-1)
    return {
        "angles_deg": list(TEST_ANGLES_DEG),
        "peak_error_px": [float(value) for value in errors],
        "maximum_peak_error_px": float(errors.max()),
        "pass": bool((errors <= 1.0).all()),
    }


def _adjacent_mask_audit(masks: np.ndarray) -> dict[str, Any]:
    source = torch.from_numpy(masks[:-1])
    target = masks[1:]
    correct_values: list[float] = []
    wrong_values: list[float] = []
    for start in range(0, len(source), 8):
        batch = source[start : start + 8]
        correct = warp_masks(batch, torch.full((len(batch),), 2.0)).numpy()
        wrong = warp_masks(batch, torch.full((len(batch),), -2.0)).numpy()
        for offset in range(len(batch)):
            correct_values.append(_iou(correct[offset], target[start + offset]))
            wrong_values.append(_iou(wrong[offset], target[start + offset]))
    correct_array = np.asarray(correct_values)
    wrong_array = np.asarray(wrong_values)
    return {
        "correct_plus_2_iou": _summary(correct_array),
        "wrong_minus_2_iou": _summary(wrong_array),
        "sample_unit": "overlapping adjacent rendered training-mask pair",
        "descriptive_not_inferential": True,
        "sem_or_confidence_interval_computed": False,
        "pass": bool(
            np.median(correct_array) >= 0.985
            and correct_array.min() >= 0.980
            and np.median(wrong_array) <= 0.750
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)
    verify_semantic_lock(args.semantic_lock)
    require_local_cpu_only()
    verify_exact_file(
        args.previous_training_predictions,
        PREVIOUS_TRAINING_PREDICTIONS_SHA256,
        "previous training predictions",
    )
    verify_exact_file(
        args.previous_evaluation_result,
        PREVIOUS_EVALUATION_RESULT_SHA256,
        "previous evaluation result",
    )
    previous_evaluation = load_json(args.previous_evaluation_result)
    require(
        previous_evaluation.get("decision", {}).get("operator_authorized") is False,
        "previous result unexpectedly authorized operator",
    )
    train_receipt = load_json(args.train_input_receipt)
    require(
        train_receipt.get("schema_version")
        == "leakage_safe_distillation_train_input_receipt.v1",
        "train receipt schema differs",
    )
    require(
        train_receipt.get("repository_head") == repository_head,
        "train receipt repository differs",
    )
    require(
        train_receipt.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256,
        "train receipt semantic lock differs",
    )
    require(
        train_receipt.get("validation_truth_received_by_training_stage") is False,
        "train receipt exposes validation truth",
    )
    require(
        train_receipt.get("full_track_archive_received_by_training_stage") is False,
        "train receipt exposes the full track archive",
    )
    require(
        np.array_equal(
            np.asarray(train_receipt.get("training_frame_indices"), dtype=np.int64),
            TRAIN_FRAMES,
        ),
        "train receipt frames differ",
    )
    train_targets = verify_record(train_receipt["train_targets"], "train targets")
    train_manifest = verify_record(
        train_receipt["train_frame_manifest"], "train manifest"
    )
    dataset, target_px, masks, _, data_controls = load_bound_training_dataset(
        train_manifest, train_targets, args.object_root
    )

    start = time.perf_counter()
    off_object = _off_object_audit(
        args.previous_training_predictions, target_px, masks
    )
    radial = _radial_and_background_audit(dataset, target_px)
    analytic = _analytic_point_audit(target_px)
    zero = _zero_angle_audit(dataset)
    impulse = _impulse_audit()
    adjacent = _adjacent_mask_audit(masks)
    gates = {
        "prior_off_object_replay": bool(
            off_object["count"] == 28
            and off_object["all_within_1_5_px_of_foreground"]
        ),
        "radial_canvas_and_background": bool(
            radial["all_targets_inside_every_rotation"]
            and radial["all_foreground_inside_every_rotation"]
            and radial["all_corners_fixed_background"]
        ),
        "analytic_sign_and_roundtrip": bool(analytic["pass"]),
        "zero_angle_identity": bool(zero["pass"]),
        "planted_impulse_alignment": bool(impulse["pass"]),
        "adjacent_mask_sign_and_pivot": bool(adjacent["pass"]),
        "training_only_information_boundary": bool(
            data_controls["only_training_frames_loaded"]
            and train_receipt["validation_truth_received_by_training_stage"] is False
        ),
    }
    all_pass = all(gates.values())
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "PAIRED_ROTATION_PREFLIGHT_RESULT.json"
    result = {
        "schema_version": "paired_rotation_augmentation_preflight_result.v1",
        "artifact_type": "training_only_known_warp_semantic_preflight",
        "repository_head": repository_head,
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "runtime_seconds": time.perf_counter() - start,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "augmentation": file_record(
                args.repo_root / "keypoint_net" / "paired_rotation_augmentation.py"
            ),
            "contract": file_record(
                args.repo_root
                / "keypoint_net"
                / "leakage_safe_distillation_contract.py"
            ),
        },
        "semantic_lock": file_record(args.semantic_lock),
        "train_input_receipt": file_record(args.train_input_receipt),
        "previous_training_predictions": file_record(
            args.previous_training_predictions
        ),
        "previous_evaluation_result": file_record(args.previous_evaluation_result),
        "off_object_audit": off_object,
        "radial_and_background_audit": radial,
        "analytic_point_audit": analytic,
        "zero_angle_audit": zero,
        "planted_impulse_audit": impulse,
        "adjacent_mask_audit": adjacent,
        "gates": gates,
        "all_preflight_gates_pass": all_pass,
        "validation_truth_received_or_opened": False,
        "operator_prediction_received_or_opened": False,
        "training_or_weight_update_performed": False,
        "statistical_scope": {
            "inference": "descriptive_only",
            "adjacent_mask_pair_count": len(TRAIN_FRAMES) - 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "PAIRED_ROTATION_PREFLIGHT_RECEIPT.json"
    receipt = {
        "schema_version": "paired_rotation_augmentation_preflight_receipt.v1",
        "repository_head": repository_head,
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "result": file_record(result_path),
        "semantic_lock": file_record(args.semantic_lock),
        "all_preflight_gates_pass": all_pass,
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
    parser.add_argument("--previous-training-predictions", type=Path, required=True)
    parser.add_argument("--previous-evaluation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"PAIRED ROTATION PREFLIGHT CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
