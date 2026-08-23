"""Privileged post-hash evaluation of the global RGB silhouette teacher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .certified_witness_capability import (
        EXPECTED_WITNESS_IDS,
        HALF_CELL_DIAGONAL_PX,
        evaluate_predictions,
    )
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from certified_witness_capability import (
        EXPECTED_WITNESS_IDS,
        HALF_CELL_DIAGONAL_PX,
        evaluate_predictions,
    )
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_LOCK_SHA256 = "bcb24fc39510b829acdf64022a3ab7e7cfa4ffc6a3c89fd2510a9ea788b27362"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_CAPABILITY_MANIFEST_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_TRACKS_SHA256 = "b9decd7440da1e35f935f5d8d443e3eb9738b1584f8b72ebebb51b1d7bfa93b6"
EXPECTED_INITIALS_SHA256 = "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
HEADER_HEIGHT = 72


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_masks(capability: dict[str, Any], object_root: Path) -> np.ndarray:
    frames = capability["dataset"]["frames"]
    require(len(frames) == EXPECTED_FRAMES, "capability frame count differs")
    masks = np.empty((EXPECTED_FRAMES, IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for expected, row in enumerate(frames):
        require(int(row["frame_index"]) == expected, "capability frame order differs")
        path = (object_root / str(row["mask_relpath"])).resolve()
        record = file_record(path, include_path=False)
        require(record["sha256"] == row["mask_sha256"], f"mask hash differs: {path}")
        with Image.open(path) as opened:
            value = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        require(value.shape == (IMAGE_SIZE, IMAGE_SIZE), f"mask shape differs: {path}")
        masks[expected] = value
    return masks


def _rigid_angle(source: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(left_centered.T @ right_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def _circular_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (np.asarray(first) - np.asarray(second) + math.pi) % (2.0 * math.pi) - math.pi


def _mark(image: Image.Image, xy: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw = ImageDraw.Draw(image)
    radius = 9
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.line((x - 13, y, x + 13, y), fill=color, width=3)
    draw.line((x, y - 13, x, y + 13), fill=color, width=3)
    draw.text((x + 11, y - 12), label, fill=color, stroke_width=2, stroke_fill="black")


def _render_worst(
    rgb_paths: list[Path],
    truth: np.ndarray,
    prediction: np.ndarray,
    error: np.ndarray,
    witness_id: np.ndarray,
    selected_angle: np.ndarray,
    selected_iou: np.ndarray,
    output_path: Path,
) -> list[dict[str, Any]]:
    flat_order = np.argsort(error.reshape(-1), kind="stable")[::-1][:4]
    canvas = Image.new("RGB", (2 * IMAGE_SIZE, 2 * (IMAGE_SIZE + HEADER_HEIGHT)), "white")
    events: list[dict[str, Any]] = []
    for panel, flat in enumerate(flat_order):
        frame, witness = np.unravel_index(int(flat), error.shape)
        with Image.open(rgb_paths[frame]) as opened:
            image = opened.convert("RGB")
        _mark(image, truth[frame, witness], (0, 210, 0), "TRUE")
        _mark(image, prediction[frame, witness], (255, 0, 255), "PRED")
        ImageDraw.Draw(image).line(
            (*truth[frame, witness], *prediction[frame, witness]),
            fill=(255, 215, 0),
            width=4,
        )
        wrapped = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE + HEADER_HEIGHT), "white")
        wrapped.paste(image, (0, HEADER_HEIGHT))
        title = (
            f"frame {frame}  witness {int(witness_id[witness])}  error {error[frame, witness]:.3f}px\n"
            f"RGB silhouette angle {math.degrees(selected_angle[frame]):.3f} deg; IoU {selected_iou[frame]:.5f}"
        )
        ImageDraw.Draw(wrapped).multiline_text(
            (10, 8), title, fill="black", font=ImageFont.load_default(), spacing=4
        )
        canvas.paste(
            wrapped,
            ((panel % 2) * IMAGE_SIZE, (panel // 2) * (IMAGE_SIZE + HEADER_HEIGHT)),
        )
        events.append(
            {
                "frame_index": int(frame),
                "witness_index": int(witness),
                "witness_id": int(witness_id[witness]),
                "error_px": float(error[frame, witness]),
                "truth_xy": truth[frame, witness].tolist(),
                "prediction_xy": prediction[frame, witness].tolist(),
                "selected_angle_deg": float(math.degrees(selected_angle[frame])),
                "selected_silhouette_iou": float(selected_iou[frame]),
                "rgb": file_record(rgb_paths[frame]),
            }
        )
    canvas.save(output_path)
    return events


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")

    lock_record = file_record(args.semantic_lock)
    require(lock_record["sha256"] == EXPECTED_LOCK_SHA256, "semantic-lock SHA-256 differs")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    capability_record = file_record(args.capability_manifest)
    require(
        capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256,
        "capability-manifest SHA-256 differs",
    )
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == EXPECTED_TRACKS_SHA256, "tracks SHA-256 differs")
    raw_receipt_record = file_record(args.raw_receipt)
    raw_receipt = load_json(args.raw_receipt)
    require(raw_receipt["schema_version"] == "raw_global_rgb_silhouette_teacher.v1", "raw schema differs")
    require(raw_receipt["sources"]["semantic_lock"] == lock_record, "raw lock binding differs")
    require(raw_receipt["sources"]["sanitized_manifest"] == manifest_record, "raw manifest binding differs")
    require(
        raw_receipt["sources"]["frame_zero_initials"]["sha256"] == EXPECTED_INITIALS_SHA256,
        "raw initials binding differs",
    )
    require(raw_receipt["controls"]["supplied_masks_opened"] is False, "raw stage opened supplied masks")
    require(raw_receipt["controls"]["non_frame_zero_truth_opened"] is False, "raw stage opened truth")
    require(
        raw_receipt["controls"]["frame_order_reversal_exact_every_array"] is True,
        "frame-order control failed",
    )
    raw_record = file_record(args.raw_predictions)
    require(raw_receipt["raw_predictions"] == raw_record, "raw prediction binding differs")

    with np.load(args.raw_predictions) as raw:
        expected_keys = {
            "witness_id",
            "initial_frame_index",
            "initial_coordinate_px",
            "foreground_mask",
            "background_rgb",
            "otsu_threshold",
            "component_label",
            "component_area",
            "centroid_xy",
            "second_moment_real_imag",
            "anisotropy",
            "candidate_angle_rad",
            "candidate_iou",
            "selected_index",
            "selected_angle_rad",
            "selected_iou",
            "ambiguity_gap_iou",
            "matrix",
            "prediction_xy",
        }
        require(set(raw.files) == expected_keys, "raw arrays expose unexpected keys")
        witness_id = np.asarray(raw["witness_id"], dtype=np.int64)
        initial_frame = int(raw["initial_frame_index"])
        initial_coordinate = np.asarray(raw["initial_coordinate_px"], dtype=np.float64)
        prediction = np.asarray(raw["prediction_xy"], dtype=np.float64)
        selected_angle = np.asarray(raw["selected_angle_rad"], dtype=np.float64)
        selected_iou = np.asarray(raw["selected_iou"], dtype=np.float64)
        ambiguity_gap = np.asarray(raw["ambiguity_gap_iou"], dtype=np.float64)
        component_area = np.asarray(raw["component_area"], dtype=np.int64)
        centroid = np.asarray(raw["centroid_xy"], dtype=np.float64)
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "witness IDs differ")
    require(initial_frame == 0, "initial frame differs")
    require(prediction.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "prediction shape differs")

    capability = load_json(args.capability_manifest)
    require(capability["portable_tracks"]["sha256"] == EXPECTED_TRACKS_SHA256, "capability track binding differs")
    with np.load(args.tracks) as archive:
        truth_witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        truth = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(witness_id, truth_witness_id), "truth witness order differs")
    require(truth.shape == prediction.shape, "truth shape differs")
    require(np.array_equal(initial_coordinate, truth[0]), "frame-zero initialization differs")
    masks = _load_masks(capability, args.object_root.resolve())
    report, derived = evaluate_predictions(
        prediction,
        truth,
        masks,
        witness_ids=tuple(int(value) for value in witness_id),
    )

    source_distances = np.linalg.norm(
        initial_coordinate[:, None, :] - initial_coordinate[None, :, :], axis=-1
    )
    predicted_distances = np.linalg.norm(
        prediction[:, :, None, :] - prediction[:, None, :, :], axis=-1
    )
    maximum_relative_distance_change = float(
        np.max(np.abs(predicted_distances - source_distances[None]))
    )
    truth_angles = np.asarray(
        [_rigid_angle(initial_coordinate, truth[frame]) for frame in range(EXPECTED_FRAMES)],
        dtype=np.float64,
    )
    angle_error_deg = np.degrees(_circular_difference(selected_angle, truth_angles))

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "GLOBAL_SILHOUETTE_TEACHER_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        truth_xy=truth,
        prediction_xy=prediction,
        material_error_px=derived["material_error_px"],
        within_half_cell=derived["within_half_cell"],
        on_object=derived["on_object"],
        identity_correct=derived["identity_correct"],
        distinct_pair=derived["distinct_pair"],
        truth_rigid_angle_rad=truth_angles,
        silhouette_angle_error_deg=angle_error_deg,
        component_area=component_area,
        centroid_xy=centroid,
        selected_silhouette_iou=selected_iou,
        orientation_ambiguity_gap_iou=ambiguity_gap,
    )
    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    visual_path = args.output_dir / "WORST_GLOBAL_SILHOUETTE_TEACHER_EVENTS.png"
    visual_events = _render_worst(
        rgb_paths,
        truth,
        prediction,
        derived["material_error_px"],
        witness_id,
        selected_angle,
        selected_iou,
        visual_path,
    )
    strict_pass = bool(report["strict_capability_pass"])
    result = {
        "schema_version": "global_rgb_silhouette_teacher_evaluation.v1",
        "artifact_type": "privileged_posthash_global_silhouette_teacher_check",
        "decision": {
            "all_ten_all_180_strict_pass": strict_pass,
            "branch": (
                "authorize_raw_site_selection_then_cnn_distillation"
                if strict_pass
                else "reject_global_rgb_silhouette_teacher"
            ),
        },
        "frozen_contract_report": report,
        "pose_diagnostics": {
            "selected_silhouette_iou": {
                "minimum": float(np.min(selected_iou)),
                "median": float(np.median(selected_iou)),
                "maximum": float(np.max(selected_iou)),
            },
            "orientation_ambiguity_gap_iou": {
                "minimum": float(np.min(ambiguity_gap)),
                "median": float(np.median(ambiguity_gap)),
            },
            "foreground_component_area_px": {
                "minimum": int(np.min(component_area)),
                "median": float(np.median(component_area)),
                "maximum": int(np.max(component_area)),
            },
            "silhouette_angle_error_deg": {
                "maximum_absolute": float(np.max(np.abs(angle_error_deg))),
                "median_absolute": float(np.median(np.abs(angle_error_deg))),
            },
            "maximum_pairwise_distance_change_px": maximum_relative_distance_change,
        },
        "thresholds": {
            "material_error_maximum_px": HALF_CELL_DIAGONAL_PX,
            "all_predictions_must_be_on_supplied_object_mask": True,
            "all_identity_assignments_must_be_correct": True,
            "all_pair_ratios_must_be_at_least": 0.5,
            "raw_selected_silhouette_iou_minimum": 0.95,
        },
        "visual_selection": "four largest material errors globally; stable descending order",
        "visual_events": visual_events,
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
            "raw_predictions": raw_record,
            "raw_receipt": raw_receipt_record,
        },
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "arrays": file_record(arrays_path),
        "visual": file_record(visual_path),
        "raw_predictions_hashed_before_truth_or_supplied_masks_open": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over one correlated 180-frame orbit; descriptive only",
    }
    result_path = args.output_dir / "GLOBAL_SILHOUETTE_TEACHER_EVALUATION_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
