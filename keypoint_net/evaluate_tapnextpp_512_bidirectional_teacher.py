"""Privileged post-hash evaluation of the TAPNext++ 512 support-point bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
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


EXPECTED_LOCK_SHA256 = (
    "5f438c7b9958ec64dd25d3344321eccb7e9b13a41e586ef5198fa8036990b78f"
)
EXPECTED_MANIFEST_SHA256 = (
    "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
)
EXPECTED_CAPABILITY_MANIFEST_SHA256 = (
    "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
)
EXPECTED_TRACKS_SHA256 = (
    "b9decd7440da1e35f935f5d8d443e3eb9738b1584f8b72ebebb51b1d7bfa93b6"
)
EXPECTED_INITIALS_SHA256 = (
    "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
)
EXPECTED_TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
EXPECTED_CHECKPOINT_SHA256 = (
    "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f"
)
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
MODEL_INPUT_RESOLUTION = 512
MODEL_COORDINATE_RESOLUTION = 256
SUPPORT_POINTS_PER_QUERY = 64
SUPPORT_RADIUS_MODEL_INPUT_PX = 32.0
EXPECTED_INTERNAL_QUERIES = EXPECTED_WITNESSES * (SUPPORT_POINTS_PER_QUERY + 1)
HEADER_HEIGHT = 76


RAW_ARRAY_KEYS = {
    "witness_id",
    "initial_frame_index",
    "initial_coordinate_px",
    "forward_prediction_xy",
    "forward_visible",
    "reverse_prediction_xy",
    "reverse_visible",
    "forward_reverse_difference_px",
    "forward_traversal",
    "reverse_traversal",
    "repeated_prefix_prediction_xy",
    "repeated_prefix_visible",
}


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


def _mark(
    image: Image.Image, xy: np.ndarray, color: tuple[int, int, int], label: str
) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw = ImageDraw.Draw(image)
    radius = 9
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline=color, width=4
    )
    draw.line((x - 13, y, x + 13, y), fill=color, width=3)
    draw.line((x, y - 13, x, y + 13), fill=color, width=3)
    draw.text((x + 11, y - 12), label, fill=color, stroke_width=2, stroke_fill="black")


def _render_worst(
    *,
    direction: str,
    rgb_paths: list[Path],
    truth: np.ndarray,
    prediction: np.ndarray,
    visible: np.ndarray,
    error: np.ndarray,
    witness_id: np.ndarray,
    output_path: Path,
) -> list[dict[str, Any]]:
    flat_order = np.argsort(error.reshape(-1), kind="stable")[::-1][:4]
    canvas = Image.new(
        "RGB", (2 * IMAGE_SIZE, 2 * (IMAGE_SIZE + HEADER_HEIGHT)), "white"
    )
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
            f"{direction}  frame {frame}  witness {int(witness_id[witness])}\n"
            f"error {error[frame, witness]:.3f}px; model visible {bool(visible[frame, witness])}"
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
                "direction": direction,
                "frame_index": int(frame),
                "witness_index": int(witness),
                "witness_id": int(witness_id[witness]),
                "error_px": float(error[frame, witness]),
                "model_visible": bool(visible[frame, witness]),
                "truth_xy": truth[frame, witness].tolist(),
                "prediction_xy": prediction[frame, witness].tolist(),
                "rgb": file_record(rgb_paths[frame]),
            }
        )
    canvas.save(output_path)
    return events


def _direction_decision(report: dict[str, Any], visible: np.ndarray) -> dict[str, Any]:
    capability_pass = bool(report["strict_capability_pass"])
    visibility_pass = bool(np.all(visible))
    return {
        "immutable_capability_pass": capability_pass,
        "all_1800_model_visibility_flags_true": visibility_pass,
        "strict_direction_pass": bool(capability_pass and visibility_pass),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(
        not args.output_dir.exists(),
        "output directory already exists; use a fresh attempt",
    )
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")

    lock_record = file_record(args.semantic_lock)
    manifest_record = file_record(args.manifest)
    require(
        manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256,
        "manifest SHA-256 differs",
    )
    capability_record = file_record(args.capability_manifest)
    require(
        capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256,
        "capability-manifest SHA-256 differs",
    )
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == EXPECTED_TRACKS_SHA256, "tracks SHA-256 differs")

    raw_receipt_record = file_record(args.raw_receipt)
    raw_receipt = load_json(args.raw_receipt)
    require(
        raw_receipt["schema_version"]
        == "raw_tapnextpp_512_support64_bidirectional_teacher.v1",
        "raw schema differs",
    )
    execution_profile = raw_receipt["execution_profile"]
    require(execution_profile == "cpu", "TAPNext++ 512 raw profile is not CPU")
    require(
        lock_record["sha256"] == EXPECTED_LOCK_SHA256,
        "TAPNext++ 512 semantic-lock SHA-256 differs",
    )
    require(
        raw_receipt["decision"]["raw_tracker_semantic_pass"] is True,
        "raw tracker gate failed; truth evaluation is not authorized",
    )
    require(
        raw_receipt["privileged_evaluation_authorized"] is True,
        "raw stage withheld evaluation",
    )
    require(
        raw_receipt["sources"]["semantic_lock"] == lock_record,
        "raw lock binding differs",
    )
    require(
        raw_receipt["sources"]["sanitized_manifest"] == manifest_record,
        "raw manifest binding differs",
    )
    require(
        raw_receipt["sources"]["frame_zero_initials"]["sha256"]
        == EXPECTED_INITIALS_SHA256,
        "raw initials binding differs",
    )
    require(
        raw_receipt["sources"]["tapnextpp_checkpoint"]["sha256"]
        == EXPECTED_CHECKPOINT_SHA256,
        "raw checkpoint binding differs",
    )
    require(
        raw_receipt["sources"]["tapnet"]["commit"] == EXPECTED_TAPNET_COMMIT,
        "raw TAPNet commit differs",
    )
    controls = raw_receipt["controls"]
    require(
        controls["supplied_masks_opened"] is False, "raw stage opened supplied masks"
    )
    require(controls["non_frame_zero_truth_opened"] is False, "raw stage opened truth")
    require(
        controls["renderer_angle_or_pivot_opened"] is False,
        "raw stage opened renderer geometry",
    )
    require(
        controls["learned_keypoint_checkpoint_or_features_opened"] is False,
        "raw stage opened a forbidden learned keypoint path",
    )
    require(controls["local_laptop_gpu_used"] is False, "raw stage used the laptop GPU")
    environment = raw_receipt["environment"]
    require(
        environment["execution_device"] == execution_profile,
        "raw device binding differs",
    )
    require(controls["local_laptop_cpu_only"] is True, "raw stage was not CPU only")
    require(controls["cluster_cuda_only"] is False, "raw CPU stage claims cluster CUDA")
    require(controls["autocast_enabled"] is False, "raw CPU stage used autocast")
    require(environment["cuda_available"] is False, "raw CPU environment exposed CUDA")
    require(
        environment["model_input_resolution"] == MODEL_INPUT_RESOLUTION,
        "raw model input resolution differs",
    )
    require(
        environment["model_coordinate_resolution"] == MODEL_COORDINATE_RESOLUTION,
        "raw model coordinate resolution differs",
    )
    require(
        raw_receipt["model_frame_call_count"] == 370, "raw model-call count differs"
    )
    pip_freeze_path = Path(raw_receipt["pip_freeze"]["absolute_path"])
    require(
        file_record(pip_freeze_path) == raw_receipt["pip_freeze"],
        "raw pip-freeze binding differs",
    )
    require(
        controls["repeated_prefix_positions_exact"] is True,
        "raw repeat positions differ",
    )
    require(
        controls["repeated_prefix_visibility_exact"] is True,
        "raw repeat visibility differs",
    )
    require(
        controls["all_forward_coordinates_finite_and_in_image"] is True,
        "raw forward coordinate control failed",
    )
    require(
        controls["all_reverse_coordinates_finite_and_in_image"] is True,
        "raw reverse coordinate control failed",
    )
    require(
        controls["official_512_input_256_coordinate_roundtrip_pass"] is True,
        "raw official coordinate round-trip control failed",
    )
    require(
        controls["official_local_support_construction_pass"] is True,
        "raw official support construction control failed",
    )
    require(
        controls["only_ten_real_query_trajectories_saved"] is True,
        "raw stage did not discard support trajectories",
    )
    support = raw_receipt["support_configuration"]
    require(support["mode"] == "local", "raw support mode differs")
    require(
        support["support_radius_space"] == "model", "raw support radius space differs"
    )
    require(
        support["support_radius_model_input_px"] == SUPPORT_RADIUS_MODEL_INPUT_PX,
        "raw support radius differs",
    )
    require(
        support["support_points_per_real_query"] == SUPPORT_POINTS_PER_QUERY,
        "raw support count per query differs",
    )
    require(
        support["real_query_count"] == EXPECTED_WITNESSES,
        "raw real query count differs",
    )
    require(
        support["support_query_count"] == EXPECTED_WITNESSES * SUPPORT_POINTS_PER_QUERY,
        "raw support query count differs",
    )
    require(
        support["internal_query_count"] == EXPECTED_INTERNAL_QUERIES,
        "raw internal query count differs",
    )
    require(support["support_trajectories_saved"] is False, "support tracks were saved")
    require(
        support["source"] == "tapnet/tapnextpp/votsp2026/tracker.py",
        "raw support source differs",
    )
    coordinate_mapping = raw_receipt["coordinate_mapping"]
    require(
        coordinate_mapping["maximum_absolute_roundtrip_error_px"] <= 1e-6,
        "raw official coordinate round-trip error exceeds tolerance",
    )
    require(
        coordinate_mapping["custom_half_pixel_correction_applied"] is False,
        "raw stage applied a custom half-pixel correction",
    )

    raw_record = file_record(args.raw_predictions)
    require(
        raw_receipt["raw_predictions"] == raw_record, "raw prediction binding differs"
    )
    with np.load(args.raw_predictions) as raw:
        require(set(raw.files) == RAW_ARRAY_KEYS, "raw arrays expose unexpected keys")
        witness_id = np.asarray(raw["witness_id"], dtype=np.int64)
        initial_frame = int(raw["initial_frame_index"])
        initial_coordinate = np.asarray(raw["initial_coordinate_px"], dtype=np.float64)
        forward_prediction = np.asarray(raw["forward_prediction_xy"], dtype=np.float64)
        forward_visible = np.asarray(raw["forward_visible"], dtype=bool)
        reverse_prediction = np.asarray(raw["reverse_prediction_xy"], dtype=np.float64)
        reverse_visible = np.asarray(raw["reverse_visible"], dtype=bool)
        direction_difference = np.asarray(
            raw["forward_reverse_difference_px"], dtype=np.float64
        )
        forward_traversal = np.asarray(raw["forward_traversal"], dtype=np.int64)
        reverse_traversal = np.asarray(raw["reverse_traversal"], dtype=np.int64)

    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "witness IDs differ")
    require(initial_frame == 0, "initial frame differs")
    require(
        initial_coordinate.shape == (EXPECTED_WITNESSES, 2),
        "initial coordinates differ",
    )
    expected_shape = (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2)
    require(
        forward_prediction.shape == expected_shape, "forward prediction shape differs"
    )
    require(
        reverse_prediction.shape == expected_shape, "reverse prediction shape differs"
    )
    require(
        forward_visible.shape == expected_shape[:2], "forward visibility shape differs"
    )
    require(
        reverse_visible.shape == expected_shape[:2], "reverse visibility shape differs"
    )
    require(
        np.array_equal(forward_traversal, np.arange(EXPECTED_FRAMES)),
        "forward traversal differs",
    )
    require(
        np.array_equal(
            reverse_traversal,
            np.asarray([0, *range(EXPECTED_FRAMES - 1, 0, -1)], dtype=np.int64),
        ),
        "reverse traversal differs",
    )
    recomputed_direction_difference = np.linalg.norm(
        forward_prediction - reverse_prediction, axis=-1
    )
    require(
        np.array_equal(direction_difference, recomputed_direction_difference),
        "stored direction disagreement differs",
    )

    # Privileged information is opened only after all raw records and controls pass.
    capability = load_json(args.capability_manifest)
    require(
        capability["portable_tracks"]["sha256"] == EXPECTED_TRACKS_SHA256,
        "capability track binding differs",
    )
    with np.load(args.tracks) as archive:
        truth_witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        truth = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(witness_id, truth_witness_id), "truth witness order differs")
    require(truth.shape == expected_shape, "truth shape differs")
    require(
        np.array_equal(initial_coordinate, truth[0]),
        "frame-zero initialization differs",
    )
    masks = _load_masks(capability, args.object_root.resolve())
    forward_report, forward_derived = evaluate_predictions(
        forward_prediction,
        truth,
        masks,
        witness_ids=tuple(int(value) for value in witness_id),
    )
    reverse_report, reverse_derived = evaluate_predictions(
        reverse_prediction,
        truth,
        masks,
        witness_ids=tuple(int(value) for value in witness_id),
    )
    forward_decision = _direction_decision(forward_report, forward_visible)
    reverse_decision = _direction_decision(reverse_report, reverse_visible)
    direction_pass_count = int(forward_decision["strict_direction_pass"]) + int(
        reverse_decision["strict_direction_pass"]
    )
    individual_channel_passes = []
    for channel, witness in enumerate(witness_id):
        forward_channel = bool(
            forward_report["per_witness"][channel]["strict_channel_pass"]
        )
        reverse_channel = bool(
            reverse_report["per_witness"][channel]["strict_channel_pass"]
        )
        forward_visibility = bool(np.all(forward_visible[:, channel]))
        reverse_visibility = bool(np.all(reverse_visible[:, channel]))
        individual_channel_passes.append(
            {
                "channel": channel,
                "witness_id": int(witness),
                "forward_localization_object_identity_pass": forward_channel,
                "reverse_localization_object_identity_pass": reverse_channel,
                "forward_visibility_pass": forward_visibility,
                "reverse_visibility_pass": reverse_visibility,
                "bidirectional_individual_channel_pass": bool(
                    forward_channel
                    and reverse_channel
                    and forward_visibility
                    and reverse_visibility
                ),
            }
        )
    individual_channel_pass_count = sum(
        int(row["bidirectional_individual_channel_pass"])
        for row in individual_channel_passes
    )
    any_pair_collapse = bool(
        forward_report["violations"]["collapsed_pair_count"]
        or reverse_report["violations"]["collapsed_pair_count"]
    )
    if direction_pass_count == 2:
        branch = "authorize_direct_leakage_safe_shared_affine_operator_fit"
    elif 6 <= individual_channel_pass_count <= 9 and not any_pair_collapse:
        branch = "preserve_diagnostic_and_require_predeclared_six_scope_decision"
    else:
        branch = "reject_tapnextpp_512_support64_privileged_bridge"

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "TAPNEXTPP_512_BIDIRECTIONAL_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        truth_xy=truth,
        forward_prediction_xy=forward_prediction,
        reverse_prediction_xy=reverse_prediction,
        forward_visible=forward_visible,
        reverse_visible=reverse_visible,
        forward_reverse_difference_px=direction_difference,
        forward_material_error_px=forward_derived["material_error_px"],
        reverse_material_error_px=reverse_derived["material_error_px"],
        forward_within_half_cell=forward_derived["within_half_cell"],
        reverse_within_half_cell=reverse_derived["within_half_cell"],
        forward_on_object=forward_derived["on_object"],
        reverse_on_object=reverse_derived["on_object"],
        forward_identity_correct=forward_derived["identity_correct"],
        reverse_identity_correct=reverse_derived["identity_correct"],
        forward_distinct_pair=forward_derived["distinct_pair"],
        reverse_distinct_pair=reverse_derived["distinct_pair"],
    )
    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    forward_visual_path = args.output_dir / "WORST_TAPNEXTPP_512_FORWARD_EVENTS.png"
    reverse_visual_path = args.output_dir / "WORST_TAPNEXTPP_512_REVERSE_EVENTS.png"
    forward_events = _render_worst(
        direction="forward",
        rgb_paths=rgb_paths,
        truth=truth,
        prediction=forward_prediction,
        visible=forward_visible,
        error=forward_derived["material_error_px"],
        witness_id=witness_id,
        output_path=forward_visual_path,
    )
    reverse_events = _render_worst(
        direction="reverse",
        rgb_paths=rgb_paths,
        truth=truth,
        prediction=reverse_prediction,
        visible=reverse_visible,
        error=reverse_derived["material_error_px"],
        witness_id=witness_id,
        output_path=reverse_visual_path,
    )

    result = {
        "schema_version": "tapnextpp_512_support64_bidirectional_teacher_evaluation.v1",
        "artifact_type": "privileged_posthash_pretrained_bidirectional_teacher_check",
        "execution_profile": execution_profile,
        "decision": {
            "both_directions_strict_pass": direction_pass_count == 2,
            "strict_direction_pass_count": direction_pass_count,
            "forward": forward_decision,
            "reverse": reverse_decision,
            "bidirectional_individual_channel_pass_count": individual_channel_pass_count,
            "individual_channels": individual_channel_passes,
            "any_pair_collapse": any_pair_collapse,
            "branch": branch,
        },
        "frozen_contract_report": {
            "forward": forward_report,
            "reverse": reverse_report,
        },
        "visibility_diagnostics": {
            "forward_visible_count": int(np.sum(forward_visible)),
            "forward_invisible_count": int(
                forward_visible.size - np.sum(forward_visible)
            ),
            "reverse_visible_count": int(np.sum(reverse_visible)),
            "reverse_invisible_count": int(
                reverse_visible.size - np.sum(reverse_visible)
            ),
        },
        "direction_disagreement_diagnostic_px": {
            "n": int(direction_difference.size),
            "mean": float(np.mean(direction_difference)),
            "median": float(np.median(direction_difference)),
            "q90": float(np.quantile(direction_difference, 0.90)),
            "maximum": float(np.max(direction_difference)),
        },
        "thresholds": {
            "material_error_maximum_px": HALF_CELL_DIAGONAL_PX,
            "all_predictions_must_be_on_supplied_object_mask": True,
            "all_identity_assignments_must_be_correct": True,
            "all_pair_ratios_must_be_at_least": 0.5,
            "all_model_visibility_flags_must_be_true": True,
        },
        "visual_selection": "four largest material errors per direction; stable descending order",
        "visual_events": {
            "forward": forward_events,
            "reverse": reverse_events,
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
            "raw_predictions": raw_record,
            "raw_receipt": raw_receipt_record,
        },
        "raw_implementation_head": raw_receipt["implementation_head"],
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "command_argv": list(sys.argv),
        "arrays": file_record(arrays_path),
        "visuals": {
            "forward": file_record(forward_visual_path),
            "reverse": file_record(reverse_visual_path),
        },
        "raw_predictions_hashed_before_truth_or_supplied_masks_open": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": (
            "ten fixed witnesses over one correlated 180-frame orbit in two causal orders; "
            "descriptive only; no SEM, confidence interval, or population inference"
        ),
    }
    result_path = args.output_dir / "TAPNEXTPP_512_BIDIRECTIONAL_EVALUATION_RESULT.json"
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
