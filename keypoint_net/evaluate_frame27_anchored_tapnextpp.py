"""Privileged post-hash evaluation of frame-27 anchored TAPNext++ tracks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    "e023344fca7abca6bf9727a409ffc00b7028b994b86b9a24fffc552cef79f4d0"
)
EXPECTED_MANIFEST_SHA256 = (
    "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
)
EXPECTED_VALIDATION_TRUTH_SHA256 = (
    "3e188cd699cbddfe10ce51a3c1de97f7be95f958012219ed2699f4c0d565819b"
)
EXPECTED_MASK_MANIFEST_SHA256 = (
    "d4a02868b6bd645f106703b88af59513d0aa2eb8ba2ebef6eac64683af667b00"
)
EXPECTED_RAW_SCHEMA = "raw_frame27_anchored_tapnextpp_512_support64.v1"
EXPECTED_FRAMES = np.arange(24, dtype=np.int64)
EXPECTED_WITNESSES = 10
TWO_CELL_SPACING_PX = 2.0 * 511.0 / 63.0
HEADER_HEIGHT = 74


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_targets(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        frame_index = np.asarray(loaded["frame_index"], dtype=np.int64)
        witness_id = np.asarray(loaded["witness_id"], dtype=np.int64)
        target_px = np.asarray(loaded["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(loaded["physical_valid"], dtype=bool)
        target_on_object = np.asarray(loaded["target_on_object"], dtype=bool)
    require(np.array_equal(frame_index, EXPECTED_FRAMES), "validation frame order differs")
    require(tuple(witness_id.tolist()) == tuple(EXPECTED_WITNESS_IDS), "validation witness order differs")
    require(target_px.shape == (24, EXPECTED_WITNESSES, 2), "validation target shape differs")
    require(bool(physical_valid.all() and target_on_object.all()), "validation target validity differs")
    return frame_index, witness_id, target_px


def _load_masks(manifest_path: Path, object_root: Path, frames: np.ndarray) -> np.ndarray:
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema_version")
        == "leakage_safe_distillation_evaluation_mask_manifest.v1",
        "mask manifest schema differs",
    )
    records = {int(row["frame_index"]): row for row in manifest["frames"]}
    missing = sorted(set(map(int, frames.tolist())) - set(records))
    require(not missing, f"mask manifest omits frames: {missing}")
    masks = np.empty((len(frames), 512, 512), dtype=bool)
    for local_index, frame in enumerate(frames.tolist()):
        row = records[frame]
        path = object_root / str(row["mask_relpath"])
        require(file_record(path)["sha256"] == row["mask_sha256"], f"mask hash differs: {frame}")
        with Image.open(path) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        require(mask.shape == (512, 512), "mask shape differs")
        masks[local_index] = mask
    return masks


def select_branch(report: dict[str, Any], all_within_two: bool) -> str:
    if bool(report["strict_capability_pass"]):
        return "heldout_wedge_fix_detect_once_track_thereafter"
    violations = report["violations"]
    if (
        all_within_two
        and int(violations["wrong_identity_count"]) == 0
        and int(violations["collapsed_pair_count"]) == 0
        and int(violations["off_object_count"]) == 0
    ):
        return "bounded_temporal_support_residual_continuous_refinement_needed"
    return "off_the_shelf_continuation_insufficient_design_domain_adapted_query_tracker"


def visibility_diagnostic(
    visible: Any, within_half: Any, material_error: Any
) -> dict[str, Any]:
    visibility = np.asarray(visible, dtype=bool)
    within = np.asarray(within_half, dtype=bool)
    error = np.asarray(material_error, dtype=np.float64)
    require(visibility.shape == within.shape == error.shape, "visibility diagnostic shape differs")
    invisible = ~visibility
    return {
        "visible_count": int(visibility.sum()),
        "invisible_count": int(invisible.sum()),
        "invisible_but_within_half_cell_count": int((invisible & within).sum()),
        "visible_but_outside_half_cell_count": int((visibility & ~within).sum()),
        "visible_material_error_mean_px": float(error[visibility].mean())
        if visibility.any()
        else None,
        "invisible_material_error_mean_px": float(error[invisible].mean())
        if invisible.any()
        else None,
        "visibility_used_to_suppress_predictions": False,
    }


def _mark(
    image: Image.Image, xy: np.ndarray, color: tuple[int, int, int], label: str
) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw = ImageDraw.Draw(image)
    radius = 9
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.line((x - 13, y, x + 13, y), fill=color, width=3)
    draw.line((x, y - 13, x, y + 13), fill=color, width=3)
    draw.text((x + 11, y - 12), label, fill=color, stroke_width=2, stroke_fill="black")


def _render_worst(
    *,
    rgb_paths: list[Path],
    truth: np.ndarray,
    prediction: np.ndarray,
    visible: np.ndarray,
    error: np.ndarray,
    witness_id: np.ndarray,
    output_path: Path,
) -> list[dict[str, Any]]:
    flat_order = np.argsort(error.reshape(-1), kind="stable")[::-1][:8]
    canvas = Image.new(
        "RGB", (4 * 512, 2 * (512 + HEADER_HEIGHT)), "white"
    )
    events: list[dict[str, Any]] = []
    for panel, flat in enumerate(flat_order):
        frame, witness = np.unravel_index(int(flat), error.shape)
        with Image.open(rgb_paths[frame]) as opened:
            image = opened.convert("RGB")
        _mark(image, truth[frame, witness], (0, 210, 0), "TRUE")
        _mark(image, prediction[frame, witness], (255, 0, 255), "TRACK")
        ImageDraw.Draw(image).line(
            (*truth[frame, witness], *prediction[frame, witness]),
            fill=(255, 215, 0),
            width=4,
        )
        wrapped = Image.new("RGB", (512, 512 + HEADER_HEIGHT), "white")
        wrapped.paste(image, (0, HEADER_HEIGHT))
        title = (
            f"frame {frame}  witness {int(witness_id[witness])}\n"
            f"error {error[frame, witness]:.3f}px; visible {bool(visible[frame, witness])}"
        )
        ImageDraw.Draw(wrapped).multiline_text(
            (10, 8), title, fill="black", font=ImageFont.load_default(), spacing=4
        )
        canvas.paste(
            wrapped,
            ((panel % 4) * 512, (panel // 4) * (512 + HEADER_HEIGHT)),
        )
        events.append(
            {
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")

    # Bind and load raw predictions before opening validation truth or masks.
    raw_receipt_record = file_record(args.raw_receipt)
    raw_receipt = load_json(args.raw_receipt)
    require(raw_receipt.get("schema_version") == EXPECTED_RAW_SCHEMA, "raw schema differs")
    require(raw_receipt.get("privileged_evaluation_authorized") is True, "raw stage blocks evaluation")
    raw_arrays_path = Path(str(raw_receipt["raw_predictions"]["absolute_path"]))
    require(file_record(raw_arrays_path) == raw_receipt["raw_predictions"], "raw prediction binding differs")
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
    require(raw_receipt["sources"]["sanitized_manifest"] == manifest_record, "raw manifest binding differs")
    for label, record in raw_receipt["implementation_sources"].items():
        require(file_record(Path(str(record["absolute_path"]))) == record, f"raw source differs: {label}")
    controls = raw_receipt["controls"]
    require(controls["frame0_to_23_truth_opened"] is False, "raw stage opened validation truth")
    require(controls["supplied_masks_opened"] is False, "raw stage opened masks")
    require(controls["validation_score_map_arrays_read"] is False, "raw stage read validation scores")
    require(controls["local_laptop_gpu_used"] is False, "raw stage used laptop GPU")
    require(controls["training_or_weight_update_performed"] is False, "raw stage trained")
    require(raw_receipt["traversal"]["order"] == list(range(27, -1, -1)), "raw traversal differs")
    require(raw_receipt["traversal"]["continuous_without_reset"] is True, "raw traversal reset")

    truth_record = file_record(args.validation_truth)
    mask_record = file_record(args.mask_manifest)
    require(truth_record["sha256"] == EXPECTED_VALIDATION_TRUTH_SHA256, "validation truth differs")
    require(mask_record["sha256"] == EXPECTED_MASK_MANIFEST_SHA256, "mask manifest differs")
    frame_index, witness_id, target_px = _load_targets(args.validation_truth)
    object_root = args.object_root.resolve(strict=True)
    masks = _load_masks(args.mask_manifest, object_root, frame_index)
    rgb_paths = resolve_rgb_paths(load_json(args.manifest), object_root_override=object_root)

    require(np.array_equal(raw["frame_index"], np.arange(28)), "raw frame index differs")
    require(tuple(raw["witness_id"].tolist()) == tuple(EXPECTED_WITNESS_IDS), "raw witness order differs")
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
    within_two = derived["material_error_px"] <= TWO_CELL_SPACING_PX + 1e-12
    all_within_two = bool(within_two.all())
    branch = select_branch(report, all_within_two)
    visibility = visibility_diagnostic(
        visible, derived["within_half_cell"], derived["material_error_px"]
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "FRAME27_ANCHORED_TAPNEXTPP_EVALUATION_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        target_coordinate_px=target_px,
        prediction_xy=prediction_px,
        visible=visible,
        material_error_px=derived["material_error_px"],
        within_half_cell=derived["within_half_cell"],
        within_two_cells=within_two,
        identity_correct=derived["identity_correct"],
        assigned_identity=derived["assigned_identity"],
        on_object=derived["on_object"],
        distinct_pair=derived["distinct_pair"],
        prediction_pair_distance_px=derived["prediction_pair_distance_px"],
        target_pair_distance_px=derived["target_pair_distance_px"],
    )
    worst_path = args.output_dir / "01_WORST_FRAME27_ANCHORED_EVENTS.png"
    worst_events = _render_worst(
        rgb_paths=rgb_paths,
        truth=target_px,
        prediction=prediction_px,
        visible=visible,
        error=derived["material_error_px"],
        witness_id=witness_id,
        output_path=worst_path,
    )
    result_path = args.output_dir / "FRAME27_ANCHORED_TAPNEXTPP_RESULT.json"
    result = {
        "schema_version": "frame27_anchored_tapnextpp_evaluation.v1",
        "artifact_type": "privileged_posthash_frame27_anchored_track_evaluation",
        "implementation_head": implementation_head,
        "raw_implementation_head": raw_head,
        "raw_loaded_and_hash_verified_before_truth_or_masks": raw_loaded_before_privileged_inputs,
        "sample_scope": {
            "object_count": 1,
            "heldout_frame_count": 24,
            "witness_count": 10,
            "event_count": 240,
            "frames_correlated": True,
            "statistics": "descriptive only; no error bars, SEM, CI, or hypothesis test",
            "cross_object_or_full_orbit_generalization_authorized": False,
        },
        "material_result": report,
        "practical_two_cell": {
            "threshold_px": TWO_CELL_SPACING_PX,
            "within_count": int(within_two.sum()),
            "event_count": int(within_two.size),
            "all_within": all_within_two,
        },
        "visibility_diagnostic": visibility,
        "decision": {
            "branch": branch,
            "strict_heldout_wedge_fix": bool(report["strict_capability_pass"]),
            "operator_authorized": False,
            "training_authorized": False,
            "gpu_run_authorized": False,
            "next_if_strict_pass": "test longer horizon and second object before operator use",
            "next_if_failure": "design renderer-supervised domain-adapted query tracker; do not resume per-frame global detection sweeps",
        },
        "thresholds": {
            "strict_half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "practical_two_cell_spacing_px": TWO_CELL_SPACING_PX,
        },
        "worst_visual_events": worst_events,
        "controls": {
            "continuous_raw_coordinates_evaluated_without_snapping": True,
            "visibility_used_to_hide_predictions": False,
            "training_or_weight_update_performed": False,
            "laptop_gpu_used": False,
            "validation_truth_opened_only_after_raw_prediction_hash": True,
        },
        "bindings": {
            "semantic_lock": lock_record,
            "raw_receipt": raw_receipt_record,
            "raw_predictions": file_record(raw_arrays_path),
            "validation_truth": truth_record,
            "mask_manifest": mask_record,
            "rgb_manifest": manifest_record,
            "evaluation_arrays": file_record(arrays_path),
            "worst_figure": file_record(worst_path),
        },
        "implementation_sources": {
            "evaluator": file_record(Path(__file__)),
            "capability_contract": file_record(
                Path(__file__).with_name("certified_witness_capability.py")
            ),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
        },
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "FRAME27_ANCHORED_TAPNEXTPP_EVALUATION_RECEIPT.json"
    receipt = {
        "schema_version": "frame27_anchored_tapnextpp_evaluation_receipt.v1",
        "implementation_head": implementation_head,
        "raw_implementation_head": raw_head,
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "worst_figure": file_record(worst_path),
        "decision_branch": branch,
        "strict_heldout_wedge_fix": bool(report["strict_capability_pass"]),
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
