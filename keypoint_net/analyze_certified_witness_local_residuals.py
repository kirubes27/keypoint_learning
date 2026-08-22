"""Audit every residual from the fixed hard-centred local-3x3 readout."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from analyze_certified_witness_readouts import _infer
from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    FEATURE_SIZE,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    require,
    sha256_file,
)
from certified_witness_local_readout import (
    category_name,
    classify_localization_failures,
    compact_information_boundary,
    readout_arrays,
)
from run_certified_witness_capability import _load_bound_inputs


SEEDS = (42, 43, 44)


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


def _point(draw: ImageDraw.ImageDraw, point: np.ndarray, color: str, kind: str) -> None:
    x = float(point[0]) * 319.0 / 511.0
    y = float(point[1]) * 319.0 / 511.0
    if kind == "cross":
        draw.line((x - 6, y, x + 6, y), fill=color, width=3)
        draw.line((x, y - 6, x, y + 6), fill=color, width=3)
    elif kind == "square":
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
    else:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)


def _heatmap_image(logits: np.ndarray) -> Image.Image:
    value = np.asarray(logits, dtype=np.float64)
    value = np.clip((value - value.max() + 12.0) / 12.0, 0.0, 1.0)
    red = np.clip(255.0 * value, 0.0, 255.0)
    green = np.clip(255.0 * value**2, 0.0, 255.0)
    blue = np.clip(100.0 * (1.0 - value), 0.0, 255.0)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).resize((320, 320), Image.Resampling.NEAREST)


def _save_seed42_residual_montage(
    path: Path,
    images: np.ndarray,
    target_px: np.ndarray,
    global_px: np.ndarray,
    hard_px: np.ndarray,
    local_px: np.ndarray,
    logits: np.ndarray,
    hard_x: np.ndarray,
    hard_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    category: np.ndarray,
    within_half_cell: np.ndarray,
    on_object: np.ndarray,
    target_rank: np.ndarray,
    inside_mass: np.ndarray,
) -> int:
    residual = np.logical_not(within_half_cell) | np.logical_not(on_object)
    event_indices = np.argwhere(residual)
    require(event_indices.shape[0] > 0, "seed42 residual montage received no events")
    tile_width = 660
    tile_height = 374
    columns = 2
    rows = int(np.ceil(event_indices.shape[0] / columns))
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#181818")
    font = ImageFont.load_default()
    for event_index, (frame_value, witness_value) in enumerate(event_indices):
        frame = int(frame_value)
        witness = int(witness_value)
        tile = Image.new("RGB", (tile_width, tile_height), "#101010")
        label_draw = ImageDraw.Draw(tile)
        code = int(category[frame, witness])
        category_label = category_name(code) if code else "off_object_only"
        error = float(np.linalg.norm(local_px[frame, witness] - target_px[frame, witness]))
        label = (
            f"seed42 f{frame} w{EXPECTED_WITNESS_IDS[witness]} {category_label} "
            f"err={error:.3f}px rank={int(target_rank[frame, witness])} "
            f"mass3x3={float(inside_mass[frame, witness]):.4f} "
            f"on={bool(on_object[frame, witness])}"
        )
        label_draw.text((6, 5), label, fill="white", font=font)

        image_panel = Image.fromarray(images[frame]).resize((320, 320), Image.Resampling.LANCZOS)
        image_draw = ImageDraw.Draw(image_panel)
        _point(image_draw, target_px[frame, witness], "#00ff66", "cross")
        _point(image_draw, global_px[frame, witness], "#ff9900", "circle")
        _point(image_draw, hard_px[frame, witness], "#ff33ff", "square")
        _point(image_draw, local_px[frame, witness], "#00ffff", "circle")
        tile.paste(image_panel, (0, 38))

        heatmap_panel = _heatmap_image(logits[frame, witness])
        heatmap_draw = ImageDraw.Draw(heatmap_panel)
        scale = 320.0 / FEATURE_SIZE
        hx = int(hard_x[frame, witness])
        hy = int(hard_y[frame, witness])
        tx = int(target_x[frame, witness])
        ty = int(target_y[frame, witness])
        x0 = max(0, hx - 1) * scale
        x1 = min(FEATURE_SIZE, hx + 2) * scale - 1
        y0 = max(0, hy - 1) * scale
        y1 = min(FEATURE_SIZE, hy + 2) * scale - 1
        heatmap_draw.rectangle((x0, y0, x1, y1), outline="#ff33ff", width=2)
        heatmap_draw.line(
            ((tx + 0.5) * scale - 5, (ty + 0.5) * scale, (tx + 0.5) * scale + 5, (ty + 0.5) * scale),
            fill="#00ff66",
            width=2,
        )
        heatmap_draw.line(
            ((tx + 0.5) * scale, (ty + 0.5) * scale - 5, (tx + 0.5) * scale, (ty + 0.5) * scale + 5),
            fill="#00ff66",
            width=2,
        )
        heatmap_draw.rectangle(
            (
                hx * scale + 2,
                hy * scale + 2,
                (hx + 1) * scale - 2,
                (hy + 1) * scale - 2,
            ),
            outline="#ff33ff",
            width=2,
        )
        tile.paste(heatmap_panel, (330, 38))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((4, 356), "image: target green, global orange, hard magenta, local cyan", fill="#cccccc", font=font)
        tile_draw.text((334, 356), "heatmap: target green, selected 3x3 magenta", fill="#cccccc", font=font)
        canvas.paste(tile, ((event_index % columns) * tile_width, (event_index // columns) * tile_height))
    canvas.save(path)
    return int(event_indices.shape[0])


def _event_list(mask: np.ndarray) -> list[dict[str, int]]:
    return [
        {
            "frame": int(frame),
            "channel": int(witness),
            "witness_id": int(EXPECTED_WITNESS_IDS[int(witness)]),
        }
        for frame, witness in np.argwhere(mask)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh audit path")
    repository_head = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(repository_head == args.expected_repo_head, "repository HEAD differs from command lock")
    repository_status = subprocess.run(
        ["git", "-C", str(args.repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(repository_status == "", "repository is not clean")
    require(sha256_file(args.audit_lock) == args.expected_audit_lock_sha256, "audit-lock SHA-256 differs")
    require(
        sha256_file(args.previous_readout_arrays)
        == args.expected_previous_readout_arrays_sha256,
        "previous readout arrays SHA-256 differs",
    )
    _, dataset, target_px, masks, controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        EXPECTED_FRAMES,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")

    with np.load(args.previous_readout_arrays) as previous:
        previous_seed = np.asarray(previous["seed"], dtype=np.int64)
        previous_global = np.asarray(previous["global_soft_prediction_px"], dtype=np.float64)
        previous_hard = np.asarray(previous["hard_argmax_prediction_px"], dtype=np.float64)
        previous_local = np.asarray(previous["local_3x3_prediction_px"], dtype=np.float64)
        previous_rank = np.asarray(previous["target_nearest_cell_rank"], dtype=np.int64)
        previous_top1 = np.asarray(previous["top1_probability"], dtype=np.float64)
        previous_top2 = np.asarray(previous["top2_probability"], dtype=np.float64)
        previous_margin = np.asarray(previous["top1_top2_probability_margin"], dtype=np.float64)
        previous_entropy = np.asarray(previous["heatmap_entropy"], dtype=np.float64)
    require(np.array_equal(previous_seed, np.asarray(SEEDS)), "previous readout seed order differs")

    run_specs = (
        (42, args.seed42_checkpoint, args.seed42_checkpoint_sha256, args.seed42_predictions, args.seed42_predictions_sha256),
        (43, args.seed43_checkpoint, args.seed43_checkpoint_sha256, args.seed43_predictions, args.seed43_predictions_sha256),
        (44, args.seed44_checkpoint, args.seed44_checkpoint_sha256, args.seed44_predictions, args.seed44_predictions_sha256),
    )
    start = time.perf_counter()
    global_predictions: list[np.ndarray] = []
    hard_predictions: list[np.ndarray] = []
    local_predictions: list[np.ndarray] = []
    hard_x_values: list[np.ndarray] = []
    hard_y_values: list[np.ndarray] = []
    target_x_values: list[np.ndarray] = []
    target_y_values: list[np.ndarray] = []
    target_rank_values: list[np.ndarray] = []
    target_inside_values: list[np.ndarray] = []
    inside_mass_values: list[np.ndarray] = []
    outside_mass_values: list[np.ndarray] = []
    category_values: list[np.ndarray] = []
    within_values: list[np.ndarray] = []
    on_object_values: list[np.ndarray] = []
    identity_values: list[np.ndarray] = []
    distinct_pair_values: list[np.ndarray] = []
    seed_rows: list[dict[str, Any]] = []
    seed42_logits: np.ndarray | None = None

    for seed_index, (seed, checkpoint_path, checkpoint_sha, predictions_path, predictions_sha) in enumerate(run_specs):
        require(sha256_file(predictions_path) == predictions_sha, f"seed {seed} prediction SHA-256 differs")
        with np.load(predictions_path) as saved:
            saved_global = np.asarray(saved["prediction_coordinate_px"], dtype=np.float64)
        checkpoint, global_prediction, logits = _infer(
            checkpoint_path, checkpoint_sha, dataset, device, args.batch_size
        )
        require(np.array_equal(global_prediction, saved_global), f"seed {seed} saved global replay differs")
        readouts = readout_arrays(logits, target_px)
        require(np.array_equal(global_prediction, previous_global[seed_index]), f"seed {seed} previous global replay differs")
        require(np.array_equal(readouts["hard_prediction_px"], previous_hard[seed_index]), f"seed {seed} previous hard replay differs")
        require(np.array_equal(readouts["local_3x3_prediction_px"], previous_local[seed_index]), f"seed {seed} previous local replay differs")
        require(np.array_equal(readouts["target_nearest_cell_rank"], previous_rank[seed_index]), f"seed {seed} previous rank replay differs")
        require(np.array_equal(readouts["top1_probability"], previous_top1[seed_index]), f"seed {seed} previous top1 replay differs")
        require(np.array_equal(readouts["top2_probability"], previous_top2[seed_index]), f"seed {seed} previous top2 replay differs")
        require(np.array_equal(readouts["top1_top2_probability_margin"], previous_margin[seed_index]), f"seed {seed} previous margin replay differs")
        require(np.array_equal(readouts["heatmap_entropy"], previous_entropy[seed_index]), f"seed {seed} previous entropy replay differs")

        report, derived = evaluate_predictions(
            readouts["local_3x3_prediction_px"], target_px, masks
        )
        category, category_counts = classify_localization_failures(
            readouts, derived["within_half_cell"]
        )
        localization_failure = np.logical_not(derived["within_half_cell"])
        per_witness = [
            {
                "channel": channel,
                "witness_id": int(EXPECTED_WITNESS_IDS[channel]),
                "localization_failure_count": int(localization_failure[:, channel].sum()),
                "off_object_count": int(np.logical_not(derived["on_object"][:, channel]).sum()),
                "wrong_identity_count": int(np.logical_not(derived["identity_correct"][:, channel]).sum()),
                "has_any_localization_failure": bool(localization_failure[:, channel].any()),
            }
            for channel in range(EXPECTED_WITNESSES)
        ]
        seed_rows.append(
            {
                "seed": seed,
                "selected_update": int(checkpoint["update"]),
                "report": _compact_report(report),
                "localization_category_counts": category_counts,
                "channels_with_any_localization_failure": int(
                    sum(row["has_any_localization_failure"] for row in per_witness)
                ),
                "per_witness": per_witness,
                "localization_failure_set": _event_list(localization_failure),
                "checkpoint": file_record(checkpoint_path),
                "saved_predictions": file_record(predictions_path),
            }
        )
        global_predictions.append(global_prediction)
        hard_predictions.append(readouts["hard_prediction_px"])
        local_predictions.append(readouts["local_3x3_prediction_px"])
        hard_x_values.append(readouts["hard_cell_x"])
        hard_y_values.append(readouts["hard_cell_y"])
        target_x_values.append(readouts["target_cell_x"])
        target_y_values.append(readouts["target_cell_y"])
        target_rank_values.append(readouts["target_nearest_cell_rank"])
        target_inside_values.append(readouts["target_cell_inside_local_window"])
        inside_mass_values.append(readouts["inside_window_probability_mass"])
        outside_mass_values.append(readouts["outside_window_probability_mass"])
        category_values.append(category)
        within_values.append(derived["within_half_cell"])
        on_object_values.append(derived["on_object"])
        identity_values.append(derived["identity_correct"])
        distinct_pair_values.append(derived["distinct_pair"])
        if seed == 42:
            seed42_logits = logits

    global_stack = np.stack(global_predictions)
    hard_stack = np.stack(hard_predictions)
    local_stack = np.stack(local_predictions)
    hard_x_stack = np.stack(hard_x_values)
    hard_y_stack = np.stack(hard_y_values)
    target_x_stack = np.stack(target_x_values)
    target_y_stack = np.stack(target_y_values)
    rank_stack = np.stack(target_rank_values)
    target_inside_stack = np.stack(target_inside_values)
    inside_mass_stack = np.stack(inside_mass_values)
    outside_mass_stack = np.stack(outside_mass_values)
    category_stack = np.stack(category_values)
    within_stack = np.stack(within_values)
    on_object_stack = np.stack(on_object_values)
    identity_stack = np.stack(identity_values)
    distinct_pair_stack = np.stack(distinct_pair_values)
    failure_stack = np.logical_not(within_stack)
    failure_union = failure_stack.any(axis=0)
    failure_intersection = failure_stack.all(axis=0)
    require(seed42_logits is not None, "seed42 logits were not captured")

    seed42_residual = failure_stack[0] | np.logical_not(on_object_stack[0])
    seed42_details: list[dict[str, Any]] = []
    for frame_value, witness_value in np.argwhere(seed42_residual):
        frame = int(frame_value)
        witness = int(witness_value)
        code = int(category_stack[0, frame, witness])
        seed42_details.append(
            {
                "frame": frame,
                "channel": witness,
                "witness_id": int(EXPECTED_WITNESS_IDS[witness]),
                "localization_failed": bool(failure_stack[0, frame, witness]),
                "localization_category": category_name(code) if code else None,
                "off_object": bool(not on_object_stack[0, frame, witness]),
                "wrong_identity": bool(not identity_stack[0, frame, witness]),
                "target_coordinate_px": target_px[frame, witness].tolist(),
                "global_coordinate_px": global_stack[0, frame, witness].tolist(),
                "hard_coordinate_px": hard_stack[0, frame, witness].tolist(),
                "local_coordinate_px": local_stack[0, frame, witness].tolist(),
                "local_material_error_px": float(
                    np.linalg.norm(local_stack[0, frame, witness] - target_px[frame, witness])
                ),
                "hard_cell_xy": [
                    int(hard_x_stack[0, frame, witness]),
                    int(hard_y_stack[0, frame, witness]),
                ],
                "target_nearest_cell_xy": [
                    int(target_x_stack[0, frame, witness]),
                    int(target_y_stack[0, frame, witness]),
                ],
                "target_cell_inside_local_window": bool(
                    target_inside_stack[0, frame, witness]
                ),
                "target_nearest_cell_rank": int(rank_stack[0, frame, witness]),
                "inside_window_probability_mass": float(
                    inside_mass_stack[0, frame, witness]
                ),
                "outside_window_probability_mass": float(
                    outside_mass_stack[0, frame, witness]
                ),
                "top1_probability": float(previous_top1[0, frame, witness]),
                "top2_probability": float(previous_top2[0, frame, witness]),
                "top1_top2_probability_margin": float(
                    previous_margin[0, frame, witness]
                ),
            }
        )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "LOCAL_READOUT_RESIDUAL_AUDIT_ARRAYS.npz"
    residual_indices = np.argwhere(seed42_residual)
    np.savez_compressed(
        arrays_path,
        seed=np.asarray(SEEDS, dtype=np.int64),
        target_coordinate_px=target_px,
        global_prediction_px=global_stack,
        hard_prediction_px=hard_stack,
        local_3x3_prediction_px=local_stack,
        hard_cell_x=hard_x_stack,
        hard_cell_y=hard_y_stack,
        target_cell_x=target_x_stack,
        target_cell_y=target_y_stack,
        target_nearest_cell_rank=rank_stack,
        target_cell_inside_local_window=target_inside_stack,
        inside_window_probability_mass=inside_mass_stack,
        outside_window_probability_mass=outside_mass_stack,
        localization_category_code=category_stack,
        within_half_cell=within_stack,
        on_object=on_object_stack,
        identity_correct=identity_stack,
        distinct_pair=distinct_pair_stack,
        localization_failure_union=failure_union,
        localization_failure_intersection=failure_intersection,
        seed42_residual_frame=residual_indices[:, 0],
        seed42_residual_channel=residual_indices[:, 1],
        seed42_residual_logits=seed42_logits[seed42_residual],
    )
    montage_path = args.output_dir / "SEED42_ALL_LOCAL_RESIDUALS.png"
    visual_event_count = _save_seed42_residual_montage(
        montage_path,
        dataset.images,
        target_px,
        global_stack[0],
        hard_stack[0],
        local_stack[0],
        seed42_logits,
        hard_x_stack[0],
        hard_y_stack[0],
        target_x_stack[0],
        target_y_stack[0],
        category_stack[0],
        within_stack[0],
        on_object_stack[0],
        rank_stack[0],
        inside_mass_stack[0],
    )
    require(visual_event_count == len(seed42_details), "visual residual event count differs")

    result = {
        "schema_version": "certified_witness_local_readout_residual_audit.v1",
        "artifact_type": "source_bound_frozen_local_readout_residual_audit",
        "weights_optimized": False,
        "batchnorm_state_changed": False,
        "information_boundary": compact_information_boundary(),
        "previous_readout_replay_exact_all_seeds": True,
        "seed_rows": seed_rows,
        "localization_failure_union_count": int(failure_union.sum()),
        "localization_failure_intersection_count": int(failure_intersection.sum()),
        "localization_failure_union": _event_list(failure_union),
        "localization_failure_intersection": _event_list(failure_intersection),
        "seed42_residual_event_count": len(seed42_details),
        "seed42_residual_events": seed42_details,
        "phase_a_gate_pass": True,
        "phase_b_confirmation_authorized": True,
        "runtime_seconds": time.perf_counter() - start,
        "statistical_scope": {
            "inference": "descriptive_only",
            "optimization_seed_count": 3,
            "object_count": 1,
            "orbit_count": 1,
            "frame_values_independent": False,
            "sem_or_confidence_interval_computed": False,
        },
    }
    result_path = args.output_dir / "LOCAL_READOUT_RESIDUAL_AUDIT_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "certified_witness_local_readout_residual_audit_config.v1",
        "repository_head": repository_head,
        "audit_source": file_record(Path(__file__)),
        "local_readout_source": file_record(
            args.repo_root / "keypoint_net" / "certified_witness_local_readout.py"
        ),
        "audit_lock": file_record(args.audit_lock),
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "previous_readout_arrays": file_record(args.previous_readout_arrays),
        "device": str(device),
        "batch_size": args.batch_size,
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_local_readout_residual_audit_receipt.v1",
        "result": file_record(result_path),
        "arrays": file_record(arrays_path),
        "visual": file_record(montage_path),
        "config": file_record(config_path),
        "phase_a_gate_pass": True,
        "phase_b_confirmation_authorized": True,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--previous-readout-arrays", type=Path, required=True)
    parser.add_argument("--expected-previous-readout-arrays-sha256", required=True)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-checkpoint-sha256", required=True)
        parser.add_argument(f"--seed{seed}-predictions", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-predictions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"LOCAL READOUT RESIDUAL AUDIT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
