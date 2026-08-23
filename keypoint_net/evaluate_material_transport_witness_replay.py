"""Privileged post-hash replay of the fixed RGB field at certified sites."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .material_transport_gate_io import file_record, load_json, require, write_json
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_gate_io import file_record, load_json, require, write_json


EXPECTED_CAPABILITY_MANIFEST_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
GRID_SIZE = 64
HALF_CELL_DIAGONAL_PX = math.sqrt(2.0) * 0.5 * (IMAGE_SIZE - 1) / (GRID_SIZE - 1)


def _grid_index(coordinate_px: np.ndarray) -> np.ndarray:
    cell = np.rint(coordinate_px * ((GRID_SIZE - 1) / (IMAGE_SIZE - 1))).astype(np.int64)
    require(bool(((0 <= cell) & (cell < GRID_SIZE)).all()), "coordinate maps outside grid")
    return cell[..., 1] * GRID_SIZE + cell[..., 0]


def _grid_pixel(flat_index: np.ndarray) -> np.ndarray:
    x = flat_index % GRID_SIZE
    y = flat_index // GRID_SIZE
    return np.stack((x, y), axis=-1).astype(np.float64) * ((IMAGE_SIZE - 1) / (GRID_SIZE - 1))


def _stable_rank(probability: np.ndarray, candidate_index: np.ndarray, target_cell: int) -> int:
    matches = np.flatnonzero(candidate_index == int(target_cell))
    if matches.size != 1:
        return 0
    column = int(matches[0])
    target_probability = probability[column]
    earlier = np.arange(probability.size) < column
    return int(np.sum((probability > target_probability) | ((probability == target_probability) & earlier))) + 1


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    capability_record = file_record(args.capability_manifest)
    require(capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256, "capability manifest SHA-256 differs")
    capability = load_json(args.capability_manifest)
    expected_tracks = capability["portable_tracks"]
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == expected_tracks["sha256"], "track SHA-256 differs")
    require(tracks_record["size_bytes"] == expected_tracks["size_bytes"], "track size differs")

    field_receipt_path = args.field_dir / "RGB_FIELD_RECEIPT.json"
    field_receipt = load_json(field_receipt_path)
    require(field_receipt["execution_scope"] == "complete", "witness replay requires complete field")
    require(field_receipt["edge_count"] == EXPECTED_FRAMES, "field edge count differs")
    require(field_receipt["controls"]["privileged_evaluation_files_opened"] is False, "raw field opened privileged files")
    require(field_receipt["controls"]["frame_processing_reversal_exact_every_array"] is True, "field frame-order control failed")
    for record in field_receipt["field_arrays"].values():
        path = Path(record["absolute_path"])
        require(file_record(path) == record, f"field array binding differs: {path}")
    require(file_record(Path(field_receipt["field_layout"]["absolute_path"])) == field_receipt["field_layout"], "field layout binding differs")

    with np.load(args.tracks) as track_archive:
        frame_index = np.asarray(track_archive["frame_index"], dtype=np.int64)
        witness_id = np.asarray(track_archive["witness_id"], dtype=np.int64)
        source_px = np.asarray(track_archive["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(track_archive["physical_valid"], dtype=bool)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "track frame order differs")
    require(witness_id.shape == (EXPECTED_WITNESSES,), "witness count differs")
    require(source_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "track coordinate shape differs")
    require(bool(physical_valid.all()), "a certified track coordinate is invalid")
    target_px = np.roll(source_px, shift=-1, axis=0)
    source_cell = _grid_index(source_px)
    target_cell = _grid_index(target_px)

    with np.load(args.field_dir / "FIELD_LAYOUT.npz") as layout:
        edge_frame_index = np.asarray(layout["edge_frame_index"], dtype=np.int64)
        candidate_index = np.asarray(layout["candidate_index"], dtype=np.int64)
    require(np.array_equal(edge_frame_index, np.arange(EXPECTED_FRAMES)), "field edge order differs")
    require(candidate_index.shape == (GRID_SIZE * GRID_SIZE, 81), "candidate layout shape differs")
    forward_probability = np.load(args.field_dir / "forward_probability.npy", mmap_mode="r")
    reverse_probability = np.load(args.field_dir / "reverse_probability.npy", mmap_mode="r")
    forward_row_valid = np.load(args.field_dir / "forward_row_valid.npy", mmap_mode="r")
    reverse_row_valid = np.load(args.field_dir / "reverse_row_valid.npy", mmap_mode="r")

    forward_prediction_cell = np.empty_like(source_cell)
    reverse_prediction_cell = np.empty_like(source_cell)
    forward_rank = np.zeros_like(source_cell)
    reverse_rank = np.zeros_like(source_cell)
    forward_valid = np.zeros_like(source_cell, dtype=bool)
    reverse_valid = np.zeros_like(source_cell, dtype=bool)
    for frame in range(EXPECTED_FRAMES):
        for witness in range(EXPECTED_WITNESSES):
            source = int(source_cell[frame, witness])
            target = int(target_cell[frame, witness])
            forward_row = np.asarray(forward_probability[frame, source], dtype=np.float64)
            reverse_row = np.asarray(reverse_probability[frame, target], dtype=np.float64)
            forward_column = int(np.argmax(forward_row))
            reverse_column = int(np.argmax(reverse_row))
            forward_prediction_cell[frame, witness] = candidate_index[source, forward_column]
            reverse_prediction_cell[frame, witness] = candidate_index[target, reverse_column]
            forward_rank[frame, witness] = _stable_rank(forward_row, candidate_index[source], target)
            reverse_rank[frame, witness] = _stable_rank(reverse_row, candidate_index[target], source)
            forward_valid[frame, witness] = bool(forward_row_valid[frame, source])
            reverse_valid[frame, witness] = bool(reverse_row_valid[frame, target])

    forward_prediction_px = _grid_pixel(forward_prediction_cell)
    reverse_prediction_px = _grid_pixel(reverse_prediction_cell)
    forward_error_px = np.linalg.norm(forward_prediction_px - target_px, axis=-1)
    reverse_error_px = np.linalg.norm(reverse_prediction_px - source_px, axis=-1)
    forward_pass = forward_valid & (forward_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    reverse_pass = reverse_valid & (reverse_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    witness_pass = np.all(forward_pass & reverse_pass, axis=0)

    reports = []
    for witness, identifier in enumerate(witness_id):
        reports.append(
            {
                "witness_id": int(identifier),
                "strict_bidirectional_all_edges": bool(witness_pass[witness]),
                "forward_pass_edges": int(np.sum(forward_pass[:, witness])),
                "reverse_pass_edges": int(np.sum(reverse_pass[:, witness])),
                "forward_top1_physical_cell_edges": int(np.sum(forward_rank[:, witness] == 1)),
                "reverse_top1_physical_cell_edges": int(np.sum(reverse_rank[:, witness] == 1)),
                "forward_maximum_error_px": float(np.max(forward_error_px[:, witness])),
                "reverse_maximum_error_px": float(np.max(reverse_error_px[:, witness])),
                "forward_median_rank": float(np.median(forward_rank[:, witness])),
                "reverse_median_rank": float(np.median(reverse_rank[:, witness])),
            }
        )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "WITNESS_FIELD_REPLAY_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        source_coordinate_px=source_px,
        target_coordinate_px=target_px,
        source_cell=source_cell,
        target_cell=target_cell,
        forward_prediction_cell=forward_prediction_cell,
        reverse_prediction_cell=reverse_prediction_cell,
        forward_prediction_px=forward_prediction_px,
        reverse_prediction_px=reverse_prediction_px,
        forward_error_px=forward_error_px,
        reverse_error_px=reverse_error_px,
        forward_rank=forward_rank,
        reverse_rank=reverse_rank,
        forward_valid=forward_valid,
        reverse_valid=reverse_valid,
        forward_pass=forward_pass,
        reverse_pass=reverse_pass,
        witness_pass=witness_pass,
    )
    all_ten_pass = bool(np.all(witness_pass))
    result = {
        "schema_version": "certified_witness_rgb_field_replay.v1",
        "artifact_type": "privileged_posthash_rgb_field_representation_check",
        "decision": {
            "all_ten_bidirectional_all_edges_pass": all_ten_pass,
            "branch": "authorize_free_logit_optimization" if all_ten_pass else "stop_field_representation_mismatch",
        },
        "thresholds": {
            "grid_size": GRID_SIZE,
            "image_size": IMAGE_SIZE,
            "half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "strict_rule": "valid and top1 grid-centre error <= half-cell diagonal in both directions on every cyclic edge",
        },
        "aggregate": {
            "strict_witness_count": int(np.sum(witness_pass)),
            "witness_count": EXPECTED_WITNESSES,
            "forward_pass_cases": int(np.sum(forward_pass)),
            "reverse_pass_cases": int(np.sum(reverse_pass)),
            "case_count_per_direction": EXPECTED_FRAMES * EXPECTED_WITNESSES,
            "forward_maximum_error_px": float(np.max(forward_error_px)),
            "reverse_maximum_error_px": float(np.max(reverse_error_px)),
        },
        "witness_reports": reports,
        "sources": {
            "capability_manifest": capability_record,
            "tracks": tracks_record,
            "field_receipt": file_record(field_receipt_path),
            "field_layout": field_receipt["field_layout"],
        },
        "arrays": file_record(arrays_path),
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over 180 correlated frames; descriptive only",
    }
    result_path = args.output_dir / "WITNESS_FIELD_REPLAY_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", required=True, type=Path)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
