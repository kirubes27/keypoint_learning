"""Privileged full-distribution replay at the certified continuous sites."""

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
EXPECTED_AMENDMENT_SHA256 = "7a9da0b1f92635be149ba25720b69e367463f4f6c3f31f164b1241b2b0c2489b"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
GRID_SIZE = 64
CELL_STEP_PX = (IMAGE_SIZE - 1) / (GRID_SIZE - 1)
HALF_CELL_DIAGONAL_PX = math.sqrt(2.0) * 0.5 * CELL_STEP_PX


def bilinear_grid_distribution(coordinate_px: np.ndarray) -> np.ndarray:
    """Splat finite pixel xy coordinates to endpoint-aligned r64 distributions."""

    coordinate = np.asarray(coordinate_px, dtype=np.float64)
    require(coordinate.shape[-1:] == (2,), "coordinate must end in xy")
    require(bool(np.isfinite(coordinate).all()), "coordinate contains non-finite values")
    cell = coordinate / CELL_STEP_PX
    require(bool(((cell >= 0.0) & (cell <= GRID_SIZE - 1)).all()), "coordinate lies outside grid")
    x0 = np.floor(cell[..., 0]).astype(np.int64)
    y0 = np.floor(cell[..., 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, GRID_SIZE - 1)
    y1 = np.minimum(y0 + 1, GRID_SIZE - 1)
    wx = cell[..., 0] - x0
    wy = cell[..., 1] - y0
    output = np.zeros(coordinate.shape[:-1] + (GRID_SIZE * GRID_SIZE,), dtype=np.float64)
    flat = output.reshape(-1, GRID_SIZE * GRID_SIZE)
    rows = np.arange(flat.shape[0])
    values = (
        (y0 * GRID_SIZE + x0, (1.0 - wx) * (1.0 - wy)),
        (y0 * GRID_SIZE + x1, wx * (1.0 - wy)),
        (y1 * GRID_SIZE + x0, (1.0 - wx) * wy),
        (y1 * GRID_SIZE + x1, wx * wy),
    )
    for indices, weights in values:
        np.add.at(flat, (rows, np.asarray(indices).reshape(-1)), np.asarray(weights).reshape(-1))
    require(bool(np.allclose(flat.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0)), "bilinear mass differs")
    grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float64), GRID_SIZE)
    grid_y = np.repeat(np.arange(GRID_SIZE, dtype=np.float64), GRID_SIZE)
    reproduced = np.stack((flat @ grid_x, flat @ grid_y), axis=-1).reshape(coordinate.shape)
    require(bool(np.allclose(reproduced, cell, atol=1.0e-12, rtol=0.0)), "bilinear coordinate reproduction differs")
    return output


def transport_distribution(
    source_probability: np.ndarray,
    conditional: np.ndarray,
    candidate_index: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_probability, dtype=np.float64)
    field = np.asarray(conditional, dtype=np.float64)
    require(source.shape == (GRID_SIZE * GRID_SIZE,), "source probability shape differs")
    require(field.shape == candidate_index.shape, "conditional shape differs")
    output = np.zeros_like(source)
    nonzero = np.flatnonzero(source > 0.0)
    for source_cell in nonzero:
        np.add.at(
            output,
            candidate_index[source_cell],
            source[source_cell] * field[source_cell],
        )
    require(bool(np.isfinite(output).all()), "transported distribution is non-finite")
    require(abs(float(output.sum()) - 1.0) <= 2.0e-6, "transported mass differs")
    return output


def hard_centered_local_readout(probability: np.ndarray) -> dict[str, Any]:
    value = np.asarray(probability, dtype=np.float64)
    require(value.shape == (GRID_SIZE * GRID_SIZE,), "probability shape differs")
    require(bool(np.isfinite(value).all()) and bool((value >= 0.0).all()), "probability is invalid")
    hard = int(np.argmax(value))
    hard_x = hard % GRID_SIZE
    hard_y = hard // GRID_SIZE
    x0 = max(0, hard_x - 1)
    x1 = min(GRID_SIZE - 1, hard_x + 1)
    y0 = max(0, hard_y - 1)
    y1 = min(GRID_SIZE - 1, hard_y + 1)
    grid = value.reshape(GRID_SIZE, GRID_SIZE)
    window = grid[y0 : y1 + 1, x0 : x1 + 1]
    mass = float(window.sum())
    require(mass > 0.0 and math.isfinite(mass), "local window has no finite mass")
    yy, xx = np.meshgrid(
        np.arange(y0, y1 + 1, dtype=np.float64),
        np.arange(x0, x1 + 1, dtype=np.float64),
        indexing="ij",
    )
    coordinate_cell = np.asarray(
        [float(np.sum(window * xx) / mass), float(np.sum(window * yy) / mass)],
        dtype=np.float64,
    )
    return {
        "hard_cell": hard,
        "hard_x": hard_x,
        "hard_y": hard_y,
        "window_bounds": (x0, x1, y0, y1),
        "window_mass": mass,
        "coordinate_cell": coordinate_cell,
        "coordinate_px": coordinate_cell * CELL_STEP_PX,
    }


def _js(first: np.ndarray, second: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).tiny
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    midpoint = 0.5 * (first + second)
    first_mask = first > 0.0
    second_mask = second > 0.0
    first_kl = np.sum(first[first_mask] * (np.log(first[first_mask]) - np.log(np.maximum(midpoint[first_mask], epsilon))))
    second_kl = np.sum(second[second_mask] * (np.log(second[second_mask]) - np.log(np.maximum(midpoint[second_mask], epsilon))))
    return float(0.5 * (first_kl + second_kl))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    capability_record = file_record(args.capability_manifest)
    require(capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256, "capability manifest SHA-256 differs")
    capability = load_json(args.capability_manifest)
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == capability["portable_tracks"]["sha256"], "track SHA-256 differs")
    field_receipt_path = args.field_dir / "RGB_FIELD_RECEIPT.json"
    field_receipt = load_json(field_receipt_path)
    require(field_receipt["execution_scope"] == "complete", "distribution replay requires complete field")
    require(field_receipt["controls"]["frame_processing_reversal_exact_every_array"] is True, "field order control failed")
    for record in field_receipt["field_arrays"].values():
        require(file_record(Path(record["absolute_path"])) == record, "field array binding differs")

    with np.load(args.tracks) as archive:
        frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        source_px = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "track frame order differs")
    require(source_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "track shape differs")
    target_px = np.roll(source_px, shift=-1, axis=0)
    source_oracle = bilinear_grid_distribution(source_px)
    target_oracle = bilinear_grid_distribution(target_px)

    with np.load(args.field_dir / "FIELD_LAYOUT.npz") as layout:
        candidate_index = np.asarray(layout["candidate_index"], dtype=np.int64)
        require(np.array_equal(layout["edge_frame_index"], np.arange(EXPECTED_FRAMES)), "field edge order differs")
    forward_probability = np.load(args.field_dir / "forward_probability.npy", mmap_mode="r")
    reverse_probability = np.load(args.field_dir / "reverse_probability.npy", mmap_mode="r")
    forward_row_valid = np.load(args.field_dir / "forward_row_valid.npy", mmap_mode="r")
    reverse_row_valid = np.load(args.field_dir / "reverse_row_valid.npy", mmap_mode="r")

    shape = (EXPECTED_FRAMES, EXPECTED_WITNESSES)
    forward_transport = np.empty(shape + (GRID_SIZE * GRID_SIZE,), dtype=np.float32)
    reverse_transport = np.empty_like(forward_transport)
    forward_prediction_px = np.empty(shape + (2,), dtype=np.float64)
    reverse_prediction_px = np.empty_like(forward_prediction_px)
    forward_hard_cell = np.empty(shape, dtype=np.int64)
    reverse_hard_cell = np.empty(shape, dtype=np.int64)
    forward_window_mass = np.empty(shape, dtype=np.float64)
    reverse_window_mass = np.empty(shape, dtype=np.float64)
    forward_target_in_window = np.zeros(shape, dtype=bool)
    reverse_target_in_window = np.zeros(shape, dtype=bool)
    forward_support_valid = np.zeros(shape, dtype=bool)
    reverse_support_valid = np.zeros(shape, dtype=bool)
    forward_js = np.empty(shape, dtype=np.float64)
    reverse_js = np.empty(shape, dtype=np.float64)

    for frame in range(EXPECTED_FRAMES):
        for witness in range(EXPECTED_WITNESSES):
            source_distribution = source_oracle[frame, witness]
            target_distribution = target_oracle[frame, witness]
            forward = transport_distribution(
                source_distribution,
                forward_probability[frame],
                candidate_index,
            )
            reverse = transport_distribution(
                target_distribution,
                reverse_probability[frame],
                candidate_index,
            )
            forward_transport[frame, witness] = forward.astype(np.float32)
            reverse_transport[frame, witness] = reverse.astype(np.float32)
            forward_readout = hard_centered_local_readout(forward)
            reverse_readout = hard_centered_local_readout(reverse)
            forward_prediction_px[frame, witness] = forward_readout["coordinate_px"]
            reverse_prediction_px[frame, witness] = reverse_readout["coordinate_px"]
            forward_hard_cell[frame, witness] = forward_readout["hard_cell"]
            reverse_hard_cell[frame, witness] = reverse_readout["hard_cell"]
            forward_window_mass[frame, witness] = forward_readout["window_mass"]
            reverse_window_mass[frame, witness] = reverse_readout["window_mass"]
            target_cell_xy = np.rint(target_px[frame, witness] / CELL_STEP_PX).astype(np.int64)
            source_cell_xy = np.rint(source_px[frame, witness] / CELL_STEP_PX).astype(np.int64)
            fx0, fx1, fy0, fy1 = forward_readout["window_bounds"]
            rx0, rx1, ry0, ry1 = reverse_readout["window_bounds"]
            forward_target_in_window[frame, witness] = bool(
                fx0 <= target_cell_xy[0] <= fx1 and fy0 <= target_cell_xy[1] <= fy1
            )
            reverse_target_in_window[frame, witness] = bool(
                rx0 <= source_cell_xy[0] <= rx1 and ry0 <= source_cell_xy[1] <= ry1
            )
            forward_support = np.flatnonzero(source_distribution > 0.0)
            reverse_support = np.flatnonzero(target_distribution > 0.0)
            forward_support_valid[frame, witness] = bool(np.all(forward_row_valid[frame, forward_support]))
            reverse_support_valid[frame, witness] = bool(np.all(reverse_row_valid[frame, reverse_support]))
            forward_js[frame, witness] = _js(forward, target_distribution)
            reverse_js[frame, witness] = _js(reverse, source_distribution)

    forward_error_px = np.linalg.norm(forward_prediction_px - target_px, axis=-1)
    reverse_error_px = np.linalg.norm(reverse_prediction_px - source_px, axis=-1)
    forward_pass = (
        forward_support_valid
        & forward_target_in_window
        & (forward_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    )
    reverse_pass = (
        reverse_support_valid
        & reverse_target_in_window
        & (reverse_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    )
    witness_pass = np.all(forward_pass & reverse_pass, axis=0)

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "WITNESS_DISTRIBUTION_REPLAY_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        source_coordinate_px=source_px,
        target_coordinate_px=target_px,
        source_oracle_distribution=source_oracle.astype(np.float32),
        target_oracle_distribution=target_oracle.astype(np.float32),
        forward_transport_distribution=forward_transport,
        reverse_transport_distribution=reverse_transport,
        forward_prediction_px=forward_prediction_px,
        reverse_prediction_px=reverse_prediction_px,
        forward_hard_cell=forward_hard_cell,
        reverse_hard_cell=reverse_hard_cell,
        forward_window_mass=forward_window_mass,
        reverse_window_mass=reverse_window_mass,
        forward_target_in_window=forward_target_in_window,
        reverse_target_in_window=reverse_target_in_window,
        forward_support_valid=forward_support_valid,
        reverse_support_valid=reverse_support_valid,
        forward_js=forward_js,
        reverse_js=reverse_js,
        forward_error_px=forward_error_px,
        reverse_error_px=reverse_error_px,
        forward_pass=forward_pass,
        reverse_pass=reverse_pass,
        witness_pass=witness_pass,
    )
    reports = []
    for witness, identifier in enumerate(witness_id):
        reports.append(
            {
                "witness_id": int(identifier),
                "strict_bidirectional_all_edges": bool(witness_pass[witness]),
                "forward_pass_edges": int(np.sum(forward_pass[:, witness])),
                "reverse_pass_edges": int(np.sum(reverse_pass[:, witness])),
                "forward_maximum_error_px": float(np.max(forward_error_px[:, witness])),
                "reverse_maximum_error_px": float(np.max(reverse_error_px[:, witness])),
                "forward_maximum_js": float(np.max(forward_js[:, witness])),
                "reverse_maximum_js": float(np.max(reverse_js[:, witness])),
                "forward_minimum_local_window_mass": float(np.min(forward_window_mass[:, witness])),
                "reverse_minimum_local_window_mass": float(np.min(reverse_window_mass[:, witness])),
            }
        )
    all_ten = bool(np.all(witness_pass))
    result = {
        "schema_version": "certified_witness_rgb_field_distribution_replay.v1",
        "artifact_type": "privileged_posthash_full_distribution_representation_check",
        "decision": {
            "all_ten_bidirectional_all_edges_pass": all_ten,
            "branch": "authorize_free_logit_optimization" if all_ten else "reject_exact_r64_centered_field",
        },
        "thresholds": {
            "half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "target_nearest_cell_must_be_inside_hard_centered_3x3": True,
            "all_nonzero_source_support_rows_must_be_valid": True,
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
            "amendment": amendment_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
            "field_receipt": file_record(field_receipt_path),
        },
        "arrays": file_record(arrays_path),
        "raw_field_or_objective_changed_after_single_cell_replay": False,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over 180 correlated frames; descriptive only",
    }
    result_path = args.output_dir / "WITNESS_DISTRIBUTION_REPLAY_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", required=True, type=Path)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
