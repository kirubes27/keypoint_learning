"""Replay certified witness distributions through the radius-two RGB field."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evaluate_material_transport_witness_distribution_replay import (
        CELL_STEP_PX,
        GRID_SIZE,
        HALF_CELL_DIAGONAL_PX,
        _js,
        hard_centered_local_readout,
        transport_distribution,
    )
    from .material_transport_gate_io import file_record, load_json, require, write_json
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_material_transport_witness_distribution_replay import (
        CELL_STEP_PX,
        GRID_SIZE,
        HALF_CELL_DIAGONAL_PX,
        _js,
        hard_centered_local_readout,
        transport_distribution,
    )
    from material_transport_gate_io import file_record, load_json, require, write_json


EXPECTED_AMENDMENT_SHA256 = "fd063aaef58741f962e0ca0e9b8e9514bf70ad284fafc8b3014e75bc2c261bd3"
EXPECTED_PREVIOUS_ARRAYS_SHA256 = "6227a50533284d0ef03bb95b52fefdebde828e6e601e524d8bb8e015f83ba120"
EXPECTED_PREVIOUS_RESULT_SHA256 = "2909b1bbfeb864a81544ac400cd8ff3a93d4565351b268c300bc2f35e1bcbafa"
EXPECTED_FIELD_RECEIPT_SHA256 = "2b23b3198d62b44893c3539e09ddc1c457f8a3afb2965ee71c58f8bf73732e60"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
LOCAL_RADIUS_CELLS = 2
EXPECTED_RETAINED_COLUMNS = 25
COLUMN_REVERSAL_ATOL = 1.0e-7


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def radius_column_mask(offsets_xy: np.ndarray, radius_cells: int = LOCAL_RADIUS_CELLS) -> np.ndarray:
    offsets = np.asarray(offsets_xy, dtype=np.int64)
    require(offsets.ndim == 2 and offsets.shape[1] == 2, "offset layout differs")
    require(radius_cells == LOCAL_RADIUS_CELLS, "only the frozen radius is allowed")
    mask = np.max(np.abs(offsets), axis=1) <= radius_cells
    require(int(np.sum(mask)) == EXPECTED_RETAINED_COLUMNS, "retained column count differs")
    return mask


def condition_probability(probability: np.ndarray, keep_columns: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    keep = np.asarray(keep_columns, dtype=bool)
    require(value.ndim == 2 and keep.shape == (value.shape[1],), "conditioning shape differs")
    selected = value[:, keep]
    require(bool(np.isfinite(selected).all()) and bool((selected >= 0.0).all()), "selected probability is invalid")
    mass = np.sum(selected, axis=1, keepdims=True, dtype=np.float64)
    require(bool(np.isfinite(mass).all()) and bool((mass > 0.0).all()), "conditioned row has no mass")
    output = selected / mass
    require(
        bool(np.allclose(output.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0)),
        "conditioned row mass differs",
    )
    return output


def _target_in_window(readout: dict[str, Any], target_px: np.ndarray) -> bool:
    cell_xy = np.rint(np.asarray(target_px, dtype=np.float64) / CELL_STEP_PX).astype(np.int64)
    x0, x1, y0, y1 = readout["window_bounds"]
    return bool(x0 <= cell_xy[0] <= x1 and y0 <= cell_xy[1] <= y1)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    previous_arrays_record = file_record(args.previous_arrays)
    require(previous_arrays_record["sha256"] == EXPECTED_PREVIOUS_ARRAYS_SHA256, "previous arrays SHA-256 differs")
    previous_result_record = file_record(args.previous_result)
    require(previous_result_record["sha256"] == EXPECTED_PREVIOUS_RESULT_SHA256, "previous result SHA-256 differs")
    previous_result = load_json(args.previous_result)
    require(previous_result["decision"]["branch"] == "reject_exact_r64_centered_field", "previous replay branch differs")
    field_receipt_path = args.field_dir / "RGB_FIELD_RECEIPT.json"
    field_receipt_record = file_record(field_receipt_path)
    require(field_receipt_record["sha256"] == EXPECTED_FIELD_RECEIPT_SHA256, "field receipt SHA-256 differs")
    field_receipt = load_json(field_receipt_path)
    for record in field_receipt["field_arrays"].values():
        require(file_record(Path(record["absolute_path"])) == record, "field array binding differs")

    with np.load(args.previous_arrays) as previous:
        frame_index = np.asarray(previous["frame_index"], dtype=np.int64)
        witness_id = np.asarray(previous["witness_id"], dtype=np.int64)
        source_px = np.asarray(previous["source_coordinate_px"], dtype=np.float64)
        target_px = np.asarray(previous["target_coordinate_px"], dtype=np.float64)
        source_oracle = np.asarray(previous["source_oracle_distribution"], dtype=np.float64)
        target_oracle = np.asarray(previous["target_oracle_distribution"], dtype=np.float64)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "frame order differs")
    require(source_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "witness coordinate shape differs")
    require(source_oracle.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, GRID_SIZE * GRID_SIZE), "source distribution shape differs")

    with np.load(args.field_dir / "FIELD_LAYOUT.npz") as layout:
        offsets_xy = np.asarray(layout["offsets_xy"], dtype=np.int64)
        candidate_index_full = np.asarray(layout["candidate_index"], dtype=np.int64)
        self_column = int(layout["self_column"])
        require(np.array_equal(layout["edge_frame_index"], np.arange(EXPECTED_FRAMES)), "field edge order differs")
    keep = radius_column_mask(offsets_xy)
    retained_offsets = offsets_xy[keep]
    candidate_index = candidate_index_full[:, keep]
    require(bool(keep[self_column]), "self column was not retained")
    require(bool((np.max(np.abs(retained_offsets), axis=1) <= LOCAL_RADIUS_CELLS).all()), "retained offset exceeds radius")

    forward_probability = np.load(args.field_dir / "forward_probability.npy", mmap_mode="r")
    reverse_probability = np.load(args.field_dir / "reverse_probability.npy", mmap_mode="r")
    forward_row_valid = np.load(args.field_dir / "forward_row_valid.npy", mmap_mode="r")
    reverse_row_valid = np.load(args.field_dir / "reverse_row_valid.npy", mmap_mode="r")

    shape = (EXPECTED_FRAMES, EXPECTED_WITNESSES)
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
    maximum_column_reversal_difference = 0.0
    minimum_conditioned_mass = math.inf
    maximum_conditioned_mass = -math.inf

    for frame in range(EXPECTED_FRAMES):
        conditioned_forward = condition_probability(forward_probability[frame], keep)
        conditioned_reverse = condition_probability(reverse_probability[frame], keep)
        for full, conditioned in (
            (forward_probability[frame], conditioned_forward),
            (reverse_probability[frame], conditioned_reverse),
        ):
            reversed_conditioned = condition_probability(np.asarray(full)[:, ::-1], keep[::-1])[:, ::-1]
            maximum_column_reversal_difference = max(
                maximum_column_reversal_difference,
                float(np.max(np.abs(conditioned - reversed_conditioned))),
            )
            row_mass = conditioned.sum(axis=1)
            minimum_conditioned_mass = min(minimum_conditioned_mass, float(np.min(row_mass)))
            maximum_conditioned_mass = max(maximum_conditioned_mass, float(np.max(row_mass)))

        for witness in range(EXPECTED_WITNESSES):
            source_distribution = source_oracle[frame, witness]
            target_distribution = target_oracle[frame, witness]
            forward = transport_distribution(source_distribution, conditioned_forward, candidate_index)
            reverse = transport_distribution(target_distribution, conditioned_reverse, candidate_index)
            forward_readout = hard_centered_local_readout(forward)
            reverse_readout = hard_centered_local_readout(reverse)
            forward_prediction_px[frame, witness] = forward_readout["coordinate_px"]
            reverse_prediction_px[frame, witness] = reverse_readout["coordinate_px"]
            forward_hard_cell[frame, witness] = forward_readout["hard_cell"]
            reverse_hard_cell[frame, witness] = reverse_readout["hard_cell"]
            forward_window_mass[frame, witness] = forward_readout["window_mass"]
            reverse_window_mass[frame, witness] = reverse_readout["window_mass"]
            forward_target_in_window[frame, witness] = _target_in_window(forward_readout, target_px[frame, witness])
            reverse_target_in_window[frame, witness] = _target_in_window(reverse_readout, source_px[frame, witness])
            forward_support = np.flatnonzero(source_distribution > 0.0)
            reverse_support = np.flatnonzero(target_distribution > 0.0)
            forward_support_valid[frame, witness] = bool(np.all(forward_row_valid[frame, forward_support]))
            reverse_support_valid[frame, witness] = bool(np.all(reverse_row_valid[frame, reverse_support]))
            forward_js[frame, witness] = _js(forward, target_distribution)
            reverse_js[frame, witness] = _js(reverse, source_distribution)

    require(maximum_column_reversal_difference <= COLUMN_REVERSAL_ATOL, "column reversal control failed")
    forward_error_px = np.linalg.norm(forward_prediction_px - target_px, axis=-1)
    reverse_error_px = np.linalg.norm(reverse_prediction_px - source_px, axis=-1)
    forward_pass = forward_support_valid & forward_target_in_window & (forward_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    reverse_pass = reverse_support_valid & reverse_target_in_window & (reverse_error_px <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    witness_pass = np.all(forward_pass & reverse_pass, axis=0)

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "LOCAL_MOTION_REPLAY_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index,
        witness_id=witness_id,
        retained_offsets_xy=retained_offsets,
        source_coordinate_px=source_px,
        target_coordinate_px=target_px,
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
    witness_reports = []
    for witness, identifier in enumerate(witness_id):
        witness_reports.append(
            {
                "witness_id": int(identifier),
                "strict_bidirectional_all_edges": bool(witness_pass[witness]),
                "forward_pass_edges": int(np.sum(forward_pass[:, witness])),
                "reverse_pass_edges": int(np.sum(reverse_pass[:, witness])),
                "forward_maximum_error_px": float(np.max(forward_error_px[:, witness])),
                "reverse_maximum_error_px": float(np.max(reverse_error_px[:, witness])),
                "forward_maximum_js": float(np.max(forward_js[:, witness])),
                "reverse_maximum_js": float(np.max(reverse_js[:, witness])),
            }
        )
    all_ten = bool(np.all(witness_pass))
    result = {
        "schema_version": "certified_witness_rgb_field_local_motion_replay.v1",
        "artifact_type": "privileged_posthash_radius_two_representation_check",
        "decision": {
            "all_ten_bidirectional_all_edges_pass": all_ten,
            "branch": "derive_radius_two_site_costs_and_run_free_logits" if all_ten else "reject_r64_grid_centered_patch_queries",
        },
        "field_amendment": {
            "search_radius_cells": LOCAL_RADIUS_CELLS,
            "retained_offset_columns": int(np.sum(keep)),
            "original_offset_columns": int(keep.size),
            "cell_step_px": CELL_STEP_PX,
            "per_axis_coverage_px": LOCAL_RADIUS_CELLS * CELL_STEP_PX,
            "temperature_or_similarity_changed": False,
        },
        "controls": {
            "retained_column_count_exact": int(np.sum(keep)) == EXPECTED_RETAINED_COLUMNS,
            "self_column_retained": bool(keep[self_column]),
            "every_retained_offset_within_radius": bool((np.max(np.abs(retained_offsets), axis=1) <= LOCAL_RADIUS_CELLS).all()),
            "minimum_conditioned_row_mass": minimum_conditioned_mass,
            "maximum_conditioned_row_mass": maximum_conditioned_mass,
            "maximum_candidate_column_reversal_difference": maximum_column_reversal_difference,
            "candidate_column_reversal_atol": COLUMN_REVERSAL_ATOL,
            "candidate_column_reversal_pass": maximum_column_reversal_difference <= COLUMN_REVERSAL_ATOL,
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
        "witness_reports": witness_reports,
        "sources": {
            "amendment": amendment_record,
            "previous_arrays": previous_arrays_record,
            "previous_result": previous_result_record,
            "field_receipt": field_receipt_record,
        },
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "arrays": file_record(arrays_path),
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over 180 correlated frames; descriptive only",
    }
    result_path = args.output_dir / "LOCAL_MOTION_REPLAY_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", required=True, type=Path)
    parser.add_argument("--previous-arrays", required=True, type=Path)
    parser.add_argument("--previous-result", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
