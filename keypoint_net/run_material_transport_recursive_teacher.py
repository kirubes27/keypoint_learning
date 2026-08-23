"""Run prediction-only recursive continuous-RGB teacher tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np

try:
    from .evaluate_material_transport_continuous_query import (
        MINIMUM_QUERY_RMS,
        ORDER_ATOL_PX,
        _ImageCache,
        continuous_rgb_match,
    )
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
    from .rgb_material_observability import RGBObservabilityConfig
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_material_transport_continuous_query import (
        MINIMUM_QUERY_RMS,
        ORDER_ATOL_PX,
        _ImageCache,
        continuous_rgb_match,
    )
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
    from rgb_material_observability import RGBObservabilityConfig


EXPECTED_AMENDMENT_SHA256 = "913398162f3795c062793f1423d548edcd7109b62d7d95417d1097ec02d41854"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_direction(
    *,
    paths: list[Path],
    initial_coordinate_px: np.ndarray,
    direction: str,
    witness_order: Iterable[int],
) -> dict[str, np.ndarray | float]:
    require(direction in {"forward", "reverse"}, "recursive direction differs")
    order = [int(value) for value in witness_order]
    require(sorted(order) == list(range(EXPECTED_WITNESSES)), "witness order is not a permutation")
    cache = _ImageCache(paths)
    config = RGBObservabilityConfig(minimum_query_rms=MINIMUM_QUERY_RMS)
    current = np.asarray(initial_coordinate_px, dtype=np.float64).copy()
    query = np.empty((EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), dtype=np.float64)
    prediction = np.empty_like(query)
    hard = np.empty_like(query)
    query_rms = np.empty((EXPECTED_FRAMES, EXPECTED_WITNESSES), dtype=np.float64)
    local_mass = np.empty_like(query_rms)
    source_frame_index = np.empty(EXPECTED_FRAMES, dtype=np.int64)
    target_frame_index = np.empty(EXPECTED_FRAMES, dtype=np.int64)
    maximum_reversal_difference = 0.0

    for step in range(EXPECTED_FRAMES):
        if direction == "forward":
            source_frame = step
            target_frame = (step + 1) % EXPECTED_FRAMES
        else:
            source_frame = (-step) % EXPECTED_FRAMES
            target_frame = (-step - 1) % EXPECTED_FRAMES
        source_frame_index[step] = source_frame
        target_frame_index[step] = target_frame
        source_rgb = cache.get(source_frame)
        target_rgb = cache.get(target_frame)
        next_coordinate = np.empty_like(current)
        for witness in order:
            query[step, witness] = current[witness]
            match = continuous_rgb_match(
                source_rgb,
                target_rgb,
                current[witness],
                config=config,
                verify_enumeration=True,
            )
            next_coordinate[witness] = match["coordinate_px"]
            prediction[step, witness] = match["coordinate_px"]
            hard[step, witness] = match["hard_coordinate_px"]
            query_rms[step, witness] = match["source_patch_rms"]
            local_mass[step, witness] = match["local_mass"]
            maximum_reversal_difference = max(
                maximum_reversal_difference,
                float(match["candidate_enumeration_reversal_difference_px"]),
            )
        current = next_coordinate

    return {
        "source_frame_index": source_frame_index,
        "target_frame_index": target_frame_index,
        "query_coordinate_px": query,
        "prediction_coordinate_px": prediction,
        "hard_coordinate_px": hard,
        "query_rms": query_rms,
        "local_mass": local_mass,
        "maximum_candidate_enumeration_reversal_difference_px": maximum_reversal_difference,
    }


def _require_exact(first: dict[str, np.ndarray | float], second: dict[str, np.ndarray | float]) -> None:
    require(set(first) == set(second), "recursive control keys differ")
    for key in first:
        left = first[key]
        right = second[key]
        if isinstance(left, np.ndarray):
            require(isinstance(right, np.ndarray) and np.array_equal(left, right), f"witness-order control changed {key}")
        else:
            require(float(left) == float(right), f"witness-order control changed {key}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    manifest = load_json(args.manifest)
    paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    initials_receipt_record = file_record(args.initials_receipt)
    initials_receipt = load_json(args.initials_receipt)
    require(initials_receipt["schema_version"] == "recursive_continuous_teacher_initials.v1", "initials schema differs")
    require(initials_receipt["sources"]["amendment"] == amendment_record, "initials amendment differs")
    initials_record = file_record(args.initials)
    require(initials_receipt["initials"] == initials_record, "initials binding differs")
    with np.load(args.initials) as archive:
        require(
            set(archive.files) == {"witness_id", "initial_frame_index", "initial_coordinate_px"},
            "initials expose unexpected arrays",
        )
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        initial_frame_index = int(archive["initial_frame_index"])
        initial_coordinate_px = np.asarray(archive["initial_coordinate_px"], dtype=np.float64)
    require(initial_frame_index == 0, "initial frame differs")
    require(witness_id.shape == (EXPECTED_WITNESSES,), "witness IDs differ")
    require(initial_coordinate_px.shape == (EXPECTED_WITNESSES, 2), "initial coordinate shape differs")

    canonical_order = list(range(EXPECTED_WITNESSES))
    reversed_order = list(reversed(canonical_order))
    forward = _run_direction(
        paths=paths,
        initial_coordinate_px=initial_coordinate_px,
        direction="forward",
        witness_order=canonical_order,
    )
    reverse = _run_direction(
        paths=paths,
        initial_coordinate_px=initial_coordinate_px,
        direction="reverse",
        witness_order=canonical_order,
    )
    forward_control = _run_direction(
        paths=paths,
        initial_coordinate_px=initial_coordinate_px,
        direction="forward",
        witness_order=reversed_order,
    )
    reverse_control = _run_direction(
        paths=paths,
        initial_coordinate_px=initial_coordinate_px,
        direction="reverse",
        witness_order=reversed_order,
    )
    _require_exact(forward, forward_control)
    _require_exact(reverse, reverse_control)
    maximum_enumeration_difference = max(
        float(forward["maximum_candidate_enumeration_reversal_difference_px"]),
        float(reverse["maximum_candidate_enumeration_reversal_difference_px"]),
    )
    require(maximum_enumeration_difference <= ORDER_ATOL_PX, "candidate enumeration control failed")

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_RECURSIVE_TEACHER_PREDICTIONS.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(initial_frame_index, dtype=np.int64),
        initial_coordinate_px=initial_coordinate_px,
        forward_source_frame_index=forward["source_frame_index"],
        forward_target_frame_index=forward["target_frame_index"],
        forward_query_coordinate_px=forward["query_coordinate_px"],
        forward_prediction_coordinate_px=forward["prediction_coordinate_px"],
        forward_hard_coordinate_px=forward["hard_coordinate_px"],
        forward_query_rms=forward["query_rms"],
        forward_local_mass=forward["local_mass"],
        reverse_source_frame_index=reverse["source_frame_index"],
        reverse_target_frame_index=reverse["target_frame_index"],
        reverse_query_coordinate_px=reverse["query_coordinate_px"],
        reverse_prediction_coordinate_px=reverse["prediction_coordinate_px"],
        reverse_hard_coordinate_px=reverse["hard_coordinate_px"],
        reverse_query_rms=reverse["query_rms"],
        reverse_local_mass=reverse["local_mass"],
    )
    receipt = {
        "schema_version": "raw_recursive_continuous_teacher_predictions.v1",
        "artifact_type": "prediction_only_recursive_rgb_tracks",
        "sources": {
            "amendment": amendment_record,
            "sanitized_manifest": manifest_record,
            "initials": initials_record,
            "initials_receipt": initials_receipt_record,
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "continuous_matcher": file_record(Path(__file__).with_name("evaluate_material_transport_continuous_query.py")),
            "rgb_primitives": file_record(Path(__file__).with_name("rgb_material_observability.py")),
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_rgb_hashes_rechecked_before_open": True,
            "only_frame_zero_initial_coordinates_opened": True,
            "non_frame_zero_truth_opened": False,
            "forward_reverse_independent": True,
            "witness_order_reversal_exact_every_array": True,
            "maximum_candidate_enumeration_reversal_difference_px": maximum_enumeration_difference,
            "candidate_enumeration_reversal_atol_px": ORDER_ATOL_PX,
            "model_optimizer_feature_geometry_or_operator_opened": False,
        },
        "runtime_seconds": float(time.time() - started),
        "training_or_weight_update_performed": False,
    }
    receipt_path = args.output_dir / "RAW_RECURSIVE_TEACHER_RECEIPT.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--initials", required=True, type=Path)
    parser.add_argument("--initials-receipt", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
