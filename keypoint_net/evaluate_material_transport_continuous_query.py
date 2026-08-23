"""Evaluate continuous-coordinate adjacent RGB matching on certified witnesses."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image

try:
    from .evaluate_material_transport_witness_distribution_replay import HALF_CELL_DIAGONAL_PX
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
    from .rgb_material_observability import (
        PATCH_SIZES,
        RGBObservabilityConfig,
        candidate_coordinate_grids,
        patch_inside,
        rgb_correlation_map,
    )
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_material_transport_witness_distribution_replay import HALF_CELL_DIAGONAL_PX
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
    from rgb_material_observability import (
        PATCH_SIZES,
        RGBObservabilityConfig,
        candidate_coordinate_grids,
        patch_inside,
        rgb_correlation_map,
    )


EXPECTED_AMENDMENT_SHA256 = "6b1fdac13fe262ed80da9030ba64b645725221755cee09a868b030e0d0703d23"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
EXPECTED_CAPABILITY_MANIFEST_SHA256 = "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
PATCH_SIZE = 35
SEARCH_RADIUS_PX = 16.0
LOCAL_RADIUS_PX = 1.0
TEMPERATURE = 0.05
MINIMUM_QUERY_RMS = 1.0e-8
ORDER_ATOL_PX = 1.0e-10


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    require(array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"RGB shape differs: {path}")
    return np.ascontiguousarray(array)


class _ImageCache:
    def __init__(self, paths: list[Path], maximum_entries: int = 3) -> None:
        self.paths = paths
        self.maximum_entries = maximum_entries
        self.values: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame: int) -> np.ndarray:
        if frame in self.values:
            value = self.values.pop(frame)
            self.values[frame] = value
            return value
        value = _load_rgb(self.paths[frame])
        self.values[frame] = value
        while len(self.values) > self.maximum_entries:
            self.values.popitem(last=False)
        return value


def decode_continuous_score_map(
    scores: np.ndarray,
    source_coordinate_px: np.ndarray,
    *,
    reverse_enumeration: bool = False,
) -> dict[str, Any]:
    """Decode a local score mode with enumeration-independent tie semantics."""

    value = np.asarray(scores, dtype=np.float32)
    source = np.asarray(source_coordinate_px, dtype=np.float64)
    require(source.shape == (2,) and bool(np.isfinite(source).all()), "source coordinate is invalid")
    xx, yy = candidate_coordinate_grids(value.shape, PATCH_SIZE)
    allowed = (
        (np.abs(xx - source[0]) <= SEARCH_RADIUS_PX)
        & (np.abs(yy - source[1]) <= SEARCH_RADIUS_PX)
        & np.isfinite(value)
    )
    canonical_index = np.flatnonzero(allowed.reshape(-1))
    require(canonical_index.size > 0, "continuous matcher has no candidates")
    enumeration = canonical_index[::-1] if reverse_enumeration else canonical_index
    selected_score = value.reshape(-1)[enumeration].astype(np.float64)
    maximum = float(np.max(selected_score))
    unnormalized = np.exp((selected_score - maximum) / TEMPERATURE)
    require(bool(np.isfinite(unnormalized).all()) and bool((unnormalized >= 0.0).all()), "conditional is invalid")
    mass = float(np.sum(unnormalized, dtype=np.float64))
    require(mass > 0.0 and math.isfinite(mass), "conditional has no mass")
    probability = unnormalized / mass
    canonical_probability = np.zeros(value.size, dtype=np.float64)
    canonical_probability[enumeration] = probability
    require(
        abs(float(np.sum(canonical_probability)) - 1.0) <= 1.0e-12,
        "conditional mass differs",
    )

    allowed_scores = value.reshape(-1)[canonical_index]
    hard_score = float(np.max(allowed_scores))
    tied = canonical_index[allowed_scores == hard_score]
    hard_flat = int(np.min(tied))
    hard_y, hard_x = np.unravel_index(hard_flat, value.shape)
    hard_coordinate = np.asarray([xx[hard_y, hard_x], yy[hard_y, hard_x]], dtype=np.float64)
    local = allowed & (
        (np.abs(xx - hard_coordinate[0]) <= LOCAL_RADIUS_PX)
        & (np.abs(yy - hard_coordinate[1]) <= LOCAL_RADIUS_PX)
    )
    local_probability = canonical_probability.reshape(value.shape) * local
    local_mass = float(np.sum(local_probability, dtype=np.float64))
    require(local_mass > 0.0 and math.isfinite(local_mass), "local conditional has no mass")
    coordinate = np.asarray(
        [
            float(np.sum(local_probability * xx, dtype=np.float64) / local_mass),
            float(np.sum(local_probability * yy, dtype=np.float64) / local_mass),
        ],
        dtype=np.float64,
    )
    return {
        "coordinate_px": coordinate,
        "hard_coordinate_px": hard_coordinate,
        "hard_score": hard_score,
        "local_mass": local_mass,
        "candidate_count": int(canonical_index.size),
        "conditional_sum": float(np.sum(canonical_probability)),
        "conditional_minimum": float(np.min(canonical_probability[canonical_index])),
        "conditional_maximum": float(np.max(canonical_probability[canonical_index])),
    }


def continuous_rgb_match(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_coordinate_px: np.ndarray,
    *,
    config: RGBObservabilityConfig,
    verify_enumeration: bool,
) -> dict[str, Any]:
    scores, evidence = rgb_correlation_map(
        source_rgb,
        target_rgb,
        source_coordinate_px,
        PATCH_SIZE,
        config=config,
    )
    require(scores is not None, "continuous query patch is invalid")
    decoded = decode_continuous_score_map(scores, source_coordinate_px)
    reversal_difference = 0.0
    if verify_enumeration:
        reversed_decoded = decode_continuous_score_map(
            scores,
            source_coordinate_px,
            reverse_enumeration=True,
        )
        reversal_difference = float(
            np.max(np.abs(decoded["coordinate_px"] - reversed_decoded["coordinate_px"]))
        )
        require(reversal_difference <= ORDER_ATOL_PX, "candidate enumeration control failed")
    return {
        **decoded,
        "source_patch_rms": float(evidence["source_patch_rms"]),
        "candidate_enumeration_reversal_difference_px": reversal_difference,
    }


def target_in_selected_window(target_px: np.ndarray, hard_coordinate_px: np.ndarray) -> bool:
    target = np.asarray(target_px, dtype=np.float64)
    hard = np.asarray(hard_coordinate_px, dtype=np.float64)
    nearest = np.rint(target)
    return bool(
        patch_inside(nearest, PATCH_SIZE)
        and np.max(np.abs(nearest - hard)) <= LOCAL_RADIUS_PX
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    amendment_record = file_record(args.amendment)
    require(amendment_record["sha256"] == EXPECTED_AMENDMENT_SHA256, "amendment SHA-256 differs")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    capability_record = file_record(args.capability_manifest)
    require(capability_record["sha256"] == EXPECTED_CAPABILITY_MANIFEST_SHA256, "capability SHA-256 differs")
    capability = load_json(args.capability_manifest)
    tracks_record = file_record(args.tracks)
    require(tracks_record["sha256"] == capability["portable_tracks"]["sha256"], "track SHA-256 differs")
    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)

    with np.load(args.tracks) as archive:
        frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        source_px = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "track frame order differs")
    require(source_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "track shape differs")
    target_px = np.roll(source_px, shift=-1, axis=0)
    edge_count = EXPECTED_FRAMES if args.edge_limit is None else int(args.edge_limit)
    require(1 <= edge_count <= EXPECTED_FRAMES, "edge limit is outside 1..180")

    config = RGBObservabilityConfig(minimum_query_rms=MINIMUM_QUERY_RMS)
    require(PATCH_SIZE in PATCH_SIZES, "patch size is not supported")
    cache = _ImageCache(rgb_paths)
    shape = (edge_count, EXPECTED_WITNESSES)
    forward_prediction = np.empty(shape + (2,), dtype=np.float64)
    reverse_prediction = np.empty_like(forward_prediction)
    forward_hard = np.empty_like(forward_prediction)
    reverse_hard = np.empty_like(forward_prediction)
    forward_error = np.empty(shape, dtype=np.float64)
    reverse_error = np.empty(shape, dtype=np.float64)
    forward_target_window = np.zeros(shape, dtype=bool)
    reverse_target_window = np.zeros(shape, dtype=bool)
    forward_query_rms = np.empty(shape, dtype=np.float64)
    reverse_query_rms = np.empty(shape, dtype=np.float64)
    forward_local_mass = np.empty(shape, dtype=np.float64)
    reverse_local_mass = np.empty(shape, dtype=np.float64)
    forward_candidate_count = np.empty(shape, dtype=np.int64)
    reverse_candidate_count = np.empty(shape, dtype=np.int64)
    maximum_reversal_difference = 0.0

    for frame in range(edge_count):
        target_frame = (frame + 1) % EXPECTED_FRAMES
        source_rgb = cache.get(frame)
        target_rgb = cache.get(target_frame)
        for witness in range(EXPECTED_WITNESSES):
            forward = continuous_rgb_match(
                source_rgb,
                target_rgb,
                source_px[frame, witness],
                config=config,
                verify_enumeration=True,
            )
            reverse = continuous_rgb_match(
                target_rgb,
                source_rgb,
                target_px[frame, witness],
                config=config,
                verify_enumeration=True,
            )
            forward_prediction[frame, witness] = forward["coordinate_px"]
            reverse_prediction[frame, witness] = reverse["coordinate_px"]
            forward_hard[frame, witness] = forward["hard_coordinate_px"]
            reverse_hard[frame, witness] = reverse["hard_coordinate_px"]
            forward_error[frame, witness] = float(
                np.linalg.norm(forward["coordinate_px"] - target_px[frame, witness])
            )
            reverse_error[frame, witness] = float(
                np.linalg.norm(reverse["coordinate_px"] - source_px[frame, witness])
            )
            forward_target_window[frame, witness] = target_in_selected_window(
                target_px[frame, witness], forward["hard_coordinate_px"]
            )
            reverse_target_window[frame, witness] = target_in_selected_window(
                source_px[frame, witness], reverse["hard_coordinate_px"]
            )
            forward_query_rms[frame, witness] = forward["source_patch_rms"]
            reverse_query_rms[frame, witness] = reverse["source_patch_rms"]
            forward_local_mass[frame, witness] = forward["local_mass"]
            reverse_local_mass[frame, witness] = reverse["local_mass"]
            forward_candidate_count[frame, witness] = forward["candidate_count"]
            reverse_candidate_count[frame, witness] = reverse["candidate_count"]
            maximum_reversal_difference = max(
                maximum_reversal_difference,
                forward["candidate_enumeration_reversal_difference_px"],
                reverse["candidate_enumeration_reversal_difference_px"],
            )

    forward_pass = forward_target_window & (forward_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    reverse_pass = reverse_target_window & (reverse_error <= HALF_CELL_DIAGONAL_PX + 1.0e-12)
    complete = edge_count == EXPECTED_FRAMES
    witness_pass = np.all(forward_pass & reverse_pass, axis=0) if complete else np.zeros(EXPECTED_WITNESSES, dtype=bool)
    all_ten = bool(complete and np.all(witness_pass))

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "CONTINUOUS_QUERY_REPLAY_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        frame_index=frame_index[:edge_count],
        witness_id=witness_id,
        source_coordinate_px=source_px[:edge_count],
        target_coordinate_px=target_px[:edge_count],
        forward_prediction_px=forward_prediction,
        reverse_prediction_px=reverse_prediction,
        forward_hard_coordinate_px=forward_hard,
        reverse_hard_coordinate_px=reverse_hard,
        forward_error_px=forward_error,
        reverse_error_px=reverse_error,
        forward_target_in_window=forward_target_window,
        reverse_target_in_window=reverse_target_window,
        forward_query_rms=forward_query_rms,
        reverse_query_rms=reverse_query_rms,
        forward_local_mass=forward_local_mass,
        reverse_local_mass=reverse_local_mass,
        forward_candidate_count=forward_candidate_count,
        reverse_candidate_count=reverse_candidate_count,
        forward_pass=forward_pass,
        reverse_pass=reverse_pass,
        witness_pass=witness_pass,
    )
    witness_reports = []
    for witness, identifier in enumerate(witness_id):
        witness_reports.append(
            {
                "witness_id": int(identifier),
                "strict_bidirectional_all_edges": bool(witness_pass[witness]) if complete else None,
                "forward_pass_edges": int(np.sum(forward_pass[:, witness])),
                "reverse_pass_edges": int(np.sum(reverse_pass[:, witness])),
                "evaluated_edges": edge_count,
                "forward_maximum_error_px": float(np.max(forward_error[:, witness])),
                "reverse_maximum_error_px": float(np.max(reverse_error[:, witness])),
            }
        )
    result = {
        "schema_version": "certified_witness_continuous_rgb_query_replay.v1",
        "artifact_type": "privileged_posthash_continuous_query_representation_check",
        "execution_scope": "complete" if complete else "smoke",
        "decision": {
            "all_ten_bidirectional_all_edges_pass": all_ten if complete else None,
            "branch": (
                "implement_differentiable_continuous_matcher"
                if all_ten
                else "reject_continuous_rgb_identity_substrate"
                if complete
                else "smoke_only_no_scientific_branch"
            ),
        },
        "matcher": {
            "patch_size_px": PATCH_SIZE,
            "search_radius_px": SEARCH_RADIUS_PX,
            "local_readout_radius_px": LOCAL_RADIUS_PX,
            "temperature": TEMPERATURE,
            "minimum_query_rms": MINIMUM_QUERY_RMS,
            "method": "cv2.TM_CCOEFF_NORMED",
        },
        "controls": {
            "all_rgb_hashes_rechecked_before_open": True,
            "forward_and_reverse_independently_computed": True,
            "maximum_candidate_enumeration_reversal_difference_px": maximum_reversal_difference,
            "candidate_enumeration_reversal_atol_px": ORDER_ATOL_PX,
            "candidate_enumeration_reversal_pass": maximum_reversal_difference <= ORDER_ATOL_PX,
            "model_optimizer_feature_geometry_or_operator_opened_by_matcher": False,
        },
        "thresholds": {
            "half_cell_diagonal_px": HALF_CELL_DIAGONAL_PX,
            "target_nearest_integer_candidate_must_be_inside_hard_centered_3x3": True,
        },
        "aggregate": {
            "strict_witness_count": int(np.sum(witness_pass)) if complete else None,
            "witness_count": EXPECTED_WITNESSES,
            "evaluated_edges": edge_count,
            "forward_pass_cases": int(np.sum(forward_pass)),
            "reverse_pass_cases": int(np.sum(reverse_pass)),
            "case_count_per_direction": edge_count * EXPECTED_WITNESSES,
            "forward_maximum_error_px": float(np.max(forward_error)),
            "reverse_maximum_error_px": float(np.max(reverse_error)),
        },
        "witness_reports": witness_reports,
        "sources": {
            "amendment": amendment_record,
            "sanitized_manifest": manifest_record,
            "capability_manifest": capability_record,
            "tracks": tracks_record,
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "continuous_evaluator": file_record(Path(__file__)),
            "rgb_primitives": file_record(Path(__file__).with_name("rgb_material_observability.py")),
            "gate_io": file_record(Path(__file__).with_name("material_transport_gate_io.py")),
        },
        "arrays": file_record(arrays_path),
        "runtime_seconds": float(time.time() - started),
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over correlated adjacent edges; descriptive only",
    }
    result_path = args.output_dir / "CONTINUOUS_QUERY_REPLAY_RESULT.json"
    write_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capability-manifest", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--edge-limit", type=int)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
