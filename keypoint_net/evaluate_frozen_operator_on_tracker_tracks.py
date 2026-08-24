"""Transfer six frozen oracle-trained affine operators to tracker tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from material_transport_gate_io import file_record, load_json, require, write_json
from run_frame27_anchored_tapnextpp import EXPECTED_WITNESS_IDS


EXPECTED_LOCK_SHA256 = (
    "a9c13caf2244e266b204dda9e2ce0526522fbaf432f5518080a7c52f40539696"
)
EXPECTED_HISTORICAL_HEAD = "97b1bd144f5da2d825c63d03e62829d876873e76"
EXPECTED_HISTORICAL_RESULT_SHA256 = (
    "9026155947d5f3d0dfe0867d7683abe4d408d3525663861b1b001770cee21bac"
)
EXPECTED_HISTORICAL_RESULT_CONTENT_SHA256 = (
    "6667caa27054a84b4a1559aef7783973513f82ca995d8620c64a6fbba65351e2"
)
EXPECTED_HISTORICAL_MANIFEST_SHA256 = (
    "69ab744a53481712507939cba31b6926355ca04f9ec1ff0f316f9acc3358cad0"
)
EXPECTED_HISTORICAL_METRIC_SOURCE_SHA256 = (
    "bac722cb7118d89e48ca98bf54c7cc1b1731cdaf933e1a19b455680e3952946f"
)
EXPECTED_VALIDATION_PAIRS_SHA256 = (
    "3e71c4f862a99d1882a8704140f8706612460d804a6ed7a833c8dee9f35514a4"
)
EXPECTED_CERTIFIED_ARRAYS_SHA256 = (
    "28a709f2e4694bf3a73969eac9abad6838c87ec5f2a58884f17d74e68295b3c5"
)
EXPECTED_DETECTOR_ARRAYS_SHA256 = (
    "703be9111822d102b4e40425acaf19c916d7b3a6c63ee8948404225121be5d04"
)
EXPECTED_OPERATOR_KEYS = tuple(
    (recipe, seed)
    for recipe in ("task55_clean", "task80_assisted")
    for seed in (42, 43, 44)
)
EXPECTED_FRAMES = np.arange(24, dtype=np.int64)


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rotation_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def pixel_xy_to_normalized(points_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    require(points.shape[-1] == 2, "point coordinate shape differs")
    require(bool(np.isfinite(points).all()), "point coordinate is non-finite")
    return 2.0 * points / 511.0 - 1.0


def operator_metrics(
    source: Any,
    target: Any,
    A: Any,
    bias: Any,
    criteria: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    matrix = np.asarray(A, dtype=np.float64)
    offset = np.asarray(bias, dtype=np.float64)
    require(source_array.shape == target_array.shape, "operator source/target shape differs")
    require(source_array.ndim == 3 and source_array.shape[1:] == (10, 2), "operator pair shape differs")
    require(matrix.shape == (2, 2) and offset.shape == (2,), "operator parameter shape differs")
    require(
        bool(
            np.isfinite(source_array).all()
            and np.isfinite(target_array).all()
            and np.isfinite(matrix).all()
            and np.isfinite(offset).all()
        ),
        "operator input is non-finite",
    )

    target_A = rotation_matrix(6.0)
    prediction = source_array @ matrix.T + offset
    squared_error = (prediction - target_array) ** 2
    mse = float(np.mean(squared_error))
    identity_mse = float(np.mean((source_array - target_array) ** 2))
    u, _singular_values, vt = np.linalg.svd(matrix)
    proper = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
    angle = math.degrees(math.atan2(float(proper[1, 0]), float(proper[0, 0])))
    signed_error = (angle - 6.0 + 180.0) % 360.0 - 180.0
    determinant = float(np.linalg.det(matrix))
    metrics = {
        "learned_A": matrix.tolist(),
        "learned_bias": offset.tolist(),
        "proper_rotation_angle_degrees": angle,
        "signed_angle_error_degrees": signed_error,
        "absolute_angle_error_degrees": abs(signed_error),
        "matrix_frobenius_error": float(np.linalg.norm(matrix - target_A, ord="fro")),
        "bias_l2": float(np.linalg.norm(offset)),
        "determinant": determinant,
        "wrong_locked_sign": angle < 0.0,
        "improper_or_reflection": determinant <= 0.0,
        "validation_pair_mse": mse,
        "validation_identity_mse": identity_mse,
        "validation_identity_normalized_mse": (
            mse / identity_mse if identity_mse > 0.0 else math.inf
        ),
    }
    checks = {
        "absolute_angle_error_degrees": metrics["absolute_angle_error_degrees"]
        <= float(criteria["absolute_angle_error_degrees_max"]),
        "matrix_frobenius_error": metrics["matrix_frobenius_error"]
        <= float(criteria["matrix_frobenius_error_max"]),
        "bias_l2": metrics["bias_l2"] <= float(criteria["bias_l2_max"]),
        "validation_pair_mse": metrics["validation_pair_mse"]
        <= float(criteria["validation_pair_mse_max"]),
        "validation_identity_normalized_mse": metrics[
            "validation_identity_normalized_mse"
        ]
        <= float(criteria["validation_identity_normalized_mse_max"]),
        "wrong_locked_sign": metrics["wrong_locked_sign"] is False,
        "improper_or_reflection": metrics["improper_or_reflection"] is False,
    }
    metrics["criterion_checks"] = checks
    metrics["passes_all_criteria"] = bool(all(checks.values()))
    pair_mse = squared_error.mean(axis=(1, 2))
    return metrics, pair_mse


def validate_pair_rows(pair_index: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    require(pair_index.get("schema_version") == "representation_pair_index.v1", "pair schema differs")
    require(pair_index.get("split") == "validation", "pair split differs")
    require(int(pair_index.get("pair_count", -1)) == 21, "pair count differs")
    rows = pair_index.get("pairs")
    require(isinstance(rows, list) and len(rows) == 21, "pair rows differ")
    source: list[int] = []
    target: list[int] = []
    pair_ids: list[str] = []
    for row in rows:
        require(row.get("model_name") == "engineers_hammer_vray", "pair object differs")
        require(row.get("object_role") == "development", "pair role differs")
        require(row.get("direction") == "forward", "pair direction differs")
        require(row.get("physical_axis") == "world_z", "pair axis differs")
        require(int(row.get("stride", -1)) == 3, "pair stride differs")
        require(float(row.get("signed_generator", math.nan)) == 6.0, "pair generator differs")
        source.append(int(row["src_frame_index"]))
        target.append(int(row["dst_frame_index"]))
        pair_ids.append(str(row["pair_id"]))
    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    require(np.array_equal(source_array, np.arange(21)), "source frame set differs")
    require(np.array_equal(target_array, np.arange(3, 24)), "target frame set differs")
    return source_array, target_array, pair_ids


def aggregate_arm(cells: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(cells) == 6, "arm must contain six operator cells")
    recipes: dict[str, Any] = {}
    for recipe in ("task55_clean", "task80_assisted"):
        selected = [cell for cell in cells if cell["recipe"] == recipe]
        require(len(selected) == 3, f"{recipe} cell count differs")
        passing = int(sum(bool(cell["metrics"]["passes_all_criteria"]) for cell in selected))
        recipes[recipe] = {
            "seed_count": 3,
            "passing_seed_count": passing,
            "recipe_succeeds": passing >= 2,
            "validation_pair_mse": {
                "mean_over_three_fixed_operator_seeds": float(
                    np.mean([cell["metrics"]["validation_pair_mse"] for cell in selected])
                ),
                "sample_standard_deviation_ddof_1": float(
                    np.std(
                        [cell["metrics"]["validation_pair_mse"] for cell in selected],
                        ddof=1,
                    )
                ),
                "n_fixed_operator_seeds": 3,
                "descriptive_only": True,
            },
            "validation_identity_normalized_mse": {
                "mean_over_three_fixed_operator_seeds": float(
                    np.mean(
                        [
                            cell["metrics"]["validation_identity_normalized_mse"]
                            for cell in selected
                        ]
                    )
                ),
                "sample_standard_deviation_ddof_1": float(
                    np.std(
                        [
                            cell["metrics"]["validation_identity_normalized_mse"]
                            for cell in selected
                        ],
                        ddof=1,
                    )
                ),
                "n_fixed_operator_seeds": 3,
                "descriptive_only": True,
            },
        }
    return {
        "recipes": recipes,
        "arm_succeeds": bool(all(record["recipe_succeeds"] for record in recipes.values())),
    }


def select_decision(aggregates: Mapping[str, Mapping[str, Any]]) -> str:
    reference = bool(aggregates["reference_material_targets"]["arm_succeeds"])
    certified = bool(aggregates["certified_anchor_tracker"]["arm_succeeds"])
    detector = bool(aggregates["detector_initialized_tracker"]["arm_succeeds"])
    if not reference:
        return "stop_reference_coordinate_or_parity_failure"
    if certified and detector:
        return "both_tracker_arms_operator_compatible_short_horizon"
    if certified and not detector:
        return "certified_tracker_compatible_detector_initialization_blocks_bridge"
    return "tracker_coordinates_not_operator_compatible"


def _load_track_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"frame_index", "witness_id", "prediction_xy", "target_coordinate_px"}
        require(required.issubset(loaded.files), "tracker evaluation arrays omit required fields")
        value = {name: np.asarray(loaded[name]) for name in required}
    require(np.array_equal(value["frame_index"], EXPECTED_FRAMES), "tracker frame order differs")
    require(tuple(value["witness_id"].tolist()) == tuple(EXPECTED_WITNESS_IDS), "tracker witness order differs")
    require(value["prediction_xy"].shape == (24, 10, 2), "tracker prediction shape differs")
    require(value["target_coordinate_px"].shape == (24, 10, 2), "tracker target shape differs")
    return value


def _historical_metric_parity(
    historical_metrics: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> dict[str, Any]:
    keys = (
        "proper_rotation_angle_degrees",
        "signed_angle_error_degrees",
        "absolute_angle_error_degrees",
        "matrix_frobenius_error",
        "bias_l2",
        "determinant",
    )
    differences = {
        key: abs(float(recomputed[key]) - float(historical_metrics[key])) for key in keys
    }
    require(max(differences.values()) <= 1e-12, "historical matrix metric parity failed")
    return {
        "absolute_differences": differences,
        "maximum_absolute_difference": max(differences.values()),
        "tolerance": 1e-12,
        "pass": True,
    }


def _render(cells_by_arm: Mapping[str, list[dict[str, Any]]], output_path: Path) -> None:
    arm_order = (
        "reference_material_targets",
        "certified_anchor_tracker",
        "detector_initialized_tracker",
    )
    labels = ("material target\nreference", "certified-init\ntracker", "detector-init\ntracker")
    colors = ("#1b9e77", "#377eb8", "#e66101")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for cell_index, operator_key in enumerate(EXPECTED_OPERATOR_KEYS):
        mse = []
        normalized = []
        for arm in arm_order:
            cell = next(
                value
                for value in cells_by_arm[arm]
                if (value["recipe"], value["seed"]) == operator_key
            )
            mse.append(cell["metrics"]["validation_pair_mse"])
            normalized.append(cell["metrics"]["validation_identity_normalized_mse"])
        operator_label = f"{operator_key[0].replace('_', ' ')} s{operator_key[1]}"
        axes[0].plot(range(3), mse, marker="o", alpha=0.72, label=operator_label)
        axes[1].plot(range(3), normalized, marker="o", alpha=0.72, label=operator_label)
    axes[0].axhline(1e-4, color="black", linestyle="--", label="pass threshold")
    axes[1].axhline(0.1, color="black", linestyle="--", label="pass threshold")
    for axis, title, ylabel in (
        (axes[0], "Frozen operator pair error", "validation pair MSE"),
        (axes[1], "Recovery relative to identity baseline", "pair MSE / identity MSE"),
    ):
        axis.set_yscale("log")
        axis.set_xticks(range(3), labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    figure.suptitle("Six frozen oracle-trained operators; all 21 held-out pairs and ten points")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    require(_git(args.repo_root, "status", "--porcelain") == "", "implementation repo is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(implementation_head == args.expected_repo_head, "implementation HEAD differs")
    require(
        _git(args.historical_repo_root, "status", "--porcelain", "--untracked-files=no") == "",
        "historical tracked worktree is dirty",
    )
    historical_head = _git(args.historical_repo_root, "rev-parse", "HEAD")
    require(historical_head == EXPECTED_HISTORICAL_HEAD, "historical HEAD differs")

    records = {
        "semantic_lock": file_record(args.semantic_lock),
        "historical_result": file_record(args.historical_result),
        "historical_manifest": file_record(args.historical_manifest),
        "historical_metric_source": file_record(args.historical_metric_source),
        "validation_pairs": file_record(args.validation_pairs),
        "certified_tracker_arrays": file_record(args.certified_tracker_arrays),
        "detector_tracker_arrays": file_record(args.detector_tracker_arrays),
    }
    expected_hashes = {
        "semantic_lock": EXPECTED_LOCK_SHA256,
        "historical_result": EXPECTED_HISTORICAL_RESULT_SHA256,
        "historical_manifest": EXPECTED_HISTORICAL_MANIFEST_SHA256,
        "historical_metric_source": EXPECTED_HISTORICAL_METRIC_SOURCE_SHA256,
        "validation_pairs": EXPECTED_VALIDATION_PAIRS_SHA256,
        "certified_tracker_arrays": EXPECTED_CERTIFIED_ARRAYS_SHA256,
        "detector_tracker_arrays": EXPECTED_DETECTOR_ARRAYS_SHA256,
    }
    for label, expected in expected_hashes.items():
        require(records[label]["sha256"] == expected, f"{label} hash differs")

    historical = load_json(args.historical_result)
    content_copy = dict(historical)
    observed_content_hash = str(content_copy.pop("result_content_sha256", ""))
    require(observed_content_hash == EXPECTED_HISTORICAL_RESULT_CONTENT_SHA256, "historical content field differs")
    require(_canonical_sha256(content_copy) == observed_content_hash, "historical canonical content hash differs")
    require(historical.get("causal_interpretation_authorized") is True, "historical result is not authorized")
    require(historical.get("outcome") == "representation_or_identity_sliding_bottleneck", "historical outcome differs")

    manifest = load_json(args.historical_manifest)
    criteria = manifest.get("positive_pass_criteria", {}).get("per_cell")
    require(isinstance(criteria, Mapping), "historical criteria are missing")
    require(
        criteria
        == {
            "absolute_angle_error_degrees_max": 0.25,
            "matrix_frobenius_error_max": 0.01,
            "bias_l2_max": 0.005,
            "validation_pair_mse_max": 0.0001,
            "validation_identity_normalized_mse_max": 0.1,
            "wrong_locked_sign": False,
            "improper_or_reflection": False,
        },
        "historical pass criteria differ",
    )

    historical_cells = historical.get("oracle_cells")
    require(isinstance(historical_cells, list) and len(historical_cells) == 6, "historical oracle cells differ")
    cell_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    for cell in historical_cells:
        key = (str(cell["recipe"]), int(cell["seed"]))
        require(key not in cell_map, "historical operator cell repeats")
        require(bool(cell["metrics"]["passes_all_criteria"]), "historical oracle cell did not pass")
        cell_map[key] = cell
    require(tuple(sorted(cell_map)) == tuple(sorted(EXPECTED_OPERATOR_KEYS)), "historical operator set differs")

    pair_index = load_json(args.validation_pairs)
    source_frames, target_frames, pair_ids = validate_pair_rows(pair_index)
    certified = _load_track_arrays(args.certified_tracker_arrays)
    detector = _load_track_arrays(args.detector_tracker_arrays)
    require(
        np.array_equal(certified["target_coordinate_px"], detector["target_coordinate_px"]),
        "tracker validation targets differ",
    )
    arms_px = {
        "reference_material_targets": certified["target_coordinate_px"],
        "certified_anchor_tracker": certified["prediction_xy"],
        "detector_initialized_tracker": detector["prediction_xy"],
    }
    arms = {label: pixel_xy_to_normalized(value) for label, value in arms_px.items()}

    cells_by_arm: dict[str, list[dict[str, Any]]] = {label: [] for label in arms}
    pair_errors: dict[str, dict[str, list[float]]] = {label: {} for label in arms}
    parity_records: dict[str, Any] = {}
    for operator_key in EXPECTED_OPERATOR_KEYS:
        historical_cell = cell_map[operator_key]
        historical_metrics = historical_cell["metrics"]
        A = np.asarray(historical_metrics["learned_A"], dtype=np.float64)
        bias = np.asarray(historical_metrics["learned_bias"], dtype=np.float64)
        for arm_label, coordinates in arms.items():
            metrics, per_pair_mse = operator_metrics(
                coordinates[source_frames],
                coordinates[target_frames],
                A,
                bias,
                criteria,
            )
            case_id = f"{arm_label}__{operator_key[0]}__seed{operator_key[1]}"
            cells_by_arm[arm_label].append(
                {
                    "case_id": case_id,
                    "recipe": operator_key[0],
                    "seed": operator_key[1],
                    "frozen_historical_case_id": historical_cell["case_id"],
                    "metrics": metrics,
                }
            )
            pair_errors[arm_label][case_id] = per_pair_mse.tolist()
            if arm_label == "reference_material_targets":
                parity_records[historical_cell["case_id"]] = _historical_metric_parity(
                    historical_metrics, metrics
                )

    aggregates = {
        arm_label: aggregate_arm(cells) for arm_label, cells in cells_by_arm.items()
    }
    decision = select_decision(aggregates)
    args.output_dir.mkdir(parents=True)
    pair_arrays_path = args.output_dir / "FROZEN_OPERATOR_TRACKER_TRANSFER_PAIR_ERRORS.npz"
    np.savez_compressed(
        pair_arrays_path,
        pair_id=np.asarray(pair_ids),
        source_frame=source_frames,
        target_frame=target_frames,
        witness_id=np.asarray(EXPECTED_WITNESS_IDS, dtype=np.int64),
        reference_coordinate_normalized=arms["reference_material_targets"],
        certified_tracker_coordinate_normalized=arms["certified_anchor_tracker"],
        detector_tracker_coordinate_normalized=arms["detector_initialized_tracker"],
        **{
            f"pair_mse__{arm_label}__{recipe}__seed{seed}": np.asarray(
                pair_errors[arm_label][f"{arm_label}__{recipe}__seed{seed}"],
                dtype=np.float64,
            )
            for arm_label in arms
            for recipe, seed in EXPECTED_OPERATOR_KEYS
        },
    )
    figure_path = args.output_dir / "01_FROZEN_OPERATOR_TRACKER_TRANSFER.png"
    _render(cells_by_arm, figure_path)
    result_path = args.output_dir / "FROZEN_OPERATOR_TRACKER_TRANSFER_RESULT.json"
    result = {
        "schema_version": "frozen_operator_tracker_transfer.v1",
        "artifact_type": "posthash_frozen_operator_transfer_evaluation",
        "implementation_head": implementation_head,
        "historical_head": historical_head,
        "decision": {
            "branch": decision,
            "reference_arm_succeeds": bool(aggregates["reference_material_targets"]["arm_succeeds"]),
            "certified_tracker_arm_succeeds": bool(aggregates["certified_anchor_tracker"]["arm_succeeds"]),
            "detector_tracker_arm_succeeds": bool(aggregates["detector_initialized_tracker"]["arm_succeeds"]),
            "operator_training_performed": False,
            "gpu_used": False,
            "full_orbit_or_cross_object_claim_authorized": False,
        },
        "criteria": dict(criteria),
        "pair_scope": {
            "pair_count": 21,
            "fixed_witness_count": 10,
            "source_frames": source_frames.tolist(),
            "target_frames": target_frames.tolist(),
            "stride_frames": 3,
            "signed_generator_degrees": 6.0,
            "pairs_correlated": True,
        },
        "cells_by_arm": cells_by_arm,
        "aggregates": aggregates,
        "historical_matrix_metric_parity": parity_records,
        "controls": {
            "all_six_frozen_operators_evaluated": True,
            "operator_training_optimizer_or_gradient_used": False,
            "visibility_filtering_used": False,
            "point_rematching_or_pruning_used": False,
            "coordinate_normalization": "endpoint-aligned x,y: 2*p/511 - 1; image y not flipped",
            "reference_arm_is_privileged_and_not_autonomous": True,
            "laptop_gpu_used": False,
        },
        "statistical_scope": {
            "inference": "descriptive engineering transfer gate only",
            "operator_cells": "two fixed recipes times three historical training seeds",
            "frame_pairs_correlated": True,
            "uncertainty": "per-recipe sample standard deviation uses ddof=1 over three fixed operator seeds; no SEM, CI, or hypothesis test",
        },
        "bindings": {
            **records,
            "pair_error_arrays": file_record(pair_arrays_path),
            "figure": file_record(figure_path),
        },
        "implementation_source": file_record(Path(__file__)),
    }
    write_json(result_path, result)
    receipt_path = args.output_dir / "FROZEN_OPERATOR_TRACKER_TRANSFER_RECEIPT.json"
    receipt = {
        "schema_version": "frozen_operator_tracker_transfer_receipt.v1",
        "implementation_head": implementation_head,
        "result": file_record(result_path),
        "pair_error_arrays": file_record(pair_arrays_path),
        "figure": file_record(figure_path),
        "decision_branch": decision,
        "reference_arm_succeeds": result["decision"]["reference_arm_succeeds"],
        "certified_tracker_arm_succeeds": result["decision"]["certified_tracker_arm_succeeds"],
        "detector_tracker_arm_succeeds": result["decision"]["detector_tracker_arm_succeeds"],
        "operator_training_performed": False,
        "gpu_used": False,
        "command_argv": list(sys.argv),
    }
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--historical-repo-root", required=True, type=Path)
    parser.add_argument("--historical-result", required=True, type=Path)
    parser.add_argument("--historical-manifest", required=True, type=Path)
    parser.add_argument("--historical-metric-source", required=True, type=Path)
    parser.add_argument("--validation-pairs", required=True, type=Path)
    parser.add_argument("--certified-tracker-arrays", required=True, type=Path)
    parser.add_argument("--detector-tracker-arrays", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
