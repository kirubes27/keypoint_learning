"""Independent numeric reference for representation-oracle tolerances.

This script deliberately does not import ``eval_representation``.  It computes
the preregistered exact estimator and affine fixtures through a separate NumPy
path, then applies

    T = max(32 * machine_epsilon * S, 4 * E_ref)

for each registry entry.  It is calibration evidence, not a scientific
threshold and not a production-oracle invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "representation-oracle-numeric-calibration-v1"
PROVISIONAL_CEILINGS = {
    "float64_direct_geometry_operator": 1e-10,
    "float64_estimator_coordinates": 1e-10,
    "float32_estimator_metric_coordinates": 1e-5,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expectation_from_logits(
    logits: np.ndarray, temperature: float, dtype: np.dtype[Any]
) -> np.ndarray:
    values = logits.astype(dtype, copy=False)
    scaled = values / dtype.type(temperature)
    maximum = np.max(scaled)
    mass = np.exp(scaled - maximum).astype(dtype, copy=False)
    probability = mass / np.sum(mass, dtype=dtype)
    height, width = values.shape
    xs = np.linspace(-1.0, 1.0, width, dtype=dtype)
    ys = np.linspace(-1.0, 1.0, height, dtype=dtype)
    x = np.sum(probability * xs[None, :], dtype=dtype)
    y = np.sum(probability * ys[:, None], dtype=dtype)
    return np.asarray([x, y], dtype=dtype)


def _grid_coordinate(index: int, resolution: int) -> float:
    return -1.0 + 2.0 * float(index) / float(resolution - 1)


def _estimator_errors(
    dtype: np.dtype[Any],
    fixtures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        resolution = int(fixture["resolution"])
        v = int(fixture["v"])
        u = int(fixture["u"])
        temperature = float(fixture["temperature"])
        construction = fixture["construction"]
        if construction == "single_finite_grid_cell":
            logits = np.full((resolution, resolution), -np.inf, dtype=dtype)
            logits[v, u] = dtype.type(0.0)
            actual = _expectation_from_logits(logits, temperature, dtype)
            expected = np.asarray(
                [_grid_coordinate(u, resolution), _grid_coordinate(v, resolution)],
                dtype=np.float64,
            )
        elif construction == "analytic_bilinear_probability_mass":
            fraction_y = float(fixture["fraction_y"])
            fraction_x = float(fixture["fraction_x"])
            weights = np.asarray(
                [
                    [(1.0 - fraction_y) * (1.0 - fraction_x),
                     (1.0 - fraction_y) * fraction_x],
                    [fraction_y * (1.0 - fraction_x), fraction_y * fraction_x],
                ],
                dtype=dtype,
            )
            probability = np.zeros((resolution, resolution), dtype=dtype)
            probability[v : v + 2, u : u + 2] = weights
            temperature_value = dtype.type(temperature)
            constant = dtype.type(float(fixture["logit_constant"]))
            logits = np.full((resolution, resolution), -np.inf, dtype=dtype)
            positive = probability > 0
            logits[positive] = (
                temperature_value * np.log(probability[positive]) + constant
            )
            actual = _expectation_from_logits(
                logits, float(temperature_value), dtype
            )
            x0 = _grid_coordinate(u, resolution)
            x1 = _grid_coordinate(u + 1, resolution)
            y0 = _grid_coordinate(v, resolution)
            y1 = _grid_coordinate(v + 1, resolution)
            expected = np.asarray(
                [
                    (1.0 - fraction_x) * x0 + fraction_x * x1,
                    (1.0 - fraction_y) * y0 + fraction_y * y1,
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unknown estimator construction {construction!r}")
        error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "resolution": resolution,
                "construction": construction,
                "expected_xy": [float(item) for item in expected],
                "actual_xy": [float(item) for item in actual],
                "maximum_absolute_error": error,
            }
        )
    return rows


def _rotation(angle_deg: float, dtype: np.dtype[Any]) -> np.ndarray:
    radians = dtype.type(math.radians(angle_deg))
    cosine = dtype.type(math.cos(float(radians)))
    sine = dtype.type(math.sin(float(radians)))
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=dtype)


def _add_error(
    errors: dict[str, list[dict[str, Any]]],
    metric: str,
    fixture_id: str,
    actual: float | np.ndarray,
    expected: float | np.ndarray,
) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    error = float(np.max(np.abs(actual_array - expected_array)))
    errors.setdefault(metric, []).append(
        {
            "fixture_id": fixture_id,
            "actual": actual_array.tolist(),
            "expected": expected_array.tolist(),
            "maximum_absolute_error": error,
        }
    )


def _metric_scales_from_manifest(
    fixtures: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    scales = {metric: 1.0 for metric in (
        "affine_matrix_element",
        "affine_bias_element",
        "matrix_error_fro",
        "bias_error_l2",
        "proper_rotation_matrix_element",
        "proper_rotation_angle_deg",
        "signed_angle_error_deg",
        "determinant_A",
        "mean_scale",
        "anisotropy",
        "off_diagonal_shear_residual",
        "composition_matrix_element",
        "composition_bias_element",
        "composition_bias_error_l2",
        "rollout_model_mse",
        "rollout_identity_normalized_ratio",
        "full_rollout_auc",
        "closure_model_mse",
        "canonical_drift_rms",
        "float32_affine_coordinate",
    )}
    for fixture in fixtures:
        case = fixture["case"]
        if case == "proper_rotation_polar":
            scales["proper_rotation_angle_deg"] = max(
                scales["proper_rotation_angle_deg"], abs(float(fixture["angle_deg"]))
            )
            # The error is zero, but it is derived from this degree-valued angle.
            scales["signed_angle_error_deg"] = max(
                scales["signed_angle_error_deg"], abs(float(fixture["angle_deg"]))
            )
        elif case == "uniform_scale_svd":
            scales["mean_scale"] = max(scales["mean_scale"], abs(float(fixture["scale"])))
        elif case == "affine_composition":
            horizon = int(fixture["horizon"])
            bias = np.asarray(fixture["b"], dtype=np.float64)
            magnitude = float(np.max(np.abs(horizon * bias)))
            scales["composition_bias_element"] = max(
                scales["composition_bias_element"], magnitude
            )
            scales["composition_bias_error_l2"] = max(
                scales["composition_bias_error_l2"], magnitude
            )
    return scales


def _geometry_errors(
    dtype: np.dtype[Any],
    fixtures: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    errors: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        fixture_id = str(fixture["fixture_id"])
        case = fixture["case"]
        if case == "affine_fit":
            angle_deg = float(fixture["angle_deg"])
            expected_A = _rotation(angle_deg, np.dtype(np.float64))
            expected_centre = np.asarray(fixture["centre_xy"], dtype=np.float64)
            expected_b = expected_centre - expected_A @ expected_centre
            source = np.asarray(fixture["source_points_xy"], dtype=dtype)
            A = expected_A.astype(dtype)
            centre = expected_centre.astype(dtype)
            b = centre - A @ centre
            target = source @ A.T + b
            design = np.concatenate(
                [source, np.ones((source.shape[0], 1), dtype=dtype)], axis=1
            )
            coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            fitted_A = coefficients[:2, :].T
            fitted_b = coefficients[2, :]
            _add_error(errors, "affine_matrix_element", fixture_id, fitted_A, expected_A)
            _add_error(errors, "affine_bias_element", fixture_id, fitted_b, expected_b)
            _add_error(
                errors,
                "matrix_error_fro",
                fixture_id,
                np.linalg.norm(fitted_A.astype(np.float64) - expected_A, ord="fro"),
                0.0,
            )
            _add_error(
                errors,
                "bias_error_l2",
                fixture_id,
                np.linalg.norm(fitted_b.astype(np.float64) - expected_b),
                0.0,
            )
        elif case == "proper_rotation_polar":
            angle_deg = float(fixture["angle_deg"])
            expected_rotation = _rotation(angle_deg, np.dtype(np.float64))
            stretch = np.diag(
                np.asarray(fixture["right_stretch_diagonal"], dtype=dtype)
            )
            matrix = expected_rotation.astype(dtype) @ stretch
            U, _, Vt = np.linalg.svd(matrix)
            correction = np.eye(2, dtype=dtype)
            correction[-1, -1] = np.linalg.det(U @ Vt)
            proper = U @ correction @ Vt
            angle = math.degrees(
                math.atan2(float(proper[1, 0]), float(proper[0, 0]))
            )
            signed_error = (angle - angle_deg + 180.0) % 360.0 - 180.0
            expected_det = float(np.prod(fixture["right_stretch_diagonal"]))
            _add_error(
                errors,
                "proper_rotation_matrix_element",
                fixture_id,
                proper,
                expected_rotation,
            )
            _add_error(
                errors, "proper_rotation_angle_deg", fixture_id, angle, angle_deg
            )
            _add_error(
                errors, "signed_angle_error_deg", fixture_id, signed_error, 0.0
            )
            _add_error(
                errors, "determinant_A", fixture_id, np.linalg.det(matrix), expected_det
            )
        elif case == "reflection_classification":
            matrix = np.asarray(fixture["A"], dtype=dtype)
            actual = "improper_or_reflection" if np.linalg.det(matrix) <= 0.0 else "proper"
            if actual != fixture["expected"]:
                raise RuntimeError(f"{fixture_id} exact reflection category mismatch")
        elif case == "uniform_scale_svd":
            scale = float(fixture["scale"])
            matrix = dtype.type(scale) * np.eye(2, dtype=dtype)
            singular = np.linalg.svd(matrix, compute_uv=False)
            _add_error(errors, "mean_scale", fixture_id, np.mean(singular), scale)
            _add_error(
                errors, "anisotropy", fixture_id, abs(singular[0] - singular[1]), 0.0
            )
            _add_error(
                errors,
                "off_diagonal_shear_residual",
                fixture_id,
                np.linalg.norm(matrix - np.diag(np.diag(matrix)), ord="fro"),
                0.0,
            )
        elif case == "affine_composition":
            A = np.asarray(fixture["A"], dtype=dtype)
            b = np.asarray(fixture["b"], dtype=dtype)
            horizon = int(fixture["horizon"])
            result_A = np.eye(2, dtype=dtype)
            result_b = np.zeros(2, dtype=dtype)
            for _ in range(horizon):
                result_b = A @ result_b + b
                result_A = A @ result_A
            expected_A = np.linalg.matrix_power(
                np.asarray(fixture["A"], dtype=np.float64), horizon
            )
            expected_b = horizon * np.asarray(fixture["b"], dtype=np.float64)
            _add_error(
                errors, "composition_matrix_element", fixture_id, result_A, expected_A
            )
            _add_error(
                errors, "composition_bias_element", fixture_id, result_b, expected_b
            )
            _add_error(
                errors,
                "composition_bias_error_l2",
                fixture_id,
                np.linalg.norm(result_b.astype(np.float64) - expected_b),
                0.0,
            )
        elif case == "rollout":
            frame_count = int(fixture["frame_count"])
            physical_step = float(fixture["physical_step_deg"])
            stride = int(fixture["operator_stride_frames"])
            operator_angle = float(fixture["operator_angle_deg"])
            canonical = np.asarray(fixture["canonical_points_xy"], dtype=dtype)
            frames = np.stack(
                [
                    canonical @ _rotation(index * physical_step, dtype).T
                    for index in range(frame_count)
                ],
                axis=0,
            )
            operator = _rotation(operator_angle, dtype)
            ratios = []
            model_mses = []
            for horizon in range(1, 60):
                power = np.eye(2, dtype=dtype)
                for _ in range(horizon):
                    power = operator @ power
                starts = np.arange(frame_count)
                targets = (starts + horizon * stride) % frame_count
                predicted = frames @ power.T
                model_mse = float(np.mean((predicted - frames[targets]) ** 2))
                identity_mse = float(np.mean((frames - frames[targets]) ** 2))
                ratio = model_mse / identity_mse
                model_mses.append(model_mse)
                ratios.append(ratio)
            closure_power = np.eye(2, dtype=dtype)
            for _ in range(60):
                closure_power = operator @ closure_power
            closure_mse = float(np.mean((frames @ closure_power.T - frames) ** 2))
            _add_error(
                errors, "rollout_model_mse", fixture_id, max(model_mses), 0.0
            )
            _add_error(
                errors,
                "rollout_identity_normalized_ratio",
                fixture_id,
                max(ratios),
                0.0,
            )
            _add_error(
                errors, "full_rollout_auc", fixture_id, np.mean(ratios), 0.0
            )
            _add_error(
                errors, "closure_model_mse", fixture_id, closure_mse, 0.0
            )
        elif case == "canonical_drift":
            frame_count = int(fixture["frame_count"])
            physical_step = float(fixture["physical_step_deg"])
            canonical = np.asarray(fixture["canonical_points_xy"], dtype=dtype)
            recovered = []
            for index in range(frame_count):
                state = _rotation(index * physical_step, dtype)
                observed = canonical @ state.T
                recovered.append(observed @ np.linalg.inv(state).T)
            recovered_array = np.stack(recovered, axis=0).astype(np.float64)
            centred = recovered_array - np.mean(recovered_array, axis=0, keepdims=True)
            residual = centred / (
                2.0 * float(fixture["bbox_diagonal_image01"])
            )
            rms = np.sqrt(np.mean(np.sum(residual * residual, axis=2), axis=0))
            _add_error(
                errors, "canonical_drift_rms", fixture_id, np.max(rms), 0.0
            )
        elif case == "affine_coordinate":
            angle_deg = float(fixture["angle_deg"])
            A = _rotation(angle_deg, dtype)
            centre = np.asarray(fixture["centre_xy"], dtype=dtype)
            b = centre - A @ centre
            source = np.asarray(fixture["source_points_xy"], dtype=dtype)
            actual = source @ A.T + b
            expected_A = _rotation(angle_deg, np.dtype(np.float64))
            expected_centre = np.asarray(fixture["centre_xy"], dtype=np.float64)
            expected_b = expected_centre - expected_A @ expected_centre
            expected = (
                np.asarray(fixture["source_points_xy"], dtype=np.float64)
                @ expected_A.T
                + expected_b
            )
            _add_error(
                errors, "float32_affine_coordinate", fixture_id, actual, expected
            )
        else:
            raise ValueError(f"unknown geometry case {case!r}")
    return errors


def _registry_entry(
    *,
    dtype: np.dtype[Any],
    rows: list[dict[str, Any]],
    expected_scale: float,
    ceiling: float,
) -> dict[str, Any]:
    reference_error = max(float(row["maximum_absolute_error"]) for row in rows)
    epsilon = float(np.finfo(dtype).eps)
    tolerance = max(32.0 * epsilon * max(1.0, expected_scale), 4.0 * reference_error)
    return {
        "dtype": dtype.name,
        "machine_epsilon": epsilon,
        "expected_magnitude_scale_S": expected_scale,
        "maximum_reference_error_E_ref": reference_error,
        "formula": "max(32 * machine_epsilon * S, 4 * E_ref)",
        "tolerance": tolerance,
        "provisional_ceiling": ceiling,
        "within_provisional_ceiling": bool(tolerance <= ceiling),
        "fixtures": rows,
    }


def build_calibration(
    fixture_manifest_path: Path,
    *,
    fixture_manifest_commit: str,
    reference_source_commit: str,
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    with fixture_manifest_path.open("r", encoding="utf-8") as handle:
        fixture_manifest = json.load(handle)
    estimator_fixtures = fixture_manifest["estimator_fixtures"]
    geometry_fixtures = fixture_manifest["geometry_fixtures"]
    estimator64 = _estimator_errors(np.dtype(np.float64), estimator_fixtures)
    estimator32 = _estimator_errors(np.dtype(np.float32), estimator_fixtures)
    geometry64 = _geometry_errors(np.dtype(np.float64), geometry_fixtures)
    geometry32 = _geometry_errors(np.dtype(np.float32), geometry_fixtures)
    scales = _metric_scales_from_manifest(geometry_fixtures)
    metric_registry: dict[str, dict[str, Any]] = {}
    for metric in fixture_manifest["metric_registry"]["float64"]:
        if metric not in geometry64:
            raise RuntimeError(f"preregistered float64 metric {metric} lacks a fixture")
        metric_registry[f"float64::{metric}"] = _registry_entry(
            dtype=np.dtype(np.float64),
            rows=geometry64[metric],
            expected_scale=scales[metric],
            ceiling=PROVISIONAL_CEILINGS["float64_direct_geometry_operator"],
        )
    metric_registry["float64::spatial_expectation_coordinate"] = _registry_entry(
        dtype=np.dtype(np.float64),
        rows=estimator64,
        expected_scale=1.0,
        ceiling=PROVISIONAL_CEILINGS["float64_estimator_coordinates"],
    )
    metric_registry["float32::spatial_expectation_coordinate"] = _registry_entry(
        dtype=np.dtype(np.float32),
        rows=estimator32,
        expected_scale=1.0,
        ceiling=PROVISIONAL_CEILINGS["float32_estimator_metric_coordinates"],
    )
    metric_registry["float32::affine_coordinate"] = _registry_entry(
        dtype=np.dtype(np.float32),
        rows=geometry32["float32_affine_coordinate"],
        expected_scale=scales["float32_affine_coordinate"],
        ceiling=PROVISIONAL_CEILINGS["float32_estimator_metric_coordinates"],
    )
    if not all(
        entry["within_provisional_ceiling"] for entry in metric_registry.values()
    ):
        raise RuntimeError("at least one calibrated tolerance exceeds its provisional ceiling")
    result = {
        "schema_version": SCHEMA_VERSION,
        "fixture_manifest": {
            "relative_path": (
                "docs/decisions/2026-07-26/representation_oracle_calibration/"
                "CALIBRATION_FIXTURE_MANIFEST.json"
            ),
            "sha256": _file_sha256(fixture_manifest_path),
            "git_commit": fixture_manifest_commit,
            "status": fixture_manifest["status"],
        },
        "reference_implementation": {
            "relative_path": "keypoint_net/diagnostics/calibrate_representation_oracle_tolerances.py",
            "sha256": _file_sha256(source_path),
            "git_commit": reference_source_commit,
            "imports_production_evaluator": False,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "metric_registries": metric_registry,
        "provisional_ceilings": PROVISIONAL_CEILINGS,
        "scope": "numeric_implementation_correctness_only_not_scientific_decision_thresholds",
        "exact_fields": fixture_manifest["metric_registry"]["exact"],
        "exact_field_rule": "exact_equality",
    }
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixture-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, value in (
        ("fixture-commit", args.fixture_commit),
        ("source-commit", args.source_commit),
    ):
        if not (
            7 <= len(value) <= 64
            and all(char in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{name} must be a lowercase hexadecimal commit id")
    result = build_calibration(
        args.fixtures,
        fixture_manifest_commit=args.fixture_commit,
        reference_source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")
    print(json.dumps({
        "output": str(args.output),
        "content_sha256": result["content_sha256"],
        "maximum_tolerance_by_dtype": {
            dtype: max(
                value["tolerance"]
                for key, value in result["metric_registries"].items()
                if key.startswith(f"{dtype}::")
            )
            for dtype in ("float64", "float32")
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
