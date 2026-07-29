"""Strict, transformation-aware representation evaluation.

This module is intentionally independent of the training loop.  Planted
fixtures and saved-checkpoint bundles enter through :func:`evaluate_bundle`;
there is no oracle-only metric implementation.

The public bundle schema is JSON-compatible and represents one
``(object, seed, transform, direction, stride, partition)`` stratum.
Held-out/full-corpus evidence lives only in ``evaluation``.  When an affine
must be fitted, its training frames and rows live in a separate ``fit``
section; the evaluator never concatenates the two collections.  Keeping
strata separate is deliberate: pooling them can hide a semantic failure.

This evaluator reports descriptive values only.  It never computes SEM and it
does not collapse operator accuracy and representation health into one score.
Scientific decision thresholds are not defined here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from keypoint_net import representation_evaluation_provenance as provenance_contract
from keypoint_net import representation_array_codec as array_codec


BUNDLE_SCHEMA_VERSION = "representation-evaluation-bundle-v1"
RESULT_SCHEMA_VERSION = "representation-evaluation-result-v2"
SUPPORTED_FAMILIES = {"roll", "translation", "scale", "yaw", "pitch"}
HEX_DIGITS = frozenset("0123456789abcdef")
AUTHORITATIVE_NUMERIC_REGISTRY_RELATIVE_PATH = Path(
    "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "NUMERIC_CALIBRATION_v1_1.json"
)
AUTHORITATIVE_NUMERIC_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / AUTHORITATIVE_NUMERIC_REGISTRY_RELATIVE_PATH
).resolve()
AUTHORITATIVE_NUMERIC_REGISTRY_FILE_SHA256 = (
    "40115c7325858137595ba6e90e36fbd52c9a72da2f16b64d9fd1ad4f7764f420"
)
AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256 = (
    "24a2d0eb28d1ec8f26c14064abdb4afcaa51e7cf63d9560e1c2f1c9b4a8118c8"
)
AUTHORITATIVE_NUMERIC_REGISTRY_SCHEMA_VERSION = (
    "representation-oracle-numeric-calibration-v1"
)
BASE_EVALUATION_CONFIG_FIELDS = {
    "protocol",
    "representation_thresholds",
    "motion_reference_magnitude_image01",
    "motion_fraction_min",
    "on_object_rate_min",
    "minimum_eligible_channels",
    "operator_composition_horizons",
}
ROLL_EVALUATION_CONFIG_FIELDS = {
    "full_rollout_horizons",
    "role_scoped_holdout_frame_ids",
    "holdout_rollout_horizons",
    "closure_horizon",
}
CHECKPOINT_AUTHORIZATION_MODULE = (
    "keypoint_net.representation_checkpoint_authorization"
)
CHECKPOINT_AUTHORIZATION_VALIDATOR = (
    "validate_checkpoint_evaluator_authorization"
)
CHECKPOINT_PROVENANCE_RECEIPT_CONSUMER = (
    "consume_checkpoint_provenance_load_receipt"
)
CHECKPOINT_AUTHORIZATION_RECEIPT_FIELDS = {
    "checkpoint_evaluation_authorized",
    "source_commit",
    "task_id",
    "fixture_id",
    "checkpoint_sha256",
    "runtime_source_manifest_file_sha256",
    "checkpoint_hash_preflight_file_sha256",
    "training_or_weight_update_authorized",
    "selection_use_authorized",
}


class EvaluationContractError(ValueError):
    """Raised when a bundle cannot support the claimed evaluation."""

    DEFAULT_CODE = "evaluation_contract_violation"

    def __init__(
        self,
        message: str,
        *,
        code: str = DEFAULT_CODE,
    ) -> None:
        _code_is_valid = (
            isinstance(code, str)
            and bool(code)
            and code.replace("_", "").isalnum()
            and code.lower() == code
        )
        if not _code_is_valid:
            raise ValueError("EvaluationContractError code must be stable snake_case")
        self.code = code
        super().__init__(message)


_LOADED_EVALUATOR_SOURCE_DIGEST = (
    provenance_contract.capture_loaded_source_digest(
        "evaluator_source",
        Path(__file__).absolute(),
    )
)
_LOADED_ARRAY_CODEC_SOURCE_DIGEST = {
    "absolute_path": str(Path(array_codec.__file__).resolve(strict=True)),
    "sha256": str(array_codec.__representation_import_sha256__),
}
__representation_import_sha256__ = _LOADED_EVALUATOR_SOURCE_DIGEST["sha256"]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository-wide canonical JSON representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvaluationContractError(
                f"JSON input contains duplicate JSON key {key!r}",
                code="strict_json_invalid",
            )
        value[key] = item
    return value


def _strict_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise EvaluationContractError(
            f"JSON input contains non-finite JSON number {token!r}",
            code="strict_json_invalid",
        )
    return value


def _reject_nonfinite_json_constant(token: str) -> None:
    raise EvaluationContractError(
        f"JSON input contains non-finite JSON constant {token!r}",
        code="strict_json_invalid",
    )


def _load_strict_json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_float=_strict_json_float,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except EvaluationContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationContractError(
            f"{name} is not strict valid JSON",
            code="strict_json_invalid",
        ) from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX_DIGITS for char in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_planted_harness_digest(
    record: Mapping[str, Any],
) -> dict[str, str]:
    committed = record.get("committed_files")
    _require(
        isinstance(committed, list),
        "planted provenance committed_files must be a list",
    )
    harness_records = [
        item
        for item in committed
        if isinstance(item, Mapping)
        and item.get("role") == "oracle_harness_source"
    ]
    _require(
        len(harness_records) == 1,
        "planted provenance must bind exactly one oracle harness",
    )
    relative_path = harness_records[0].get("repo_relative_path")
    _require(
        isinstance(relative_path, str) and bool(relative_path),
        "planted oracle harness path is invalid",
    )
    expected_path = (
        Path(__file__).resolve().parents[1] / relative_path
    ).resolve(strict=True)
    matches: set[tuple[str, str]] = set()
    for module in tuple(sys.modules.values()):
        captured = getattr(module, "_LOADED_HARNESS_SOURCE_DIGEST", None)
        if not isinstance(captured, Mapping):
            continue
        path_value = captured.get("absolute_path")
        digest = captured.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            continue
        try:
            loaded_path = Path(path_value).resolve(strict=True)
        except OSError:
            continue
        if loaded_path == expected_path:
            matches.add((str(loaded_path), digest))
    _require(
        len(matches) == 1,
        "the exact planted oracle harness lacks one import-time digest",
    )
    path_value, digest = next(iter(matches))
    return {"absolute_path": path_value, "sha256": digest}


def _production_loaded_source_digests(
    record: Mapping[str, Any],
    *,
    case_kind: str,
) -> dict[str, dict[str, str]]:
    """Return only sources whose import-time bytes are actually captured.

    The strict provenance helper separately checks the adapter, harness, and
    every other committed role against Git.  They are not mislabeled as
    ``loaded_sources`` merely because their files exist on disk.
    """

    loaded = {
        "evaluator_source": dict(_LOADED_EVALUATOR_SOURCE_DIGEST),
        "array_codec_source": dict(_LOADED_ARRAY_CODEC_SOURCE_DIGEST),
    }
    if case_kind == "planted":
        loaded["oracle_harness_source"] = _active_planted_harness_digest(
            record
        )
    return loaded


def _validate_production_provenance(
    record: Mapping[str, Any],
    *,
    case_kind: str,
    fit_from_pairs: bool,
    checkpoint_provenance_load_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Production provenance boundary.

    This indirection is an intentionally private unit-test seam.  Public calls
    always use the strict helper against this repository and cannot choose a
    repository root.
    """

    try:
        return provenance_contract.validate_evaluation_provenance(
            record,
            case_kind=case_kind,
            fit_from_pairs=fit_from_pairs,
            loaded_source_digests=_production_loaded_source_digests(
                record,
                case_kind=case_kind,
            ),
            checkpoint_provenance_load_receipt=(
                checkpoint_provenance_load_receipt
            ),
        )
    except provenance_contract.ProvenanceContractError as exc:
        raise EvaluationContractError(
            f"provenance validation failed: {exc}",
            code="provenance_invalid",
        ) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    _require(not missing, f"{name} missing fields: {missing}")
    _require(not extra, f"{name} has unknown fields: {extra}")


def _absolute_existing_file(value: object, *, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} must be a path string")
    path = Path(value)
    _require(path.is_absolute(), f"{name} must be absolute")
    _require(path.is_file(), f"{name} does not resolve to a file")
    return path


def _strict_bool_array(
    value: object,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    if array_codec.is_compact_array_record(value):
        try:
            array = array_codec.decode_bool_packbits_array(value)
        except array_codec.ArrayCodecError as exc:
            raise EvaluationContractError(
                f"{name} compact boolean array is invalid: {exc}",
                code="compact_array_invalid",
            ) from exc
    else:
        array = np.asarray(value)
    _require(array.dtype == np.dtype(np.bool_), f"{name} must contain JSON booleans")
    if shape is not None:
        _require(array.ndim == len(shape), f"{name} must have {len(shape)} dimensions")
        for axis, expected in enumerate(shape):
            if expected is not None:
                _require(
                    array.shape[axis] == expected,
                    f"{name} axis {axis} must have length {expected}, got {array.shape[axis]}",
                )
    return array


def _finite_array(
    value: object,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    if array_codec.is_compact_array_record(value):
        try:
            compact = array_codec.decode_float32_array(value)
        except array_codec.ArrayCodecError as exc:
            raise EvaluationContractError(
                f"{name} compact float32 array is invalid: {exc}",
                code="compact_array_invalid",
            ) from exc
        array = compact.astype(np.float64)
    else:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(f"{name} is not a numeric array") from exc
    if shape is not None:
        _require(array.ndim == len(shape), f"{name} must have {len(shape)} dimensions")
        for axis, expected in enumerate(shape):
            if expected is not None:
                _require(
                    array.shape[axis] == expected,
                    f"{name} axis {axis} must have length {expected}, got {array.shape[axis]}",
                )
    _require(bool(np.all(np.isfinite(array))), f"{name} contains a non-finite value")
    return array


def _descriptive(
    values: Iterable[float],
    *,
    sample_unit: str,
    aggregation_hierarchy: Sequence[str],
) -> dict[str, Any]:
    """Named descriptive summary with explicit sample semantics and no SEM."""

    array = np.asarray(list(values), dtype=np.float64)
    _require(array.size > 0, "cannot summarize an empty sample")
    _require(bool(np.all(np.isfinite(array))), "descriptive sample contains non-finite values")
    _require(
        isinstance(sample_unit, str) and bool(sample_unit),
        "descriptive sample_unit must be a non-empty string",
    )
    _require(
        isinstance(aggregation_hierarchy, Sequence)
        and not isinstance(aggregation_hierarchy, (str, bytes))
        and bool(aggregation_hierarchy)
        and all(
            isinstance(level, str) and bool(level)
            for level in aggregation_hierarchy
        ),
        "descriptive aggregation_hierarchy must contain named levels",
    )
    q25, median, q75 = np.percentile(array, [25.0, 50.0, 75.0])
    return {
        "n": int(array.size),
        "sample_unit": sample_unit,
        "aggregation_hierarchy": list(aggregation_hierarchy),
        "inference": "descriptive",
        "mean": float(np.mean(array)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "values": [float(item) for item in array],
    }


def spatial_expectation(
    logits: np.ndarray | Sequence[Any],
    *,
    temperature: float,
    softmax_dtype: str,
) -> np.ndarray:
    """Endpoint-aligned spatial expectation used by the production model.

    ``logits`` has shape ``(..., H, W)``.  Negative infinity is permitted for
    zero probability mass, but every heatmap must contain at least one finite
    logit.  NaN, positive infinity, and all-negative-infinity maps are fatal.
    """

    _require(
        isinstance(temperature, (int, float)) and math.isfinite(float(temperature)),
        "temperature must be finite",
    )
    _require(float(temperature) > 0.0, "temperature must be positive")
    _require(softmax_dtype in {"float32", "float64"}, "unsupported softmax_dtype")
    dtype = np.float32 if softmax_dtype == "float32" else np.float64
    values = np.asarray(logits, dtype=dtype)
    _require(values.ndim >= 2, "logits must end in heatmap height and width")
    _require(values.shape[-2] >= 2 and values.shape[-1] >= 2, "heatmap is too small")
    _require(not bool(np.any(np.isnan(values))), "logits contain NaN")
    _require(not bool(np.any(np.isposinf(values))), "logits contain positive infinity")
    finite_any = np.any(np.isfinite(values), axis=(-2, -1))
    _require(bool(np.all(finite_any)), "a heatmap has no finite logit")

    scaled = values / dtype(float(temperature))
    maximum = np.max(scaled, axis=(-2, -1), keepdims=True)
    weights = np.exp(scaled - maximum, dtype=dtype)
    denominator = np.sum(weights, axis=(-2, -1), keepdims=True, dtype=dtype)
    _require(bool(np.all(np.isfinite(denominator))), "softmax denominator is non-finite")
    _require(bool(np.all(denominator > 0)), "softmax denominator is zero")
    probabilities = weights / denominator

    height, width = values.shape[-2:]
    xs = np.linspace(-1.0, 1.0, width, dtype=dtype)
    ys = np.linspace(-1.0, 1.0, height, dtype=dtype)
    x = np.sum(probabilities * xs.reshape((1,) * (values.ndim - 1) + (width,)),
               axis=(-2, -1), dtype=dtype)
    y = np.sum(probabilities * ys.reshape((1,) * (values.ndim - 2) + (height, 1)),
               axis=(-2, -1), dtype=dtype)
    return np.stack([x, y], axis=-1)


def probabilities_from_logits(
    logits: np.ndarray | Sequence[Any],
    *,
    temperature: float,
    softmax_dtype: str,
) -> np.ndarray:
    """Return validated spatial probabilities for health diagnostics."""

    dtype = np.float32 if softmax_dtype == "float32" else np.float64
    values = np.asarray(logits, dtype=dtype)
    # Reuse every semantic check in the estimator path.
    spatial_expectation(
        values, temperature=temperature, softmax_dtype=softmax_dtype
    )
    scaled = values / dtype(float(temperature))
    maximum = np.max(scaled, axis=(-2, -1), keepdims=True)
    weights = np.exp(scaled - maximum, dtype=dtype)
    return weights / np.sum(weights, axis=(-2, -1), keepdims=True, dtype=dtype)


def _validated_logits_and_points(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, int, int, int],
    estimator: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate declared logits and derive their fixed spatial expectations."""

    if array_codec.is_compact_array_record(value):
        _require(
            estimator["logit_dtype"] == "float32",
            "compact float32 logits require estimator logit_dtype=float32",
        )
        try:
            compact_logits = array_codec.decode_float32_array(value)
        except array_codec.ArrayCodecError as exc:
            raise EvaluationContractError(
                f"{name} compact float32 array is invalid: {exc}",
                code="compact_array_invalid",
            ) from exc
        raw_logits: np.ndarray = compact_logits
    else:
        raw_logits = np.asarray(value, dtype=object)
        for item in raw_logits.reshape(-1):
            is_finite_number = (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            )
            _require(
                is_finite_number or item == "-inf",
                f"{name} must contain finite numbers or the exact '-inf' zero-mass sentinel",
            )
    declared_logit_dtype = (
        np.float32 if estimator["logit_dtype"] == "float32" else np.float64
    )
    logits = np.asarray(
        raw_logits if array_codec.is_compact_array_record(value) else raw_logits.tolist(),
        dtype=declared_logit_dtype,
    )
    _require(
        logits.shape == expected_shape,
        f"{name} shape disagrees with estimator metadata",
    )
    points = spatial_expectation(
        logits,
        temperature=float(estimator["temperature"]),
        softmax_dtype=str(estimator["softmax_dtype"]),
    ).astype(np.float64)
    return logits, points


def apply_affine(points: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply column-vector semantics to an array whose last axis is xy."""

    return np.matmul(points, A.T) + b


def affine_power(A: np.ndarray, b: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    _require(isinstance(steps, int) and steps >= 0, "steps must be a non-negative integer")
    result_A = np.eye(2, dtype=np.float64)
    result_b = np.zeros(2, dtype=np.float64)
    for _ in range(steps):
        result_b = A @ result_b + b
        result_A = A @ result_A
    return result_A, result_b


def fit_shared_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of ``target = A source + b``."""

    src = _finite_array(source, name="fit source")
    dst = _finite_array(target, name="fit target")
    _require(src.shape == dst.shape, "fit source and target shapes differ")
    _require(src.ndim == 2 and src.shape[1] == 2, "fit arrays must be [n,2]")
    _require(src.shape[0] >= 3, "at least three correspondences are required")
    design = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    _require(np.linalg.matrix_rank(design) == 3, "affine fit is rank deficient")
    coefficients, _, _, _ = np.linalg.lstsq(design, dst, rcond=None)
    return coefficients[:2, :].T, coefficients[2, :]


def _circular_difference_degrees(actual: float, target: float) -> float:
    return float((actual - target + 180.0) % 360.0 - 180.0)


def proper_rotation_metrics(A: np.ndarray, *, target_angle_deg: float) -> dict[str, Any]:
    """Extract the closest proper rotation without accepting reflections."""

    matrix = _finite_array(A, name="operator A", shape=(2, 2))
    determinant = float(np.linalg.det(matrix))
    base = {
        "determinant_A": determinant,
        "improper_or_reflection": bool(determinant <= 0.0),
    }
    if determinant <= 0.0:
        base.update(
            {
                "proper_rotation_angle_deg": None,
                "signed_angle_error_deg": None,
                "absolute_angle_error_deg": None,
            }
        )
        return base

    U, _, Vt = np.linalg.svd(matrix)
    correction = np.eye(2, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(U @ Vt)
    rotation = U @ correction @ Vt
    _require(np.linalg.det(rotation) > 0.0, "polar extraction did not produce SO(2)")
    angle = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    signed_error = _circular_difference_degrees(angle, float(target_angle_deg))
    base.update(
        {
            "proper_rotation": rotation.tolist(),
            "proper_rotation_angle_deg": float(angle),
            "signed_angle_error_deg": signed_error,
            "absolute_angle_error_deg": abs(signed_error),
        }
    )
    return base


def _operator_metrics(
    family: str,
    A: np.ndarray,
    b: np.ndarray,
    target_A: np.ndarray,
    target_b: np.ndarray,
    transform: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    determinant = float(np.linalg.det(A))
    singular_values = np.linalg.svd(A, compute_uv=False)
    result: dict[str, Any] = {
        "determinant_A": determinant,
        "singular_values": [float(item) for item in singular_values],
        "improper_or_reflection": bool(determinant <= 0.0),
        "learned_A": A.tolist(),
        "learned_b": b.tolist(),
        "target_A": target_A.tolist(),
        "target_b": target_b.tolist(),
    }
    if family in {"yaw", "pitch"}:
        result.update(
            {
                "matrix_error_fro": None,
                "bias_error_l2": None,
                "operator_comparison_status": (
                    "not_applicable_target_is_same_train_fitted_reference"
                ),
            }
        )
    else:
        result.update(
            {
                "matrix_error_fro": float(
                    np.linalg.norm(A - target_A, ord="fro")
                ),
                "bias_error_l2": float(np.linalg.norm(b - target_b)),
                "operator_comparison_status": "available_locked_target",
            }
        )
    if family == "roll":
        generator_units = transform["generator_units"]
        _require(generator_units == "degrees", "roll signed_generator must use degrees")
        result.update(
            proper_rotation_metrics(
                A, target_angle_deg=float(transform["signed_generator"])
            )
        )
        actual_angle = result["proper_rotation_angle_deg"]
        result["wrong_locked_sign"] = bool(
            actual_angle is not None
            and float(actual_angle) * float(transform["signed_generator"]) < 0.0
        )
        centre = transform.get("projected_centre_xy")
        if centre is not None:
            centre_array = _finite_array(centre, name="projected centre", shape=(2,))
            expected_from_centre = centre_array - A @ centre_array
            result["centre_implied_bias_error_l2"] = float(
                np.linalg.norm(b - expected_from_centre)
            )
    elif family == "translation":
        axis = transform["physical_axis"]
        _require(axis in {"world_x", "world_y", "image_x", "image_y"}, "invalid translation axis")
        expected_component = int(transform["expected_image_component"])
        _require(expected_component in {0, 1}, "expected_image_component must be 0 or 1")
        other = 1 - expected_component
        result.update(
            {
                "A_minus_identity_fro": float(np.linalg.norm(A - np.eye(2), ord="fro")),
                "signed_bias_component_error": float(b[expected_component] - target_b[expected_component]),
                "orthogonal_bias_leakage": float(b[other] - target_b[other]),
                "wrong_locked_sign": bool(
                    float(b[expected_component])
                    * float(target_b[expected_component])
                    < 0.0
                ),
            }
        )
    elif family == "scale":
        mean_scale = float(np.mean(singular_values))
        target_ratio = math.exp(float(transform["signed_generator"]))
        result.update(
            {
                "mean_scale": mean_scale,
                "target_scale_ratio": target_ratio,
                "mean_scale_error": float(mean_scale - target_ratio),
                "anisotropy": float(abs(singular_values[0] - singular_values[1])),
                "off_diagonal_shear_residual": float(np.linalg.norm(A - np.diag(np.diag(A)), ord="fro")),
            }
        )
        centre = _finite_array(
            transform["projected_centre_xy"],
            name="projected centre",
            shape=(2,),
        )
        result["centre_implied_bias_error_l2"] = float(
            np.linalg.norm(b - (centre - A @ centre))
        )
        result["centre_target_bias_error_l2"] = float(np.linalg.norm(b - target_b))
    else:
        result["claim_scope"] = "local_affine_approximation"

    if family in {"yaw", "pitch"}:
        result["composition"] = []
        result["composition_status"] = (
            "not_applicable_finite_arc_comparison_to_self_fitted_reference"
        )
        return result

    horizons = config.get("operator_composition_horizons", [])
    composition_rows = []
    for value in horizons:
        _require(isinstance(value, int) and value >= 1, "composition horizons must be positive integers")
        learned_Ak, learned_bk = affine_power(A, b, value)
        target_Ak, target_bk = affine_power(target_A, target_b, value)
        composition_rows.append(
            {
                "k": value,
                "matrix_error_fro": float(np.linalg.norm(learned_Ak - target_Ak, ord="fro")),
                "bias_error_l2": float(np.linalg.norm(learned_bk - target_bk)),
                "learned_b": learned_bk.tolist(),
                "target_b": target_bk.tolist(),
            }
        )
    result["composition"] = composition_rows
    result["composition_status"] = "available_locked_target"
    return result


def _longest_true_run(flags: np.ndarray, *, cyclic: bool) -> int:
    values = np.asarray(flags, dtype=bool)
    if values.size == 0 or not bool(values.any()):
        return 0
    if bool(values.all()):
        return int(values.size)
    scan = np.concatenate([values, values]) if cyclic else values
    best = current = 0
    for item in scan:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(min(best, values.size))


def _pairwise_representation(
    points: np.ndarray,
    bbox_diagonal_image01: np.ndarray,
    visibility: np.ndarray,
    frame_ids: Sequence[Any],
    channel_indices: Sequence[int],
    *,
    cyclic: bool,
    thresholds: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected = [int(index) for index in channel_indices]
    if len(selected) < 2:
        return None
    points01 = (points[:, selected, :] + 1.0) / 2.0
    selected_visibility = visibility[:, selected]
    distance_cube = np.linalg.norm(
        points01[:, :, None, :] - points01[:, None, :, :], axis=-1
    ) / bbox_diagonal_image01[:, None, None]

    nearest_rows: list[dict[str, Any]] = []
    nearest_values: list[float] = []
    for frame_index in range(points.shape[0]):
        visible_local = np.flatnonzero(selected_visibility[frame_index])
        for local_index, channel in enumerate(selected):
            if not selected_visibility[frame_index, local_index]:
                status = "void_channel_not_visible"
                value = None
            elif visible_local.size < 2:
                status = "void_no_jointly_visible_neighbour"
                value = None
            else:
                candidates = visible_local[visible_local != local_index]
                value = float(
                    np.min(distance_cube[frame_index, local_index, candidates])
                )
                status = "available"
                nearest_values.append(value)
            nearest_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_ids[frame_index],
                    "channel": channel,
                    "status": status,
                    "nearest_distance_objdiag": value,
                }
            )

    close_threshold = float(thresholds["close_distance_objdiag"])
    persistent_fraction = float(thresholds["persistent_fraction"])
    recurrent_fraction = float(thresholds["recurrent_fraction"])
    transient_longest_fraction = float(thresholds["transient_longest_fraction"])
    clustered_median = float(thresholds["clustered_median_objdiag"])
    pair_rows: list[dict[str, Any]] = []
    category_counts = {
        "persistent_duplicate": 0,
        "recurrent_close_pair": 0,
        "transient_crossing": 0,
        "clustered_pair": 0,
        "separate_pair": 0,
        "void_no_joint_visibility": 0,
    }
    for local_i, channel_i in enumerate(selected):
        for local_j in range(local_i + 1, len(selected)):
            channel_j = selected[local_j]
            jointly_visible = (
                selected_visibility[:, local_i]
                & selected_visibility[:, local_j]
            )
            jointly_visible_count = int(np.sum(jointly_visible))
            if jointly_visible_count == 0:
                category_counts["void_no_joint_visibility"] += 1
                pair_rows.append(
                    {
                        "channel_i": channel_i,
                        "channel_j": channel_j,
                        "status": "void_no_joint_visibility",
                        "jointly_visible_frame_count": 0,
                        "jointly_visible_frame_ids": [],
                        "median_distance_objdiag": None,
                        "minimum_distance_objdiag": None,
                        "close_fraction": None,
                        "longest_close_run_frames": None,
                        "longest_close_run_fraction": None,
                        "category": None,
                        "raw_distance_objdiag": [],
                    }
                )
                continue
            series = distance_cube[jointly_visible, local_i, local_j]
            close_visible = series < close_threshold
            close_full = np.zeros(points.shape[0], dtype=bool)
            close_full[jointly_visible] = close_visible
            fraction = float(np.mean(close_visible))
            longest = _longest_true_run(close_full, cyclic=cyclic)
            longest_fraction = float(longest / jointly_visible_count)
            median_distance = float(np.median(series))
            if fraction >= persistent_fraction:
                category = "persistent_duplicate"
            elif fraction >= recurrent_fraction:
                category = "recurrent_close_pair"
            elif fraction > 0.0 and longest_fraction <= transient_longest_fraction:
                category = "transient_crossing"
            elif median_distance < clustered_median:
                category = "clustered_pair"
            else:
                category = "separate_pair"
            category_counts[category] += 1
            pair_rows.append(
                {
                    "channel_i": channel_i,
                    "channel_j": channel_j,
                    "status": "available",
                    "jointly_visible_frame_count": jointly_visible_count,
                    "jointly_visible_frame_ids": [
                        frame_ids[index]
                        for index in np.flatnonzero(jointly_visible)
                    ],
                    "median_distance_objdiag": median_distance,
                    "minimum_distance_objdiag": float(np.min(series)),
                    "close_fraction": fraction,
                    "longest_close_run_frames": longest,
                    "longest_close_run_fraction": longest_fraction,
                    "category": category,
                    "raw_distance_objdiag": [float(item) for item in series],
                }
            )

    pair_count = len(pair_rows)
    void_pair_count = category_counts["void_no_joint_visibility"]
    evaluable_pair_count = pair_count - void_pair_count
    persistent_rate = (
        category_counts["persistent_duplicate"] / evaluable_pair_count
        if evaluable_pair_count
        else None
    )
    recurrent_rate = (
        category_counts["recurrent_close_pair"] / evaluable_pair_count
        if evaluable_pair_count
        else None
    )
    nearest_summary = (
        _descriptive(
            nearest_values,
            sample_unit="visible_frame_channel_nearest_neighbour_distance_objdiag",
            aggregation_hierarchy=("stratum", "frame", "channel"),
        )
        if nearest_values
        else None
    )
    return {
        "channel_indices": selected,
        "channel_count": len(selected),
        "pair_count": pair_count,
        "evaluable_pair_count": evaluable_pair_count,
        "void_pair_count": void_pair_count,
        "all_selected_channels_visible_in_all_frames": bool(
            np.all(selected_visibility)
        ),
        "nearest_neighbour": {
            "summary": nearest_summary,
            "rows": nearest_rows,
        },
        "trajectory_median_nearest_neighbour_objdiag": (
            float(np.median(nearest_values)) if nearest_values else None
        ),
        "trajectory_minimum_nearest_neighbour_objdiag": (
            float(np.min(nearest_values)) if nearest_values else None
        ),
        "persistent_duplicate_pair_rate": (
            float(persistent_rate) if persistent_rate is not None else None
        ),
        "recurrent_close_pair_rate": (
            float(recurrent_rate) if recurrent_rate is not None else None
        ),
        "all_pairs_persistent_duplicate": bool(
            evaluable_pair_count == pair_count
            and category_counts["persistent_duplicate"] == pair_count
        ),
        "visibility_rule": (
            "distances_use_only_frames_where_both_channels_are_visible;"
            "visibility_gaps_break_close_runs"
        ),
        "category_counts": category_counts,
        "pairs": pair_rows,
    }


def _canonical_drift(
    points: np.ndarray,
    state_A: np.ndarray,
    state_b: np.ndarray,
    bbox_diagonal_image01: np.ndarray,
    visibility: np.ndarray,
) -> dict[str, Any]:
    canonical = np.empty_like(points, dtype=np.float64)
    for frame_index in range(points.shape[0]):
        determinant = float(np.linalg.det(state_A[frame_index]))
        _require(abs(determinant) > 1e-12, "a state transform is singular")
        inverse = np.linalg.inv(state_A[frame_index])
        canonical[frame_index] = apply_affine(
            points[frame_index] - state_b[frame_index], inverse, np.zeros(2)
        )

    channel_rows = []
    values = []
    for channel in range(points.shape[1]):
        valid = visibility[:, channel]
        _require(bool(np.any(valid)), f"channel {channel} has no visible frame")
        channel_points = canonical[valid, channel]
        mean = np.mean(channel_points, axis=0)
        residual = (channel_points - mean) / 2.0
        normalized = residual / bbox_diagonal_image01[valid, None]
        norms = np.linalg.norm(normalized, axis=1)
        rms = float(np.sqrt(np.mean(norms * norms)))
        values.append(rms)
        channel_rows.append(
            {
                "channel": channel,
                "visible_frame_count": int(np.sum(valid)),
                "canonical_mean_xy": mean.tolist(),
                "rms_objdiag": rms,
                "raw_residual_norm_objdiag": [float(item) for item in norms],
            }
        )
    return {
        "per_channel": channel_rows,
        "median_rms_objdiag": float(np.median(values)),
        "maximum_rms_objdiag": float(np.max(values)),
        "summary": _descriptive(
            values,
            sample_unit="channel_canonical_rms_objdiag",
            aggregation_hierarchy=("stratum", "channel", "visible_frames"),
        ),
    }


def _pair_fit_residuals(
    points: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    visibility: np.ndarray,
    physical_states: Sequence[Mapping[str, Any]],
    transform: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    frame_index_by_id: Mapping[Any, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rms_values: list[float] = []
    for row in pair_rows:
        source_index = frame_index_by_id[row["source_frame"]]
        target_index = frame_index_by_id[row["target_frame"]]
        source_state = dict(physical_states[source_index])
        target_state = dict(physical_states[target_index])
        state_fields = {
            "pair_id": row["pair_id"],
            "source_physical_state": source_state,
            "target_physical_state": target_state,
            "source_theta_deg": float(source_state["theta_deg"]),
            "target_theta_deg": float(target_state["theta_deg"]),
            "source_projection_model": source_state["projection_model"],
            "target_projection_model": target_state["projection_model"],
            "source_depth_configuration": source_state[
                "depth_configuration"
            ],
            "target_depth_configuration": target_state[
                "depth_configuration"
            ],
            "direction": row["direction"],
            "stride": row["stride"],
            "evaluation_partition": row["partition"],
        }
        valid = visibility[source_index] & visibility[target_index]
        if not bool(np.any(valid)):
            rows.append(
                {
                    "source_frame_id": row["source_frame"],
                    "target_frame_id": row["target_frame"],
                    "partition": row["partition"],
                    **state_fields,
                    "visible_correspondence_count": 0,
                    "rms_normalized_xy": None,
                    "status": "void_no_jointly_visible_correspondence",
                }
            )
            continue
        predicted = apply_affine(points[source_index, valid], A, b)
        residual = predicted - points[target_index, valid]
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        rms_values.append(rms)
        rows.append(
            {
                "source_frame_id": row["source_frame"],
                "target_frame_id": row["target_frame"],
                "partition": row["partition"],
                **state_fields,
                "visible_correspondence_count": int(np.sum(valid)),
                "rms_normalized_xy": rms,
                "status": "available",
            }
        )
    _require(bool(rms_values), "no visible pair residual is available")
    return {
        "terminology": "fitted_reference_residual_not_canonical_exactness",
        "reference_transform": transform["reference_transform"],
        "summary": _descriptive(
            rms_values,
            sample_unit="evaluation_pair_rms_normalized_xy",
            aggregation_hierarchy=(
                "stratum",
                "evaluation_pair",
                "jointly_visible_channels",
            ),
        ),
        "pairs": rows,
    }


def _bbox_diagonals_from_masks(masks: np.ndarray) -> np.ndarray:
    """Return mask bounding-box diagonals on the endpoint-aligned [0,1] grid."""

    height, width = masks.shape[-2:]
    diagonals: list[float] = []
    for frame_index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        _require(rows.size > 0, f"mask frame {frame_index} is empty")
        width01 = float(columns.max() - columns.min()) / float(width - 1)
        height01 = float(rows.max() - rows.min()) / float(height - 1)
        diagonal = math.hypot(width01, height01)
        _require(diagonal > 0.0, f"mask frame {frame_index} has zero bounding-box diagonal")
        diagonals.append(diagonal)
    return np.asarray(diagonals, dtype=np.float64)


def _sample_masks(
    points: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = masks.shape[-2:]
    u = np.rint((points[..., 0] + 1.0) * (width - 1) / 2.0).astype(np.int64)
    v = np.rint((points[..., 1] + 1.0) * (height - 1) / 2.0).astype(np.int64)
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside = np.zeros(in_image.shape, dtype=bool)
    for frame_index in range(points.shape[0]):
        valid_channels = np.flatnonzero(in_image[frame_index])
        inside[frame_index, valid_channels] = masks[
            frame_index,
            v[frame_index, valid_channels],
            u[frame_index, valid_channels],
        ]
    return inside, in_image


def _channel_health(
    points: np.ndarray,
    masks: np.ndarray,
    logits: np.ndarray | None,
    visibility: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    frame_index_by_id: Mapping[Any, int],
    estimator: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    displacement_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    for row in pair_rows:
        source_index = frame_index_by_id[row["source_frame"]]
        target_index = frame_index_by_id[row["target_frame"]]
        displacement_rows.append(
            np.linalg.norm(
                ((points[target_index] - points[source_index]) / 2.0),
                axis=1,
            )
        )
        valid_rows.append(visibility[source_index] & visibility[target_index])
    displacements = np.stack(displacement_rows, axis=0)
    valid_motion = np.stack(valid_rows, axis=0)
    reference = float(config["motion_reference_magnitude_image01"])
    _require(math.isfinite(reference) and reference > 0.0, "motion reference must be positive")
    visible_transition_count = np.sum(valid_motion, axis=0)
    motion_evidence_available = visible_transition_count > 0
    motion = np.zeros(points.shape[1], dtype=np.float64)
    for channel in range(points.shape[1]):
        if motion_evidence_available[channel]:
            motion[channel] = float(
                np.mean(displacements[valid_motion[:, channel], channel])
            )
    fraction = motion / reference
    motion_threshold = float(config["motion_fraction_min"])
    _require(
        math.isfinite(motion_threshold) and motion_threshold > 0.0,
        "motion threshold must be strictly positive",
    )
    active = (fraction >= motion_threshold) & motion_evidence_available

    inside, in_image = _sample_masks(points, masks)
    on_object_rate = np.zeros(points.shape[1], dtype=np.float64)
    visible_frame_count = np.sum(visibility, axis=0)
    on_object_evidence_available = visible_frame_count > 0
    for channel in range(points.shape[1]):
        if on_object_evidence_available[channel]:
            on_object_rate[channel] = float(
                np.mean(inside[visibility[:, channel], channel])
            )
    on_object_threshold = float(config["on_object_rate_min"])
    _require(
        math.isfinite(on_object_threshold) and 0.0 <= on_object_threshold <= 1.0,
        "invalid on-object threshold",
    )
    on_object = (on_object_rate >= on_object_threshold) & on_object_evidence_available

    dead: np.ndarray | None = None
    entropy_median: list[float | None] = [None] * points.shape[1]
    peak_median: list[float | None] = [None] * points.shape[1]
    if logits is not None:
        probabilities = probabilities_from_logits(
            logits,
            temperature=float(estimator["temperature"]),
            softmax_dtype=str(estimator["softmax_dtype"]),
        ).astype(np.float64)
        flat = probabilities.reshape(points.shape[0], points.shape[1], -1)
        safe_log = np.zeros_like(flat)
        positive = flat > 0
        safe_log[positive] = np.log(flat[positive])
        entropy = -np.sum(flat * safe_log, axis=2)
        peak = np.max(flat, axis=2)
        cells = flat.shape[-1]
        dead = (
            np.median(entropy, axis=0) > math.log(cells) - 0.5
        ) & (
            np.median(peak, axis=0) < 2.0 / cells
        )
        entropy_median = [float(item) for item in np.median(entropy, axis=0)]
        peak_median = [float(item) for item in np.median(peak, axis=0)]

    rows = []
    for channel in range(points.shape[1]):
        if not active[channel] and not on_object[channel]:
            label = "inactive_off_object"
        elif not active[channel]:
            label = "motion_inactive"
        elif not on_object[channel]:
            label = "active_off_object"
        else:
            label = "active_on_object"
        rows.append(
            {
                "channel": channel,
                "motion_energy_image01": float(motion[channel]),
                "fraction_of_known_transform_magnitude": float(fraction[channel]),
                "motion_active": bool(active[channel]),
                "visible_transition_count": int(visible_transition_count[channel]),
                "motion_evidence_void": bool(
                    not motion_evidence_available[channel]
                ),
                "on_object_rate": float(on_object_rate[channel]),
                "on_object": bool(on_object[channel]),
                "visible_frame_count": int(visible_frame_count[channel]),
                "on_object_evidence_void": bool(
                    not on_object_evidence_available[channel]
                ),
                "in_image_rate": float(np.mean(in_image[:, channel])),
                "heatmap_entropy_median": entropy_median[channel],
                "heatmap_peak_probability_median": peak_median[channel],
                "heatmap_flat_dead": (
                    bool(dead[channel]) if dead is not None else None
                ),
                "health_label": label,
            }
        )
    eligible = active & on_object
    return {
        "channels": rows,
        "motion_active_count": int(np.sum(active)),
        "on_object_count": int(np.sum(on_object)),
        "active_on_object_count": int(np.sum(active & on_object)),
        "heatmap_flat_dead_count": (
            int(np.sum(dead)) if dead is not None else None
        ),
        "heatmap_health_evidence": (
            "available" if dead is not None else "void_missing_logits"
        ),
        "eligibility_includes_heatmap_health": False,
        "eligibility_definition": "motion_active_and_on_object_only",
        "eligible_count": int(np.sum(eligible)),
        "eligible_indices": [int(item) for item in np.flatnonzero(eligible)],
    }


def _assignment_diagnostics(cost: np.ndarray) -> dict[str, Any]:
    """Return the best assignment and the two lowest exact costs.

    Exact equality of the two lowest costs means channel identity is
    unidentifiable from this transition.  The deterministic lexicographic
    representative remains useful for reproducibility, but must not be
    interpreted as evidence for one identity assignment.
    """

    matrix = _finite_array(cost, name="assignment cost")
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "cost must be square")
    size = matrix.shape[0]
    _require(size <= 16, "exact assignment supports at most 16 channels")
    states: dict[int, list[tuple[float, tuple[int, ...]]]] = {
        0: [(0.0, ())]
    }
    for row in range(size):
        candidates_by_mask: dict[
            int,
            list[tuple[float, tuple[int, ...]]],
        ] = {}
        for mask, ranked in states.items():
            for total, assignment in ranked:
                for column in range(size):
                    bit = 1 << column
                    if mask & bit:
                        continue
                    candidate = (
                        total + float(matrix[row, column]),
                        assignment + (column,),
                    )
                    candidates_by_mask.setdefault(mask | bit, []).append(
                        candidate
                    )
        next_states: dict[int, list[tuple[float, tuple[int, ...]]]] = {}
        for mask, candidates in candidates_by_mask.items():
            candidates.sort()
            next_states[mask] = candidates[:2]
        states = next_states
    ranked = states[(1 << size) - 1]
    best_cost, best_assignment = ranked[0]
    second_best_cost = ranked[1][0] if len(ranked) > 1 else None
    margin = (
        float(second_best_cost - best_cost)
        if second_best_cost is not None
        else None
    )
    exact_tie = bool(
        second_best_cost is not None and second_best_cost == best_cost
    )
    return {
        "deterministic_representative_assignment": list(best_assignment),
        "best_cost": float(best_cost),
        "second_best_cost": (
            float(second_best_cost)
            if second_best_cost is not None
            else None
        ),
        "best_vs_second_margin": margin,
        "exact_tie": exact_tie,
        "unique_optimum": not exact_tie,
    }


def _optimal_assignment(cost: np.ndarray) -> list[int]:
    """Compatibility wrapper returning the deterministic representative."""

    return _assignment_diagnostics(cost)[
        "deterministic_representative_assignment"
    ]


def _pair_graph_components(
    pair_rows: Sequence[Mapping[str, Any]],
    frame_index_by_id: Mapping[Any, int],
) -> dict[str, Any]:
    """Decompose a one-in/one-out transition graph into paths and cycles."""

    source_to_transition: dict[Any, int] = {}
    target_to_transition: dict[Any, int] = {}
    duplicate_sources: list[Any] = []
    duplicate_targets: list[Any] = []
    for transition_index, row in enumerate(pair_rows):
        source = row["source_frame"]
        target = row["target_frame"]
        if source in source_to_transition:
            duplicate_sources.append(source)
        else:
            source_to_transition[source] = transition_index
        if target in target_to_transition:
            duplicate_targets.append(target)
        else:
            target_to_transition[target] = transition_index
    if duplicate_sources or duplicate_targets:
        return {
            "status": "not_computed_branching_or_merging_pair_graph",
            "duplicate_source_frame_ids": sorted(
                set(duplicate_sources),
                key=lambda frame_id: frame_index_by_id[frame_id],
            ),
            "duplicate_target_frame_ids": sorted(
                set(duplicate_targets),
                key=lambda frame_id: frame_index_by_id[frame_id],
            ),
            "components": [],
        }

    visited: set[int] = set()
    components: list[dict[str, Any]] = []

    def walk(start_transition: int) -> list[int]:
        transition_indices: list[int] = []
        current = start_transition
        while current not in visited:
            visited.add(current)
            transition_indices.append(current)
            next_source = pair_rows[current]["target_frame"]
            next_transition = source_to_transition.get(next_source)
            if next_transition is None:
                break
            current = next_transition
        return transition_indices

    path_starts = [
        transition_index
        for transition_index, row in enumerate(pair_rows)
        if row["source_frame"] not in target_to_transition
    ]
    path_starts.sort(
        key=lambda transition_index: frame_index_by_id[
            pair_rows[transition_index]["source_frame"]
        ]
    )
    for start_transition in path_starts:
        if start_transition in visited:
            continue
        transition_indices = walk(start_transition)
        components.append(
            {
                "kind": "path",
                "cyclic": False,
                "transition_indices": transition_indices,
            }
        )

    remaining = sorted(
        set(range(len(pair_rows))) - visited,
        key=lambda transition_index: frame_index_by_id[
            pair_rows[transition_index]["source_frame"]
        ],
    )
    for start_transition in remaining:
        if start_transition in visited:
            continue
        transition_indices = walk(start_transition)
        cyclic = bool(
            pair_rows[transition_indices[-1]]["target_frame"]
            == pair_rows[transition_indices[0]]["source_frame"]
        )
        _require(
            cyclic,
            "pair graph component without a path start is not a cycle",
        )
        components.append(
            {
                "kind": "cycle",
                "cyclic": True,
                "transition_indices": transition_indices,
            }
        )

    _require(
        len(visited) == len(pair_rows),
        "pair graph decomposition omitted a transition",
    )
    normalized_components = []
    for component_index, component in enumerate(components):
        indices = component["transition_indices"]
        normalized_components.append(
            {
                "component_id": component_index,
                "kind": component["kind"],
                "cyclic": component["cyclic"],
                "transition_indices": indices,
                "transition_count": len(indices),
                "pair_ids": [
                    pair_rows[index]["pair_id"] for index in indices
                ],
                "source_frame_ids": [
                    pair_rows[index]["source_frame"] for index in indices
                ],
                "target_frame_ids": [
                    pair_rows[index]["target_frame"] for index in indices
                ],
            }
        )
    return {
        "status": "computed_disjoint_pair_graph_paths_and_cycles",
        "duplicate_source_frame_ids": [],
        "duplicate_target_frame_ids": [],
        "components": normalized_components,
    }


def _true_run_lengths(
    values: Sequence[bool | None],
    *,
    cyclic: bool,
) -> list[int]:
    """Count consecutive true transitions, with void/ambiguous values breaking runs."""

    if not values:
        return []
    if all(value is True for value in values):
        return [len(values)]
    run_lengths: list[int] = []
    current = 0
    for value in values:
        if value is True:
            current += 1
        elif current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    if (
        cyclic
        and values[0] is True
        and values[-1] is True
        and len(run_lengths) >= 2
    ):
        run_lengths[0] += run_lengths[-1]
        run_lengths.pop()
    return run_lengths


def _physical_change_duration_summary(
    series_by_channel: Sequence[Sequence[bool | None]],
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize switch persistence along each directed path or cycle."""

    channel_rows: list[dict[str, Any]] = []
    for channel, series in enumerate(series_by_channel):
        component_rows: list[dict[str, Any]] = []
        all_run_lengths: list[int] = []
        for component in components:
            indices = component["transition_indices"]
            values = [series[index] for index in indices]
            observed_count = sum(value is not None for value in values)
            void_count = sum(value is None for value in values)
            if observed_count == 0:
                component_evidence_status = (
                    "void_no_unambiguous_transitions"
                )
            elif void_count:
                component_evidence_status = (
                    "partial_ambiguous_or_void_breaks_runs"
                )
            else:
                component_evidence_status = "complete"
            run_lengths = _true_run_lengths(
                values,
                cyclic=bool(component["cyclic"]),
            )
            all_run_lengths.extend(run_lengths)
            component_rows.append(
                {
                    "component_id": component["component_id"],
                    "kind": component["kind"],
                    "cyclic": component["cyclic"],
                    "transition_count": len(values),
                    "duration_evidence_status": component_evidence_status,
                    "observed_unambiguous_transition_count": observed_count,
                    "ambiguous_or_void_transition_count": void_count,
                    "changed_transition_count": sum(
                        value is True for value in values
                    ),
                    "change_durations_transitions": run_lengths,
                }
            )
        observed_total = sum(value is not None for value in series)
        void_total = sum(value is None for value in series)
        if observed_total == 0:
            evidence_status = "void_no_unambiguous_transitions"
        elif void_total:
            evidence_status = "partial_ambiguous_or_void_breaks_runs"
        else:
            evidence_status = "complete"
        channel_rows.append(
            {
                "channel": channel,
                "duration_evidence_status": evidence_status,
                "observed_unambiguous_transition_count": observed_total,
                "ambiguous_or_void_transition_count": void_total,
                "components": component_rows,
                "all_change_durations_transitions": all_run_lengths,
                "maximum_change_duration_transitions": (
                    (
                        max(all_run_lengths)
                        if all_run_lengths
                        else 0
                    )
                    if observed_total > 0
                    else None
                ),
            }
        )
    return channel_rows


def _switching(
    points: np.ndarray,
    reference_A: np.ndarray,
    reference_b: np.ndarray,
    reference_label: str,
    logits: np.ndarray | None,
    visibility: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    frame_index_by_id: Mapping[Any, int],
) -> dict[str, Any]:
    transition_rows = []
    changes_by_channel: list[list[bool | None]] = [
        [] for _ in range(points.shape[1])
    ]
    mode_changes_by_channel: list[list[bool | None]] = [
        [] for _ in range(points.shape[1])
    ]
    modes_vu = None
    if logits is not None:
        flat = logits.reshape(logits.shape[0], logits.shape[1], -1)
        modes = np.argmax(flat, axis=2)
        width = logits.shape[-1]
        modes_vu = np.stack([modes // width, modes % width], axis=-1)

    ordered_pair_rows = sorted(
        pair_rows,
        key=lambda row: (
            frame_index_by_id[row["source_frame"]],
            frame_index_by_id[row["target_frame"]],
            str(row["pair_id"]),
        ),
    )
    for row in ordered_pair_rows:
        source_index = frame_index_by_id[row["source_frame"]]
        target_index = frame_index_by_id[row["target_frame"]]
        visible_indices = np.flatnonzero(
            visibility[source_index] & visibility[target_index]
        )
        if visible_indices.size < 2:
            for channel in range(points.shape[1]):
                changes_by_channel[channel].append(None)
                mode_changes_by_channel[channel].append(None)
            transition_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "source_frame_id": row["source_frame"],
                    "target_frame_id": row["target_frame"],
                    "partition": row["partition"],
                    "visible_channels": [int(item) for item in visible_indices],
                    "evidence_status": "void_fewer_than_two_jointly_visible_channels",
                    "assignment": None,
                    "changed_channels": None,
                    "changed_channel_count": None,
                    "assignment_cost": None,
                    "assignment_second_best_cost": None,
                    "assignment_best_vs_second_margin": None,
                    "assignment_unique": None,
                    "assignment_status": "void",
                    "mode_transform_residual_by_channel": None,
                    "mode_assignment": None,
                    "mode_changed_channels": None,
                    "mode_changed_channel_count": None,
                    "mode_assignment_cost": None,
                    "mode_assignment_second_best_cost": None,
                    "mode_assignment_best_vs_second_margin": None,
                    "mode_assignment_unique": None,
                    "mode_assignment_status": "void",
                }
            )
            continue

        predicted = apply_affine(
            points[source_index, visible_indices],
            reference_A,
            reference_b,
        )
        cost = np.linalg.norm(
            predicted[:, None, :]
            - points[target_index, visible_indices][None, :, :],
            axis=2,
        )
        assignment_diagnostics = _assignment_diagnostics(cost)
        local_assignment = assignment_diagnostics[
            "deterministic_representative_assignment"
        ]
        assignment = {
            int(visible_indices[local_source]): int(visible_indices[local_target])
            for local_source, local_target in enumerate(local_assignment)
        }
        assignment_unique = bool(assignment_diagnostics["unique_optimum"])
        changed = (
            [
                source_channel
                for source_channel, target_channel in assignment.items()
                if source_channel != target_channel
            ]
            if assignment_unique
            else None
        )
        for channel in range(points.shape[1]):
            changes_by_channel[channel].append(
                (
                    channel in changed
                    if assignment_unique and channel in assignment
                    else None
                )
            )

        mode_residual = None
        mode_assignment = None
        mode_changed = None
        mode_assignment_diagnostics = None
        if modes_vu is not None:
            height, width = logits.shape[-2:]
            source_modes = modes_vu[source_index, visible_indices]
            target_modes = modes_vu[target_index, visible_indices]
            source_xy = np.stack(
                [
                    -1.0 + 2.0 * source_modes[:, 1] / float(width - 1),
                    -1.0 + 2.0 * source_modes[:, 0] / float(height - 1),
                ],
                axis=1,
            )
            target_xy = np.stack(
                [
                    -1.0 + 2.0 * target_modes[:, 1] / float(width - 1),
                    -1.0 + 2.0 * target_modes[:, 0] / float(height - 1),
                ],
                axis=1,
            )
            predicted_modes = apply_affine(
                source_xy,
                reference_A,
                reference_b,
            )
            residuals = np.linalg.norm(predicted_modes - target_xy, axis=1)
            mode_residual = [
                {
                    "channel": int(channel),
                    "source_mode_vu": [
                        int(item) for item in source_mode
                    ],
                    "target_mode_vu": [
                        int(item) for item in target_mode
                    ],
                    "residual_normalized_xy": float(residual),
                }
                for channel, source_mode, target_mode, residual in zip(
                    visible_indices,
                    source_modes,
                    target_modes,
                    residuals,
                )
            ]
            mode_cost = np.linalg.norm(
                predicted_modes[:, None, :] - target_xy[None, :, :],
                axis=2,
            )
            mode_assignment_diagnostics = _assignment_diagnostics(mode_cost)
            mode_local_assignment = mode_assignment_diagnostics[
                "deterministic_representative_assignment"
            ]
            mode_assignment = {
                int(visible_indices[local_source]): int(
                    visible_indices[local_target]
                )
                for local_source, local_target in enumerate(
                    mode_local_assignment
                )
            }
            mode_assignment_unique = bool(
                mode_assignment_diagnostics["unique_optimum"]
            )
            mode_changed = (
                [
                    source_channel
                    for source_channel, target_channel in mode_assignment.items()
                    if source_channel != target_channel
                ]
                if mode_assignment_unique
                else None
            )
            for channel in range(points.shape[1]):
                mode_changes_by_channel[channel].append(
                    (
                        channel in mode_changed
                        if mode_assignment_unique
                        and channel in mode_assignment
                        else None
                    )
                )
        else:
            for channel in range(points.shape[1]):
                mode_changes_by_channel[channel].append(None)
        transition_rows.append(
            {
                "pair_id": row["pair_id"],
                "source_frame_id": row["source_frame"],
                "target_frame_id": row["target_frame"],
                "partition": row["partition"],
                "visible_channels": [int(item) for item in visible_indices],
                "evidence_status": "available",
                "assignment": assignment,
                "changed_channels": changed,
                "changed_channel_count": (
                    len(changed) if changed is not None else None
                ),
                "assignment_cost": assignment_diagnostics["best_cost"],
                "assignment_second_best_cost": assignment_diagnostics[
                    "second_best_cost"
                ],
                "assignment_best_vs_second_margin": (
                    assignment_diagnostics["best_vs_second_margin"]
                ),
                "assignment_unique": assignment_unique,
                "assignment_status": (
                    "unique_optimum"
                    if assignment_unique
                    else "ambiguous_exact_tie"
                ),
                "mode_transform_residual_by_channel": mode_residual,
                "mode_assignment": mode_assignment,
                "mode_changed_channels": mode_changed,
                "mode_changed_channel_count": (
                    len(mode_changed) if mode_changed is not None else None
                ),
                "mode_assignment_cost": (
                    mode_assignment_diagnostics["best_cost"]
                    if mode_assignment_diagnostics is not None
                    else None
                ),
                "mode_assignment_second_best_cost": (
                    mode_assignment_diagnostics["second_best_cost"]
                    if mode_assignment_diagnostics is not None
                    else None
                ),
                "mode_assignment_best_vs_second_margin": (
                    mode_assignment_diagnostics["best_vs_second_margin"]
                    if mode_assignment_diagnostics is not None
                    else None
                ),
                "mode_assignment_unique": (
                    mode_assignment_diagnostics["unique_optimum"]
                    if mode_assignment_diagnostics is not None
                    else None
                ),
                "mode_assignment_status": (
                    (
                        "unique_optimum"
                        if mode_assignment_diagnostics["unique_optimum"]
                        else "ambiguous_exact_tie"
                    )
                    if mode_assignment_diagnostics is not None
                    else "void_missing_logits"
                ),
            }
        )

    changed_transition_count = sum(
        value is True
        for series in changes_by_channel
        for value in series
    )
    mode_changed_transition_count = sum(
        value is True
        for series in mode_changes_by_channel
        for value in series
    )
    pair_graph = _pair_graph_components(
        ordered_pair_rows,
        frame_index_by_id,
    )
    physical_durations_available = (
        pair_graph["status"]
        == "computed_disjoint_pair_graph_paths_and_cycles"
    )
    assignment_physical_durations = (
        _physical_change_duration_summary(
            changes_by_channel,
            pair_graph["components"],
        )
        if physical_durations_available
        else None
    )
    mode_physical_durations = (
        _physical_change_duration_summary(
            mode_changes_by_channel,
            pair_graph["components"],
        )
        if physical_durations_available and modes_vu is not None
        else None
    )
    pair_graph_report = {
        "status": pair_graph["status"],
        "duplicate_source_frame_ids": pair_graph[
            "duplicate_source_frame_ids"
        ],
        "duplicate_target_frame_ids": pair_graph[
            "duplicate_target_frame_ids"
        ],
        "components": [
            {
                key: value
                for key, value in component.items()
                if key != "transition_indices"
            }
            for component in pair_graph["components"]
        ],
    }

    result: dict[str, Any] = {
        "reference_transform": reference_label,
        "reference_A": reference_A.tolist(),
        "reference_b": reference_b.tolist(),
        "assignment_changed_channel_event_count": changed_transition_count,
        "assignment_ambiguous_transition_count": sum(
            row["assignment_status"] == "ambiguous_exact_tie"
            for row in transition_rows
        ),
        "assignment_physical_change_durations_by_channel": (
            assignment_physical_durations
        ),
        "heatmap_mode_assignment_changed_channel_event_count": (
            mode_changed_transition_count if modes_vu is not None else None
        ),
        "heatmap_mode_assignment_ambiguous_transition_count": (
            sum(
                row["mode_assignment_status"] == "ambiguous_exact_tie"
                for row in transition_rows
            )
            if modes_vu is not None
            else None
        ),
        "heatmap_mode_assignment_physical_change_durations_by_channel": (
            mode_physical_durations
        ),
        "transitions": transition_rows,
        "transition_ordering": (
            "source_frame_index_then_target_frame_index_then_pair_id"
        ),
        "pair_graph_duration_decomposition": pair_graph_report,
        "physical_change_duration_status": pair_graph["status"],
        "physical_change_duration_unit": (
            "consecutive_changed_transitions_along_one_pair_graph_path_or_cycle"
        ),
        "physical_change_duration_boundary_rule": (
            "void_or_ambiguous_transition_breaks_run_and_cycle_end_joins_start"
        ),
        "assignment_event_count_definition": (
            "sum_over_unique_minimum_cost_assignments_only"
        ),
        "heatmap_mode_assignment_definition": (
            "minimum_cost_one_to_one_assignment_between_reference_transformed_"
            "source_argmax_cells_and_target_argmax_cells"
        ),
        "heatmap_mode_evidence": (
            "available" if logits is not None else "void_missing_logits"
        ),
    }
    if modes_vu is not None:
        result["heatmap_modes_vu"] = modes_vu.tolist()
    return result


def _rollout_horizon(
    points: np.ndarray,
    frame_ids: Sequence[Any],
    A: np.ndarray,
    b: np.ndarray,
    *,
    starts: Sequence[int],
    targets: Sequence[int],
    k: int,
    require_identity_denominator: bool = True,
) -> dict[str, Any]:
    _require(len(starts) == len(targets) and len(starts) > 0, "empty rollout horizon")
    Ak, bk = affine_power(A, b, k)
    model_values = []
    identity_values = []
    rows = []
    for source_index, target_index in zip(starts, targets):
        predicted = apply_affine(points[source_index], Ak, bk)
        actual = points[target_index]
        model_mse = float(np.mean((predicted - actual) ** 2))
        identity_mse = float(np.mean((points[source_index] - actual) ** 2))
        model_values.append(model_mse)
        identity_values.append(identity_mse)
        rows.append(
            {
                "source_index": int(source_index),
                "target_index": int(target_index),
                "source_frame_id": frame_ids[source_index],
                "target_frame_id": frame_ids[target_index],
                "model_mse": model_mse,
                "identity_mse": identity_mse,
            }
        )
    denominator = float(np.mean(identity_values))
    numerator = float(np.mean(model_values))
    _require(math.isfinite(denominator), "non-finite identity denominator")
    identity_valid = denominator > 0.0
    if require_identity_denominator:
        _require(identity_valid, "invalid identity denominator")
    return {
        "k": k,
        "model_mse": _descriptive(
            model_values,
            sample_unit="rollout_start_target_pair_model_mse",
            aggregation_hierarchy=(
                "stratum",
                "horizon",
                "start_target_pair",
                "channels",
                "xy",
            ),
        ),
        "identity_mse": _descriptive(
            identity_values,
            sample_unit="rollout_start_target_pair_identity_mse",
            aggregation_hierarchy=(
                "stratum",
                "horizon",
                "start_target_pair",
                "channels",
                "xy",
            ),
        ),
        "model_mse_mean": numerator,
        "identity_mse_mean": denominator,
        "identity_relative_valid": identity_valid,
        "model_over_identity": float(numerator / denominator) if identity_valid else None,
        "starts": [int(item) for item in starts],
        "targets": [int(item) for item in targets],
        "start_frame_ids": [frame_ids[int(item)] for item in starts],
        "target_frame_ids": [frame_ids[int(item)] for item in targets],
        "rows": rows,
    }


def _rollout_metrics(
    points: np.ndarray,
    frame_ids: Sequence[Any],
    A: np.ndarray,
    b: np.ndarray,
    transform: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require(bool(transform["cyclic"]), "roll full-corpus rollout requires cyclic=true")
    stride = int(transform["stride"])
    count = points.shape[0]
    full_horizons = config["full_rollout_horizons"]
    _require(
        isinstance(full_horizons, list) and bool(full_horizons),
        "full rollout horizons must be a non-empty list",
    )
    _require(
        full_horizons == list(range(1, max(full_horizons) + 1)),
        "full rollout horizons must be consecutive from one",
    )
    full_rows = []
    for k in full_horizons:
        targets = [int((start + k * stride) % count) for start in range(count)]
        full_rows.append(
            _rollout_horizon(
                points,
                frame_ids,
                A,
                b,
                starts=list(range(count)),
                targets=targets,
                k=int(k),
            )
        )
    full_auc = float(np.mean([row["model_over_identity"] for row in full_rows]))

    holdout_ids = config["role_scoped_holdout_frame_ids"]
    index_by_id = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    _require(len(holdout_ids) == len(set(holdout_ids)), "duplicate holdout frame id")
    _require(all(item in index_by_id for item in holdout_ids), "unknown holdout frame id")
    holdout_indices = sorted(index_by_id[item] for item in holdout_ids)
    holdout_set = set(holdout_indices)
    holdout_rows = []
    for k in config["holdout_rollout_horizons"]:
        starts = [
            index
            for index in holdout_indices
            if index + int(k) * stride in holdout_set
        ]
        targets = [index + int(k) * stride for index in starts]
        holdout_rows.append(
            _rollout_horizon(
                points,
                frame_ids,
                A,
                b,
                starts=starts,
                targets=targets,
                k=int(k),
            )
        )
    holdout_auc = float(np.mean([row["model_over_identity"] for row in holdout_rows]))

    closure_k = int(config["closure_horizon"])
    closure_targets = [
        int((start + closure_k * stride) % count) for start in range(count)
    ]
    closure = _rollout_horizon(
        points,
        frame_ids,
        A,
        b,
        starts=list(range(count)),
        targets=closure_targets,
        k=closure_k,
        require_identity_denominator=False,
    )
    return {
        "full_corpus_identity_normalized_auc": {
            "value": full_auc,
            "horizons": full_rows,
            "selection_status": "diagnostic_only_not_heldout",
            "aggregation": "unweighted_mean_of_horizon_ratios",
        },
        "role_scoped_holdout_identity_normalized_auc": {
            "value": holdout_auc,
            "horizons": holdout_rows,
            "selection_status": (
                "role_scoped_endpoint_subset_not_heldout_without_"
                "proven_non_exposure"
            ),
            "training_exposure_status": (
                "not_established_by_frame_membership"
            ),
            "aggregation": "unweighted_mean_of_horizon_ratios",
        },
        "closure": {
            **closure,
            "selection_status": "separate_diagnostic_never_in_auc",
        },
    }


def _validate_estimator(estimator: Mapping[str, Any]) -> None:
    required = {
        "input_height",
        "input_width",
        "heatmap_height",
        "heatmap_width",
        "endpoint_grid",
        "temperature",
        "logit_dtype",
        "softmax_dtype",
        "crop",
        "resize",
        "align_corners",
    }
    _require_exact_keys(estimator, required=required, name="estimator metadata")
    for key in ("input_height", "input_width", "heatmap_height", "heatmap_width"):
        _require(isinstance(estimator[key], int) and estimator[key] > 1, f"invalid {key}")
    _require(estimator["endpoint_grid"] is True, "endpoint_grid must be true")
    _require(
        isinstance(estimator["temperature"], (int, float))
        and math.isfinite(float(estimator["temperature"]))
        and float(estimator["temperature"]) > 0.0,
        "estimator temperature must be finite and positive",
    )
    _require(estimator["logit_dtype"] in {"float32", "float64"}, "unsupported logit_dtype")
    _require(estimator["softmax_dtype"] in {"float32", "float64"}, "unsupported softmax_dtype")
    resize = estimator["resize"]
    _require(
        isinstance(resize, list)
        and len(resize) == 2
        and all(isinstance(item, int) and item > 1 for item in resize),
        "estimator resize must be [height,width]",
    )
    _require(
        resize == [estimator["input_height"], estimator["input_width"]],
        "estimator resize disagrees with observed input shape",
    )
    _require(
        isinstance(estimator["align_corners"], (bool, type(None))),
        "align_corners must be bool or null",
    )


def _validate_file_record(record: Mapping[str, Any], *, name: str) -> dict[str, str]:
    _require(isinstance(record, Mapping), f"{name} must be a mapping")
    _require_exact_keys(
        record,
        required={"role", "absolute_path", "sha256"},
        name=name,
    )
    role = record["role"]
    _require(isinstance(role, str) and bool(role), f"{name} role is empty")
    path = _absolute_existing_file(record["absolute_path"], name=f"{name} absolute_path")
    claimed = record["sha256"]
    _require(_is_sha256(claimed), f"{name} sha256 is invalid")
    actual = _file_sha256(path)
    _require(actual == claimed, f"{name} sha256 mismatch")
    return {"role": role, "absolute_path": str(path), "sha256": actual}


def _validate_numeric_registry(
    *,
    estimator: Mapping[str, Any],
    validated_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    committed_files = validated_provenance.get("committed_files")
    _require(
        isinstance(committed_files, list),
        "validated provenance lacks committed_files",
    )
    records = [
        record
        for record in committed_files
        if isinstance(record, Mapping)
        and record.get("role") == "numeric_registry"
    ]
    _require(
        len(records) == 1,
        "validated provenance does not uniquely bind numeric_registry",
    )
    record = records[0]
    _require(
        record["absolute_path"] == str(AUTHORITATIVE_NUMERIC_REGISTRY_PATH),
        "numeric registry path is not the authoritative v1.1 path",
    )
    path = _absolute_existing_file(
        record["absolute_path"], name="numeric_registry absolute_path"
    )
    _require(
        record.get("file_sha256")
        == AUTHORITATIVE_NUMERIC_REGISTRY_FILE_SHA256,
        "numeric registry binding does not use the authoritative v1.1 file hash",
    )
    actual_file_hash = _file_sha256(path)
    _require(
        actual_file_hash == AUTHORITATIVE_NUMERIC_REGISTRY_FILE_SHA256,
        "authoritative numeric registry file hash mismatch",
    )
    registry = _load_strict_json(path, name="authoritative numeric registry")
    _require(isinstance(registry, Mapping), "numeric registry must be a mapping")
    _require(
        registry.get("schema_version")
        == AUTHORITATIVE_NUMERIC_REGISTRY_SCHEMA_VERSION,
        "authoritative numeric registry schema mismatch",
    )
    claimed_content = registry.get("content_sha256")
    _require(
        claimed_content == AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256,
        "authoritative numeric registry internal content hash mismatch",
    )
    payload = dict(registry)
    payload.pop("content_sha256")
    computed_content = canonical_sha256(payload)
    _require(
        computed_content == AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256,
        "authoritative numeric registry canonical content hash mismatch",
    )
    expected_key = (
        f"{estimator['softmax_dtype']}::spatial_expectation_coordinate"
    )
    entries = registry.get("metric_registries")
    _require(isinstance(entries, Mapping), "numeric registry lacks metric_registries")
    _require(expected_key in entries, f"numeric registry lacks {expected_key}")
    entry = entries[expected_key]
    _require(isinstance(entry, Mapping), "numeric registry coordinate entry is invalid")
    tolerance = entry.get("tolerance")
    _require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) >= 0.0,
        "numeric registry coordinate tolerance is invalid",
    )
    return {
        "absolute_path": str(path),
        "file_sha256": actual_file_hash,
        "content_sha256": AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256,
        "schema_version": AUTHORITATIVE_NUMERIC_REGISTRY_SCHEMA_VERSION,
        "coordinate_tolerance_key": expected_key,
        "coordinate_tolerance": float(tolerance),
    }


def _validate_evaluation_config(
    config: Mapping[str, Any], *, family: str
) -> None:
    protocol = config.get("protocol")
    _require(
        protocol in {"generic", "full_primary_roll"},
        "unsupported evaluation protocol",
    )
    required = set(BASE_EVALUATION_CONFIG_FIELDS)
    if protocol == "full_primary_roll":
        _require(family == "roll", "full_primary_roll protocol requires roll")
        required |= ROLL_EVALUATION_CONFIG_FIELDS
    _require_exact_keys(config, required=required, name="evaluation_config")
    horizons = config["operator_composition_horizons"]
    _require(
        isinstance(horizons, list)
        and all(isinstance(item, int) and item >= 1 for item in horizons)
        and len(horizons) == len(set(horizons)),
        "operator composition horizons are invalid",
    )
    _require(
        isinstance(config["motion_reference_magnitude_image01"], (int, float))
        and math.isfinite(float(config["motion_reference_magnitude_image01"]))
        and float(config["motion_reference_magnitude_image01"]) > 0.0,
        "motion reference magnitude is invalid",
    )
    _require(
        isinstance(config["motion_fraction_min"], (int, float))
        and not isinstance(config["motion_fraction_min"], bool)
        and math.isfinite(float(config["motion_fraction_min"]))
        and float(config["motion_fraction_min"]) > 0.0,
        "motion_fraction_min must be finite and strictly positive",
    )
    _require(
        isinstance(config["on_object_rate_min"], (int, float))
        and math.isfinite(float(config["on_object_rate_min"]))
        and 0.0 <= float(config["on_object_rate_min"]) <= 1.0,
        "on_object_rate_min is invalid",
    )
    _require(
        isinstance(config["minimum_eligible_channels"], int)
        and not isinstance(config["minimum_eligible_channels"], bool)
        and config["minimum_eligible_channels"] >= 2,
        "minimum_eligible_channels must be an integer of at least two",
    )
    if protocol == "full_primary_roll":
        for key in ("full_rollout_horizons", "holdout_rollout_horizons"):
            values = config[key]
            _require(
                isinstance(values, list)
                and bool(values)
                and all(isinstance(item, int) and item >= 1 for item in values)
                and len(values) == len(set(values)),
                f"{key} is invalid",
            )
        holdout_ids = config["role_scoped_holdout_frame_ids"]
        _require(
            isinstance(holdout_ids, list)
            and bool(holdout_ids)
            and len(holdout_ids) == len(set(holdout_ids)),
            "role_scoped_holdout_frame_ids is invalid",
        )
        _require(
            isinstance(config["closure_horizon"], int)
            and config["closure_horizon"] >= 1,
            "closure_horizon is invalid",
        )


def _rotation_matrix_degrees(angle_deg: float) -> np.ndarray:
    radians = math.radians(float(angle_deg))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _require_direction_matches_signed_generator(
    direction: object, signed_generator: float, *, name: str
) -> None:
    _require(direction in {"forward", "reverse"}, f"{name} direction is invalid")
    if direction == "forward":
        _require(signed_generator > 0.0, f"{name} forward direction requires positive generator")
    else:
        _require(signed_generator < 0.0, f"{name} reverse direction requires negative generator")


def _validate_transform(
    transform: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    common = {
        "family",
        "physical_axis",
        "direction",
        "signed_generator",
        "generator_units",
        "stride",
        "stride_units",
        "cyclic",
        "expected_2d_family",
    }
    family = transform.get("family")
    _require(family in SUPPORTED_FAMILIES, f"unknown transform family {family!r}")
    family_fields = {
        "roll": {"projected_centre_xy"},
        "translation": {
            "expected_image_component",
            "expected_image_sign",
            "translation_source",
        },
        "scale": {"projected_centre_xy"},
        "yaw": {"reference_transform"},
        "pitch": {"reference_transform"},
    }[str(family)]
    if (
        family == "translation"
        and transform.get("translation_source") == "rendered_world_calibrated"
    ):
        family_fields = family_fields | {"calibrated_image_delta_xy"}
    _require_exact_keys(
        transform,
        required=common | family_fields,
        name="transform",
    )
    _require(isinstance(transform["stride"], int) and transform["stride"] > 0, "invalid stride")
    _require(isinstance(transform["cyclic"], bool), "cyclic must be bool")
    _require(
        isinstance(transform["signed_generator"], (int, float))
        and not isinstance(transform["signed_generator"], bool)
        and math.isfinite(float(transform["signed_generator"])),
        "signed_generator must be finite",
    )
    generator = float(transform["signed_generator"])
    target_A: np.ndarray | None = None
    target_b: np.ndarray | None = None

    if family == "roll":
        _require(transform["physical_axis"] == "world_z", "roll physical_axis must be world_z")
        _require(transform["generator_units"] == "degrees", "roll generator units must be degrees")
        _require(transform["stride_units"] == "frames", "roll stride_units must be frames")
        _require(transform["cyclic"] is True, "roll must be cyclic")
        _require(
            transform["expected_2d_family"]
            == "planar_rotation_about_projected_center",
            "invalid roll expected_2d_family",
        )
        _require_direction_matches_signed_generator(
            transform["direction"], generator, name="roll"
        )
        expected_A = _rotation_matrix_degrees(generator)
        centre = _finite_array(
            transform["projected_centre_xy"],
            name="projected centre",
            shape=(2,),
        )
        expected_b = centre - expected_A @ centre
        target_A = expected_A
        target_b = expected_b
    elif family == "translation":
        source = transform["translation_source"]
        _require(
            transform["physical_axis"] in {"world_x", "world_y", "image_x", "image_y"},
            "invalid translation physical_axis",
        )
        _require(transform["cyclic"] is False, "translation must be non-cyclic")
        _require_direction_matches_signed_generator(
            transform["direction"], generator, name="translation"
        )
        if source == "rendered_world_calibrated":
            _require(
                transform["physical_axis"] in {"world_x", "world_y"},
                "rendered translation axis must be world_x or world_y",
            )
            _require(
                transform["generator_units"] == "world_units",
                "rendered translation generator_units must be world_units",
            )
            _require(
                transform["stride_units"] == "steps",
                "rendered translation stride_units must be steps",
            )
            _require(
                transform["expected_2d_family"]
                == "image_plane_translation_after_calibration",
                "invalid rendered translation expected_2d_family",
            )
        elif source == "synthetic_image_fixture":
            _require(
                transform["physical_axis"] in {"image_x", "image_y"},
                "synthetic image translation axis must be image_x or image_y",
            )
            _require(
                transform["generator_units"] == "normalized_image",
                "synthetic image translation units must be normalized_image",
            )
            _require(
                transform["stride_units"] == "frames",
                "synthetic image translation stride_units must be frames",
            )
            _require(
                transform["expected_2d_family"]
                == "synthetic_image_plane_translation",
                "invalid synthetic translation expected_2d_family",
            )
        else:
            raise EvaluationContractError(
                "translation_source must explicitly identify rendered calibration "
                "or a synthetic image fixture"
            )
        component = transform["expected_image_component"]
        expected_component = (
            0 if transform["physical_axis"] in {"world_x", "image_x"} else 1
        )
        _require(
            isinstance(component, int) and not isinstance(component, bool),
            "expected_image_component must be an integer",
        )
        _require(component == expected_component, "translation image component disagrees with axis")
        image_sign = transform["expected_image_sign"]
        _require(
            isinstance(image_sign, int)
            and not isinstance(image_sign, bool)
            and image_sign in {-1, 1},
            "expected_image_sign must be -1 or 1",
        )
        target_A = np.eye(2, dtype=np.float64)
        other = 1 - int(component)
        if source == "rendered_world_calibrated":
            target_b = _finite_array(
                transform["calibrated_image_delta_xy"],
                name="calibrated_image_delta_xy",
                shape=(2,),
            )
            _require(
                abs(float(target_b[other])) <= tolerance,
                "calibrated translation delta leaks orthogonally",
            )
        else:
            target_b = np.zeros(2, dtype=np.float64)
            target_b[int(component)] = generator * int(image_sign)
        _require(
            float(target_b[int(component)])
            * generator
            * int(image_sign)
            > 0.0,
            "translation target_b has the wrong locked image sign",
        )
    elif family == "scale":
        _require(transform["physical_axis"] == "uniform", "invalid scale axis")
        _require(transform["generator_units"] == "log_scale", "scale units must be log_scale")
        _require(transform["stride_units"] == "frames", "scale stride_units must be frames")
        _require(transform["cyclic"] is False, "scale must be non-cyclic")
        _require(
            transform["expected_2d_family"]
            == "uniform_scale_about_projected_center",
            "invalid scale expected_2d_family",
        )
        _require_direction_matches_signed_generator(
            transform["direction"], generator, name="scale"
        )
        expected_ratio = math.exp(generator)
        expected_A = expected_ratio * np.eye(2)
        centre = _finite_array(
            transform["projected_centre_xy"],
            name="projected centre",
            shape=(2,),
        )
        expected_b = centre - expected_A @ centre
        target_A = expected_A
        target_b = expected_b
    else:
        expected_axis = "world_y" if family == "yaw" else "world_x"
        _require(transform["physical_axis"] == expected_axis, f"{family} physical_axis is wrong")
        _require(transform["generator_units"] == "degrees", f"{family} units must be degrees")
        _require(transform["stride_units"] == "frames", f"{family} stride_units must be frames")
        _require(transform["cyclic"] is False, f"{family} must be non-cyclic")
        _require(
            transform["expected_2d_family"] == "local_affine_approximation",
            f"invalid {family} expected_2d_family",
        )
        _require(
            transform["reference_transform"] == "fitted_reference",
            f"{family} target transform must be explicitly labelled fitted_reference",
        )
        _require_direction_matches_signed_generator(
            transform["direction"], generator, name=str(family)
        )
    return str(family), target_A, target_b


def _finite_state_scalar(
    state: Mapping[str, Any],
    key: str,
    *,
    frame_id: Any,
) -> float:
    value = state[key]
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"physical state {frame_id!r} field {key} must be finite",
    )
    return float(value)


def _validate_and_derive_physical_states(
    *,
    family: str,
    physical_states: object,
    frame_ids: Sequence[Any],
    frame_index_by_id: Mapping[Any, int],
    transform: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    target_A: np.ndarray,
    target_b: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
    _require(
        isinstance(physical_states, list)
        and len(physical_states) == len(frame_ids),
        "physical_states must contain exactly one record per frame",
    )
    normalized: list[dict[str, Any]] = []
    state_A: np.ndarray | None = None
    state_b: np.ndarray | None = None
    if family in {"roll", "scale", "translation"}:
        state_A = np.empty((len(frame_ids), 2, 2), dtype=np.float64)
        state_b = np.empty((len(frame_ids), 2), dtype=np.float64)

    centre = None
    if family in {"roll", "scale"}:
        centre = _finite_array(
            transform["projected_centre_xy"],
            name="projected centre",
            shape=(2,),
        )

    for frame_index, (frame_id, raw_state) in enumerate(
        zip(frame_ids, physical_states)
    ):
        _require(
            isinstance(raw_state, Mapping),
            f"physical state {frame_id!r} must be a mapping",
        )
        state = dict(raw_state)
        if family == "roll":
            _require_exact_keys(
                state,
                required={"theta_deg"},
                name=f"roll physical state {frame_id!r}",
            )
            theta = _finite_state_scalar(state, "theta_deg", frame_id=frame_id)
            derived_A = _rotation_matrix_degrees(theta)
            derived_b = centre - derived_A @ centre
            state["theta_deg"] = theta
        elif family == "scale":
            _require_exact_keys(
                state,
                required={"s_rel"},
                name=f"scale physical state {frame_id!r}",
            )
            scale = _finite_state_scalar(state, "s_rel", frame_id=frame_id)
            _require(scale > 0.0, f"scale physical state {frame_id!r} must be positive")
            derived_A = scale * np.eye(2, dtype=np.float64)
            derived_b = centre - derived_A @ centre
            state["s_rel"] = scale
        elif family == "translation":
            if transform["translation_source"] == "rendered_world_calibrated":
                _require_exact_keys(
                    state,
                    required={
                        "dx_world",
                        "dy_world",
                        "grid_x",
                        "grid_y",
                        "calibrated_image_offset_xy",
                    },
                    name=f"rendered translation physical state {frame_id!r}",
                )
                state["dx_world"] = _finite_state_scalar(
                    state, "dx_world", frame_id=frame_id
                )
                state["dy_world"] = _finite_state_scalar(
                    state, "dy_world", frame_id=frame_id
                )
                for key in ("grid_x", "grid_y"):
                    _require(
                        isinstance(state[key], int)
                        and not isinstance(state[key], bool),
                        f"physical state {frame_id!r} field {key} must be an integer",
                    )
                offset = _finite_array(
                    state["calibrated_image_offset_xy"],
                    name=f"calibrated image offset for frame {frame_id!r}",
                    shape=(2,),
                )
                state["calibrated_image_offset_xy"] = offset.tolist()
            else:
                _require_exact_keys(
                    state,
                    required={"image_offset_xy"},
                    name=f"synthetic translation physical state {frame_id!r}",
                )
                offset = _finite_array(
                    state["image_offset_xy"],
                    name=f"synthetic image offset for frame {frame_id!r}",
                    shape=(2,),
                )
                state["image_offset_xy"] = offset.tolist()
            derived_A = np.eye(2, dtype=np.float64)
            derived_b = offset
        else:
            _require_exact_keys(
                state,
                required={
                    "theta_deg",
                    "projection_model",
                    "depth_configuration",
                },
                name=f"{family} physical state {frame_id!r}",
            )
            state["theta_deg"] = _finite_state_scalar(
                state, "theta_deg", frame_id=frame_id
            )
            for key in ("projection_model", "depth_configuration"):
                _require(
                    isinstance(state[key], str) and bool(state[key]),
                    f"physical state {frame_id!r} field {key} must be a non-empty string",
                )
        if state_A is not None and state_b is not None:
            state_A[frame_index] = derived_A
            state_b[frame_index] = derived_b
        normalized.append(state)

    if family in {"yaw", "pitch"}:
        projection_models = {state["projection_model"] for state in normalized}
        depth_configurations = {
            state["depth_configuration"] for state in normalized
        }
        _require(
            len(projection_models) == 1,
            f"{family} physical states mix projection models",
        )
        _require(
            len(depth_configurations) == 1,
            f"{family} physical states mix depth configurations",
        )

    for row_number, row in enumerate(pair_rows):
        source_index = frame_index_by_id[row["source_frame"]]
        target_index = frame_index_by_id[row["target_frame"]]
        source_state = normalized[source_index]
        target_state = normalized[target_index]
        if family in {"roll", "scale", "translation"}:
            _require(state_A is not None and state_b is not None, "missing derived state transform")
            inverse_source = np.linalg.inv(state_A[source_index])
            relative_A = state_A[target_index] @ inverse_source
            relative_b = (
                state_b[target_index] - relative_A @ state_b[source_index]
            )
            _require(
                float(np.max(np.abs(relative_A - target_A))) <= tolerance,
                f"pair row {row_number} physical states disagree with target_A",
            )
            _require(
                float(np.max(np.abs(relative_b - target_b))) <= tolerance,
                f"pair row {row_number} physical states disagree with target_b",
            )
            if (
                family == "translation"
                and transform["translation_source"]
                == "rendered_world_calibrated"
            ):
                component_key = (
                    "dx_world"
                    if transform["physical_axis"] == "world_x"
                    else "dy_world"
                )
                other_key = (
                    "dy_world"
                    if transform["physical_axis"] == "world_x"
                    else "dx_world"
                )
                grid_key = (
                    "grid_x"
                    if transform["physical_axis"] == "world_x"
                    else "grid_y"
                )
                other_grid_key = (
                    "grid_y"
                    if transform["physical_axis"] == "world_x"
                    else "grid_x"
                )
                _require(
                    abs(
                        float(target_state[component_key])
                        - float(source_state[component_key])
                        - float(transform["signed_generator"])
                    )
                    <= tolerance,
                    f"pair row {row_number} world displacement disagrees with signed_generator",
                )
                _require(
                    abs(
                        float(target_state[other_key])
                        - float(source_state[other_key])
                    )
                    <= tolerance,
                    f"pair row {row_number} leaks into the orthogonal world axis",
                )
                _require(
                    int(target_state[grid_key]) - int(source_state[grid_key])
                    == (
                        int(transform["stride"])
                        if transform["direction"] == "forward"
                        else -int(transform["stride"])
                    ),
                    f"pair row {row_number} grid step disagrees with direction and stride",
                )
                _require(
                    int(target_state[other_grid_key])
                    == int(source_state[other_grid_key]),
                    f"pair row {row_number} changes the orthogonal grid coordinate",
                )
        else:
            _require(
                abs(
                    float(target_state["theta_deg"])
                    - float(source_state["theta_deg"])
                    - float(transform["signed_generator"])
                )
                <= tolerance,
                f"pair row {row_number} angle states disagree with signed_generator",
            )
    return normalized, state_A, state_b


def _validate_primary_roll_protocol(
    *,
    frame_ids: Sequence[Any],
    transform: Mapping[str, Any],
    config: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the full-orbit diagnostic bundle.

    This bundle is deliberately distinct from a train/validation split bundle:
    it contains every one of the 180 cyclic +3 edges exactly once and labels
    every edge ``full_corpus``.  Frozen train/validation roles remain in the
    separately hash-bound split artifacts and role-scoped holdout frame list.
    Consequently this pair index may drive full-orbit descriptive metrics but
    may never be used by ``fit_from_pairs``.
    """

    if config["protocol"] != "full_primary_roll":
        return
    _require(frame_ids == list(range(180)), "full_primary_roll frame_ids must be ordered 0..179")
    _require(transform["stride"] == 3, "full_primary_roll stride must be 3")
    _require(
        float(transform["signed_generator"]) == 6.0,
        "full_primary_roll generator must be +6 degrees",
    )
    _require(transform["direction"] == "forward", "full_primary_roll direction must be forward")
    _require(
        config["full_rollout_horizons"] == list(range(1, 60)),
        "full_primary_roll full horizons must be k=1..59",
    )
    _require(
        config["holdout_rollout_horizons"] == list(range(1, 8)),
        "full_primary_roll holdout horizons must be k=1..7",
    )
    _require(config["closure_horizon"] == 60, "full_primary_roll closure must be k=60")
    _require(
        config["role_scoped_holdout_frame_ids"] == list(range(24)),
        "full_primary_roll holdout frame ids must be 0..23",
    )
    _require(
        len(pair_rows) == 180,
        "full_primary_roll pair_index must contain exactly 180 edges",
    )
    source_frames = [row["source_frame"] for row in pair_rows]
    _require(
        sorted(source_frames) == list(range(180)),
        "full_primary_roll pair_index must contain every source frame exactly once",
    )
    for row_number, row in enumerate(pair_rows):
        _require(
            row["target_frame"] == (row["source_frame"] + 3) % 180,
            f"full_primary_roll pair row {row_number} is not a +3 edge",
        )
        _require(
            row["partition"] == "full_corpus",
            f"full_primary_roll pair row {row_number} must use partition=full_corpus",
        )


def _validate_pair_rows(
    raw_rows: object,
    *,
    frame_ids: Sequence[Any],
    required_partition: str,
    transform: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    _require(
        isinstance(raw_rows, list) and bool(raw_rows),
        f"{name} is empty",
    )
    frame_set = set(frame_ids)
    seen_endpoints: set[tuple[Any, Any]] = set()
    seen_pair_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows):
        _require(
            isinstance(row, Mapping),
            f"{name} row {row_number} is not a mapping",
        )
        _require_exact_keys(
            row,
            required={
                "pair_id",
                "source_frame",
                "target_frame",
                "direction",
                "stride",
                "partition",
            },
            name=f"{name} row {row_number}",
        )
        _require(
            isinstance(row["pair_id"], str) and bool(row["pair_id"]),
            f"{name} row {row_number} pair_id is empty",
        )
        _require(
            row["pair_id"] not in seen_pair_ids,
            f"{name} contains duplicate pair_id",
        )
        seen_pair_ids.add(row["pair_id"])
        _require(
            row["source_frame"] in frame_set,
            f"{name} row {row_number} source is unknown",
        )
        _require(
            row["target_frame"] in frame_set,
            f"{name} row {row_number} target is unknown",
        )
        _require(
            row["source_frame"] != row["target_frame"],
            f"{name} row {row_number} is self-pair",
        )
        _require(
            row["direction"] == transform["direction"],
            f"{name} mixes pair directions",
        )
        _require(
            row["stride"] == transform["stride"],
            f"{name} mixes pair strides",
        )
        _require(
            row["partition"] == required_partition,
            f"{name} must contain only {required_partition} rows",
        )
        endpoints = (row["source_frame"], row["target_frame"])
        _require(
            endpoints not in seen_endpoints,
            f"{name} contains duplicate pair endpoints",
        )
        seen_endpoints.add(endpoints)
        normalized.append(dict(row))
    return normalized


def _resolve_operator(
    operator: Mapping[str, Any],
    *,
    fit: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    _require(isinstance(operator, Mapping), "operator must be a mapping")
    mode = operator.get("mode")
    if mode == "supplied":
        _require(
            fit is None,
            "supplied operator forbids a fit section",
        )
        _require_exact_keys(
            operator,
            required={"mode", "A", "b"},
            name="supplied operator",
        )
        A = _finite_array(operator["A"], name="operator A", shape=(2, 2))
        b = _finite_array(operator["b"], name="operator b", shape=(2,))
        return A, b, mode
    if mode == "fit_from_pairs":
        _require_exact_keys(
            operator,
            required={"mode"},
            name="fit_from_pairs operator",
        )
        _require(
            fit is not None,
            "fit_from_pairs requires a separate fit section",
        )
        source_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        points = fit["points"]
        visibility = fit["visibility"]
        frame_index_by_id = fit["frame_index_by_id"]
        for row in fit["pair_rows"]:
            source_index = frame_index_by_id[row["source_frame"]]
            target_index = frame_index_by_id[row["target_frame"]]
            valid = visibility[source_index] & visibility[target_index]
            if bool(np.any(valid)):
                source_rows.append(points[source_index, valid])
                target_rows.append(points[target_index, valid])
        _require(bool(source_rows), "fit_from_pairs has no visible training correspondence")
        source = np.concatenate(source_rows, axis=0)
        target = np.concatenate(target_rows, axis=0)
        A, b = fit_shared_affine(source, target)
        return A, b, mode
    raise EvaluationContractError("operator mode must be supplied or fit_from_pairs")


def _validate_frame_ids(value: object, *, name: str) -> list[Any]:
    _require(isinstance(value, list) and len(value) >= 2, f"invalid {name}")
    _require(
        all(
            isinstance(item, (int, str)) and not isinstance(item, bool)
            for item in value
        ),
        f"{name} must be integers or strings",
    )
    _require(len(value) == len(set(value)), f"duplicate {name[:-1]}")
    return list(value)


def _require_checkpoint_evaluator_authorization(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Validate committed checkpoint execution authority before external access.

    The trusted authorization module verifies the reviewed source/manifests and
    returns an audit receipt.  No field in the caller's bundle can directly
    enable this path: if the module or its validator is absent, malformed, or
    raises for any reason, checkpoint evaluation remains blocked.
    """

    try:
        authorization = importlib.import_module(CHECKPOINT_AUTHORIZATION_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != CHECKPOINT_AUTHORIZATION_MODULE:
            raise EvaluationContractError(
                f"checkpoint authorization module failed to import: {exc}",
                code="checkpoint_replay_blocked",
            ) from exc
        raise EvaluationContractError(
            "saved-checkpoint replay is blocked: checkpoint authorization module is absent",
            code="checkpoint_replay_blocked",
        ) from exc
    except Exception as exc:
        raise EvaluationContractError(
            f"checkpoint authorization module failed closed: {exc}",
            code="checkpoint_replay_blocked",
        ) from exc

    validator = getattr(authorization, CHECKPOINT_AUTHORIZATION_VALIDATOR, None)
    _require(
        callable(validator),
        "checkpoint authorization validator is absent",
    )
    try:
        receipt = validator(bundle)
    except Exception as exc:
        raise EvaluationContractError(
            f"saved-checkpoint replay is blocked: authorization failed: {exc}",
            code="checkpoint_replay_blocked",
        ) from exc

    raw_provenance = bundle.get("provenance")
    _require(
        isinstance(raw_provenance, Mapping),
        "checkpoint bundle provenance must be a mapping",
    )
    external_files = raw_provenance.get("external_files")
    _require(
        isinstance(external_files, list),
        "checkpoint provenance external_files must be a list",
    )
    checkpoint_records = [
        record
        for record in external_files
        if isinstance(record, Mapping) and record.get("role") == "checkpoint"
    ]
    _require(
        len(checkpoint_records) == 1,
        "checkpoint authorization requires one external checkpoint role",
    )
    consumer = getattr(
        authorization,
        CHECKPOINT_PROVENANCE_RECEIPT_CONSUMER,
        None,
    )
    if not callable(consumer):
        raise EvaluationContractError(
            "saved-checkpoint replay is blocked: checkpoint provenance "
            "receipt consumer is absent",
            code="checkpoint_replay_blocked",
        )
    try:
        checkpoint_provenance_load_receipt = consumer(
            source_commit=raw_provenance.get("source_commit"),
            checkpoint_record=checkpoint_records[0],
        )
    except Exception as exc:
        raise EvaluationContractError(
            "saved-checkpoint replay is blocked: checkpoint provenance "
            f"receipt consumption failed: {exc}",
            code="checkpoint_replay_blocked",
        ) from exc
    if not isinstance(checkpoint_provenance_load_receipt, Mapping):
        raise EvaluationContractError(
            "saved-checkpoint replay is blocked: checkpoint provenance "
            "receipt consumer returned no validated receipt",
            code="checkpoint_replay_blocked",
        )

    _require(
        isinstance(receipt, Mapping),
        "checkpoint authorization validator returned no audit receipt",
    )
    _require_exact_keys(
        receipt,
        required=CHECKPOINT_AUTHORIZATION_RECEIPT_FIELDS,
        name="checkpoint authorization receipt",
    )
    _require(
        receipt["checkpoint_evaluation_authorized"] is True,
        "checkpoint authorization receipt does not authorize evaluation",
    )
    _require(
        receipt["training_or_weight_update_authorized"] is False,
        "checkpoint authorization receipt must forbid training and weight updates",
    )
    _require(
        receipt["selection_use_authorized"] is False,
        "checkpoint authorization receipt must forbid selection use",
    )
    _require(
        isinstance(receipt["task_id"], int)
        and not isinstance(receipt["task_id"], bool)
        and receipt["task_id"] in {20, 55, 80},
        "checkpoint authorization receipt has an invalid task_id",
    )
    _require(
        isinstance(receipt["fixture_id"], str)
        and bool(receipt["fixture_id"])
        and receipt["fixture_id"] == bundle.get("case_id"),
        "checkpoint authorization fixture_id differs from case_id",
    )
    for field in (
        "checkpoint_sha256",
        "runtime_source_manifest_file_sha256",
        "checkpoint_hash_preflight_file_sha256",
    ):
        _require(
            _is_sha256(receipt[field]),
            f"checkpoint authorization receipt field {field} is invalid",
        )

    _require(
        receipt["source_commit"] == raw_provenance.get("source_commit"),
        "checkpoint authorization receipt binds a different source_commit",
    )
    _require(
        checkpoint_records[0].get("file_sha256")
        == receipt["checkpoint_sha256"],
        "checkpoint authorization receipt binds different checkpoint bytes",
    )
    return dict(receipt), checkpoint_provenance_load_receipt


def validate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an immutable evaluation bundle."""

    _require(isinstance(bundle, Mapping), "bundle must be a mapping")
    required = {
        "schema_version",
        "case_id",
        "case_kind",
        "provenance",
        "evaluation_config",
        "estimator_metadata",
        "transform",
        "evaluation",
        "operator",
        "bundle_content_sha256",
    }
    _require_exact_keys(
        bundle,
        required=required,
        optional={"fit"},
        name="bundle",
    )
    _require(bundle["schema_version"] == BUNDLE_SCHEMA_VERSION, "unsupported bundle schema")
    _require(isinstance(bundle["case_id"], str) and bundle["case_id"], "case_id is empty")
    _require(bundle["case_kind"] in {"planted", "checkpoint"}, "invalid case_kind")
    claimed_bundle_hash = bundle["bundle_content_sha256"]
    _require(_is_sha256(claimed_bundle_hash), "bundle_content_sha256 is invalid")
    bundle_payload = dict(bundle)
    bundle_payload.pop("bundle_content_sha256")
    _require(
        canonical_sha256(bundle_payload) == claimed_bundle_hash,
        "bundle content hash mismatch",
    )
    checkpoint_authorization = None
    checkpoint_provenance_load_receipt = None
    if bundle["case_kind"] == "checkpoint":
        (
            checkpoint_authorization,
            checkpoint_provenance_load_receipt,
        ) = _require_checkpoint_evaluator_authorization(
            bundle,
        )

    operator_hint = bundle["operator"]
    _require(isinstance(operator_hint, Mapping), "operator must be a mapping")
    operator_mode_hint = operator_hint.get("mode")
    _require(
        operator_mode_hint in {"supplied", "fit_from_pairs"},
        "operator mode must be supplied or fit_from_pairs",
    )
    provenance = bundle["provenance"]
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    try:
        provenance_kwargs: dict[str, Any] = {
            "case_kind": str(bundle["case_kind"]),
            "fit_from_pairs": operator_mode_hint == "fit_from_pairs",
        }
        if checkpoint_provenance_load_receipt is not None:
            provenance_kwargs["checkpoint_provenance_load_receipt"] = (
                checkpoint_provenance_load_receipt
            )
        validated_provenance = _validate_production_provenance(
            provenance,
            **provenance_kwargs,
        )
    except EvaluationContractError:
        raise
    except provenance_contract.ProvenanceContractError as exc:
        raise EvaluationContractError(
            f"provenance validation failed: {exc}",
            code="provenance_invalid",
        ) from exc
    except Exception as exc:
        raise EvaluationContractError(
            f"provenance validator failed closed: {exc}",
            code="provenance_invalid",
        ) from exc
    _require(
        isinstance(validated_provenance, Mapping)
        and validated_provenance.get("source_commit_verified") is True,
        "provenance validator did not return a verified source commit",
    )

    estimator = bundle["estimator_metadata"]
    _require(isinstance(estimator, Mapping), "estimator_metadata must be a mapping")
    _validate_estimator(estimator)

    transform = bundle["transform"]
    _require(isinstance(transform, Mapping), "transform must be a mapping")
    family_hint = transform.get("family")
    _require(family_hint in SUPPORTED_FAMILIES, f"unknown transform family {family_hint!r}")
    config = bundle["evaluation_config"]
    _require(isinstance(config, Mapping), "evaluation_config must be a mapping")
    _validate_evaluation_config(config, family=str(family_hint))
    numeric_registry = _validate_numeric_registry(
        estimator=estimator,
        validated_provenance=validated_provenance,
    )
    family, target_A, target_b = _validate_transform(
        transform,
        tolerance=numeric_registry["coordinate_tolerance"],
    )

    checkpoint_binding = None
    if bundle["case_kind"] == "checkpoint":
        external_files = validated_provenance.get("external_files")
        _require(
            isinstance(external_files, list),
            "validated checkpoint provenance lacks external_files",
        )
        checkpoint_records = [
            record
            for record in external_files
            if isinstance(record, Mapping)
            and record.get("role") == "checkpoint"
        ]
        _require(
            len(checkpoint_records) == 1,
            "validated provenance does not uniquely bind checkpoint",
        )
        checkpoint_record = checkpoint_records[0]
        checkpoint_binding = {
            "absolute_path": checkpoint_record["absolute_path"],
            "sha256": checkpoint_record["file_sha256"],
            "size_bytes": checkpoint_record["size_bytes"],
        }

    evaluation_record = bundle["evaluation"]
    _require(
        isinstance(evaluation_record, Mapping),
        "evaluation must be a mapping",
    )
    evaluation_required = {
        "object_id",
        "seed",
        "partition",
        "frame_ids",
        "points",
        "physical_states",
        "bbox_diagonal_image01",
        "visibility",
        "masks",
        "pair_rows",
    }
    _require_exact_keys(
        evaluation_record,
        required=evaluation_required,
        optional={"logits"},
        name="evaluation",
    )
    _require(
        isinstance(evaluation_record["object_id"], str)
        and bool(evaluation_record["object_id"]),
        "evaluation object_id is empty",
    )
    _require(
        isinstance(evaluation_record["seed"], int)
        and not isinstance(evaluation_record["seed"], bool),
        "evaluation seed must be an integer",
    )
    _require(
        evaluation_record["partition"] in {"validation", "test", "full_corpus"},
        "evaluation partition must be validation, test, or full_corpus",
    )
    evaluation_partition = str(evaluation_record["partition"])
    evaluation_frame_ids = _validate_frame_ids(
        evaluation_record["frame_ids"],
        name="evaluation frame_ids",
    )
    evaluation_frame_index_by_id = {
        frame_id: index
        for index, frame_id in enumerate(evaluation_frame_ids)
    }
    evaluation_points = _finite_array(
        evaluation_record["points"],
        name="evaluation points",
        shape=(len(evaluation_frame_ids), None, 2),
    )
    _require(
        evaluation_points.shape[1] >= 2,
        "at least two evaluation channels are required",
    )
    _require(
        bool(
            np.all(
                (evaluation_points >= -1.0)
                & (evaluation_points <= 1.0)
            )
        ),
        "evaluation points lie outside closed normalized coordinate square [-1,1]^2",
    )
    evaluation_bbox = _finite_array(
        evaluation_record["bbox_diagonal_image01"],
        name="evaluation bbox_diagonal_image01",
        shape=(len(evaluation_frame_ids),),
    )
    _require(
        bool(np.all(evaluation_bbox > 0.0)),
        "evaluation bbox diagonals must be positive",
    )
    evaluation_visibility = _strict_bool_array(
        evaluation_record["visibility"],
        name="evaluation visibility",
        shape=evaluation_points.shape[:2],
    )
    evaluation_pairs = _validate_pair_rows(
        evaluation_record["pair_rows"],
        frame_ids=evaluation_frame_ids,
        required_partition=evaluation_partition,
        transform=transform,
        name="evaluation pair_rows",
    )
    _validate_primary_roll_protocol(
        frame_ids=evaluation_frame_ids,
        transform=transform,
        config=config,
        pair_rows=evaluation_pairs,
    )

    masks_record = evaluation_record["masks"]
    _require(
        isinstance(masks_record, Mapping),
        "evaluation masks must be a mapping",
    )
    _require_exact_keys(
        masks_record,
        required={"values", "geometry"},
        name="evaluation masks",
    )
    evaluation_masks = _strict_bool_array(
        masks_record["values"],
        name="evaluation masks values",
        shape=(
            len(evaluation_frame_ids),
            estimator["input_height"],
            estimator["input_width"],
        ),
    )
    geometry = masks_record["geometry"]
    _require(isinstance(geometry, Mapping), "mask geometry must be a mapping")
    geometry_fields = {"input_height", "input_width", "crop", "resize", "align_corners"}
    _require_exact_keys(geometry, required=geometry_fields, name="mask geometry")
    for key in geometry_fields:
        _require(geometry[key] == estimator[key], f"mask/image geometry differs at {key}")
    recomputed_bbox = _bbox_diagonals_from_masks(evaluation_masks)
    maximum_bbox_error = float(
        np.max(np.abs(recomputed_bbox - evaluation_bbox))
    )
    _require(
        maximum_bbox_error <= numeric_registry["coordinate_tolerance"],
        f"supplied bbox diagonal disagrees with mask endpoint grid: maximum error {maximum_bbox_error}",
    )
    evaluation_bbox = recomputed_bbox

    evaluation_logits = None
    evaluation_estimator_consistency: dict[str, Any] = {
        "status": "void_missing_logits",
        "maximum_absolute_coordinate_error": None,
    }
    if "logits" in evaluation_record:
        expected_shape = (
            len(evaluation_frame_ids),
            evaluation_points.shape[1],
            estimator["heatmap_height"],
            estimator["heatmap_width"],
        )
        evaluation_logits, derived = _validated_logits_and_points(
            evaluation_record["logits"],
            name="evaluation logits",
            expected_shape=expected_shape,
            estimator=estimator,
        )
        tolerance = numeric_registry["coordinate_tolerance"]
        maximum_error = float(
            np.max(np.abs(derived - evaluation_points))
        )
        _require(
            maximum_error <= tolerance,
            f"evaluation points disagree with logits: maximum error {maximum_error}",
        )
        evaluation_points = derived
        evaluation_estimator_consistency = {
            "status": "verified_and_points_recomputed_from_logits",
            "maximum_absolute_coordinate_error": maximum_error,
        }
    elif bundle["case_kind"] == "checkpoint":
        raise EvaluationContractError(
            "checkpoint case requires logits for heatmap evidence",
            code="checkpoint_evaluation_logits_required",
        )

    fit: dict[str, Any] | None = None
    fit_estimator_consistency: dict[str, Any] = {
        "status": "not_applicable_no_fit_section",
        "maximum_absolute_coordinate_error": None,
    }
    if "fit" in bundle:
        fit_record = bundle["fit"]
        _require(isinstance(fit_record, Mapping), "fit must be a mapping")
        _require_exact_keys(
            fit_record,
            required={
                "object_id",
                "seed",
                "partition",
                "frame_ids",
                "points",
                "visibility",
                "pair_rows",
            },
            optional={"logits"},
            name="fit",
        )
        _require(
            isinstance(fit_record["object_id"], str)
            and bool(fit_record["object_id"]),
            "fit object_id is empty",
        )
        _require(
            isinstance(fit_record["seed"], int)
            and not isinstance(fit_record["seed"], bool),
            "fit seed must be an integer",
        )
        _require(
            fit_record["object_id"] == evaluation_record["object_id"],
            "fit and evaluation object_id differ",
        )
        _require(
            fit_record["seed"] == evaluation_record["seed"],
            "fit and evaluation seed differ",
        )
        _require(
            fit_record["partition"] == "train",
            "fit partition must be train",
        )
        fit_frame_ids = _validate_frame_ids(
            fit_record["frame_ids"],
            name="fit frame_ids",
        )
        fit_frame_index_by_id = {
            frame_id: index
            for index, frame_id in enumerate(fit_frame_ids)
        }
        fit_points = _finite_array(
            fit_record["points"],
            name="fit points",
            shape=(
                len(fit_frame_ids),
                evaluation_points.shape[1],
                2,
            ),
        )
        _require(
            bool(np.all((fit_points >= -1.0) & (fit_points <= 1.0))),
            "fit points lie outside closed normalized coordinate square [-1,1]^2",
        )
        fit_logits = None
        if "logits" in fit_record:
            expected_fit_logit_shape = (
                len(fit_frame_ids),
                evaluation_points.shape[1],
                estimator["heatmap_height"],
                estimator["heatmap_width"],
            )
            fit_logits, derived_fit_points = _validated_logits_and_points(
                fit_record["logits"],
                name="fit logits",
                expected_shape=expected_fit_logit_shape,
                estimator=estimator,
            )
            fit_maximum_error = float(
                np.max(np.abs(derived_fit_points - fit_points))
            )
            _require(
                fit_maximum_error <= numeric_registry["coordinate_tolerance"],
                "fit points disagree with logits: "
                f"maximum error {fit_maximum_error}",
            )
            fit_points = derived_fit_points
            fit_estimator_consistency = {
                "status": "verified_and_points_recomputed_from_logits",
                "maximum_absolute_coordinate_error": fit_maximum_error,
            }
        elif bundle["case_kind"] == "checkpoint":
            raise EvaluationContractError(
                "checkpoint fit section requires logits",
                code="checkpoint_fit_logits_required",
            )
        else:
            fit_estimator_consistency = {
                "status": "void_missing_logits_for_planted_fit",
                "maximum_absolute_coordinate_error": None,
            }
        fit_visibility = _strict_bool_array(
            fit_record["visibility"],
            name="fit visibility",
            shape=fit_points.shape[:2],
        )
        fit_pairs = _validate_pair_rows(
            fit_record["pair_rows"],
            frame_ids=fit_frame_ids,
            required_partition="train",
            transform=transform,
            name="fit pair_rows",
        )
        fit_endpoints = {
            endpoint
            for row in fit_pairs
            for endpoint in (row["source_frame"], row["target_frame"])
        }
        evaluation_endpoints = {
            endpoint
            for row in evaluation_pairs
            for endpoint in (row["source_frame"], row["target_frame"])
        }
        _require(
            fit_endpoints.isdisjoint(evaluation_endpoints),
            "training and evaluation endpoints overlap",
        )
        evaluation_pair_ids = {
            row["pair_id"] for row in evaluation_pairs
        }
        fit_pair_ids = {row["pair_id"] for row in fit_pairs}
        _require(
            evaluation_pair_ids.isdisjoint(fit_pair_ids),
            "training and evaluation pair_id values overlap",
        )
        fit = {
            "frame_ids": fit_frame_ids,
            "frame_index_by_id": fit_frame_index_by_id,
            "points": fit_points,
            "logits": fit_logits,
            "visibility": fit_visibility,
            "pair_rows": fit_pairs,
        }

    operator_record = bundle["operator"]
    _require(isinstance(operator_record, Mapping), "operator must be a mapping")
    if config["protocol"] == "full_primary_roll":
        _require(
            operator_record.get("mode") == "supplied" and fit is None,
            "full_primary_roll forbids fit_from_pairs and a fit section",
        )
    if family in {"yaw", "pitch"}:
        _require(
            operator_record.get("mode") == "fit_from_pairs",
            f"{family} requires fit_from_pairs and a separate train fit section",
        )
    A, b, operator_mode = _resolve_operator(
        operator_record,
        fit=fit,
    )
    if family in {"yaw", "pitch"}:
        target_A = A.copy()
        target_b = b.copy()
    else:
        _require(
            target_A is not None and target_b is not None,
            "exact family target geometry was not derived internally",
        )

    physical_states, state_A, state_b = _validate_and_derive_physical_states(
        family=family,
        physical_states=evaluation_record["physical_states"],
        frame_ids=evaluation_frame_ids,
        frame_index_by_id=evaluation_frame_index_by_id,
        transform=transform,
        pair_rows=evaluation_pairs,
        target_A=target_A,
        target_b=target_b,
        tolerance=numeric_registry["coordinate_tolerance"],
    )

    return {
        "family": family,
        "target_A": target_A,
        "target_b": target_b,
        "evaluation_partition": evaluation_partition,
        "evaluation_pair_index": evaluation_pairs,
        "fit_pair_index": fit["pair_rows"] if fit is not None else [],
        "evaluation_frame_ids": evaluation_frame_ids,
        "evaluation_frame_index_by_id": evaluation_frame_index_by_id,
        "evaluation_points": evaluation_points,
        "evaluation_physical_states": physical_states,
        "evaluation_state_A": state_A,
        "evaluation_state_b": state_b,
        "evaluation_bbox": evaluation_bbox,
        "evaluation_visibility": evaluation_visibility,
        "A": A,
        "b": b,
        "evaluation_masks": evaluation_masks,
        "evaluation_logits": evaluation_logits,
        "operator_mode": operator_mode,
        "bundle_content_sha256": claimed_bundle_hash,
        "validated_provenance": dict(validated_provenance),
        "numeric_registry": numeric_registry,
        "checkpoint": checkpoint_binding,
        "checkpoint_authorization": checkpoint_authorization,
        "estimator_consistency": {
            "definition": (
                "supplied_points_are_checked_against_and_replaced_by_fixed_"
                "spatial_expectation_of_the_bound_logits"
            ),
            "coordinate_tolerance_key": numeric_registry[
                "coordinate_tolerance_key"
            ],
            "coordinate_tolerance": numeric_registry[
                "coordinate_tolerance"
            ],
            "evaluation": evaluation_estimator_consistency,
            "fit": fit_estimator_consistency,
        },
        "bbox_definition": (
            "boolean_mask_bounding_box_diagonal_on_endpoint_aligned_[0,1]_sample_centres"
        ),
    }


def evaluate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one strict stratum and return canonical-JSON-safe metrics."""

    normalized = validate_bundle(bundle)
    config = bundle["evaluation_config"]
    thresholds = config["representation_thresholds"]
    required_thresholds = {
        "close_distance_objdiag",
        "persistent_fraction",
        "recurrent_fraction",
        "transient_longest_fraction",
        "clustered_median_objdiag",
    }
    _require(isinstance(thresholds, Mapping), "representation_thresholds must be a mapping")
    _require_exact_keys(
        thresholds,
        required=required_thresholds,
        name="representation_thresholds",
    )
    _require(
        float(thresholds["close_distance_objdiag"]) == 0.06
        and float(thresholds["persistent_fraction"]) == 0.50
        and float(thresholds["recurrent_fraction"]) == 0.10
        and float(thresholds["transient_longest_fraction"]) == 0.10
        and float(thresholds["clustered_median_objdiag"]) == 0.12,
        "descriptive close-pair definitions differ from the frozen specification",
    )

    evaluation_points = normalized["evaluation_points"]
    evaluation_masks = normalized["evaluation_masks"]
    evaluation_visibility = normalized["evaluation_visibility"]
    evaluation_bbox = normalized["evaluation_bbox"]
    evaluation_logits = normalized["evaluation_logits"]
    evaluation_frame_ids = normalized["evaluation_frame_ids"]
    evaluation_frame_index_by_id = normalized[
        "evaluation_frame_index_by_id"
    ]
    evaluation_physical_states = normalized[
        "evaluation_physical_states"
    ]
    evaluation_state_A = normalized["evaluation_state_A"]
    evaluation_state_b = normalized["evaluation_state_b"]

    health = _channel_health(
        evaluation_points,
        evaluation_masks,
        evaluation_logits,
        evaluation_visibility,
        normalized["evaluation_pair_index"],
        evaluation_frame_index_by_id,
        bundle["estimator_metadata"],
        config,
    )
    all_indices = list(range(evaluation_points.shape[1]))
    all_channel = _pairwise_representation(
        evaluation_points,
        evaluation_bbox,
        evaluation_visibility,
        evaluation_frame_ids,
        all_indices,
        cyclic=bool(bundle["transform"]["cyclic"]),
        thresholds=thresholds,
    )
    _require(all_channel is not None, "all-channel representation metric is void")
    minimum_eligible = int(config["minimum_eligible_channels"])
    _require(minimum_eligible >= 2, "minimum_eligible_channels must be at least two")
    eligible_indices = health["eligible_indices"]
    eligible_metric = (
        _pairwise_representation(
            evaluation_points,
            evaluation_bbox,
            evaluation_visibility,
            evaluation_frame_ids,
            eligible_indices,
            cyclic=bool(bundle["transform"]["cyclic"]),
            thresholds=thresholds,
        )
        if len(eligible_indices) >= minimum_eligible
        else None
    )
    flat_dead_indices = [
        int(row["channel"])
        for row in health["channels"]
        if row["heatmap_flat_dead"] is True
    ]
    channels_not_confirmed_flat_dead_indices = [
        int(row["channel"])
        for row in health["channels"]
        if row["heatmap_flat_dead"] is not True
    ]
    channels_not_confirmed_flat_dead_metric = (
        _pairwise_representation(
            evaluation_points,
            evaluation_bbox,
            evaluation_visibility,
            evaluation_frame_ids,
            channels_not_confirmed_flat_dead_indices,
            cyclic=bool(bundle["transform"]["cyclic"]),
            thresholds=thresholds,
        )
        if len(channels_not_confirmed_flat_dead_indices) >= minimum_eligible
        else None
    )

    operator = _operator_metrics(
        normalized["family"],
        normalized["A"],
        normalized["b"],
        normalized["target_A"],
        normalized["target_b"],
        bundle["transform"],
        config,
    )
    critical_failures = []
    if operator.get("improper_or_reflection"):
        critical_failures.append("operator_is_improper_or_reflection")
    if operator.get("wrong_locked_sign"):
        critical_failures.append("operator_has_wrong_locked_sign")

    if normalized["family"] in {"yaw", "pitch"}:
        canonical_or_residual = {
            "canonical_drift": {
                "status": "not_applicable",
                "reason": "yaw_pitch_are_not_exact_planar_canonical_transforms",
            },
            "fitted_reference_residual": _pair_fit_residuals(
                evaluation_points,
                normalized["target_A"],
                normalized["target_b"],
                evaluation_visibility,
                evaluation_physical_states,
                bundle["transform"],
                normalized["evaluation_pair_index"],
                evaluation_frame_index_by_id,
            ),
        }
    else:
        _require(
            evaluation_state_A is not None and evaluation_state_b is not None,
            "exact planar family lacks derived physical-state transforms",
        )
        canonical_or_residual = {
            "canonical_drift": _canonical_drift(
                evaluation_points,
                evaluation_state_A,
                evaluation_state_b,
                evaluation_bbox,
                evaluation_visibility,
            ),
            "fitted_reference_residual": {
                "status": "not_applicable_exact_planar_family"
            },
        }

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": bundle["case_id"],
        "case_kind": bundle["case_kind"],
        "stratum": {
            "object_id": bundle["evaluation"]["object_id"],
            "seed": bundle["evaluation"]["seed"],
            "partition": bundle["evaluation"]["partition"],
            "transform_family": normalized["family"],
            "direction": bundle["transform"]["direction"],
            "stride": bundle["transform"]["stride"],
        },
        "provenance": normalized["validated_provenance"],
        "validated_numeric_registry": normalized["numeric_registry"],
        "validated_checkpoint": normalized["checkpoint"],
        "bundle_content_sha256": normalized["bundle_content_sha256"],
        "evaluation_config": bundle["evaluation_config"],
        "evaluation_config_sha256": canonical_sha256(
            bundle["evaluation_config"]
        ),
        "estimator_metadata": bundle["estimator_metadata"],
        "estimator_consistency": normalized["estimator_consistency"],
        "transform": bundle["transform"],
        "evaluation_pair_index": normalized["evaluation_pair_index"],
        "operator_fit_pair_index": normalized["fit_pair_index"],
        "evaluation_frame_ids": evaluation_frame_ids,
        "evaluation_physical_states": evaluation_physical_states,
        "operator_input_mode": normalized["operator_mode"],
        "operator": operator,
        **canonical_or_residual,
        "channel_health": health,
        "trajectory_separation": {
            "total_channel_count": evaluation_points.shape[1],
            "eligible_channel_count": len(eligible_indices),
            "minimum_eligible_channel_count": minimum_eligible,
            "eligible_metric_void": eligible_metric is None,
            "all_channels": all_channel,
            "eligible_active_on_object": eligible_metric,
            "channels_not_confirmed_flat_dead_count": len(
                channels_not_confirmed_flat_dead_indices
            ),
            "channels_not_confirmed_flat_dead_indices": (
                channels_not_confirmed_flat_dead_indices
            ),
            "confirmed_flat_dead_channel_indices": flat_dead_indices,
            "channels_not_confirmed_flat_dead_metric_void": (
                channels_not_confirmed_flat_dead_metric is None
            ),
            "channels_not_confirmed_flat_dead": (
                channels_not_confirmed_flat_dead_metric
            ),
        },
        "switching": _switching(
            evaluation_points,
            normalized["target_A"],
            normalized["target_b"],
            (
                "fitted_reference"
                if normalized["family"] in {"yaw", "pitch"}
                else "locked_exact_pair_transform"
            ),
            evaluation_logits,
            evaluation_visibility,
            normalized["evaluation_pair_index"],
            evaluation_frame_index_by_id,
        ),
        "evidence_availability": {
            "logits": {
                "available": evaluation_logits is not None,
                "required_for_case_kind": bundle["case_kind"] == "checkpoint",
                "missing_status": (
                    None
                    if evaluation_logits is not None
                    else "void_for_planted_coordinate_only_case"
                ),
            },
            "bbox_diagonal": {
                "status": "recomputed_and_verified",
                "definition": normalized["bbox_definition"],
            },
        },
        "critical_failures": critical_failures,
        "scientific_threshold_status": "not_defined_by_evaluator",
        "statistical_scope": {
            "frame_channel_horizon_values": "correlated_descriptive",
            "replication_units": ["seed", "object"],
            "sem_computed": False,
        },
    }
    if normalized["checkpoint_authorization"] is not None:
        result["checkpoint_authorization"] = normalized[
            "checkpoint_authorization"
        ]
    if (
        normalized["family"] == "roll"
        and config["protocol"] == "full_primary_roll"
    ):
        result["rollout"] = _rollout_metrics(
            evaluation_points,
            evaluation_frame_ids,
            normalized["A"],
            normalized["b"],
            bundle["transform"],
            config,
        )
    elif normalized["family"] == "roll":
        result["rollout"] = {
            "status": "not_full_primary_roll",
            "selection_status": (
                "generic_roll_forbids_full_corpus_holdout_and_closure_metrics"
            ),
        }
    else:
        result["rollout"] = {
            "status": "not_a_roll_metric",
            "selection_status": "rotation_auc_forbidden_for_this_family",
        }

    result["collapse_evidence"] = {
        "structural_negative_control_collapse": bool(
            all_channel["all_pairs_persistent_duplicate"]
        ),
        "structural_negative_control_definition": (
            "every_all_channel_pair_is_evaluable_and_has_the_frozen_"
            "persistent_duplicate_category"
        ),
        "structural_negative_control_is_coincidence_trigger": False,
        "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead": (
            bool(
                channels_not_confirmed_flat_dead_metric[
                    "all_pairs_persistent_duplicate"
                ]
            )
            if channels_not_confirmed_flat_dead_metric is not None
            else None
        ),
        "structural_negative_control_definition_v2_excluding_confirmed_flat_dead": (
            "after_excluding_only_channels_with_heatmap_flat_dead_true_"
            "at_least_minimum_eligible_channels_remain_and_every_possible_"
            "retained_pair_is_evaluable_and_has_the_frozen_persistent_"
            "duplicate_category"
        ),
        "structural_negative_control_status_v2_excluding_confirmed_flat_dead": (
            "available"
            if channels_not_confirmed_flat_dead_metric is not None
            else "void_below_minimum_channels_not_confirmed_flat_dead"
        ),
        "channels_not_confirmed_flat_dead_indices": (
            channels_not_confirmed_flat_dead_indices
        ),
        "confirmed_flat_dead_channel_indices": flat_dead_indices,
        "channels_not_confirmed_flat_dead_count": len(
            channels_not_confirmed_flat_dead_indices
        ),
        "minimum_channels_not_confirmed_flat_dead": minimum_eligible,
        "all_pairs_persistent_duplicate": all_channel["all_pairs_persistent_duplicate"],
        "persistent_duplicate_pair_rate": all_channel["persistent_duplicate_pair_rate"],
        "trajectory_median_nearest_neighbour_objdiag": all_channel[
            "trajectory_median_nearest_neighbour_objdiag"
        ],
        "eligible_only_structural_negative_control_collapse": (
            bool(eligible_metric["all_pairs_persistent_duplicate"])
            if eligible_metric is not None
            else None
        ),
        "eligible_only_status": (
            "additional_evidence"
            if eligible_metric is not None
            else "void_below_minimum_eligible_channels"
        ),
        "interpretation": (
            "structural_negative_control_flag_only_not_the_later_"
            "coincidence_trigger_threshold"
        ),
    }
    result["result_content_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = _load_strict_json(args.bundle, name="evaluation bundle")
    _require(
        isinstance(bundle, Mapping),
        "evaluation bundle top level must be a mapping",
    )
    result = evaluate_bundle(bundle)
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
