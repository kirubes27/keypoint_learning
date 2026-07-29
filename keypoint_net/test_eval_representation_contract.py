"""Mutation tests for the strict representation-evaluator contract.

These tests are intentionally pure CPU fixtures.  They must not be executed
until the numeric calibration amendment named by the project specification is
committed.
"""

from __future__ import annotations

import copy
import contextlib
import functools
import hashlib
import inspect
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from keypoint_net import eval_representation as evaluator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_NUMERIC_REGISTRY_PATH = (
    REPOSITORY_ROOT
    / "docs/decisions/2026-07-26/representation_oracle_calibration/"
    "NUMERIC_CALIBRATION_v1_1.json"
).resolve()
AUTHORITATIVE_NUMERIC_REGISTRY_FILE_SHA256 = (
    "40115c7325858137595ba6e90e36fbd52c9a72da2f16b64d9fd1ad4f7764f420"
)
AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256 = (
    "24a2d0eb28d1ec8f26c14064abdb4afcaa51e7cf63d9560e1c2f1c9b4a8118c8"
)


def _unit_provenance_validator(
    record: dict,
    *,
    case_kind: str,
    fit_from_pairs: bool,
) -> dict:
    """Isolated seam for evaluator-unit tests.

    Production provenance is tested independently against a temporary Git
    repository.  These evaluator tests use disposable files that cannot exist
    in the production source commit, so this seam rehashes those files while
    returning the normalized shape expected from the strict validator.
    """

    del fit_from_pairs
    expected = {
        "source_commit",
        "evaluator_config_sha256",
        "files",
        "numeric_registry",
    }
    if case_kind == "checkpoint":
        expected.add("checkpoint")
    missing = expected - set(record)
    extra = set(record) - expected
    if missing or extra:
        raise evaluator.EvaluationContractError(
            f"unit provenance keys differ: missing={sorted(missing)}, extra={sorted(extra)}",
            code="provenance_invalid",
        )
    if record["evaluator_config_sha256"] != evaluator.canonical_sha256(
        _UNIT_CURRENT_BUNDLE_CONFIG
    ):
        raise evaluator.EvaluationContractError(
            "evaluator config hash mismatch",
            code="provenance_invalid",
        )
    files = record["files"]
    if not isinstance(files, list) or not files:
        raise evaluator.EvaluationContractError(
            "provenance files are empty",
            code="provenance_invalid",
        )
    normalized_files = []
    roles = []
    paths = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {
            "role",
            "absolute_path",
            "sha256",
        }:
            raise evaluator.EvaluationContractError(
                f"provenance file {index} has invalid fields",
                code="provenance_invalid",
            )
        path = Path(item["absolute_path"])
        if not path.is_absolute() or not path.is_file():
            raise evaluator.EvaluationContractError(
                f"provenance file {index} does not resolve to a file",
                code="provenance_invalid",
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise evaluator.EvaluationContractError(
                f"provenance file {index} sha256 mismatch",
                code="provenance_invalid",
            )
        roles.append(item["role"])
        paths.append(str(path.resolve()))
        normalized_files.append(
            {
                "role": item["role"],
                "absolute_path": str(path.resolve()),
                "file_sha256": actual,
            }
        )
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise evaluator.EvaluationContractError(
            "duplicate provenance file role or path",
            code="provenance_invalid",
        )
    numeric = record["numeric_registry"]
    if not isinstance(numeric, dict) or set(numeric) != {
        "absolute_path",
        "file_sha256",
        "content_sha256",
    }:
        raise evaluator.EvaluationContractError(
            "numeric_registry binding has invalid fields",
            code="provenance_invalid",
        )
    numeric_files = [
        item for item in normalized_files if item["role"] == "numeric_registry"
    ]
    if len(numeric_files) != 1:
        raise evaluator.EvaluationContractError(
            "provenance files lack numeric_registry role",
            code="provenance_invalid",
        )
    numeric_file = numeric_files[0]
    committed_files = [
        {
            **numeric_file,
            "repo_relative_path": (
                "docs/decisions/2026-07-26/"
                "representation_oracle_calibration/"
                "NUMERIC_CALIBRATION_v1_1.json"
            ),
            "git_blob_oid": "unit-test-only",
        }
    ]
    committed_files[0]["absolute_path"] = numeric["absolute_path"]
    committed_files[0]["file_sha256"] = numeric["file_sha256"]
    external_files = []
    if case_kind == "checkpoint":
        checkpoint = record["checkpoint"]
        checkpoint_path = Path(checkpoint["absolute_path"])
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        external_files.append(
            {
                "role": "checkpoint",
                "absolute_path": str(checkpoint_path.resolve()),
                "file_sha256": checkpoint_hash,
                "size_bytes": checkpoint_path.stat().st_size,
            }
        )
    return {
        "source_commit_verified": True,
        "source_commit": "0" * 40,
        "committed_files": committed_files,
        "external_files": external_files,
    }


_UNIT_CURRENT_BUNDLE_CONFIG: dict = {}


@contextlib.contextmanager
def _assert_raises_regex(
    expected_exception: type[BaseException],
    match: str,
):
    try:
        yield
    except expected_exception as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(
                f"{str(exc)!r} does not match {match!r}"
            ) from exc
    else:
        raise AssertionError(
            f"{expected_exception.__name__} matching {match!r} was not raised"
        )


def _temporary_path(function):
    @functools.wraps(function)
    def wrapped(*_args, **_kwargs):
        global _UNIT_CURRENT_BUNDLE_CONFIG
        directory = Path(
            tempfile.mkdtemp(prefix="representation-evaluator-contract-")
        )
        with mock.patch.object(
            evaluator,
            "_validate_production_provenance",
            _unit_provenance_validator,
        ):
            _UNIT_CURRENT_BUNDLE_CONFIG = {}
            function(directory)

    return wrapped


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authoritative_numeric_registry_binding() -> dict:
    return {
        "absolute_path": str(AUTHORITATIVE_NUMERIC_REGISTRY_PATH),
        "file_sha256": AUTHORITATIVE_NUMERIC_REGISTRY_FILE_SHA256,
        "content_sha256": AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256,
    }


def _file_record(role: str, path: Path) -> dict:
    return {
        "role": role,
        "absolute_path": str(path),
        "sha256": _sha256(path),
    }


def _seal(bundle: dict) -> dict:
    global _UNIT_CURRENT_BUNDLE_CONFIG
    bundle = copy.deepcopy(bundle)
    bundle.pop("bundle_content_sha256", None)
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    _UNIT_CURRENT_BUNDLE_CONFIG = copy.deepcopy(bundle["evaluation_config"])
    return bundle


def _with_evaluation_logits(
    bundle: dict,
    per_channel_logits: list[list[list[float]]],
) -> dict:
    bundle = copy.deepcopy(bundle)
    channel_logits = np.asarray(per_channel_logits, dtype=np.float64)
    frame_count = len(bundle["evaluation"]["frame_ids"])
    logits = np.repeat(channel_logits[None, :, :, :], frame_count, axis=0)
    points = evaluator.spatial_expectation(
        logits,
        temperature=float(bundle["estimator_metadata"]["temperature"]),
        softmax_dtype=str(bundle["estimator_metadata"]["softmax_dtype"]),
    )
    bundle["evaluation"]["logits"] = logits.tolist()
    bundle["evaluation"]["points"] = points.tolist()
    bundle["evaluation"]["visibility"] = [
        [True] * channel_logits.shape[0]
        for _ in range(frame_count)
    ]
    return _seal(bundle)


def _base_bundle(tmp_path: Path, *, case_kind: str = "planted") -> dict:
    registry_path = AUTHORITATIVE_NUMERIC_REGISTRY_PATH
    registry = _authoritative_numeric_registry_binding()
    input_path = tmp_path / "fixture_input.json"
    input_path.write_text("{}\n", encoding="utf-8")
    config = {
        "protocol": "generic",
        "representation_thresholds": {
            "close_distance_objdiag": 0.06,
            "persistent_fraction": 0.50,
            "recurrent_fraction": 0.10,
            "transient_longest_fraction": 0.10,
            "clustered_median_objdiag": 0.12,
        },
        "motion_reference_magnitude_image01": 0.1,
        "motion_fraction_min": 0.01,
        "on_object_rate_min": 0.5,
        "minimum_eligible_channels": 2,
        "operator_composition_horizons": [],
    }
    points = [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]
    bundle = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": "contract-fixture",
        "case_kind": case_kind,
        "provenance": {
            "source_commit": "deadbee",
            "evaluator_config_sha256": evaluator.canonical_sha256(config),
            "files": [
                _file_record("numeric_registry", registry_path),
                _file_record("fixture_input", input_path),
            ],
            "numeric_registry": registry,
        },
        "evaluation_config": config,
        "estimator_metadata": {
            "input_height": 4,
            "input_width": 4,
            "heatmap_height": 2,
            "heatmap_width": 2,
            "endpoint_grid": True,
            "temperature": 1.0,
            "logit_dtype": "float64",
            "softmax_dtype": "float64",
            "crop": None,
            "resize": [4, 4],
            "align_corners": None,
        },
        "transform": {
            "family": "translation",
            "physical_axis": "image_x",
            "direction": "forward",
            "signed_generator": 0.1,
            "generator_units": "normalized_image",
            "stride": 1,
            "stride_units": "frames",
            "cyclic": False,
            "expected_2d_family": "synthetic_image_plane_translation",
            "expected_image_component": 0,
            "expected_image_sign": 1,
            "translation_source": "synthetic_image_fixture",
        },
        "evaluation": {
            "object_id": "synthetic",
            "seed": 0,
            "partition": "validation",
            "frame_ids": [100, 101],
            "points": [points, points],
            "physical_states": [
                {"image_offset_xy": [0.0, 0.0]},
                {"image_offset_xy": [0.1, 0.0]},
            ],
            "bbox_diagonal_image01": [math.sqrt(2.0), math.sqrt(2.0)],
            "visibility": [[True, True, True], [True, True, True]],
            "pair_rows": [
                {
                    "pair_id": "fit:100:101",
                    "source_frame": 100,
                    "target_frame": 101,
                    "direction": "forward",
                    "stride": 1,
                    "partition": "validation",
                }
            ],
            "masks": {
                "values": [
                    [[True] * 4 for _ in range(4)],
                    [[True] * 4 for _ in range(4)],
                ],
                "geometry": {
                    "input_height": 4,
                    "input_width": 4,
                    "crop": None,
                    "resize": [4, 4],
                    "align_corners": None,
                },
            },
        },
        "operator": {
            "mode": "supplied",
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "b": [0.1, 0.0],
        },
    }
    if case_kind == "checkpoint":
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.write_bytes(b"synthetic checkpoint bytes")
        bundle["provenance"]["checkpoint"] = {
            "absolute_path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        }
        bundle["provenance"]["files"].append(
            _file_record("checkpoint", checkpoint_path)
        )
    return _seal(bundle)


def _reseal_after_config_mutation(bundle: dict) -> dict:
    bundle["provenance"]["evaluator_config_sha256"] = evaluator.canonical_sha256(
        bundle["evaluation_config"]
    )
    return _seal(bundle)


def _add_translation_fit(
    bundle: dict,
    *,
    displacement: float = 0.1,
) -> dict:
    source = np.asarray(
        bundle["evaluation"]["points"][0],
        dtype=np.float64,
    )
    target = source + np.asarray([displacement, 0.0])
    bundle["fit"] = {
        "object_id": bundle["evaluation"]["object_id"],
        "seed": bundle["evaluation"]["seed"],
        "partition": "train",
        "frame_ids": [0, 1],
        "points": [source.tolist(), target.tolist()],
        "visibility": [[True] * source.shape[0] for _ in range(2)],
        "pair_rows": [
            {
                "pair_id": "evaluation:0:1",
                "source_frame": 0,
                "target_frame": 1,
                "direction": "forward",
                "stride": 1,
                "partition": "train",
            }
        ],
    }
    bundle["operator"] = {"mode": "fit_from_pairs"}
    return bundle


def _yaw_transform() -> dict:
    return {
        "family": "yaw",
        "physical_axis": "world_y",
        "direction": "forward",
        "signed_generator": 6.0,
        "generator_units": "degrees",
        "stride": 1,
        "stride_units": "frames",
        "cyclic": False,
        "expected_2d_family": "local_affine_approximation",
        "reference_transform": "fitted_reference",
    }


@_temporary_path
def test_bundle_content_hash_is_required(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle.pop("bundle_content_sha256")
    with _assert_raises_regex(evaluator.EvaluationContractError, match="bundle missing"):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_bundle_content_mutation_is_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["case_id"] = "mutated-after-sealing"
    with _assert_raises_regex(evaluator.EvaluationContractError, match="bundle content hash mismatch"):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_actual_file_rehash_rejects_mutation(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    input_path = Path(bundle["provenance"]["files"][1]["absolute_path"])
    input_path.write_text('{"mutated":true}\n', encoding="utf-8")
    with _assert_raises_regex(evaluator.EvaluationContractError, match="sha256 mismatch"):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_caller_coordinate_tolerance_is_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation_config"]["coordinate_consistency_tolerance"] = 1e-3
    bundle = _reseal_after_config_mutation(bundle)
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="coordinate_consistency_tolerance",
    ):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_non_authoritative_numeric_registry_copy_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path)
    copied_registry = tmp_path / "NUMERIC_CALIBRATION_v1_1.json"
    copied_registry.write_bytes(AUTHORITATIVE_NUMERIC_REGISTRY_PATH.read_bytes())
    bundle["provenance"]["numeric_registry"]["absolute_path"] = str(
        copied_registry
    )
    bundle["provenance"]["files"][0] = _file_record(
        "numeric_registry", copied_registry
    )
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="not the authoritative v1.1 path",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_numeric_registry_binding_hashes_are_exact(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["provenance"]["numeric_registry"]["file_sha256"] = "0" * 64
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="authoritative v1.1 file hash",
    ):
        evaluator.validate_bundle(_seal(bundle))

    validated = evaluator.validate_bundle(_base_bundle(tmp_path))
    assert (
        validated["numeric_registry"]["content_sha256"]
        == AUTHORITATIVE_NUMERIC_REGISTRY_CONTENT_SHA256
    )


@_temporary_path
def test_numeric_registry_json_decoder_is_strict(
    tmp_path: Path,
) -> None:
    for payload, message in [
        ('{"key":1,"key":2}', "duplicate JSON key"),
        ('{"key":NaN}', "non-finite JSON constant"),
        ('{"key":1e999}', "non-finite JSON number"),
    ]:
        path = tmp_path / "invalid_registry.json"
        path.write_text(payload, encoding="utf-8")
        with _assert_raises_regex(
            evaluator.EvaluationContractError,
            match=message,
        ):
            evaluator._load_strict_json(path, name="numeric registry")


@_temporary_path
def test_checkpoint_replay_is_blocked_before_provenance_or_checkpoint_access(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path, case_kind="checkpoint")
    with mock.patch.object(
        evaluator,
        "_validate_production_provenance",
        side_effect=AssertionError("provenance must not run"),
    ):
        with _assert_raises_regex(
            evaluator.EvaluationContractError,
            match="saved-checkpoint replay is blocked",
        ):
            evaluator.validate_bundle(bundle)


@_temporary_path
def test_planted_coordinate_only_heatmap_evidence_is_void(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    result = evaluator.evaluate_bundle(bundle)
    assert result["evaluation_config_sha256"] == evaluator.canonical_sha256(
        bundle["evaluation_config"]
    )
    assert result["evidence_availability"]["logits"] == {
        "available": False,
        "required_for_case_kind": False,
        "missing_status": "void_for_planted_coordinate_only_case",
    }
    assert result["channel_health"]["heatmap_flat_dead_count"] is None
    assert result["switching"]["heatmap_mode_evidence"] == "void_missing_logits"


@_temporary_path
def test_bbox_is_recomputed_and_supplied_mismatch_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation"]["bbox_diagonal_image01"][0] += 0.1
    bundle = _seal(bundle)
    with _assert_raises_regex(evaluator.EvaluationContractError, match="bbox diagonal disagrees"):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_boolean_evidence_is_strict(tmp_path: Path) -> None:
    for field in ("visibility", "mask"):
        bundle = _base_bundle(tmp_path)
        if field == "visibility":
            bundle["evaluation"]["visibility"][0][0] = 1
        else:
            bundle["evaluation"]["masks"]["values"][0][0][0] = 1
        bundle = _seal(bundle)
        with _assert_raises_regex(
            evaluator.EvaluationContractError,
            match="JSON booleans",
        ):
            evaluator.validate_bundle(bundle)


@_temporary_path
def test_duplicate_pair_endpoints_are_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    duplicate = copy.deepcopy(bundle["evaluation"]["pair_rows"][0])
    duplicate["pair_id"] = "evaluation:duplicate-endpoints"
    bundle["evaluation"]["pair_rows"].append(duplicate)
    bundle = _seal(bundle)
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="duplicate pair endpoints",
    ):
        evaluator.validate_bundle(bundle)


@_temporary_path
def test_fit_from_pairs_uses_train_partition_only(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    source = np.asarray(
        [[-0.5, -0.5], [0.4, -0.4], [0.0, 0.5]], dtype=np.float64
    )
    train_target = source + np.asarray([0.1, 0.0])
    validation_target = source + np.asarray([0.4, 0.0])
    bundle["evaluation"]["frame_ids"] = [20, 21]
    bundle["evaluation"]["points"] = [
        source.tolist(),
        validation_target.tolist(),
    ]
    bundle["evaluation"]["physical_states"] = [
        {"image_offset_xy": [0.0, 0.0]},
        {"image_offset_xy": [0.1, 0.0]},
    ]
    bundle["evaluation"]["pair_rows"] = [
        {
            "pair_id": "evaluation:20:21",
            "source_frame": 20,
            "target_frame": 21,
            "direction": "forward",
            "stride": 1,
            "partition": "validation",
        },
    ]
    bundle["fit"] = {
        "object_id": "synthetic",
        "seed": 0,
        "partition": "train",
        "frame_ids": [0, 1],
        "points": [source.tolist(), train_target.tolist()],
        "visibility": [[True, True, True], [True, True, True]],
        "pair_rows": [
            {
                "pair_id": "fit:0:1",
                "source_frame": 0,
                "target_frame": 1,
                "direction": "forward",
                "stride": 1,
                "partition": "train",
            }
        ],
    }
    bundle["operator"] = {"mode": "fit_from_pairs"}
    result = evaluator.evaluate_bundle(_seal(bundle))
    assert np.allclose(result["operator"]["learned_A"], np.eye(2), atol=1e-12)
    assert np.allclose(result["operator"]["learned_b"], [0.1, 0.0], atol=1e-12)
    assert [row["partition"] for row in result["operator_fit_pair_index"]] == [
        "train"
    ]
    assert [row["partition"] for row in result["evaluation_pair_index"]] == [
        "validation"
    ]
    assert result["evaluation_frame_ids"] == [20, 21]


@_temporary_path
def test_fit_from_pairs_requires_separate_fit_section(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["operator"] = {"mode": "fit_from_pairs"}
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="requires a separate fit section",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_supplied_operator_rejects_fit_section(tmp_path: Path) -> None:
    bundle = _add_translation_fit(_base_bundle(tmp_path))
    bundle["operator"] = {
        "mode": "supplied",
        "A": [[1.0, 0.0], [0.0, 1.0]],
        "b": [0.1, 0.0],
    }
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="supplied operator forbids a fit section",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_fit_section_rejects_non_train_rows(tmp_path: Path) -> None:
    for forbidden_partition in ("validation", "test"):
        bundle = _add_translation_fit(_base_bundle(tmp_path))
        bundle["fit"]["pair_rows"][0]["partition"] = forbidden_partition
        with _assert_raises_regex(
            evaluator.EvaluationContractError,
            match="fit pair_rows must contain only train rows",
        ):
            evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_fit_is_independent_of_evaluation_points_and_sensitive_to_fit_points(
    tmp_path: Path,
) -> None:
    baseline_bundle = _add_translation_fit(_base_bundle(tmp_path))
    baseline = evaluator.evaluate_bundle(_seal(baseline_bundle))
    baseline_A = np.asarray(baseline["operator"]["learned_A"])
    baseline_b = np.asarray(baseline["operator"]["learned_b"])

    evaluation_mutation = copy.deepcopy(baseline_bundle)
    evaluation_mutation["evaluation"]["points"][1][0] = [0.9, 0.8]
    evaluation_result = evaluator.evaluate_bundle(
        _seal(evaluation_mutation)
    )
    assert np.allclose(
        evaluation_result["operator"]["learned_A"],
        baseline_A,
        atol=1e-12,
    )
    assert np.allclose(
        evaluation_result["operator"]["learned_b"],
        baseline_b,
        atol=1e-12,
    )

    fit_mutation = copy.deepcopy(baseline_bundle)
    source = np.asarray(fit_mutation["fit"]["points"][0])
    fit_mutation["fit"]["points"][1] = (
        source + np.asarray([0.2, 0.0])
    ).tolist()
    fit_result = evaluator.evaluate_bundle(_seal(fit_mutation))
    assert not np.allclose(
        fit_result["operator"]["learned_b"],
        baseline_b,
        atol=1e-12,
    )
    assert np.allclose(
        fit_result["operator"]["learned_b"],
        [0.2, 0.0],
        atol=1e-12,
    )


@_temporary_path
def test_evaluation_section_rejects_train_partition(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation"]["partition"] = "train"
    bundle["evaluation"]["pair_rows"][0]["partition"] = "train"
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="evaluation partition must be validation, test, or full_corpus",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_wrong_translation_sign_is_critical(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["operator"]["b"] = [-0.1, 0.0]
    result = evaluator.evaluate_bundle(_seal(bundle))
    assert "operator_has_wrong_locked_sign" in result["critical_failures"]


@_temporary_path
def test_legacy_direction_vocabulary_is_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["transform"]["direction"] = "positive"
    bundle["evaluation"]["pair_rows"][0]["direction"] = "positive"
    with _assert_raises_regex(evaluator.EvaluationContractError, match="direction is invalid"):
        evaluator.validate_bundle(_seal(bundle))


def test_scale_uses_signed_log_step_and_exp_ratio() -> None:
    for direction, log_step in (
        ("forward", math.log(1.1)),
        ("reverse", math.log(0.9)),
    ):
        ratio = math.exp(log_step)
        transform = {
            "family": "scale",
            "physical_axis": "uniform",
            "direction": direction,
            "signed_generator": log_step,
            "generator_units": "log_scale",
            "stride": 1,
            "stride_units": "frames",
            "cyclic": False,
            "expected_2d_family": "uniform_scale_about_projected_center",
            "projected_centre_xy": [0.0, 0.0],
        }
        family, target_A, _ = evaluator._validate_transform(
            transform,
            tolerance=1e-12,
        )
        assert family == "scale"
        assert np.allclose(target_A, ratio * np.eye(2), atol=1e-12)


@_temporary_path
def test_exact_target_geometry_cannot_be_supplied_by_caller(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["transform"]["target_A"] = [[1.0, 0.0], [0.0, 1.0]]
    bundle["transform"]["target_b"] = [0.1, 0.0]
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="transform has unknown fields.*target_A.*target_b",
    ):
        evaluator.validate_bundle(_seal(bundle))


def test_rendered_translation_uses_world_steps_and_calibrated_offsets() -> None:
    transform = {
        "family": "translation",
        "physical_axis": "world_y",
        "direction": "forward",
        "signed_generator": 0.048,
        "generator_units": "world_units",
        "stride": 3,
        "stride_units": "steps",
        "cyclic": False,
        "expected_2d_family": "image_plane_translation_after_calibration",
        "calibrated_image_delta_xy": [0.0, 0.02],
        "expected_image_component": 1,
        "expected_image_sign": 1,
        "translation_source": "rendered_world_calibrated",
    }
    family, target_A, target_b = evaluator._validate_transform(
        transform,
        tolerance=1e-12,
    )
    states, state_A, state_b = evaluator._validate_and_derive_physical_states(
        family=family,
        physical_states=[
            {
                "dx_world": 0.064,
                "dy_world": -0.08,
                "grid_x": 9,
                "grid_y": 0,
                "calibrated_image_offset_xy": [0.0, 0.0],
            },
            {
                "dx_world": 0.064,
                "dy_world": -0.032,
                "grid_x": 9,
                "grid_y": 3,
                "calibrated_image_offset_xy": [0.0, 0.02],
            },
        ],
        frame_ids=[0, 3],
        frame_index_by_id={0: 0, 3: 1},
        transform=transform,
        pair_rows=[
            {
                "pair_id": "fit:0:1",
                "source_frame": 0,
                "target_frame": 3,
                "direction": "forward",
                "stride": 3,
                "partition": "train",
            }
        ],
        target_A=target_A,
        target_b=target_b,
        tolerance=1e-12,
    )
    assert states[1]["dy_world"] == -0.032
    assert state_A is not None and state_b is not None
    assert np.allclose(state_b[1], [0.0, 0.02], atol=1e-12)


@_temporary_path
def test_structural_negative_control_flag_uses_all_channels(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    coincident = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    bundle["evaluation"]["points"] = [coincident, coincident]
    result = evaluator.evaluate_bundle(_seal(bundle))
    evidence = result["collapse_evidence"]
    assert evidence["structural_negative_control_collapse"] is True
    assert evidence["structural_negative_control_is_coincidence_trigger"] is False
    assert (
        evidence["structural_negative_control_definition"]
        == "every_all_channel_pair_is_evaluable_and_has_the_frozen_persistent_duplicate_category"
    )
    assert evidence["eligible_only_structural_negative_control_collapse"] is None
    assert evidence["eligible_only_status"] == "void_below_minimum_eligible_channels"
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is True
    )
    assert (
        evidence[
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead"
        ]
        == "available"
    )
    assert evidence["channels_not_confirmed_flat_dead_indices"] == [0, 1, 2]
    assert evidence["confirmed_flat_dead_channel_indices"] == []


@_temporary_path
def test_v2_duplicate_cluster_plus_flat_dead_is_collapse(
    tmp_path: Path,
) -> None:
    sharp_duplicate = [[30.0, -30.0], [-30.0, -30.0]]
    flat_dead = [[0.0, 0.0], [0.0, 0.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [sharp_duplicate, sharp_duplicate, sharp_duplicate, flat_dead],
    )
    result = evaluator.evaluate_bundle(bundle)
    evidence = result["collapse_evidence"]
    assert evidence["structural_negative_control_collapse"] is False
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is True
    )
    assert evidence["channels_not_confirmed_flat_dead_indices"] == [0, 1, 2]
    assert evidence["confirmed_flat_dead_channel_indices"] == [3]
    retained = result["trajectory_separation"][
        "channels_not_confirmed_flat_dead"
    ]
    assert retained["pair_count"] == 3
    assert retained["evaluable_pair_count"] == 3
    assert retained["category_counts"]["persistent_duplicate"] == 3


@_temporary_path
def test_v2_all_sharp_duplicate_channels_are_collapse(
    tmp_path: Path,
) -> None:
    sharp_duplicate = [[30.0, -30.0], [-30.0, -30.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [sharp_duplicate, sharp_duplicate, sharp_duplicate],
    )
    evidence = evaluator.evaluate_bundle(bundle)["collapse_evidence"]
    assert evidence["structural_negative_control_collapse"] is True
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is True
    )
    assert evidence["confirmed_flat_dead_channel_indices"] == []


@_temporary_path
def test_v2_separated_sharp_channels_plus_flat_dead_are_not_collapse(
    tmp_path: Path,
) -> None:
    sharp_top_left = [[30.0, -30.0], [-30.0, -30.0]]
    sharp_bottom_right = [[-30.0, -30.0], [-30.0, 30.0]]
    flat_dead = [[0.0, 0.0], [0.0, 0.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [sharp_top_left, sharp_bottom_right, flat_dead],
    )
    evidence = evaluator.evaluate_bundle(bundle)["collapse_evidence"]
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is False
    )
    assert evidence["channels_not_confirmed_flat_dead_indices"] == [0, 1]
    assert evidence["confirmed_flat_dead_channel_indices"] == [2]


@_temporary_path
def test_v2_live_separated_channel_prevents_duplicate_cluster_collapse(
    tmp_path: Path,
) -> None:
    sharp_top_left = [[30.0, -30.0], [-30.0, -30.0]]
    sharp_bottom_right = [[-30.0, -30.0], [-30.0, 30.0]]
    flat_dead = [[0.0, 0.0], [0.0, 0.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [
            sharp_top_left,
            sharp_top_left,
            sharp_top_left,
            sharp_bottom_right,
            flat_dead,
        ],
    )
    result = evaluator.evaluate_bundle(bundle)
    evidence = result["collapse_evidence"]
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is False
    )
    assert evidence["channels_not_confirmed_flat_dead_indices"] == [0, 1, 2, 3]
    assert evidence["confirmed_flat_dead_channel_indices"] == [4]
    assert result["channel_health"]["motion_active_count"] == 0


@_temporary_path
def test_v2_below_minimum_non_flat_dead_channels_is_void(
    tmp_path: Path,
) -> None:
    sharp = [[30.0, -30.0], [-30.0, -30.0]]
    flat_dead = [[0.0, 0.0], [0.0, 0.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [sharp, flat_dead, flat_dead],
    )
    result = evaluator.evaluate_bundle(bundle)
    evidence = result["collapse_evidence"]
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is None
    )
    assert (
        evidence[
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead"
        ]
        == "void_below_minimum_channels_not_confirmed_flat_dead"
    )
    assert evidence["channels_not_confirmed_flat_dead_indices"] == [0]
    assert evidence["confirmed_flat_dead_channel_indices"] == [1, 2]
    assert (
        result["trajectory_separation"]["channels_not_confirmed_flat_dead"]
        is None
    )


@_temporary_path
def test_v2_void_retained_pair_is_not_collapse(
    tmp_path: Path,
) -> None:
    sharp_duplicate = [[30.0, -30.0], [-30.0, -30.0]]
    bundle = _base_bundle(tmp_path)
    bundle = _with_evaluation_logits(
        bundle,
        [sharp_duplicate, sharp_duplicate, sharp_duplicate],
    )
    bundle["evaluation"]["visibility"] = [
        [True, True, False],
        [True, False, True],
    ]
    result = evaluator.evaluate_bundle(_seal(bundle))
    evidence = result["collapse_evidence"]
    retained = result["trajectory_separation"][
        "channels_not_confirmed_flat_dead"
    ]
    assert retained["void_pair_count"] == 1
    assert retained["persistent_duplicate_pair_rate"] == 1.0
    assert retained["all_pairs_persistent_duplicate"] is False
    assert (
        evidence[
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead"
        ]
        is False
    )


@_temporary_path
def test_pairwise_visibility_void_is_explicit(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation"]["visibility"] = [
        [True, True, False],
        [True, False, True],
    ]
    result = evaluator.evaluate_bundle(_seal(bundle))
    all_channels = result["trajectory_separation"]["all_channels"]
    assert all_channels["pair_count"] == 3
    assert all_channels["evaluable_pair_count"] == 2
    assert all_channels["void_pair_count"] == 1
    pair = next(
        row
        for row in all_channels["pairs"]
        if row["channel_i"] == 1 and row["channel_j"] == 2
    )
    assert pair["status"] == "void_no_joint_visibility"
    assert pair["category"] is None


@_temporary_path
def test_all_visible_pairwise_values_retain_original_definition(tmp_path: Path) -> None:
    result = evaluator.evaluate_bundle(_base_bundle(tmp_path))
    all_channels = result["trajectory_separation"]["all_channels"]
    assert all_channels["void_pair_count"] == 0
    assert all_channels["all_selected_channels_visible_in_all_frames"] is True
    assert math.isclose(
        all_channels["trajectory_median_nearest_neighbour_objdiag"],
        0.5 / math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    nearest_summary = all_channels["nearest_neighbour"]["summary"]
    assert nearest_summary["sample_unit"] == (
        "visible_frame_channel_nearest_neighbour_distance_objdiag"
    )
    assert nearest_summary["aggregation_hierarchy"] == [
        "stratum",
        "frame",
        "channel",
    ]
    drift_summary = result["canonical_drift"]["summary"]
    assert drift_summary["sample_unit"] == "channel_canonical_rms_objdiag"
    assert drift_summary["aggregation_hierarchy"] == [
        "stratum",
        "channel",
        "visible_frames",
    ]


def test_reflection_is_flagged_for_every_family() -> None:
    for family in ("roll", "translation", "scale", "yaw", "pitch"):
        target_A = np.eye(2)
        target_b = np.zeros(2)
        transform = {
            "signed_generator": 6.0,
            "generator_units": "degrees",
            "physical_axis": "world_z",
            "expected_image_component": 0,
            "projected_centre_xy": [0.0, 0.0],
        }
        if family == "translation":
            transform.update(
                {
                    "signed_generator": 0.1,
                    "physical_axis": "image_x",
                    "expected_image_component": 0,
                }
            )
            target_b = np.asarray([0.1, 0.0])
        elif family == "scale":
            transform.update(
                {
                    "signed_generator": math.log(1.1),
                    "physical_axis": "uniform",
                }
            )
            target_A = 1.1 * np.eye(2)
        metrics = evaluator._operator_metrics(
            family,
            np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
            np.zeros(2),
            target_A,
            target_b,
            transform,
            {"operator_composition_horizons": []},
        )
        assert metrics["improper_or_reflection"] is True


@_temporary_path
def test_caller_supplied_state_transforms_are_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation"]["state_A"] = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="state_A",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_validation_endpoints_must_not_overlap_train(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    points = bundle["evaluation"]["points"][0]
    bundle["fit"] = {
        "object_id": "synthetic",
        "seed": 0,
        "partition": "train",
        "frame_ids": [0, 100],
        "points": [points, points],
        "visibility": [[True, True, True], [True, True, True]],
        "pair_rows": [
            {
                "pair_id": "fit:0:1",
                "source_frame": 0,
                "target_frame": 100,
                "direction": "forward",
                "stride": 1,
                "partition": "train",
            }
        ],
    }
    bundle["operator"] = {"mode": "fit_from_pairs"}
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="training and evaluation endpoints overlap",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_generic_roll_suppresses_full_orbit_metrics(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    angle = math.radians(6.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    source = np.asarray(
        bundle["evaluation"]["points"][0],
        dtype=np.float64,
    )
    bundle["transform"] = {
        "family": "roll",
        "physical_axis": "world_z",
        "direction": "forward",
        "signed_generator": 6.0,
        "generator_units": "degrees",
        "stride": 1,
        "stride_units": "frames",
        "cyclic": True,
        "expected_2d_family": "planar_rotation_about_projected_center",
        "projected_centre_xy": [0.0, 0.0],
    }
    bundle["evaluation"]["points"] = [
        source.tolist(),
        evaluator.apply_affine(source, rotation, np.zeros(2)).tolist(),
    ]
    bundle["evaluation"]["physical_states"] = [
        {"theta_deg": 0.0},
        {"theta_deg": 6.0},
    ]
    bundle["operator"] = {
        "mode": "supplied",
        "A": rotation.tolist(),
        "b": [0.0, 0.0],
    }
    result = evaluator.evaluate_bundle(_seal(bundle))
    assert result["rollout"]["status"] == "not_full_primary_roll"
    assert "full_corpus_identity_normalized_auc" not in result["rollout"]
    assert "role_scoped_holdout_identity_normalized_auc" not in result["rollout"]
    assert "closure" not in result["rollout"]


@_temporary_path
def test_yaw_rejects_caller_supplied_reference_matrix(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["transform"] = _yaw_transform()
    bundle["transform"]["target_A"] = [[1.0, 0.0], [0.0, 1.0]]
    bundle["transform"]["target_b"] = [0.05, 0.0]
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="transform has unknown fields.*target_A.*target_b",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_yaw_requires_fit_from_pairs(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["transform"] = _yaw_transform()
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="yaw requires fit_from_pairs",
    ):
        evaluator.validate_bundle(_seal(bundle))


@_temporary_path
def test_yaw_residual_rows_record_physical_metadata(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    source = np.asarray(
        bundle["evaluation"]["points"][0],
        dtype=np.float64,
    )
    target = source + np.asarray([0.05, 0.0])
    bundle["transform"] = _yaw_transform()
    bundle["evaluation"]["points"] = [source.tolist(), target.tolist()]
    bundle["evaluation"]["physical_states"] = [
        {
            "theta_deg": 0.0,
            "projection_model": "synthetic_perspective",
            "depth_configuration": "fixed_depth_control",
        },
        {
            "theta_deg": 6.0,
            "projection_model": "synthetic_perspective",
            "depth_configuration": "fixed_depth_control",
        },
    ]
    bundle["fit"] = {
        "object_id": "synthetic",
        "seed": 0,
        "partition": "train",
        "frame_ids": [0, 1],
        "points": [source.tolist(), target.tolist()],
        "visibility": [[True, True, True], [True, True, True]],
        "pair_rows": [
            {
                "pair_id": "fit:0:1",
                "source_frame": 0,
                "target_frame": 1,
                "direction": "forward",
                "stride": 1,
                "partition": "train",
            }
        ],
    }
    bundle["operator"] = {"mode": "fit_from_pairs"}
    result = evaluator.evaluate_bundle(_seal(bundle))
    assert result["canonical_drift"]["status"] == "not_applicable"
    assert np.allclose(
        result["operator"]["target_A"],
        result["operator"]["learned_A"],
        atol=1e-12,
    )
    assert np.allclose(
        result["operator"]["target_b"],
        result["operator"]["learned_b"],
        atol=1e-12,
    )
    row = result["fitted_reference_residual"]["pairs"][0]
    assert row["source_theta_deg"] == 0.0
    assert row["target_theta_deg"] == 6.0
    assert row["source_projection_model"] == "synthetic_perspective"
    assert row["target_projection_model"] == "synthetic_perspective"
    assert row["source_depth_configuration"] == "fixed_depth_control"
    assert row["target_depth_configuration"] == "fixed_depth_control"
    assert row["direction"] == "forward"
    assert row["stride"] == 1
    assert row["evaluation_partition"] == "validation"
    assert result["switching"]["reference_transform"] == "fitted_reference"
    residual_summary = result["fitted_reference_residual"]["summary"]
    assert residual_summary["sample_unit"] == (
        "evaluation_pair_rms_normalized_xy"
    )
    assert residual_summary["aggregation_hierarchy"] == [
        "stratum",
        "evaluation_pair",
        "jointly_visible_channels",
    ]


@_temporary_path
def test_zero_motion_threshold_is_rejected(tmp_path: Path) -> None:
    bundle = _base_bundle(tmp_path)
    bundle["evaluation_config"]["motion_fraction_min"] = 0.0
    bundle = _reseal_after_config_mutation(bundle)
    with _assert_raises_regex(
        evaluator.EvaluationContractError,
        match="strictly positive",
    ):
        evaluator.validate_bundle(bundle)


def test_flat_dead_heatmaps_do_not_change_active_on_object_eligibility() -> None:
    source = np.asarray(
        [[-0.5, -0.5], [0.0, -0.5], [0.5, -0.5]],
        dtype=np.float64,
    )
    points = np.stack([source, source + np.asarray([0.2, 0.0])])
    health = evaluator._channel_health(
        points,
        np.ones((2, 4, 4), dtype=bool),
        np.zeros((2, 3, 2, 2), dtype=np.float64),
        np.ones((2, 3), dtype=bool),
        [
            {
                "pair_id": "fit:0:1",
                "source_frame": 0,
                "target_frame": 1,
                "direction": "forward",
                "stride": 1,
                "partition": "train",
            }
        ],
        {0: 0, 1: 1},
        {
            "temperature": 1.0,
            "softmax_dtype": "float64",
        },
        {
            "motion_reference_magnitude_image01": 0.1,
            "motion_fraction_min": 0.5,
            "on_object_rate_min": 0.5,
        },
    )
    assert health["heatmap_flat_dead_count"] == 3
    assert health["active_on_object_count"] == 3
    assert health["eligible_count"] == 3
    assert health["eligibility_includes_heatmap_health"] is False


@_temporary_path
def test_public_evaluator_keeps_moving_high_entropy_channels_eligible(
    tmp_path: Path,
) -> None:
    bundle = _base_bundle(tmp_path)
    frame_probabilities = np.asarray(
        [
            [[0.4, 0.1], [0.4, 0.1]],
            [[0.1, 0.4], [0.1, 0.4]],
        ],
        dtype=np.float64,
    )
    logits = np.log(
        np.repeat(frame_probabilities[:, None, :, :], 3, axis=1)
    )
    points = evaluator.spatial_expectation(
        logits,
        temperature=1.0,
        softmax_dtype="float64",
    )
    bundle["evaluation"]["logits"] = logits.tolist()
    bundle["evaluation"]["points"] = points.tolist()
    result = evaluator.evaluate_bundle(_seal(bundle))
    assert result["channel_health"]["heatmap_flat_dead_count"] == 3
    assert result["channel_health"]["motion_active_count"] == 3
    assert result["channel_health"]["active_on_object_count"] == 3
    assert result["channel_health"]["eligible_count"] == 3
    assert (
        result["channel_health"]["eligibility_includes_heatmap_health"]
        is False
    )


def test_public_evaluator_has_no_provenance_validator_override() -> None:
    for function in (evaluator.validate_bundle, evaluator.evaluate_bundle):
        assert "_provenance_validator" not in inspect.signature(
            function
        ).parameters
    assert evaluator.provenance_contract.LOADED_SOURCE_ROLES == frozenset(
        {"evaluator_source", "array_codec_source"}
    )
    assert evaluator.provenance_contract.required_loaded_source_roles(
        "planted"
    ) == frozenset(
        {
            "evaluator_source",
            "array_codec_source",
            "oracle_harness_source",
        }
    )


def _full_primary_roll_contract() -> tuple[dict, dict, list[dict]]:
    config = {
        "protocol": "full_primary_roll",
        "full_rollout_horizons": list(range(1, 60)),
        "holdout_rollout_horizons": list(range(1, 8)),
        "closure_horizon": 60,
        "role_scoped_holdout_frame_ids": list(range(24)),
    }
    transform = {
        "stride": 3,
        "signed_generator": 6.0,
        "direction": "forward",
        "cyclic": True,
    }
    pair_rows = [
        {
            "pair_id": f"full:{source}:{(source + 3) % 180}",
            "source_frame": source,
            "target_frame": (source + 3) % 180,
            "direction": "forward",
            "stride": 3,
            "partition": "full_corpus",
        }
        for source in range(180)
    ]
    return config, transform, pair_rows


def test_rollout_summaries_name_sample_unit_and_aggregation() -> None:
    points = np.asarray(
        [
            [[-0.5, -0.5], [0.5, 0.5]],
            [[-0.4, -0.5], [0.6, 0.5]],
        ],
        dtype=np.float64,
    )
    summary = evaluator._rollout_horizon(
        points,
        [0, 1],
        np.eye(2),
        np.asarray([0.1, 0.0]),
        starts=[0],
        targets=[1],
        k=1,
    )
    assert summary["model_mse"]["sample_unit"] == (
        "rollout_start_target_pair_model_mse"
    )
    assert summary["identity_mse"]["sample_unit"] == (
        "rollout_start_target_pair_identity_mse"
    )
    expected_hierarchy = [
        "stratum",
        "horizon",
        "start_target_pair",
        "channels",
        "xy",
    ]
    assert summary["model_mse"]["aggregation_hierarchy"] == expected_hierarchy
    assert summary["identity_mse"]["aggregation_hierarchy"] == expected_hierarchy


def test_full_primary_roll_horizon_mutation_is_rejected() -> None:
    config, transform, pair_rows = _full_primary_roll_contract()
    config["full_rollout_horizons"] = [1]
    with _assert_raises_regex(evaluator.EvaluationContractError, match="k=1..59"):
        evaluator._validate_primary_roll_protocol(
            frame_ids=list(range(180)),
            transform=transform,
            config=config,
            pair_rows=pair_rows,
        )


def test_full_primary_roll_requires_every_cyclic_edge_once() -> None:
    config, transform, pair_rows = _full_primary_roll_contract()
    evaluator._validate_primary_roll_protocol(
        frame_ids=list(range(180)),
        transform=transform,
        config=config,
        pair_rows=pair_rows,
    )
    with _assert_raises_regex(evaluator.EvaluationContractError, match="exactly 180 edges"):
        evaluator._validate_primary_roll_protocol(
            frame_ids=list(range(180)),
            transform=transform,
            config=config,
            pair_rows=pair_rows[:-1],
        )


def test_full_primary_roll_requires_full_corpus_partition_label() -> None:
    config, transform, pair_rows = _full_primary_roll_contract()
    pair_rows[0]["partition"] = "validation"
    with _assert_raises_regex(evaluator.EvaluationContractError, match="partition=full_corpus"):
        evaluator._validate_primary_roll_protocol(
            frame_ids=list(range(180)),
            transform=transform,
            config=config,
            pair_rows=pair_rows,
        )


def test_full_primary_roll_duration_graph_is_three_sixty_edge_cycles() -> None:
    _config, _transform, pair_rows = _full_primary_roll_contract()
    graph = evaluator._pair_graph_components(
        pair_rows,
        {frame_id: frame_id for frame_id in range(180)},
    )
    assert graph["status"] == (
        "computed_disjoint_pair_graph_paths_and_cycles"
    )
    assert len(graph["components"]) == 3
    assert all(
        component["kind"] == "cycle"
        and component["cyclic"] is True
        and component["transition_count"] == 60
        for component in graph["components"]
    )
    assert [
        component["source_frame_ids"][0]
        for component in graph["components"]
    ] == [0, 1, 2]


def test_physical_change_duration_joins_only_a_cyclic_boundary() -> None:
    assert evaluator._true_run_lengths(
        [True, False, True],
        cyclic=False,
    ) == [1, 1]
    assert evaluator._true_run_lengths(
        [True, False, True],
        cyclic=True,
    ) == [2]
    assert evaluator._true_run_lengths(
        [True, True, True],
        cyclic=True,
    ) == [3]
    assert evaluator._true_run_lengths(
        [True, None, True],
        cyclic=True,
    ) == [2]


def test_switching_is_pair_order_invariant_and_reports_peak_assignments() -> None:
    points = np.asarray(
        [
            [[-1.0, -1.0], [1.0, 1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
        ],
        dtype=np.float64,
    )
    logits = np.full((4, 2, 3, 3), -100.0, dtype=np.float64)
    peak_vu = (
        ((0, 0), (2, 2)),
        ((2, 2), (0, 0)),
        ((0, 0), (2, 2)),
        ((2, 2), (0, 0)),
    )
    for frame_index, frame_peaks in enumerate(peak_vu):
        for channel, (v, u) in enumerate(frame_peaks):
            logits[frame_index, channel, v, u] = 0.0
    pair_rows = [
        {
            "pair_id": "p01",
            "source_frame": 0,
            "target_frame": 1,
            "partition": "validation",
        },
        {
            "pair_id": "p23",
            "source_frame": 2,
            "target_frame": 3,
            "partition": "validation",
        },
    ]
    arguments = {
        "points": points,
        "reference_A": np.eye(2),
        "reference_b": np.zeros(2),
        "reference_label": "locked_exact_pair_transform",
        "logits": logits,
        "visibility": np.ones((4, 2), dtype=bool),
        "frame_index_by_id": {0: 0, 1: 1, 2: 2, 3: 3},
    }
    forward = evaluator._switching(
        **arguments,
        pair_rows=pair_rows,
    )
    reversed_input = evaluator._switching(
        **arguments,
        pair_rows=list(reversed(pair_rows)),
    )
    assert forward == reversed_input
    assert [
        row["pair_id"] for row in forward["transitions"]
    ] == ["p01", "p23"]
    assert (
        forward["heatmap_mode_assignment_changed_channel_event_count"]
        == 4
    )
    assert [
        row["mode_changed_channel_count"]
        for row in forward["transitions"]
    ] == [2, 2]
    assert [
        row["all_change_durations_transitions"]
        for row in forward[
            "heatmap_mode_assignment_physical_change_durations_by_channel"
        ]
    ] == [[1, 1], [1, 1]]
    assert all(
        row["duration_evidence_status"] == "complete"
        for row in forward[
            "heatmap_mode_assignment_physical_change_durations_by_channel"
        ]
    )
    assert forward["physical_change_duration_status"] == (
        "computed_disjoint_pair_graph_paths_and_cycles"
    )
    components = forward["pair_graph_duration_decomposition"]["components"]
    assert [component["kind"] for component in components] == ["path", "path"]
    assert [component["pair_ids"] for component in components] == [
        ["p01"],
        ["p23"],
    ]
    assert "assignment_change_durations_by_channel" not in forward
    assert (
        "heatmap_mode_assignment_change_durations_by_channel"
        not in forward
    )
    assert (
        "assignment_changed_run_lengths_in_sorted_report_rows_by_channel"
        not in forward
    )


def test_exact_assignment_ties_are_reported_as_ambiguous() -> None:
    diagnostics = evaluator._assignment_diagnostics(
        np.zeros((3, 3), dtype=np.float64)
    )
    assert diagnostics["deterministic_representative_assignment"] == [0, 1, 2]
    assert diagnostics["best_cost"] == 0.0
    assert diagnostics["second_best_cost"] == 0.0
    assert diagnostics["best_vs_second_margin"] == 0.0
    assert diagnostics["exact_tie"] is True
    assert diagnostics["unique_optimum"] is False

    switching = evaluator._switching(
        np.zeros((2, 3, 2), dtype=np.float64),
        np.eye(2),
        np.zeros(2),
        "locked_exact_pair_transform",
        np.zeros((2, 3, 2, 2), dtype=np.float64),
        np.ones((2, 3), dtype=bool),
        [
            {
                "pair_id": "p01",
                "source_frame": 0,
                "target_frame": 1,
                "partition": "validation",
            }
        ],
        {0: 0, 1: 1},
    )
    transition = switching["transitions"][0]
    assert transition["assignment_status"] == "ambiguous_exact_tie"
    assert transition["assignment_unique"] is False
    assert transition["changed_channels"] is None
    assert transition["changed_channel_count"] is None
    assert transition["mode_assignment_status"] == "ambiguous_exact_tie"
    assert transition["mode_assignment_unique"] is False
    assert transition["mode_changed_channels"] is None
    assert transition["mode_changed_channel_count"] is None
    assert switching["assignment_changed_channel_event_count"] == 0
    assert switching["assignment_ambiguous_transition_count"] == 1
    assert (
        switching["heatmap_mode_assignment_changed_channel_event_count"]
        == 0
    )
    assert (
        switching["heatmap_mode_assignment_ambiguous_transition_count"]
        == 1
    )
    assert switching[
        "assignment_physical_change_durations_by_channel"
    ][0]["all_change_durations_transitions"] == []
    assert switching[
        "assignment_physical_change_durations_by_channel"
    ][0]["maximum_change_duration_transitions"] is None
    assert switching[
        "assignment_physical_change_durations_by_channel"
    ][0]["duration_evidence_status"] == (
        "void_no_unambiguous_transitions"
    )


def test_role_subset_is_not_labelled_heldout_without_exposure_proof() -> None:
    config, transform, _pair_rows = _full_primary_roll_contract()
    base = np.asarray(
        [[-0.4, -0.2], [0.5, -0.25], [0.3, 0.45]],
        dtype=np.float64,
    )
    points = []
    for frame_index in range(180):
        theta = math.radians(2.0 * frame_index)
        rotation = np.asarray(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float64,
        )
        points.append(evaluator.apply_affine(base, rotation, np.zeros(2)))
    step = math.radians(6.0)
    operator_A = np.asarray(
        [
            [math.cos(step), -math.sin(step)],
            [math.sin(step), math.cos(step)],
        ],
        dtype=np.float64,
    )
    result = evaluator._rollout_metrics(
        np.stack(points),
        list(range(180)),
        operator_A,
        np.zeros(2),
        transform,
        config,
    )
    subset = result["role_scoped_holdout_identity_normalized_auc"]
    assert subset["selection_status"] == (
        "role_scoped_endpoint_subset_not_heldout_without_proven_non_exposure"
    )
    assert (
        subset["training_exposure_status"]
        == "not_established_by_frame_membership"
    )


def test_operator_singular_values_have_a_known_answer() -> None:
    A = np.asarray([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64)
    transform = {
        "generator_units": "degrees",
        "signed_generator": 0.0,
    }
    metrics = evaluator._operator_metrics(
        "roll",
        A,
        np.zeros(2),
        np.eye(2),
        np.zeros(2),
        transform,
        {"operator_composition_horizons": []},
    )
    np.testing.assert_allclose(
        metrics["singular_values"],
        [2.0, 0.5],
        rtol=0.0,
        atol=1e-12,
    )


def load_tests(loader, standard_tests, pattern):
    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(
                unittest.FunctionTestCase(function, description=name)
            )
    return suite


if __name__ == "__main__":
    unittest.main()
