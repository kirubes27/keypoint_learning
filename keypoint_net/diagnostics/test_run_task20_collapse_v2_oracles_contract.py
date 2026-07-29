"""Focused pre-review contract checks for the Task-20 collapse-v2 harness.

These tests exercise only deterministic planted construction and the
production evaluator.  They do not authorize or execute a checkpoint replay.
The candidate sources are not yet provenance-addressable at one exact commit,
so evaluator calls use a narrow in-process provenance seam.  Production
harness code has no such seam.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import eval_representation as evaluator
from keypoint_net import representation_checkpoint_authorization as authorization
from keypoint_net.diagnostics import (
    run_task20_collapse_v2_oracles as harness,
)


def _unit_provenance_validator(
    record: dict,
    *,
    case_kind: str,
    fit_from_pairs: bool,
) -> dict:
    if case_kind != "planted" or fit_from_pairs:
        raise evaluator.EvaluationContractError(
            "unit seam accepts only supplied planted cases",
            code="provenance_invalid",
        )
    if record.get("external_files") != []:
        raise evaluator.EvaluationContractError(
            "planted unit case unexpectedly binds an external file",
            code="provenance_invalid",
        )
    path = evaluator.AUTHORITATIVE_NUMERIC_REGISTRY_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_commit_verified": True,
        "source_commit": "0" * 40,
        "committed_files": [
            {
                "role": "numeric_registry",
                "repo_relative_path": str(
                    evaluator.AUTHORITATIVE_NUMERIC_REGISTRY_RELATIVE_PATH
                ),
                "absolute_path": str(path),
                "file_sha256": digest,
                "git_blob_oid": "unit-seam-only",
            }
        ],
        "external_files": [],
    }


class Task20CollapseV2OracleHarnessContractTest(unittest.TestCase):
    def test_exact_manifest_and_seven_deterministic_constructions(self) -> None:
        manifest, cases = harness.load_manifest()
        expected_case_ids = [
            case_id
            for case_id, _pattern, _semantic_role in harness.EXPECTED_CASES
        ]
        self.assertEqual(len(cases), 7)
        self.assertEqual(
            [case["case_id"] for case in cases],
            expected_case_ids,
        )
        self.assertEqual(
            list(harness.INDEPENDENT_OUTCOME_MATRIX),
            expected_case_ids,
        )
        self.assertEqual(
            manifest["semantic_lock"]["excluded_channel_predicate"],
            "heatmap_flat_dead_is_exactly_true",
        )
        self.assertEqual(
            manifest["semantic_lock"]["retained_channel_predicate"],
            "heatmap_flat_dead_is_not_exactly_true",
        )
        self.assertEqual(
            manifest["semantic_lock"]["forbidden_channel_filters"],
            [
                "motion_active",
                "on_object",
                "health_label",
                "eligible_indices",
            ],
        )
        self.assertEqual(
            manifest["semantic_lock"]["below_minimum_disposition"],
            "void_null_not_false",
        )
        self.assertFalse(
            manifest["semantic_lock"]["saved_checkpoint_execution_authority"]
        )
        self.assertFalse(
            manifest["semantic_lock"]["task_20_55_80_scientific_authority"]
        )

        case_paths = harness._case_paths(manifest)
        expected_config_hashes = {
            entry["case_id"]: entry["expected_evaluation_config_sha256"]
            for entry in manifest["ordered_case_definitions"]
        }
        with mock.patch.object(
            evaluator,
            "_validate_production_provenance",
            _unit_provenance_validator,
        ):
            for case in cases:
                case_id = case["case_id"]
                first_bundle = harness.construct_case_bundle(
                    case,
                    case_path=case_paths[case_id],
                    source_commit="0" * 40,
                )
                second_bundle = harness.construct_case_bundle(
                    case,
                    case_path=case_paths[case_id],
                    source_commit="0" * 40,
                )
                self.assertEqual(
                    evaluator.canonical_json_bytes(first_bundle),
                    evaluator.canonical_json_bytes(second_bundle),
                )
                self.assertEqual(first_bundle["case_kind"], "planted")
                self.assertEqual(first_bundle["transform"]["family"], "roll")
                self.assertEqual(
                    first_bundle["transform"]["physical_axis"],
                    "world_z",
                )
                self.assertNotIn("fit", first_bundle)
                self.assertEqual(
                    first_bundle["provenance"]["external_files"],
                    [],
                )
                self.assertEqual(
                    evaluator.canonical_sha256(
                        first_bundle["evaluation_config"]
                    ),
                    expected_config_hashes[case_id],
                )
                if (
                    case_id
                    == "coordinate_only_missing_logits_retains_all_channels"
                ):
                    self.assertNotIn("logits", first_bundle["evaluation"])
                else:
                    self.assertIn("logits", first_bundle["evaluation"])

                first_result = evaluator.evaluate_bundle(first_bundle)
                second_result = evaluator.evaluate_bundle(second_bundle)
                self.assertEqual(
                    evaluator.canonical_json_bytes(first_result),
                    evaluator.canonical_json_bytes(second_result),
                )
                self.assertEqual(
                    first_result["evaluation_config_sha256"],
                    expected_config_hashes[case_id],
                )
                config_check = harness.validate_result_evaluation_config(
                    case_id,
                    first_result,
                    expected_sha256=expected_config_hashes[case_id],
                )
                self.assertTrue(config_check["passed"])
                harness.validate_case_result(case, first_result)
                independent_check = harness.validate_independent_outcome(
                    case_id,
                    first_result,
                )
                self.assertTrue(independent_check["passed"])
                self.assertEqual(
                    independent_check["matrix_source"],
                    "hard_coded_in_v2_harness_not_case_json",
                )

                if case_id == expected_case_ids[0]:
                    tampered_result = dict(first_result)
                    tampered_result["evaluation_config_sha256"] = "0" * 64
                    with self.assertRaises(
                        harness.Task20CollapseV2OracleError
                    ):
                        harness.validate_result_evaluation_config(
                            case_id,
                            tampered_result,
                            expected_sha256=expected_config_hashes[case_id],
                        )

    def test_frozen_execution_environment_is_exact_and_fail_closed(self) -> None:
        self.assertEqual(
            harness.frozen_execution_environment_record(),
            {
                "python_implementation": "CPython",
                "python_version": "3.10.19",
                "numpy_version": "2.2.6",
            },
        )
        with mock.patch.object(
            harness.platform,
            "python_version",
            return_value="3.10.18",
        ):
            with self.assertRaises(harness.Task20CollapseV2OracleError):
                harness.frozen_execution_environment_record()
        with mock.patch.object(harness.np, "__version__", "2.2.5"):
            with self.assertRaises(harness.Task20CollapseV2OracleError):
                harness.frozen_execution_environment_record()

    def test_parser_has_only_the_exact_new_output_argument(self) -> None:
        parser = harness.build_argument_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(
            option_strings,
            {"-h", "--help", "--output-directory"},
        )
        for forbidden in (
            "--manifest",
            "--checkpoint",
            "--model",
            "--dataset",
            "--split",
            "--train",
        ):
            self.assertNotIn(forbidden, option_strings)

    def test_reviewed_fileset_matches_checkpoint_authorization(self) -> None:
        self.assertEqual(
            harness.REVIEWED_PLANTED_INPUT_PATHS,
            authorization.TASK20_V2_REVIEWED_FILE_PATHS,
        )

    def test_suite_fails_before_any_work_without_review_authority(self) -> None:
        with mock.patch.object(
            harness,
            "frozen_execution_environment_record",
            return_value=dict(harness.FROZEN_EXECUTION_ENVIRONMENT),
        ), mock.patch.object(
            harness,
            "_validate_fable_review_authority",
            side_effect=harness.Task20CollapseV2OracleError(
                "review is absent"
            ),
        ), mock.patch.object(
            harness,
            "load_manifest",
            side_effect=AssertionError(
                "manifest must not load before review authority"
            ),
        ):
            with self.assertRaisesRegex(
                harness.Task20CollapseV2OracleError,
                "review is absent",
            ):
                harness.run_suite(
                    output_directory=Path("/not/reached")
                )

    def test_manifest_and_output_guards_reject_v1_or_nonexact_paths(self) -> None:
        v1_manifest = (
            harness.REPOSITORY_ROOT
            / harness.V1_CASE_PREFIX
            / "ORACLE_CASE_MANIFEST.json"
        )
        with self.assertRaises(harness.Task20CollapseV2OracleError):
            harness.load_manifest(v1_manifest)

        source_commit = "0" * 40
        exact = (
            harness.REPOSITORY_ROOT
            / harness.RESULT_PREFIX
            / source_commit
            / "attempt_0001"
        )
        self.assertEqual(
            harness._validated_output_directory(
                exact,
                source_commit=source_commit,
            ),
            exact.absolute(),
        )
        with self.assertRaises(harness.Task20CollapseV2OracleError):
            harness._validated_output_directory(
                harness.REPOSITORY_ROOT
                / harness.V1_RESULT_PREFIX
                / source_commit,
                source_commit=source_commit,
            )
        with self.assertRaises(harness.Task20CollapseV2OracleError):
            harness._validated_output_directory(
                exact.parent / "wrong_name",
                source_commit=source_commit,
            )
        with self.assertRaises(harness.Task20CollapseV2OracleError):
            harness._validated_output_directory(
                exact / "nested",
                source_commit=source_commit,
            )

    def test_output_file_writer_uses_exclusive_nofollow_creation(self) -> None:
        payload = b"abcdef"
        with mock.patch.object(
            harness.os,
            "open",
            return_value=17,
        ) as open_mock, mock.patch.object(
            harness.os,
            "write",
            side_effect=[2, 4],
        ) as write_mock, mock.patch.object(
            harness.os,
            "fsync",
        ) as fsync_mock, mock.patch.object(
            harness.os,
            "close",
        ) as close_mock:
            harness._write_new_file_exclusive(
                directory_fd=11,
                filename="result.json",
                payload=payload,
            )

        flags = open_mock.call_args.args[1]
        self.assertTrue(flags & harness.os.O_CREAT)
        self.assertTrue(flags & harness.os.O_EXCL)
        if hasattr(harness.os, "O_NOFOLLOW"):
            self.assertTrue(flags & harness.os.O_NOFOLLOW)
        self.assertEqual(open_mock.call_args.kwargs["dir_fd"], 11)
        self.assertEqual(
            write_mock.call_args_list,
            [
                mock.call(17, b"abcdef"),
                mock.call(17, b"cdef"),
            ],
        )
        fsync_mock.assert_called_once_with(17)
        close_mock.assert_called_once_with(17)

        with self.assertRaises(harness.Task20CollapseV2OracleError):
            harness._write_new_file_exclusive(
                directory_fd=11,
                filename="../escape.json",
                payload=payload,
            )


if __name__ == "__main__":
    unittest.main()
