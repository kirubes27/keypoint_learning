"""Focused pre-review contract checks for the planted-only oracle harness.

These tests are not an official oracle execution.  The candidate sources are
not yet provenance-addressable at one exact commit, so the sole evaluator test
uses an in-process provenance seam.  Production harness code has no such seam.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import eval_representation as evaluator
from keypoint_net.diagnostics import run_representation_oracles as harness


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


class PlantedOracleHarnessContractTest(unittest.TestCase):
    def test_manifest_and_all_planted_constructions_use_production_evaluator(self) -> None:
        manifest, cases = harness.load_manifest()
        self.assertEqual(
            manifest["blocked_dataset_backed_families"],
            ["scale", "translation", "yaw", "pitch"],
        )
        self.assertEqual(
            manifest["planted_threshold_semantics"]["scientific_authority"],
            "none",
        )
        self.assertFalse(
            manifest["planted_threshold_semantics"][
                "task_20_55_80_pass_fail_authority"
            ]
        )
        self.assertFalse(
            manifest["legacy_role_subset_semantics"][
                "task_20_55_80_role_subset_is_scientific_holdout"
            ]
        )
        self.assertEqual(len(cases), 16)
        case_paths = harness._case_path_by_id(manifest)
        expected_config_hashes = {
            entry["case_id"]: entry["expected_evaluation_config_sha256"]
            for entry in manifest["ordered_case_definitions"]
        }
        numeric_registry = harness._load_strict_json(
            harness.REPOSITORY_ROOT / harness.NUMERIC_REGISTRY_RELATIVE_PATH,
            name="numeric registry",
        )

        with mock.patch.object(
            evaluator,
            "_validate_production_provenance",
            _unit_provenance_validator,
        ):
            for case in cases:
                case_id = case["case_id"]
                first = harness.construct_case_bundle(
                    case,
                    case_path=case_paths[case_id],
                    source_commit="0" * 40,
                )
                second = harness.construct_case_bundle(
                    case,
                    case_path=case_paths[case_id],
                    source_commit="0" * 40,
                )
                self.assertEqual(
                    evaluator.canonical_json_bytes(first),
                    evaluator.canonical_json_bytes(second),
                )
                self.assertEqual(first["case_kind"], "planted")
                self.assertEqual(first["transform"]["family"], "roll")
                self.assertEqual(first["transform"]["physical_axis"], "world_z")
                self.assertNotIn("fit", first)
                self.assertEqual(first["provenance"]["external_files"], [])
                self.assertEqual(
                    evaluator.canonical_sha256(first["evaluation_config"]),
                    expected_config_hashes[case_id],
                )
                result = evaluator.evaluate_bundle(first)
                self.assertEqual(
                    result["evaluation_config_sha256"],
                    expected_config_hashes[case_id],
                )
                harness._validate_case_result(
                    case,
                    result,
                    numeric_registry=numeric_registry,
                )

    def test_parser_has_no_checkpoint_model_dataset_or_training_argument(self) -> None:
        parser = harness.build_argument_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(
            option_strings,
            {"-h", "--help", "--manifest", "--output-directory"},
        )
        for forbidden in (
            "--checkpoint",
            "--model",
            "--dataset",
            "--split",
            "--train",
        ):
            self.assertNotIn(forbidden, option_strings)


if __name__ == "__main__":
    unittest.main()
