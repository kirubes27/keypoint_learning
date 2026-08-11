from __future__ import annotations

import copy
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import ocr_zncc_outcome_evaluator as outcome
from keypoint_net import ocr_zncc_task80_training_authorization as task80


SOURCE_COMMIT = "a" * 40
SOURCE_BRANCH = task80.EXPECTED_BRANCH
MANIFEST_RECORD = {
    "absolute_path": "/external/MANIFEST.json",
    "sha256": "b" * 64,
    "size_bytes": 100,
    "content_hash_sha256": "c" * 64,
    "validation_index": {
        "absolute_path": "/repo/validation.json",
        "sha256": "d" * 64,
        "size_bytes": 200,
    },
}
RESULT_RECORDS = [
    {
        "absolute_path": f"/external/result_{index}.json",
        "sha256": str(index + 1) * 64,
        "size_bytes": 1000 + index,
        "content_hash_sha256": str(index + 2) * 64,
        "cell_id": f"cell_{index}",
    }
    for index in range(6)
]
DECISION_SOURCE_RECORD = {
    "relative_path": "keypoint_net/ocr_zncc_task55_outcome_decision.py",
    "absolute_path": "/repo/keypoint_net/ocr_zncc_task55_outcome_decision.py",
    "sha256": "e" * 64,
    "size_bytes": 12345,
}


def _recomputed_pass() -> dict:
    return {
        "outcome": "task55_pass_candidate_for_separate_task80_authorization",
        "task55_pass": True,
        "task80_training_authorized_by_this_artifact": False,
        "criterion_results": {
            "selected_drift_ratio_at_most_0p80_in_at_least_two_of_three": True,
            "final_epoch_1000_drift_nonregression_in_at_least_two_of_three": True,
            "selected_ocr_positive_roll_sign_in_all_three": True,
            "selected_ocr_absolute_angle_error_nonworse_in_at_least_two_of_three": True,
            "selected_ocr_active_on_object_count_nonlower_in_all_three": True,
            "no_provenance_nonfinite_or_reflection_blocker": True,
        },
        "support_counts": {
            "paired_seed_count": 3,
            "selected_drift_ratio_at_most_0p80": 2,
            "final_epoch_1000_drift_nonregression": 2,
            "selected_absolute_angle_error_nonworse": 2,
        },
        "critical_blockers": [],
        "pairs": [{"seed": seed} for seed in (42, 43, 44)],
        "final_checkpoint_diagnostics_not_used_for_decision": {
            "ocr_positive_roll_sign_in_all_three": False,
            "ocr_absolute_angle_error_nonworse_support_count": 0,
            "ocr_active_on_object_count_nonlower_in_all_three": False,
        },
    }


def _decision_document(recomputed: dict | None = None) -> dict:
    payload = recomputed or _recomputed_pass()
    document = {
        "schema_version": "ocr_zncc_task55_paired_outcome_decision.v1",
        "artifact_type": "frozen_task55_paired_ocr_zncc_outcome_decision",
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "decision_rule": copy.deepcopy(task80.EXPECTED_TASK55_DECISION_RULE),
        "experiment_manifest": copy.deepcopy(MANIFEST_RECORD),
        "cell_result_records": copy.deepcopy(RESULT_RECORDS),
        "committed_decision_source": copy.deepcopy(DECISION_SOURCE_RECORD),
        **copy.deepcopy(payload),
        "next_phase_started": False,
    }
    document["content_hash_sha256"] = outcome.canonical_sha256(document)
    return document


def _task55_training_authorization() -> dict:
    document = {
        "schema_version": "ocr_zncc_task55_training_authorization.v1",
        "artifact_type": "bounded_task55_ocr_zncc_training_authorization",
        "status": "authorize_task55_matched_experiment",
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "coefficient": 0.314,
        "coefficient_rule": {"target": "ten_percent_total_gradient"},
        "ocr_zncc_config": {
            "peak_margin_exclusion_radius_cells": 1,
            "patch_size": 7,
        },
        "execution_boundary": {
            "authorized_recipe": "task55_clean",
            "matched_arms": ["control", "ocr_zncc"],
            "seeds": [42, 43, 44],
            "epochs": 1000,
            "task80_authorized": False,
            "other_objects_or_operators_authorized": False,
            "test_split_authorized": False,
        },
    }
    document["content_hash_sha256"] = outcome.canonical_sha256(document)
    return document


def _task55_deep_validation(*, radius: int = 1) -> dict:
    return {
        "authorization_valid": True,
        "status": "authorize_task55_matched_experiment",
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "coefficient": 0.314,
        "ocr_zncc_config": {
            "peak_margin_exclusion_radius_cells": radius,
            "patch_size": 7,
        },
    }


def _source_binding() -> dict:
    return {
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "research_base_commit": task80.RESEARCH_BASE_COMMIT,
        "research_base_is_ancestor": True,
        "full_status_clean": True,
        "committed_sources": {},
    }


def _live_prerequisite_validation() -> dict:
    return {
        "coefficient": 0.314,
        "coefficient_rule": {"target": "ten_percent_total_gradient"},
        "ocr_zncc_config": {
            "peak_margin_exclusion_radius_cells": 1,
            "patch_size": 7,
        },
        "task55_recomputation": {
            "outcome": "task55_pass_candidate_for_separate_task80_authorization",
            "task55_pass": True,
            "critical_blockers": [],
        },
    }


def _task80_document() -> dict:
    document = {
        "schema_version": task80.SCHEMA_VERSION,
        "artifact_type": "bounded_task80_ocr_zncc_training_authorization",
        "status": task80.STATUS,
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "research_base": {
            "commit": task80.RESEARCH_BASE_COMMIT,
            "is_ancestor_of_source_commit": True,
        },
        "decision": task80.TASK80_DECISION_TEXT,
        "coefficient": 0.314,
        "coefficient_rule": {"target": "ten_percent_total_gradient"},
        "ocr_zncc_config": {
            "peak_margin_exclusion_radius_cells": 1,
            "patch_size": 7,
        },
        "task55_prerequisites": {"placeholder": True},
        "source_binding": _source_binding(),
        "execution_boundary": copy.deepcopy(
            task80.EXPECTED_TASK80_EXECUTION_BOUNDARY
        ),
        "statistical_scope": copy.deepcopy(task80.EXPECTED_STATISTICAL_SCOPE),
    }
    document["content_hash_sha256"] = outcome.canonical_sha256(document)
    return document


class Task80AuthorizationTests(unittest.TestCase):
    def _validate_decision(self, document: dict, recomputed: dict) -> None:
        task80._validate_supplied_task55_decision(  # noqa: SLF001
            document,
            source_commit=SOURCE_COMMIT,
            source_branch=SOURCE_BRANCH,
            recomputed=recomputed,
            result_records=RESULT_RECORDS,
            manifest_record=MANIFEST_RECORD,
            decision_source_record=DECISION_SOURCE_RECORD,
        )

    def test_exact_live_recomputed_pass_is_accepted(self) -> None:
        recomputed = _recomputed_pass()
        self._validate_decision(_decision_document(recomputed), recomputed)

    def test_self_hashed_fabricated_pass_cannot_override_live_failure(self) -> None:
        supplied = _decision_document(_recomputed_pass())
        live = _recomputed_pass()
        live["outcome"] = "task55_fail_stop_before_task80"
        live["task55_pass"] = False
        live["criterion_results"][
            "selected_drift_ratio_at_most_0p80_in_at_least_two_of_three"
        ] = False
        with self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "disagrees with live recomputation",
        ):
            self._validate_decision(supplied, live)

    def test_rehashed_mutated_cell_binding_is_rejected(self) -> None:
        recomputed = _recomputed_pass()
        supplied = _decision_document(recomputed)
        supplied["cell_result_records"][0]["sha256"] = "f" * 64
        supplied["content_hash_sha256"] = outcome.canonical_sha256({
            key: value for key, value in supplied.items()
            if key != "content_hash_sha256"
        })
        with self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "cell-result bindings differ",
        ):
            self._validate_decision(supplied, recomputed)

    def test_rehashed_extra_authorization_field_is_rejected(self) -> None:
        recomputed = _recomputed_pass()
        supplied = _decision_document(recomputed)
        supplied["fabricated_task80_authorized"] = True
        supplied["content_hash_sha256"] = outcome.canonical_sha256({
            key: value for key, value in supplied.items()
            if key != "content_hash_sha256"
        })
        with self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "field set differs",
        ):
            self._validate_decision(supplied, recomputed)

    def test_live_blocker_cannot_authorize_task80(self) -> None:
        live = _recomputed_pass()
        live["critical_blockers"] = [{"reason": "operator_is_improper_or_reflection"}]
        live["criterion_results"][
            "no_provenance_nonfinite_or_reflection_blocker"
        ] = False
        live["task55_pass"] = False
        live["outcome"] = "stop_invalid_or_critical_evaluator_evidence"
        supplied = _decision_document(live)
        with self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "does not pass without blockers",
        ):
            self._validate_decision(supplied, live)

    def test_task55_coefficient_and_radius_are_inherited(self) -> None:
        document = _task55_training_authorization()
        with mock.patch.object(
            task80.task55_training_decision,
            "validate_authorization_document",
            return_value=_task55_deep_validation(),
        ):
            coefficient, inherited = task80._validate_task55_training_authorization(  # noqa: SLF001
                document,
                source_commit=SOURCE_COMMIT,
                source_branch=SOURCE_BRANCH,
                repo_root=Path("/repo"),
            )
        self.assertEqual(0.314, coefficient)
        self.assertEqual(1, inherited["ocr_zncc_config"][
            "peak_margin_exclusion_radius_cells"
        ])

    def test_radius_zero_task55_authorization_is_rejected(self) -> None:
        document = _task55_training_authorization()
        with mock.patch.object(
            task80.task55_training_decision,
            "validate_authorization_document",
            return_value=_task55_deep_validation(radius=0),
        ), self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "configuration differs",
        ):
            task80._validate_task55_training_authorization(  # noqa: SLF001
                document,
                source_commit=SOURCE_COMMIT,
                source_branch=SOURCE_BRANCH,
                repo_root=Path("/repo"),
            )

    def test_repo_bound_task55_authorization_requires_deep_evidence(self) -> None:
        document = _task55_training_authorization()
        with mock.patch.object(
            task80.task55_training_decision,
            "validate_authorization_document",
            side_effect=ValueError("primary evidence differs"),
        ), self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "deep training authorization failed",
        ):
            task80._validate_task55_training_authorization(  # noqa: SLF001
                document,
                source_commit=SOURCE_COMMIT,
                source_branch=SOURCE_BRANCH,
                repo_root=Path("/repo"),
            )

    def test_deep_task55_validator_rechecks_primary_gate_inputs(self) -> None:
        document = _task55_training_authorization()
        with mock.patch.object(
            task80.task55_training_decision,
            "validate_authorization_document",
            return_value=_task55_deep_validation(),
        ) as validator:
            coefficient, inherited = task80._validate_task55_training_authorization(  # noqa: SLF001
                document,
                source_commit=SOURCE_COMMIT,
                source_branch=SOURCE_BRANCH,
                repo_root=Path("/repo"),
            )
        validator.assert_called_once_with(Path("/repo"), document)
        self.assertEqual(0.314, coefficient)
        self.assertEqual(1, inherited["ocr_zncc_config"][
            "peak_margin_exclusion_radius_cells"
        ])

    def test_validate_authorization_document_rechecks_live_inheritance(self) -> None:
        document = _task80_document()
        with mock.patch.object(
            task80, "_validate_source", return_value=_source_binding()
        ), mock.patch.object(
            task80,
            "_revalidate_task55_prerequisites",
            return_value=_live_prerequisite_validation(),
        ):
            receipt = task80.validate_authorization_document(Path.cwd(), document)
        self.assertTrue(receipt["authorization_valid"])
        self.assertEqual(0.314, receipt["coefficient"])
        self.assertTrue(receipt["task55_pass"])

    def test_self_hashed_task80_coefficient_fabrication_fails_live_check(self) -> None:
        document = _task80_document()
        document["coefficient"] = 0.5
        document["content_hash_sha256"] = outcome.canonical_sha256({
            key: value for key, value in document.items()
            if key != "content_hash_sha256"
        })
        with mock.patch.object(
            task80, "_validate_source", return_value=_source_binding()
        ), mock.patch.object(
            task80,
            "_revalidate_task55_prerequisites",
            return_value=_live_prerequisite_validation(),
        ), self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "coefficient differs from live Task55",
        ):
            task80.validate_authorization_document(Path.cwd(), document)

    def test_self_hashed_task80_boundary_expansion_fails(self) -> None:
        document = _task80_document()
        document["execution_boundary"]["other_objects_or_operators_authorized"] = True
        document["content_hash_sha256"] = outcome.canonical_sha256({
            key: value for key, value in document.items()
            if key != "content_hash_sha256"
        })
        with mock.patch.object(
            task80, "_validate_source", return_value=_source_binding()
        ), mock.patch.object(
            task80,
            "_revalidate_task55_prerequisites",
            return_value=_live_prerequisite_validation(),
        ), self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "execution boundary differs",
        ):
            task80.validate_authorization_document(Path.cwd(), document)

    def test_validate_source_rejects_untracked_worktree_content(self) -> None:
        def fake_git(_root, *arguments, check=True):
            del check
            if arguments == ("rev-parse", "HEAD"):
                return subprocess.CompletedProcess([], 0, stdout=SOURCE_COMMIT + "\n", stderr="")
            if arguments == ("branch", "--show-current"):
                return subprocess.CompletedProcess([], 0, stdout=SOURCE_BRANCH + "\n", stderr="")
            if arguments == ("status", "--porcelain"):
                return subprocess.CompletedProcess([], 0, stdout="?? rogue.txt\n", stderr="")
            raise AssertionError(arguments)

        with mock.patch.object(task80, "_git", side_effect=fake_git), self.assertRaisesRegex(
            task80.OCRTask80AuthorizationError,
            "not completely clean",
        ):
            task80._validate_source(  # noqa: SLF001
                Path("/repo"),
                expected_commit=SOURCE_COMMIT,
                expected_branch=SOURCE_BRANCH,
            )


if __name__ == "__main__":
    unittest.main()
