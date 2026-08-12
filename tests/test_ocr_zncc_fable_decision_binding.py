"""Pure CPU tests for cross-machine Fable/evidence identity binding."""

from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_net import ocr_zncc_fable_review as fable_review
from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import ocr_zncc_training_decision as training_decision


_COMMIT = "1" * 40
_HASH = "a" * 64
_ACCESS_LINE = f"{fable_review.ACCESS_PROOF_PREFIX}{'7' * 64}"
_ACCESS_HASH = hashlib.sha256((_ACCESS_LINE + "\n").encode("ascii")).hexdigest()


def _evidence_record(logical_name: str, *, digest: str = _HASH) -> dict:
    return {
        "logical_name": logical_name,
        "review_path": f"/original/mac/path/{logical_name}.json",
        "sha256": digest,
        "size_bytes": 123,
        "content_hash_sha256": "b" * 64,
    }


def _briefing() -> tuple[dict, dict[str, dict]]:
    primary = {
        name: _evidence_record(name)
        for name in fable_review.EXPECTED_PRIMARY_LOGICAL_NAMES
    }
    direction = {
        f"stage0_direction_cell:{cell_id}": _evidence_record(
            f"stage0_direction_cell:{cell_id}"
        )
        for cell_id in fable_review.EXPECTED_DIRECTION_CELL_IDS
    }
    source_records = [
        {
            "logical_name": f"committed_source:{relative}",
            "review_path": f"/original/mac/worktree/{relative}",
            "relative_path": relative,
            "sha256": _HASH,
            "size_bytes": 123,
        }
        for relative in authorization.RUN_SOURCE_PATHS
    ]
    calibration = [
        {
            "logical_name": name,
            "review_path": f"/original/mac/reviews/{name}",
            "sha256": digest,
            "size_bytes": size,
        }
        for name, (digest, size) in fable_review.EXPECTED_CALIBRATION_EXAMPLES.items()
    ]
    document = {
        "schema_version": fable_review.BRIEFING_SCHEMA_VERSION,
        "artifact_type": "independent_fable_high_ocr_zncc_review_briefing",
        "source_commit": _COMMIT,
        "source_branch": authorization.EXPECTED_BRANCH,
        "original_task": "independent adversarial review",
        "decision_question": "is Task55 evidence sufficient?",
        "constraints": [
            f"Never read, stat, open, or otherwise touch the protected file {fable_review.PROTECTED_PATH}."
        ],
        "protected_path": fable_review.PROTECTED_PATH,
        "enumerated_files_only": True,
        "access_challenge": {
            "logical_name": "fable_access_challenge",
            "review_path": "/original/mac/review/FABLE_ACCESS_PROOF.txt",
            "sha256": _ACCESS_HASH,
            "size_bytes": len(_ACCESS_LINE) + 1,
        },
        "adversarial_checklist": list(fable_review.ADVERSARIAL_CHECKLIST),
        "committed_run_sources": source_records,
        "primary_evidence": {
            "artifacts": list(primary.values()),
            "direction_cells": list(direction.values()),
            "interpretation": "raw_current_evidence_not_a_supplied_conclusion",
        },
        "reviewer_calibration_examples": {
            "status": "historical_advisory_stale_until_directly_verified",
            "same_prompt_independent_pair": calibration[:3],
            "historical_pro_design_preflight": calibration[3:],
            "authority": "none",
        },
        "requested_deliverable": ["line-level blockers"],
        "independence": {
            "codex_draft_supplied": False,
            "codex_current_conclusion_supplied": False,
            "single_fable_call": True,
            "model_alias": "fable",
            "effort": "high",
            "permission_mode": "plan_read_only",
            "session_persistence": False,
        },
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    return document, {**primary, **direction}


class FableBriefingBindingTests(unittest.TestCase):
    def _validate(self, document: dict, primary: dict[str, dict]) -> dict:
        def source_record(path: Path, *, logical_name: str, relative_path: str | None = None):
            del path
            result = {
                "logical_name": logical_name,
                "review_path": f"/cluster/live/{relative_path or logical_name}",
                "sha256": _HASH,
                "size_bytes": 123,
            }
            if relative_path is not None:
                result["relative_path"] = relative_path
            return result

        with (
            mock.patch.object(
                authorization,
                "_source_state",
                return_value=(_COMMIT, authorization.EXPECTED_BRANCH),
            ),
            mock.patch.object(fable_review, "_record", side_effect=source_record),
        ):
            return fable_review.validate_briefing_document(
                REPO_ROOT,
                document,
                primary_identities=primary,
            )

    def test_cross_machine_paths_do_not_define_reviewed_file_identity(self) -> None:
        document, primary = _briefing()
        result = self._validate(document, primary)
        self.assertEqual(result["source_commit"], _COMMIT)

    def test_fable_briefing_primary_hash_mismatch_fails_closed(self) -> None:
        document, primary = _briefing()
        mismatched = copy.deepcopy(primary)
        mismatched["exact_gradient_audit"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(
            fable_review.OCRFableReviewError,
            "Fable/live evidence bytes differ: exact_gradient_audit",
        ):
            self._validate(document, mismatched)

    def test_codex_conclusion_flag_cannot_be_rehashed_into_valid_briefing(self) -> None:
        document, primary = _briefing()
        document["independence"]["codex_current_conclusion_supplied"] = True
        document["content_hash_sha256"] = authorization.canonical_sha256(
            {key: value for key, value in document.items() if key != "content_hash_sha256"}
        )
        with self.assertRaisesRegex(
            fable_review.OCRFableReviewError,
            "Fable independence contract differs",
        ):
            self._validate(document, primary)

    def test_access_denial_cannot_masquerade_as_substantive_review(self) -> None:
        path = mock.Mock(spec=Path)
        path.is_file.return_value = True
        path.is_symlink.return_value = False
        path.stat.return_value.st_size = 5000
        path.read_text.return_value = (
            f"{_ACCESS_LINE}\n{fable_review.ACCESS_STATUS_VERIFIED}\n"
            f"{fable_review.ACCESS_COUNT_VERIFIED}\n"
            + "I could not open any of the 36 evidence files. " * 100
        )
        self.assertFalse(
            fable_review._substantive_raw(  # noqa: SLF001
                path, access_challenge_sha256=_ACCESS_HASH
            )
        )
        path.read_text.return_value = (
            "Independent review of the supplied packet.\n"
            f"{_ACCESS_LINE}\n{fable_review.ACCESS_STATUS_VERIFIED}\n"
            f"{fable_review.ACCESS_COUNT_VERIFIED}\n"
            + "Line-anchored independent review evidence. " * 100
        )
        self.assertTrue(
            fable_review._substantive_raw(  # noqa: SLF001
                path, access_challenge_sha256=_ACCESS_HASH
            )
        )

    def test_deep_decision_accepts_only_complete_clean_checkout_receipt(self) -> None:
        checkout = {
            "commit": _COMMIT,
            "branch": authorization.EXPECTED_BRANCH,
            "authorized_base_commit": authorization.AUTHORIZED_BASE_COMMIT,
            "authorized_base_is_ancestor": True,
            "complete_status": "",
        }
        training_decision._validate_checkout(  # noqa: SLF001
            checkout,
            commit=_COMMIT,
            branch=authorization.EXPECTED_BRANCH,
            name="Stage0",
        )
        stale_schema = dict(checkout)
        stale_schema.pop("complete_status")
        stale_schema["tracked_status"] = ""
        with self.assertRaisesRegex(
            training_decision.OCRTask55AuthorizationError,
            "complete checkout was dirty",
        ):
            training_decision._validate_checkout(  # noqa: SLF001
                stale_schema,
                commit=_COMMIT,
                branch=authorization.EXPECTED_BRANCH,
                name="Stage0",
            )


if __name__ == "__main__":
    unittest.main()
