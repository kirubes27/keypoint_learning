"""CPU-only contract tests for the saved-checkpoint replay registry.

These tests validate registry text and Git provenance only.  They never stat,
open, hash, import, or load any checkpoint candidate.
"""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import representation_checkpoint_replay as replay


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / replay.REGISTRY_REPO_RELATIVE_PATH


def _registry_document() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _encoded_with_recomputed_content_hash(document: dict) -> bytes:
    normalized = copy.deepcopy(document)
    normalized["content_hash_sha256"] = replay.replay_registry_content_hash(
        normalized
    )
    return (
        json.dumps(
            normalized,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class RepresentationCheckpointReplayTests(unittest.TestCase):
    @staticmethod
    def _synthetic_evaluator_result() -> dict:
        def metric_record(k: int) -> dict:
            return {
                "k": k,
                "model_mse": {
                    "mean": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0,
                    "n": 180,
                },
            }

        return {
            "operator": {
                "learned_A": [[1.0, 0.0], [0.0, 1.0]],
                "learned_b": [0.0, 0.0],
                "determinant_A": 1.0,
                "singular_values": [1.0, 1.0],
                "proper_rotation_angle_deg": 6.0,
            },
            "rollout": {
                "full_corpus_identity_normalized_auc": {
                    "horizons": [
                        metric_record(k)
                        for k in range(1, 60)
                    ],
                },
                "closure": metric_record(60),
            },
        }

    def test_exact_registry_payload_validates_without_checkpoint_touch(self) -> None:
        data = _encoded_with_recomputed_content_hash(_registry_document())
        with mock.patch.object(
            replay,
            "_hash_checkpoint_candidate",
            side_effect=AssertionError("checkpoint path was touched"),
        ) as checkpoint_hash:
            receipt = replay.validate_replay_registry_bytes(data)
        checkpoint_hash.assert_not_called()
        self.assertEqual(
            receipt.content_hash_sha256,
            replay.EXPECTED_REGISTRY_CONTENT_HASH_SHA256,
        )
        self.assertEqual(
            tuple(item.task_id for item in receipt.checkpoint_candidates),
            (20, 55, 80),
        )
        self.assertEqual(
            tuple(item.role for item in receipt.checkpoint_candidates),
            (
                "collapsed_negative_control",
                "clean_baseline_replay_only",
                "assisted_baseline_replay_only",
            ),
        )

    def test_checked_in_registry_claim_matches_canonical_payload(self) -> None:
        document = _registry_document()
        self.assertEqual(
            document["content_hash_sha256"],
            replay.replay_registry_content_hash(document),
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            replay.ReplayRegistryError,
            "duplicate JSON object key",
        ):
            replay.validate_replay_registry_bytes(
                b'{"schema_version":"a","schema_version":"b"}'
            )

    def test_nonfinite_json_constant_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            replay.ReplayRegistryError,
            "non-standard/non-finite",
        ):
            replay.validate_replay_registry_bytes(b'{"value":NaN}')

    def test_wrong_content_hash_is_rejected_before_semantic_use(self) -> None:
        data = _encoded_with_recomputed_content_hash(_registry_document())
        document = json.loads(data)
        document["content_hash_sha256"] = "0" * 64
        corrupted = json.dumps(document).encode("utf-8")
        with self.assertRaisesRegex(
            replay.ReplayRegistryError,
            "content hash mismatch",
        ):
            replay.validate_replay_registry_bytes(corrupted)

    def test_task_role_hash_and_path_are_frozen(self) -> None:
        mutations = (
            ("role", "not_a_negative_control"),
            ("checkpoint.expected_sha256", "0" * 64),
            (
                "checkpoint.candidate_absolute_path",
                "/tmp/unregistered-checkpoint.pt",
            ),
        )
        for field_name, replacement in mutations:
            with self.subTest(field_name=field_name):
                document = _registry_document()
                fixture = document["fixtures"][0]
                if field_name.startswith("checkpoint."):
                    fixture["checkpoint"][field_name.split(".", 1)[1]] = replacement
                else:
                    fixture[field_name] = replacement
                data = _encoded_with_recomputed_content_hash(document)
                with self.assertRaisesRegex(
                    replay.ReplayRegistryError,
                    "does not match the frozen replay registry",
                ):
                    replay.validate_replay_registry_bytes(data)

    def test_replay_tolerance_and_stop_conditions_are_frozen(self) -> None:
        mutations = (
            (
                ("replay_rule", "absolute_difference_max"),
                0.1,
            ),
            (
                ("replay_rule", "relative_difference_max"),
                0.1,
            ),
            (
                ("execution_boundary", "hash_mismatch_action"),
                "continue_anyway",
            ),
            (
                (
                    "execution_boundary",
                    "training_or_weight_update_authorized",
                ),
                True,
            ),
            (
                ("execution_boundary", "selection_use_authorized"),
                True,
            ),
            (
                (
                    "execution_boundary",
                    "official_replay_requires_fable_reviewed_execution_commit",
                ),
                False,
            ),
        )
        for path, replacement in mutations:
            with self.subTest(path=path):
                document = _registry_document()
                document[path[0]][path[1]] = replacement
                data = _encoded_with_recomputed_content_hash(document)
                with self.assertRaisesRegex(
                    replay.ReplayRegistryError,
                    "does not match the frozen replay registry",
                ):
                    replay.validate_replay_registry_bytes(data)

    def test_all_production_pointers_resolve_against_actual_synthetic_result(
        self,
    ) -> None:
        result = self._synthetic_evaluator_result()
        receipt = replay.validate_production_pointer_contract(result)
        self.assertEqual(receipt["resolved_value_count"], 245)
        self.assertEqual(receipt["intermediate_horizon_count"], 59)
        self.assertEqual(receipt["closure_k"], 60)
        self.assertEqual(
            receipt["status"],
            "all_frozen_production_pointers_resolved",
        )

    def test_missing_evaluator_field_fails_pointer_contract(self) -> None:
        result = self._synthetic_evaluator_result()
        del result["operator"]["singular_values"]
        with self.assertRaisesRegex(
            replay.ReplayRegistryError,
            "JSON pointer does not resolve",
        ):
            replay.validate_production_pointer_contract(result)

    def test_plain_or_forged_provenance_receipt_cannot_authorize_touch(self) -> None:
        registry = replay.validate_replay_registry_bytes(
            _encoded_with_recomputed_content_hash(_registry_document())
        )
        forged = replay.VerifiedRegistryProvenanceReceipt(
            source_commit="0" * 40,
            repository_root=str(REPO_ROOT),
            registry_absolute_path=str(REGISTRY_PATH),
            registry_file_sha256=registry.registry_file_sha256,
            registry_content_hash_sha256=registry.content_hash_sha256,
            git_blob_oid="0" * 40,
            source_commit_verified=True,
        )
        with mock.patch.object(
            replay,
            "_hash_checkpoint_candidate",
            side_effect=AssertionError("checkpoint path was touched"),
        ) as checkpoint_hash:
            with self.assertRaisesRegex(
                replay.ReplayRegistryError,
                "requires a verified registry provenance receipt",
            ):
                replay._require_preflight_authorization(registry, forged)
        checkpoint_hash.assert_not_called()

    def test_geometry_authorizes_hash_preflight_but_not_checkpoint_loading(
        self,
    ) -> None:
        registry = replay.load_and_validate_replay_registry(REGISTRY_PATH)
        source_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
        provenance = replay.VerifiedRegistryProvenanceReceipt(
            source_commit=source_commit,
            repository_root=str(REPO_ROOT),
            registry_absolute_path=registry.registry_absolute_path,
            registry_file_sha256=registry.registry_file_sha256,
            registry_content_hash_sha256=registry.content_hash_sha256,
            git_blob_oid="0" * 40,
            source_commit_verified=True,
            _capability=replay._VERIFIED_PROVENANCE_CAPABILITY,
        )
        with mock.patch.object(
            replay,
            "_hash_checkpoint_candidate",
            side_effect=lambda candidate: {
                "fixture_id": candidate.fixture_id,
                "task_id": candidate.task_id,
                "verification_status": "test_double_no_checkpoint_touch",
            },
        ) as checkpoint_hash:
            result = replay.preflight_checkpoint_candidates(
                registry,
                provenance=provenance,
            )
        self.assertEqual(checkpoint_hash.call_count, 3)
        self.assertTrue(result["checkpoint_hash_preflight_passed"])
        self.assertFalse(result["checkpoint_load_authorized_by_this_receipt"])
        self.assertFalse(result["training_or_weight_update_authorized"])
        self.assertFalse(result["selection_use_authorized"])

    def test_mutated_geometry_binding_fails_before_checkpoint_touch(self) -> None:
        registry = replay.load_and_validate_replay_registry(REGISTRY_PATH)
        source_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
        provenance = replay.VerifiedRegistryProvenanceReceipt(
            source_commit=source_commit,
            repository_root=str(REPO_ROOT),
            registry_absolute_path=registry.registry_absolute_path,
            registry_file_sha256=registry.registry_file_sha256,
            registry_content_hash_sha256=registry.content_hash_sha256,
            git_blob_oid="0" * 40,
            source_commit_verified=True,
            _capability=replay._VERIFIED_PROVENANCE_CAPABILITY,
        )
        original_reader = replay._regular_file_bytes

        def mutated_reader(path: Path, *, context: str) -> tuple[Path, bytes]:
            resolved, data = original_reader(path, context=context)
            if str(path).endswith(
                replay.HAMMER_ROLL_GEOMETRY_BINDING_REPO_RELATIVE_PATH
            ):
                return resolved, data + b" "
            return resolved, data

        with mock.patch.object(
            replay,
            "_regular_file_bytes",
            side_effect=mutated_reader,
        ), mock.patch.object(
            replay,
            "_hash_checkpoint_candidate",
            side_effect=AssertionError("checkpoint path was touched"),
        ) as checkpoint_hash:
            with self.assertRaisesRegex(
                replay.ReplayRegistryError,
                "geometry binding file hash mismatch",
            ):
                replay.preflight_checkpoint_candidates(
                    registry,
                    provenance=provenance,
                )
        checkpoint_hash.assert_not_called()

    def test_committed_registry_verifier_reads_git_not_checkpoints(self) -> None:
        persistent_root = Path(
            tempfile.mkdtemp(prefix="replay-registry-provenance-test-")
        ).resolve()
        repo = persistent_root / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.name", "Replay Registry Test")
        _git(
            repo,
            "config",
            "user.email",
            "replay-registry-test@example.invalid",
        )
        path = repo / replay.REGISTRY_REPO_RELATIVE_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(
            _encoded_with_recomputed_content_hash(_registry_document())
        )
        _git(repo, "add", replay.REGISTRY_REPO_RELATIVE_PATH)
        _git(repo, "commit", "-q", "-m", "commit replay registry")
        source_commit = _git(repo, "rev-parse", "HEAD")
        registry = replay.load_and_validate_replay_registry(path)

        with mock.patch.object(
            replay,
            "_hash_checkpoint_candidate",
            side_effect=AssertionError("checkpoint path was touched"),
        ) as checkpoint_hash:
            receipt = replay._verify_committed_registry_for_repo(
                registry,
                source_commit=source_commit,
                repo_root=repo,
            )
        checkpoint_hash.assert_not_called()
        self.assertTrue(receipt.source_commit_verified)
        self.assertEqual(receipt.source_commit, source_commit)
        self.assertEqual(
            receipt.registry_file_sha256,
            registry.registry_file_sha256,
        )
        print(f"REPLAY_REGISTRY_TEST_TEMP_ROOT={persistent_root}", flush=True)


if __name__ == "__main__":
    unittest.main()
