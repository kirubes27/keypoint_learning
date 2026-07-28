"""Contract tests for representation-evaluation provenance.

Every test repository is created under one persistent ``mkdtemp`` root.  The
root is printed once and is deliberately never deleted so failed fixtures
remain available for inspection.
"""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from keypoint_net import representation_evaluation_provenance as provenance


_DEFAULT_CHECKPOINT_RECEIPT = object()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _record_for_role(bundle: dict[str, Any], role: str) -> dict[str, Any]:
    for record in bundle["committed_files"]:
        if record["role"] == role:
            return record
    raise AssertionError(f"missing committed fixture role {role}")


def _external_record_for_role(
    bundle: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    for record in bundle["external_files"]:
        if record["role"] == role:
            return record
    raise AssertionError(f"missing external fixture role {role}")


def _checkpoint_load_receipt(fixture: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _external_record_for_role(
        fixture["bundle"],
        "checkpoint",
    )
    return {
        **checkpoint,
        "task_id": 20,
        "fixture_id": "unit_task20_checkpoint",
        "source_commit": fixture["bundle"]["source_commit"],
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }


class RepresentationEvaluationProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.persistent_root = Path(
            tempfile.mkdtemp(prefix="representation-provenance-tests-")
        ).resolve()
        print(
            f"PROVENANCE_TEST_TEMP_ROOT={cls.persistent_root}",
            flush=True,
        )

    def _fixture(
        self,
        *,
        case_kind: str = "planted",
        fit_from_pairs: bool = False,
        symlink_role: str | None = None,
    ) -> dict[str, Any]:
        repo = Path(
            tempfile.mkdtemp(
                prefix=f"repo-{case_kind}-",
                dir=self.persistent_root,
            )
        ).resolve()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.name", "Provenance Test")
        _git(repo, "config", "user.email", "provenance-test@example.invalid")

        committed_roles, external_roles = provenance.required_role_profile(
            case_kind,
            fit_from_pairs=fit_from_pairs,
        )
        paths_by_role: dict[str, Path] = {}
        relative_by_role: dict[str, str] = {}
        for role in sorted(committed_roles):
            relative = (
                provenance.CORE_ROLE_PATHS.get(role)
                or provenance.FIXED_ROLE_PATHS.get(role)
                or (
                    provenance.ROLE_PATH_PREFIXES[role] + f"{role}.json"
                    if role in provenance.ROLE_PATH_PREFIXES
                    else f"evidence/{role}.json"
                )
            )
            path = repo.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            if role == symlink_role:
                target = path.with_name(f"{path.name}.target")
                target.write_bytes(f"target for {role}\n".encode("utf-8"))
                os.symlink(target.name, path)
            else:
                path.write_bytes(f"fixture bytes for {role}\n".encode("utf-8"))
            paths_by_role[role] = path
            relative_by_role[role] = relative

        _git(repo, "add", "--all")
        _git(repo, "commit", "-q", "-m", "fixture provenance sources")
        source_commit = _git(repo, "rev-parse", "HEAD")

        committed_files = [
            {
                "role": role,
                "repo_relative_path": relative_by_role[role],
                "file_sha256": _sha256(paths_by_role[role]),
            }
            for role in sorted(committed_roles)
        ]

        external_root = Path(
            tempfile.mkdtemp(
                prefix=f"external-{case_kind}-",
                dir=self.persistent_root,
            )
        ).resolve()
        external_paths: dict[str, Path] = {}
        external_files: list[dict[str, Any]] = []
        for role in sorted(external_roles):
            path = external_root / f"{role}.bin"
            payload = f"non-model test bytes for {role}\n".encode("utf-8")
            path.write_bytes(payload)
            external_paths[role] = path
            external_files.append(
                {
                    "role": role,
                    "absolute_path": str(path),
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )

        bundle = {
            "schema_version": provenance.PROVENANCE_SCHEMA_VERSION,
            "source_commit": source_commit,
            "committed_files": committed_files,
            "external_files": external_files,
        }
        loaded_sources = {
            role: provenance.capture_loaded_source_digest(
                role,
                paths_by_role[role].absolute(),
            )
            for role in provenance.LOADED_SOURCE_ROLES
        }
        provenance_loaded_source = provenance.capture_loaded_source_digest(
            "provenance_source",
            paths_by_role["provenance_source"].absolute(),
        )
        return {
            "repo": repo,
            "bundle": bundle,
            "case_kind": case_kind,
            "fit_from_pairs": fit_from_pairs,
            "paths_by_role": paths_by_role,
            "external_paths": external_paths,
            "loaded_sources": loaded_sources,
            "provenance_loaded_source": provenance_loaded_source,
        }

    def _validate(
        self,
        fixture: dict[str, Any],
        *,
        checkpoint_receipt: object = _DEFAULT_CHECKPOINT_RECEIPT,
    ) -> dict[str, Any]:
        if checkpoint_receipt is _DEFAULT_CHECKPOINT_RECEIPT:
            checkpoint_receipt = (
                _checkpoint_load_receipt(fixture)
                if fixture["case_kind"] == "checkpoint"
                else None
            )
        return provenance._validate_evaluation_provenance_for_repo(
            fixture["bundle"],
            case_kind=fixture["case_kind"],
            fit_from_pairs=fixture["fit_from_pairs"],
            loaded_source_digests=fixture["loaded_sources"],
            repo_root=fixture["repo"],
            provenance_loaded_source_digest=fixture[
                "provenance_loaded_source"
            ],
            checkpoint_provenance_load_receipt=checkpoint_receipt,
        )

    def test_valid_full_existing_commit_and_planted_profile(self) -> None:
        fixture = self._fixture()
        result = self._validate(fixture)
        self.assertTrue(result["source_commit_verified"])
        self.assertEqual(result["source_commit"], fixture["bundle"]["source_commit"])
        self.assertEqual(result["case_kind"], "planted")
        self.assertEqual(result["external_files"], [])

    def test_deadbee_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["source_commit"] = "deadbee"
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "full lowercase 40-hex",
        ):
            self._validate(fixture)

    def test_nonexistent_40_hex_commit_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["source_commit"] = "0" * 40
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "not an existing commit",
        ):
            self._validate(fixture)

    def test_tree_object_is_not_a_commit(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["source_commit"] = _git(
            fixture["repo"],
            "rev-parse",
            "HEAD^{tree}",
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "not an existing commit",
        ):
            self._validate(fixture)

    def test_blob_object_is_not_a_commit(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["source_commit"] = _git(
            fixture["repo"],
            "rev-parse",
            "HEAD:keypoint_net/eval_representation.py",
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "not an existing commit",
        ):
            self._validate(fixture)

    def test_core_role_wrong_path_is_rejected(self) -> None:
        fixture = self._fixture()
        record = _record_for_role(fixture["bundle"], "evaluator_source")
        replacement = fixture["paths_by_role"]["split_adapter_source"]
        record["repo_relative_path"] = provenance.CORE_ROLE_PATHS[
            "split_adapter_source"
        ]
        record["file_sha256"] = _sha256(replacement)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "fixed role evaluator_source must use",
        ):
            self._validate(fixture)

    def test_repo_path_traversal_is_rejected(self) -> None:
        fixture = self._fixture()
        record = _record_for_role(
            fixture["bundle"],
            "planted_case_definition",
        )
        record["repo_relative_path"] = "../outside.json"
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "unsafe repo-relative path",
        ):
            self._validate(fixture)

    def test_repo_dot_path_is_rejected(self) -> None:
        fixture = self._fixture()
        record = _record_for_role(
            fixture["bundle"],
            "planted_case_definition",
        )
        record["repo_relative_path"] = "."
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "unsafe repo-relative path",
        ):
            self._validate(fixture)

    def test_committed_symlink_is_rejected(self) -> None:
        fixture = self._fixture(symlink_role="planted_case_definition")
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "path contains a symlink",
        ):
            self._validate(fixture)

    def test_duplicate_role_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["committed_files"].append(
            copy.deepcopy(fixture["bundle"]["committed_files"][0])
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "duplicate committed file role",
        ):
            self._validate(fixture)

    def test_duplicate_resolved_path_is_rejected(self) -> None:
        fixture = self._fixture(case_kind="dataset", fit_from_pairs=True)
        source = _record_for_role(
            fixture["bundle"],
            "evaluation_pair_artifact",
        )
        target = _record_for_role(
            fixture["bundle"],
            "fit_pair_artifact",
        )
        target["repo_relative_path"] = source["repo_relative_path"]
        target["file_sha256"] = source["file_sha256"]
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "duplicate committed file path",
        ):
            self._validate(fixture)

    def test_unknown_role_is_rejected(self) -> None:
        fixture = self._fixture()
        record = _record_for_role(
            fixture["bundle"],
            "planted_case_definition",
        )
        record["role"] = "unknown_substitution"
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "committed role profile mismatch.*unknown_substitution",
        ):
            self._validate(fixture)

    def test_missing_mandatory_role_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["bundle"]["committed_files"] = [
            record
            for record in fixture["bundle"]["committed_files"]
            if record["role"] != "oracle_case_manifest"
        ]
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "committed role profile mismatch.*oracle_case_manifest",
        ):
            self._validate(fixture)

    def test_working_mutation_fails_even_if_claim_and_loaded_digest_are_resealed(
        self,
    ) -> None:
        fixture = self._fixture()
        path = fixture["paths_by_role"]["evaluator_source"]
        path.write_bytes(b"mutated evaluator bytes after commit\n")
        new_digest = _sha256(path)
        _record_for_role(
            fixture["bundle"],
            "evaluator_source",
        )["file_sha256"] = new_digest
        fixture["loaded_sources"]["evaluator_source"] = {
            "absolute_path": str(path.resolve()),
            "sha256": new_digest,
        }
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "working bytes differ from Git blob",
        ):
            self._validate(fixture)

    def test_untracked_working_file_cannot_substitute_for_missing_git_path(
        self,
    ) -> None:
        fixture = self._fixture()
        relative = (
            provenance.ROLE_PATH_PREFIXES["planted_case_definition"]
            + "untracked_definition.json"
        )
        path = fixture["repo"].joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"untracked definition\n")
        record = _record_for_role(
            fixture["bundle"],
            "planted_case_definition",
        )
        record["repo_relative_path"] = relative
        record["file_sha256"] = _sha256(path)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "absent or ambiguous",
        ):
            self._validate(fixture)

    def test_planted_definition_role_substitution_is_rejected(self) -> None:
        fixture = self._fixture()
        replacement = fixture["paths_by_role"][
            "calibration_fixture_manifest"
        ]
        record = _record_for_role(
            fixture["bundle"],
            "planted_case_definition",
        )
        record["repo_relative_path"] = provenance.FIXED_ROLE_PATHS[
            "calibration_fixture_manifest"
        ]
        record["file_sha256"] = _sha256(replacement)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "role planted_case_definition must use one JSON file below",
        ):
            self._validate(fixture)

    def test_loaded_source_digest_mismatch_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["loaded_sources"]["evaluator_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "loaded source evaluator_source digest differs",
        ):
            self._validate(fixture)

    def test_loaded_source_path_substitution_is_rejected(self) -> None:
        fixture = self._fixture()
        fixture["loaded_sources"]["evaluator_source"]["absolute_path"] = str(
            fixture["paths_by_role"]["split_adapter_source"].resolve()
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "loaded source evaluator_source path mismatch",
        ):
            self._validate(fixture)

    def test_loaded_source_symlink_alias_is_rejected(self) -> None:
        fixture = self._fixture()
        target = fixture["paths_by_role"]["evaluator_source"]
        alias = fixture["repo"] / "evaluator-source-alias.py"
        os.symlink(target, alias)
        fixture["loaded_sources"]["evaluator_source"]["absolute_path"] = str(alias)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "loaded source evaluator_source must not be a symlink",
        ):
            self._validate(fixture)

    def test_dataset_fit_role_is_exact_and_conditional(self) -> None:
        with_fit = self._fixture(case_kind="dataset", fit_from_pairs=True)
        self._validate(with_fit)
        with_fit["bundle"]["committed_files"] = [
            record
            for record in with_fit["bundle"]["committed_files"]
            if record["role"] != "fit_pair_artifact"
        ]
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "committed role profile mismatch.*fit_pair_artifact",
        ):
            self._validate(with_fit)

        without_fit = self._fixture(case_kind="dataset", fit_from_pairs=False)
        extra = copy.deepcopy(
            _record_for_role(
                without_fit["bundle"],
                "evaluation_pair_artifact",
            )
        )
        extra["role"] = "fit_pair_artifact"
        without_fit["bundle"]["committed_files"].append(extra)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "committed role profile mismatch.*fit_pair_artifact",
        ):
            self._validate(without_fit)

    def test_checkpoint_receipt_prevents_reopen_but_config_and_history_open(
        self,
    ) -> None:
        fixture = self._fixture(case_kind="checkpoint")
        checkpoint_path = fixture["external_paths"]["checkpoint"]
        config_path = fixture["external_paths"]["checkpoint_config"]
        history_path = fixture["external_paths"]["checkpoint_metadata"]
        opened_external_paths: list[Path] = []
        original_open = Path.open

        def guarded_open(path: Path, *args: Any, **kwargs: Any):
            if path == checkpoint_path:
                raise AssertionError("provenance reopened the checkpoint")
            if path in {config_path, history_path}:
                opened_external_paths.append(path)
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            result = self._validate(fixture)
        normalized = {
            record["role"]: record for record in result["external_files"]
        }
        self.assertEqual(
            normalized["checkpoint"]["size_bytes"],
            checkpoint_path.stat().st_size,
        )
        self.assertEqual(
            opened_external_paths.count(config_path),
            1,
        )
        self.assertEqual(
            opened_external_paths.count(history_path),
            1,
        )

    def test_checkpoint_case_requires_validated_load_receipt(self) -> None:
        fixture = self._fixture(case_kind="checkpoint")
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "requires a validated load receipt",
        ):
            self._validate(fixture, checkpoint_receipt=None)

    def test_checkpoint_load_receipt_is_cross_bound(self) -> None:
        fixture = self._fixture(case_kind="checkpoint")
        receipt = _checkpoint_load_receipt(fixture)
        receipt["file_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "load receipt SHA-256 mismatch",
        ):
            self._validate(fixture, checkpoint_receipt=receipt)

    def test_external_same_size_mutation_is_rejected_by_sha256(self) -> None:
        fixture = self._fixture(case_kind="checkpoint")
        config_path = fixture["external_paths"]["checkpoint_config"]
        original_size = config_path.stat().st_size
        config_path.write_bytes(b"z" * original_size)
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "external role checkpoint_config SHA-256 mismatch",
        ):
            self._validate(fixture)

    def test_external_role_cannot_move_into_committed_files(self) -> None:
        fixture = self._fixture(case_kind="checkpoint")
        external = fixture["bundle"]["external_files"].pop()
        fixture["bundle"]["committed_files"].append(
            {
                "role": external["role"],
                "repo_relative_path": "evidence/not-an-external-role.bin",
                "file_sha256": external["file_sha256"],
            }
        )
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "committed role profile mismatch|external role profile mismatch",
        ):
            self._validate(fixture)

    def test_git_replacement_objects_are_disabled(self) -> None:
        fixture = self._fixture()
        old_commit = fixture["bundle"]["source_commit"]
        path = fixture["paths_by_role"]["evaluator_source"]
        path.write_bytes(b"replacement-commit evaluator bytes\n")
        _git(fixture["repo"], "add", "--all")
        _git(fixture["repo"], "commit", "-q", "-m", "replacement commit")
        replacement_commit = _git(fixture["repo"], "rev-parse", "HEAD")
        _git(fixture["repo"], "replace", old_commit, replacement_commit)

        new_digest = _sha256(path)
        _record_for_role(
            fixture["bundle"],
            "evaluator_source",
        )["file_sha256"] = new_digest
        fixture["loaded_sources"]["evaluator_source"] = {
            "absolute_path": str(path.resolve()),
            "sha256": new_digest,
        }
        with self.assertRaisesRegex(
            provenance.ProvenanceContractError,
            "working bytes differ from Git blob",
        ):
            self._validate(fixture)

    def test_git_dir_environment_override_is_ignored(self) -> None:
        fixture = self._fixture()
        other_repo = Path(
            tempfile.mkdtemp(
                prefix="other-repo-",
                dir=self.persistent_root,
            )
        ).resolve()
        _git(other_repo, "init", "-q")
        with mock.patch.dict(
            os.environ,
            {"GIT_DIR": str(other_repo / ".git")},
        ):
            result = self._validate(fixture)
        self.assertTrue(result["source_commit_verified"])


if __name__ == "__main__":
    unittest.main()
