"""Focused synthetic tests for the frozen primary package decision."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import roll_head_package_decision as decision


def _cell(
    recipe: str,
    head: str,
    seed: int,
    *,
    auc: float,
    drift: float,
    duplicates: int = 1,
    recurrent: int = 2,
    active: int = 8,
    angle: float = 0.2,
    eligible: bool = True,
) -> decision.CellEvidence:
    cell_id = f"{recipe}__r{head}__seed{seed}"
    return decision.CellEvidence(
        cell_id=cell_id,
        recipe=recipe,
        head_package=head,
        seed=seed,
        eligible=eligible,
        ineligibility_reasons=() if eligible else ("evaluator_critical_failure",),
        angle_error_deg=angle if eligible else None,
        validation_auc=auc if eligible else None,
        canonical_drift=drift if eligible else None,
        persistent_duplicate_count=duplicates if eligible else None,
        recurrent_duplicate_count=recurrent if eligible else None,
        active_on_object_count=active if eligible else None,
        source_commit="1" * 40,
        evaluation_config_sha256="4" * 64,
        result_content_sha256=("a" if head == "64" else "b") * 64,
        checkpoint_sha256="2" * 64,
        completed_run_receipt_sha256="3" * 64,
    )


def _matrix(
    *,
    auc_wins: dict[str, set[int]] | None = None,
    drift_wins: dict[str, set[int]] | None = None,
) -> list[decision.CellEvidence]:
    auc_wins = auc_wins or {recipe: set(decision.PRIMARY_SEEDS) for recipe in decision.RECIPES}
    drift_wins = drift_wins or {recipe: set(decision.PRIMARY_SEEDS) for recipe in decision.RECIPES}
    cells = []
    for recipe in decision.RECIPES:
        for seed in decision.PRIMARY_SEEDS:
            cells.append(_cell(recipe, "64", seed, auc=1.0, drift=1.0))
            cells.append(_cell(
                recipe,
                "128",
                seed,
                auc=0.8 if seed in auc_wins[recipe] else 1.0,
                drift=0.8 if seed in drift_wins[recipe] else 1.0,
            ))
    return cells


def _evaluator_result(cell_id: str) -> dict:
    recipe, head_token, seed_token = cell_id.split("__")
    head = head_token.removeprefix("r")
    seed = int(seed_token.removeprefix("seed"))
    source_commit = "1" * 40
    result = {
        "schema_version": decision.RESULT_SCHEMA_VERSION,
        "case_id": cell_id,
        "case_kind": "checkpoint",
        "stratum": {
            "object_id": "engineers_hammer_vray",
            "seed": seed,
            "partition": "full_corpus",
            "transform_family": "roll",
            "direction": "forward",
            "stride": 3,
        },
        "transform": {
            "family": "roll",
            "physical_axis": "world_z",
            "direction": "forward",
            "stride": 3,
            "cyclic": True,
            "signed_generator": 6.0,
        },
        "evaluation_config": copy.deepcopy(decision.FROZEN_EVALUATION_CONFIG),
        "checkpoint_authorization": {
            "checkpoint_evaluation_authorized": True,
            "source_commit": source_commit,
            "cell_id": cell_id,
            "checkpoint_sha256": "2" * 64,
            "completed_run_receipt_sha256": "3" * 64,
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": True,
        },
        "provenance": {
            "source_commit": source_commit,
            "source_commit_verified": True,
            "committed_files": [{
                "role": "evaluator_source",
                "repo_relative_path": "keypoint_net/eval_representation.py",
                "file_sha256": "a" * 64,
            }],
            "external_files": [
                {
                    "role": role,
                    "absolute_path": f"/tmp/{cell_id}/{role}",
                    "file_sha256": digest,
                    "size_bytes": 1,
                }
                for role, digest in (
                    ("checkpoint", "2" * 64),
                    ("checkpoint_config", "8" * 64),
                    ("checkpoint_metadata", "9" * 64),
                    ("completed_run_receipt", "3" * 64),
                )
            ],
        },
        "critical_failures": [],
        "operator": {
            "wrong_locked_sign": False,
            "improper_or_reflection": False,
            "absolute_angle_error_deg": 0.2 if head == "64" else 0.1,
        },
        "collapse_evidence": {
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead": (
                "available"
            ),
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead": (
                False
            ),
        },
        "channel_health": {"active_on_object_count": 8},
        "trajectory_separation": {
            "eligible_channel_count": 8,
            "eligible_active_on_object": {
                "category_counts": {
                    "persistent_duplicate": 1,
                    "recurrent_close_pair": 2,
                }
            },
        },
        "rollout": {
            "role_scoped_holdout_identity_normalized_auc": {
                "value": 1.0 if head == "64" else 0.8,
            }
        },
        "canonical_drift": {
            "median_rms_objdiag": 1.0 if head == "64" else 0.8,
        },
    }
    result["evaluation_config_sha256"] = decision.canonical_sha256(
        result["evaluation_config"]
    )
    result["result_content_sha256"] = decision.canonical_sha256(result)
    return result


def _write_matrix_lock(
    root: Path, *, source_commit: str = "1" * 40
) -> tuple[Path, str, str, str]:
    (root / "runs").mkdir(parents=True)
    (root / "evaluations").mkdir()
    environment_lock = root / "FRESH_ROLL_ENVIRONMENT.json"
    slurm_script = root / "primary.slurm"
    environment_lock.write_text("{}\n", encoding="utf-8")
    slurm_script.write_text("#!/bin/bash\n", encoding="utf-8")

    def record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "absolute_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    document = {
        "schema_version": decision.PRIMARY_MATRIX_SCHEMA_VERSION,
        "artifact_type": "roll_head_package_primary_matrix_launch_lock",
        "matrix_root": str(root),
        "source_commit": source_commit,
        "source_branch": decision.EXPECTED_BRANCH,
        "slurm_script": record(slurm_script),
        "environment_lock": record(environment_lock),
        "primary_cell_ids_by_task_index": sorted(decision._expected_cell_ids()),
        "task_count": 12,
        "gpu_count_per_task": 1,
        "max_concurrent_gpu_nodes": 1,
        "output_layout": {
            "runs_directory": str(root / "runs"),
            "evaluations_directory": str(root / "evaluations"),
            "evaluation_filename_template": "{cell_id}.json",
        },
        "submission": {
            "performed_by_prepare_command": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
        },
    }
    document["content_hash_sha256"] = decision.canonical_sha256(document)
    lock_path = root / decision.PRIMARY_MATRIX_LOCK_NAME
    lock_path.write_bytes(decision.canonical_json_bytes(document, newline=True))
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return (
        lock_path,
        lock_sha,
        document["environment_lock"]["sha256"],
        document["slurm_script"]["sha256"],
    )


def _launch_chain(lock_path: Path) -> dict[str, dict]:
    document = json.loads(lock_path.read_text(encoding="utf-8"))

    def record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "absolute_path": str(path),
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    return {
        "primary_matrix_launch_lock": record(lock_path),
        "environment_lock": record(
            Path(document["environment_lock"]["absolute_path"])
        ),
        "slurm_job_script": record(
            Path(document["slurm_script"]["absolute_path"])
        ),
    }


class RollHeadPackageDecisionTests(unittest.TestCase):
    def test_all_required_primary_conditions_adopt_128(self) -> None:
        result = decision.decide_primary_package(_matrix())
        self.assertEqual(result["decision"], "adopt_128")
        self.assertEqual(result["selected_head_package"], "128")
        self.assertFalse(result["extension_required"])

    def test_exactly_two_of_three_required_wins_triggers_only_extension(self) -> None:
        two = {recipe: {42, 43} for recipe in decision.RECIPES}
        result = decision.decide_primary_package(
            _matrix(auc_wins=two, drift_wins=two)
        )
        self.assertEqual(result["decision"], "run_extension_45_46")
        self.assertIsNone(result["selected_head_package"])
        self.assertEqual(result["extension_seeds"], [45, 46])

    def test_insufficient_required_axis_or_guardrail_retains_64(self) -> None:
        one = {recipe: {42} for recipe in decision.RECIPES}
        result = decision.decide_primary_package(_matrix(auc_wins=one))
        self.assertEqual(result["decision"], "retain_64")
        two = {recipe: {42, 43} for recipe in decision.RECIPES}
        cells = _matrix(auc_wins=two, drift_wins=two)
        for index, cell in enumerate(cells):
            if (
                cell.recipe == "task55_clean"
                and cell.head_package == "128"
                and cell.seed in {42, 43}
            ):
                cells[index] = _cell(
                    cell.recipe, cell.head_package, cell.seed,
                    auc=0.8, drift=0.8, duplicates=3, active=7,
                )
        self.assertEqual(
            decision.decide_primary_package(cells)["decision"], "retain_64"
        )

    def test_ineligible_cell_retains_64_but_incomplete_or_duplicate_matrix_rejects(self) -> None:
        cells = _matrix()
        failed = cells[0]
        cells[0] = _cell(
            failed.recipe, failed.head_package, failed.seed,
            auc=1.0, drift=1.0, eligible=False,
        )
        result = decision.decide_primary_package(cells)
        self.assertEqual(result["decision"], "retain_64")
        self.assertIn(failed.cell_id, result["ineligible_cells"])
        with self.assertRaisesRegex(decision.PackageDecisionError, "cell set differs"):
            decision.decide_primary_package(cells[:-1])
        with self.assertRaisesRegex(decision.PackageDecisionError, "duplicate cell"):
            decision.decide_primary_package([*cells, cells[-1]])

    def test_hashed_fresh_evaluator_result_extracts_and_tamper_rejects(self) -> None:
        result = _evaluator_result("task80_assisted__r128__seed44")
        cell = decision.extract_cell_evidence(result)
        self.assertTrue(cell.eligible)
        self.assertEqual((cell.recipe, cell.head_package, cell.seed),
                         ("task80_assisted", "128", 44))
        tampered = copy.deepcopy(result)
        tampered["operator"]["absolute_angle_error_deg"] = 9.0
        with self.assertRaisesRegex(decision.PackageDecisionError, "content hash differs"):
            decision.extract_cell_evidence(tampered)

    def test_full_evaluation_config_is_exact_and_fail_closed(self) -> None:
        result = _evaluator_result("task55_clean__r64__seed42")
        for mutation in ("omit", "change", "bool_for_int"):
            candidate = copy.deepcopy(result)
            if mutation == "omit":
                candidate["evaluation_config"].pop("motion_fraction_min")
            elif mutation == "change":
                candidate["evaluation_config"]["motion_fraction_min"] = 0.2
            else:
                candidate["evaluation_config"]["full_rollout_horizons"][0] = True
            candidate["evaluation_config_sha256"] = decision.canonical_sha256(
                candidate["evaluation_config"]
            )
            candidate.pop("result_content_sha256")
            candidate["result_content_sha256"] = decision.canonical_sha256(candidate)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                decision.PackageDecisionError, "frozen evaluation config differs"
            ):
                decision.extract_cell_evidence(candidate)

    def test_empty_or_mismatched_provenance_and_false_selection_reject(self) -> None:
        cell_id = "task80_assisted__r128__seed44"
        for name, mutate, message in (
            (
                "empty committed provenance",
                lambda value: value["provenance"].update({"committed_files": []}),
                "committed provenance is empty",
            ),
            (
                "mismatched receipt",
                lambda value: value["provenance"]["external_files"][3].update(
                    {"file_sha256": "f" * 64}
                ),
                "completed_run_receipt provenance binding differs",
            ),
            (
                "selection false",
                lambda value: value["checkpoint_authorization"].update(
                    {"selection_use_authorized": False}
                ),
                "fresh selection authorization differs",
            ),
        ):
            candidate = _evaluator_result(cell_id)
            mutate(candidate)
            candidate.pop("result_content_sha256")
            candidate["result_content_sha256"] = decision.canonical_sha256(candidate)
            with self.subTest(name=name), self.assertRaisesRegex(
                decision.PackageDecisionError, message
            ):
                decision.extract_cell_evidence(candidate)

    def test_live_provenance_is_independently_rehashed(self) -> None:
        cell_id = "task55_clean__r64__seed42"
        result = _evaluator_result(cell_id)
        repo_root = Path(decision.__file__).resolve().parents[1]
        committed = []
        for role, relative_path in decision.FRESH_COMMITTED_ROLE_PATHS.items():
            path = repo_root / relative_path
            data = path.read_bytes()
            committed.append({
                "role": role,
                "repo_relative_path": relative_path,
                "absolute_path": str(path),
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_oid": "0" * 40,
            })
        external_root = Path(tempfile.mkdtemp(prefix="decision_live_provenance_"))
        external = []
        for role in sorted(decision.FRESH_EXTERNAL_ROLES):
            path = external_root / role
            path.write_text(f"{role}\n", encoding="utf-8")
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            external.append({
                "role": role,
                "absolute_path": str(path),
                "file_sha256": digest,
                "size_bytes": len(data),
            })
            authorization_key = decision.FRESH_EXTERNAL_HASH_BINDINGS.get(role)
            if authorization_key is not None:
                result["checkpoint_authorization"][authorization_key] = digest
        external_by_role = {record["role"]: record for record in external}
        launch_chain = {}
        for role in (
            "primary_matrix_launch_lock",
            "environment_lock",
            "slurm_job_script",
        ):
            path = external_root / f"launch_{role}"
            path.write_text(f"{role}\n", encoding="utf-8")
            data = path.read_bytes()
            launch_chain[role] = {
                "absolute_path": str(path),
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        receipt_document = {
            "schema_version": "roll_head_package_completed_run_receipt.v3",
            "cell_id": cell_id,
            "source_commit": "1" * 40,
            "files": {
                receipt_role: {
                    "absolute_path": external_by_role[external_role]["absolute_path"],
                    "sha256": external_by_role[external_role]["file_sha256"],
                    "size_bytes": external_by_role[external_role]["size_bytes"],
                }
                for receipt_role, external_role in (
                    ("checkpoint", "checkpoint"),
                    ("config", "checkpoint_config"),
                    ("history", "checkpoint_metadata"),
                )
            },
            "execution": {
                "training_completed": True,
                "selection_use_authorized": True,
                "test_loader_constructed": False,
                "runtime_environment": {
                    role: {
                        "absolute_path": launch_chain[role]["absolute_path"],
                        "sha256": launch_chain[role]["file_sha256"],
                        "size_bytes": launch_chain[role]["size_bytes"],
                    }
                    for role in (
                        "primary_matrix_launch_lock",
                        "environment_lock",
                        "slurm_job_script",
                    )
                },
            },
        }
        receipt_document["content_hash_sha256"] = decision.canonical_sha256(
            receipt_document
        )
        receipt_path = external_root / "completed_run_receipt"
        receipt_path.write_bytes(
            decision.canonical_json_bytes(receipt_document, newline=True)
        )
        receipt_data = receipt_path.read_bytes()
        external_by_role["completed_run_receipt"].update({
            "file_sha256": hashlib.sha256(receipt_data).hexdigest(),
            "size_bytes": len(receipt_data),
        })
        result["checkpoint_authorization"][
            "completed_run_receipt_sha256"
        ] = hashlib.sha256(receipt_data).hexdigest()
        result["provenance"] = {
            "schema_version": "representation_evaluation_provenance.v1",
            "source_commit": "1" * 40,
            "source_commit_verified": True,
            "repository_root": str(repo_root),
            "case_kind": "checkpoint",
            "fit_from_pairs": False,
            "committed_files": committed,
            "external_files": external,
            "loaded_sources": {
                role: {
                    "absolute_path": next(
                        record["absolute_path"]
                        for record in committed if record["role"] == role
                    ),
                    "sha256": next(
                        record["file_sha256"]
                        for record in committed if record["role"] == role
                    ),
                }
                for role in ("array_codec_source", "evaluator_source")
            },
            "provenance_loaded_source": {
                "absolute_path": next(
                    record["absolute_path"]
                    for record in committed
                    if record["role"] == "provenance_source"
                ),
                "sha256": next(
                    record["file_sha256"]
                    for record in committed
                    if record["role"] == "provenance_source"
                ),
            },
        }
        self.assertEqual(decision._validate_live_result_provenance(
            result, repo_root=repo_root, expected_commit="1" * 40
        ), launch_chain)
        checkpoint = external_root / "checkpoint"
        checkpoint.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(
            decision.PackageDecisionError,
            "external role checkpoint live binding differs",
        ):
            decision._validate_live_result_provenance(
                result, repo_root=repo_root, expected_commit="1" * 40
            )

    def test_direct_decision_rejects_dirty_or_wrong_commit(self) -> None:
        repo_root = Path("/tmp/fake-repository")
        with mock.patch.object(
            decision,
            "_git",
            side_effect=["0" * 40],
        ):
            with self.assertRaisesRegex(
                decision.PackageDecisionError, "commit differs"
            ):
                decision._validate_decision_source(
                    repo_root, expected_commit="1" * 40
                )
        with mock.patch.object(
            decision,
            "_git",
            side_effect=["1" * 40, decision.EXPECTED_BRANCH, " M changed.py"],
        ):
            with self.assertRaisesRegex(
                decision.PackageDecisionError, "worktree is dirty"
            ):
                decision._validate_decision_source(
                    repo_root, expected_commit="1" * 40
                )

    def test_shared_result_commit_must_equal_matrix_lock_commit(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="roll_head_commit_bind_test_"))
        lock_path, _, _, _ = _write_matrix_lock(
            root, source_commit="0" * 40
        )
        arguments: list[str] = []
        for recipe in decision.RECIPES:
            for head in decision.HEAD_PACKAGES:
                for seed in decision.PRIMARY_SEEDS:
                    cell_id = f"{recipe}__r{head}__seed{seed}"
                    result_path = root / "evaluations" / f"{cell_id}.json"
                    result_path.write_text(json.dumps(_evaluator_result(cell_id)))
                    arguments.extend(["--result", str(result_path)])
        with self.assertRaisesRegex(
            decision.PackageDecisionError,
            "decision inputs differ from the matrix source commit",
        ):
            with (
                mock.patch.object(decision, "_validate_decision_source"),
                mock.patch.object(
                    decision,
                    "_validate_live_result_provenance",
                    return_value=_launch_chain(lock_path),
                ),
            ):
                decision.main([
                    *arguments,
                    "--matrix-lock", str(lock_path),
                    "--expected-commit", "0" * 40,
                    "--output", str(root / "decision.json"),
                ])
    def test_cli_consumes_exactly_twelve_results_and_writes_exclusively(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="roll_head_decision_test_"))
        lock_path, _, _, _ = _write_matrix_lock(root)
        arguments: list[str] = []
        for recipe in decision.RECIPES:
            for head in decision.HEAD_PACKAGES:
                for seed in decision.PRIMARY_SEEDS:
                    cell_id = f"{recipe}__r{head}__seed{seed}"
                    path = root / "evaluations" / f"{cell_id}.json"
                    path.write_text(json.dumps(_evaluator_result(cell_id)))
                    arguments.extend(("--result", str(path)))
        output = root / "PRIMARY_PACKAGE_DECISION.json"
        fixed = [
            "--matrix-lock", str(lock_path),
            "--expected-commit", "1" * 40,
            "--output", str(output),
        ]
        with (
            mock.patch.object(decision, "_validate_decision_source"),
            mock.patch.object(
                decision,
                "_validate_live_result_provenance",
                return_value=_launch_chain(lock_path),
            ),
        ):
            self.assertEqual(decision.main([*arguments, *fixed]), 0)
        written = json.loads(output.read_text())
        self.assertEqual(written["decision"], "adopt_128")
        self.assertEqual(len(written["input_files"]), 12)
        with self.assertRaisesRegex(
            decision.PackageDecisionError, "exclusively create"
        ):
            with (
                mock.patch.object(decision, "_validate_decision_source"),
                mock.patch.object(
                    decision,
                    "_validate_live_result_provenance",
                    return_value=_launch_chain(lock_path),
                ),
            ):
                decision.main([*arguments, *fixed])


if __name__ == "__main__":
    unittest.main()
