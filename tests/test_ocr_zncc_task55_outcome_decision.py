from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from keypoint_net import ocr_zncc_outcome_evaluator as evaluator
from keypoint_net import ocr_zncc_task55_outcome_decision as decision


RECEIPT_RECORD = {
    "absolute_path": "/evidence/OCR_ZNCC_COMPLETED_RUN_RECEIPT.json",
    "sha256": "b" * 64,
    "size_bytes": 1234,
    "content_hash_sha256": "c" * 64,
}


def _live_result_document() -> dict:
    checkpoint_results = {}
    for role, epoch, selector in (
        ("best_model", 20, "minimum_base_validation_loss"),
        ("final_model", 1000, "fixed_epoch_1000"),
    ):
        checkpoint_results[role] = {
            "checkpoint_role": role,
            "checkpoint_epoch": epoch,
            "checkpoint_selector": selector,
            "canonical_drift": {"median_rms_objdiag": 0.10},
            "operator": {
                "proper_rotation_angle_deg": 5.5,
                "absolute_angle_error_deg": 0.5,
                "improper_or_reflection": False,
            },
            "channel_health": {"active_on_object_count": 8},
            "all_numeric_outputs_finite": True,
            "critical_failures": [],
        }
    document = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "artifact_type": "frozen_paired_ocr_zncc_outcome_cell",
        "evaluation_status": "complete",
        "cell_id": "task55_clean__ocr_zncc__seed42",
        "recipe": "task55_clean",
        "arm": "ocr_zncc",
        "seed": 42,
        "source_commit": "a" * 40,
        "source_branch": "agent/ocr-zncc-training-20260811",
        "metric_lock": {},
        "lineage": {"completed_run_receipt": copy.deepcopy(RECEIPT_RECORD)},
        "checkpoint_results": checkpoint_results,
        "statistical_scope": {},
        "training_or_weight_update_performed": False,
        "selection_performed_by_evaluator": False,
    }
    document["content_hash_sha256"] = evaluator.canonical_sha256(document)
    return document


def _pairing_evidence(position: str) -> dict:
    return {
        "initial_model_state_sha256": "1" * 64,
        "post_diagnostic_model_state_sha256": "2" * 64,
        "diagnostic_pair_order_sha256": "3" * 64,
        "diagnostic_pair_order_sample_count": 16,
        "train_sampler_generator_seed": 10_000_061,
        "epoch_pair_order_sequence_sha256": "4" * 64,
        "execution_device": "cuda",
        "runtime_pair_execution": {
            "pair_id": "task55_clean__seed42",
            "position": position,
            "wrapper_pid": 12345,
            "slurm_job_id": "98765",
            "cuda_visible_devices": "0",
            "allocated_gpu_count": 1,
            "stage_launch_lock_content_hash_sha256": "5" * 64,
            "runtime_provenance_files": {
                "stage_launch_lock": {"sha256": "6" * 64},
                "environment_lock": {"sha256": "7" * 64},
                "slurm_script": {"sha256": "8" * 64},
            },
        },
    }


def _checkpoint(
    *,
    drift: float,
    angle: float = 6.0,
    angle_error: float = 0.0,
    active: int = 8,
    reflection: bool = False,
) -> decision.CheckpointEvidence:
    return decision.CheckpointEvidence(
        canonical_drift=drift,
        proper_rotation_angle_deg=None if reflection else angle,
        absolute_angle_error_deg=None if reflection else angle_error,
        active_on_object_count=active,
        improper_or_reflection=reflection,
        all_numeric_outputs_finite=True,
        critical_failures=("operator_is_improper_or_reflection",) if reflection else (),
    )


def _matrix() -> list[decision.CellEvidence]:
    cells = []
    for seed in decision.SEEDS:
        cells.append(decision.CellEvidence(
            cell_id=f"task55_clean__control__seed{seed}",
            arm="control",
            seed=seed,
            source_commit="a" * 40,
            result_content_sha256=str(seed) * 64,
            best_model=_checkpoint(drift=1.0, angle=5.0, angle_error=1.0, active=7),
            final_model=_checkpoint(drift=1.0, angle=5.0, angle_error=1.0, active=7),
        ))
        best_drift = 0.70 if seed in {42, 43} else 0.90
        final_drift = 0.90 if seed in {42, 43} else 1.10
        best_error = 0.50 if seed in {42, 43} else 1.50
        cells.append(decision.CellEvidence(
            cell_id=f"task55_clean__ocr_zncc__seed{seed}",
            arm="ocr_zncc",
            seed=seed,
            source_commit="a" * 40,
            result_content_sha256=str(seed + 1) * 64,
            best_model=_checkpoint(
                drift=best_drift, angle=5.5, angle_error=best_error, active=7
            ),
            final_model=_checkpoint(
                drift=final_drift, angle=5.5, angle_error=best_error, active=7
            ),
        ))
    return cells


class OCRTask55DecisionTests(unittest.TestCase):
    def test_generic_evaluation_config_hash_is_frozen(self) -> None:
        config = evaluator.evaluation_config()
        self.assertEqual(
            evaluator.GENERIC_EVALUATION_CONFIG_SHA256,
            evaluator.canonical_sha256(config),
        )
        self.assertEqual(list(range(24)), list(evaluator.VALIDATION_FRAME_IDS))
        self.assertEqual(list(range(10)), list(evaluator.SOFT_COORDINATE_CHANNELS))

    def test_roll_inventory_must_match_live_corpus_and_committed_source(self) -> None:
        run = mock.Mock()
        run.repo_root = Path("/repo")
        run.training_repo_root = Path("/training-repo")
        run.source_commit = "a" * 40
        run.evaluation_source_files = {
            "keypoint_net/representation_corpus_inventory.py": {
                "relative_path": "keypoint_net/representation_corpus_inventory.py",
                "sha256": "9" * 64,
            },
        }
        inventory_record = {
            "relative_path": evaluator.ROLL_CORPUS_INVENTORY_RELATIVE_PATH,
            "absolute_path": "/repo/inventory.json",
            "sha256": "a" * 64,
            "size_bytes": 100,
        }
        validated_inventory = mock.Mock()
        validated_inventory.content_hash_sha256 = evaluator.ROLL_CORPUS_CONTENT_SHA256
        with mock.patch.object(
            evaluator,
            "_committed_file_bytes_and_record",
            return_value=(b"inventory", inventory_record),
        ), mock.patch.object(
            evaluator.corpus_inventory,
            "validate_corpus_inventory",
            return_value=validated_inventory,
        ):
            binding = evaluator._validate_roll_corpus_inventory(  # noqa: SLF001
                run, data_root=Path("/bound/data")
            )
        self.assertTrue(binding["exact_live_corpus_match"])
        self.assertEqual(
            evaluator.ROLL_CORPUS_CONTENT_SHA256,
            binding["inventory_file"]["content_hash_sha256"],
        )
        self.assertEqual(
            run.evaluation_source_files[
                "keypoint_net/representation_corpus_inventory.py"
            ],
            binding["validator_source"],
        )

    def test_checkpoint_float32_coordinate_policy_accepts_roundoff_but_rejects_drift(
        self,
    ) -> None:
        class DummyModel(torch.nn.Module):
            def extractor(self, batch):
                return (
                    torch.zeros((batch.shape[0], 20), dtype=torch.float32),
                    torch.zeros((batch.shape[0], 10, 64, 64), dtype=torch.float32),
                )

        model = DummyModel().eval()
        images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(24)]
        with mock.patch.object(
            evaluator.representation_evaluator,
            "spatial_expectation",
            return_value=np.full((24, 10, 2), 2e-6, dtype=np.float32),
        ):
            _points, _logits, record = evaluator._infer_soft_coordinates(  # noqa: SLF001
                model, images
            )
        self.assertTrue(record["coordinate_consistency_pass"])
        self.assertEqual(1e-4, record["coordinate_consistency_tolerance"])
        self.assertEqual(
            evaluator.representation_evaluator.
            CHECKPOINT_FLOAT32_CROSS_BACKEND_COORDINATE_TOLERANCE_KEY,
            record["coordinate_consistency_tolerance_key"],
        )

        with mock.patch.object(
            evaluator.representation_evaluator,
            "spatial_expectation",
            return_value=np.full((24, 10, 2), 2e-4, dtype=np.float32),
        ), self.assertRaisesRegex(
            evaluator.OCROutcomeEvaluationError,
            "checkpoint float32 policy",
        ):
            evaluator._infer_soft_coordinates(model, images)  # noqa: SLF001

    def test_evaluator_commit_is_distinct_and_descends_from_training_commit(self) -> None:
        training_commit = "a" * 40
        evaluation_commit = "b" * 40
        source_calls = []

        def fake_git(_root, *arguments):
            return {
                ("rev-parse", "HEAD"): evaluation_commit,
                ("branch", "--show-current"):
                    "agent/ocr-zncc-training-20260811",
                ("status", "--porcelain"): "",
            }[arguments]

        def fake_source(_root, relative, *, source_commit):
            source_calls.append((relative, source_commit))
            return {"relative_path": relative, "sha256": "c" * 64}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator, "_git", side_effect=fake_git
        ), mock.patch.object(
            evaluator, "_committed_source_record", side_effect=fake_source
        ), mock.patch.object(
            evaluator.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ):
            checkout = evaluator._validate_checkout(  # noqa: SLF001
                Path(directory),
                source_commit=training_commit,
                source_branch="agent/ocr-zncc-training-20260811",
            )
        self.assertEqual(training_commit, checkout["training_source_commit"])
        self.assertEqual(evaluation_commit, checkout["evaluation_source_commit"])
        self.assertEqual(
            {(relative, evaluation_commit) for relative in evaluator.SOURCE_RELATIVE_PATHS},
            set(source_calls),
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator,
            "_git",
            side_effect=lambda _root, *arguments: {
                ("rev-parse", "HEAD"): evaluation_commit,
                ("branch", "--show-current"):
                    "agent/ocr-zncc-training-20260811",
                ("status", "--porcelain"): "modified",
            }[arguments],
        ), self.assertRaisesRegex(
            evaluator.OCROutcomeEvaluationError,
            "not completely clean",
        ):
            evaluator._validate_checkout(  # noqa: SLF001
                Path(directory),
                source_commit=training_commit,
                source_branch="agent/ocr-zncc-training-20260811",
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator, "_git", side_effect=fake_git
        ), mock.patch.object(
            evaluator.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CalledProcessError(1, []),
            ],
        ), self.assertRaisesRegex(
            evaluator.OCROutcomeEvaluationError,
            "training source commit is not an ancestor",
        ):
            evaluator._validate_checkout(  # noqa: SLF001
                Path(directory),
                source_commit=training_commit,
                source_branch="agent/ocr-zncc-training-20260811",
            )

    def test_mutated_live_corpus_inventory_binding_is_rejected(self) -> None:
        run = mock.Mock()
        run.repo_root = Path("/repo")
        run.training_repo_root = Path("/training-repo")
        run.source_commit = "a" * 40
        run.evaluation_source_files = {
            "keypoint_net/representation_corpus_inventory.py": {"sha256": "9" * 64},
        }
        mutated = mock.Mock()
        mutated.content_hash_sha256 = "0" * 64
        with mock.patch.object(
            evaluator,
            "_committed_file_bytes_and_record",
            return_value=(b"inventory", {"sha256": "a" * 64}),
        ), mock.patch.object(
            evaluator.corpus_inventory,
            "validate_corpus_inventory",
            return_value=mutated,
        ), self.assertRaisesRegex(
            evaluator.OCROutcomeEvaluationError,
            "content hash differs from frozen dataset binding",
        ):
            evaluator._validate_roll_corpus_inventory(  # noqa: SLF001
                run, data_root=Path("/bound/data")
            )

    def test_pairing_requires_same_wrapper_allocation_and_exact_positions(self) -> None:
        control = _pairing_evidence("1_control")
        ocr = _pairing_evidence("2_ocr_zncc")
        decision.validate_matched_pairing_evidence(
            control, ocr, recipe="task55_clean", seed=42
        )
        ocr["runtime_pair_execution"]["position"] = "1_control"
        with self.assertRaisesRegex(
            decision.OCRTask55DecisionError,
            "not one control-first/OCR-second GPU pair",
        ):
            decision.validate_matched_pairing_evidence(
                control, ocr, recipe="task55_clean", seed=42
            )

    def test_runtime_pair_evidence_is_artifact_bound_and_position_checked(self) -> None:
        artifacts = {
            "stage_launch_lock": {
                "absolute_path": "/evidence/stage.json",
                "sha256": "1" * 64,
                "size_bytes": 100,
            },
            "environment_lock": {
                "absolute_path": "/evidence/environment.json",
                "sha256": "2" * 64,
                "size_bytes": 200,
            },
            "slurm_script": {
                "absolute_path": "/repo/cluster/ocr_zncc_paired_stage.slurm",
                "sha256": "3" * 64,
                "size_bytes": 300,
            },
        }
        runtime = {
            "device_type": "cuda",
            "slurm_job_id": "98765",
            "ocr_stage_launch_lock": copy.deepcopy(artifacts["stage_launch_lock"]),
            "environment_lock": copy.deepcopy(artifacts["environment_lock"]),
            "slurm_job_script": copy.deepcopy(artifacts["slurm_script"]),
            "ocr_pair_execution": {
                "pair_id": "task55_clean__seed42",
                "position": "1_control",
                "wrapper_pid": 12345,
                "slurm_job_id": "98765",
                "cuda_visible_devices": "0",
                "allocated_gpu_count": 1,
            },
        }
        manifest_record = {
            "absolute_path": "/evidence/manifest.json",
            "sha256": "4" * 64,
            "size_bytes": 400,
            "content_hash_sha256": "5" * 64,
        }
        stage = {
            "schema_version": "ocr_zncc_paired_stage_launch_lock.v1",
            "artifact_type": "ocr_zncc_paired_stage_launch_lock",
            "source_commit": "a" * 40,
            "source_branch": "agent/ocr-zncc-training-20260811",
            "recipe": "task55_clean",
            "manifest": copy.deepcopy(manifest_record),
            "environment_lock": copy.deepcopy(artifacts["environment_lock"]),
            "slurm_script": copy.deepcopy(artifacts["slurm_script"]),
            "pair_order": [
                {
                    "seed": seed,
                    "first": f"task55_clean__control__seed{seed}",
                    "second": f"task55_clean__ocr_zncc__seed{seed}",
                }
                for seed in decision.SEEDS
            ],
            "partial_failure_policy": "new_attempt_root_required_no_resume",
        }
        stage["content_hash_sha256"] = evaluator.canonical_sha256(stage)
        receipt = {
            "execution": {
                "device": "cuda",
                "runtime_environment": copy.deepcopy(runtime),
            },
            "source_files": {
                "cluster/ocr_zncc_paired_stage.slurm": copy.deepcopy(
                    artifacts["slurm_script"]
                ),
            },
        }
        config = {"runtime_environment": copy.deepcopy(runtime)}

        def verify(claim, **_kwargs):
            return dict(claim)

        with mock.patch.object(
            evaluator, "_verify_bound_record", side_effect=verify
        ), mock.patch.object(
            evaluator,
            "_load_json",
            return_value=(stage, copy.deepcopy(artifacts["stage_launch_lock"])),
        ):
            pair = evaluator._validate_runtime_pair_execution(  # noqa: SLF001
                receipt=receipt,
                config=config,
                source_commit="a" * 40,
                source_branch="agent/ocr-zncc-training-20260811",
                recipe="task55_clean",
                arm="control",
                seed=42,
                manifest_record=manifest_record,
            )
        self.assertEqual("1_control", pair["position"])
        self.assertEqual(artifacts, pair["runtime_provenance_files"])

    def test_live_result_validator_returns_only_exact_recomputation(self) -> None:
        expected = _live_result_document()
        run = mock.Mock()
        run.receipt_record = copy.deepcopy(RECEIPT_RECORD)
        run.receipt = {
            "expected_training_arguments": {"data_root": "/bound/data"},
        }
        with mock.patch.object(
            evaluator,
            "_load_json",
            return_value=(copy.deepcopy(expected), {
                "absolute_path": "/evidence/result.json",
                "sha256": "d" * 64,
                "size_bytes": 5678,
            }),
        ), mock.patch.object(
            evaluator, "validate_completed_run", return_value=run
        ), mock.patch.object(
            evaluator, "load_frozen_validation_data", return_value={"frozen": True}
        ) as load_validation, mock.patch.object(
            evaluator, "_build_result_document", return_value=copy.deepcopy(expected)
        ):
            validated = evaluator.validate_live_result(
                Path("/repo"), Path("/evidence/result.json")
            )
        self.assertEqual(expected, validated.document)
        self.assertIs(run, validated.run)
        self.assertEqual(expected["content_hash_sha256"],
                         validated.result_record["content_hash_sha256"])
        load_validation.assert_called_once_with(run, data_root="/bound/data")

    def test_raw_self_hashed_result_cannot_be_extracted_for_decision(self) -> None:
        with self.assertRaisesRegex(
            decision.OCRTask55DecisionError,
            "requires a live deep-validated outcome result",
        ):
            decision.extract_cell_evidence(_live_result_document())  # type: ignore[arg-type]

    def test_rehashed_metric_mutations_fail_before_decision_extraction(self) -> None:
        expected = _live_result_document()
        run = mock.Mock()
        run.receipt_record = copy.deepcopy(RECEIPT_RECORD)
        run.receipt = {
            "expected_training_arguments": {"data_root": "/bound/data"},
        }
        mutations = {
            "canonical drift": (
                "canonical_drift", "median_rms_objdiag", 0.001
            ),
            "operator": ("operator", "proper_rotation_angle_deg", 6.0),
            "channel health": ("channel_health", "active_on_object_count", 10),
        }
        for name, (section, field, value) in mutations.items():
            with self.subTest(metric=name):
                supplied = copy.deepcopy(expected)
                supplied["checkpoint_results"]["best_model"][section][field] = value
                supplied["content_hash_sha256"] = evaluator.canonical_sha256({
                    key: item for key, item in supplied.items()
                    if key != "content_hash_sha256"
                })
                with mock.patch.object(
                    evaluator,
                    "_load_json",
                    return_value=(supplied, {
                        "absolute_path": "/evidence/result.json",
                        "sha256": "e" * 64,
                        "size_bytes": 5678,
                    }),
                ), mock.patch.object(
                    evaluator, "validate_completed_run", return_value=run
                ), mock.patch.object(
                    evaluator,
                    "load_frozen_validation_data",
                    return_value={"frozen": True},
                ), mock.patch.object(
                    evaluator,
                    "_build_result_document",
                    return_value=copy.deepcopy(expected),
                ), self.assertRaisesRegex(
                    evaluator.OCROutcomeEvaluationError,
                    "differs from exact live metric recomputation",
                ):
                    decision._load_and_bind_result(  # noqa: SLF001
                        Path("/repo"), Path("/evidence/result.json")
                    )

    def test_exact_two_of_three_support_passes(self) -> None:
        result = decision.decide(_matrix())
        self.assertTrue(result["task55_pass"])
        self.assertEqual(
            "task55_pass_candidate_for_separate_task80_authorization",
            result["outcome"],
        )
        self.assertFalse(result["task80_training_authorized_by_this_artifact"])
        self.assertEqual(2, result["support_counts"][
            "selected_drift_ratio_at_most_0p80"
        ])
        self.assertEqual(2, result["support_counts"][
            "final_epoch_1000_drift_nonregression"
        ])

    def test_final_safety_diagnostics_cannot_change_pass(self) -> None:
        cells = _matrix()
        changed = []
        for cell in cells:
            if cell.arm == "ocr_zncc":
                changed.append(decision.CellEvidence(
                    **{
                        **cell.__dict__,
                        "final_model": _checkpoint(
                            drift=cell.final_model.canonical_drift,
                            angle=-6.0,
                            angle_error=12.0,
                            active=0,
                        ),
                    }
                ))
            else:
                changed.append(cell)
        result = decision.decide(changed)
        self.assertTrue(result["task55_pass"])
        self.assertFalse(result[
            "final_checkpoint_diagnostics_not_used_for_decision"
        ]["ocr_positive_roll_sign_in_all_three"])
        self.assertFalse(result[
            "final_checkpoint_diagnostics_not_used_for_decision"
        ]["ocr_active_on_object_count_nonlower_in_all_three"])

    def test_selected_active_guardrail_fails_on_one_seed(self) -> None:
        cells = _matrix()
        changed = []
        for cell in cells:
            if cell.cell_id == "task55_clean__ocr_zncc__seed44":
                changed.append(decision.CellEvidence(
                    **{
                        **cell.__dict__,
                        "best_model": _checkpoint(
                            drift=cell.best_model.canonical_drift,
                            angle=5.5,
                            angle_error=cell.best_model.absolute_angle_error_deg or 0.0,
                            active=6,
                        ),
                    }
                ))
            else:
                changed.append(cell)
        result = decision.decide(changed)
        self.assertFalse(result["task55_pass"])
        self.assertFalse(result["criterion_results"][
            "selected_ocr_active_on_object_count_nonlower_in_all_three"
        ])

    def test_reflection_blocks_even_if_drift_supports(self) -> None:
        cells = _matrix()
        changed = []
        for cell in cells:
            if cell.cell_id == "task55_clean__ocr_zncc__seed42":
                changed.append(decision.CellEvidence(
                    **{
                        **cell.__dict__,
                        "best_model": _checkpoint(drift=0.1, reflection=True),
                    }
                ))
            else:
                changed.append(cell)
        result = decision.decide(changed)
        self.assertFalse(result["task55_pass"])
        self.assertEqual("stop_invalid_or_critical_evaluator_evidence", result["outcome"])
        self.assertTrue(any(
            row["reason"] == "operator_is_improper_or_reflection"
            for row in result["critical_blockers"]
        ))


if __name__ == "__main__":
    unittest.main()
