"""Focused tests for the generic-GPU descriptor execution and decision gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import torch

from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net import descriptor_attachment_decision as decision
from keypoint_net import model as model_module
from keypoint_net import representation_fresh_checkpoint_runtime as runtime
from keypoint_net import run_descriptor_attachment_matrix as matrix


REPO_ROOT = Path(__file__).resolve().parents[1]


def _evidence(
    recipe: str,
    arm: str,
    seed: int,
    *,
    eligible: bool = True,
    drift: float = 1.0,
    auc: float = 1.0,
    persistent: int = 2,
    recurrent: int = 1,
    active: int = 4,
    angle: float = 1.0,
) -> decision.DescriptorEvidence:
    return decision.DescriptorEvidence(
        cell_id=f"{recipe}__{arm}__seed{seed}",
        recipe=recipe,
        arm=arm,
        seed=seed,
        eligible=eligible,
        ineligibility_reasons=() if eligible else ("planted_failure",),
        angle_error_deg=angle if eligible else None,
        validation_auc=auc if eligible else None,
        canonical_drift=drift if eligible else None,
        persistent_duplicate_count=persistent if eligible else None,
        recurrent_duplicate_count=recurrent if eligible else None,
        active_on_object_count=active if eligible else None,
        source_commit="a" * 40,
        result_content_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        completed_run_receipt_sha256="d" * 64,
    )


def _passing_cells() -> list[decision.DescriptorEvidence]:
    cells = []
    for recipe in decision.RECIPES:
        for seed in decision.SEEDS:
            cells.append(_evidence(recipe, "control", seed))
            cells.append(_evidence(
                recipe, "attachment", seed,
                drift=0.8, auc=0.9, persistent=1, recurrent=1,
                active=5, angle=0.9,
            ))
    return cells


class DescriptorGpuGateTests(unittest.TestCase):
    def test_slurm_requests_generic_gpu_and_pairs_six_tasks(self):
        smoke = (REPO_ROOT / "cluster/descriptor_attachment_gpu_smoke.slurm").read_text()
        paired = (REPO_ROOT / "cluster/descriptor_attachment_paired_matrix.slurm").read_text()
        for script in (smoke, paired):
            self.assertIn("#SBATCH --gres=gpu:1", script)
            self.assertNotIn("#SBATCH --constraint", script)
            self.assertNotIn("#SBATCH --account", script)
        self.assertIn("#SBATCH --array=0-5%6", paired)
        self.assertEqual(matrix.TASK_COUNT, 6)
        self.assertEqual(matrix.MAX_CONCURRENT_GPU_NODES, 6)
        self.assertEqual(matrix._cell_ids("task55_clean", 42), (
            "task55_clean__control__seed42",
            "task55_clean__attachment__seed42",
        ))

    def test_frozen_decision_retains_only_when_both_recipes_pass(self):
        result = decision.decide(_passing_cells())
        self.assertEqual(result["outcome"], "retain_descriptor_attachment")
        self.assertTrue(all(
            item["recipe_pass"] for item in result["recipe_summaries"].values()
        ))

        cells = _passing_cells()
        for index, cell in enumerate(cells):
            if cell.recipe == "task80_assisted" and cell.arm == "attachment":
                cells[index] = _evidence(
                    cell.recipe, cell.arm, cell.seed,
                    drift=1.2, auc=1.1, persistent=3, recurrent=2,
                    active=3, angle=1.1,
                )
        result = decision.decide(cells)
        self.assertEqual(result["outcome"], "reject_exact_descriptor_attachment")

    def test_any_ineligible_cell_stops_instead_of_rejecting_hypothesis(self):
        cells = _passing_cells()
        planted = cells[0]
        cells[0] = _evidence(
            planted.recipe, planted.arm, planted.seed, eligible=False
        )
        result = decision.decide(cells)
        self.assertEqual(result["outcome"], "stop_ineligible_result")
        self.assertFalse(result["all_twelve_evaluator_results_eligible"])

    def test_descriptor_one_shot_route_uses_existing_fresh_evaluator_hook(self):
        temporary = Path(tempfile.mkdtemp(prefix="descriptor_runtime_route_"))
        checkpoint = temporary / "best_model.pt"
        checkpoint.write_bytes(b"checkpoint")
        capability = authorization.DescriptorCompletedRunCapability(
            cell_id="task55_clean__control__seed42",
            arm="control",
            base_cell_id="task55_clean__r64__seed42",
            source_commit="a" * 40,
            checkpoint_absolute_path=str(checkpoint),
            checkpoint_sha256="b" * 64,
            checkpoint_size_bytes=len(b"checkpoint"),
            config_absolute_path=str(temporary / "config.json"),
            config_sha256="c" * 64,
            config_size_bytes=1,
            history_absolute_path=str(temporary / "history.json"),
            history_sha256="d" * 64,
            history_size_bytes=1,
            receipt_absolute_path=str(temporary / "receipt.json"),
            receipt_file_sha256="e" * 64,
            receipt_size_bytes=1,
            expected_training_arguments=MappingProxyType({}),
            expected_base_config=MappingProxyType({}),
            _capability=object(),
        )
        model = model_module.PhaseAModel(
            num_keypoints=10, base_channels=32, heatmap_res=64,
            operator_type="shared_affine",
        )
        record = {
            "absolute_path": capability.checkpoint_absolute_path,
            "file_sha256": capability.checkpoint_sha256,
            "size_bytes": capability.checkpoint_size_bytes,
            "same_open_file_descriptor_hash_and_load": True,
            "weights_only": True,
        }
        with mock.patch.object(
            authorization, "require_completed_run_capability",
            side_effect=lambda value: value,
        ):
            runtime.register_descriptor_checkpoint_load(capability, record, model=model)
            loaded = runtime._LOADED_DESCRIPTOR_CHECKPOINT.get()
            inference = {"token": object()}
            loaded["inference_record"] = inference
            bundle = {
                "case_kind": "checkpoint",
                "checkpoint_authority": "fresh_run",
                "case_id": capability.cell_id,
            }
            from keypoint_net import eval_representation as evaluator
            bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
            loaded["built_bundle_sha256"] = bundle["bundle_content_sha256"]
            runtime.arm_descriptor_checkpoint_evaluator(
                bundle, capability=capability, inference_record=inference
            )
            auth = runtime.validate_fresh_checkpoint_evaluator_authorization(bundle)
            self.assertEqual(auth["cell_id"], capability.cell_id)
            provenance = runtime.consume_fresh_checkpoint_provenance_load_receipt(
                source_commit=capability.source_commit,
                checkpoint_record={
                    "role": "checkpoint",
                    "absolute_path": capability.checkpoint_absolute_path,
                    "file_sha256": capability.checkpoint_sha256,
                    "size_bytes": capability.checkpoint_size_bytes,
                },
            )
            self.assertTrue(provenance["same_open_file_descriptor_hash_and_load"])
            with self.assertRaises(runtime.FreshCheckpointRuntimeError):
                runtime.validate_fresh_checkpoint_evaluator_authorization(bundle)


if __name__ == "__main__":
    unittest.main()
