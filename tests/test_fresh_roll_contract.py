"""Focused tests for the fresh roll cell/receipt/runtime boundary.

The project deletion policy forbids automatic cleanup, so these tests leave
their small uniquely named fixtures below the operating-system temp root.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keypoint_net import eval_representation as evaluator
from keypoint_net import model as model_module
from keypoint_net import representation_fresh_checkpoint_authorization as fresh
from keypoint_net import representation_fresh_checkpoint_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2"


def _model_for(cell: fresh.FrozenFreshCell) -> model_module.PhaseAModel:
    training = cell.expected_config["training"]
    return model_module.PhaseAModel(
        num_keypoints=training["num_keypoints"],
        base_channels=training["base_channels"],
        temperature=training["temperature"],
        num_action_classes=0,
        padding_mode=training["padding_mode"],
        operator_type=training["operator_type"],
        learn_inverse_operator=training["learn_inverse_operator"],
        heatmap_res=training["heatmap_res"],
    )


def _completed_fixture(cell_id: str):
    output_root = Path(tempfile.mkdtemp(prefix="fresh_roll_contract_test_"))
    arguments = fresh.training_arguments(
        REPO_ROOT,
        cell_id,
        data_root=DATA_ROOT,
        output_root=output_root,
    )
    binding = fresh.bind_training_namespace(
        REPO_ROOT, argparse.Namespace(**arguments)
    )
    run_dir = Path(binding.run_directory)
    run_dir.mkdir()
    config = {
        "cell_id": cell_id,
        "source_commit": binding.cell.source_commit,
        "fresh_run_contract": dict(binding.cell.expected_config),
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "history.json").write_text(json.dumps([{"epoch": 1, "val_loss": 1.0}]))
    model = _model_for(binding.cell)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "loss": 1.0,
            "config": config,
        },
        run_dir / "best_model.pt",
    )
    receipt = fresh.write_completed_run_receipt(
        REPO_ROOT,
        binding,
        device="cpu",
        optimizer_step_count=1,
        determinism={
            "seed": binding.cell.seed,
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "data_loader_workers": 0,
        },
    )
    return binding, receipt, model


class FreshRollContractTests(unittest.TestCase):
    def test_frozen_cells_construct_exact_64_and_128_heads(self) -> None:
        clean = fresh.resolve_fresh_cell(REPO_ROOT, "task55_clean__r64__seed42")
        assisted = fresh.resolve_fresh_cell(
            REPO_ROOT, "task80_assisted__r128__seed44"
        )
        self.assertEqual(clean.expected_config["training"]["lambda_inv"], 0.0)
        self.assertEqual(assisted.expected_config["training"]["lambda_inv"], 0.5)
        model64 = _model_for(clean)
        model128 = _model_for(assisted)
        head64 = sum(parameter.numel() for parameter in model64.extractor.heatmap_head.parameters())
        head128 = sum(parameter.numel() for parameter in model128.extractor.heatmap_head.parameters())
        head128 += sum(parameter.numel() for parameter in model128.extractor.head_upsample.parameters())
        self.assertEqual((head64, head128, head128 - head64), (1290, 74570, 73280))
        with torch.inference_mode():
            self.assertEqual(model64.extractor(torch.zeros(1, 3, 512, 512))[1].shape[-2:], (64, 64))
            self.assertEqual(model128.extractor(torch.zeros(1, 3, 512, 512))[1].shape[-2:], (128, 128))

    def test_wrong_cell_argument_rejects_before_cell_output_exists(self) -> None:
        output_root = Path(tempfile.mkdtemp(prefix="fresh_roll_reject_test_"))
        arguments = fresh.training_arguments(
            REPO_ROOT,
            "task55_clean__r64__seed42",
            data_root=DATA_ROOT,
            output_root=output_root,
        )
        arguments["heatmap_res"] = 128
        with self.assertRaisesRegex(
            fresh.FreshCheckpointAuthorizationError,
            "heatmap_res",
        ):
            fresh.bind_training_namespace(REPO_ROOT, argparse.Namespace(**arguments))
        self.assertFalse((output_root / "task55_clean__r64__seed42").exists())

    def test_completed_receipt_authorizes_then_checkpoint_tamper_rejects(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")
        capability = fresh.authorize_completed_fresh_run(
            REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
        )
        self.assertEqual(capability.checkpoint_sha256,
                         json.loads(receipt.read_text())["files"]["checkpoint"]["sha256"])
        with (Path(binding.run_directory) / "best_model.pt").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(
            fresh.FreshCheckpointAuthorizationError,
            "size mismatch|hash mismatch",
        ):
            fresh.authorize_completed_fresh_run(
                REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
            )

    def test_safe_reload_routes_small_bundle_through_production_evaluator(self) -> None:
        binding, receipt, original_model = _completed_fixture(
            "task80_assisted__r64__seed42"
        )
        model, capability, load_record = runtime.load_authorized_fresh_checkpoint(
            REPO_ROOT, receipt, cell_id=binding.cell.cell_id
        )
        self.assertEqual(load_record["file_sha256"], capability.checkpoint_sha256)
        for key, value in original_model.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        with torch.inference_mode():
            points, logits = model.extractor(torch.zeros(2, 3, 512, 512))
        points_array = points.reshape(2, 10, 2).numpy().copy()
        logits_array = logits.numpy().copy()
        masks = np.zeros((2, 512, 512), dtype=bool)
        masks[:, 128:384, 128:384] = True
        bundle = runtime.build_fresh_checkpoint_bundle(
            REPO_ROOT,
            capability,
            model,
            points=points_array,
            logits=logits_array,
            masks=masks,
            frame_ids=[0, 3],
            physical_states=[{"theta_deg": 0.0}, {"theta_deg": 6.0}],
            pair_rows=[{
                "pair_id": "validation:0:3",
                "source_frame": 0,
                "target_frame": 3,
                "direction": "forward",
                "stride": 3,
                "partition": "validation",
            }],
            partition="validation",
            full_primary_roll=False,
        )
        runtime.arm_fresh_checkpoint_evaluator(
            bundle,
            capability=capability,
            inference_record={
                "state_unchanged": True,
                "gradients_enabled_inside_inference": False,
            },
        )
        result = evaluator.evaluate_bundle(bundle)
        self.assertEqual(result["case_id"], binding.cell.cell_id)
        self.assertTrue(
            result["checkpoint_authorization"]["checkpoint_evaluation_authorized"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
