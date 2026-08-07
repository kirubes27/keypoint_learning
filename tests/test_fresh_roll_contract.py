"""Focused tests for the fresh roll cell/receipt/runtime boundary.

The project deletion policy forbids automatic cleanup, so these tests leave
their small uniquely named fixtures below the operating-system temp root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from keypoint_net import eval_representation as evaluator
from keypoint_net import fresh_roll_determinism as fresh_det
from keypoint_net import model as model_module
from keypoint_net import representation_fresh_checkpoint_authorization as fresh
from keypoint_net import representation_fresh_checkpoint_runtime as runtime
from keypoint_net.diagnostics import run_fresh_roll_cuda_smoke as cuda_smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2"
KEYPOINT_NET_ROOT = REPO_ROOT / "keypoint_net"
if str(KEYPOINT_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_NET_ROOT))
from keypoint_net import train as train_module


def fresh_provenance_roles() -> set[str]:
    return {
        "checkpoint",
        "checkpoint_config",
        "checkpoint_metadata",
        "completed_run_receipt",
    }


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
    parent = Path(tempfile.mkdtemp(prefix="fresh_roll_contract_test_")).resolve()
    matrix_root = parent / "matrix"
    output_root = matrix_root / "runs"
    output_root.mkdir(parents=True)
    (matrix_root / "evaluations").mkdir()
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
    full_command = [
        sys.executable,
        "keypoint_net/train.py",
        "--fresh_cell_id",
        cell_id,
    ]
    environment_lock = parent / "FRESH_ROLL_ENVIRONMENT.json"
    environment_lock.write_text("{}\n", encoding="utf-8")
    slurm_script = REPO_ROOT / "cluster/fresh_roll_primary_matrix.slurm"

    def record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "absolute_path": str(path.absolute()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    matrix_document = {
        "schema_version": "roll_head_package_primary_matrix_launch.v1",
        "artifact_type": "roll_head_package_primary_matrix_launch_lock",
        "matrix_root": str(matrix_root),
        "source_commit": binding.cell.source_commit,
        "source_branch": fresh.EXPECTED_BRANCH,
        "slurm_script": record(slurm_script),
        "environment_lock": record(environment_lock),
        "primary_cell_ids_by_task_index": [
            f"{recipe}__r{head}__seed{seed}"
            for recipe in ("task55_clean", "task80_assisted")
            for head in (64, 128)
            for seed in (42, 43, 44)
        ],
        "task_count": 12,
        "gpu_count_per_task": 1,
        "max_concurrent_gpu_nodes": 1,
        "output_layout": {
            "runs_directory": str(output_root),
            "evaluations_directory": str(matrix_root / "evaluations"),
            "evaluation_filename_template": "{cell_id}.json",
        },
        "submission": {
            "performed_by_prepare_command": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
        },
    }
    matrix_document["content_hash_sha256"] = fresh.canonical_sha256(
        matrix_document
    )
    matrix_lock = matrix_root / "PRIMARY_MATRIX_LAUNCH_LOCK.json"
    matrix_lock.write_bytes(fresh.canonical_json_bytes(matrix_document, newline=True))
    runtime_environment = {
        "python_implementation": "CPython",
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": str(np.__version__),
        "pytorch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": "cpu",
        "gpu_name": None,
        "nvidia_driver_version": None,
        "driver_visible_cuda_version": None,
        "slurm_job_id": "synthetic-test-job",
        "primary_matrix_launch_lock": record(matrix_lock),
        "environment_lock": record(environment_lock),
        "slurm_job_script": record(slurm_script),
    }
    policy = fresh_det.policy_for_heatmap_resolution(
        REPO_ROOT,
        binding.cell.expected_config["training"]["heatmap_res"],
    )
    determinism = fresh_det.configure_torch_determinism(
        binding.cell.seed,
        policy,
    )
    determinism["data_loader_workers"] = 0
    nondeterminism_evidence = fresh_det.finalize_warning_evidence(
        fresh_det.new_warning_evidence(policy),
        policy=policy,
        device_type="cpu",
    )
    config = {
        **arguments,
        "cell_id": cell_id,
        "source_commit": binding.cell.source_commit,
        "training_mode": "development",
        "effective_epochs": binding.cell.expected_config["training"]["epochs"],
        "checkpoint_policy": binding.cell.expected_config["training"][
            "checkpoint_policy"
        ],
        "fresh_run_contract": dict(binding.cell.expected_config),
        "determinism": determinism,
        "determinism_amendment": fresh_det.amendment_record(REPO_ROOT),
        "full_command": full_command,
        "runtime_environment": runtime_environment,
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    validation_epochs = [1, *range(10, 1001, 10)]
    history = [
        {"epoch": epoch, "val_loss": 1.0 + (epoch - 1) / 10000.0}
        for epoch in validation_epochs
    ]
    (run_dir / "history.json").write_text(json.dumps(history))
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
        optimizer_step_count=10000,
        determinism=determinism,
        nondeterminism_evidence=nondeterminism_evidence,
        full_command=full_command,
        runtime_environment=runtime_environment,
    )
    return binding, receipt, model


def _refresh_receipt_file_record(receipt: Path, role: str) -> None:
    document = json.loads(receipt.read_text(encoding="utf-8"))
    path = Path(document["files"][role]["absolute_path"])
    data = path.read_bytes()
    document["files"][role] = {
        "absolute_path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    document.pop("content_hash_sha256")
    document["content_hash_sha256"] = fresh.canonical_sha256(document)
    receipt.write_bytes(fresh.canonical_json_bytes(document, newline=True))


def _rewrite_receipt(receipt: Path, mutate) -> None:
    document = json.loads(receipt.read_text(encoding="utf-8"))
    mutate(document)
    document.pop("content_hash_sha256")
    document["content_hash_sha256"] = fresh.canonical_sha256(document)
    receipt.write_bytes(fresh.canonical_json_bytes(document, newline=True))


class FreshRollContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        runtime._LOADED_CHECKPOINT.set(None)
        runtime._PENDING_EVALUATOR.set(None)
        runtime._PENDING_PROVENANCE.set(None)

    def test_smoke_fixed_batch_inference_forces_eval_mode(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Dropout2d(p=0.5),
        )
        model.train()
        self.assertIs(cuda_smoke._fixed_batch_eval(model), model)
        self.assertFalse(model.training)
        self.assertTrue(all(not module.training for module in model.modules()))

    def test_authorization_git_disables_replacements_and_sanitizes_environment(self) -> None:
        completed = mock.Mock(stdout="value\n")
        with (
            mock.patch.dict(
                os.environ,
                {"GIT_DIR": "/tmp/hostile", "GIT_CONFIG_COUNT": "1"},
            ),
            mock.patch.object(
                fresh.subprocess, "run", return_value=completed
            ) as run,
        ):
            self.assertEqual(fresh._git(REPO_ROOT, "rev-parse", "HEAD"), "value")
        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command[:2], ["git", "--no-replace-objects"])
        self.assertNotIn("GIT_DIR", environment)
        self.assertNotIn("GIT_CONFIG_COUNT", environment)
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")

    def test_runtime_environment_requires_and_captures_full_launch_chain(self) -> None:
        parent = Path(tempfile.mkdtemp(prefix="fresh_runtime_env_test_"))
        paths = {}
        for name in ("matrix", "environment", "slurm"):
            path = parent / f"{name}.txt"
            path.write_text(f"{name}\n", encoding="utf-8")
            paths[name] = path

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        environment = {
            "SLURM_JOB_ID": "12345",
            "FRESH_ROLL_PRIMARY_MATRIX_LOCK": str(paths["matrix"]),
            "FRESH_ROLL_PRIMARY_MATRIX_LOCK_SHA256": digest(paths["matrix"]),
            "FRESH_ROLL_ENV_LOCK": str(paths["environment"]),
            "FRESH_ROLL_ENV_LOCK_SHA256": digest(paths["environment"]),
            "FRESH_ROLL_SLURM_SCRIPT": str(paths["slurm"]),
            "FRESH_ROLL_SLURM_SCRIPT_SHA256": digest(paths["slurm"]),
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            captured = train_module._runtime_environment(torch.device("cpu"))
        self.assertEqual(captured["slurm_job_id"], "12345")
        self.assertEqual(
            captured["primary_matrix_launch_lock"]["sha256"],
            environment["FRESH_ROLL_PRIMARY_MATRIX_LOCK_SHA256"],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_ID"):
                train_module._runtime_environment(torch.device("cpu"))

    def test_head_specific_warning_policy_is_hash_bound_and_exact(self) -> None:
        policy64 = fresh_det.policy_for_heatmap_resolution(REPO_ROOT, 64)
        policy128 = fresh_det.policy_for_heatmap_resolution(REPO_ROOT, 128)
        self.assertEqual(
            policy64["allowed_operations"],
            [fresh_det.REFLECTION_OPERATION],
        )
        self.assertEqual(
            policy128["allowed_operations"],
            [fresh_det.REFLECTION_OPERATION, fresh_det.BILINEAR_OPERATION],
        )
        self.assertEqual(policy64["amendment"], policy128["amendment"])
        self.assertRegex(policy64["amendment"]["file_sha256"], r"^[0-9a-f]{64}$")

    def test_warning_gate_steps_only_after_allowed_warning(self) -> None:
        class WarningLoss:
            def __init__(self, operation: str):
                self.operation = operation

            def backward(self) -> None:
                warnings.warn(
                    f"{self.operation} does not have a deterministic implementation",
                    UserWarning,
                )

        policy64 = fresh_det.policy_for_heatmap_resolution(REPO_ROOT, 64)
        allowed_optimizer = mock.Mock()
        allowed_evidence = fresh_det.new_warning_evidence(policy64)
        fresh_det.backward_and_step(
            WarningLoss(fresh_det.REFLECTION_OPERATION),
            allowed_optimizer,
            policy=policy64,
            evidence=allowed_evidence,
        )
        allowed_optimizer.step.assert_called_once_with()
        self.assertEqual(
            allowed_evidence["observed_operation_counts"],
            {fresh_det.REFLECTION_OPERATION: 1},
        )

        rejected_optimizer = mock.Mock()
        rejected_evidence = fresh_det.new_warning_evidence(policy64)
        with self.assertRaisesRegex(
            fresh_det.FreshRollDeterminismError,
            fresh_det.BILINEAR_OPERATION,
        ):
            fresh_det.backward_and_step(
                WarningLoss(fresh_det.BILINEAR_OPERATION),
                rejected_optimizer,
                policy=policy64,
                evidence=rejected_evidence,
            )
        rejected_optimizer.step.assert_not_called()

    def test_cuda_positive_control_is_required_but_128_bilinear_is_optional(self) -> None:
        policy128 = fresh_det.policy_for_heatmap_resolution(REPO_ROOT, 128)
        empty = fresh_det.new_warning_evidence(policy128)
        with self.assertRaisesRegex(
            fresh_det.FreshRollDeterminismError,
            "positive-control",
        ):
            fresh_det.finalize_warning_evidence(
                empty,
                policy=policy128,
                device_type="cuda",
            )
        observed = fresh_det.new_warning_evidence(policy128)
        observed["backward_calls"] = 1
        observed["captured_warning_count"] = 1
        observed["observed_operation_counts"][fresh_det.REFLECTION_OPERATION] = 1
        completed = fresh_det.finalize_warning_evidence(
            observed,
            policy=policy128,
            device_type="cuda",
        )
        self.assertTrue(completed["positive_control_observed"])
        self.assertEqual(
            completed["observed_operation_counts"][fresh_det.BILINEAR_OPERATION],
            0,
        )

    def test_nvidia_smi_provenance_accepts_legacy_and_kmd_headers(self) -> None:
        headers = (
            (
                "NVIDIA-SMI 580.173.02 Driver Version: 580.173.02 "
                "CUDA Version: 13.0\n",
                ("580.173.02", "13.0"),
            ),
            (
                "NVIDIA-SMI 610.43.02 KMD Version: 610.43.02 "
                "CUDA UMD Version: 13.3\n",
                ("610.43.02", "13.3"),
            ),
        )
        for stdout, expected in headers:
            with self.subTest(stdout=stdout), mock.patch.object(
                train_module.subprocess,
                "run",
                return_value=mock.Mock(stdout=stdout),
            ):
                self.assertEqual(expected, train_module._nvidia_driver_facts())

        with mock.patch.object(
            train_module.subprocess,
            "run",
            return_value=mock.Mock(stdout="NVIDIA-SMI unknown format\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot parse"):
                train_module._nvidia_driver_facts()

    def test_real_corpus_dry_run_constructs_exact_64_and_128_data_plans(self) -> None:
        output_root = Path(tempfile.mkdtemp(prefix="fresh_roll_dry_run_test_"))
        for cell_id in (
            "task55_clean__r64__seed42",
            "task55_clean__r128__seed42",
        ):
            arguments = fresh.training_arguments(
                REPO_ROOT,
                cell_id,
                data_root=DATA_ROOT,
                output_root=output_root,
            )
            namespace = argparse.Namespace(**arguments)
            binding = fresh.bind_training_namespace(REPO_ROOT, namespace)
            plan = train_module._prepare_training_data(
                namespace,
                include_backward=False,
            )
            train_module._validate_fresh_data_plan(
                plan,
                binding,
                data_root=str(DATA_ROOT),
            )
            self.assertEqual(len(plan.train_dataset), 147)
            self.assertEqual(len(plan.val_dataset), 21)
            self.assertFalse(Path(binding.run_directory).exists())

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

    def test_incomplete_schedule_and_false_selection_reject(self) -> None:
        for mutation, expected in (
            (
                "optimizer_steps",
                "optimizer step count differs",
            ),
            (
                "selection_false",
                "selection use is not authorized",
            ),
            (
                "history_incomplete",
                "history validation-entry count differs",
            ),
        ):
            binding, receipt, _ = _completed_fixture(
                "task55_clean__r64__seed42"
            )
            if mutation == "optimizer_steps":
                _rewrite_receipt(
                    receipt,
                    lambda document: document["execution"].update(
                        {"optimizer_step_count": 9999}
                    ),
                )
            elif mutation == "selection_false":
                _rewrite_receipt(
                    receipt,
                    lambda document: document["execution"].update(
                        {"selection_use_authorized": False}
                    ),
                )
            else:
                history_path = Path(binding.run_directory) / "history.json"
                history = json.loads(history_path.read_text(encoding="utf-8"))
                history_path.write_text(json.dumps(history[:-1]), encoding="utf-8")
                _refresh_receipt_file_record(receipt, "history")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                fresh.FreshCheckpointAuthorizationError, expected
            ):
                fresh.authorize_completed_fresh_run(
                    REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
                )

    def test_best_checkpoint_epoch_and_loss_must_match_first_history_minimum(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")
        checkpoint_path = Path(binding.run_directory) / "best_model.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        payload["epoch"] = 10
        torch.save(payload, checkpoint_path)
        _refresh_receipt_file_record(receipt, "checkpoint")
        with self.assertRaisesRegex(
            fresh.FreshCheckpointAuthorizationError,
            "checkpoint epoch is not the first validation minimum",
        ):
            runtime.load_authorized_fresh_checkpoint(
                REPO_ROOT, receipt, cell_id=binding.cell.cell_id
            )

    def test_live_environment_lock_tamper_rejects_receipt_chain(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")
        document = json.loads(receipt.read_text(encoding="utf-8"))
        environment_lock = Path(
            document["execution"]["runtime_environment"]["environment_lock"][
                "absolute_path"
            ]
        )
        environment_lock.write_text('{"tampered":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            fresh.FreshCheckpointAuthorizationError,
            "environment lock size mismatch|environment lock hash mismatch",
        ):
            fresh.authorize_completed_fresh_run(
                REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
            )

    def test_cuda_indexed_device_requires_complete_gpu_evidence(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")

        def mutate(document: dict) -> None:
            document["execution"]["device"] = "cuda:0"
            document["execution"]["runtime_environment"]["device_type"] = "cuda"

        _rewrite_receipt(receipt, mutate)
        with mock.patch.object(
            fresh_det, "validate_final_warning_evidence", return_value=None
        ):
            with self.assertRaisesRegex(
                fresh.FreshCheckpointAuthorizationError,
                "CUDA environment .* is invalid",
            ):
                fresh.authorize_completed_fresh_run(
                    REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
                )

    def test_config_and_history_use_same_captured_bytes_and_symlinks_reject(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")
        original_read_bytes = Path.read_bytes

        def forbid_config_history_reopen(path: Path) -> bytes:
            if path.name in {"config.json", "history.json"}:
                raise AssertionError("config/history path was reopened")
            return original_read_bytes(path)

        with mock.patch.object(Path, "read_bytes", forbid_config_history_reopen):
            capability = fresh.authorize_completed_fresh_run(
                REPO_ROOT, receipt, expected_cell_id=binding.cell.cell_id
            )
        self.assertEqual(capability.cell_id, binding.cell.cell_id)

        symlink_root = Path(tempfile.mkdtemp(prefix="fresh_symlink_reject_test_"))
        run_directory = symlink_root / "run"
        run_directory.mkdir()
        target = symlink_root / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        symlink = run_directory / "history.json"
        symlink.symlink_to(target)
        data = target.read_bytes()
        with self.assertRaisesRegex(
            fresh.FreshCheckpointAuthorizationError, "cannot open history"
        ):
            fresh._validated_run_file(
                {
                    "absolute_path": str(symlink),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                },
                name="history",
                run_directory=run_directory,
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
        images = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(2)]
        points_array, logits_array, inference = runtime._infer(
            model, images, heatmap_res=64
        )
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
            inference_record=inference,
        )
        result = evaluator.evaluate_bundle(bundle)
        self.assertEqual(result["case_id"], binding.cell.cell_id)
        self.assertTrue(
            result["checkpoint_authorization"]["checkpoint_evaluation_authorized"]
        )
        self.assertEqual(
            {record["role"] for record in result["provenance"]["external_files"]},
            fresh_provenance_roles(),
        )

    def test_bundle_rejects_other_model_and_arbitrary_inference_arrays(self) -> None:
        binding, receipt, _ = _completed_fixture("task55_clean__r64__seed42")
        model, capability, _ = runtime.load_authorized_fresh_checkpoint(
            REPO_ROOT, receipt, cell_id=binding.cell.cell_id
        )
        images = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(2)]
        points, logits, _ = runtime._infer(model, images, heatmap_res=64)
        masks = np.zeros((2, 512, 512), dtype=bool)
        masks[:, 128:384, 128:384] = True
        common = {
            "masks": masks,
            "frame_ids": [0, 3],
            "physical_states": [{"theta_deg": 0.0}, {"theta_deg": 6.0}],
            "pair_rows": [{
                "pair_id": "validation:0:3",
                "source_frame": 0,
                "target_frame": 3,
                "direction": "forward",
                "stride": 3,
                "partition": "validation",
            }],
            "partition": "validation",
            "full_primary_roll": False,
        }
        other_model = _model_for(binding.cell)
        with self.assertRaisesRegex(
            runtime.FreshCheckpointRuntimeError, "exact loaded checkpoint model"
        ):
            runtime.build_fresh_checkpoint_bundle(
                REPO_ROOT, capability, other_model,
                points=points, logits=logits, **common,
            )
        arbitrary_points = points.copy()
        arbitrary_points[0, 0, 0] += 0.25
        with self.assertRaisesRegex(
            runtime.FreshCheckpointRuntimeError,
            "arrays/model differ from the live inference context",
        ):
            runtime.build_fresh_checkpoint_bundle(
                REPO_ROOT, capability, model,
                points=arbitrary_points, logits=logits, **common,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
