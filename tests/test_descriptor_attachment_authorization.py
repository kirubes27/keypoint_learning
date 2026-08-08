"""Fail-closed provenance tests for descriptor experiment training cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from keypoint_net import descriptor_attachment as attachment
from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net import descriptor_attachment_calibration as calibration
from keypoint_net import prepare_descriptor_attachment_experiment as prepare_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_canonical(path: Path, document: dict) -> str:
    path.write_bytes(attachment.canonical_json_bytes(document) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _module_hashes() -> dict:
    records = {}
    for relative in (
        "keypoint_net/descriptor_attachment.py",
        "keypoint_net/descriptor_attachment_calibration.py",
        "keypoint_net/model.py",
        "keypoint_net/dataset.py",
    ):
        records[relative] = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
    records["aggregate_sha256"] = attachment.canonical_sha256(records)
    return records


def _training_arguments(root: Path, *, lambda_attach: float) -> dict:
    values = {
        "data_root": str(root),
        "object": "engineers_hammer_vray",
        "pairs_index": None,
        "indexed_mode": "development",
        "train_pairs_index": str(root / "train.json"),
        "val_pairs_index": str(root / "validation.json"),
        "test_pairs_index": None,
        "dataset_binding_sha256": "1" * 64,
        "img_size": 512,
        "num_keypoints": 10,
        "base_channels": 32,
        "heatmap_res": 64,
        "temperature": 1.0,
        "padding_mode": "reflect",
        "operator_type": "shared_affine",
        "learn_inverse_operator": False,
        "lambda_smooth": 0.0,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.0,
        "lambda_cycle": 0.0,
        "lambda_attach": lambda_attach,
        "sigma": 0.1,
        "loc_bg_threshold": 30.0,
        "num_action_classes": 0,
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "center_crop": None,
        "epochs": 1000,
        "frozen_epochs": None,
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "seed": 42,
        "output_dir": str(root / "outputs"),
        "save_every": 100,
        "log_every": 10,
        "auto_eval": False,
        "auto_eval_checkpoint": "best",
        "eval_frames_dir": None,
        "eval_max_k": 10,
    }
    assert set(values) == authorization.BOUND_TRAINING_ARGUMENTS
    return values


def _fixture(*, arm: str = "attachment"):
    root = Path(tempfile.mkdtemp(prefix="descriptor_authorization_test_")).resolve()
    commit = _source_commit()
    chosen = calibration.choose_global_lambda([
        {
            "cell_id": cell_id,
            "base_gradient_norm": 1.0,
            "attachment_gradient_norm": 1.0,
        }
        for cell_id in calibration.CALIBRATION_CELL_IDS
    ])
    receipt = {
        "schema_version": calibration.SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_weight_receipt",
        "status": chosen["status"],
        "source": {
            "commit": commit,
            "branch": authorization.EXPECTED_BRANCH,
            "module_hashes": _module_hashes(),
        },
        "loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
        "checkpoint_selector": calibration.CHECKPOINT_SELECTOR,
        "calibration_protocol": {},
        "dataset": {},
        "cell_norms": chosen["cell_norms"],
        "median_gradient_ratio": chosen["median_gradient_ratio"],
        "maximum_gradient_ratio": chosen["maximum_gradient_ratio"],
        "lambda_attach": chosen["lambda_attach"],
        "scaled_median_contribution": chosen["scaled_median_contribution"],
        "scaled_maximum_contribution": chosen["scaled_maximum_contribution"],
        "caps": chosen["caps"],
        "runtime": {},
    }
    receipt["content_hash_sha256"] = attachment.canonical_sha256(receipt)
    receipt_path = root / "weight_receipt.json"
    receipt_file_hash = _write_canonical(receipt_path, receipt)

    weight = 0.0 if arm == "control" else receipt["lambda_attach"]
    training_arguments = _training_arguments(root, lambda_attach=weight)
    cell_id = f"task55_clean__{arm}__seed42"
    manifest = {
        "schema_version": authorization.EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": authorization.EXPERIMENT_ARTIFACT_TYPE,
        "source_commit": commit,
        "source_branch": authorization.EXPECTED_BRANCH,
        "loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
        "checkpoint_selector": calibration.CHECKPOINT_SELECTOR,
        "weight_receipt": {
            "absolute_path": str(receipt_path),
            "file_sha256": receipt_file_hash,
            "content_hash_sha256": receipt["content_hash_sha256"],
        },
        "cells": [{
            "cell_id": cell_id,
            "arm": arm,
            "training_arguments": training_arguments,
        }],
    }
    manifest["content_hash_sha256"] = attachment.canonical_sha256(manifest)
    manifest_path = root / "experiment_manifest.json"
    manifest_file_hash = _write_canonical(manifest_path, manifest)
    namespace = argparse.Namespace(
        **training_arguments,
        fresh_cell_id=None,
        descriptor_cell_id=cell_id,
        descriptor_experiment_manifest=str(manifest_path),
        descriptor_experiment_manifest_sha256=manifest_file_hash,
        descriptor_weight_receipt=str(receipt_path),
        descriptor_weight_receipt_sha256=receipt_file_hash,
        descriptor_loss_spec_sha256=attachment.LOSS_SPEC_SHA256,
    )
    return root, receipt, manifest, namespace


def _completed_run_fixture():
    root, weight_receipt, manifest, namespace = _fixture()
    source = (weight_receipt["source"]["commit"], authorization.EXPECTED_BRANCH)
    base_config = {"split": {"train": {"pair_count_after_object_filter": 147}}}
    amendment = {"schema": "planted-amendment"}
    with mock.patch.object(authorization, "_source_state", return_value=source):
        binding = authorization.bind_training_namespace(REPO_ROOT, namespace)
    run_dir = Path(binding.run_directory)
    run_dir.mkdir(parents=True)
    provenance_files = []
    for name in ("matrix.json", "environment.json", "paired.slurm"):
        path = root / name
        path.write_text(name)
        provenance_files.append(authorization._file_record(path, name=name))
    runtime = {
        "python_implementation": "CPython",
        "python_version": "3.11.14",
        "pytorch_version": "2.5.1+cu121",
        "torchvision_version": "0.20.1+cu121",
        "numpy_version": "1.26.4",
        "pytorch_cuda_version": "12.1",
        "cudnn_version": 90100,
        "device_type": "cuda",
        "gpu_name": "planted generic GPU",
        "nvidia_driver_version": "570.0",
        "driver_visible_cuda_version": "12.8",
        "slurm_job_id": "12345_0",
        "primary_matrix_launch_lock": provenance_files[0],
        "environment_lock": provenance_files[1],
        "slurm_job_script": provenance_files[2],
    }
    determinism = {"seed": 42, "planted": True}
    full_command = ["python", "train.py", "--descriptor_cell_id", namespace.descriptor_cell_id]
    descriptor_binding = {
        "cell_id": binding.cell_id,
        "arm": binding.arm,
        "loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
        "experiment_manifest_file_sha256": binding.experiment_manifest_file_sha256,
        "experiment_manifest_content_sha256": binding.experiment_manifest_content_sha256,
        "weight_receipt_file_sha256": binding.weight_receipt_file_sha256,
        "weight_receipt_content_sha256": binding.weight_receipt_content_sha256,
    }
    config = {
        **vars(namespace),
        "device": "cuda",
        "source_commit": source[0],
        "checkpoint_policy": {"selection": "minimum_base_validation_loss"},
        "index_provenance": {"planted": True},
        "determinism": determinism,
        "determinism_amendment": amendment,
        "full_command": full_command,
        "runtime_environment": runtime,
        "cell_id": binding.cell_id,
        "descriptor_attachment_binding": descriptor_binding,
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    validation_epochs = [1, *range(10, 1001, 10)]
    history = [
        {
            "epoch": epoch,
            "val_base_loss": 1.0 if epoch == 1 else 1.0 + epoch / 1000.0,
            "val_attachment_loss": 2.0,
            "val_train_loss": 1.2,
        }
        for epoch in validation_epochs
    ]
    (run_dir / "history.json").write_text(json.dumps(history))
    training = manifest["cells"][0]["training_arguments"]
    checkpoint_config = {
        "source_commit": source[0],
        "cell_id": binding.cell_id,
        "descriptor_attachment_binding": descriptor_binding,
        "descriptor_loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
        "num_keypoints": training["num_keypoints"],
        "base_channels": training["base_channels"],
        "heatmap_res": training["heatmap_res"],
        "temperature": training["temperature"],
        "padding_mode": training["padding_mode"],
        "operator_type": training["operator_type"],
        "lambda_attach": training["lambda_attach"],
        "seed": training["seed"],
        "data_root": training["data_root"],
        "object": training["object"],
        "train_pairs_index": training["train_pairs_index"],
        "val_pairs_index": training["val_pairs_index"],
        "test_pairs_index": training["test_pairs_index"],
        "dataset_binding_sha256": training["dataset_binding_sha256"],
        "training_mode": training["indexed_mode"],
        "effective_epochs": training["epochs"],
        "checkpoint_policy": config["checkpoint_policy"],
        "index_provenance": config["index_provenance"],
        "determinism": determinism,
        "determinism_amendment": amendment,
    }
    torch.save({
        "epoch": 1,
        "model_state_dict": {"planted": torch.tensor([1.0])},
        "optimizer_state_dict": {},
        "loss": 1.0,
        "selection_loss_name": "base_validation_loss",
        "base_validation_loss": 1.0,
        "attachment_validation_loss": 2.0,
        "augmented_validation_loss": 1.2,
        "config": checkpoint_config,
    }, run_dir / "best_model.pt")
    patches = (
        mock.patch.object(authorization, "_source_state", return_value=source),
        mock.patch.object(
            authorization.fresh,
            "resolve_fresh_cell",
            return_value=SimpleNamespace(expected_config=base_config),
        ),
        mock.patch.object(
            authorization.fresh_roll_determinism,
            "policy_for_heatmap_resolution",
            return_value={},
        ),
        mock.patch.object(
            authorization.fresh_roll_determinism,
            "validate_determinism_record",
        ),
        mock.patch.object(
            authorization.fresh_roll_determinism,
            "validate_final_warning_evidence",
        ),
        mock.patch.object(
            authorization.fresh_roll_determinism,
            "amendment_record",
            return_value=amendment,
        ),
    )
    for patcher in patches:
        patcher.start()
    try:
        receipt_path = authorization.write_completed_run_receipt(
            REPO_ROOT,
            binding,
            device="cuda",
            optimizer_step_count=10000,
            determinism=determinism,
            nondeterminism_evidence={"planted": True},
            full_command=full_command,
            runtime_environment=runtime,
        )
        capability = authorization.authorize_completed_run(
            REPO_ROOT, receipt_path, expected_cell_id=binding.cell_id
        )
    finally:
        for patcher in reversed(patches):
            patcher.stop()
    return capability, provenance_files[0]["absolute_path"], patches


class AuthorizationTests(unittest.TestCase):
    def test_manifest_generator_freezes_exact_paired_twelve_cells(self):
        root, receipt, _, namespace = _fixture()
        data_root = REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2"
        source = (receipt["source"]["commit"], authorization.EXPECTED_BRANCH)
        with mock.patch.object(prepare_experiment, "_source_state", return_value=source):
            manifest = prepare_experiment.build_experiment_manifest(
                repo_root=REPO_ROOT,
                data_root=data_root,
                output_root=root / "future_runs",
                weight_receipt_path=Path(namespace.descriptor_weight_receipt),
                weight_receipt_file_sha256=namespace.descriptor_weight_receipt_sha256,
            )
        self.assertEqual(len(manifest["cells"]), 12)
        self.assertEqual(
            {cell["arm"] for cell in manifest["cells"]},
            {"control", "attachment"},
        )
        for cell in manifest["cells"]:
            self.assertEqual(
                set(cell["training_arguments"]),
                authorization.BOUND_TRAINING_ARGUMENTS,
            )
            expected = 0.0 if cell["arm"] == "control" else receipt["lambda_attach"]
            self.assertEqual(cell["training_arguments"]["lambda_attach"], expected)

    def test_attachment_and_exact_zero_control_bind(self):
        for arm in ("attachment", "control"):
            _, receipt, _, namespace = _fixture(arm=arm)
            with mock.patch.object(
                authorization,
                "_source_state",
                return_value=(receipt["source"]["commit"], authorization.EXPECTED_BRANCH),
            ):
                binding = authorization.bind_training_namespace(REPO_ROOT, namespace)
            self.assertEqual(binding.arm, arm)
            if arm == "control":
                self.assertEqual(binding.lambda_attach, 0.0)
            else:
                self.assertEqual(binding.lambda_attach, receipt["lambda_attach"])

    def test_lambda_manifest_spec_and_selector_changes_fail(self):
        _, receipt, _, namespace = _fixture()
        source = (receipt["source"]["commit"], authorization.EXPECTED_BRANCH)
        with mock.patch.object(authorization, "_source_state", return_value=source):
            namespace.lambda_attach *= 2.0
            with self.assertRaises(ValueError):
                authorization.bind_training_namespace(REPO_ROOT, namespace)

        _, receipt, _, namespace = _fixture()
        namespace.descriptor_loss_spec_sha256 = "0" * 64
        with mock.patch.object(authorization, "_source_state", return_value=source):
            with self.assertRaises(ValueError):
                authorization.bind_training_namespace(REPO_ROOT, namespace)

        root, receipt, manifest, namespace = _fixture()
        manifest["checkpoint_selector"] = "minimum_augmented_validation_loss"
        manifest["content_hash_sha256"] = attachment.canonical_sha256(
            {key: value for key, value in manifest.items() if key != "content_hash_sha256"}
        )
        namespace.descriptor_experiment_manifest_sha256 = _write_canonical(
            root / "experiment_manifest.json", manifest
        )
        with mock.patch.object(authorization, "_source_state", return_value=source):
            with self.assertRaises(ValueError):
                authorization.bind_training_namespace(REPO_ROOT, namespace)

    def test_missing_or_changed_weight_receipt_fails(self):
        _, receipt, _, namespace = _fixture()
        source = (receipt["source"]["commit"], authorization.EXPECTED_BRANCH)
        namespace.descriptor_weight_receipt_sha256 = "0" * 64
        with mock.patch.object(authorization, "_source_state", return_value=source):
            with self.assertRaises(ValueError):
                authorization.bind_training_namespace(REPO_ROOT, namespace)

    def test_unbound_descriptor_arguments_are_rejected(self):
        baseline = argparse.Namespace(
            lambda_attach=0.0,
            descriptor_experiment_manifest=None,
            descriptor_experiment_manifest_sha256=None,
            descriptor_weight_receipt=None,
            descriptor_weight_receipt_sha256=None,
            descriptor_loss_spec_sha256=None,
        )
        authorization.reject_unbound_descriptor_arguments(baseline)
        baseline.lambda_attach = 0.1
        with self.assertRaises(ValueError):
            authorization.reject_unbound_descriptor_arguments(baseline)

    def test_completed_run_receipt_authorizes_exact_checkpoint_and_launch_chain(self):
        capability, matrix_path, _ = _completed_run_fixture()
        self.assertEqual(capability.cell_id, "task55_clean__attachment__seed42")
        self.assertEqual(Path(capability.checkpoint_absolute_path).name, "best_model.pt")
        self.assertRegex(capability.receipt_file_sha256, r"^[0-9a-f]{64}$")
        self.assertTrue(Path(matrix_path).is_file())


if __name__ == "__main__":
    unittest.main()
