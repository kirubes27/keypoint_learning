"""Focused authorization/runtime-boundary tests with no real checkpoint open."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from keypoint_net import representation_checkpoint_authorization as authorization
from keypoint_net import representation_checkpoint_replay as replay_registry
from keypoint_net import representation_checkpoint_runtime as runtime


SOURCE_COMMIT = "a" * 40
REVIEWED_COMMIT = "9" * 40
STATE_SHA256 = "1" * 64
RUNTIME_MANIFEST_SHA256 = "2" * 64
RUNTIME_MANIFEST_CONTENT_SHA256 = "3" * 64
PREFLIGHT_SHA256 = "4" * 64
EXECUTION_AUTHORIZATION_SHA256 = "5" * 64
FABLE_REVIEW_SHA256 = "6" * 64


def _runtime_authorization(
    *,
    task_id: int = 20,
) -> authorization.CheckpointRuntimeAuthorization:
    binding = authorization._TASK_BINDINGS[task_id]
    run_directory = str(binding["run_directory"])
    task20_bound = task_id in {55, 80}
    return authorization.CheckpointRuntimeAuthorization(
        task_id=task_id,
        fixture_id=str(binding["fixture_id"]),
        role=str(binding["role"]),
        runtime_source_commit=SOURCE_COMMIT,
        repository_root="/repository",
        checkpoint_absolute_path=f"{run_directory}/best_model.pt",
        checkpoint_sha256=str(binding["checkpoint_sha256"]),
        checkpoint_size_bytes=int(binding["checkpoint_size_bytes"]),
        config_absolute_path=f"{run_directory}/config.json",
        config_sha256=str(binding["config_sha256"]),
        config_size_bytes=int(binding["config_size_bytes"]),
        history_absolute_path=f"{run_directory}/history.json",
        history_sha256=str(binding["history_sha256"]),
        history_size_bytes=int(binding["history_size_bytes"]),
        dataset_root="/dataset",
        reviewed_candidate_commit=REVIEWED_COMMIT,
        runtime_source_manifest_file_sha256=RUNTIME_MANIFEST_SHA256,
        runtime_source_manifest_content_hash_sha256=(
            RUNTIME_MANIFEST_CONTENT_SHA256
        ),
        checkpoint_hash_preflight_file_sha256=PREFLIGHT_SHA256,
        checkpoint_execution_authorization_file_sha256=(
            EXECUTION_AUTHORIZATION_SHA256
        ),
        fable_execution_review_file_sha256=FABLE_REVIEW_SHA256,
        expected_learn_inverse_operator=bool(
            binding["learn_inverse_operator"]
        ),
        task20_gate_result_file_sha256=("7" * 64 if task20_bound else None),
        task20_gate_result_content_hash_sha256=(
            "8" * 64 if task20_bound else None
        ),
        task20_gate_source_commit=(
            REVIEWED_COMMIT if task20_bound else None
        ),
        _capability=authorization._VERIFIED_RUNTIME_CAPABILITY,
    )


def _checkpoint_load_record(
    runtime: authorization.CheckpointRuntimeAuthorization,
) -> dict:
    return {
        "checkpoint_epoch": 100,
        "checkpoint_loss": 0.25,
        "state_key_count": 42,
        "embedded_config_verified": True,
        "checkpoint_absolute_path": runtime.checkpoint_absolute_path,
        "checkpoint_size_bytes": runtime.checkpoint_size_bytes,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "load_device": "cpu",
        "weights_only": True,
        "same_open_file_descriptor_hash_and_load": True,
        "unsafe_fallback_used": False,
    }


def _inference_record() -> dict:
    return {
        "frame_count": 180,
        "channel_count": 10,
        "batch_size": 8,
        "device": "cpu",
        "model_training_flag": False,
        "gradients_enabled_inside_inference": False,
        "state_sha256_before": STATE_SHA256,
        "state_sha256_after": STATE_SHA256,
        "state_unchanged": True,
    }


def _compact_float32_record(shape: list[int], label: str) -> dict:
    record = {
        "schema_version": "representation_compact_array.v1",
        "encoding": "raw_float32_le_base64",
        "dtype": "float32",
        "shape": shape,
        "order": "C",
        # The authorization layer binds the complete record; the array-codec
        # validator later decodes the full payload.  A tiny payload keeps this
        # boundary test independent of a 29 MB synthetic logits allocation.
        "payload_base64": label,
        "payload_nbytes": 4
        * __import__("math").prod(shape),
        "payload_sha256": hashlib.sha256(label.encode("ascii")).hexdigest(),
    }
    record["content_sha256"] = hashlib.sha256(
        authorization._canonical_json_bytes(record)
    ).hexdigest()
    return record


def _bundle(
    runtime: authorization.CheckpointRuntimeAuthorization,
) -> dict:
    bundle = {
        "case_id": runtime.fixture_id,
        "case_kind": "checkpoint",
        "provenance": {
            "source_commit": runtime.runtime_source_commit,
            "external_files": [
                {
                    "role": "checkpoint",
                    "absolute_path": runtime.checkpoint_absolute_path,
                    "file_sha256": runtime.checkpoint_sha256,
                    "size_bytes": runtime.checkpoint_size_bytes,
                },
                {
                    "role": "checkpoint_config",
                    "absolute_path": runtime.config_absolute_path,
                    "file_sha256": runtime.config_sha256,
                    "size_bytes": runtime.config_size_bytes,
                },
                {
                    "role": "checkpoint_metadata",
                    "absolute_path": runtime.history_absolute_path,
                    "file_sha256": runtime.history_sha256,
                    "size_bytes": runtime.history_size_bytes,
                },
            ],
        },
        "evaluation": {
            "points": _compact_float32_record([180, 10, 2], "cG9pbnRz"),
            "logits": _compact_float32_record(
                [180, 10, 64, 64],
                "bG9naXRz",
            ),
        },
    }
    bundle["bundle_content_sha256"] = hashlib.sha256(
        authorization._canonical_json_bytes(bundle)
    ).hexdigest()
    return bundle


def _candidate(task_id: int) -> replay_registry.CheckpointCandidate:
    binding = authorization._TASK_BINDINGS[task_id]
    return replay_registry.CheckpointCandidate(
        fixture_id=str(binding["fixture_id"]),
        task_id=task_id,
        role=str(binding["role"]),
        absolute_path=f"{binding['run_directory']}/best_model.pt",
        relative_path="best_model.pt",
        expected_sha256=str(binding["checkpoint_sha256"]),
        expected_size_bytes=int(binding["checkpoint_size_bytes"]),
    )


def _synthetic_external_config(task_id: int) -> dict:
    task = runtime._EXPECTED_TASK_CONFIG[task_id]
    return {
        "data_root": "./_tdw_world_z_roll_base_panel_512_v2",
        "object": "engineers_hammer_vray",
        "pairs_index": (
            "_tdw_world_z_roll_base_panel_512_v2/indices/"
            "pairs_skip3_cyclic.json"
        ),
        "img_size": 512,
        "num_keypoints": 10,
        "base_channels": 32,
        "temperature": 1.0,
        "padding_mode": "reflect",
        "operator_type": "shared_affine",
        "learn_inverse_operator": False,
        "num_action_classes": 2,
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "center_crop": None,
        "seed": 42,
        "sigma": 0.1,
        "loc_bg_threshold": 30.0,
        **{
            name: task[name]
            for name in (
                "lambda_smooth",
                "lambda_disp",
                "lambda_ent",
                "lambda_act",
                "lambda_loc",
                "lambda_inv",
                "lambda_cycle",
                "learn_inverse_operator_effective",
            )
        },
    }


def _synthetic_embedded_config(
    *,
    external_config: dict,
    architecture: dict,
) -> dict:
    return {
        "num_keypoints": architecture["num_keypoints"],
        "base_channels": architecture["base_channels"],
        "heatmap_res": architecture["heatmap_res"],
        "temperature": architecture["temperature"],
        "padding_mode": architecture["padding_mode"],
        "operator_type": architecture["operator_type"],
        "learn_inverse_operator": architecture["learn_inverse_operator"],
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "lambda_smooth": external_config["lambda_smooth"],
        "lambda_disp": external_config["lambda_disp"],
        "lambda_ent": external_config["lambda_ent"],
        "lambda_act": external_config["lambda_act"],
        "lambda_loc": external_config["lambda_loc"],
        "lambda_inv": external_config["lambda_inv"],
        "lambda_cycle": external_config["lambda_cycle"],
        "num_action_classes": 0,
        "sigma": external_config["sigma"],
        "loc_bg_threshold": external_config["loc_bg_threshold"],
        "img_size": 512,
        "center_crop": None,
        "seed": 42,
        "data_root": external_config["data_root"],
        "object": "engineers_hammer_vray",
        "pairs_index": external_config["pairs_index"],
    }


def _registry(task_id: int) -> replay_registry.ValidatedReplayRegistry:
    return replay_registry.ValidatedReplayRegistry(
        registry_absolute_path="/repository/registry.json",
        registry_file_sha256="a" * 64,
        content_hash_sha256="b" * 64,
        schema_version="fixture",
        checkpoint_candidates=(_candidate(task_id),),
    )


def _review_authorization() -> dict:
    return {
        "reviewed_candidate_commit": REVIEWED_COMMIT,
        "runtime_source_manifest": {
            "file_sha256": RUNTIME_MANIFEST_SHA256,
            "content_hash_sha256": RUNTIME_MANIFEST_CONTENT_SHA256,
            "sources": {},
        },
        "runtime_source_manifest_file_sha256": RUNTIME_MANIFEST_SHA256,
        "runtime_source_manifest_content_hash_sha256": (
            RUNTIME_MANIFEST_CONTENT_SHA256
        ),
        "checkpoint_hash_preflight_file_sha256": PREFLIGHT_SHA256,
        "checkpoint_hash_preflight": {},
        "checkpoint_execution_authorization_file_sha256": (
            EXECUTION_AUTHORIZATION_SHA256
        ),
        "fable_execution_review_file_sha256": FABLE_REVIEW_SHA256,
    }


def _task20_result_document(
    *,
    collapse: bool = True,
    historical_passed: bool = True,
) -> tuple[dict, dict, bytes]:
    source_bytes = b"reviewed source bytes\n"
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest_sources = {
        role: {
            "absolute_path": f"/repository/{path}",
            "sha256": source_sha256,
        }
        for role, path in authorization._EXPECTED_RUNTIME_SOURCES
    }
    loaded_sources = {
        role: {
            "module": f"fixture.{role}",
            "absolute_path": record["absolute_path"],
            "import_time_sha256": source_sha256,
        }
        for role, record in manifest_sources.items()
    }
    runtime = _runtime_authorization(task_id=20)
    evaluator_result = {
        "collapse_evidence": {
            "structural_negative_control_collapse": collapse,
        }
    }
    evaluator_result["result_content_sha256"] = hashlib.sha256(
        authorization._canonical_json_bytes(evaluator_result)
    ).hexdigest()
    result = {
        "schema_version": "representation_checkpoint_replay_result.v1",
        "artifact_type": "representation_checkpoint_replay_result",
        "task_id": 20,
        "fixture_id": runtime.fixture_id,
        "fixture_role": runtime.role,
        "source_commit": SOURCE_COMMIT,
        "candidate_eligibility": "forbidden",
        "execution_boundary": {
            "checkpoint_loaded": True,
            "training_performed": False,
            "optimizer_constructed": False,
            "weight_update_performed": False,
            "device": "cpu",
            "weights_only": True,
            "selection_use_authorized": False,
        },
        "environment": {},
        "architecture": {},
        "checkpoint": _checkpoint_load_record(runtime),
        "live_inputs": {},
        "inference": _inference_record(),
        "loaded_source_digests": loaded_sources,
        "bundle_content_sha256": "c" * 64,
        "evaluator_result": evaluator_result,
        "historical_comparison": {
            "frozen_record_count": 245,
            "failed_record_count": 0 if historical_passed else 1,
            "all_definition_identical_fields_passed": historical_passed,
        },
        "gate": {
            "expected_structural_negative_control_collapse": True,
            "observed_structural_negative_control_collapse": collapse,
            "classification_passed": collapse,
            "historical_245_record_comparison_passed": historical_passed,
            "passed": collapse and historical_passed,
            "next_task_authorized": collapse and historical_passed,
        },
    }
    result["content_hash_sha256"] = hashlib.sha256(
        authorization._canonical_json_bytes(result)
    ).hexdigest()
    data = authorization._canonical_json_bytes(result) + b"\n"
    manifest = {
        "file_sha256": RUNTIME_MANIFEST_SHA256,
        "content_hash_sha256": RUNTIME_MANIFEST_CONTENT_SHA256,
        "sources": manifest_sources,
    }
    return result, manifest, data


class CheckpointAuthorizationRuntimeTests(TestCase):
    def setUp(self) -> None:
        authorization._REGISTERED_LOADED_CHECKPOINT.set(None)
        authorization._PENDING_EVALUATOR_AUTHORIZATION.set(None)
        authorization._PENDING_PROVENANCE_LOADED_CHECKPOINT.set(None)

    def tearDown(self) -> None:
        authorization._REGISTERED_LOADED_CHECKPOINT.set(None)
        authorization._PENDING_EVALUATOR_AUTHORIZATION.set(None)
        authorization._PENDING_PROVENANCE_LOADED_CHECKPOINT.set(None)

    def test_authorization_import_digest_captures_exact_loaded_bytes(self) -> None:
        self.assertEqual(
            authorization.__representation_import_sha256__,
            hashlib.sha256(
                Path(authorization.__file__).absolute().read_bytes()
            ).hexdigest(),
        )

    def test_synthetic_checkpoint_safe_load_and_legacy_architecture_smoke(
        self,
    ) -> None:
        """Exercise the real loader without opening any registered checkpoint."""

        torch = __import__("torch")
        model_module = __import__(
            "keypoint_net.model",
            fromlist=["PhaseAModel"],
        )
        runtime_receipt = _runtime_authorization(task_id=20)
        external_config = _synthetic_external_config(20)
        architecture = runtime._validate_external_config(
            external_config,
            task_id=20,
            runtime_authorization=runtime_receipt,
        )
        source_model = model_module.PhaseAModel(**architecture)
        synthetic_directory = Path(
            tempfile.mkdtemp(
                prefix="representation-synthetic-checkpoint-",
                dir="/private/tmp",
            )
        )
        checkpoint_path = synthetic_directory / "synthetic_task20.pt"
        torch.save(
            {
                "epoch": 3,
                "model_state_dict": source_model.state_dict(),
                "optimizer_state_dict": {},
                "loss": 0.25,
                "config": _synthetic_embedded_config(
                    external_config=external_config,
                    architecture=architecture,
                ),
            },
            checkpoint_path,
        )
        checkpoint_bytes = checkpoint_path.read_bytes()
        runtime_receipt = replace(
            runtime_receipt,
            checkpoint_absolute_path=str(checkpoint_path),
            checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            checkpoint_size_bytes=len(checkpoint_bytes),
        )

        loaded_model, load_record = (
            runtime._safe_load_checkpoint_same_descriptor(
                runtime_receipt,
                architecture=architecture,
                external_config=external_config,
            )
        )
        self.assertIsNotNone(loaded_model.inverse_operator)
        self.assertIsNone(loaded_model.action_classifier)
        self.assertEqual(loaded_model.extractor.heatmap_res, 64)
        self.assertIsNone(loaded_model.extractor.head_upsample)
        self.assertFalse(loaded_model.training)
        self.assertTrue(
            all(
                parameter.requires_grad is False
                for parameter in loaded_model.parameters()
            )
        )
        before = runtime._state_digest(loaded_model)
        with torch.inference_mode():
            points, logits = loaded_model.extractor(
                torch.zeros((1, 3, 512, 512), dtype=torch.float32)
            )
        after = runtime._state_digest(loaded_model)
        self.assertEqual(tuple(points.shape), (1, 20))
        self.assertEqual(tuple(logits.shape), (1, 10, 64, 64))
        self.assertEqual(before, after)
        self.assertEqual(
            load_record["checkpoint_sha256"],
            runtime_receipt.checkpoint_sha256,
        )
        self.assertTrue(
            load_record["same_open_file_descriptor_hash_and_load"]
        )
        self.assertTrue(load_record["weights_only"])
        self.assertFalse(load_record["unsafe_fallback_used"])

    def test_forged_runtime_token_is_rejected(self) -> None:
        forged = copy.copy(_runtime_authorization())
        object.__setattr__(forged, "_capability", object())
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "live checkpoint runtime capability",
        ):
            authorization.require_checkpoint_runtime_authorization(forged)

    def test_load_inference_evaluator_provenance_receipts_are_one_shot(
        self,
    ) -> None:
        runtime = _runtime_authorization()
        checkpoint_record = _checkpoint_load_record(runtime)
        bundle = _bundle(runtime)
        with mock.patch.object(
            authorization,
            "_regular_file_bytes",
            side_effect=AssertionError("checkpoint was reopened"),
        ):
            authorization._register_loaded_checkpoint_after_same_descriptor_load(
                runtime_authorization=runtime,
                checkpoint_record=checkpoint_record,
            )
            authorization._mint_checkpoint_evaluator_context_after_inference(
                bundle,
                runtime_authorization=runtime,
                inference_record=_inference_record(),
            )
            evaluator_receipt = (
                authorization.validate_checkpoint_evaluator_authorization(
                    bundle
                )
            )
            checkpoint_external_record = bundle["provenance"][
                "external_files"
            ][0]
            provenance_receipt = (
                authorization.consume_checkpoint_provenance_load_receipt(
                    source_commit=SOURCE_COMMIT,
                    checkpoint_record=checkpoint_external_record,
                )
            )
        self.assertTrue(
            evaluator_receipt["checkpoint_evaluation_authorized"]
        )
        self.assertEqual(
            provenance_receipt["file_sha256"],
            runtime.checkpoint_sha256,
        )
        self.assertTrue(
            provenance_receipt["same_open_file_descriptor_hash_and_load"]
        )
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "one-shot load receipt",
        ):
            authorization.consume_checkpoint_provenance_load_receipt(
                source_commit=SOURCE_COMMIT,
                checkpoint_record=checkpoint_external_record,
            )

    def test_evaluator_context_requires_registered_checkpoint_load(self) -> None:
        runtime = _runtime_authorization()
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "registered same-descriptor checkpoint load",
        ):
            authorization._mint_checkpoint_evaluator_context_after_inference(
                _bundle(runtime),
                runtime_authorization=runtime,
                inference_record=_inference_record(),
            )

    def test_changed_bundle_burns_the_evaluator_receipt(self) -> None:
        runtime = _runtime_authorization()
        bundle = _bundle(runtime)
        authorization._register_loaded_checkpoint_after_same_descriptor_load(
            runtime_authorization=runtime,
            checkpoint_record=_checkpoint_load_record(runtime),
        )
        authorization._mint_checkpoint_evaluator_context_after_inference(
            bundle,
            runtime_authorization=runtime,
            inference_record=_inference_record(),
        )
        changed = copy.deepcopy(bundle)
        changed["evaluation"]["points"]["payload_base64"] = "changed"
        with self.assertRaises(
            authorization.CheckpointAuthorizationError
        ):
            authorization.validate_checkpoint_evaluator_authorization(changed)
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "live one-shot runtime context",
        ):
            authorization.validate_checkpoint_evaluator_authorization(bundle)

    def test_bad_inference_state_is_rejected_and_load_receipt_is_burned(
        self,
    ) -> None:
        runtime = _runtime_authorization()
        authorization._register_loaded_checkpoint_after_same_descriptor_load(
            runtime_authorization=runtime,
            checkpoint_record=_checkpoint_load_record(runtime),
        )
        bad_inference = _inference_record()
        bad_inference["state_sha256_after"] = "f" * 64
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "model state after inference",
        ):
            authorization._mint_checkpoint_evaluator_context_after_inference(
                _bundle(runtime),
                runtime_authorization=runtime,
                inference_record=bad_inference,
            )
        with self.assertRaisesRegex(
            authorization.CheckpointAuthorizationError,
            "registered same-descriptor checkpoint load",
        ):
            authorization._mint_checkpoint_evaluator_context_after_inference(
                _bundle(runtime),
                runtime_authorization=runtime,
                inference_record=_inference_record(),
            )

    def test_committed_task20_gate_accepts_only_full_pass(self) -> None:
        _result, manifest, data = _task20_result_document()
        source_bytes = b"reviewed source bytes\n"

        def fake_git(
            _repo_root: Path,
            arguments: list[str],
            *,
            name: str,
        ) -> bytes:
            del name
            if arguments[0] == "merge-base":
                return b""
            self.assertEqual(arguments[0], "show")
            return source_bytes

        with mock.patch.object(
            authorization,
            "_committed_working_file",
            return_value=(Path("/repository/task20.json"), data),
        ), mock.patch.object(
            authorization,
            "_git",
            side_effect=fake_git,
        ):
            receipt = authorization._validate_committed_task20_result(
                repo_root=Path("/repository"),
                source_commit=SOURCE_COMMIT,
                runtime_source_manifest=manifest,
            )
        self.assertEqual(
            receipt["file_sha256"],
            hashlib.sha256(data).hexdigest(),
        )

        _failed, failed_manifest, failed_data = _task20_result_document(
            collapse=False
        )
        with mock.patch.object(
            authorization,
            "_committed_working_file",
            return_value=(Path("/repository/task20.json"), failed_data),
        ), mock.patch.object(
            authorization,
            "_git",
            side_effect=fake_git,
        ):
            with self.assertRaisesRegex(
                authorization.CheckpointAuthorizationError,
                "structural collapse flag",
            ):
                authorization._validate_committed_task20_result(
                    repo_root=Path("/repository"),
                    source_commit=SOURCE_COMMIT,
                    runtime_source_manifest=failed_manifest,
                )

    def test_public_task55_authorization_cannot_bypass_task20_file(self) -> None:
        registry = _registry(55)
        provenance = SimpleNamespace(source_commit=SOURCE_COMMIT)
        common_patches = (
            mock.patch.object(
                replay_registry,
                "_require_preflight_authorization",
            ),
            mock.patch.object(
                authorization,
                "_validate_execution_review_roles",
                return_value=_review_authorization(),
            ),
            mock.patch.object(
                authorization,
                "_validate_checkpoint_preflight",
            ),
            mock.patch.object(
                authorization,
                "_validate_actual_runtime_environment",
            ),
        )
        for patcher in common_patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        with mock.patch.object(
            authorization,
            "_validate_committed_task20_result",
            side_effect=authorization.CheckpointAuthorizationError(
                "Task 20 checkpoint replay result cannot be read"
            ),
        ):
            with self.assertRaisesRegex(
                authorization.CheckpointAuthorizationError,
                "Task 20 checkpoint replay result cannot be read",
            ):
                authorization.authorize_checkpoint_runtime(
                    registry=registry,
                    provenance=provenance,
                    checkpoint_hash_preflight={},
                    task_id=55,
                    runtime_source_commit=SOURCE_COMMIT,
                    evaluation_provenance={},
                )

        task20_gate = {
            "file_sha256": "7" * 64,
            "content_hash_sha256": "8" * 64,
            "source_commit": REVIEWED_COMMIT,
        }
        with mock.patch.object(
            authorization,
            "_validate_committed_task20_result",
            return_value=task20_gate,
        ):
            runtime = authorization.authorize_checkpoint_runtime(
                registry=registry,
                provenance=provenance,
                checkpoint_hash_preflight={},
                task_id=55,
                runtime_source_commit=SOURCE_COMMIT,
                evaluation_provenance={},
            )
        self.assertEqual(
            runtime.task20_gate_result_file_sha256,
            task20_gate["file_sha256"],
        )
        authorization.require_checkpoint_runtime_authorization(
            runtime,
            task_id=55,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
