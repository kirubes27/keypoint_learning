"""Decision-critical CPU tests for the held-out roll finalizer."""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from keypoint_net import eval_representation as evaluator
from keypoint_net.dataset import IndexPairManifest
from keypoint_net import representation_evaluation_provenance as provenance
from keypoint_net import representation_fixed_final_authorization as authorization
from keypoint_net import representation_fixed_final_runtime as runtime
from keypoint_net import representation_split_adapter as adapter


def _binding(root: Path) -> authorization.FixedFinalRunBinding:
    return authorization.FixedFinalRunBinding(
        run_id="banana-roll-task55-seed42",
        recipe_id="task55_clean",
        object_id="b03_banana_01_high",
        object_role="confirmation",
        seed=42,
        frozen_epochs=40,
        source_commit="a" * 40,
        source_branch="agent/heldout-roll-finalizer-20260809",
        run_directory=str(root),
        manifest_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/manifests/run.json"
        ),
        manifest_absolute_path=str(root / "run.json"),
        manifest_file_sha256="1" * 64,
        manifest_content_sha256="2" * 64,
        train_pair_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
            "roll__world_z__forward__train.json"
        ),
        train_pair_file_sha256="7" * 64,
        train_pair_content_sha256="8" * 64,
        test_pair_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
            "roll__world_z__forward__test.json"
        ),
        test_pair_file_sha256="9" * 64,
        test_pair_content_sha256="a" * 64,
        dataset_binding_sha256="b" * 64,
        corpus_inventory_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/inventories/"
            "CORPUS_INVENTORY__roll.json"
        ),
        geometry_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
            "b03_banana_01_high__roll__world_z__v1.json"
        ),
        implementation_lock_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/implementation_locks/"
            "implementation.json"
        ),
        decision_spec_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/decisions/decision.json"
        ),
        pro_review_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/reviews/pro_review.json"
        ),
        fable_review_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/reviews/fable_review.json"
        ),
        user_approval_repo_relative_path=(
            "docs/decisions/heldout_roll_fixed_final/v1/approvals/approval.json"
        ),
        expected_training_arguments={
            "num_keypoints": 10,
            "heatmap_res": 64,
            "temperature": 1.0,
        },
        _capability=authorization._CAPABILITY_TOKEN,
    )


def _capability(root: Path) -> authorization.FixedFinalCheckpointCapability:
    return authorization.FixedFinalCheckpointCapability(
        binding=_binding(root),
        checkpoint_absolute_path=str(root / "final_model.pt"),
        checkpoint_sha256="3" * 64,
        checkpoint_size_bytes=10,
        config_absolute_path=str(root / "config.json"),
        config_sha256="4" * 64,
        config_size_bytes=11,
        history_absolute_path=str(root / "history.json"),
        history_sha256="5" * 64,
        history_size_bytes=12,
        receipt_absolute_path=str(root / authorization.TRAINING_RECEIPT_NAME),
        receipt_file_sha256="6" * 64,
        receipt_size_bytes=13,
        _capability=authorization._CAPABILITY_TOKEN,
    )


class FixedFinalContractTests(unittest.TestCase):
    def test_authority_records_must_approve_exact_run_and_implementation(self) -> None:
        manifest_hash = "1" * 64
        implementation_hash = "2" * 64
        decision_hash = "3" * 64
        pro_hash = "4" * 64
        fable_hash = "5" * 64
        document = {
            "run_id": "banana-roll-task55-seed42",
            "object": {"id": "b03_banana_01_high", "role": "confirmation"},
            "recipe": {"id": "task55_clean"},
            "training_arguments": {"frozen_epochs": 40},
            "implementation_lock": {
                "repo_relative_path": (
                    "docs/decisions/heldout_roll_fixed_final/v1/"
                    "implementation_locks/lock.json"
                )
            },
            "decision_spec": {"repo_relative_path": "unused"},
            "pro_review": {"repo_relative_path": "unused"},
            "fable_review": {"repo_relative_path": "unused"},
            "user_approval": {"repo_relative_path": "unused"},
        }

        def sealed(value):
            result = copy.deepcopy(value)
            result["content_hash_sha256"] = authorization.canonical_sha256(result)
            return result

        common = {
            "run_id": document["run_id"],
            "manifest_content_sha256": manifest_hash,
        }
        decision = sealed({
            "schema_version": "heldout_roll_fixed_final_decision_spec.v1",
            "artifact_type": "heldout_roll_fixed_final_decision_spec",
            **common,
            "object": document["object"],
            "recipe_id": "task55_clean",
            "frozen_epochs": 40,
        })

        def review(reviewer, raw_name):
            return sealed({
                "schema_version": "heldout_roll_fixed_final_review.v1",
                "artifact_type": "heldout_roll_fixed_final_review",
                "reviewer": reviewer,
                "verdict": "PASS_WITH_REQUIRED_FIXES",
                **common,
                "decision_spec_file_sha256": decision_hash,
                "implementation_lock_file_sha256": implementation_hash,
                "raw_report": {
                    "repo_relative_path": (
                        "docs/decisions/heldout_roll_fixed_final/v1/reports/"
                        f"{raw_name}.txt"
                    ),
                    "file_sha256": "6" * 64,
                },
            })

        pro = review("chatgpt_pro", "pro")
        fable = review("fable", "fable")
        approval = sealed({
            "schema_version": "heldout_roll_fixed_final_user_approval.v1",
            "artifact_type": "heldout_roll_fixed_final_user_approval",
            "affirmative_authorization": True,
            "authorization": "authorize_exact_fixed_final_run",
            "reviewer_findings_are_advisory": True,
            **common,
            "object": document["object"],
            "recipe_id": "task55_clean",
            "frozen_epochs": 40,
            "decision_spec_file_sha256": decision_hash,
            "pro_review_file_sha256": pro_hash,
            "fable_review_file_sha256": fable_hash,
            "implementation_lock_file_sha256": implementation_hash,
        })
        paths = {
            "decision specification": (
                "docs/decisions/heldout_roll_fixed_final/v1/decisions/d.json",
                decision,
                decision_hash,
            ),
            "pro review": (
                "docs/decisions/heldout_roll_fixed_final/v1/reviews/pro.json",
                pro,
                pro_hash,
            ),
            "fable review": (
                "docs/decisions/heldout_roll_fixed_final/v1/reviews/fable.json",
                fable,
                fable_hash,
            ),
            "user approval": (
                "docs/decisions/heldout_roll_fixed_final/v1/approvals/user.json",
                approval,
                "7" * 64,
            ),
        }

        def load_record(_root, _commit, _record, *, name, prefix):
            del prefix
            return paths[name]

        def load_raw_report(_root, _commit, _record, *, name, prefix):
            del prefix
            return ("pro.txt" if name.startswith("pro") else "fable.txt", "6" * 64)

        with mock.patch.object(
            authorization, "_validate_json_path_reference", side_effect=load_record
        ), mock.patch.object(
            authorization, "_validate_file_record", side_effect=load_raw_report
        ):
            valid = authorization._validate_authority_records(
                Path("/repo"),
                "a" * 40,
                document,
                manifest_content_sha256=manifest_hash,
                implementation_lock_sha256=implementation_hash,
            )
            self.assertEqual(valid["decision_relative"], paths["decision specification"][0])

            for label, mutate in (
                ("affirmative", lambda item: item.__setitem__("affirmative_authorization", False)),
                ("manifest", lambda item: item.__setitem__("manifest_content_sha256", "8" * 64)),
                ("implementation", lambda item: item.__setitem__("implementation_lock_file_sha256", "9" * 64)),
            ):
                changed = copy.deepcopy(approval)
                changed.pop("content_hash_sha256")
                mutate(changed)
                changed = sealed(changed)
                paths["user approval"] = (paths["user approval"][0], changed, "7" * 64)
                with self.subTest(label=label), self.assertRaises(
                    authorization.FixedFinalAuthorizationError
                ):
                    authorization._validate_authority_records(
                        Path("/repo"),
                        "a" * 40,
                        document,
                        manifest_content_sha256=manifest_hash,
                        implementation_lock_sha256=implementation_hash,
                    )
            paths["user approval"] = (
                paths["pro review"][0], approval, "7" * 64
            )
            with self.assertRaisesRegex(
                authorization.FixedFinalAuthorizationError, "paths must be distinct"
            ):
                authorization._validate_authority_records(
                    Path("/repo"),
                    "a" * 40,
                    document,
                    manifest_content_sha256=manifest_hash,
                    implementation_lock_sha256=implementation_hash,
                )

    def test_heldout_roll_path_is_not_mislabeled_as_a_cycle(self) -> None:
        path_rows = [
            {"source_frame": 0, "target_frame": 3},
            {"source_frame": 3, "target_frame": 6},
        ]
        self.assertFalse(
            evaluator._evaluation_frame_sequence_is_cyclic(
                path_rows,
                [0, 3, 6],
                transform_cyclic=True,
                protocol="generic",
            )
        )
        cycle_rows = [
            {"source_frame": 0, "target_frame": 3},
            {"source_frame": 3, "target_frame": 6},
            {"source_frame": 6, "target_frame": 0},
        ]
        self.assertTrue(
            evaluator._evaluation_frame_sequence_is_cyclic(
                cycle_rows,
                [0, 3, 6],
                transform_cyclic=True,
                protocol="generic",
            )
        )
        self.assertFalse(
            evaluator._evaluation_frame_sequence_is_cyclic(
                cycle_rows,
                [0, 6, 3],
                transform_cyclic=True,
                protocol="generic",
            )
        )
        full_roll_rows = [
            {"source_frame": frame, "target_frame": (frame + 3) % 180}
            for frame in range(180)
        ]
        self.assertTrue(
            evaluator._evaluation_frame_sequence_is_cyclic(
                full_roll_rows,
                list(range(180)),
                transform_cyclic=True,
                protocol="full_primary_roll",
            )
        )

    def test_provenance_profile_is_exact_and_has_both_split_roles(self) -> None:
        self.assertFalse(hasattr(authorization, "write_evidence_receipt"))
        committed, external = provenance.required_role_profile(
            "checkpoint", fit_from_pairs=False, checkpoint_profile="fixed_final"
        )
        self.assertEqual(committed, provenance.FIXED_FINAL_CHECKPOINT_COMMITTED_ROLES)
        self.assertIn("training_pair_artifact", committed)
        self.assertIn("evaluation_pair_artifact", committed)
        self.assertIn("geometry_manifest", committed)
        self.assertEqual(
            external,
            frozenset(
                {
                    "checkpoint", "checkpoint_config", "checkpoint_metadata",
                    "completed_run_receipt",
                }
            ),
        )
        with self.assertRaisesRegex(Exception, "forbids fit_from_pairs"):
            provenance.required_role_profile(
                "checkpoint", fit_from_pairs=True,
                checkpoint_profile="fixed_final",
            )

    def test_strict_test_manifest_adapter_keeps_role_partition_and_geometry(self) -> None:
        metadata = {
            "transform": {
                "family": "roll", "physical_axis": "world_z",
                "direction": "forward", "signed_generator": 6.0,
                "generator_units": "degrees", "stride": 3,
                "stride_units": "frames", "cyclic": True,
                "expected_2d_family": "planar_rotation_about_projected_center",
            },
            "dataset_binding_sha256": "7" * 64,
            "dataset_semantic_lock_sha256": "8" * 64,
            "object_roles": {"b03_banana_01_high": "confirmation"},
        }
        rows = (
            {
                "pair_id": "banana-test-0-3", "model_name": "b03_banana_01_high",
                "object_role": "confirmation", "split": "test",
                "src_frame_index": 0, "dst_frame_index": 3,
                "src_image_relpath": "train/banana/frames/a/img_0000.png",
                "dst_image_relpath": "train/banana/frames/a/img_0003.png",
                "src_mask_relpath": "train/banana/masks/a/mask_0000.png",
                "dst_mask_relpath": "train/banana/masks/a/mask_0003.png",
                "src_state": {"theta_deg": 0.0},
                "dst_state": {"theta_deg": 6.0},
                "direction": "forward", "stride": 3,
            },
        )
        manifest = SimpleNamespace(
            strict_metadata=True, split="test", metadata=metadata, pairs=rows,
            dataset_binding_sha256="7" * 64, index_path=Path("test.json"),
            index_sha256="9" * 64, content_hash_sha256="a" * 64,
        )
        geometry = {
            "content_hash_sha256": "b" * 64,
            "parameters": {"projected_centre_xy": [0.125, -0.25]},
        }
        registry_record = {
            "content_hash_sha256": "b" * 64,
        }
        with mock.patch.object(
            adapter, "_authorize_registered_dataset_geometry",
            return_value=registry_record,
        ), mock.patch.object(adapter, "_load_geometry", return_value=geometry):
            result = adapter.build_index_manifest_adapter_rows(
                manifest=manifest,
                object_id="b03_banana_01_high",
                geometry_binding_path=Path("/registered/banana-roll.json"),
            )
        self.assertEqual(result["stratum"]["object_role"], "confirmation")
        self.assertEqual(result["evaluation"]["partition"], "test")
        self.assertEqual(result["evaluation"]["rows"][0]["partition"], "test")
        self.assertEqual(
            result["evaluator_transform"]["projected_centre_xy"],
            [0.125, -0.25],
        )

    def test_test_content_open_phase_is_one_shot(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="fixed_final_open_test_")).resolve()
        capability = _capability(root)
        geometry_path = root / capability.binding.geometry_repo_relative_path
        geometry_path.parent.mkdir(parents=True)
        geometry_path.write_bytes(b"{}\n")
        test_pair_path = root / capability.binding.test_pair_repo_relative_path
        test_pair_path.parent.mkdir(parents=True, exist_ok=True)
        test_pair_path.write_bytes(b"{}\n")
        image_buffer = io.BytesIO()
        Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(
            image_buffer, format="PNG"
        )
        mask_buffer = io.BytesIO()
        Image.fromarray(np.ones((512, 512), dtype=np.uint8) * 255).save(
            mask_buffer, format="PNG"
        )
        frame = {
            "frame_id": 0,
            "image_relpath": "frame.png",
            "mask_relpath": "mask.png",
            "physical_state": {"theta_deg": 0.0},
        }
        adapted = {
            "stratum": {"transform_family": "roll", "evaluation_partition": "test"},
            "evaluation": {
                "frames": [frame],
                "rows": [{
                    "pair_id": "p", "source_frame": 0, "target_frame": 3,
                    "direction": "forward", "stride": 3, "partition": "test",
                }],
            },
            "pair_index_binding": {
                "absolute_path": str(test_pair_path),
                "file_sha256": capability.binding.test_pair_file_sha256,
                "content_hash_sha256": capability.binding.test_pair_content_sha256,
                "dataset_binding_sha256": capability.binding.dataset_binding_sha256,
            },
            "evaluator_transform": {"family": "roll"},
        }
        opened = {
            "frame.png": image_buffer.getvalue(),
            "mask.png": mask_buffer.getvalue(),
        }

        def read_bound(_root, relative, _inventory, *, name):
            raw = opened[relative]
            return raw, {
                "relative_path": relative, "absolute_path": f"/dataset/{relative}",
                "sha256": "c" * 64, "size_bytes": len(raw),
            }

        token = runtime._LOADED.set(
            {"capability": capability, "opening_ledger": None}
        )
        manifest = IndexPairManifest(
            data_root=root,
            index_path=test_pair_path,
            index_sha256=capability.binding.test_pair_file_sha256,
            content_hash_sha256=capability.binding.test_pair_content_sha256,
            metadata={"dataset_binding_sha256": capability.binding.dataset_binding_sha256},
            pairs=(),
            source_endpoint_ids=frozenset(),
            target_endpoint_ids=frozenset(),
            endpoint_ids=frozenset(),
            strict_metadata=True,
        )
        try:
            with mock.patch.object(
                runtime.split_adapter, "build_index_manifest_adapter_rows",
                return_value=adapted,
            ), mock.patch.object(runtime, "_load_inventory", return_value={}), \
                    mock.patch.object(runtime, "_read_bound_dataset_file",
                                      side_effect=read_bound):
                output = runtime._open_test_frames_once(
                    root, root, capability, manifest
                )
                self.assertEqual(output[-1]["test_content_open_phase_count"], 1)
                with self.assertRaisesRegex(runtime.FixedFinalRuntimeError,
                                            "only once"):
                    runtime._open_test_frames_once(
                        root, root, capability, manifest
                    )
        finally:
            runtime._LOADED.reset(token)

        restart_token = runtime._LOADED.set(
            {"capability": capability, "opening_ledger": None}
        )
        try:
            with mock.patch.object(
                runtime.split_adapter,
                "build_index_manifest_adapter_rows",
                return_value=adapted,
            ), mock.patch.object(runtime, "_load_inventory", return_value={}):
                with self.assertRaisesRegex(
                    runtime.FixedFinalRuntimeError, "durable test-open marker"
                ):
                    runtime._open_test_frames_once(
                        root, root, capability, manifest
                    )
        finally:
            runtime._LOADED.reset(restart_token)

    def test_substituted_test_manifest_fails_before_content_open(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="fixed_final_substitute_test_")).resolve()
        capability = _capability(root)
        test_pair_path = root / capability.binding.test_pair_repo_relative_path
        test_pair_path.parent.mkdir(parents=True)
        test_pair_path.write_bytes(b"{}\n")
        substituted = IndexPairManifest(
            data_root=root,
            index_path=test_pair_path,
            index_sha256="f" * 64,
            content_hash_sha256=capability.binding.test_pair_content_sha256,
            metadata={
                "dataset_binding_sha256": capability.binding.dataset_binding_sha256
            },
            pairs=(),
            source_endpoint_ids=frozenset(),
            target_endpoint_ids=frozenset(),
            endpoint_ids=frozenset(),
            strict_metadata=True,
        )
        adapter_mock = mock.Mock()
        read_mock = mock.Mock()
        token = runtime._LOADED.set(
            {"capability": capability, "opening_ledger": None}
        )
        try:
            with mock.patch.object(
                runtime.split_adapter,
                "build_index_manifest_adapter_rows",
                adapter_mock,
            ), mock.patch.object(runtime, "_read_bound_dataset_file", read_mock):
                with self.assertRaisesRegex(
                    runtime.FixedFinalRuntimeError, "approved test pair artifact"
                ):
                    runtime._open_test_frames_once(
                        root, root, capability, substituted
                    )
        finally:
            runtime._LOADED.reset(token)
        adapter_mock.assert_not_called()
        read_mock.assert_not_called()
        self.assertFalse((root / runtime.TEST_OPEN_MARKER_NAME).exists())

    def test_evaluator_routes_fixed_final_authority_and_binds_run_id(self) -> None:
        checkpoint = {
            "role": "checkpoint", "absolute_path": "/run/final_model.pt",
            "file_sha256": "d" * 64, "size_bytes": 10,
        }
        bundle = {
            "case_id": "banana-roll-task55-seed42",
            "checkpoint_authority": "fixed_final",
            "provenance": {
                "source_commit": "e" * 40,
                "external_files": [checkpoint],
            },
        }
        receipt = {
            "checkpoint_evaluation_authorized": True,
            "source_commit": "e" * 40,
            "run_id": bundle["case_id"],
            "checkpoint_sha256": "d" * 64,
            "completed_run_receipt_sha256": "f" * 64,
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": True,
        }
        module = SimpleNamespace(
            validate_fixed_final_checkpoint_evaluator_authorization=(
                lambda _bundle: receipt
            ),
            consume_fixed_final_checkpoint_provenance_load_receipt=(
                lambda **_kwargs: {
                    **checkpoint,
                    "same_open_file_descriptor_hash_and_load": True,
                    "weights_only": True,
                }
            ),
        )
        with mock.patch.object(evaluator.importlib, "import_module", return_value=module):
            validated, loaded = evaluator._require_checkpoint_evaluator_authorization(
                bundle
            )
        self.assertEqual(validated["run_id"], bundle["case_id"])
        self.assertTrue(loaded["same_open_file_descriptor_hash_and_load"])

    def test_runtime_load_receipt_satisfies_provenance_cross_binding(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="fixed_final_receipt_test_")).resolve()
        capability = _capability(root)
        checkpoint_record = {
            "role": "checkpoint",
            "absolute_path": capability.checkpoint_absolute_path,
            "file_sha256": capability.checkpoint_sha256,
            "size_bytes": capability.checkpoint_size_bytes,
        }
        token = runtime._PENDING_PROVENANCE.set({"capability": capability})
        try:
            receipt = runtime.consume_fixed_final_checkpoint_provenance_load_receipt(
                source_commit=capability.binding.source_commit,
                checkpoint_record=checkpoint_record,
            )
        finally:
            runtime._PENDING_PROVENANCE.reset(token)
        normalized = provenance._validate_checkpoint_provenance_load_receipt(
            receipt,
            source_commit=capability.binding.source_commit,
            checkpoint_shape=(
                "checkpoint",
                capability.checkpoint_absolute_path,
                capability.checkpoint_sha256,
                capability.checkpoint_size_bytes,
            ),
        )
        self.assertEqual(normalized, checkpoint_record)


if __name__ == "__main__":
    unittest.main()
