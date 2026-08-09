"""Decision-critical CPU tests for the held-out roll finalizer."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from keypoint_net import eval_representation as evaluator
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
            "docs/decisions/2026-08-09/heldout_roll_fixed_final/manifests/run.json"
        ),
        manifest_absolute_path=str(root / "run.json"),
        manifest_file_sha256="1" * 64,
        manifest_content_sha256="2" * 64,
        train_pair_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
            "roll__world_z__forward__train.json"
        ),
        test_pair_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/pairs/"
            "roll__world_z__forward__test.json"
        ),
        corpus_inventory_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_splits/inventories/"
            "CORPUS_INVENTORY__roll.json"
        ),
        geometry_repo_relative_path=(
            "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
            "b03_banana_01_high__roll__world_z__v1.json"
        ),
        decision_spec_repo_relative_path=(
            "docs/decisions/2026-08-09/heldout_roll_fixed_final/decision.json"
        ),
        pro_review_repo_relative_path=(
            "docs/decisions/2026-08-09/heldout_roll_fixed_final/pro_review.json"
        ),
        fable_review_repo_relative_path=(
            "docs/decisions/2026-08-09/heldout_roll_fixed_final/fable_review.json"
        ),
        user_approval_repo_relative_path=(
            "docs/decisions/2026-08-09/heldout_roll_fixed_final/approval.json"
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
    def test_heldout_roll_path_is_not_mislabeled_as_a_cycle(self) -> None:
        path_rows = [
            {"source_frame": 0, "target_frame": 3},
            {"source_frame": 3, "target_frame": 6},
        ]
        self.assertFalse(
            evaluator._pair_graph_is_one_complete_cycle(path_rows, [0, 3, 6])
        )
        cycle_rows = [
            {"source_frame": 0, "target_frame": 3},
            {"source_frame": 3, "target_frame": 6},
            {"source_frame": 6, "target_frame": 0},
        ]
        self.assertTrue(
            evaluator._pair_graph_is_one_complete_cycle(cycle_rows, [0, 3, 6])
        )

    def test_provenance_profile_is_exact_and_has_both_split_roles(self) -> None:
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
        try:
            with mock.patch.object(
                runtime.split_adapter, "build_index_manifest_adapter_rows",
                return_value=adapted,
            ), mock.patch.object(runtime, "_load_inventory", return_value={}), \
                    mock.patch.object(runtime, "_read_bound_dataset_file",
                                      side_effect=read_bound):
                output = runtime._open_test_frames_once(
                    root, root, capability, SimpleNamespace()
                )
                self.assertEqual(output[-1]["test_content_open_phase_count"], 1)
                with self.assertRaisesRegex(runtime.FixedFinalRuntimeError,
                                            "only once"):
                    runtime._open_test_frames_once(
                        root, root, capability, SimpleNamespace()
                    )
        finally:
            runtime._LOADED.reset(token)

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
