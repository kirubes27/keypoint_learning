"""CPU-only contract tests for the frozen split-to-evaluator adapter."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from keypoint_net import representation_split_adapter as adapter
from keypoint_net.representation_splits import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPOSITORY_ROOT / adapter.FROZEN_BUNDLE_RELPATH
SPEC_PATH = (
    REPOSITORY_ROOT
    / "docs/decisions/2026-07-26/REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md"
)
HAMMER_ROLL_GEOMETRY = (
    REPOSITORY_ROOT
    / "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
    "engineers_hammer_vray__roll__world_z__v1.json"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(relative_path: str) -> dict:
    return json.loads((BUNDLE_ROOT / relative_path).read_text(encoding="utf-8"))


def _write_geometry(
    root: Path,
    *,
    filename: str,
    family: str,
    object_id: str,
    inventory_hash: str,
    semantic_hash: str,
    parameters: dict,
) -> Path:
    document = {
        "schema_version": adapter.GEOMETRY_SCHEMA_VERSION,
        "artifact_type": "representation_geometry_binding",
        "object_id": object_id,
        "transform_family": family,
        "inventory_content_hash_sha256": inventory_hash,
        "dataset_semantic_lock_sha256": semantic_hash,
        "estimator_geometry": {
            "coordinate_system": "normalized_minus1_plus1_endpoint_aligned_xy",
            "image_y_direction": "down",
            "crop": {"mode": "none"},
            "resize": [512, 512],
            "align_corners": None,
        },
        "parameters": parameters,
        "evidence_files": [
            {
                "role": "contract_test_geometry_evidence",
                "absolute_path": str(SPEC_PATH.resolve()),
                "sha256": _sha256_file(SPEC_PATH),
            }
        ],
    }
    document["content_hash_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    path = root / filename
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    return path


class SplitAdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Deliberately not auto-cleaned: project deletion guardrails prohibit
        # silently deleting even test-created temporary artifacts.
        cls.temp_root = Path(tempfile.mkdtemp(prefix="representation_adapter_test_"))
        scale_pair = _read_json("pairs/scale__uniform__forward__test.json")
        scale_inventory = _read_json(
            "inventories/CORPUS_INVENTORY__scale.json"
        )
        cls.scale_geometry = _write_geometry(
            cls.temp_root,
            filename="scale_geometry.json",
            family="scale",
            object_id="b03_banana_01_high",
            inventory_hash=scale_inventory["content_hash_sha256"],
            semantic_hash=scale_pair["dataset_semantic_lock_sha256"],
            parameters={"projected_centre_xy": [0.125, -0.25]},
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Intentionally leave cls.temp_root in place; do not delete or move it.
        pass

    def test_public_dataset_adapter_is_blocked_without_registered_geometry(self) -> None:
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "no unique frozen registry entry",
        ):
            adapter.build_split_adapter_rows(
                split_bundle_root=BUNDLE_ROOT,
                evaluation_pair_relpath="pairs/scale__uniform__forward__test.json",
                fit_pair_relpath="pairs/scale__uniform__forward__train.json",
                object_id="b03_banana_01_high",
                fit_from_pairs=True,
                geometry_binding_path=self.scale_geometry,
            )

    def test_registered_hammer_roll_geometry_builds_exact_validation_stratum(
        self,
    ) -> None:
        result = adapter.build_split_adapter_rows(
            split_bundle_root=BUNDLE_ROOT,
            evaluation_pair_relpath=(
                "pairs/roll__world_z__forward__validation.json"
            ),
            object_id="engineers_hammer_vray",
            fit_from_pairs=False,
            geometry_binding_path=HAMMER_ROLL_GEOMETRY,
        )
        self.assertEqual(result["stratum"]["signed_generator"], 6.0)
        self.assertEqual(
            result["evaluator_transform"]["projected_centre_xy"],
            [0.0, 0.0],
        )
        self.assertEqual(
            result["geometry_binding"]["parameters"],
            {"projected_centre_xy": [0.0, 0.0]},
        )

    def test_scale_uses_exp_of_signed_log_generator(self) -> None:
        generator = math.log(1.07)
        evaluator_transform, derivations = adapter._derive_evaluator_transform(
            transform={
                "family": "scale",
                "signed_generator": generator,
            },
            family="scale",
            geometry={
                "content_hash_sha256": "a" * 64,
                "parameters": {"projected_centre_xy": [0.125, -0.25]},
            },
            evaluation_frames=[],
            evaluation_rows=[],
        )
        self.assertEqual(
            derivations["image_ratio"],
            math.exp(generator),
        )
        self.assertNotEqual(derivations["image_ratio"], generator)
        self.assertEqual(
            evaluator_transform["projected_centre_xy"],
            [0.125, -0.25],
        )

    def test_rendered_world_y_uses_frozen_negative_image_sign(self) -> None:
        evaluator_transform, derivations = adapter._derive_evaluator_transform(
            transform={
                "family": "translation",
                "physical_axis": "world_y",
                "signed_generator": 0.4,
            },
            family="translation",
            geometry={
                "content_hash_sha256": "b" * 64,
                "parameters": {},
            },
            evaluation_frames=[
                {
                    "frame_id": 10,
                    "physical_state": {
                        "calibrated_image_offset_xy": [0.0, 0.0]
                    },
                },
                {
                    "frame_id": 11,
                    "physical_state": {
                        "calibrated_image_offset_xy": [0.0, -0.08]
                    },
                },
            ],
            evaluation_rows=[{"source_frame": 10, "target_frame": 11}],
        )
        self.assertEqual(evaluator_transform["expected_image_component"], 1)
        self.assertEqual(evaluator_transform["expected_image_sign"], -1)
        self.assertEqual(
            evaluator_transform["calibrated_image_delta_xy"],
            [0.0, -0.08],
        )
        self.assertEqual(derivations["expected_image_sign"], -1)

    def test_rendered_translation_wrong_camera_sign_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "violates the frozen camera sign",
        ):
            adapter._derive_evaluator_transform(
                transform={
                    "family": "translation",
                    "physical_axis": "world_y",
                    "signed_generator": 0.4,
                },
                family="translation",
                geometry={
                    "content_hash_sha256": "b" * 64,
                    "parameters": {},
                },
                evaluation_frames=[
                    {
                        "frame_id": 10,
                        "physical_state": {
                            "calibrated_image_offset_xy": [0.0, 0.0]
                        },
                    },
                    {
                        "frame_id": 11,
                        "physical_state": {
                            "calibrated_image_offset_xy": [0.0, 0.08]
                        },
                    },
                ],
                evaluation_rows=[{"source_frame": 10, "target_frame": 11}],
            )

    def test_rendered_translation_swapped_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "orthogonal image leakage",
        ):
            adapter._derive_evaluator_transform(
                transform={
                    "family": "translation",
                    "physical_axis": "world_y",
                    "signed_generator": 0.4,
                },
                family="translation",
                geometry={
                    "content_hash_sha256": "b" * 64,
                    "parameters": {},
                },
                evaluation_frames=[
                    {
                        "frame_id": 10,
                        "physical_state": {
                            "calibrated_image_offset_xy": [0.0, 0.0]
                        },
                    },
                    {
                        "frame_id": 11,
                        "physical_state": {
                            "calibrated_image_offset_xy": [-0.08, 0.0]
                        },
                    },
                ],
                evaluation_rows=[{"source_frame": 10, "target_frame": 11}],
            )

    def test_frozen_bundle_binds_exact_manifest_pair_and_inventory_hashes(self) -> None:
        bundle, manifest = adapter._load_frozen_bundle(BUNDLE_ROOT)
        self.assertEqual(
            hashlib.sha256(bundle["SPLIT_MANIFEST.json"]).hexdigest(),
            adapter.FROZEN_MANIFEST_FILE_SHA256,
        )
        pair_path = "pairs/scale__uniform__forward__test.json"
        pair = adapter._load_pair(bundle, manifest, pair_path)
        inventory_path = "inventories/CORPUS_INVENTORY__scale.json"
        inventory = adapter._decode_json(
            bundle[inventory_path],
            name=inventory_path,
        )
        self.assertEqual(
            pair["dataset_binding_sha256"],
            inventory["content_hash_sha256"],
        )

    def test_in_memory_one_byte_mutation_fails_git_blob_binding(self) -> None:
        current = {
            path.relative_to(BUNDLE_ROOT).as_posix(): path.read_bytes()
            for path in BUNDLE_ROOT.rglob("*")
            if path.is_file()
        }
        mutated = dict(current)
        raw = bytearray(mutated["SPLIT_VERIFIER_REPORT.json"])
        raw[-2] = raw[-2] ^ 1
        mutated["SPLIT_VERIFIER_REPORT.json"] = bytes(raw)
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "working bytes differ from frozen artifact commit",
        ):
            adapter._verify_frozen_git_blobs(mutated)

    def test_heldout_fitting_requires_explicit_train_artifact(self) -> None:
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "requires a train artifact",
        ):
            adapter.build_split_adapter_rows(
                split_bundle_root=BUNDLE_ROOT,
                evaluation_pair_relpath="pairs/scale__uniform__forward__test.json",
                object_id="b03_banana_01_high",
                fit_from_pairs=True,
                geometry_binding_path=self.scale_geometry,
            )

    def test_roll_or_scale_geometry_without_projected_centre_fails(self) -> None:
        pair = _read_json("pairs/scale__uniform__forward__test.json")
        inventory = _read_json("inventories/CORPUS_INVENTORY__scale.json")
        path = _write_geometry(
            self.temp_root,
            filename="scale_missing_centre.json",
            family="scale",
            object_id="b03_banana_01_high",
            inventory_hash=inventory["content_hash_sha256"],
            semantic_hash=pair["dataset_semantic_lock_sha256"],
            parameters={},
        )
        with self.assertRaisesRegex(adapter.SplitAdapterError, "projected_centre_xy"):
            adapter._load_geometry(
                path,
                family="scale",
                object_id="b03_banana_01_high",
                inventory_content_hash=inventory["content_hash_sha256"],
                dataset_semantic_lock_hash=pair[
                    "dataset_semantic_lock_sha256"
                ],
                required_frame_ids={53, 57},
            )

    def test_translation_requires_calibrated_offset_for_every_frame(self) -> None:
        pair = _read_json("pairs/translation__world_x__forward__test.json")
        inventory = _read_json(
            "inventories/CORPUS_INVENTORY__translation.json"
        )
        path = _write_geometry(
            self.temp_root,
            filename="translation_incomplete.json",
            family="translation",
            object_id="b03_banana_01_high",
            inventory_hash=inventory["content_hash_sha256"],
            semantic_hash=pair["dataset_semantic_lock_sha256"],
            parameters={
                "calibration_method": "contract-test-only",
                "frame_calibrated_offsets": [
                    {
                        "frame_id": 0,
                        "calibrated_image_offset_xy": [0.0, 0.0],
                    }
                ],
            },
        )
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "does not cover all selected frames",
        ):
            adapter._load_geometry(
                path,
                family="translation",
                object_id="b03_banana_01_high",
                inventory_content_hash=inventory["content_hash_sha256"],
                dataset_semantic_lock_hash=pair[
                    "dataset_semantic_lock_sha256"
                ],
                required_frame_ids={0, 3},
            )

    def test_yaw_requires_projection_and_depth_for_every_frame(self) -> None:
        pair = _read_json("pairs/yaw__world_y__forward__test.json")
        inventory = _read_json("inventories/CORPUS_INVENTORY__yaw.json")
        path = _write_geometry(
            self.temp_root,
            filename="yaw_incomplete.json",
            family="yaw",
            object_id="b03_banana_01_high",
            inventory_hash=inventory["content_hash_sha256"],
            semantic_hash=pair["dataset_semantic_lock_sha256"],
            parameters={
                "frame_projection_geometry": [
                    {
                        "frame_id": 53,
                        "projection_model": "pinhole",
                        "depth_configuration": "planar",
                    }
                ]
            },
        )
        with self.assertRaisesRegex(
            adapter.SplitAdapterError,
            "does not cover all selected frames",
        ):
            adapter._load_geometry(
                path,
                family="yaw",
                object_id="b03_banana_01_high",
                inventory_content_hash=inventory["content_hash_sha256"],
                dataset_semantic_lock_hash=pair[
                    "dataset_semantic_lock_sha256"
                ],
                required_frame_ids={53, 59},
            )

    def test_geometry_evidence_is_rehashed(self) -> None:
        document = json.loads(self.scale_geometry.read_text(encoding="utf-8"))
        document["evidence_files"][0]["sha256"] = "0" * 64
        document.pop("content_hash_sha256")
        document["content_hash_sha256"] = hashlib.sha256(
            canonical_json_bytes(document)
        ).hexdigest()
        path = self.temp_root / "scale_bad_evidence_hash.json"
        path.write_bytes(canonical_json_bytes(document) + b"\n")
        pair = _read_json("pairs/scale__uniform__forward__test.json")
        inventory = _read_json("inventories/CORPUS_INVENTORY__scale.json")
        with self.assertRaisesRegex(adapter.SplitAdapterError, "evidence hash mismatch"):
            adapter._load_geometry(
                path,
                family="scale",
                object_id="b03_banana_01_high",
                inventory_content_hash=inventory["content_hash_sha256"],
                dataset_semantic_lock_hash=pair[
                    "dataset_semantic_lock_sha256"
                ],
                required_frame_ids={53, 57},
            )


if __name__ == "__main__":
    unittest.main()
