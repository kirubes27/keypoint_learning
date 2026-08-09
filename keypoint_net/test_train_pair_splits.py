"""Focused CPU tests for strict indexed training modes.

Run without pytest:
    /opt/anaconda3/envs/phd/bin/python -m unittest \
        keypoint_net/test_train_pair_splits.py -v

The project deletion policy forbids automatic cleanup, so these tests leave
small uniquely named fixture directories under the operating-system temp root.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


KEYPOINT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from dataset import IndexPairDataset, inspect_index_pair_manifest  # noqa: E402
import train  # noqa: E402


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


OBJECT_ROLES = {
    "engineers_hammer_vray": "development",
    "b03_banana_01_high": "confirmation",
    "kettle": "confirmation",
    "dewalt_compact_drill_vray": "final_test",
    "b03_trumpet_vray": "final_test",
    "toy_monkey_medium": "final_test",
}
INCLUDED_OBJECTS = {
    "train": list(OBJECT_ROLES),
    "validation": ["engineers_hammer_vray"],
    "test": [
        name
        for name, role in OBJECT_ROLES.items()
        if role in {"confirmation", "final_test"}
    ],
}


def _new_corpus(_role: str) -> tuple[Path, str]:
    root = Path(tempfile.mkdtemp(prefix="keypoint_index_mode_test_")).resolve()
    binding = _sha256(f"binding:{root.name}".encode("utf-8"))
    source_index = root / "indices" / "all_pairs.json"
    source_index.parent.mkdir(parents=True)
    source_pairs = []
    for object_name in OBJECT_ROLES:
        for src, dst in ((0, 1), (3, 4)):
            source_pairs.append({
                "model_name": object_name,
                "src_frame_index": src,
                "dst_frame_index": dst,
                "src_image_relpath": (
                    f"train/{object_name}/frames/a/img_{src:04d}.png"
                ),
                "dst_image_relpath": (
                    f"train/{object_name}/frames/a/img_{dst:04d}.png"
                ),
                "src_mask_relpath": (
                    f"train/{object_name}/masks/a/mask_{src:04d}.png"
                ),
                "dst_mask_relpath": (
                    f"train/{object_name}/masks/a/mask_{dst:04d}.png"
                ),
            })
    source_index.write_bytes(
        _canonical_json_bytes({"pairs": source_pairs}) + b"\n"
    )
    (root / "splits").mkdir()
    return root, binding


def _write_pair_index(
    root: Path,
    binding: str,
    *,
    split: str,
    src: int,
    dst: int,
) -> Path:
    source_relpath = "indices/all_pairs.json"
    source_path = root / source_relpath
    transform = {
        "family": "translation",
        "physical_axis": "camera_x",
        "direction": "positive",
        "signed_generator": 0.1,
        "generator_units": "world_units",
        "stride": 1,
        "stride_units": "grid_steps",
        "cyclic": False,
        "expected_2d_family": "translation",
    }
    pairs = []
    for object_name in INCLUDED_OBJECTS[split]:
        src_image = f"train/{object_name}/frames/a/img_{src:04d}.png"
        dst_image = f"train/{object_name}/frames/a/img_{dst:04d}.png"
        src_mask = f"train/{object_name}/masks/a/mask_{src:04d}.png"
        dst_mask = f"train/{object_name}/masks/a/mask_{dst:04d}.png"
        for relpath in (src_image, dst_image, src_mask, dst_mask):
            path = root / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"fixture")

        pairs.append({
            "pair_id": f"{split}-{object_name}-{src}-{dst}",
            "model_name": object_name,
            "object_role": OBJECT_ROLES[object_name],
            "split": split,
            "transform_family": transform["family"],
            "physical_axis": transform["physical_axis"],
            "direction": transform["direction"],
            "signed_generator": transform["signed_generator"],
            "generator_units": transform["generator_units"],
            "stride": transform["stride"],
            "stride_units": transform["stride_units"],
            "cyclic": transform["cyclic"],
            "src_frame_index": src,
            "dst_frame_index": dst,
            "src_state": {
                "grid_x": src,
                "grid_y": 0,
                "dx_world": src / 10,
            },
            "dst_state": {
                "grid_x": dst,
                "grid_y": 0,
                "dx_world": dst / 10,
            },
            "src_image_relpath": src_image,
            "dst_image_relpath": dst_image,
            "src_mask_relpath": src_mask,
            "dst_mask_relpath": dst_mask,
        })
    artifact = {
        "schema_version": "representation_pair_index.v1",
        "artifact_type": "representation_pair_index",
        "dataset_basename": root.name,
        "dataset_binding_sha256": binding,
        "dataset_semantic_lock_sha256": _sha256(b"semantic-lock"),
        "source_pair_index_relpath": source_relpath,
        "source_pair_index_sha256": _sha256(source_path.read_bytes()),
        "unversioned_source": True,
        "generator": {
            "name": "unit-test-fixture",
            "commit": "1" * 40,
            "config_sha256": _sha256(b"generator-config"),
        },
        "split": split,
        "object_roles": OBJECT_ROLES,
        "included_objects": INCLUDED_OBJECTS[split],
        "transform": transform,
        "frame_partition": {
            "train_frame_indices": [0, 1],
            "holdout_frame_indices": [3, 4],
            "guard_frame_indices": [2],
        },
        "pair_count": len(pairs),
        "pair_counts_by_object": {
            object_name: 1 for object_name in INCLUDED_OBJECTS[split]
        },
        "pairs": pairs,
    }
    artifact["content_hash_sha256"] = _sha256(_canonical_json_bytes(artifact))
    output_path = root / "splits" / f"{split}_{src}_{dst}.json"
    output_path.write_bytes(_canonical_json_bytes(artifact) + b"\n")
    return output_path


def _args(
    root: Path,
    binding: str,
    *,
    mode: str,
    train_index: Path,
    val_index: Path | None = None,
    test_index: Path | None = None,
    epochs: int = 7,
    frozen_epochs: int | None = None,
    auto_eval: bool = False,
    object_name: str = "engineers_hammer_vray",
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(root),
        object=object_name,
        pairs_index=None,
        indexed_mode=mode,
        train_pairs_index=str(train_index),
        val_pairs_index=None if val_index is None else str(val_index),
        test_pairs_index=None if test_index is None else str(test_index),
        dataset_binding_sha256=binding,
        img_size=64,
        center_crop=None,
        frame_skip=1,
        batch_size=2,
        epochs=epochs,
        frozen_epochs=frozen_epochs,
        auto_eval=auto_eval,
        lambda_act=0.0,
    )


class StrictIndexManifestTests(unittest.TestCase):
    def test_generic_transform_metadata_and_hashes_are_exposed(self) -> None:
        root, binding = _new_corpus("development")
        index_path = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        manifest = inspect_index_pair_manifest(
            str(root),
            str(index_path),
            expected_split="train",
            object_name="engineers_hammer_vray",
            strict_metadata=True,
            expected_dataset_binding_sha256=binding,
        )
        dataset = IndexPairDataset(
            str(root),
            str(index_path),
            strict_metadata=True,
            expected_split="train",
            expected_index_sha256=manifest.index_sha256,
            expected_dataset_binding_sha256=binding,
            manifest=manifest,
        )
        self.assertEqual(dataset.transform_family, "translation")
        self.assertEqual(dataset.signed_generator, 0.1)
        self.assertEqual(dataset.split_role, "train")
        self.assertEqual(dataset.index_sha256, manifest.index_sha256)
        self.assertEqual(
            json.loads(dataset.samples[0]["pair_metadata_json"])[
                "transform_family"
            ],
            "translation",
        )
        self.assertNotIn("delta_theta_deg", dataset.transform_metadata)

    def test_changed_index_file_fails_preflight_hash(self) -> None:
        root, binding = _new_corpus("development")
        index_path = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        manifest = inspect_index_pair_manifest(
            str(root),
            str(index_path),
            strict_metadata=True,
            expected_split="train",
            expected_dataset_binding_sha256=binding,
        )
        index_path.write_bytes(index_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            inspect_index_pair_manifest(
                str(root),
                str(index_path),
                strict_metadata=True,
                expected_split="train",
                expected_index_sha256=manifest.index_sha256,
                expected_dataset_binding_sha256=binding,
            )

    def test_external_dataset_binding_is_required_and_compared(self) -> None:
        root, binding = _new_corpus("development")
        index_path = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        with self.assertRaisesRegex(ValueError, "independently supplied"):
            inspect_index_pair_manifest(
                str(root),
                str(index_path),
                strict_metadata=True,
                expected_split="train",
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            inspect_index_pair_manifest(
                str(root),
                str(index_path),
                strict_metadata=True,
                expected_split="train",
                expected_dataset_binding_sha256="0" * 64,
            )

    def test_pair_endpoint_must_belong_to_declared_split_partition(self) -> None:
        root, binding = _new_corpus("development")
        index_path = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        artifact = json.loads(index_path.read_text())
        artifact["pairs"][0]["src_frame_index"] = 3
        del artifact["content_hash_sha256"]
        artifact["content_hash_sha256"] = _sha256(
            _canonical_json_bytes(artifact)
        )
        index_path.write_bytes(_canonical_json_bytes(artifact) + b"\n")
        with self.assertRaisesRegex(ValueError, "declared train frame partition"):
            inspect_index_pair_manifest(
                str(root),
                str(index_path),
                strict_metadata=True,
                expected_split="train",
                expected_dataset_binding_sha256=binding,
            )

    def test_signed_generator_must_be_a_finite_real_scalar(self) -> None:
        root, binding = _new_corpus("development")
        index_path = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        artifact = json.loads(index_path.read_text())
        artifact["transform"]["signed_generator"] = "not-numeric"
        for pair in artifact["pairs"]:
            pair["signed_generator"] = "not-numeric"
        del artifact["content_hash_sha256"]
        artifact["content_hash_sha256"] = _sha256(
            _canonical_json_bytes(artifact)
        )
        index_path.write_bytes(_canonical_json_bytes(artifact) + b"\n")
        with self.assertRaisesRegex(ValueError, "finite real scalar"):
            inspect_index_pair_manifest(
                str(root),
                str(index_path),
                strict_metadata=True,
                expected_split="train",
                expected_dataset_binding_sha256=binding,
            )


class TrainingModePlanTests(unittest.TestCase):
    def test_development_has_distinct_validation_and_best_policy(self) -> None:
        root, binding = _new_corpus("development")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        val_index = _write_pair_index(
            root,
            binding,
            split="validation",
            src=3,
            dst=4,
        )
        args = _args(
            root,
            binding,
            mode="development",
            train_index=train_index,
            val_index=val_index,
        )
        plan = train._prepare_training_data(args, include_backward=False)
        self.assertEqual(plan.mode, "development")
        self.assertIsNotNone(plan.val_dataset)
        self.assertIsNone(plan.test_manifest)
        self.assertIsNot(plan.train_dataset, plan.val_dataset)
        self.assertFalse(
            plan.train_dataset.endpoint_ids & plan.val_dataset.endpoint_ids
        )
        self.assertEqual(
            plan.checkpoint_policy["authoritative_checkpoint"],
            "best_model.pt",
        )
        self.assertEqual(plan.checkpoint_policy["post_training_test_evaluations"], 0)
        self.assertEqual(set(plan.index_provenance), {"train", "validation"})

    def test_fixed_final_defers_test_dataset_and_uses_final_policy(self) -> None:
        root, binding = _new_corpus("final_test")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        test_index = _write_pair_index(
            root,
            binding,
            split="test",
            src=3,
            dst=4,
        )
        args = _args(
            root,
            binding,
            mode="fixed-final",
            train_index=train_index,
            test_index=test_index,
            epochs=7,
            frozen_epochs=7,
            object_name="dewalt_compact_drill_vray",
        )
        plan = train._prepare_training_data(args, include_backward=False)
        self.assertEqual(plan.mode, "fixed-final")
        self.assertIsNone(plan.val_dataset)
        self.assertIsNotNone(plan.test_manifest)
        self.assertEqual(
            plan.checkpoint_policy["authoritative_checkpoint"],
            "final_model.pt",
        )
        self.assertFalse(plan.checkpoint_policy["best_model_written"])
        self.assertEqual(plan.checkpoint_policy["post_training_test_evaluations"], 1)

        self.assertFalse(hasattr(train, "_build_fixed_final_test_loader"))

    def test_source_row_tampering_fails_before_output_directory_exists(self) -> None:
        root, binding = _new_corpus("development")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        val_index = _write_pair_index(
            root,
            binding,
            split="validation",
            src=3,
            dst=4,
        )
        val_artifact = json.loads(val_index.read_text())
        val_artifact["pairs"][0]["src_image_relpath"] = (
            "train/engineers_hammer_vray/frames/a/img_0001.png"
        )
        del val_artifact["content_hash_sha256"]
        val_artifact["content_hash_sha256"] = _sha256(
            _canonical_json_bytes(val_artifact)
        )
        val_index.write_bytes(_canonical_json_bytes(val_artifact) + b"\n")
        args = _args(
            root,
            binding,
            mode="development",
            train_index=train_index,
            val_index=val_index,
        )
        output_dir = root / "must_not_exist"
        args.output_dir = str(output_dir)
        with self.assertRaisesRegex(ValueError, "bound source pair-index row"):
            train._prepare_training_data(args, include_backward=False)
        self.assertFalse(output_dir.exists())

    def test_endpoint_overlap_guard_rejects_prevalidated_manifests(self) -> None:
        root, binding = _new_corpus("development")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        val_index = _write_pair_index(
            root,
            binding,
            split="validation",
            src=3,
            dst=4,
        )
        train_manifest = inspect_index_pair_manifest(
            str(root),
            str(train_index),
            expected_split="train",
            object_name="engineers_hammer_vray",
            strict_metadata=True,
            expected_dataset_binding_sha256=binding,
        )
        val_manifest = inspect_index_pair_manifest(
            str(root),
            str(val_index),
            expected_split="validation",
            object_name="engineers_hammer_vray",
            strict_metadata=True,
            expected_dataset_binding_sha256=binding,
        )
        aliased_val_manifest = replace(
            val_manifest,
            source_endpoint_ids=train_manifest.source_endpoint_ids,
            target_endpoint_ids=train_manifest.target_endpoint_ids,
            endpoint_ids=train_manifest.endpoint_ids,
        )
        with self.assertRaisesRegex(ValueError, "share source/target"):
            train._validate_manifest_pair(
                train_manifest,
                aliased_val_manifest,
                object_name="engineers_hammer_vray",
                mode="development",
            )

    def test_index_only_legacy_call_and_auto_eval_fail_closed(self) -> None:
        root, binding = _new_corpus("development")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        args = _args(
            root,
            binding,
            mode="development",
            train_index=train_index,
            val_index=train_index,
        )
        args.indexed_mode = None
        args.train_pairs_index = None
        args.val_pairs_index = None
        args.pairs_index = str(train_index)
        with self.assertRaisesRegex(ValueError, "explicit --indexed_mode"):
            train._resolve_index_argument_policy(args)

        args = _args(
            root,
            binding,
            mode="development",
            train_index=train_index,
            val_index=train_index,
            auto_eval=True,
        )
        with self.assertRaisesRegex(ValueError, "not partition-aware"):
            train._resolve_index_argument_policy(args)

    def test_fixed_final_epoch_mismatch_fails(self) -> None:
        root, binding = _new_corpus("final_test")
        train_index = _write_pair_index(
            root,
            binding,
            split="train",
            src=0,
            dst=1,
        )
        test_index = _write_pair_index(
            root,
            binding,
            split="test",
            src=3,
            dst=4,
        )
        args = _args(
            root,
            binding,
            mode="fixed-final",
            train_index=train_index,
            test_index=test_index,
            epochs=8,
            frozen_epochs=7,
            object_name="dewalt_compact_drill_vray",
        )
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            train._resolve_index_argument_policy(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
