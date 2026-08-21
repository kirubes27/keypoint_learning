"""Focused CPU tests for the frozen representation primary split gate."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from keypoint_net.representation_corpus_inventory import (
    CorpusInventoryError,
    _scan_regular_files,
    build_all_corpus_inventories,
    canonical_json_bytes as inventory_json_bytes,
    inventory_content_hash,
    validate_corpus_inventory,
)
from keypoint_net.representation_split_bundle import (
    SplitBundleError,
    build_primary_split_bundle,
    validate_primary_split_bundle,
)
from keypoint_net.representation_split_verifier import (
    SplitVerificationError,
    verify_primary_split_artifacts,
)
from keypoint_net.representation_splits import (
    OBJECT_ROLES,
    PRIMARY_GROUPS,
    SplitGenerationError,
    canonical_json_bytes,
    content_hash_sha256,
    generate_primary_split_artifacts,
    validate_source_pair_document,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FUTURE_ROOT = Path("/Users/kirubeso.r/Documents/PhD/data/active")
DATASET_ROOTS = {
    "roll": REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2",
    "yaw": FUTURE_ROOT / "_tdw_world_y_yaw_arc60_step1_base_panel_512_v1",
    "pitch": FUTURE_ROOT / "_tdw_world_x_pitch_arc60_step1_base_panel_512_v1",
    "scale": FUTURE_ROOT / "_tdw_uniform_scale_loghalf_to_one_base_panel_512_v1",
    "translation": FUTURE_ROOT / "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
}
GENERATOR_COMMIT = "a" * 40
ROLL_INVENTORY_PATH = (
    REPO_ROOT
    / "docs/decisions/2026-07-26/representation_oracle_splits/inventories/"
    "CORPUS_INVENTORY__roll.json"
)


def _decode(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def _replace_artifact(
    artifacts: dict[str, bytes],
    filename: str,
    mutation,
    *,
    repair_content_hash: bool = True,
) -> dict[str, bytes]:
    changed = dict(artifacts)
    document = _decode(changed[filename])
    mutation(document)
    if repair_content_hash:
        document["content_hash_sha256"] = content_hash_sha256(document)
    changed[filename] = canonical_json_bytes(document, trailing_newline=True)
    return changed


def _replace_inventory(
    inventories: dict[str, bytes],
    dataset_key: str,
    mutation,
    *,
    repair_content_hash: bool = True,
) -> dict[str, bytes]:
    changed = dict(inventories)
    document = _decode(changed[dataset_key])
    mutation(document)
    if repair_content_hash:
        document["content_hash_sha256"] = inventory_content_hash(document)
    changed[dataset_key] = inventory_json_bytes(document, trailing_newline=True)
    return changed


def _replace_manifest(bundle: dict[str, bytes], mutation) -> dict[str, bytes]:
    changed = dict(bundle)
    document = _decode(changed["SPLIT_MANIFEST.json"])
    mutation(document)
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    document["content_hash_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    changed["SPLIT_MANIFEST.json"] = canonical_json_bytes(
        document,
        trailing_newline=True,
    )
    return changed


class CorpusInventoryPortabilityTests(unittest.TestCase):
    def test_inventory_allows_only_source_root_relocation(self) -> None:
        original = ROLL_INVENTORY_PATH.read_bytes()
        relocated = _decode(original)
        relocated["source_root_provenance"] = (
            "/work/scratch/example/_tdw_world_z_roll_base_panel_512_v2"
        )
        relocated["content_hash_sha256"] = inventory_content_hash(relocated)
        relocated_bytes = inventory_json_bytes(relocated, trailing_newline=True)

        with patch(
            "keypoint_net.representation_corpus_inventory.build_corpus_inventory",
            return_value=relocated_bytes,
        ):
            validated = validate_corpus_inventory(
                original,
                "roll",
                DATASET_ROOTS["roll"],
            )
        self.assertEqual(
            _decode(original)["content_hash_sha256"],
            validated.content_hash_sha256,
        )

        relocated["files"][0]["size_bytes"] += 1
        relocated["content_hash_sha256"] = inventory_content_hash(relocated)
        changed_bytes = inventory_json_bytes(relocated, trailing_newline=True)
        with patch(
            "keypoint_net.representation_corpus_inventory.build_corpus_inventory",
            return_value=changed_bytes,
        ):
            with self.assertRaisesRegex(
                CorpusInventoryError,
                "does not match the current exact corpus contents/metadata",
            ):
                validate_corpus_inventory(
                    original,
                    "roll",
                    DATASET_ROOTS["roll"],
                )


class RepresentationSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [str(path) for path in DATASET_ROOTS.values() if not path.is_dir()]
        if missing:
            raise AssertionError(f"required immutable dataset roots are missing: {missing}")
        cls.inventories = build_all_corpus_inventories(DATASET_ROOTS)
        cls.validated_inventories = {
            key: validate_corpus_inventory(cls.inventories[key], key, DATASET_ROOTS[key])
            for key in DATASET_ROOTS
        }
        cls.first = generate_primary_split_artifacts(
            DATASET_ROOTS,
            corpus_inventories=cls.inventories,
            generator_commit=GENERATOR_COMMIT,
        )
        cls.second = generate_primary_split_artifacts(
            DATASET_ROOTS,
            corpus_inventories=cls.inventories,
            generator_commit=GENERATOR_COMMIT,
        )

    def _verify(self, artifacts: dict[str, bytes], *, regenerated=None) -> dict:
        return verify_primary_split_artifacts(
            artifacts,
            DATASET_ROOTS,
            corpus_inventories=self.inventories,
            expected_generator_commit=GENERATOR_COMMIT,
            regenerated_artifacts=regenerated,
        )

    def test_exact_artifact_set_and_byte_identical_regeneration(self) -> None:
        self.assertEqual(33, len(self.first))
        self.assertEqual(self.first, self.second)
        report = self._verify(self.first, regenerated=self.second)
        self.assertTrue(report["structurally_valid"])
        self.assertTrue(report["validated_corpus_inventories"])
        self.assertEqual(33, report["artifact_count"])
        self.assertTrue(report["byte_identical_regeneration"])
        self.assertTrue(report["gate_pass"])

    def test_partial_verification_is_not_a_gate_pass(self) -> None:
        report = self._verify(self.first)
        self.assertTrue(report["structurally_valid"])
        self.assertTrue(report["validated_corpus_inventories"])
        self.assertFalse(report["byte_identical_regeneration"])
        self.assertFalse(report["gate_pass"])

    def test_exact_roles_splits_and_per_object_counts(self) -> None:
        expected_counts = {
            "roll": (147, 21),
            "yaw": (82, 9),
            "pitch": (82, 9),
            "scale": (90, 11),
            "translation": (56, 16),
        }
        for filename, raw in self.first.items():
            document = _decode(raw)
            family = document["transform"]["family"]
            split = document["split"]
            per_object = expected_counts[family][0 if split == "train" else 1]
            if split == "train":
                expected_objects = list(OBJECT_ROLES)
            elif split == "validation":
                expected_objects = ["engineers_hammer_vray"]
            else:
                expected_objects = [
                    model for model, role in OBJECT_ROLES.items() if role != "development"
                ]
            self.assertEqual(expected_objects, document["included_objects"], filename)
            self.assertEqual(
                {model: per_object for model in expected_objects},
                document["pair_counts_by_object"],
                filename,
            )
            self.assertEqual(
                per_object * len(expected_objects),
                document["pair_count"],
                filename,
            )
            self.assertEqual(document["pair_count"], len(document["pairs"]), filename)
            for pair in document["pairs"]:
                self.assertEqual(split, pair["split"], filename)
                self.assertEqual(OBJECT_ROLES[pair["model_name"]], pair["object_role"])

    def test_endpoint_disjointness_guards_wrap_and_translation_axes(self) -> None:
        documents = {filename: _decode(raw) for filename, raw in self.first.items()}
        for group in PRIMARY_GROUPS:
            stem = group.artifact_stem
            train = documents[f"{stem}__train.json"]
            validation = documents[f"{stem}__validation.json"]
            test = documents[f"{stem}__test.json"]
            guards = set(train["frame_partition"]["guard_frame_indices"])
            train_by_object: dict[str, set[int]] = {}
            holdout_by_object: dict[str, set[int]] = {}
            for pair in train["pairs"]:
                endpoints = train_by_object.setdefault(pair["model_name"], set())
                endpoints.update((pair["src_frame_index"], pair["dst_frame_index"]))
                self.assertFalse(
                    guards.intersection((pair["src_frame_index"], pair["dst_frame_index"]))
                )
            for heldout_document in (validation, test):
                for pair in heldout_document["pairs"]:
                    endpoints = holdout_by_object.setdefault(pair["model_name"], set())
                    endpoints.update((pair["src_frame_index"], pair["dst_frame_index"]))
                    self.assertFalse(
                        guards.intersection((pair["src_frame_index"], pair["dst_frame_index"]))
                    )
            for model in OBJECT_ROLES:
                self.assertTrue(
                    train_by_object[model].isdisjoint(holdout_by_object[model]),
                    f"{stem}/{model}: train and held-out endpoints overlap",
                )

            if group.family == "roll":
                self.assertTrue(train["transform"]["cyclic"])
                for document in (train, validation, test):
                    self.assertTrue(
                        all(
                            pair["dst_frame_index"] > pair["src_frame_index"]
                            for pair in document["pairs"]
                        ),
                        "wrap-crossing roll pairs must be dropped by the frozen guards",
                    )
            if group.family == "translation":
                for document in (train, validation, test):
                    for pair in document["pairs"]:
                        src, dst = pair["src_state"], pair["dst_state"]
                        if group.physical_axis == "world_x":
                            self.assertEqual(src["grid_y"], dst["grid_y"])
                            self.assertEqual(3, abs(src["grid_x"] - dst["grid_x"]))
                        else:
                            self.assertEqual(src["grid_x"], dst["grid_x"])
                            self.assertEqual(3, abs(src["grid_y"] - dst["grid_y"]))

    def test_scale_absolute_holdout_is_absent(self) -> None:
        scale_files = [
            filename for filename in self.first if filename.startswith("scale__uniform__")
        ]
        self.assertEqual(6, len(scale_files))
        for filename in scale_files:
            document = _decode(self.first[filename])
            self.assertNotIn("pairs_eval_abs_holdout_ratio_r4", document["source_pair_index_relpath"])
            for pair in document["pairs"]:
                for key in (
                    "src_image_relpath",
                    "dst_image_relpath",
                    "src_mask_relpath",
                    "dst_mask_relpath",
                ):
                    self.assertNotIn("eval_abs_holdout", pair[key], filename)

    def test_strict_source_schema_rejects_missing_pair_field(self) -> None:
        group = PRIMARY_GROUPS[0]
        source_path = DATASET_ROOTS["roll"] / group.source_pair_index_relpath
        source = json.loads(source_path.read_text(encoding="utf-8"))
        malformed = copy.deepcopy(source)
        del malformed["pairs"][0]["cyclic"]
        with self.assertRaisesRegex(SplitGenerationError, "schema mismatch"):
            validate_source_pair_document(
                malformed,
                group,
                DATASET_ROOTS["roll"],
                inventory=self.validated_inventories["roll"],
            )

    def test_strict_source_schema_rejects_numeric_type_drift(self) -> None:
        group = PRIMARY_GROUPS[0]
        source_path = DATASET_ROOTS["roll"] / group.source_pair_index_relpath
        malformed = json.loads(source_path.read_text(encoding="utf-8"))
        malformed["skip_frames"] = 3.0
        with self.assertRaisesRegex(SplitGenerationError, "wrong skip_frames"):
            validate_source_pair_document(
                malformed,
                group,
                DATASET_ROOTS["roll"],
                inventory=self.validated_inventories["roll"],
            )

    def test_generator_requires_complete_inventory_bindings(self) -> None:
        incomplete = dict(self.inventories)
        del incomplete["scale"]
        with self.assertRaisesRegex(SplitGenerationError, "expected exact keys"):
            generate_primary_split_artifacts(
                DATASET_ROOTS,
                corpus_inventories=incomplete,
                generator_commit=GENERATOR_COMMIT,
            )

    def test_fake_inventory_binding_is_rejected(self) -> None:
        changed = _replace_inventory(
            self.inventories,
            "translation",
            lambda document: document.__setitem__("content_hash_sha256", "0" * 64),
            repair_content_hash=False,
        )
        with self.assertRaisesRegex(CorpusInventoryError, "content hash mismatch"):
            generate_primary_split_artifacts(
                DATASET_ROOTS,
                corpus_inventories=changed,
                generator_commit=GENERATOR_COMMIT,
            )

    def test_source_path_must_match_frame_index_for_roll_and_translation(self) -> None:
        groups = [
            next(group for group in PRIMARY_GROUPS if group.family == "roll"),
            next(
                group
                for group in PRIMARY_GROUPS
                if group.family == "translation"
                and group.physical_axis == "world_x"
                and group.direction == "forward"
            ),
        ]
        for group in groups:
            root = DATASET_ROOTS[group.dataset_key]
            source = json.loads(
                (root / group.source_pair_index_relpath).read_text(encoding="utf-8")
            )
            malformed = copy.deepcopy(source)
            row = malformed["pairs"][0]
            self.assertEqual(0, row["src_frame_index"])
            row["src_image_relpath"] = row["src_image_relpath"].replace(
                "img_0000.png", "img_0001.png"
            )
            with self.assertRaisesRegex(SplitGenerationError, "expected canonical path"):
                validate_source_pair_document(
                    malformed,
                    group,
                    root,
                    inventory=self.validated_inventories[group.dataset_key],
                )

    def test_inventory_metadata_state_tamper_is_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["frame_records"][0]["physical_state"]["theta_deg"] = 999.0

        changed = _replace_inventory(self.inventories, "yaw", mutate)
        with self.assertRaisesRegex(
            CorpusInventoryError,
            "does not match the current exact corpus contents/metadata",
        ):
            validate_corpus_inventory(changed["yaw"], "yaw", DATASET_ROOTS["yaw"])

    def test_inventory_file_addition_and_removal_tamper_are_rejected(self) -> None:
        def add_file(document: dict) -> None:
            document["files"].append(
                {
                    "relative_path": "invented.bin",
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                }
            )
            document["files"].sort(key=lambda item: item["relative_path"])
            document["file_count"] += 1

        added = _replace_inventory(self.inventories, "translation", add_file)
        with self.assertRaisesRegex(
            CorpusInventoryError,
            "does not match the current exact corpus contents/metadata",
        ):
            validate_corpus_inventory(
                added["translation"],
                "translation",
                DATASET_ROOTS["translation"],
            )

        def remove_file(document: dict) -> None:
            removed = document["files"].pop()
            document["file_count"] -= 1
            document["total_bytes"] -= removed["size_bytes"]

        removed = _replace_inventory(self.inventories, "translation", remove_file)
        with self.assertRaisesRegex(
            CorpusInventoryError,
            "does not match the current exact corpus contents/metadata",
        ):
            validate_corpus_inventory(
                removed["translation"],
                "translation",
                DATASET_ROOTS["translation"],
            )

    def test_ds_store_symlink_and_nonregular_entry_are_rejected(self) -> None:
        fake_path = str(DATASET_ROOTS["roll"] / ".DS_Store")
        unsafe_entries = (
            (
                SimpleNamespace(
                    name=".DS_Store",
                    path=fake_path,
                    is_symlink=lambda: True,
                ),
                "symlinks are forbidden",
            ),
            (
                SimpleNamespace(
                    name=".DS_Store",
                    path=fake_path,
                    is_symlink=lambda: False,
                    stat=lambda **_: SimpleNamespace(st_mode=stat.S_IFIFO),
                ),
                "non-regular filesystem entry",
            ),
        )
        for entry, expected_error in unsafe_entries:
            with self.subTest(expected_error=expected_error):
                with patch(
                    "keypoint_net.representation_corpus_inventory.os.scandir",
                    return_value=[entry],
                ):
                    with self.assertRaisesRegex(CorpusInventoryError, expected_error):
                        _scan_regular_files(DATASET_ROOTS["roll"])

    def test_verifier_rejects_content_hash_tampering(self) -> None:
        filename = "roll__world_z__forward__validation.json"

        def mutate(document: dict) -> None:
            document["content_hash_sha256"] = "0" * 64

        changed = _replace_artifact(
            self.first,
            filename,
            mutate,
            repair_content_hash=False,
        )
        with self.assertRaisesRegex(SplitVerificationError, "content hash mismatch"):
            self._verify(changed)

    def test_verifier_rejects_guard_or_boundary_pair_even_with_valid_hash(self) -> None:
        filename = "roll__world_z__forward__train.json"

        def mutate(document: dict) -> None:
            pair = document["pairs"][0]
            pair["src_frame_index"] = 24
            pair["dst_frame_index"] = 27

        changed = _replace_artifact(self.first, filename, mutate)
        with self.assertRaisesRegex(
            SplitVerificationError,
            "endpoint outside named split|guard endpoint retained",
        ):
            self._verify(changed)

    def test_verifier_rejects_swapped_translation_axis_even_with_valid_hash(self) -> None:
        filename = "translation__world_x__forward__validation.json"

        def mutate(document: dict) -> None:
            document["transform"]["physical_axis"] = "world_y"

        changed = _replace_artifact(self.first, filename, mutate)
        with self.assertRaisesRegex(SplitVerificationError, "transform metadata mismatch"):
            self._verify(changed)

    def test_verifier_rejects_nonidentical_regeneration(self) -> None:
        filename = "yaw__world_y__reverse__test.json"
        changed = dict(self.second)
        changed[filename] = changed[filename] + b" "
        with self.assertRaisesRegex(SplitVerificationError, "not byte-identical"):
            self._verify(self.first, regenerated=changed)

    def test_complete_bundle_contains_manifest_report_inventories_and_pairs(self) -> None:
        with patch(
            "keypoint_net.representation_split_bundle."
            "verify_commit_contains_generator_sources"
        ):
            bundle = build_primary_split_bundle(
                DATASET_ROOTS,
                generator_commit=GENERATOR_COMMIT,
            )
            validated_manifest = validate_primary_split_bundle(
                bundle,
                verify_corpus_contents=True,
            )
        self.assertEqual(40, len(bundle))
        manifest = _decode(bundle["SPLIT_MANIFEST.json"])
        report = _decode(bundle["SPLIT_VERIFIER_REPORT.json"])
        self.assertEqual(manifest, validated_manifest)
        self.assertEqual(33, manifest["pair_artifact_count"])
        self.assertEqual(11, len(manifest["group_summaries"]))
        self.assertTrue(manifest["regeneration"]["corpus_inventory_byte_identical"])
        self.assertTrue(manifest["regeneration"]["pair_artifact_byte_identical"])
        self.assertEqual(
            0,
            manifest["scale_absolute_holdout_nonmembership"][
                "primary_pair_path_intersection_count"
            ],
        )
        self.assertTrue(report["structurally_valid"])
        self.assertTrue(report["validated_corpus_inventories"])
        self.assertTrue(report["byte_identical_regeneration"])
        self.assertTrue(report["gate_pass"])

    def test_public_bundle_builder_cannot_bypass_commit_binding(self) -> None:
        with self.assertRaisesRegex(SplitBundleError, "generator commit does not exist"):
            build_primary_split_bundle(
                DATASET_ROOTS,
                generator_commit=GENERATOR_COMMIT,
            )

    def test_rehashed_false_manifest_summaries_are_rejected(self) -> None:
        with patch(
            "keypoint_net.representation_split_bundle."
            "verify_commit_contains_generator_sources"
        ):
            bundle = build_primary_split_bundle(
                DATASET_ROOTS,
                generator_commit=GENERATOR_COMMIT,
            )
            mutations = (
                (
                    lambda document: document["object_role_lock"]["roles"].__setitem__(
                        "engineers_hammer_vray",
                        "final_test",
                    ),
                    "object-role lock mismatch",
                ),
                (
                    lambda document: document["pair_artifacts"][0].__setitem__(
                        "pair_count",
                        0,
                    ),
                    "pair summaries disagree",
                ),
                (
                    lambda document: document["group_summaries"][0].__setitem__(
                        "dropped_pair_count",
                        0,
                    ),
                    "group summaries disagree",
                ),
                (
                    lambda document: document[
                        "scale_absolute_holdout_nonmembership"
                    ].__setitem__("excluded_frame_count", 0),
                    "scale absolute-holdout proof mismatch",
                ),
            )
            for mutation, expected_error in mutations:
                with self.subTest(expected_error=expected_error):
                    changed = _replace_manifest(bundle, mutation)
                    with self.assertRaisesRegex(SplitBundleError, expected_error):
                        validate_primary_split_bundle(
                            changed,
                            verify_corpus_contents=False,
                        )


if __name__ == "__main__":
    unittest.main()
