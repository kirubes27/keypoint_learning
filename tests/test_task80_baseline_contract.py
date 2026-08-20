from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "docs" / "baselines" / "task80-assisted-roll"
CONFIG_PATH = BASELINE_DIR / "config.json"
MANIFEST_PATH = BASELINE_DIR / "MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


class Task80StaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(CONFIG_PATH)
        cls.manifest = load_json(MANIFEST_PATH)

    def test_bound_config_is_byte_exact(self) -> None:
        binding = self.manifest["configuration"]
        self.assertEqual(CONFIG_PATH.stat().st_size, binding["size_bytes"])
        self.assertEqual(sha256_file(CONFIG_PATH), binding["sha256"])

    def test_core_source_files_match_the_baseline(self) -> None:
        for relative_path, expected_hash in self.manifest["source"]["core_files_sha256"].items():
            self.assertEqual(sha256_file(ROOT / relative_path), expected_hash, relative_path)

    def test_baseline_commit_is_an_ancestor(self) -> None:
        base = self.manifest["source"]["base_commit"]
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"{base} is not an ancestor of HEAD")

    def test_active_and_inert_config_semantics(self) -> None:
        cfg = self.config
        self.assertEqual(cfg["operator_type"], "shared_affine")
        self.assertEqual(cfg["padding_mode"], "reflect")
        self.assertEqual(cfg["num_keypoints"], 10)
        self.assertNotIn("heatmap_res", cfg)
        self.assertNotIn("true_quarter_res", cfg)
        effective_inverse = bool(
            cfg["learn_inverse_operator"]
            or cfg["lambda_inv"] > 0
            or cfg["lambda_cycle"] > 0
        )
        effective_action_classes = cfg["num_action_classes"] if cfg["lambda_act"] > 0 else 0
        self.assertTrue(effective_inverse)
        self.assertTrue(cfg["learn_inverse_operator_effective"])
        self.assertEqual(effective_action_classes, 0)

    def test_model_architecture_contract(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"PyTorch unavailable: {exc}")
        sys.path.insert(0, str(ROOT / "keypoint_net"))
        from model import PhaseAModel

        cfg = self.config
        effective_inverse = bool(
            cfg["learn_inverse_operator"]
            or cfg["lambda_inv"] > 0
            or cfg["lambda_cycle"] > 0
        )
        model = PhaseAModel(
            num_keypoints=cfg["num_keypoints"],
            base_channels=cfg["base_channels"],
            temperature=cfg["temperature"],
            num_action_classes=0,
            padding_mode=cfg["padding_mode"],
            operator_type=cfg["operator_type"],
            learn_inverse_operator=effective_inverse,
        )
        expected = self.manifest["model_contract"]
        self.assertEqual(sum(p.numel() for p in model.parameters()), expected["trainable_parameters"])
        self.assertEqual(sum(p.numel() for p in model.operator.parameters()), expected["forward_affine_parameters"])
        self.assertIsNotNone(model.inverse_operator)
        self.assertEqual(sum(p.numel() for p in model.inverse_operator.parameters()), expected["inverse_affine_parameters"])
        self.assertIsNone(model.action_classifier)
        self.assertEqual(model.extractor.padding_mode, expected["padding_mode"])
        self.assertEqual(model.extractor.heatmap_head.out_channels, expected["num_keypoints"])


class Task80ExternalArtifactContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        checkpoint_value = os.environ.get("TASK80_CHECKPOINT")
        data_root_value = os.environ.get("TASK80_DATA_ROOT")
        if not checkpoint_value or not data_root_value:
            raise unittest.SkipTest(
                "set TASK80_CHECKPOINT and TASK80_DATA_ROOT to run the bound external-artifact check"
            )
        cls.checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        cls.data_root = Path(data_root_value).expanduser().resolve()
        cls.config = load_json(CONFIG_PATH)
        cls.manifest = load_json(MANIFEST_PATH)

    def test_bound_checkpoint_dataset_and_forward_behavior(self) -> None:
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"PyTorch unavailable: {exc}")
        sys.path.insert(0, str(ROOT / "keypoint_net"))
        from dataset import IndexPairDataset
        from model import PhaseAModel

        cfg = self.config
        manifest = self.manifest
        checkpoint_binding = manifest["checkpoint"]
        dataset_binding = manifest["dataset"]
        fixture = manifest["behavior_fixture"]

        self.assertTrue(self.checkpoint_path.is_file())
        self.assertTrue(self.data_root.is_dir())
        self.assertEqual(self.checkpoint_path.stat().st_size, checkpoint_binding["size_bytes"])
        self.assertEqual(sha256_file(self.checkpoint_path), checkpoint_binding["sha256"])
        self.assertEqual(sha256_file(self.data_root / "dataset_index.json"), dataset_binding["dataset_index_sha256"])

        pair_index = self.data_root / dataset_binding["pair_index_relative_path"]
        self.assertEqual(sha256_file(pair_index), dataset_binding["pair_index_sha256"])
        source_image = self.data_root / fixture["source_image_relative_path"]
        target_image = self.data_root / fixture["target_image_relative_path"]
        self.assertEqual(sha256_file(source_image), fixture["source_image_sha256"])
        self.assertEqual(sha256_file(target_image), fixture["target_image_sha256"])

        dataset = IndexPairDataset(
            str(self.data_root),
            str(pair_index),
            img_size=cfg["img_size"],
            center_crop=cfg["center_crop"],
            include_backward=False,
            object_name=cfg["object"],
        )
        sample = dataset[0]
        self.assertEqual((sample["t"], sample["t1"]), (fixture["source_frame"], fixture["target_frame"]))
        self.assertEqual(sha256_tensor(sample["x_t"]), fixture["source_input_tensor_sha256"])
        self.assertEqual(sha256_tensor(sample["x_t1"]), fixture["target_input_tensor_sha256"])

        effective_inverse = bool(
            cfg["learn_inverse_operator"]
            or cfg["lambda_inv"] > 0
            or cfg["lambda_cycle"] > 0
        )
        effective_action_classes = cfg["num_action_classes"] if cfg["lambda_act"] > 0 else 0
        model = PhaseAModel(
            num_keypoints=cfg["num_keypoints"],
            base_channels=cfg["base_channels"],
            temperature=cfg["temperature"],
            num_action_classes=effective_action_classes,
            padding_mode=cfg["padding_mode"],
            operator_type=cfg["operator_type"],
            learn_inverse_operator=effective_inverse,
        ).cpu().eval()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertEqual(checkpoint["epoch"], checkpoint_binding["epoch"])

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.inference_mode():
            first = model(sample["x_t"].unsqueeze(0), sample["x_t1"].unsqueeze(0))
            second = model(sample["x_t"].unsqueeze(0), sample["x_t1"].unsqueeze(0))

        comparison = fixture["coordinate_comparison"]
        for key in ("p_t", "p_t1", "p_hat_t1"):
            expected = torch.tensor(fixture["outputs"][key]["values"], dtype=first[key].dtype).reshape(fixture["outputs"][key]["shape"])
            torch.testing.assert_close(first[key].cpu(), expected, rtol=comparison["rtol"], atol=comparison["atol"])
            self.assertTrue(torch.equal(first[key], second[key]), key)

        for key in ("heatmaps_t", "heatmaps_t1"):
            self.assertEqual(list(first[key].shape), fixture["outputs"][key]["shape"])
            self.assertTrue(torch.equal(first[key], second[key]), key)

        if os.environ.get("TASK80_REQUIRE_EXACT_OUTPUT_HASHES", "0") == "1":
            for key, expected in fixture["outputs"].items():
                self.assertEqual(sha256_tensor(first[key]), expected["sha256"], key)


if __name__ == "__main__":
    unittest.main()
