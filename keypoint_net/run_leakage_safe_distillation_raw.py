"""Run truth-free RGB inference from the frozen distilled detector."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    CapabilityContractError,
    file_record,
    model_state_sha256,
    require,
)
from leakage_safe_distillation_contract import (
    SEMANTIC_LOCK_SHA256,
    TRAIN_FRAMES,
    VALIDATION_FRAMES,
    load_json,
    predict_model_readouts,
    require_local_cpu_only,
    verify_clean_repository,
    verify_exact_file,
    verify_record,
    verify_semantic_lock,
    write_json,
)
from model import KeypointExtractor
from run_certified_witness_capability import BoundCapabilityDataset


def _load_truth_free_dataset(
    manifest_path: Path,
    object_root: Path,
) -> tuple[BoundCapabilityDataset, np.ndarray, dict[str, object]]:
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema_version")
        == "leakage_safe_distillation_raw_rgb_manifest.v1",
        "raw RGB manifest schema differs",
    )
    require(manifest.get("contains_target_coordinates") is False, "raw manifest contains targets")
    require(manifest.get("contains_mask_paths_or_hashes") is False, "raw manifest contains masks")
    frames = np.asarray(manifest.get("frame_indices"), dtype=np.int64)
    expected = np.concatenate((VALIDATION_FRAMES, TRAIN_FRAMES))
    require(np.array_equal(frames, expected), "raw RGB frame order differs")
    records = manifest.get("frames")
    require(isinstance(records, list) and len(records) == len(frames), "raw RGB records differ")
    images = np.empty((len(frames), 512, 512, 3), dtype=np.uint8)
    verified = 0
    for local_index, (frame, record) in enumerate(zip(frames.tolist(), records, strict=True)):
        require(int(record["frame_index"]) == frame, "raw RGB record order differs")
        require("mask_relpath" not in record and "mask_sha256" not in record, "raw RGB record exposes mask")
        image_path = object_root / str(record["image_relpath"])
        verify_exact_file(image_path, str(record["image_sha256"]), f"raw RGB frame {frame}")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        require(image.shape == (512, 512, 3), "raw RGB shape differs")
        images[local_index] = image
        verified += 1
    dummy_masks = np.zeros((len(frames), 512, 512), dtype=bool)
    dummy_targets = np.zeros((len(frames), EXPECTED_WITNESSES, 2), dtype=np.float32)
    dataset = BoundCapabilityDataset(
        images,
        dummy_masks,
        dummy_targets,
        np.arange(len(frames), dtype=np.int64),
    )
    return dataset, frames, {
        "rgb_hashes_verified": verified,
        "target_artifact_opened": False,
        "mask_path_hash_or_pixels_opened": False,
        "frame_indices": frames.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    require(not args.output_dir.exists(), "output directory already exists")
    require("SEALED_VALIDATION_TRUTH" not in " ".join(sys.argv), "validation truth entered raw command")
    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)
    verify_semantic_lock(args.semantic_lock)
    environment = require_local_cpu_only()

    raw_input = load_json(args.raw_input_receipt)
    require(
        raw_input.get("schema_version")
        == "leakage_safe_distillation_raw_input_receipt.v1",
        "raw input receipt schema differs",
    )
    require(raw_input.get("semantic_lock_sha256") == SEMANTIC_LOCK_SHA256, "raw input lock differs")
    require(raw_input.get("repository_head") == repository_head, "raw input repository head differs")
    require(raw_input.get("contains_target_or_mask_artifact") is False, "raw input receipt exposes truth")
    raw_manifest = verify_record(raw_input["raw_rgb_manifest"], "raw RGB manifest")

    training_receipt = load_json(args.training_receipt)
    require(
        training_receipt.get("schema_version")
        == "leakage_safe_witness_distillation_training_receipt.v1",
        "training receipt schema differs",
    )
    require(training_receipt.get("run_kind") == "full", "raw inference requires full training")
    require(
        training_receipt.get("decision_branch")
        == "freeze_checkpoint_and_run_truth_free_inference",
        "training receipt does not authorize raw inference",
    )
    training_result_path = verify_record(training_receipt["result"], "training result")
    training_result = load_json(training_result_path)
    require(training_result.get("validation_truth_received_or_opened") is False, "training opened validation truth")
    model_path = verify_record(training_receipt["selected_model"], "selected detector")
    expected_model_state_hash = str(training_result["selected_model_state_sha256"])

    dataset, original_frames, input_controls = _load_truth_free_dataset(
        raw_manifest, args.object_root
    )
    torch.manual_seed(42)
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    )
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    require(
        payload.get("schema_version")
        == "leakage_safe_witness_distillation_checkpoint.v1",
        "detector checkpoint schema differs",
    )
    model.load_state_dict(payload["extractor_state_dict"], strict=True)
    actual_model_state_hash = model_state_sha256(model)
    require(actual_model_state_hash == expected_model_state_hash, "detector state hash differs")

    start = time.perf_counter()
    arrays, no_global_report, no_local_report = predict_model_readouts(
        model, dataset, torch.device("cpu"), args.batch_size
    )
    require(no_global_report is None and no_local_report is None, "truth-free inference produced report")
    replay, replay_global, replay_local = predict_model_readouts(
        model, dataset, torch.device("cpu"), args.batch_size
    )
    require(replay_global is None and replay_local is None, "truth-free replay produced report")
    for name in arrays:
        require(np.array_equal(arrays[name], replay[name]), f"raw {name} replay differs")
    runtime_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True)
    arrays["frame_index"] = original_frames
    arrays_path = args.output_dir / "RAW_DISTILLED_DETECTOR_PREDICTIONS.npz"
    np.savez_compressed(arrays_path, **arrays)
    pip_freeze_path = args.output_dir / "PIP_FREEZE.txt"
    pip_freeze_path.write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    receipt_path = args.output_dir / "RAW_DISTILLED_DETECTOR_RECEIPT.json"
    receipt = {
        "schema_version": "leakage_safe_distillation_raw_receipt.v1",
        "artifact_type": "truth_free_frozen_detector_predictions",
        "command_argv": list(sys.argv),
        "repository_head": repository_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "contract": file_record(
                args.repo_root
                / "keypoint_net"
                / "leakage_safe_distillation_contract.py"
            ),
            "local_readout": file_record(
                args.repo_root
                / "keypoint_net"
                / "certified_witness_local_readout.py"
            ),
            "model": file_record(args.repo_root / "keypoint_net" / "model.py"),
        },
        "semantic_lock": file_record(args.semantic_lock),
        "raw_input_receipt": file_record(args.raw_input_receipt),
        "training_receipt": file_record(args.training_receipt),
        "selected_detector": file_record(model_path),
        "selected_model_state_sha256": actual_model_state_hash,
        "raw_arrays": file_record(arrays_path),
        "pip_freeze": file_record(pip_freeze_path),
        "runtime_seconds": runtime_seconds,
        "environment": environment,
        "controls": {
            **input_controls,
            "all_coordinates_finite": bool(
                np.isfinite(arrays["global_soft_prediction_px"]).all()
                and np.isfinite(arrays["local_3x3_prediction_px"]).all()
            ),
            "full_prediction_replay_exact": True,
            "target_artifact_received_or_opened": False,
            "mask_artifact_received_or_opened": False,
            "operator_prediction_received_or_opened": False,
            "previous_frame_or_tracker_input_used": False,
            "training_or_weight_update_performed": False,
            "laptop_gpu_used": False,
        },
        "privileged_evaluation_authorized": True,
    }
    require(receipt["controls"]["all_coordinates_finite"] is True, "raw coordinates are non-finite")
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--raw-input-receipt", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"DISTILLATION RAW CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
