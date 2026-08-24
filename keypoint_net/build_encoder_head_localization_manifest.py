"""Build the validation-blind manifest for the encoder/head localization audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from certified_witness_capability import EXPECTED_WITNESS_IDS, file_record, require
from encoder_head_localization import REPRESENTATION_NAMES, pixel_to_normalized


SCHEMA_VERSION = "augmented_encoder_head_localization_manifest.v1"
SEMANTIC_LOCK_SHA256 = "56fac1053fe264d1c24cb66093efc9272d8881562796951e0598516110900a9b"
SOURCE_TRAINING_HEAD = "403514387fe6dc13478f314e2cddbdcc2034a334"
EXPECTED_BINDINGS = {
    "checkpoint": "7e5d81241b1251254d46a420022dd1eda60f87c530196ea6063407c9ffb4e6cc",
    "training_receipt": "a0bc3137624b53c72e0e7acc194024a0bf7c4c586bd3adf0086ef66339608664",
    "raw_rgb_manifest": "5fdad8b65a438cdc52d7f1b4080d772d5312bb3c01076a66715226fe2754c596",
    "train_targets": "d7685fd3866d906fa47378a01608051faff5b819fc0dbfe026948284a5288d72",
    "prior_raw_arrays": "06ebff3fa0fc479c32fa6a31279163c26cf3581b7cb898c705edb58290dba6ef",
}
IMPLEMENTATION_FILES = (
    "keypoint_net/encoder_head_localization.py",
    "keypoint_net/build_encoder_head_localization_manifest.py",
    "keypoint_net/run_encoder_head_localization_raw.py",
    "keypoint_net/evaluate_encoder_head_localization.py",
    "keypoint_net/model.py",
    "keypoint_net/frozen_feature_decode.py",
)
ANCHOR_FRAME = 27
VALIDATION_FRAMES = tuple(range(24))


def _clean_head(repo_root: Path, expected_head: str) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == expected_head, "implementation HEAD differs")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "implementation repository is not clean")
    return head


def _bound(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == expected_hash, f"{label} SHA-256 differs")
    return record


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root differs: {path}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output.exists(), "manifest output already exists")
    implementation_head = _clean_head(args.repo_root, args.expected_repo_head)
    semantic_lock = _bound(args.semantic_lock, SEMANTIC_LOCK_SHA256, "semantic lock")
    checkpoint = _bound(args.checkpoint, EXPECTED_BINDINGS["checkpoint"], "checkpoint")
    training_receipt_record = _bound(
        args.training_receipt, EXPECTED_BINDINGS["training_receipt"], "training receipt"
    )
    rgb_manifest_record = _bound(
        args.raw_rgb_manifest, EXPECTED_BINDINGS["raw_rgb_manifest"], "RGB manifest"
    )
    train_targets_record = _bound(
        args.train_targets, EXPECTED_BINDINGS["train_targets"], "training targets"
    )
    prior_raw_record = _bound(
        args.prior_raw_arrays, EXPECTED_BINDINGS["prior_raw_arrays"], "prior raw arrays"
    )

    receipt = _load_json(args.training_receipt)
    require(
        receipt.get("schema_version") == "leakage_safe_witness_distillation_training_receipt.v1",
        "training receipt schema differs",
    )
    require(receipt.get("paired_arm") == "candidate", "checkpoint is not augmented candidate")
    require(receipt.get("repository_head") == SOURCE_TRAINING_HEAD, "training source HEAD differs")
    require(receipt.get("selected_model", {}).get("sha256") == checkpoint["sha256"], "selected model differs")
    require(
        Path(str(receipt.get("selected_model", {}).get("absolute_path", ""))).resolve()
        == args.checkpoint.resolve(),
        "selected model path differs",
    )
    result_path = Path(str(receipt["result"]["absolute_path"]))
    result_record = file_record(result_path)
    require(result_record == receipt["result"], "training result binding differs")
    training_result = _load_json(result_path)
    expected_model_state = str(training_result["selected_model_state_sha256"])

    with np.load(args.train_targets) as loaded:
        frames = np.asarray(loaded["frame_index"], dtype=np.int64)
        witness_ids = np.asarray(loaded["witness_id"], dtype=np.int64)
        targets_px = np.asarray(loaded["target_coordinate_px"], dtype=np.float64)
        physical_valid = np.asarray(loaded["physical_valid"], dtype=bool)
        target_on_object = np.asarray(loaded["target_on_object"], dtype=bool)
    require(np.array_equal(frames, np.arange(27, 177)), "training target frames differ")
    require(tuple(witness_ids.tolist()) == EXPECTED_WITNESS_IDS, "witness identity order differs")
    require(targets_px.shape == (150, 10, 2), "training target shape differs")
    require(bool(physical_valid.all() and target_on_object.all()), "training target validity differs")
    anchor_index = int(np.flatnonzero(frames == ANCHOR_FRAME)[0])
    anchor_px = targets_px[anchor_index]
    anchor_normalized = pixel_to_normalized(anchor_px)

    rgb_manifest = _load_json(args.raw_rgb_manifest)
    require(
        rgb_manifest.get("schema_version") == "leakage_safe_distillation_raw_rgb_manifest.v1",
        "RGB manifest schema differs",
    )
    require(rgb_manifest.get("contains_target_coordinates") is False, "RGB manifest contains targets")
    records = {int(row["frame_index"]): row for row in rgb_manifest["frames"]}
    selected_frames = (ANCHOR_FRAME,) + VALIDATION_FRAMES
    require(all(frame in records for frame in selected_frames), "RGB manifest omits selected frame")
    selected_records: list[dict[str, Any]] = []
    for frame in selected_frames:
        record = records[frame]
        require("mask_relpath" not in record and "mask_sha256" not in record, "RGB record exposes mask")
        image_path = args.object_root / str(record["image_relpath"])
        observed = file_record(image_path)
        require(observed["sha256"] == record["image_sha256"], f"RGB frame {frame} differs")
        selected_records.append(
            {
                "frame_index": frame,
                "image_relpath": str(record["image_relpath"]),
                "image_sha256": str(record["image_sha256"]),
            }
        )

    with np.load(args.prior_raw_arrays) as loaded:
        prior_frames = np.asarray(loaded["frame_index"], dtype=np.int64)
        require("native_heatmap_logits" in loaded.files, "prior raw arrays omit logits")
        prior_logits_shape = tuple(np.asarray(loaded["native_heatmap_logits"]).shape)
    require(np.array_equal(prior_frames[:24], np.arange(24)), "prior raw validation frames differ")
    require(ANCHOR_FRAME in prior_frames, "prior raw arrays omit anchor")
    require(prior_logits_shape == (174, 10, 64, 64), "prior raw logit shape differs")

    sources = {relative: file_record(args.repo_root / relative) for relative in IMPLEMENTATION_FILES}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "validation_blind_augmented_encoder_head_localization_manifest",
        "command_argv": list(sys.argv),
        "implementation_head": implementation_head,
        "source_training_head": SOURCE_TRAINING_HEAD,
        "implementation_sources": sources,
        "semantic_lock": semantic_lock,
        "checkpoint": checkpoint,
        "expected_model_state_sha256": expected_model_state,
        "training_receipt": training_receipt_record,
        "training_result": result_record,
        "raw_rgb_manifest": rgb_manifest_record,
        "train_targets_source": train_targets_record,
        "prior_raw_detector_arrays": prior_raw_record,
        "object_root": str(args.object_root.resolve(strict=True)),
        "anchor": {
            "frame_index": ANCHOR_FRAME,
            "witness_id": list(EXPECTED_WITNESS_IDS),
            "target_coordinate_px": anchor_px.tolist(),
            "target_coordinate_normalized": anchor_normalized.tolist(),
            "source_is_training_only": True,
        },
        "evaluation": {
            "frame_indices": list(VALIDATION_FRAMES),
            "frame_records": selected_records[1:],
            "validation_target_received_or_opened": False,
        },
        "anchor_frame_record": selected_records[0],
        "representations": [
            {
                "name": REPRESENTATION_NAMES[0],
                "module": "extractor.encoder[8]",
                "shape_without_batch": [128, 64, 64],
                "score": "cosine to bilinear frame-27 certified-site descriptor",
            },
            {
                "name": REPRESENTATION_NAMES[1],
                "module": "extractor.encoder[11]",
                "shape_without_batch": [128, 64, 64],
                "score": "cosine to bilinear frame-27 certified-site descriptor",
            },
            {
                "name": REPRESENTATION_NAMES[2],
                "module": "extractor.heatmap_head",
                "shape_without_batch": [10, 64, 64],
                "score": "native corresponding-channel logit",
            },
        ],
        "information_boundary": {
            "opened": ["checkpoint", "bound RGB corpus", "frame-27 training targets", "prior truth-free arrays"],
            "forbidden_and_not_opened": [
                "validation targets",
                "masks",
                "prior evaluation classifications",
                "operator prediction",
                "tracker output",
            ],
        },
        "training_or_weight_update_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--raw-rgb-manifest", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--prior-raw-arrays", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps({"output": str(args.output.resolve()), "schema_version": manifest["schema_version"]}, sort_keys=True))


if __name__ == "__main__":
    main()
