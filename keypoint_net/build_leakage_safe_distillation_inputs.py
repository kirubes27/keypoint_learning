"""Build disjoint train, raw-RGB, and privileged evaluation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    file_record,
    require,
)
from leakage_safe_distillation_contract import (
    CAPABILITY_MANIFEST_SHA256,
    FULL_TRACKS_SHA256,
    GUARD_FRAMES,
    OBJECT_NAME,
    SEMANTIC_LOCK_SHA256,
    TRAIN_FRAMES,
    TRAIN_PAIRS_SHA256,
    TRAIN_PAIR_SOURCE_FRAMES,
    TRAIN_PAIR_TARGET_FRAMES,
    VALIDATION_FRAMES,
    VALIDATION_PAIRS_SHA256,
    VALIDATION_PAIR_SOURCE_FRAMES,
    VALIDATION_PAIR_TARGET_FRAMES,
    load_json,
    verify_clean_repository,
    verify_exact_file,
    verify_semantic_lock,
    write_json,
)


def _pair_rows(path: Path, split: str) -> list[dict[str, Any]]:
    value = load_json(path)
    rows = [
        row
        for row in value.get("pairs", [])
        if row.get("model_name") == OBJECT_NAME
    ]
    require(rows, f"no {OBJECT_NAME} pairs in {split} index")
    require(all(row.get("split") == split for row in rows), f"{split} row tag differs")
    require(all(int(row.get("stride", -1)) == 3 for row in rows), f"{split} stride differs")
    require(
        all(float(row.get("signed_generator", 0.0)) == 6.0 for row in rows),
        f"{split} generator differs",
    )
    return rows


def _assert_pairs(
    rows: list[dict[str, Any]],
    expected_source: np.ndarray,
    expected_target: np.ndarray,
    split: str,
) -> None:
    source = np.asarray([int(row["src_frame_index"]) for row in rows], dtype=np.int64)
    target = np.asarray([int(row["dst_frame_index"]) for row in rows], dtype=np.int64)
    require(np.array_equal(source, expected_source), f"{split} pair sources differ")
    require(np.array_equal(target, expected_target), f"{split} pair targets differ")
    require(len({str(row["pair_id"]) for row in rows}) == len(rows), f"{split} pair IDs repeat")


def _frame_records(capability_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = capability_manifest.get("dataset", {}).get("frames")
    require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "capability frame inventory differs")
    by_frame: dict[int, dict[str, Any]] = {}
    for row in frames:
        frame = int(row["frame_index"])
        require(frame not in by_frame, "capability frame repeats")
        by_frame[frame] = row
    require(set(by_frame) == set(range(EXPECTED_FRAMES)), "capability frame indices differ")
    return by_frame


def _target_payload(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frame_index": np.asarray(arrays["frame_index"], dtype=np.int64)[indices],
        "witness_id": np.asarray(arrays["witness_id"], dtype=np.int64),
        "target_coordinate_px": np.asarray(
            arrays["target_coordinate_px"], dtype=np.float64
        )[indices],
        "target_coordinate_normalized": np.asarray(
            arrays["target_coordinate_normalized"], dtype=np.float32
        )[indices],
        "physical_valid": np.asarray(arrays["physical_valid"], dtype=bool)[indices],
        "target_on_object": np.asarray(arrays["target_on_object"], dtype=bool)[indices],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists")
    verify_semantic_lock(args.semantic_lock)
    verify_exact_file(args.capability_manifest, CAPABILITY_MANIFEST_SHA256, "capability manifest")
    verify_exact_file(args.full_tracks, FULL_TRACKS_SHA256, "full tracks")
    verify_exact_file(args.train_pairs, TRAIN_PAIRS_SHA256, "train pairs")
    verify_exact_file(args.validation_pairs, VALIDATION_PAIRS_SHA256, "validation pairs")
    repository_head = verify_clean_repository(args.repo_root, args.expected_repo_head)

    train_rows = _pair_rows(args.train_pairs, "train")
    validation_rows = _pair_rows(args.validation_pairs, "validation")
    _assert_pairs(
        train_rows, TRAIN_PAIR_SOURCE_FRAMES, TRAIN_PAIR_TARGET_FRAMES, "train"
    )
    _assert_pairs(
        validation_rows,
        VALIDATION_PAIR_SOURCE_FRAMES,
        VALIDATION_PAIR_TARGET_FRAMES,
        "validation",
    )
    require(len(train_rows) == 147, "hammer train pair count differs")
    require(len(validation_rows) == 21, "hammer validation pair count differs")
    train_frame_union = np.unique(
        np.concatenate((TRAIN_PAIR_SOURCE_FRAMES, TRAIN_PAIR_TARGET_FRAMES))
    )
    validation_frame_union = np.unique(
        np.concatenate(
            (VALIDATION_PAIR_SOURCE_FRAMES, VALIDATION_PAIR_TARGET_FRAMES)
        )
    )
    require(np.array_equal(train_frame_union, TRAIN_FRAMES), "train frame union differs")
    require(
        np.array_equal(validation_frame_union, VALIDATION_FRAMES),
        "validation frame union differs",
    )
    require(
        len(set(TRAIN_FRAMES.tolist()) & set(VALIDATION_FRAMES.tolist())) == 0,
        "train and validation frames overlap",
    )
    require(
        len(set(GUARD_FRAMES.tolist()) & set(TRAIN_FRAMES.tolist())) == 0
        and len(set(GUARD_FRAMES.tolist()) & set(VALIDATION_FRAMES.tolist())) == 0,
        "guard frames enter a primary partition",
    )

    capability = load_json(args.capability_manifest)
    require(
        capability.get("schema_version")
        == "certified_witness_supervised_capability_manifest.v1",
        "capability manifest schema differs",
    )
    by_frame = _frame_records(capability)
    with np.load(args.full_tracks) as loaded:
        full = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(np.array_equal(np.asarray(full["frame_index"], dtype=np.int64), np.arange(EXPECTED_FRAMES)), "full-track frames differ")
    require(tuple(np.asarray(full["witness_id"], dtype=np.int64).tolist()) == EXPECTED_WITNESS_IDS, "full-track witness order differs")
    require(
        np.asarray(full["target_coordinate_px"]).shape
        == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2),
        "full-track coordinate shape differs",
    )

    args.output_dir.mkdir(parents=True)
    train_targets = args.output_dir / "DISTILLATION_TRAIN_TARGETS.npz"
    validation_truth = args.output_dir / "SEALED_VALIDATION_TRUTH.npz"
    np.savez_compressed(train_targets, **_target_payload(full, TRAIN_FRAMES))
    np.savez_compressed(validation_truth, **_target_payload(full, VALIDATION_FRAMES))

    train_manifest = args.output_dir / "DISTILLATION_TRAIN_FRAME_MANIFEST.json"
    write_json(
        train_manifest,
        {
            "schema_version": "leakage_safe_distillation_train_manifest.v1",
            "artifact_type": "training_only_rgb_mask_manifest",
            "object_name": OBJECT_NAME,
            "frame_indices": TRAIN_FRAMES.tolist(),
            "frames": [by_frame[int(frame)] for frame in TRAIN_FRAMES],
        },
    )
    raw_rgb_manifest = args.output_dir / "DISTILLATION_RAW_RGB_MANIFEST.json"
    raw_frames = np.concatenate((VALIDATION_FRAMES, TRAIN_FRAMES))
    write_json(
        raw_rgb_manifest,
        {
            "schema_version": "leakage_safe_distillation_raw_rgb_manifest.v1",
            "artifact_type": "truth_and_mask_free_rgb_manifest",
            "object_name": OBJECT_NAME,
            "frame_indices": raw_frames.tolist(),
            "validation_frame_indices": VALIDATION_FRAMES.tolist(),
            "training_frame_indices": TRAIN_FRAMES.tolist(),
            "frames": [
                {
                    "frame_index": int(frame),
                    "image_relpath": by_frame[int(frame)]["image_relpath"],
                    "image_sha256": by_frame[int(frame)]["image_sha256"],
                }
                for frame in raw_frames
            ],
            "contains_target_coordinates": False,
            "contains_mask_paths_or_hashes": False,
        },
    )
    evaluation_mask_manifest = (
        args.output_dir / "DISTILLATION_EVALUATION_MASK_MANIFEST.json"
    )
    write_json(
        evaluation_mask_manifest,
        {
            "schema_version": "leakage_safe_distillation_evaluation_mask_manifest.v1",
            "artifact_type": "privileged_posthash_mask_manifest",
            "object_name": OBJECT_NAME,
            "frame_indices": raw_frames.tolist(),
            "frames": [
                {
                    "frame_index": int(frame),
                    "mask_relpath": by_frame[int(frame)]["mask_relpath"],
                    "mask_sha256": by_frame[int(frame)]["mask_sha256"],
                }
                for frame in raw_frames
            ],
        },
    )

    train_receipt_path = args.output_dir / "TRAIN_INPUT_RECEIPT.json"
    train_receipt = {
        "schema_version": "leakage_safe_distillation_train_input_receipt.v1",
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "repository_head": repository_head,
        "train_targets": file_record(train_targets),
        "train_frame_manifest": file_record(train_manifest),
        "training_frame_indices": TRAIN_FRAMES.tolist(),
        "validation_truth_received_by_training_stage": False,
        "full_track_archive_received_by_training_stage": False,
    }
    write_json(train_receipt_path, train_receipt)

    raw_receipt_path = args.output_dir / "RAW_INPUT_RECEIPT.json"
    raw_receipt = {
        "schema_version": "leakage_safe_distillation_raw_input_receipt.v1",
        "semantic_lock_sha256": SEMANTIC_LOCK_SHA256,
        "repository_head": repository_head,
        "raw_rgb_manifest": file_record(raw_rgb_manifest),
        "frame_indices": raw_frames.tolist(),
        "contains_target_or_mask_artifact": False,
    }
    write_json(raw_receipt_path, raw_receipt)

    evaluation_receipt_path = args.output_dir / "EVALUATION_INPUT_RECEIPT.json"
    evaluation_receipt = {
        "schema_version": "leakage_safe_distillation_evaluation_input_receipt.v1",
        "semantic_lock": file_record(args.semantic_lock),
        "capability_manifest": file_record(args.capability_manifest),
        "full_track_source": file_record(args.full_tracks),
        "train_pairs": file_record(args.train_pairs),
        "validation_pairs": file_record(args.validation_pairs),
        "train_targets": file_record(train_targets),
        "validation_truth": file_record(validation_truth),
        "raw_rgb_manifest": file_record(raw_rgb_manifest),
        "evaluation_mask_manifest": file_record(evaluation_mask_manifest),
        "training_frame_indices": TRAIN_FRAMES.tolist(),
        "validation_frame_indices": VALIDATION_FRAMES.tolist(),
        "guard_frame_indices": GUARD_FRAMES.tolist(),
        "training_pair_count": len(train_rows),
        "validation_pair_count": len(validation_rows),
        "frame_partitions_disjoint": True,
    }
    write_json(evaluation_receipt_path, evaluation_receipt)

    result_path = args.output_dir / "PREPARATION_RESULT.json"
    result = {
        "schema_version": "leakage_safe_distillation_preparation_result.v1",
        "artifact_type": "privileged_disjoint_distillation_input_preparation",
        "repository_head": repository_head,
        "builder": file_record(Path(__file__)),
        "contract": file_record(
            args.repo_root
            / "keypoint_net"
            / "leakage_safe_distillation_contract.py"
        ),
        "semantic_lock": file_record(args.semantic_lock),
        "train_input_receipt": file_record(train_receipt_path),
        "raw_input_receipt": file_record(raw_receipt_path),
        "evaluation_input_receipt": file_record(evaluation_receipt_path),
        "full_truth_opened_only_by_privileged_preparation": True,
        "training_and_validation_frames_disjoint": True,
        "guards_excluded": True,
        "operator_train_pair_count": 147,
        "operator_validation_pair_count": 21,
    }
    write_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--full-tracks", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"DISTILLATION PREPARATION CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
