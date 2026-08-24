"""Run the validation-blind frozen encoder/head representation extraction."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, __version__ as pillow_version

from certified_witness_capability import file_record, model_state_sha256, require
from encoder_head_localization import EXPECTED_WITNESSES, REPRESENTATION_NAMES
from frozen_feature_decode import cosine_correlation_maps, sample_anchor_descriptors
from model import KeypointExtractor


SCHEMA_VERSION = "augmented_encoder_head_localization_raw_receipt.v1"
EXPECTED_MANIFEST_SCHEMA = "augmented_encoder_head_localization_manifest.v1"
MAX_PROJECTED_SECONDS = 3600.0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root differs: {path}")
    return value


def _verify_record(record: dict[str, Any], label: str) -> Path:
    require(isinstance(record, dict), f"{label} record missing")
    path = Path(str(record.get("absolute_path", "")))
    observed = file_record(path)
    require(observed == record, f"{label} binding differs")
    return path


def _verify_clean_head(repo_root: Path, expected_head: str) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == expected_head, "repository HEAD differs")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "repository is not clean")
    return head


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(array.shape == (512, 512, 3), f"RGB shape differs: {path}")
    return array


def _preprocess(paths: Sequence[Path]) -> torch.Tensor:
    array = np.stack([_load_rgb(path) for path in paths])
    batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def _extract_stages(
    model: KeypointExtractor, paths: Sequence[Path]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    batch = _preprocess(paths)
    with torch.inference_mode():
        penultimate = model.encoder[:9](batch)
        final = model.encoder[9:](penultimate)
        logits = model.heatmap_head(final)
        standard_points, standard_logits, standard_final = model(
            batch, return_descriptor_features=True
        )
    require(tuple(penultimate.shape[1:]) == (128, 64, 64), "penultimate shape differs")
    require(tuple(final.shape[1:]) == (128, 64, 64), "final feature shape differs")
    require(tuple(logits.shape[1:]) == (10, 64, 64), "logit shape differs")
    require(bool(torch.isfinite(penultimate).all()), "penultimate features non-finite")
    require(bool(torch.isfinite(final).all()), "final features non-finite")
    require(bool(torch.isfinite(logits).all()), "logits non-finite")
    final_difference = float(torch.max(torch.abs(final - standard_final)).item())
    logit_difference = float(torch.max(torch.abs(logits - standard_logits)).item())
    require(final_difference == 0.0, "manual final feature differs from ordinary extractor")
    require(logit_difference == 0.0, "manual logits differ from ordinary extractor")
    return (
        penultimate,
        final,
        logits,
        standard_points.reshape(len(paths), EXPECTED_WITNESSES, 2),
        {
            "manual_vs_standard_final_max_abs": final_difference,
            "manual_vs_standard_logits_max_abs": logit_difference,
        },
    )


def _image_path(root: Path, record: dict[str, Any]) -> Path:
    path = (root / str(record["image_relpath"])).resolve(strict=True)
    observed = file_record(path)
    require(observed["sha256"] == record["image_sha256"], f"RGB frame {record['frame_index']} differs")
    return path


def _load_model(manifest: dict[str, Any]) -> tuple[KeypointExtractor, dict[str, Any]]:
    checkpoint_path = _verify_record(manifest["checkpoint"], "checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        payload.get("schema_version") == "leakage_safe_witness_distillation_checkpoint.v1",
        "checkpoint schema differs",
    )
    require(payload.get("config", {}).get("paired_arm") == "candidate", "checkpoint arm differs")
    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    )
    model.load_state_dict(payload["extractor_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_hash = model_state_sha256(model)
    require(state_hash == manifest["expected_model_state_sha256"], "model state hash differs")
    return model, {
        "checkpoint_schema_version": payload["schema_version"],
        "checkpoint_config": payload["config"],
        "model_state_sha256": state_hash,
    }


def _score_order(
    model: KeypointExtractor,
    image_paths: list[Path],
    order: list[int],
    penultimate_anchor: torch.Tensor,
    final_anchor: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    score_maps = np.empty((3, len(image_paths), 10, 64, 64), dtype=np.float32)
    logits = np.empty((len(image_paths), 10, 64, 64), dtype=np.float32)
    points = np.empty((len(image_paths), 10, 2), dtype=np.float32)
    maximum_final_difference = 0.0
    maximum_logit_difference = 0.0
    for start in range(0, len(order), batch_size):
        selected = order[start : start + batch_size]
        penultimate, final, batch_logits, batch_points, replay = _extract_stages(
            model, [image_paths[index] for index in selected]
        )
        with torch.inference_mode():
            penultimate_score = cosine_correlation_maps(
                penultimate_anchor, penultimate
            )[0]
            final_score = cosine_correlation_maps(final_anchor, final)[0]
        for local, target_index in enumerate(selected):
            score_maps[0, target_index] = penultimate_score[local].cpu().numpy()
            score_maps[1, target_index] = final_score[local].cpu().numpy()
            score_maps[2, target_index] = batch_logits[local].cpu().numpy()
            logits[target_index] = batch_logits[local].cpu().numpy()
            points[target_index] = batch_points[local].cpu().numpy()
        maximum_final_difference = max(
            maximum_final_difference, replay["manual_vs_standard_final_max_abs"]
        )
        maximum_logit_difference = max(
            maximum_logit_difference, replay["manual_vs_standard_logits_max_abs"]
        )
    return score_maps, logits, points, {
        "manual_vs_standard_final_max_abs": maximum_final_difference,
        "manual_vs_standard_logits_max_abs": maximum_logit_difference,
    }


def _environment(torch_threads: int) -> dict[str, Any]:
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": mps_available,
        "execution_device": "cpu",
        "torch_num_threads": torch_threads,
        "laptop_gpu_used": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "raw output directory already exists")
    require(args.batch_size > 0 and args.torch_threads > 0, "runtime settings invalid")
    require("SEALED_VALIDATION_TRUTH" not in " ".join(sys.argv), "validation truth entered raw command")
    manifest = _load_json(args.manifest)
    require(manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "manifest schema differs")
    manifest_record = file_record(args.manifest)
    repository_head = _verify_clean_head(args.repo_root, args.expected_repo_head)
    require(manifest.get("implementation_head") == repository_head, "manifest HEAD differs")
    for relative, record in manifest["implementation_sources"].items():
        require(file_record(args.repo_root / relative) == record, f"implementation differs: {relative}")
    _verify_record(manifest["semantic_lock"], "semantic lock")
    _verify_record(manifest["training_receipt"], "training receipt")
    _verify_record(manifest["training_result"], "training result")
    prior_path = _verify_record(manifest["prior_raw_detector_arrays"], "prior raw arrays")

    torch.set_num_threads(args.torch_threads)
    model, model_binding = _load_model(manifest)
    state_before = model_state_sha256(model)
    root = Path(str(manifest["object_root"])).resolve(strict=True)
    anchor_path = _image_path(root, manifest["anchor_frame_record"])
    frame_records = manifest["evaluation"]["frame_records"]
    require([int(row["frame_index"]) for row in frame_records] == list(range(24)), "evaluation frames differ")
    image_paths = [_image_path(root, record) for record in frame_records]

    anchor_penultimate, anchor_final, anchor_logits, anchor_points, anchor_replay = _extract_stages(
        model, [anchor_path]
    )
    anchor_coordinate = torch.tensor(
        manifest["anchor"]["target_coordinate_normalized"], dtype=torch.float32
    ).unsqueeze(0)
    penultimate_anchor = sample_anchor_descriptors(anchor_penultimate, anchor_coordinate)
    final_anchor = sample_anchor_descriptors(anchor_final, anchor_coordinate)
    require(tuple(penultimate_anchor.shape) == (1, 10, 128), "penultimate anchor shape differs")
    require(tuple(final_anchor.shape) == (1, 10, 128), "final anchor shape differs")

    smoke_count = min(args.batch_size, 8, len(image_paths))
    smoke_start = time.perf_counter()
    _score_order(
        model,
        image_paths[:smoke_count],
        list(range(smoke_count)),
        penultimate_anchor,
        final_anchor,
        batch_size=smoke_count,
    )
    smoke_seconds = time.perf_counter() - smoke_start
    projected_seconds = smoke_seconds / smoke_count * len(image_paths) * 2.0 + smoke_seconds
    require(projected_seconds <= MAX_PROJECTED_SECONDS, "CPU timing projection exceeds one hour")

    full_start = time.perf_counter()
    forward = _score_order(
        model,
        image_paths,
        list(range(len(image_paths))),
        penultimate_anchor,
        final_anchor,
        batch_size=args.batch_size,
    )
    reverse = _score_order(
        model,
        image_paths,
        list(reversed(range(len(image_paths)))),
        penultimate_anchor,
        final_anchor,
        batch_size=args.batch_size,
    )
    runtime_seconds = time.perf_counter() - full_start
    names = ("score_maps", "native_heatmap_logits", "global_soft_prediction_normalized")
    order_differences: dict[str, float] = {}
    for name, forward_value, reverse_value in zip(names, forward[:3], reverse[:3], strict=True):
        difference = float(np.max(np.abs(forward_value - reverse_value)))
        order_differences[name] = difference
        require(difference == 0.0, f"frame-order reversal changed {name}")

    with np.load(prior_path) as loaded:
        prior_frames = np.asarray(loaded["frame_index"], dtype=np.int64)
        prior_logits = np.asarray(loaded["native_heatmap_logits"], dtype=np.float32)
        prior_points = np.asarray(loaded["global_soft_prediction_normalized"], dtype=np.float32)
    lookup = {int(frame): index for index, frame in enumerate(prior_frames.tolist())}
    validation_indices = np.asarray([lookup[frame] for frame in range(24)], dtype=np.int64)
    anchor_index = lookup[int(manifest["anchor"]["frame_index"])]
    validation_logit_difference = float(
        np.max(np.abs(forward[1] - prior_logits[validation_indices]))
    )
    validation_point_difference = float(
        np.max(np.abs(forward[2] - prior_points[validation_indices]))
    )
    anchor_logit_difference = float(
        np.max(np.abs(anchor_logits[0].cpu().numpy() - prior_logits[anchor_index]))
    )
    anchor_point_difference = float(
        np.max(np.abs(anchor_points[0].cpu().numpy() - prior_points[anchor_index]))
    )
    require(validation_logit_difference <= 1e-6, "validation logits do not replay prior frozen run")
    require(validation_point_difference <= 1e-6, "validation points do not replay prior frozen run")
    require(anchor_logit_difference <= 1e-6, "anchor logits do not replay prior frozen run")
    require(anchor_point_difference <= 1e-6, "anchor points do not replay prior frozen run")
    state_after = model_state_sha256(model)
    require(state_before == state_after, "model state changed during inference")

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_ENCODER_HEAD_SCORE_MAPS.npz"
    np.savez_compressed(
        arrays_path,
        representation_name=np.asarray(REPRESENTATION_NAMES),
        frame_index=np.arange(24, dtype=np.int64),
        witness_id=np.asarray(manifest["anchor"]["witness_id"], dtype=np.int64),
        score_maps=forward[0],
        native_heatmap_logits=forward[1],
        global_soft_prediction_normalized=forward[2],
        anchor_frame_index=np.asarray(int(manifest["anchor"]["frame_index"]), dtype=np.int64),
        anchor_target_coordinate_px=np.asarray(manifest["anchor"]["target_coordinate_px"], dtype=np.float64),
        penultimate_anchor_descriptor=penultimate_anchor[0].cpu().numpy(),
        final_anchor_descriptor=final_anchor[0].cpu().numpy(),
    )
    pip_path = args.output_dir / "PIP_FREEZE.txt"
    pip_path.write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        encoding="utf-8",
    )
    receipt_path = args.output_dir / "RAW_ENCODER_HEAD_RECEIPT.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "validation_blind_frozen_augmented_encoder_head_scores",
        "command_argv": list(sys.argv),
        "repository_head": repository_head,
        "manifest": manifest_record,
        "semantic_lock": manifest["semantic_lock"],
        "checkpoint": manifest["checkpoint"],
        "model": model_binding,
        "raw_arrays": file_record(arrays_path),
        "pip_freeze": file_record(pip_path),
        "representation_names": list(REPRESENTATION_NAMES),
        "runtime": {
            "smoke_frame_count": smoke_count,
            "smoke_seconds": smoke_seconds,
            "projected_full_seconds": projected_seconds,
            "maximum_authorized_seconds": MAX_PROJECTED_SECONDS,
            "full_forward_reverse_seconds": runtime_seconds,
            "batch_size": args.batch_size,
        },
        "environment": _environment(args.torch_threads),
        "controls": {
            "anchor_manual_replay": anchor_replay,
            "forward_manual_replay": forward[3],
            "reverse_manual_replay": reverse[3],
            "frame_order_reversal_maximum_absolute_difference": order_differences,
            "frame_order_reversal_exact": True,
            "prior_validation_logit_max_abs": validation_logit_difference,
            "prior_validation_point_max_abs": validation_point_difference,
            "prior_anchor_logit_max_abs": anchor_logit_difference,
            "prior_anchor_point_max_abs": anchor_point_difference,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": True,
            "validation_target_received_or_opened": False,
            "mask_received_or_opened": False,
            "prior_evaluation_classification_opened": False,
            "operator_or_tracker_input_opened": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "training_or_weight_update_performed": False,
            "laptop_gpu_used": False,
        },
        "privileged_evaluation_authorized": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
