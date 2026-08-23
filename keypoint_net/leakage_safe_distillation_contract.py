"""Pure contracts shared by the leakage-safe witness-distillation stages."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, __version__ as pillow_version
from torch.utils.data import DataLoader

from certified_witness_capability import (
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    evaluate_predictions,
    file_record,
    normalized_to_pixel,
    require,
    sha256_file,
)
from certified_witness_local_readout import readout_arrays
from model import KeypointExtractor
from run_certified_witness_capability import BoundCapabilityDataset


SEMANTIC_LOCK_SHA256 = (
    "aabd05a0437f77cda143171d68c59c9578f0a09e8bfaea2e1321a4a0e6a2f0fe"
)
CAPABILITY_MANIFEST_SHA256 = (
    "1f94e0baf1c0a1b01e8897f0a5dc8419fccbd52c865ff5963253fcd098bd44dd"
)
FULL_TRACKS_SHA256 = (
    "b9decd7440da1e35f935f5d8d443e3eb9738b1584f8b72ebebb51b1d7bfa93b6"
)
TRAIN_PAIRS_SHA256 = (
    "f4317b96e05562c22c8e51b96a290c33b0fc3e1f6ba2b9b6771ffe9df9daa063"
)
VALIDATION_PAIRS_SHA256 = (
    "3e71c4f862a99d1882a8704140f8706612460d804a6ed7a833c8dee9f35514a4"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "bdda2eb08575c55b8a3569e706dd23e3dab4fbdee5db2f9a459ea0577cb49ade"
)
OBJECT_NAME = "engineers_hammer_vray"
TRAIN_FRAMES = np.arange(27, 177, dtype=np.int64)
VALIDATION_FRAMES = np.arange(0, 24, dtype=np.int64)
GUARD_FRAMES = np.asarray((24, 25, 26, 177, 178, 179), dtype=np.int64)
TRAIN_PAIR_SOURCE_FRAMES = np.arange(27, 174, dtype=np.int64)
TRAIN_PAIR_TARGET_FRAMES = TRAIN_PAIR_SOURCE_FRAMES + 3
VALIDATION_PAIR_SOURCE_FRAMES = np.arange(0, 21, dtype=np.int64)
VALIDATION_PAIR_TARGET_FRAMES = VALIDATION_PAIR_SOURCE_FRAMES + 3
HALF_CELL_DIAGONAL_PX = (511.0 / 63.0) / math.sqrt(2.0)
TWO_CELL_SPACING_PX = 2.0 * 511.0 / 63.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_exact_file(path: Path, expected_sha256: str, label: str) -> None:
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == expected_sha256, f"{label} SHA-256 differs")


def verify_semantic_lock(path: Path) -> None:
    verify_exact_file(path, SEMANTIC_LOCK_SHA256, "semantic lock")


def verify_clean_repository(repo_root: Path, expected_head: str | None = None) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected_head is not None:
        require(head == expected_head, "repository HEAD differs from command lock")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(status == "", "repository is not clean")
    return head


def verify_record(record: dict[str, Any], label: str) -> Path:
    require(isinstance(record, dict), f"{label} record missing")
    path = Path(str(record.get("absolute_path", "")))
    require(path.is_file(), f"{label} file missing")
    require(int(record.get("size_bytes", -1)) == path.stat().st_size, f"{label} size differs")
    require(str(record.get("sha256", "")) == sha256_file(path), f"{label} hash differs")
    return path


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_capability_pass": bool(report["strict_capability_pass"]),
        "violations": report["violations"],
        "material_error_px": report["material_error_px"],
        "within_half_cell_rate": report["within_half_cell_rate"],
        "on_object_rate": report["on_object_rate"],
        "identity_assignment_rate": report["identity_assignment_rate"],
        "minimum_predicted_pair_distance_px": report[
            "minimum_predicted_pair_distance_px"
        ],
        "minimum_predicted_to_physical_pair_ratio": report[
            "minimum_predicted_to_physical_pair_ratio"
        ],
        "per_witness": report["per_witness"],
    }


def runtime_environment() -> dict[str, Any]:
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
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def require_local_cpu_only() -> dict[str, Any]:
    environment = runtime_environment()
    require(environment["cuda_available"] is False, "CUDA is visible in CPU-only run")
    require(environment["mps_available"] is False, "MPS is visible in CPU-only run")
    return environment


def _record_by_frame(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = manifest.get("frames")
    require(isinstance(records, list), "frame manifest records missing")
    by_frame: dict[int, dict[str, Any]] = {}
    for record in records:
        frame = int(record["frame_index"])
        require(frame not in by_frame, "duplicate frame record")
        by_frame[frame] = record
    return by_frame


def load_bound_training_dataset(
    train_manifest_path: Path,
    train_targets_path: Path,
    object_root: Path,
) -> tuple[BoundCapabilityDataset, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only the prepared train frames, targets, and masks."""

    manifest = load_json(train_manifest_path)
    require(
        manifest.get("schema_version") == "leakage_safe_distillation_train_manifest.v1",
        "train manifest schema differs",
    )
    by_frame = _record_by_frame(manifest)
    require(np.array_equal(np.asarray(sorted(by_frame), dtype=np.int64), TRAIN_FRAMES), "train manifest frames differ")

    with np.load(train_targets_path) as arrays:
        frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
        witness_id = np.asarray(arrays["witness_id"], dtype=np.int64)
        target_px = np.asarray(arrays["target_coordinate_px"], dtype=np.float64)
        target_normalized = np.asarray(
            arrays["target_coordinate_normalized"], dtype=np.float32
        )
        physical_valid = np.asarray(arrays["physical_valid"], dtype=bool)
        target_on_object_recorded = np.asarray(arrays["target_on_object"], dtype=bool)
    require(np.array_equal(frame_index, TRAIN_FRAMES), "train-target frames differ")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "witness order differs")
    require(target_px.shape == (len(TRAIN_FRAMES), EXPECTED_WITNESSES, 2), "train target shape differs")
    require(bool(physical_valid.all()), "train target is physically invalid")
    require(bool(target_on_object_recorded.all()), "train target is recorded off object")

    images = np.empty((len(TRAIN_FRAMES), 512, 512, 3), dtype=np.uint8)
    masks = np.empty((len(TRAIN_FRAMES), 512, 512), dtype=bool)
    verified: list[dict[str, Any]] = []
    for local_index, frame in enumerate(TRAIN_FRAMES.tolist()):
        record = by_frame[frame]
        image_path = object_root / str(record["image_relpath"])
        mask_path = object_root / str(record["mask_relpath"])
        verify_exact_file(image_path, str(record["image_sha256"]), f"train RGB frame {frame}")
        verify_exact_file(mask_path, str(record["mask_sha256"]), f"train mask frame {frame}")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        require(image.shape == (512, 512, 3), "train RGB shape differs")
        require(mask.shape == (512, 512), "train mask shape differs")
        images[local_index] = image
        masks[local_index] = mask
        verified.append(
            {
                "frame_index": frame,
                "image_sha256": str(record["image_sha256"]),
                "mask_sha256": str(record["mask_sha256"]),
            }
        )

    rounded = np.rint(target_px).astype(np.int64)
    replay = masks[
        np.arange(len(TRAIN_FRAMES))[:, None], rounded[..., 1], rounded[..., 0]
    ]
    require(bool(replay.all()), "prepared train target is off loaded object mask")
    require(np.array_equal(replay, target_on_object_recorded), "train mask replay differs")
    # Local row indices keep existing evaluation helpers independent of original frame IDs.
    dataset = BoundCapabilityDataset(
        images,
        masks,
        target_normalized,
        np.arange(len(TRAIN_FRAMES), dtype=np.int64),
    )
    controls = {
        "only_training_frames_loaded": True,
        "training_frame_count": len(TRAIN_FRAMES),
        "original_frame_indices": TRAIN_FRAMES.tolist(),
        "rgb_and_mask_hashes_verified": len(verified),
        "all_training_targets_physical_and_on_object": True,
    }
    return dataset, target_px, masks, frame_index, controls


@torch.no_grad()
def predict_model_readouts(
    model: KeypointExtractor,
    dataset: BoundCapabilityDataset,
    device: torch.device,
    batch_size: int,
    *,
    target_px: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None, dict[str, Any] | None]:
    """Predict global and fixed-local coordinates, optionally evaluating truth."""

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    global_values: list[np.ndarray] = []
    logit_values: list[np.ndarray] = []
    frame_values: list[np.ndarray] = []
    for batch in loader:
        flat, logits = model(batch["image"].to(device))
        global_values.append(flat.view(-1, EXPECTED_WITNESSES, 2).cpu().numpy())
        logit_values.append(logits.cpu().numpy())
        frame_values.append(batch["frame"].numpy())
    frame_index = np.concatenate(frame_values)
    order = np.argsort(frame_index)
    require(
        np.array_equal(frame_index[order], np.arange(len(dataset))),
        "prediction frame order incomplete",
    )
    global_normalized = np.concatenate(global_values)[order]
    logits = np.concatenate(logit_values)[order]
    global_px = normalized_to_pixel(global_normalized)
    local = readout_arrays(logits, target_px)
    arrays = {
        "native_heatmap_logits": logits,
        "global_soft_prediction_normalized": global_normalized,
        "global_soft_prediction_px": global_px,
        "hard_cell_x": local["hard_cell_x"],
        "hard_cell_y": local["hard_cell_y"],
        "hard_prediction_px": local["hard_prediction_px"],
        "local_3x3_prediction_px": local["local_3x3_prediction_px"],
        "inside_window_probability_mass": local["inside_window_probability_mass"],
        "outside_window_probability_mass": local["outside_window_probability_mass"],
        "top1_probability": local["top1_probability"],
        "top2_probability": local["top2_probability"],
        "top1_top2_probability_margin": local["top1_top2_probability_margin"],
        "heatmap_entropy": local["heatmap_entropy"],
    }
    if target_px is None:
        require(masks is None, "masks supplied without targets")
        return arrays, None, None
    require(masks is not None, "targets supplied without masks")
    global_report, _ = evaluate_predictions(global_px, target_px, masks)
    local_report, _ = evaluate_predictions(
        arrays["local_3x3_prediction_px"], target_px, masks
    )
    return arrays, global_report, local_report


def operational_near_pass(report: dict[str, Any]) -> bool:
    violations = report["violations"]
    return bool(
        violations["wrong_identity_count"] == 0
        and violations["collapsed_pair_count"] == 0
        and violations["off_object_count"] == 0
        and float(report["material_error_px"]["median"])
        <= HALF_CELL_DIAGONAL_PX + 1e-12
        and float(report["material_error_px"]["maximum"])
        <= TWO_CELL_SPACING_PX + 1e-12
    )


def source_record(path: Path) -> dict[str, Any]:
    return file_record(path)


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
