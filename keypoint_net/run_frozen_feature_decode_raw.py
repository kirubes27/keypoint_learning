"""Run geometry-blind full-orbit feature-anchored decoding for one frozen role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from .frozen_feature_decode import (
        cosine_correlation_maps,
        endpoint_cells_to_coordinates,
        sample_anchor_descriptors,
        sample_target_similarities,
        stable_spatial_top_two,
    )
    from .run_frozen_wobble_forensics import _construct_frozen_model, _same_fd_checkpoint_load, _state_sha256
except ImportError:
    from frozen_feature_decode import (  # type: ignore
        cosine_correlation_maps,
        endpoint_cells_to_coordinates,
        sample_anchor_descriptors,
        sample_target_similarities,
        stable_spatial_top_two,
    )
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )


SCHEMA_VERSION = "frozen_feature_decode_raw_receipt.v1"
EXPECTED_MANIFEST_SCHEMA = "frozen_feature_decode_manifest.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
FEATURE_SIZE = 64
FEATURE_CHANNELS = 128
BASIS_NAMES = ("hard_peak", "soft_coordinate")


class RawFeatureDecodeError(ValueError):
    """Raised when a geometry-blind frozen decode cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawFeatureDecodeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {"absolute_path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "manifest is not a JSON object")
    _require(manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA, "manifest schema differs")
    _require(manifest.get("artifact_type") == "geometry_blind_frozen_feature_decode_manifest", "manifest type differs")
    implementation = manifest.get("implementation", {}).get("implementation_sources")
    _require(isinstance(implementation, Mapping), "implementation binding is missing")
    for source_record in implementation.values():
        _require(_file_record(str(source_record["absolute_path"])) == dict(source_record), "implementation source differs")
    return manifest, record


def _image_paths(manifest: Mapping[str, Any]) -> list[Path]:
    corpus = manifest.get("corpus")
    _require(isinstance(corpus, Mapping), "corpus binding is missing")
    root = Path(str(corpus.get("rgb_object_root"))).resolve(strict=True)
    frames = corpus.get("frames")
    _require(isinstance(frames, list) and len(frames) == EXPECTED_FRAMES, "RGB frame count differs")
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        _require(int(frame.get("frame_index", -1)) == index, "RGB frame index differs")
        path = (root / str(frame["image_relpath"])).resolve(strict=True)
        _require(_sha256(path) == frame.get("image_sha256"), f"RGB SHA-256 differs at frame {index}")
        paths.append(path)
    return paths


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    _require(rgb.shape == (512, 512, 3), f"RGB frame shape differs: {path}")
    return rgb


def _preprocess(paths: Sequence[Path]) -> torch.Tensor:
    array = np.stack([_load_rgb(path) for path in paths])
    batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    means = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
    return batch.sub_(means).div_(stds)


def _extract(model: torch.nn.Module, paths: Sequence[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        flattened, logits, encoder = model.extractor(_preprocess(paths), return_descriptor_features=True)
    points = flattened.reshape(len(paths), EXPECTED_CHANNELS, 2)
    _require(tuple(logits.shape[1:]) == (EXPECTED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), "logit shape differs")
    _require(tuple(encoder.shape[1:]) == (FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), "feature shape differs")
    _require(bool(torch.isfinite(points).all()) and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(encoder).all()), "inference produced non-finite values")
    return points, logits, encoder


def _hard_coordinates(logits: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(logits.shape[0], logits.shape[1], -1)
    cell = torch.argmax(flat, dim=-1)
    y = torch.div(cell, FEATURE_SIZE, rounding_mode="floor")
    x = torch.remainder(cell, FEATURE_SIZE)
    coordinate = endpoint_cells_to_coordinates(x.cpu().numpy(), y.cpu().numpy(), size=FEATURE_SIZE)
    return torch.from_numpy(coordinate.astype(np.float32, copy=False))


def _decode_order(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    order: Sequence[int],
    *,
    anchor_frame: int,
    batch_size: int,
    second_peak_radius: float,
) -> dict[str, np.ndarray]:
    anchor_soft, anchor_logits, anchor_field = _extract(model, [image_paths[anchor_frame]])
    anchor_hard = _hard_coordinates(anchor_logits)
    anchor_coordinates = torch.stack((anchor_hard[0], anchor_soft[0]), dim=0)
    anchors = sample_anchor_descriptors(anchor_field, anchor_coordinates)

    shape_bfk = (len(BASIS_NAMES), EXPECTED_FRAMES, EXPECTED_CHANNELS)
    result = {
        "decoded_coordinate": np.empty(shape_bfk + (2,), dtype=np.float64),
        "top1_similarity": np.empty(shape_bfk, dtype=np.float64),
        "top2_similarity": np.empty(shape_bfk, dtype=np.float64),
        "top1_top2_margin": np.empty(shape_bfk, dtype=np.float64),
        "second_peak_coordinate": np.empty(shape_bfk + (2,), dtype=np.float64),
        "detector_similarity": np.empty(shape_bfk, dtype=np.float64),
        "detector_coordinate": np.empty(shape_bfk + (2,), dtype=np.float64),
        "anchor_coordinate": anchor_coordinates.cpu().numpy().astype(np.float64),
        "anchor_descriptor": anchors.cpu().numpy().astype(np.float64),
    }
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            selected = list(order[start : start + batch_size])
            soft, logits, field = _extract(model, [image_paths[index] for index in selected])
            hard = _hard_coordinates(logits)
            detector_coordinates = torch.stack((hard, soft), dim=0)
            correlation = cosine_correlation_maps(anchors, field).cpu().numpy()
            detector_similarity = sample_target_similarities(anchors, field, detector_coordinates).cpu().numpy()
            top = stable_spatial_top_two(correlation, exclusion_radius_cells=second_peak_radius)
            decoded = endpoint_cells_to_coordinates(top["top1_x_cell"], top["top1_y_cell"], size=FEATURE_SIZE)
            second = endpoint_cells_to_coordinates(top["top2_x_cell"], top["top2_y_cell"], size=FEATURE_SIZE)
            for local, frame in enumerate(selected):
                result["decoded_coordinate"][:, frame] = decoded[:, local]
                result["top1_similarity"][:, frame] = top["top1_similarity"][:, local]
                result["top2_similarity"][:, frame] = top["top2_similarity"][:, local]
                result["top1_top2_margin"][:, frame] = top["margin"][:, local]
                result["second_peak_coordinate"][:, frame] = second[:, local]
                result["detector_similarity"][:, frame] = detector_similarity[:, local]
                result["detector_coordinate"][:, frame] = detector_coordinates[:, local].cpu().numpy()
    return result


def run_raw_decode(
    manifest: Mapping[str, Any],
    role_key: str,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require(batch_size > 0, "batch size must be positive")
    matches = [role for role in manifest["roles"] if role.get("role_key") == role_key]
    _require(len(matches) == 1, "role key is absent or duplicated")
    role = matches[0]
    checkpoint = role["checkpoint"]
    payload, checkpoint_record = _same_fd_checkpoint_load(
        checkpoint["absolute_path"], expected_sha256=checkpoint["sha256"], expected_size=checkpoint["size_bytes"]
    )
    model, config = _construct_frozen_model(
        payload,
        cell_id=role["cell_id"],
        checkpoint_role=role["checkpoint_role"],
        expected_epoch=role["checkpoint_epoch"],
    )
    _require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()), "model is not frozen")
    image_paths = _image_paths(manifest)
    before = _state_sha256(model)
    forward = _decode_order(
        model,
        image_paths,
        list(range(EXPECTED_FRAMES)),
        anchor_frame=int(manifest["anchor_frame_index"]),
        batch_size=batch_size,
        second_peak_radius=float(manifest["separated_second_peak_radius_cells"]),
    )
    reverse = _decode_order(
        model,
        image_paths,
        list(reversed(range(EXPECTED_FRAMES))),
        anchor_frame=int(manifest["anchor_frame_index"]),
        batch_size=batch_size,
        second_peak_radius=float(manifest["separated_second_peak_radius_cells"]),
    )
    differences: dict[str, float] = {}
    for name, values in forward.items():
        maximum = float(np.max(np.abs(values - reverse[name])))
        differences[name] = maximum
        _require(maximum == 0.0, f"frame-order reversal changed {name} by {maximum}")
    after = _state_sha256(model)
    _require(before == after, "model state changed during frozen inference")
    arrays = {**forward, "frame_index": np.arange(EXPECTED_FRAMES, dtype=np.int64)}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "geometry_blind_frozen_feature_decode_raw_receipt",
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "basis_names": list(BASIS_NAMES),
        "anchor_frame_index": int(manifest["anchor_frame_index"]),
        "correlation_search": "stable full-image cosine argmax on endpoint-aligned 64x64 encoder field",
        "tie_rule": "lowest flat cell index",
        "separated_second_peak_radius_cells": float(manifest["separated_second_peak_radius_cells"]),
        "frame_order_reversal_maximum_absolute_difference": differences,
        "frame_order_reversal_exact": all(value == 0.0 for value in differences.values()),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "model_state_unchanged": before == after,
        "opened_rgb_paths": [str(path) for path in image_paths],
        "opened_rgb_count": len(image_paths),
        "forbidden_inputs_opened": [],
        "masks_or_geometry_opened": False,
        "optimizer_constructed": False,
        "gradients_enabled": False,
        "training_or_weight_update_performed": False,
        "device": "cpu",
        "batch_size": batch_size,
    }
    return arrays, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    torch.set_num_threads(args.torch_threads)
    manifest, manifest_record = _load_manifest(args.manifest.resolve(strict=True))
    arrays, receipt = run_raw_decode(manifest, args.role_key, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "raw_feature_decode_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    receipt["manifest"] = manifest_record
    receipt["raw_arrays"] = _file_record(arrays_path)
    receipt_path = args.output_dir / "raw_feature_decode_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path.resolve()), "raw_arrays_sha256": receipt["raw_arrays"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
