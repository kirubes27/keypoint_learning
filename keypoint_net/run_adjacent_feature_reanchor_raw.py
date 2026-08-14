"""Run geometry-blind same-frame and adjacent feature re-anchoring for one role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .adjacent_feature_reanchor import (
        cyclic_target_indices,
        paired_cosine_correlation_maps,
        sample_paired_descriptors,
        sample_paired_target_similarities,
    )
    from .frozen_feature_decode import endpoint_cells_to_coordinates, stable_spatial_top_two
    from .run_frozen_feature_decode_raw import _extract, _hard_coordinates, _image_paths
    from .run_frozen_wobble_forensics import _construct_frozen_model, _same_fd_checkpoint_load, _state_sha256
except ImportError:
    from adjacent_feature_reanchor import (  # type: ignore
        cyclic_target_indices,
        paired_cosine_correlation_maps,
        sample_paired_descriptors,
        sample_paired_target_similarities,
    )
    from frozen_feature_decode import endpoint_cells_to_coordinates, stable_spatial_top_two  # type: ignore
    from run_frozen_feature_decode_raw import _extract, _hard_coordinates, _image_paths  # type: ignore
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )


SCHEMA_VERSION = "adjacent_feature_reanchor_raw_receipt.v1"
EXPECTED_MANIFEST_SCHEMA = "adjacent_feature_reanchor_manifest.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
FEATURE_CHANNELS = 128
FEATURE_SIZE = 64


class RawAdjacentFeatureError(ValueError):
    """Raised when the geometry-blind adjacent feature pass is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawAdjacentFeatureError(message)


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
    _require(manifest.get("artifact_type") == "geometry_blind_adjacent_feature_reanchor_manifest", "manifest type differs")
    sources = manifest.get("implementation", {}).get("implementation_sources")
    _require(isinstance(sources, Mapping), "implementation binding is missing")
    for name, source_record in sources.items():
        _require(_file_record(str(source_record["absolute_path"])) == dict(source_record), f"implementation source differs: {name}")
    return manifest, record


def _extract_all(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    order: Sequence[int],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), dtype=np.float32)
    features = np.empty((EXPECTED_FRAMES, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        selected = list(order[start : start + batch_size])
        _, logits, encoder = _extract(model, [image_paths[index] for index in selected])
        hard = _hard_coordinates(logits)
        coordinates[selected] = hard.cpu().numpy()
        features[selected] = encoder.cpu().numpy()
    _require(np.isfinite(coordinates).all() and np.isfinite(features).all(), "extracted arrays are non-finite")
    return coordinates, features


def _top_arrays(correlation: torch.Tensor, *, radius: float) -> dict[str, np.ndarray]:
    top = stable_spatial_top_two(correlation.cpu().numpy(), exclusion_radius_cells=radius)
    return {
        "decoded_coordinate": endpoint_cells_to_coordinates(top["top1_x_cell"], top["top1_y_cell"], size=FEATURE_SIZE),
        "top1_similarity": top["top1_similarity"],
        "top2_similarity": top["top2_similarity"],
        "top1_top2_margin": top["margin"],
        "second_peak_coordinate": endpoint_cells_to_coordinates(
            top["top2_x_cell"], top["top2_y_cell"], size=FEATURE_SIZE
        ),
    }


def _decode_extracted(
    coordinates: np.ndarray,
    features: np.ndarray,
    *,
    batch_size: int,
    second_peak_radius: float,
) -> dict[str, np.ndarray]:
    target_index = cyclic_target_indices(EXPECTED_FRAMES)
    shape_fk = (EXPECTED_FRAMES, EXPECTED_CHANNELS)
    arrays: dict[str, np.ndarray] = {
        "source_detector_coordinate": coordinates.astype(np.float64),
        "target_detector_coordinate": coordinates[target_index].astype(np.float64),
        "source_descriptor_raw_norm": np.empty(shape_fk, dtype=np.float64),
        "source_descriptor": np.empty(shape_fk + (FEATURE_CHANNELS,), dtype=np.float64),
        "self_decoded_coordinate": np.empty(shape_fk + (2,), dtype=np.float64),
        "self_top1_similarity": np.empty(shape_fk, dtype=np.float64),
        "self_top2_similarity": np.empty(shape_fk, dtype=np.float64),
        "self_top1_top2_margin": np.empty(shape_fk, dtype=np.float64),
        "self_second_peak_coordinate": np.empty(shape_fk + (2,), dtype=np.float64),
        "adjacent_decoded_coordinate": np.empty(shape_fk + (2,), dtype=np.float64),
        "adjacent_top1_similarity": np.empty(shape_fk, dtype=np.float64),
        "adjacent_top2_similarity": np.empty(shape_fk, dtype=np.float64),
        "adjacent_top1_top2_margin": np.empty(shape_fk, dtype=np.float64),
        "adjacent_second_peak_coordinate": np.empty(shape_fk + (2,), dtype=np.float64),
        "adjacent_target_detector_similarity": np.empty(shape_fk, dtype=np.float64),
    }
    with torch.inference_mode():
        for start in range(0, EXPECTED_FRAMES, batch_size):
            selected = np.arange(start, min(start + batch_size, EXPECTED_FRAMES), dtype=np.int64)
            targets = target_index[selected]
            source_field = torch.from_numpy(features[selected])
            target_field = torch.from_numpy(features[targets])
            source_coordinates = torch.from_numpy(coordinates[selected])
            target_coordinates = torch.from_numpy(coordinates[targets])
            _, raw_norm, descriptors = sample_paired_descriptors(source_field, source_coordinates)
            self_top = _top_arrays(
                paired_cosine_correlation_maps(descriptors, source_field), radius=second_peak_radius
            )
            adjacent_top = _top_arrays(
                paired_cosine_correlation_maps(descriptors, target_field), radius=second_peak_radius
            )
            arrays["source_descriptor_raw_norm"][selected] = raw_norm.cpu().numpy()
            arrays["source_descriptor"][selected] = descriptors.cpu().numpy()
            for name, values in self_top.items():
                arrays[f"self_{name}"][selected] = values
            for name, values in adjacent_top.items():
                arrays[f"adjacent_{name}"][selected] = values
            arrays["adjacent_target_detector_similarity"][selected] = sample_paired_target_similarities(
                descriptors, target_field, target_coordinates
            ).cpu().numpy()
    return arrays


def run_raw(
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
    forward_coordinates, forward_features = _extract_all(
        model, image_paths, list(range(EXPECTED_FRAMES)), batch_size=batch_size
    )
    forward = _decode_extracted(
        forward_coordinates,
        forward_features,
        batch_size=batch_size,
        second_peak_radius=float(manifest["separated_second_peak_radius_cells"]),
    )
    del forward_coordinates, forward_features
    reverse_coordinates, reverse_features = _extract_all(
        model, image_paths, list(reversed(range(EXPECTED_FRAMES))), batch_size=batch_size
    )
    reverse = _decode_extracted(
        reverse_coordinates,
        reverse_features,
        batch_size=batch_size,
        second_peak_radius=float(manifest["separated_second_peak_radius_cells"]),
    )
    differences: dict[str, float] = {}
    for name, values in forward.items():
        maximum = float(np.max(np.abs(values - reverse[name])))
        differences[name] = maximum
        _require(maximum == 0.0, f"reversed extraction changed {name} by {maximum}")
    after = _state_sha256(model)
    _require(before == after, "model state changed during frozen inference")
    arrays = {
        **forward,
        "source_frame_index": np.arange(EXPECTED_FRAMES, dtype=np.int64),
        "target_frame_index": cyclic_target_indices(EXPECTED_FRAMES),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "geometry_blind_adjacent_feature_reanchor_raw_receipt",
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "pairing": "independent cyclic (t -> t+1) re-anchoring for all 180 sources",
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
        "previous_decode_propagated": False,
        "target_detector_used_to_choose_or_centre_search": False,
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
    arrays, receipt = run_raw(manifest, args.role_key, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "raw_adjacent_feature_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    receipt["manifest"] = manifest_record
    receipt["raw_arrays"] = _file_record(arrays_path)
    receipt_path = args.output_dir / "raw_adjacent_feature_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path.resolve()), "raw_arrays_sha256": receipt["raw_arrays"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
