"""Run one source-bound frozen detector nuisance-sensitivity role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .frozen_nuisance_sensitivity import (
        IMAGE_SIZE,
        LOCKED_SPECS,
        apply_nuisance,
        normalized_residual_cells,
        normalized_residual_pixels,
        undo_nuisance_coordinates,
    )
    from .run_frozen_feature_decode_raw import (
        EXPECTED_CHANNELS,
        EXPECTED_FRAMES,
        FEATURE_SIZE,
        _hard_coordinates,
        _image_paths,
        _load_manifest,
        _preprocess,
    )
    from .run_frozen_wobble_forensics import (
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from frozen_nuisance_sensitivity import (  # type: ignore
        IMAGE_SIZE,
        LOCKED_SPECS,
        apply_nuisance,
        normalized_residual_cells,
        normalized_residual_pixels,
        undo_nuisance_coordinates,
    )
    from run_frozen_feature_decode_raw import (  # type: ignore
        EXPECTED_CHANNELS,
        EXPECTED_FRAMES,
        FEATURE_SIZE,
        _hard_coordinates,
        _image_paths,
        _load_manifest,
        _preprocess,
    )
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )


SCHEMA_VERSION = "frozen_nuisance_sensitivity.v1"
SECOND_PEAK_RADIUS_CELLS = 4.0


class FrozenNuisanceRunError(ValueError):
    """Raised when a frozen nuisance run cannot satisfy its lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenNuisanceRunError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return {
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _spatial_peak_summary(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard cells, endpoint xy, and separated top-one/top-two margin."""
    _require(tuple(logits.shape[1:]) == (EXPECTED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), "logit shape differs")
    flat = logits.reshape(logits.shape[0], EXPECTED_CHANNELS, -1)
    hard_cell = torch.argmax(flat, dim=-1)
    hard_coordinate = _hard_coordinates(logits).to(logits.device)
    top1 = torch.gather(flat, -1, hard_cell.unsqueeze(-1)).squeeze(-1)
    top1_y = torch.div(hard_cell, FEATURE_SIZE, rounding_mode="floor")
    top1_x = torch.remainder(hard_cell, FEATURE_SIZE)
    yy, xx = torch.meshgrid(
        torch.arange(FEATURE_SIZE, device=logits.device),
        torch.arange(FEATURE_SIZE, device=logits.device),
        indexing="ij",
    )
    distance_sq = (
        (xx.view(1, 1, FEATURE_SIZE, FEATURE_SIZE) - top1_x[..., None, None]).square()
        + (yy.view(1, 1, FEATURE_SIZE, FEATURE_SIZE) - top1_y[..., None, None]).square()
    )
    separated = logits.masked_fill(distance_sq <= SECOND_PEAK_RADIUS_CELLS**2, -torch.inf)
    top2 = separated.reshape(logits.shape[0], EXPECTED_CHANNELS, -1).amax(dim=-1)
    _require(bool(torch.isfinite(top2).all()), "no separated second peak exists")
    return hard_cell, hard_coordinate, top1 - top2


def _infer(model: torch.nn.Module, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        flattened, logits = model.extractor(normalized)
    soft = flattened.reshape(normalized.shape[0], EXPECTED_CHANNELS, 2)
    hard_cell, hard_coordinate, margin = _spatial_peak_summary(logits)
    _require(
        bool(torch.isfinite(soft).all())
        and bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(margin).all()),
        "inference produced non-finite values",
    )
    return {
        "soft_coordinate": soft,
        "hard_coordinate": hard_coordinate,
        "hard_cell": hard_cell,
        "separated_logit_margin": margin,
        "logits": logits,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(flat.size > 0 and np.isfinite(flat).all(), "summary values are invalid")
    return {
        "minimum": float(np.min(flat)),
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "q90": float(np.quantile(flat, 0.90)),
        "q99": float(np.quantile(flat, 0.99)),
        "maximum": float(np.max(flat)),
    }


def run_role(
    manifest: Mapping[str, Any],
    role_key: str,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require(batch_size > 0, "batch size must be positive")
    matches = [item for item in manifest["roles"] if item.get("role_key") == role_key]
    _require(len(matches) == 1, "role key is absent or duplicated")
    role = matches[0]
    checkpoint = role["checkpoint"]
    payload, checkpoint_record = _same_fd_checkpoint_load(
        checkpoint["absolute_path"],
        expected_sha256=checkpoint["sha256"],
        expected_size=checkpoint["size_bytes"],
    )
    model, config = _construct_frozen_model(
        payload,
        cell_id=role["cell_id"],
        checkpoint_role=role["checkpoint_role"],
        expected_epoch=role["checkpoint_epoch"],
    )
    _require(not model.training, "model is not in evaluation mode")
    _require(all(not parameter.requires_grad for parameter in model.parameters()), "model is not frozen")
    image_paths = _image_paths(manifest)
    _require(len(image_paths) == EXPECTED_FRAMES, "frame count differs")

    nuisance_names = [spec.name for spec in LOCKED_SPECS]
    shape_pfk = (len(LOCKED_SPECS), EXPECTED_FRAMES, EXPECTED_CHANNELS)
    arrays = {
        "soft_residual_pixels": np.empty(shape_pfk, dtype=np.float64),
        "hard_residual_pixels": np.empty(shape_pfk, dtype=np.float64),
        "hard_residual_cells": np.empty(shape_pfk, dtype=np.float64),
        "hard_peak_changed": np.empty(shape_pfk, dtype=np.bool_),
        "baseline_soft_coordinate": np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), dtype=np.float64),
        "baseline_hard_coordinate": np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS, 2), dtype=np.float64),
        "baseline_separated_logit_margin": np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=np.float64),
        "perturbed_soft_coordinate": np.empty(shape_pfk + (2,), dtype=np.float64),
        "perturbed_hard_coordinate": np.empty(shape_pfk + (2,), dtype=np.float64),
        "recovered_soft_coordinate": np.empty(shape_pfk + (2,), dtype=np.float64),
        "recovered_hard_coordinate": np.empty(shape_pfk + (2,), dtype=np.float64),
        "perturbed_separated_logit_margin": np.empty(shape_pfk, dtype=np.float64),
    }

    state_before = _state_sha256(model)
    for start in range(0, EXPECTED_FRAMES, batch_size):
        stop = min(start + batch_size, EXPECTED_FRAMES)
        normalized = _preprocess(image_paths[start:stop])
        baseline = _infer(model, normalized)
        arrays["baseline_soft_coordinate"][start:stop] = baseline["soft_coordinate"].cpu().numpy()
        arrays["baseline_hard_coordinate"][start:stop] = baseline["hard_coordinate"].cpu().numpy()
        arrays["baseline_separated_logit_margin"][start:stop] = baseline[
            "separated_logit_margin"
        ].cpu().numpy()

        for nuisance_index, spec in enumerate(LOCKED_SPECS):
            perturbed_input = apply_nuisance(normalized, spec)
            perturbed = _infer(model, perturbed_input)
            recovered_soft = undo_nuisance_coordinates(perturbed["soft_coordinate"], spec)
            recovered_hard = undo_nuisance_coordinates(perturbed["hard_coordinate"], spec)
            soft_pixels = normalized_residual_pixels(baseline["soft_coordinate"], recovered_soft)
            hard_pixels = normalized_residual_pixels(baseline["hard_coordinate"], recovered_hard)
            hard_cells = normalized_residual_cells(baseline["hard_coordinate"], recovered_hard)

            index = (nuisance_index, slice(start, stop))
            arrays["soft_residual_pixels"][index] = soft_pixels.cpu().numpy()
            arrays["hard_residual_pixels"][index] = hard_pixels.cpu().numpy()
            arrays["hard_residual_cells"][index] = hard_cells.cpu().numpy()
            arrays["hard_peak_changed"][index] = (
                perturbed["hard_cell"] != baseline["hard_cell"]
            ).cpu().numpy()
            arrays["perturbed_soft_coordinate"][index] = perturbed["soft_coordinate"].cpu().numpy()
            arrays["perturbed_hard_coordinate"][index] = perturbed["hard_coordinate"].cpu().numpy()
            arrays["recovered_soft_coordinate"][index] = recovered_soft.cpu().numpy()
            arrays["recovered_hard_coordinate"][index] = recovered_hard.cpu().numpy()
            arrays["perturbed_separated_logit_margin"][index] = perturbed[
                "separated_logit_margin"
            ].cpu().numpy()

            if spec.kind == "identity":
                _require(torch.equal(perturbed_input, normalized), "identity input differs")
                for field in (
                    "soft_coordinate",
                    "hard_coordinate",
                    "hard_cell",
                    "separated_logit_margin",
                    "logits",
                ):
                    _require(torch.equal(perturbed[field], baseline[field]), f"identity changed {field}")

    state_after = _state_sha256(model)
    _require(state_before == state_after, "model state changed during frozen inference")
    arrays["frame_index"] = np.arange(EXPECTED_FRAMES, dtype=np.int64)
    arrays["channel_index"] = np.arange(EXPECTED_CHANNELS, dtype=np.int64)
    arrays["nuisance_name"] = np.asarray(nuisance_names)

    summary: dict[str, Any] = {}
    for nuisance_index, spec in enumerate(LOCKED_SPECS):
        soft = arrays["soft_residual_pixels"][nuisance_index]
        hard_pixels = arrays["hard_residual_pixels"][nuisance_index]
        hard_cells = arrays["hard_residual_cells"][nuisance_index]
        changed = arrays["hard_peak_changed"][nuisance_index]
        worst_flat = int(np.argmax(soft))
        worst_frame, worst_channel = np.unravel_index(worst_flat, soft.shape)
        summary[spec.name] = {
            "spec": {
                "kind": spec.kind,
                "brightness_delta": spec.brightness_delta,
                "dx_pixels": spec.dx_pixels,
                "dy_pixels": spec.dy_pixels,
                "clockwise_degrees": spec.clockwise_degrees,
            },
            "soft_residual_pixels": _quantiles(soft),
            "hard_residual_pixels": _quantiles(hard_pixels),
            "hard_residual_cells": _quantiles(hard_cells),
            "hard_peak_changed_count": int(np.count_nonzero(changed)),
            "hard_peak_observation_count": int(changed.size),
            "hard_peak_changed_fraction": float(np.mean(changed)),
            "per_channel_soft_maximum_pixels": [float(value) for value in np.max(soft, axis=0)],
            "per_channel_hard_maximum_cells": [float(value) for value in np.max(hard_cells, axis=0)],
            "worst_soft_event": {
                "frame_index": int(worst_frame),
                "channel_index": int(worst_channel),
                "residual_pixels": float(soft[worst_frame, worst_channel]),
            },
        }

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_detector_nuisance_sensitivity_role",
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "scope": {
            "frames": EXPECTED_FRAMES,
            "channels": EXPECTED_CHANNELS,
            "nuisances": nuisance_names,
            "batch_size": batch_size,
            "device": "cpu",
        },
        "summary": summary,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "identity_repeat_exact": True,
        "second_peak_exclusion_radius_cells": SECOND_PEAK_RADIUS_CELLS,
        "opened_rgb_paths": [str(path) for path in image_paths],
        "opened_rgb_count": len(image_paths),
        "forbidden_inputs_opened": [],
        "masks_theta_operator_or_material_coordinates_opened": False,
        "optimizer_constructed": False,
        "gradients_enabled": False,
        "training_or_weight_update_performed": False,
    }
    return arrays, receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)

    _require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    torch.set_num_threads(args.torch_threads)
    manifest, manifest_record = _load_manifest(args.manifest.resolve(strict=True))
    arrays, receipt = run_role(manifest, args.role_key, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "nuisance_sensitivity_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    source_root = Path(__file__).resolve().parents[1]
    receipt["implementation_sources"] = {
        relative: _file_record(source_root / relative)
        for relative in (
            "keypoint_net/frozen_nuisance_sensitivity.py",
            "keypoint_net/run_frozen_nuisance_sensitivity.py",
            "keypoint_net/run_frozen_feature_decode_raw.py",
            "keypoint_net/run_frozen_wobble_forensics.py",
            "keypoint_net/model.py",
        )
    }
    receipt["manifest"] = manifest_record
    receipt["arrays"] = _file_record(arrays_path)
    receipt_path = args.output_dir / "nuisance_sensitivity_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path.resolve()),
                "arrays_sha256": receipt["arrays"]["sha256"],
                "summary": receipt["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
