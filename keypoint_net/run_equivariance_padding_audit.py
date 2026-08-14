"""Audit whether heatmap target padding changes the frozen wobble conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .frozen_nuisance_sensitivity import (
        IMAGE_SIZE,
        apply_nuisance,
        nuisance_output_to_input_affine,
    )
    from .run_frozen_equivariance_alignment import (
        _average_ranks,
        _file_record,
        _quantiles,
        _read_json,
        _spearman,
    )
    from .run_frozen_feature_decode_raw import (
        EXPECTED_CHANNELS,
        EXPECTED_FRAMES,
        _image_paths,
        _load_manifest,
        _preprocess,
    )
    from .run_frozen_wobble_forensics import (
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from .same_frame_equivariance import (
        LOCKED_EQUIVARIANCE_SPECS,
        jensen_shannon_per_heatmap,
        spatial_probabilities,
        warp_spatial_distribution,
    )
except ImportError:  # pragma: no cover
    from frozen_nuisance_sensitivity import (  # type: ignore
        IMAGE_SIZE,
        apply_nuisance,
        nuisance_output_to_input_affine,
    )
    from run_frozen_equivariance_alignment import (  # type: ignore
        _average_ranks,
        _file_record,
        _quantiles,
        _read_json,
        _spearman,
    )
    from run_frozen_feature_decode_raw import (  # type: ignore
        EXPECTED_CHANNELS,
        EXPECTED_FRAMES,
        _image_paths,
        _load_manifest,
        _preprocess,
    )
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from same_frame_equivariance import (  # type: ignore
        LOCKED_EQUIVARIANCE_SPECS,
        jensen_shannon_per_heatmap,
        spatial_probabilities,
        warp_spatial_distribution,
    )


SCHEMA_VERSION = "equivariance_padding_audit.v1"
PADDING_POLICIES = ("zeros", "reflection")


class EquivariancePaddingAuditError(ValueError):
    """Raised when the source-bound audit differs from its semantic lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EquivariancePaddingAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _warp_with_padding(
    probabilities: torch.Tensor,
    spec: Any,
    *,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized warped mass and pre-renormalization mass per heatmap."""
    _require(padding_mode in PADDING_POLICIES, "unsupported padding policy")
    _require(probabilities.ndim == 4, "probabilities must be BxNxHxW")
    normalized = probabilities / probabilities.sum(dim=(-2, -1), keepdim=True)
    theta = nuisance_output_to_input_affine(
        spec,
        batch_size=normalized.shape[0],
        coordinate_size=IMAGE_SIZE,
        reference=normalized,
    )
    grid = F.affine_grid(theta, normalized.shape, align_corners=True)
    warped = F.grid_sample(
        normalized,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    ).clamp_min(0.0)
    mass = warped.sum(dim=(-2, -1), keepdim=True)
    _require(bool((mass > 0).all()), "warp removed all heatmap mass")
    return warped / mass, mass.squeeze(-1).squeeze(-1)


def _outer_cell_mass(probabilities: torch.Tensor) -> torch.Tensor:
    """Probability mass in the one-cell outer ring for every batch/channel."""
    _require(probabilities.ndim == 4, "probabilities must be BxNxHxW")
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask[..., 0, :] = True
    mask[..., -1, :] = True
    mask[..., :, 0] = True
    mask[..., :, -1] = True
    return torch.where(mask, probabilities, torch.zeros_like(probabilities)).sum(dim=(-2, -1))


def _top_indices(values: np.ndarray, count: int = 20) -> set[int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(flat.size >= count and np.isfinite(flat).all(), "top-index values are invalid")
    return set(np.argsort(-flat, kind="mergesort")[:count].tolist())


def _compare_policies(
    zero_js: np.ndarray,
    reflection_js: np.ndarray,
    residual_pixels: np.ndarray,
) -> dict[str, Any]:
    _require(zero_js.shape == reflection_js.shape == residual_pixels.shape, "event shapes differ")
    flat_residual = residual_pixels.reshape(-1)
    worst_index = int(np.argmax(flat_residual))
    zero_ranks = _average_ranks(zero_js)
    reflection_ranks = _average_ranks(reflection_js)
    zero_top = _top_indices(zero_js)
    reflection_top = _top_indices(reflection_js)
    overlap = len(zero_top & reflection_top)
    event_count = flat_residual.size
    result = {
        "observation_count": event_count,
        "zero_vs_reflection_js_spearman": _spearman(zero_js, reflection_js),
        "absolute_js_difference": _quantiles(np.abs(zero_js - reflection_js)),
        "top20_overlap_count": overlap,
        "top20_overlap_fraction": overlap / 20.0,
        "worst_residual_event": {
            "frame_index": int(np.unravel_index(worst_index, residual_pixels.shape)[0]),
            "channel_index": int(np.unravel_index(worst_index, residual_pixels.shape)[1]),
            "residual_pixels": float(flat_residual[worst_index]),
            "zero_js": float(zero_js.reshape(-1)[worst_index]),
            "reflection_js": float(reflection_js.reshape(-1)[worst_index]),
            "zero_js_rank_percentile": float(zero_ranks[worst_index] / event_count),
            "reflection_js_rank_percentile": float(reflection_ranks[worst_index] / event_count),
        },
    }
    result["decision_changing"] = bool(
        result["worst_residual_event"]["reflection_js_rank_percentile"] < 0.5
        or overlap < 10
    )
    return result


def run_role(
    manifest: Mapping[str, Any],
    role_key: str,
    nuisance_receipt_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require(batch_size > 0, "batch size must be positive")
    matches = [item for item in manifest["roles"] if item.get("role_key") == role_key]
    _require(len(matches) == 1, "role key is absent or duplicated")
    role = matches[0]
    nuisance_receipt, nuisance_record = _read_json(nuisance_receipt_path)
    _require(nuisance_receipt.get("role", {}).get("role_key") == role_key, "nuisance role differs")
    arrays_record = nuisance_receipt.get("arrays")
    _require(isinstance(arrays_record, dict), "nuisance arrays record is absent")
    arrays_path = Path(arrays_record["absolute_path"]).resolve(strict=True)
    _require(_sha256(arrays_path) == arrays_record.get("sha256"), "nuisance arrays hash differs")
    with np.load(arrays_path, allow_pickle=False) as nuisance_arrays:
        nuisance_names = [str(value) for value in nuisance_arrays["nuisance_name"].tolist()]
        soft_residual = np.asarray(nuisance_arrays["soft_residual_pixels"], dtype=np.float64)

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
    image_paths = _image_paths(manifest)
    _require(len(image_paths) == EXPECTED_FRAMES, "frame count differs")

    js = np.empty((2, 2, EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=np.float64)
    target_mass = np.empty_like(js)
    border_mass = np.empty((EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=np.float64)
    state_before = _state_sha256(model)
    with torch.inference_mode():
        for start in range(0, EXPECTED_FRAMES, batch_size):
            stop = min(start + batch_size, EXPECTED_FRAMES)
            normalized = _preprocess(image_paths[start:stop])
            _, base_logits = model.extractor(normalized)
            base_probability = spatial_probabilities(base_logits)
            border_mass[start:stop] = _outer_cell_mass(base_probability).cpu().numpy()
            for spec_index, spec in enumerate(LOCKED_EQUIVARIANCE_SPECS):
                _, actual_logits = model.extractor(apply_nuisance(normalized, spec))
                actual = spatial_probabilities(actual_logits)
                for policy_index, policy in enumerate(PADDING_POLICIES):
                    expected, mass = _warp_with_padding(
                        base_probability,
                        spec,
                        padding_mode=policy,
                    )
                    js[spec_index, policy_index, start:stop] = (
                        jensen_shannon_per_heatmap(expected, actual).cpu().numpy()
                    )
                    target_mass[spec_index, policy_index, start:stop] = mass.cpu().numpy()
                current = warp_spatial_distribution(base_probability, spec)
                _require(
                    bool(torch.equal(current, _warp_with_padding(base_probability, spec, padding_mode="zeros")[0])),
                    "zero-padding audit path differs from candidate",
                )
    state_after = _state_sha256(model)
    _require(state_before == state_after, "model state changed")

    summaries = {}
    residual = np.empty((2, EXPECTED_FRAMES, EXPECTED_CHANNELS), dtype=np.float64)
    for spec_index, spec in enumerate(LOCKED_EQUIVARIANCE_SPECS):
        _require(spec.name in nuisance_names, f"nuisance arrays lack {spec.name}")
        nuisance_index = nuisance_names.index(spec.name)
        residual[spec_index] = soft_residual[nuisance_index]
        summaries[spec.name] = _compare_policies(
            js[spec_index, 0],
            js[spec_index, 1],
            residual[spec_index],
        )
        summaries[spec.name]["zero_target_mass"] = _quantiles(target_mass[spec_index, 0])
        summaries[spec.name]["reflection_target_mass"] = _quantiles(target_mass[spec_index, 1])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_equivariance_target_padding_audit",
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "scope": {
            "frames": EXPECTED_FRAMES,
            "channels": EXPECTED_CHANNELS,
            "transforms": [spec.name for spec in LOCKED_EQUIVARIANCE_SPECS],
            "padding_policies": list(PADDING_POLICIES),
            "batch_size": batch_size,
            "device": "cpu",
        },
        "summary": summaries,
        "decision_changing_any_case": any(item["decision_changing"] for item in summaries.values()),
        "base_outer_one_cell_probability_mass": _quantiles(border_mass),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "nuisance_receipt": nuisance_record,
        "nuisance_arrays": _file_record(arrays_path),
        "opened_rgb_count": len(image_paths),
        "opened_rgb_paths": [str(path) for path in image_paths],
        "forbidden_inputs_opened": [],
        "masks_theta_operator_or_material_coordinates_opened": False,
        "gradients_enabled": False,
        "optimizer_constructed": False,
        "training_or_weight_update_performed": False,
    }
    arrays = {
        "nuisance_name": np.asarray([spec.name for spec in LOCKED_EQUIVARIANCE_SPECS]),
        "padding_policy": np.asarray(PADDING_POLICIES),
        "jensen_shannon": js,
        "target_mass_before_renormalization": target_mass,
        "base_outer_one_cell_probability_mass": border_mass,
        "soft_residual_pixels": residual,
        "frame_index": np.arange(EXPECTED_FRAMES, dtype=np.int64),
        "channel_index": np.arange(EXPECTED_CHANNELS, dtype=np.int64),
    }
    return arrays, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--nuisance-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    _require(not args.output_dir.exists(), "output directory already exists")
    torch.set_num_threads(args.torch_threads)
    manifest, manifest_record = _load_manifest(args.manifest.resolve(strict=True))
    arrays, receipt = run_role(
        manifest,
        args.role_key,
        args.nuisance_receipt.resolve(strict=True),
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "padding_audit_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    source_root = Path(__file__).resolve().parents[1]
    receipt["implementation_sources"] = {
        relative: _file_record(source_root / relative)
        for relative in (
            "keypoint_net/run_equivariance_padding_audit.py",
            "keypoint_net/frozen_nuisance_sensitivity.py",
            "keypoint_net/same_frame_equivariance.py",
            "keypoint_net/run_frozen_equivariance_alignment.py",
            "keypoint_net/run_frozen_feature_decode_raw.py",
            "keypoint_net/run_frozen_wobble_forensics.py",
            "keypoint_net/model.py",
        )
    }
    receipt["manifest"] = manifest_record
    receipt["arrays"] = _file_record(arrays_path)
    receipt_path = args.output_dir / "padding_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path.resolve()),
        "decision_changing_any_case": receipt["decision_changing_any_case"],
        "summary": receipt["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
