"""Check whether the candidate loss ranks the observed frozen wobble events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .frozen_nuisance_sensitivity import apply_nuisance
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
except ImportError:  # pragma: no cover - direct script compatibility
    from frozen_nuisance_sensitivity import apply_nuisance  # type: ignore
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


SCHEMA_VERSION = "frozen_equivariance_alignment.v1"
LEGACY_UNFILTERED_IMPLEMENTATION_SHA256 = (
    "3029cd6f6c5aa900d0f4a257d5ad6138f1fda5235eab1757bf09b955eb2c394f"
)


class FrozenEquivarianceAlignmentError(ValueError):
    """Raised when source binding or event alignment differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenEquivarianceAlignmentError(message)


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


def _read_json(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _file_record(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "JSON root must be an object")
    return payload, record


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with stable tie handling."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(flat.size > 0 and np.isfinite(flat).all(), "rank values are invalid")
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


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


def _summarize_alignment(
    js: np.ndarray,
    residual_pixels: np.ndarray,
    hard_changed: np.ndarray,
) -> dict[str, Any]:
    _require(js.shape == residual_pixels.shape == hard_changed.shape, "event shapes differ")
    _require(js.shape == (EXPECTED_FRAMES, EXPECTED_CHANNELS), "event matrix shape differs")
    flat_js = np.asarray(js, dtype=np.float64).reshape(-1)
    flat_residual = np.asarray(residual_pixels, dtype=np.float64).reshape(-1)
    flat_changed = np.asarray(hard_changed, dtype=np.bool_).reshape(-1)
    js_ranks = _average_ranks(flat_js)
    residual_order = np.argsort(-flat_residual, kind="mergesort")
    top_events = []
    for flat_index in residual_order[:20]:
        frame_index, channel_index = np.unravel_index(int(flat_index), js.shape)
        top_events.append(
            {
                "frame_index": int(frame_index),
                "channel_index": int(channel_index),
                "soft_residual_pixels": float(flat_residual[flat_index]),
                "jensen_shannon": float(flat_js[flat_index]),
                "jensen_shannon_rank_percentile": float(js_ranks[flat_index] / flat_js.size),
                "hard_peak_changed": bool(flat_changed[flat_index]),
            }
        )
    worst = top_events[0]
    changed_summary = _quantiles(flat_js[flat_changed]) if np.any(flat_changed) else None
    unchanged_summary = _quantiles(flat_js[~flat_changed]) if np.any(~flat_changed) else None
    return {
        "observation_count": int(flat_js.size),
        "spearman_rank_correlation_js_vs_soft_residual": _spearman(flat_js, flat_residual),
        "jensen_shannon": _quantiles(flat_js),
        "jensen_shannon_when_hard_peak_changed": changed_summary,
        "jensen_shannon_when_hard_peak_unchanged": unchanged_summary,
        "hard_peak_changed_count": int(np.count_nonzero(flat_changed)),
        "worst_soft_residual_event": worst,
        "top_20_soft_residual_events": top_events,
        "per_channel_jensen_shannon": [
            _quantiles(js[:, channel]) for channel in range(EXPECTED_CHANNELS)
        ],
        "statistical_status": (
            "descriptive only; 180 frames in one orbit and ten channels in one model "
            "are correlated and are not treated as independent population samples"
        ),
    }


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
    nuisance_receipt, nuisance_receipt_record = _read_json(nuisance_receipt_path)
    _require(nuisance_receipt.get("role", {}).get("role_key") == role_key, "nuisance role differs")
    recorded_prefilter = nuisance_receipt.get("scope", {}).get("prefilter")
    _require(recorded_prefilter in {None, "none"}, "nuisance receipt is filtered")
    if recorded_prefilter is None:
        legacy_source = nuisance_receipt.get("implementation_sources", {}).get(
            "keypoint_net/frozen_nuisance_sensitivity.py", {}
        )
        _require(
            legacy_source.get("sha256") == LEGACY_UNFILTERED_IMPLEMENTATION_SHA256,
            "implicit unfiltered receipt is not bound to the legacy no-filter implementation",
        )
    arrays_record = nuisance_receipt.get("arrays")
    _require(isinstance(arrays_record, dict), "nuisance arrays record is absent")
    arrays_path = Path(arrays_record["absolute_path"]).resolve(strict=True)
    _require(_sha256(arrays_path) == arrays_record.get("sha256"), "nuisance arrays hash differs")
    _require(arrays_path.stat().st_size == arrays_record.get("size_bytes"), "nuisance arrays size differs")
    with np.load(arrays_path, allow_pickle=False) as nuisance_arrays:
        nuisance_names = [str(value) for value in nuisance_arrays["nuisance_name"].tolist()]
        soft_residual = np.asarray(nuisance_arrays["soft_residual_pixels"], dtype=np.float64)
        hard_changed = np.asarray(nuisance_arrays["hard_peak_changed"], dtype=np.bool_)

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

    result_js = np.empty(
        (len(LOCKED_EQUIVARIANCE_SPECS), EXPECTED_FRAMES, EXPECTED_CHANNELS),
        dtype=np.float64,
    )
    state_before = _state_sha256(model)
    with torch.inference_mode():
        for start in range(0, EXPECTED_FRAMES, batch_size):
            stop = min(start + batch_size, EXPECTED_FRAMES)
            normalized = _preprocess(image_paths[start:stop])
            _, base_logits = model.extractor(normalized)
            base_probability = spatial_probabilities(base_logits)
            for spec_index, spec in enumerate(LOCKED_EQUIVARIANCE_SPECS):
                _, actual_logits = model.extractor(apply_nuisance(normalized, spec))
                expected = warp_spatial_distribution(base_probability, spec)
                actual = spatial_probabilities(actual_logits)
                js = jensen_shannon_per_heatmap(expected, actual)
                result_js[spec_index, start:stop] = js.cpu().numpy()
    state_after = _state_sha256(model)
    _require(state_before == state_after, "model state changed")

    summaries: dict[str, Any] = {}
    aligned_residual = np.empty_like(result_js)
    aligned_changed = np.empty(result_js.shape, dtype=np.bool_)
    for spec_index, spec in enumerate(LOCKED_EQUIVARIANCE_SPECS):
        _require(spec.name in nuisance_names, f"nuisance arrays lack {spec.name}")
        nuisance_index = nuisance_names.index(spec.name)
        aligned_residual[spec_index] = soft_residual[nuisance_index]
        aligned_changed[spec_index] = hard_changed[nuisance_index]
        summaries[spec.name] = _summarize_alignment(
            result_js[spec_index],
            aligned_residual[spec_index],
            aligned_changed[spec_index],
        )

    arrays = {
        "nuisance_name": np.asarray([spec.name for spec in LOCKED_EQUIVARIANCE_SPECS]),
        "jensen_shannon": result_js,
        "soft_residual_pixels": aligned_residual,
        "hard_peak_changed": aligned_changed,
        "frame_index": np.arange(EXPECTED_FRAMES, dtype=np.int64),
        "channel_index": np.arange(EXPECTED_CHANNELS, dtype=np.int64),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_equivariance_loss_wobble_alignment",
        "role": dict(role),
        "checkpoint": checkpoint_record,
        "checkpoint_configuration": dict(config),
        "scope": {
            "frames": EXPECTED_FRAMES,
            "channels": EXPECTED_CHANNELS,
            "transforms": [spec.name for spec in LOCKED_EQUIVARIANCE_SPECS],
            "observations_per_transform": EXPECTED_FRAMES * EXPECTED_CHANNELS,
            "batch_size": batch_size,
            "device": "cpu",
        },
        "summary": summaries,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "nuisance_receipt": nuisance_receipt_record,
        "nuisance_arrays": _file_record(arrays_path),
        "opened_rgb_count": len(image_paths),
        "opened_rgb_paths": [str(path) for path in image_paths],
        "forbidden_inputs_opened": [],
        "masks_theta_operator_or_material_coordinates_opened": False,
        "gradients_enabled": False,
        "optimizer_constructed": False,
        "training_or_weight_update_performed": False,
    }
    return arrays, receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--nuisance-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)

    _require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    torch.set_num_threads(args.torch_threads)
    manifest, manifest_record = _load_manifest(args.manifest.resolve(strict=True))
    arrays, receipt = run_role(
        manifest,
        args.role_key,
        args.nuisance_receipt.resolve(strict=True),
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "equivariance_alignment_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    source_root = Path(__file__).resolve().parents[1]
    receipt["implementation_sources"] = {
        relative: _file_record(source_root / relative)
        for relative in (
            "keypoint_net/frozen_nuisance_sensitivity.py",
            "keypoint_net/same_frame_equivariance.py",
            "keypoint_net/run_frozen_equivariance_alignment.py",
            "keypoint_net/run_frozen_nuisance_sensitivity.py",
            "keypoint_net/run_frozen_feature_decode_raw.py",
            "keypoint_net/run_frozen_wobble_forensics.py",
            "keypoint_net/model.py",
        )
    }
    receipt["manifest"] = manifest_record
    receipt["arrays"] = _file_record(arrays_path)
    receipt_path = args.output_dir / "equivariance_alignment_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
