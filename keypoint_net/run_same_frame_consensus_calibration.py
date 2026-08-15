"""Generate a model-blind interpolation floor for the frozen consensus decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from .same_frame_consensus import (
        LOCKED_CONSENSUS_SPECS,
        consensus_probabilities_from_view_logits,
        spatial_expectation_from_probabilities,
    )
    from .same_frame_equivariance import warp_spatial_distribution
except ImportError:  # Support direct historical script imports.
    from same_frame_consensus import (  # type: ignore
        LOCKED_CONSENSUS_SPECS,
        consensus_probabilities_from_view_logits,
        spatial_expectation_from_probabilities,
    )
    from same_frame_equivariance import warp_spatial_distribution  # type: ignore


SCHEMA_VERSION = "same_frame_consensus_planted_calibration.v1"
HEATMAP_SIZE = 64
CELL_SCALE = (HEATMAP_SIZE - 1) / 2.0


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("content_hash_sha256", None)
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "absolute_path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _grid(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, HEATMAP_SIZE, dtype=reference.dtype)
    return torch.meshgrid(axis, axis, indexing="ij")


def _topology(probability: torch.Tensor) -> dict[str, float]:
    normalized = probability / probability.sum()
    xy = spatial_expectation_from_probabilities(normalized)[0, 0]
    yy, xx = _grid(normalized)
    radius_squared_cells = (
        torch.square((xx - xy[0]) * CELL_SCALE)
        + torch.square((yy - xy[1]) * CELL_SCALE)
    )
    width = torch.sqrt((normalized[0, 0] * radius_squared_cells).sum())
    flat = normalized.flatten()
    first_index = int(torch.argmax(flat).item())
    first_y, first_x = divmod(first_index, HEATMAP_SIZE)
    y_cells, x_cells = torch.meshgrid(
        torch.arange(HEATMAP_SIZE), torch.arange(HEATMAP_SIZE), indexing="ij"
    )
    separated = torch.square(y_cells - first_y) + torch.square(x_cells - first_x) >= 16
    second_probability = normalized[0, 0][separated].max()
    first_probability = flat[first_index]
    finite_floor = torch.finfo(normalized.dtype).tiny
    margin = torch.log(first_probability.clamp_min(finite_floor)) - torch.log(
        second_probability.clamp_min(finite_floor)
    )
    entropy = -(normalized * torch.log(normalized.clamp_min(torch.finfo(normalized.dtype).tiny))).sum()
    return {
        "soft_x": float(xy[0]),
        "soft_y": float(xy[1]),
        "hard_x_cell": float(first_x),
        "hard_y_cell": float(first_y),
        "rms_width_cells": float(width),
        "entropy_nats": float(entropy),
        "first_peak_probability": float(first_probability),
        "separated_peak_radius4_logit_margin": float(margin),
    }


def _consensus_for_perfect_views(source: torch.Tensor) -> torch.Tensor:
    view_logits = []
    for spec in LOCKED_CONSENSUS_SPECS:
        view = source if spec.kind == "identity" else warp_spatial_distribution(source, spec)
        view_logits.append(torch.log(view.clamp_min(torch.finfo(view.dtype).tiny)))
    consensus, _ = consensus_probabilities_from_view_logits(torch.stack(view_logits))
    return consensus


def _case(name: str, source: torch.Tensor) -> dict[str, Any]:
    source = source / source.sum()
    consensus = _consensus_for_perfect_views(source)
    source_topology = _topology(source)
    consensus_topology = _topology(consensus)
    soft_displacement = np.linalg.norm(
        np.asarray(
            [
                consensus_topology["soft_x"] - source_topology["soft_x"],
                consensus_topology["soft_y"] - source_topology["soft_y"],
            ]
        )
    ) * CELL_SCALE
    hard_displacement = np.linalg.norm(
        np.asarray(
            [
                consensus_topology["hard_x_cell"] - source_topology["hard_x_cell"],
                consensus_topology["hard_y_cell"] - source_topology["hard_y_cell"],
            ]
        )
    )
    return {
        "name": name,
        "source": source_topology,
        "consensus": consensus_topology,
        "consensus_minus_source": {
            "soft_displacement_cells": float(soft_displacement),
            "hard_displacement_cells": float(hard_displacement),
            "rms_width_cells": consensus_topology["rms_width_cells"] - source_topology["rms_width_cells"],
            "entropy_nats": consensus_topology["entropy_nats"] - source_topology["entropy_nats"],
            "first_peak_probability": consensus_topology["first_peak_probability"] - source_topology["first_peak_probability"],
            "separated_peak_radius4_logit_margin": consensus_topology["separated_peak_radius4_logit_margin"] - source_topology["separated_peak_radius4_logit_margin"],
        },
        "consensus_probability_sum": float(consensus.sum()),
        "all_finite": bool(torch.isfinite(consensus).all()),
    }


def build_calibration(repo_root: Path) -> dict[str, Any]:
    dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(HEATMAP_SIZE, dtype=dtype),
        torch.arange(HEATMAP_SIZE, dtype=dtype),
        indexing="ij",
    )
    cases = []
    for y_cell in (16, 32, 47):
        for x_cell in (16, 32, 47):
            delta = torch.zeros((1, 1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=dtype)
            delta[0, 0, y_cell, x_cell] = 1.0
            cases.append(_case(f"delta_y{y_cell}_x{x_cell}", delta))
            for sigma_cells in (0.75, 1.0, 2.0):
                gaussian = torch.exp(
                    -(
                        torch.square(xx - x_cell) + torch.square(yy - y_cell)
                    )
                    / (2.0 * sigma_cells**2)
                ).view(1, 1, HEATMAP_SIZE, HEATMAP_SIZE)
                cases.append(
                    _case(
                        f"gaussian_sigma{sigma_cells:g}_y{y_cell}_x{x_cell}",
                        gaussian,
                    )
                )
    deltas = [row["consensus_minus_source"] for row in cases]
    interpolation_floor = {
        "maximum_soft_displacement_cells": max(row["soft_displacement_cells"] for row in deltas),
        "maximum_hard_displacement_cells": max(row["hard_displacement_cells"] for row in deltas),
        "maximum_rms_width_increase_cells": max(row["rms_width_cells"] for row in deltas),
        "maximum_entropy_increase_nats": max(row["entropy_nats"] for row in deltas),
        "maximum_first_peak_probability_decrease": max(-row["first_peak_probability"] for row in deltas),
        "maximum_separated_peak_margin_decrease": max(-row["separated_peak_radius4_logit_margin"] for row in deltas),
    }
    source_paths = (
        "keypoint_net/run_same_frame_consensus_calibration.py",
        "keypoint_net/same_frame_consensus.py",
        "keypoint_net/same_frame_equivariance.py",
        "keypoint_net/frozen_nuisance_sensitivity.py",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "model_blind_consensus_decoder_interpolation_floor",
        "model_checkpoint_opened": False,
        "training_or_weight_update_performed": False,
        "heatmap_size": HEATMAP_SIZE,
        "dtype": str(dtype),
        "planted_case_count": len(cases),
        "planted_case_scope": {
            "positions_yx_cells": [16, 32, 47],
            "distributions": ["delta", "gaussian_sigma_0.75", "gaussian_sigma_1", "gaussian_sigma_2"],
            "all_positions_interior": True,
            "perfectly_equivariant_views_generated_from_source_distribution": True,
        },
        "locked_view_specs": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "dx_pixels": float(spec.dx_pixels),
                "dy_pixels": float(spec.dy_pixels),
                "clockwise_degrees": float(spec.clockwise_degrees),
            }
            for spec in LOCKED_CONSENSUS_SPECS
        ],
        "interpolation_floor": interpolation_floor,
        "cases": cases,
        "semantic_assertions": {
            "all_cases_finite": all(row["all_finite"] for row in cases),
            "all_probability_sums_within_1e_6": all(
                abs(row["consensus_probability_sum"] - 1.0) <= 1e-6
                for row in cases
            ),
            "no_hard_cell_change": interpolation_floor["maximum_hard_displacement_cells"] == 0.0,
            "maximum_soft_displacement_below_0_01_cell": interpolation_floor["maximum_soft_displacement_cells"] < 0.01,
        },
        "statistical_scope": {
            "status": "deterministic calibration, not inferential",
            "sample_unit": "planted interior heatmap distribution",
            "n": len(cases),
            "correlation_caveat": "grid cases are deterministic controls, not population samples",
        },
        "bindings": {
            "implementation_head": head,
            "implementation_sources": {
                path: _file_record(repo_root / path) for path in source_paths
            },
        },
    }
    result["all_semantic_assertions_pass"] = all(result["semantic_assertions"].values())
    result["content_hash_sha256"] = _content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_calibration(Path(__file__).resolve().parents[1])
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": _file_record(output), "all_semantic_assertions_pass": result["all_semantic_assertions_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
