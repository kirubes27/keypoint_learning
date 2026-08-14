"""Generate the model-blind calibration for the frozen wobble forensic gate.

The calibration is deterministic and uses only planted analytic trajectories
and heatmaps.  It must complete before a saved model checkpoint is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .frozen_wobble_forensics import (
        canonicalize,
        coordinate_dynamics,
        endpoint_grid,
        geometry_smoke,
        heatmap_topology,
        phase_metrics,
        raw_from_canonical,
    )
except ImportError:  # Support direct ``python keypoint_net/...`` execution.
    from frozen_wobble_forensics import (  # type: ignore
        canonicalize,
        coordinate_dynamics,
        endpoint_grid,
        geometry_smoke,
        heatmap_topology,
        phase_metrics,
        raw_from_canonical,
    )


SCHEMA_VERSION = "frozen_wobble_oracle_calibration.v1_2"
FRAME_COUNT = 180
FRAME_STEP_DEG = 2.0
HEATMAP_SIZE = 64
GRID_STEP = 2.0 / (HEATMAP_SIZE - 1)
AMPLITUDE_MULTIPLIERS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RADII_GRID_STEPS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0)
ANCHOR_ANGLES_DEG = tuple(float(value) for value in range(0, 360, 15))
GAUSSIAN_WIDTHS_CELLS = (0.75, 1.25, 2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    normalized = dict(value)
    normalized.pop("content_hash_sha256", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_maxima(points: np.ndarray) -> dict[str, float]:
    dynamics = coordinate_dynamics(points)
    centred = points - np.mean(points, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=0))
    cyclic_second = np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)
    radius = dynamics["radius_about_mean"]
    return {
        "canonical_rms_max": float(np.max(rms)),
        "canonical_radius_max": float(np.max(radius)),
        "adjacent_step_max": float(np.max(dynamics["adjacent_step"])),
        "within_residue_step_max": float(np.max(dynamics["within_residue_step"])),
        "second_difference_max": float(np.max(dynamics["second_difference"])),
        "cyclic_second_difference_max": float(np.max(np.linalg.norm(cyclic_second, axis=-1))),
    }


def _merge_maxima(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot merge an empty metric list")
    return {key: max(row[key] for row in rows) for key in rows[0]}


def _quantize_endpoint_grid(points: np.ndarray) -> np.ndarray:
    cell = np.rint((points + 1.0) * 0.5 * (HEATMAP_SIZE - 1))
    cell = np.clip(cell, 0, HEATMAP_SIZE - 1)
    return -1.0 + 2.0 * cell / (HEATMAP_SIZE - 1)


def _clean_coordinate_oracles() -> dict[str, Any]:
    theta = np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_STEP_DEG
    exact_rows: list[dict[str, float]] = []
    quantized_rows: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []
    for radius_steps in RADII_GRID_STEPS:
        radius = radius_steps * GRID_STEP
        for anchor_angle in ANCHOR_ANGLES_DEG:
            angle = math.radians(anchor_angle)
            anchor = np.asarray((radius * math.cos(angle), radius * math.sin(angle)))
            canonical = np.broadcast_to(anchor, (FRAME_COUNT, 1, 2)).copy()
            raw = raw_from_canonical(canonical, theta)
            exact = canonicalize(raw, theta)
            quantized = canonicalize(_quantize_endpoint_grid(raw), theta)
            exact_metric = _metric_maxima(exact)
            quantized_metric = _metric_maxima(quantized)
            exact_rows.append(exact_metric)
            quantized_rows.append(quantized_metric)
            cases.append(
                {
                    "radius_grid_steps": radius_steps,
                    "anchor_angle_deg": anchor_angle,
                    "exact": exact_metric,
                    "r64_hard_quantized": quantized_metric,
                }
            )
    return {
        "case_count": len(cases),
        "radii_grid_steps": list(RADII_GRID_STEPS),
        "anchor_angles_deg": list(ANCHOR_ANGLES_DEG),
        "cases": cases,
        "exact_envelope": _merge_maxima(exact_rows),
        "r64_hard_quantized_envelope": _merge_maxima(quantized_rows),
    }


def _gaussian_logits(raw_points: np.ndarray, width_cells: float) -> np.ndarray:
    if raw_points.shape != (FRAME_COUNT, 1, 2):
        raise ValueError("raw_points must be (180,1,2)")
    x_cell = (raw_points[:, 0, 0] + 1.0) * 0.5 * (HEATMAP_SIZE - 1)
    y_cell = (raw_points[:, 0, 1] + 1.0) * 0.5 * (HEATMAP_SIZE - 1)
    yy, xx = np.meshgrid(
        np.arange(HEATMAP_SIZE, dtype=np.float64),
        np.arange(HEATMAP_SIZE, dtype=np.float64),
        indexing="ij",
    )
    squared = (
        np.square(xx[None] - x_cell[:, None, None])
        + np.square(yy[None] - y_cell[:, None, None])
    )
    return (-0.5 * squared / (width_cells * width_cells))[:, None]


def _clean_gaussian_oracles() -> dict[str, Any]:
    theta = np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_STEP_DEG
    rows: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []
    # Radii are kept away from the heatmap border, because truncation is itself
    # a border failure rather than a clean material-transport control.
    for radius_steps in (0.5, 1.0, 4.0, 8.0, 12.0, 18.0):
        radius = radius_steps * GRID_STEP
        for anchor_angle in ANCHOR_ANGLES_DEG:
            angle = math.radians(anchor_angle)
            anchor = np.asarray((radius * math.cos(angle), radius * math.sin(angle)))
            canonical = np.broadcast_to(anchor, (FRAME_COUNT, 1, 2)).copy()
            raw = raw_from_canonical(canonical, theta)
            for width in GAUSSIAN_WIDTHS_CELLS:
                topology = heatmap_topology(_gaussian_logits(raw, width))
                recovered = canonicalize(topology["soft_coordinate"], theta)
                metric = _metric_maxima(recovered)
                rows.append(metric)
                cases.append(
                    {
                        "radius_grid_steps": radius_steps,
                        "anchor_angle_deg": anchor_angle,
                        "gaussian_width_cells": width,
                        "soft_coordinate": metric,
                        "soft_hard_distance_cells_max": float(
                            np.max(topology["soft_hard_distance_cells"])
                        ),
                        "entropy_nats_max": float(np.max(topology["entropy_nats"])),
                        "effective_support_cells_max": float(np.max(topology["effective_support_cells"])),
                        "rms_width_cells_max": float(np.max(topology["rms_width_cells"])),
                        "first_peak_probability_min": float(np.min(topology["first_peak_probability"])),
                        "separated_peak_radius4_logit_margin_min": float(
                            np.min(topology["separated_peaks"]["4"]["logit_margin"])
                        ),
                        "separated_peak_radius4_probability_margin_min": float(
                            np.min(topology["separated_peaks"]["4"]["probability_margin"])
                        ),
                    }
                )
    topology_envelope = {
        "soft_hard_distance_cells_max": max(row["soft_hard_distance_cells_max"] for row in cases),
        "entropy_nats_max": max(row["entropy_nats_max"] for row in cases),
        "effective_support_cells_max": max(row["effective_support_cells_max"] for row in cases),
        "rms_width_cells_max": max(row["rms_width_cells_max"] for row in cases),
        "first_peak_probability_min": min(row["first_peak_probability_min"] for row in cases),
        "separated_peak_radius4_logit_margin_min": min(
            row["separated_peak_radius4_logit_margin_min"] for row in cases
        ),
        "separated_peak_radius4_probability_margin_min": min(
            row["separated_peak_radius4_probability_margin_min"] for row in cases
        ),
    }
    return {
        "case_count": len(cases),
        "gaussian_widths_cells": list(GAUSSIAN_WIDTHS_CELLS),
        "cases": cases,
        "soft_single_gaussian_envelope": _merge_maxima(rows),
        "single_gaussian_topology_envelope": topology_envelope,
    }


def _planted_detection_ladders(
    hard_thresholds: dict[str, float], soft_thresholds: dict[str, float]
) -> dict[str, Any]:
    theta = np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_STEP_DEG
    frame = np.arange(FRAME_COUNT, dtype=np.float64)
    anchor = np.asarray((12.0 * GRID_STEP, -4.0 * GRID_STEP))
    base = np.broadcast_to(anchor, (FRAME_COUNT, 1, 2)).copy()
    families: dict[str, list[dict[str, Any]]] = {
        "slow_canonical_slide": [],
        "constant_mod3_offsets": [],
        "isolated_spike": [],
    }
    for multiplier in AMPLITUDE_MULTIPLIERS:
        amplitude = multiplier * GRID_STEP

        slow = base.copy()
        slow[:, 0, 0] += amplitude * np.sin(2.0 * np.pi * frame / FRAME_COUNT)
        slow_metric = _metric_maxima(slow)
        families["slow_canonical_slide"].append(
            {
                "amplitude_grid_steps": multiplier,
                "amplitude_normalized": amplitude,
                "metrics": slow_metric,
                "detected_by_soft_contract": slow_metric["canonical_rms_max"]
                > soft_thresholds["canonical_rms_max"],
            }
        )

        phase = base.copy()
        phase_offsets = np.asarray(((amplitude, 0.0), (-0.5 * amplitude, 0.0), (-0.5 * amplitude, 0.0)))
        phase[:, 0] += phase_offsets[np.arange(FRAME_COUNT) % 3]
        phase_metric = _metric_maxima(phase)
        phase_report = phase_metrics(phase, theta)["channels"][0]
        families["constant_mod3_offsets"].append(
            {
                "amplitude_grid_steps": multiplier,
                "amplitude_normalized": amplitude,
                "metrics": phase_metric,
                "phase": phase_report,
                "detected_by_hard_contract": phase_metric["adjacent_step_max"]
                > hard_thresholds["adjacent_step_max"],
            }
        )

        spike = base.copy()
        spike[73, 0, 0] += amplitude
        spike_metric = _metric_maxima(spike)
        spike_report = phase_metrics(spike, theta)["channels"][0]
        families["isolated_spike"].append(
            {
                "amplitude_grid_steps": multiplier,
                "amplitude_normalized": amplitude,
                "metrics": spike_metric,
                "phase": spike_report,
                "detected_by_hard_contract": spike_metric["second_difference_max"]
                > hard_thresholds["second_difference_max"],
            }
        )

    for rows in families.values():
        detected = [row["amplitude_grid_steps"] for row in rows if any(
            bool(value) for key, value in row.items() if key.startswith("detected_by_")
        )]
        for row in rows:
            row["minimum_detected_amplitude_grid_steps"] = (
                min(detected) if detected else None
            )
    return families


def _heatmap_semantic_assertions() -> dict[str, Any]:
    blank = np.full((3, 1, HEATMAP_SIZE, HEATMAP_SIZE), -20.0, dtype=np.float64)

    mass_shift = blank.copy()
    mass_shift[:, 0, 32, 32] = 10.0
    mass_shift[:, 0, 32, 40] = np.asarray((2.0, 8.0, 9.5))
    mass = heatmap_topology(mass_shift)
    mass_hard_stable = bool(np.unique(mass["hard_x_cell"][:, 0]).size == 1)
    mass_soft_moved = bool(np.ptp(mass["soft_coordinate"][:, 0, 0]) > GRID_STEP)

    hop = blank.copy()
    hop[:, 0, 32, 32] = np.asarray((10.0, 10.0, 8.0))
    hop[:, 0, 32, 40] = np.asarray((8.0, 9.5, 10.0))
    hopped = heatmap_topology(hop)
    hard_hop_detected = bool(np.unique(hopped["hard_x_cell"][:, 0]).size > 1)

    sharp = blank.copy()
    for index, x_cell in enumerate((28, 32, 36)):
        sharp[index, 0, 30, x_cell] = 15.0
    moving = heatmap_topology(sharp)
    hard_soft_together = bool(
        np.max(
            np.abs(moving["hard_coordinate"][:, 0] - moving["soft_coordinate"][:, 0])
        )
        < 1e-8
        and np.unique(moving["hard_x_cell"][:, 0]).size == 3
    )
    return {
        "stable_hard_moving_soft_mass_shift": mass_hard_stable and mass_soft_moved,
        "hard_basin_hop_detected": hard_hop_detected,
        "sharp_hard_and_soft_move_together": hard_soft_together,
    }


def build_calibration() -> dict[str, Any]:
    geometry = geometry_smoke()
    clean_coordinate = _clean_coordinate_oracles()
    clean_gaussian = _clean_gaussian_oracles()

    # Thresholds are the next representable float above the deterministic
    # model-blind clean envelope.  There is no tuned percentage or inspection
    # of model outcomes.  Separate hard and soft envelopes remain distinct.
    hard_thresholds = {
        key: float(np.nextafter(value, np.inf))
        for key, value in clean_coordinate["r64_hard_quantized_envelope"].items()
    }
    soft_thresholds = {
        key: float(np.nextafter(value, np.inf))
        for key, value in clean_gaussian["soft_single_gaussian_envelope"].items()
    }
    topology_thresholds = {
        key: float(np.nextafter(value, -np.inf if key.endswith("_min") else np.inf))
        for key, value in clean_gaussian["single_gaussian_topology_envelope"].items()
    }
    ladders = _planted_detection_ladders(hard_thresholds, soft_thresholds)
    heatmap_assertions = _heatmap_semantic_assertions()

    theta = np.arange(FRAME_COUNT, dtype=np.float64) * FRAME_STEP_DEG
    base = np.broadcast_to(np.asarray((0.4, -0.1)), (FRAME_COUNT, 1, 2)).copy()
    phase = base.copy()
    phase[:, 0, 0] += np.asarray((GRID_STEP, -0.5 * GRID_STEP, -0.5 * GRID_STEP))[
        np.arange(FRAME_COUNT) % 3
    ]
    phase_row = phase_metrics(phase, theta)["channels"][0]
    spike = base.copy()
    spike[73, 0, 0] += 4.0 * GRID_STEP
    spike_row = phase_metrics(spike, theta)["channels"][0]

    # Equal adjacent-step energy from a fixed residue pattern and a non-phase
    # sinusoid defines the preregistered 50% "phase-dominant" boundary.
    equal_mix = base.copy()
    equal_mix[:, 0, 0] += np.asarray((GRID_STEP, -0.5 * GRID_STEP, -0.5 * GRID_STEP))[
        np.arange(FRAME_COUNT) % 3
    ]
    nonphase = np.sin(2.0 * np.pi * np.arange(FRAME_COUNT) / 11.0)
    phase_only_steps = np.diff(equal_mix[:, 0, 0])
    nonphase_steps = np.diff(nonphase)
    nonphase_scale = math.sqrt(
        float(np.sum(np.square(phase_only_steps)))
        / float(np.sum(np.square(nonphase_steps)))
    )
    equal_mix[:, 0, 1] += nonphase_scale * nonphase
    equal_mix_phase_fraction = phase_metrics(equal_mix, theta)["channels"][0][
        "adjacent_step_energy_removed_by_constant_residue_correction"
    ]

    small_radius = 0.5 * GRID_STEP
    small_canonical = np.broadcast_to(
        np.asarray((small_radius, 0.0)), (FRAME_COUNT, 1, 2)
    ).copy()
    small_raw = raw_from_canonical(small_canonical, theta)
    static_raw = np.zeros_like(small_raw)
    raw_rms = lambda value: float(np.sqrt(np.mean(np.sum(
        np.square(value - np.mean(value, axis=0, keepdims=True)), axis=-1
    ))))
    small_activity = raw_rms(small_raw)
    static_activity = raw_rms(static_raw)

    assertions = {
        "exact_derotation_numeric": geometry["correct_sign_rms_normalized"] < 1e-14,
        "wrong_sign_fails": geometry["wrong_sign_rms_normalized"] > 0.1,
        "wrong_pivot_fails": geometry["wrong_pivot_rms_normalized"] > 0.05,
        "small_radius_rotating_is_distinguishable_from_static_centre": (
            small_activity > 0.0 and static_activity == 0.0
        ),
        "constant_phase_is_substantially_removed": (
            phase_row["adjacent_step_energy_removed_by_constant_residue_correction"] > 0.999
            and phase_row["maximum_spike_excess_over_phase_bound"] <= 1e-12
        ),
        "isolated_spike_exceeds_phase_prediction": (
            spike_row["maximum_spike_excess_over_phase_bound"] > 2.0 * GRID_STEP
        ),
        "equal_phase_nonphase_step_energy_calibrates_half_boundary": (
            0.49 <= equal_mix_phase_fraction <= 0.51
        ),
        **heatmap_assertions,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "model_blind_planted_oracle_calibration",
        "model_checkpoint_opened": False,
        "training_or_weight_update_performed": False,
        "frame_count": FRAME_COUNT,
        "frame_step_deg": FRAME_STEP_DEG,
        "heatmap_size": HEATMAP_SIZE,
        "grid_step_normalized": GRID_STEP,
        "amplitude_multipliers_grid_steps": list(AMPLITUDE_MULTIPLIERS),
        "geometry": geometry,
        "clean_coordinate_oracles": clean_coordinate,
        "clean_gaussian_heatmap_oracles": clean_gaussian,
        "frozen_thresholds": {
            "derivation": "next representable float above the deterministic model-blind clean envelope",
            "hard_r64_quantized": hard_thresholds,
            "soft_single_gaussian": soft_thresholds,
            "single_gaussian_heatmap_topology": topology_thresholds,
            "phase_dominance": {
                "minimum_adjacent_step_energy_fraction": 0.5,
                "derivation": "equal planted adjacent-step energy from constant residue offsets and an orthogonal non-phase sinusoid",
                "observed_equal_mix_fraction": equal_mix_phase_fraction,
            },
            "activity": {
                "minimum_observable_radius_grid_steps": 0.5,
                "minimum_raw_orbit_rms_normalized": small_activity,
                "static_centre_raw_orbit_rms_normalized": static_activity,
            },
            "grounding_and_distinctness": {
                "required_on_object_rate": 1.0,
                "minimum_image_border_distance_px": 0.5 * GRID_STEP * 0.5 * 511.0,
                "minimum_fixed_channel_pair_distance_normalized": GRID_STEP,
                "persistent_duplicate_close_fraction": 0.5,
                "definitions": {
                    "on_object": "nearest endpoint-aligned pixel lies inside the bound binary object mask",
                    "border": "at least half one r64 grid cell from every image boundary",
                    "persistent_duplicate": "fixed-channel pair is closer than one r64 grid cell for at least half the orbit",
                },
            },
        },
        "planted_detection_ladders": ladders,
        "semantic_assertions": assertions,
        "all_semantic_assertions_pass": all(assertions.values()),
        "statistical_scope": {
            "status": "deterministic descriptive calibration",
            "sample_unit": "planted analytic trajectory or heatmap case",
            "correlation_caveat": "frames and overlapping differences are correlated; no inferential interval is reported",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = build_calibration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "output": str(args.output.resolve()),
        "file_sha256": _sha256(args.output),
        "content_hash_sha256": calibration["content_hash_sha256"],
        "all_semantic_assertions_pass": calibration["all_semantic_assertions_pass"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not calibration["all_semantic_assertions_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
