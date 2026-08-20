"""Model-blind full-resolution coordinate calibration for the SIFT bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class CalibrationError(ValueError):
    """Raised when the bridge calibration cannot be frozen safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rotate(points: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(theta_deg)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    x = cosine * points[..., 0] - sine * points[..., 1]
    y = sine * points[..., 0] + cosine * points[..., 1]
    return np.stack((x, y), axis=-1)


def quantize_full_resolution(points: np.ndarray, size: int = 512) -> np.ndarray:
    pixel = (points + 1.0) * float(size - 1) / 2.0
    pixel = np.rint(pixel)
    return 2.0 * pixel / float(size - 1) - 1.0


def clean_quantization_metrics() -> dict[str, float]:
    theta = np.arange(180, dtype=np.float64) * 2.0
    anchors = []
    for radius in (0.02, 0.08, 0.2, 0.45, 0.7, 0.9):
        for angle in np.arange(0.0, 360.0, 15.0):
            radians = math.radians(float(angle))
            anchors.append([radius * math.cos(radians), radius * math.sin(radians)])
    canonical_reference = np.asarray(anchors, dtype=np.float64)
    canonical = np.broadcast_to(
        canonical_reference,
        (theta.size, canonical_reference.shape[0], 2),
    ).copy()
    raw = rotate(canonical, theta[:, None])
    quantized = quantize_full_resolution(raw)
    recovered = rotate(quantized, -theta[:, None])
    error_px = np.linalg.norm(recovered - canonical, axis=-1) * 255.5
    centre = np.mean(recovered, axis=0, keepdims=True)
    radius_px = np.linalg.norm(recovered - centre, axis=-1) * 255.5
    adjacent_px = np.linalg.norm(np.diff(recovered, axis=0), axis=-1) * 255.5
    second_px = np.linalg.norm(
        recovered[2:] - 2.0 * recovered[1:-1] + recovered[:-2], axis=-1
    ) * 255.5
    seam_px = np.linalg.norm(recovered[0] - recovered[-1], axis=-1) * 255.5
    return {
        "case_count": int(canonical_reference.shape[0]),
        "maximum_material_error_px": float(np.max(error_px)),
        "maximum_radius_about_mean_px": float(np.max(radius_px)),
        "maximum_adjacent_step_px": float(np.max(adjacent_px)),
        "maximum_second_difference_px": float(np.max(second_px)),
        "maximum_seam_step_px": float(np.max(seam_px)),
    }


def planted_two_pixel_spike_metrics() -> dict[str, float]:
    theta = np.arange(180, dtype=np.float64) * 2.0
    reference = np.asarray([[[0.43, -0.19]]], dtype=np.float64)
    canonical = np.broadcast_to(reference, (theta.size, 1, 2)).copy()
    raw = quantize_full_resolution(rotate(canonical, theta[:, None]))
    raw[90, 0, 0] += 2.0 / 255.5
    recovered = rotate(raw, -theta[:, None])
    material = np.linalg.norm(recovered - canonical, axis=-1) * 255.5
    adjacent = np.linalg.norm(np.diff(recovered, axis=0), axis=-1) * 255.5
    second = np.linalg.norm(
        recovered[2:] - 2.0 * recovered[1:-1] + recovered[:-2], axis=-1
    ) * 255.5
    return {
        "maximum_material_error_px": float(np.max(material)),
        "maximum_adjacent_step_px": float(np.max(adjacent)),
        "maximum_second_difference_px": float(np.max(second)),
    }


def build_lock(parent: Mapping[str, Any], *, parent_sha256: str) -> dict[str, Any]:
    require(
        parent.get("schema_version") == "frozen_wobble_oracle_calibration.v1_2",
        "wrong parent calibration schema",
    )
    require(parent.get("all_semantic_assertions_pass") is True, "parent calibration failed")
    frozen = parent["frozen_thresholds"]
    grounding = frozen["grounding_and_distinctness"]
    activity = frozen["activity"]

    root_two = math.sqrt(2.0)
    next_root_two = float(np.nextafter(root_two, math.inf))
    next_two_root_two = float(np.nextafter(2.0 * root_two, math.inf))
    thresholds = {
        "required_coverage": 1.0,
        "required_on_object_rate": float(grounding["required_on_object_rate"]),
        "minimum_reference_orbit_radius_px": float(
            activity["minimum_raw_orbit_rms_normalized"] * 255.5
        ),
        "minimum_image_border_distance_px": float(
            grounding["minimum_image_border_distance_px"]
        ),
        "minimum_fixed_identity_pair_distance_px": float(
            grounding["minimum_fixed_channel_pair_distance_normalized"] * 255.5
        ),
        "maximum_material_error_px": next_root_two,
        "maximum_canonical_rms_about_mean_px": next_root_two,
        "maximum_canonical_radius_about_mean_px": next_root_two,
        "maximum_adjacent_canonical_step_px": next_root_two,
        "maximum_canonical_second_difference_px": next_two_root_two,
        "maximum_seam_canonical_step_px": next_root_two,
    }
    clean = clean_quantization_metrics()
    spike = planted_two_pixel_spike_metrics()
    assertions = {
        "parent_semantics_pass": True,
        "clean_material_error_inside_bound": clean["maximum_material_error_px"]
        <= thresholds["maximum_material_error_px"],
        "clean_radius_inside_bound": clean["maximum_radius_about_mean_px"]
        <= thresholds["maximum_canonical_radius_about_mean_px"],
        "clean_adjacent_inside_bound": clean["maximum_adjacent_step_px"]
        <= thresholds["maximum_adjacent_canonical_step_px"],
        "clean_second_difference_inside_bound": clean[
            "maximum_second_difference_px"
        ]
        <= thresholds["maximum_canonical_second_difference_px"],
        "clean_seam_inside_bound": clean["maximum_seam_step_px"]
        <= thresholds["maximum_seam_canonical_step_px"],
        "two_pixel_spike_rejected_by_material_bound": spike[
            "maximum_material_error_px"
        ]
        > thresholds["maximum_material_error_px"],
        "two_pixel_spike_rejected_by_second_difference_bound": spike[
            "maximum_second_difference_px"
        ]
        > thresholds["maximum_canonical_second_difference_px"],
    }
    require(all(assertions.values()), f"calibration assertion failed: {assertions}")
    payload: dict[str, Any] = {
        "schema": "sift_bridge_eradication_calibration_lock.r1",
        "artifact_type": "model_blind_full_resolution_quantization_and_parent_contract_binding",
        "parent_calibration_sha256": parent_sha256,
        "parent_schema": parent["schema_version"],
        "coordinate_grid": {
            "image_size": 512,
            "endpoint_aligned": True,
            "one_pixel_normalized": 2.0 / 511.0,
        },
        "derivation": {
            "single_frame_localization_error": "each x/y coordinate is at most half a pixel from exact",
            "material_or_adjacent_difference": "difference of two 2D half-pixel error vectors is at most sqrt(2) pixels",
            "second_difference": "e[t+1]-2e[t]+e[t-1] is at most 2*sqrt(2) pixels",
            "threshold_rounding": "next representable float above each analytic upper bound",
        },
        "thresholds": thresholds,
        "clean_quantization_sweep": clean,
        "planted_two_pixel_spike": spike,
        "semantic_assertions": assertions,
        "all_semantic_assertions_pass": True,
        "training_or_weight_update_performed": False,
        "model_checkpoint_opened": False,
        "statistical_scope": {
            "status": "deterministic model-blind calibration",
            "sample_unit": "analytic material point under exact rotation and full-resolution rounding",
            "correlation_caveat": "no inferential interval; frame differences overlap",
        },
    }
    payload["content_hash_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent_calibration.resolve(strict=True)
    output = args.output.resolve(strict=False)
    require(not output.exists(), f"refusing to overwrite calibration: {output}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    lock = build_lock(parent, parent_sha256=sha256_file(parent_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_bytes(lock) + b"\n")
    print(lock["content_hash_sha256"])


if __name__ == "__main__":
    main()
