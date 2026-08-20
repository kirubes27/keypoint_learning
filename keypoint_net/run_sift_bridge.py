#!/usr/bin/env python3
"""Run the frozen two-stage SIFT bridge on the 180-frame hammer roll.

Stage ``infer`` reads RGB only, fits identities on frames 27..176, predicts
every frame independently, writes deterministic raw arrays, and hashes them.
Stage ``evaluate`` is a separate process that may then load masks and physical
theta solely to score the already-frozen raw predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from keypoint_net.sift_bridge import (
    FrozenSiftBridge,
    SiftBridgeConfig,
    SiftBridgeError,
    SiftDetections,
    config_as_dict,
    create_detector,
    detect_rgb,
    fit_from_detections,
    predict_from_detections,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ID = "engineers_hammer_vray"
FRAME_COUNT = 180
TRAIN_FRAMES = tuple(range(27, 177))
HOLDOUT_FRAMES = tuple(range(0, 24))
GUARD_FRAMES = (24, 25, 26, 177, 178, 179)
IDENTITY_COLORS = plt.get_cmap("tab10")(np.arange(10))


class RunnerError(RuntimeError):
    """Raised when provenance or stage ordering would invalidate the run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def strict_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnerError(f"invalid strict JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON top level is not an object: {path}")
    return value


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite artifact: {path}")
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create timestamp-independent, uncompressed NPZ bytes."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            require(name and "/" not in name, f"invalid NPZ field name: {name!r}")
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def write_npz_no_overwrite(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite artifact: {path}")
    payload = deterministic_npz_bytes(arrays)
    with path.open("xb") as handle:
        handle.write(payload)


def git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"git command failed: git {' '.join(args)}") from exc


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == (512, 512, 3), f"unexpected RGB shape: {path}: {rgb.shape}")
    return rgb


def frame_paths(dataset_root: Path) -> tuple[Path, ...]:
    root = dataset_root / "train" / OBJECT_ID / "frames" / "a"
    paths = tuple(root / f"img_{index:04d}.png" for index in range(FRAME_COUNT))
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, f"missing RGB frames: {missing[:3]}")
    return paths


def config_from_lock(lock: Mapping[str, Any]) -> SiftBridgeConfig:
    bridge = lock["bridge"]
    sift = bridge["sift"]
    matching = bridge["matching"]
    config = SiftBridgeConfig(
        n_identities=int(bridge["n_identities"]),
        seed_frame_index=int(bridge["seed_frame_index"]),
        seed_separation_px=float(bridge["seed_separation_px"]),
        lowe_ratio=float(matching["lowe_ratio"]),
        nfeatures=int(sift["nfeatures"]),
        n_octave_layers=int(sift["nOctaveLayers"]),
        contrast_threshold=float(sift["contrastThreshold"]),
        edge_threshold=float(sift["edgeThreshold"]),
        sigma=float(sift["sigma"]),
    )
    config.validate()
    require(config == SiftBridgeConfig(), "config lock differs from R1 code defaults")
    require(lock["schema"] == "frozen_sift_bridge_config.r1", "wrong config schema")
    return config


def exact_equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def infer_stage(dataset_root: Path, config_path: Path, output_root: Path) -> None:
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    lock = strict_json(config_path)
    config = config_from_lock(lock)
    paths = frame_paths(dataset_root)

    started = time.time()
    detector = create_detector(config)
    images = [load_rgb(path) for path in paths]
    detections = [detect_rgb(image, detector) for image in images]

    # A second fresh detector must reproduce the seed observation exactly.
    repeated_seed = detect_rgb(images[config.seed_frame_index], create_detector(config))
    seed = detections[config.seed_frame_index]
    detector_repeatable = all(
        exact_equal_with_nan(np.asarray(left), np.asarray(right))
        for left, right in (
            (seed.xy_px, repeated_seed.xy_px),
            (seed.descriptors, repeated_seed.descriptors),
            (seed.response, repeated_seed.response),
            (seed.size, repeated_seed.size),
            (seed.angle_deg, repeated_seed.angle_deg),
        )
    )
    require(detector_repeatable, "fresh SIFT seed inference is not exactly repeatable")

    train_detections = {index: detections[index] for index in TRAIN_FRAMES}
    model = fit_from_detections(train_detections, TRAIN_FRAMES, config)

    def predict(order: Iterable[int]) -> dict[str, np.ndarray]:
        coordinates = np.full((FRAME_COUNT, config.n_identities, 2), np.nan, dtype=np.float64)
        accepted = np.zeros((FRAME_COUNT, config.n_identities), dtype=bool)
        detection_index = np.full((FRAME_COUNT, config.n_identities), -1, dtype=np.int64)
        distance = np.full((FRAME_COUNT, config.n_identities), np.inf, dtype=np.float64)
        row_ratio = np.full((FRAME_COUNT, config.n_identities), np.inf, dtype=np.float64)
        column_ratio = np.full((FRAME_COUNT, config.n_identities), np.inf, dtype=np.float64)
        mutual = np.zeros((FRAME_COUNT, config.n_identities), dtype=bool)
        for frame in order:
            frame_coordinates, assignment = predict_from_detections(model, detections[frame])
            coordinates[frame] = frame_coordinates
            accepted[frame] = assignment.accepted
            detection_index[frame] = assignment.detection_index
            distance[frame] = assignment.distance
            row_ratio[frame] = assignment.row_ratio
            column_ratio[frame] = assignment.column_ratio
            mutual[frame] = assignment.mutual_nearest
        return {
            "coordinate_px": coordinates,
            "accepted": accepted,
            "detection_index": detection_index,
            "distance": distance,
            "row_ratio": row_ratio,
            "column_ratio": column_ratio,
            "mutual_nearest": mutual,
        }

    forward = predict(range(FRAME_COUNT))
    reverse = predict(reversed(range(FRAME_COUNT)))
    require(
        all(exact_equal_with_nan(forward[name], reverse[name]) for name in forward),
        "predictions depend on frame processing order",
    )

    bank_offsets = [0]
    for bank in model.descriptor_banks:
        bank_offsets.append(bank_offsets[-1] + bank.shape[0])
    bank_descriptors = np.concatenate(model.descriptor_banks, axis=0)
    arrays = {
        **forward,
        "frame_index": np.arange(FRAME_COUNT, dtype=np.int64),
        "detection_count": np.asarray([value.xy_px.shape[0] for value in detections], dtype=np.int64),
        "seed_candidate_index": model.seed_candidate_indices,
        "seed_xy_px": model.seed_xy_px,
        "seed_response": model.seed_response,
        "train_coverage": model.train_coverage,
        "train_median_ratio": model.train_median_ratio,
        "bank_offset": np.asarray(bank_offsets, dtype=np.int64),
        "bank_descriptor": bank_descriptors,
    }
    raw_path = output_root / "RAW_SIFT_PREDICTIONS.npz"
    raw_payload = deterministic_npz_bytes(arrays)
    require(raw_payload == deterministic_npz_bytes(arrays), "deterministic NPZ encoding failed")
    with raw_path.open("xb") as handle:
        handle.write(raw_payload)
    raw_hash, raw_size = sha256_file(raw_path)

    input_hashes = []
    for frame, path in enumerate(paths):
        digest, size = sha256_file(path)
        input_hashes.append(
            {
                "frame_index": frame,
                "relative_path": str(path.relative_to(dataset_root)),
                "sha256": digest,
                "size_bytes": size,
            }
        )

    template_lock = {
        "schema": "frozen_sift_template_lock.r1",
        "config": config_as_dict(model.config),
        "train_frame_indices": list(model.train_frame_indices),
        "identity_rows": [
            {
                "identity": identity,
                "label": f"SIFT{identity}",
                "seed_candidate_index": int(model.seed_candidate_indices[identity]),
                "seed_xy_px": model.seed_xy_px[identity].tolist(),
                "seed_response": float(model.seed_response[identity]),
                "train_coverage": float(model.train_coverage[identity]),
                "train_median_ratio": float(model.train_median_ratio[identity]),
                "descriptor_bank_size": int(model.descriptor_banks[identity].shape[0]),
            }
            for identity in range(config.n_identities)
        ],
        "selection_information": "train RGB and descriptor matches only",
        "templates_frozen_before_evaluation": True,
    }
    write_json_no_overwrite(output_root / "RAW_SIFT_TEMPLATE_LOCK.json", template_lock)

    source_files = [
        REPO_ROOT / "keypoint_net" / "sift_bridge.py",
        REPO_ROOT / "keypoint_net" / "run_sift_bridge.py",
        config_path,
    ]
    receipt = {
        "schema": "raw_sift_inference_receipt.r1",
        "stage": "rgb_only_inference_complete",
        "completed_unix_seconds": time.time(),
        "duration_seconds": time.time() - started,
        "source": {
            "git_head": git(["rev-parse", "HEAD"]),
            "git_branch": git(["branch", "--show-current"]),
            "files": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path)[0],
                    "size_bytes": sha256_file(path)[1],
                }
                for path in source_files
            ],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "raw_predictions": {
            "path": str(raw_path),
            "sha256": raw_hash,
            "size_bytes": raw_size,
            "shape": [FRAME_COUNT, config.n_identities, 2],
            "accepted_per_identity": np.sum(forward["accepted"], axis=0).astype(int).tolist(),
            "missing_per_identity": np.sum(~forward["accepted"], axis=0).astype(int).tolist(),
        },
        "detector_repeatable_on_fresh_seed_call": detector_repeatable,
        "frame_order_invariance_exact": True,
        "missing_predictions_filled": False,
        "forbidden_inputs_loaded": [],
        "rgb_inputs": input_hashes,
    }
    write_json_no_overwrite(output_root / "RAW_SIFT_RECEIPT.json", receipt)
    print(json.dumps({"status": "inference_complete", "raw_sha256": raw_hash}, sort_keys=True))


def load_meta(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "train" / OBJECT_ID / "meta.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"invalid meta JSON line {line_number}") from exc
            require(isinstance(row, dict), "meta row is not an object")
            rows.append(row)
    require(len(rows) == FRAME_COUNT, f"expected {FRAME_COUNT} metadata rows")
    require([int(row["frame_index"]) for row in rows] == list(range(FRAME_COUNT)), "meta order differs")
    return rows


def pixel_to_normalized(points: np.ndarray, width: int, height: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    result[..., 0] = 2.0 * result[..., 0] / float(width - 1) - 1.0
    result[..., 1] = 2.0 * result[..., 1] / float(height - 1) - 1.0
    return result


def rotate_normalized(points: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(theta_deg)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    x = cosine * points[..., 0] - sine * points[..., 1]
    y = sine * points[..., 0] + cosine * points[..., 1]
    return np.stack((x, y), axis=-1)


def partition_metrics(
    name: str,
    indices: Sequence[int],
    accepted: np.ndarray,
    on_object: np.ndarray,
    material_error_px: np.ndarray,
    canonical: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    rows = []
    for identity in range(accepted.shape[1]):
        valid = accepted[selected, identity]
        errors = material_error_px[selected, identity][valid]
        points = canonical[selected, identity][valid]
        missing = selected[~valid]

        adjacent = []
        second = []
        index_set = set(int(index) for index in selected.tolist())
        for frame in selected.tolist():
            if frame + 1 in index_set and accepted[frame, identity] and accepted[frame + 1, identity]:
                adjacent.append(float(np.linalg.norm(canonical[frame + 1, identity] - canonical[frame, identity]) * 255.5))
            if (
                frame + 2 in index_set
                and accepted[frame, identity]
                and accepted[frame + 1, identity]
                and accepted[frame + 2, identity]
            ):
                value = canonical[frame + 2, identity] - 2.0 * canonical[frame + 1, identity] + canonical[frame, identity]
                second.append(float(np.linalg.norm(value) * 255.5))

        def summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                return {"n": 0, "mean": None, "p95": None, "max": None}
            return {
                "n": int(array.size),
                "mean": float(np.mean(array)),
                "p95": float(np.quantile(array, 0.95)),
                "max": float(np.max(array)),
            }

        canonical_rms = None
        canonical_radius = None
        if points.shape[0] > 0:
            centre = np.mean(points, axis=0, keepdims=True)
            radii = np.linalg.norm(points - centre, axis=1) * 255.5
            canonical_rms = float(np.sqrt(np.mean(np.square(radii))))
            canonical_radius = float(np.max(radii))
        rows.append(
            {
                "identity": identity,
                "frame_count": int(selected.size),
                "accepted_count": int(np.count_nonzero(valid)),
                "coverage": float(np.mean(valid)),
                "missing_frames": missing.astype(int).tolist(),
                "on_object_accepted_rate": (
                    float(np.mean(on_object[selected, identity][valid])) if np.any(valid) else None
                ),
                "on_object_all_frame_rate": float(np.mean(on_object[selected, identity] & valid)),
                "material_error_px": summary(errors),
                "canonical_rms_about_mean_px": canonical_rms,
                "canonical_radius_about_mean_px": canonical_radius,
                "adjacent_canonical_step_px": summary(adjacent),
                "canonical_second_difference_px": summary(second),
            }
        )
    return {
        "name": name,
        "frame_indices": list(indices),
        "statistics": {
            "status": "descriptive",
            "sample_unit": "frames, adjacent frame pairs, or overlapping frame triples per fixed identity",
            "dependence": "adjacent steps and second differences are temporally correlated and overlapping",
            "uncertainty": "none; no SEM or population CI",
        },
        "identities": rows,
    }


def save_figure_no_overwrite(path: Path, figure: plt.Figure) -> None:
    require(not path.exists(), f"refusing to overwrite figure: {path}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_visualizations(
    output_root: Path,
    images: Sequence[np.ndarray],
    coordinates_px: np.ndarray,
    expected_raw_px: np.ndarray,
    canonical: np.ndarray,
    reference_canonical: np.ndarray,
    material_error_px: np.ndarray,
    accepted: np.ndarray,
) -> list[dict[str, Any]]:
    visual_root = output_root / "visualizations"
    visual_root.mkdir(parents=True, exist_ok=False)
    artifacts = []

    selected_frames = (0, 24, 27, 45, 90, 135, 176, 179)
    figure, axes = plt.subplots(2, 4, figsize=(16, 9))
    for axis, frame in zip(axes.flat, selected_frames):
        axis.imshow(images[frame])
        for identity in range(10):
            if accepted[frame, identity]:
                x, y = coordinates_px[frame, identity]
                axis.scatter(x, y, s=38, color=IDENTITY_COLORS[identity], edgecolors="black", linewidths=0.5)
                axis.text(x + 3, y - 3, str(identity), color="white", fontsize=7, bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})
        axis.set_title(f"frame {frame}")
        axis.axis("off")
    figure.suptitle("Frozen SIFT identities: independent per-frame redetection")
    path = visual_root / "frame_overlays.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "raw independent detections on fixed frames"})

    figure, axis = plt.subplots(figsize=(9, 9))
    for identity in range(10):
        axis.plot(canonical[:, identity, 0], canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=1.0, alpha=0.8, label=f"SIFT{identity}")
        axis.scatter(reference_canonical[identity, 0], reference_canonical[identity, 1], marker="*", s=100, color=IDENTITY_COLORS[identity], edgecolors="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("canonical x (endpoint-normalized)")
    axis.set_ylabel("canonical y (endpoint-normalized)")
    axis.set_title("Physically de-rotated material trajectories (missing matches break lines)")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    path = visual_root / "canonical_trajectories.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "material sliding and wobble after exact physical de-rotation"})

    figure, axis = plt.subplots(figsize=(14, 6))
    for identity in range(10):
        axis.plot(material_error_px[:, identity], color=IDENTITY_COLORS[identity], linewidth=1.0, label=f"SIFT{identity}")
    axis.axvspan(0, 23, color="tab:blue", alpha=0.07, label="holdout")
    axis.axvspan(24, 26, color="tab:orange", alpha=0.10, label="guard")
    axis.axvspan(27, 176, color="tab:green", alpha=0.04, label="train")
    axis.axvspan(177, 179, color="tab:orange", alpha=0.10)
    axis.set_xlabel("frame")
    axis.set_ylabel("error from frozen seed material point (pixels)")
    axis.set_title("Per-identity material error; no temporal filling")
    axis.grid(alpha=0.25)
    axis.legend(ncol=5, fontsize=7)
    path = visual_root / "material_error_by_frame.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "absolute material identity error versus frozen train seed"})

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for identity in range(10):
        axes[0].plot(canonical[:, identity, 0], color=IDENTITY_COLORS[identity], linewidth=0.9, label=f"SIFT{identity}")
        axes[1].plot(canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=0.9)
    axes[0].set_ylabel("canonical x")
    axes[1].set_ylabel("canonical y")
    axes[1].set_xlabel("frame")
    axes[0].set_title("Canonical coordinates over time")
    axes[0].legend(ncol=5, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    path = visual_root / "canonical_coordinates_over_time.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "adjacent zig-zag and slow drift in each coordinate"})

    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    worst_rows = []
    for identity, axis in enumerate(axes.flat):
        missing = np.flatnonzero(~accepted[:, identity])
        if missing.size:
            frame = int(missing[0])
            reason = "first missing"
        else:
            frame = int(np.nanargmax(material_error_px[:, identity]))
            reason = "maximum material error"
        axis.imshow(images[frame])
        expected_x, expected_y = expected_raw_px[frame, identity]
        axis.scatter(expected_x, expected_y, marker="x", s=100, linewidths=2.0, color="lime", label="expected material")
        if accepted[frame, identity]:
            x, y = coordinates_px[frame, identity]
            axis.scatter(x, y, marker="o", s=70, facecolors="none", edgecolors="red", linewidths=2.0, label="SIFT")
            error = float(material_error_px[frame, identity])
        else:
            error = None
        axis.set_title(f"SIFT{identity} f{frame}\n{reason}; error={error}", fontsize=9)
        axis.axis("off")
        worst_rows.append({"identity": identity, "frame": frame, "reason": reason, "material_error_px": error})
    axes.flat[0].legend(loc="lower left", fontsize=7)
    path = visual_root / "worst_identity_events.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "deterministic worst or first-missing event per identity", "selection": worst_rows})
    return artifacts


def evaluate_stage(dataset_root: Path, config_path: Path, output_root: Path) -> None:
    lock = strict_json(config_path)
    _ = config_from_lock(lock)
    calibration_binding = lock["scientific_threshold_binding"]
    require(
        calibration_binding["status"] == "bound_before_raw_sift_inference",
        "scientific calibration was not bound before inference",
    )
    calibration_path = Path(calibration_binding["path"]).resolve(strict=True)
    require(
        sha256_file(calibration_path)[0] == calibration_binding["file_sha256"],
        "scientific calibration file hash differs",
    )
    calibration = strict_json(calibration_path)
    require(calibration["schema"] == calibration_binding["schema"], "calibration schema differs")
    require(
        calibration["content_hash_sha256"] == calibration_binding["content_hash_sha256"],
        "calibration content hash differs",
    )
    require(calibration["all_semantic_assertions_pass"] is True, "calibration assertions failed")
    thresholds = calibration["thresholds"]
    raw_path = output_root / "RAW_SIFT_PREDICTIONS.npz"
    receipt_path = output_root / "RAW_SIFT_RECEIPT.json"
    require(raw_path.is_file() and receipt_path.is_file(), "raw inference stage is incomplete")
    receipt = strict_json(receipt_path)
    raw_hash, raw_size = sha256_file(raw_path)
    require(raw_hash == receipt["raw_predictions"]["sha256"], "raw prediction hash changed")
    require(raw_size == int(receipt["raw_predictions"]["size_bytes"]), "raw prediction size changed")
    require(receipt["forbidden_inputs_loaded"] == [], "raw stage reports a forbidden input")
    require(receipt["missing_predictions_filled"] is False, "raw stage filled missing predictions")
    require(receipt["frame_order_invariance_exact"] is True, "frame order invariance failed")

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    coordinates_px = np.asarray(raw["coordinate_px"], dtype=np.float64)
    accepted = np.asarray(raw["accepted"], dtype=bool)
    require(coordinates_px.shape == (FRAME_COUNT, 10, 2), "raw coordinate shape differs")
    require(accepted.shape == (FRAME_COUNT, 10), "accepted shape differs")
    require(np.array_equal(np.isfinite(coordinates_px).all(axis=-1), accepted), "NaN/accepted semantics differ")

    # Evaluation-only information begins here, in a process separate from inference.
    dataset_index_path = dataset_root / "dataset_index.json"
    train_pair_path = REPO_ROOT / "docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__train.json"
    validation_pair_path = REPO_ROOT / "docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__validation.json"
    for path, expected in (
        (dataset_index_path, lock["dataset"]["dataset_index_sha256"]),
        (train_pair_path, lock["dataset"]["train_pair_index_sha256"]),
        (validation_pair_path, lock["dataset"]["validation_pair_index_sha256"]),
    ):
        require(sha256_file(path)[0] == expected, f"evaluation binding hash differs: {path}")

    meta = load_meta(dataset_root)
    theta = np.asarray([float(row["theta_deg"]) for row in meta], dtype=np.float64)
    require(np.array_equal(theta, np.arange(FRAME_COUNT, dtype=np.float64) * 2.0), "theta metadata differs")
    image_paths = frame_paths(dataset_root)
    images = [load_rgb(path) for path in image_paths]

    normalized = pixel_to_normalized(coordinates_px, 512, 512)
    canonical = np.full_like(normalized, np.nan)
    safe = np.where(accepted[..., None], normalized, 0.0)
    canonical_safe = rotate_normalized(safe, -theta[:, None])
    canonical[accepted] = canonical_safe[accepted]

    seed_xy = np.asarray(raw["seed_xy_px"], dtype=np.float64)
    seed_normalized = pixel_to_normalized(seed_xy, 512, 512)
    reference_canonical = rotate_normalized(
        seed_normalized, np.full(10, -theta[27], dtype=np.float64)
    )
    material_error_px = np.full((FRAME_COUNT, 10), np.nan, dtype=np.float64)
    material_error_px[accepted] = (
        np.linalg.norm(canonical[accepted] - np.broadcast_to(reference_canonical, canonical.shape)[accepted], axis=-1)
        * 255.5
    )
    expected_normalized = rotate_normalized(
        np.broadcast_to(reference_canonical, (FRAME_COUNT, 10, 2)), theta[:, None]
    )
    expected_raw_px = np.empty_like(expected_normalized)
    expected_raw_px[..., 0] = (expected_normalized[..., 0] + 1.0) * 255.5
    expected_raw_px[..., 1] = (expected_normalized[..., 1] + 1.0) * 255.5

    mask_root = dataset_root / "train" / OBJECT_ID / "masks" / "a"
    on_object = np.zeros((FRAME_COUNT, 10), dtype=bool)
    border_distance_px = np.full((FRAME_COUNT, 10), np.nan, dtype=np.float64)
    for frame in range(FRAME_COUNT):
        with Image.open(mask_root / f"mask_{frame:04d}.png") as image:
            mask = np.asarray(image.convert("L")) > 0
        require(mask.shape == (512, 512), f"mask shape differs at frame {frame}")
        for identity in range(10):
            if not accepted[frame, identity]:
                continue
            x, y = coordinates_px[frame, identity]
            xi = int(np.clip(np.rint(x), 0, 511))
            yi = int(np.clip(np.rint(y), 0, 511))
            on_object[frame, identity] = bool(mask[yi, xi])
            border_distance_px[frame, identity] = float(min(x, y, 511.0 - x, 511.0 - y))

    pairwise_min_px = np.full(FRAME_COUNT, np.nan, dtype=np.float64)
    pairwise_argmin = np.full((FRAME_COUNT, 2), -1, dtype=np.int64)
    for frame in range(FRAME_COUNT):
        valid_identities = np.flatnonzero(accepted[frame])
        if valid_identities.size < 2:
            continue
        points = coordinates_px[frame, valid_identities]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances[np.diag_indices_from(distances)] = np.inf
        flat = int(np.argmin(distances))
        first, second = np.unravel_index(flat, distances.shape)
        pairwise_min_px[frame] = float(distances[first, second])
        pairwise_argmin[frame] = [int(valid_identities[first]), int(valid_identities[second])]

    partitions = [
        partition_metrics("train", TRAIN_FRAMES, accepted, on_object, material_error_px, canonical),
        partition_metrics("holdout", HOLDOUT_FRAMES, accepted, on_object, material_error_px, canonical),
        partition_metrics("guard", GUARD_FRAMES, accepted, on_object, material_error_px, canonical),
    ]
    seam_rows = []
    for identity in range(10):
        seam_valid = bool(accepted[179, identity] and accepted[0, identity])
        seam_rows.append(
            {
                "identity": identity,
                "valid": seam_valid,
                "canonical_step_px": (
                    float(np.linalg.norm(canonical[0, identity] - canonical[179, identity]) * 255.5)
                    if seam_valid
                    else None
                ),
            }
        )

    complete_identity = np.all(accepted, axis=0)
    raw_gate_failures = []
    if not bool(np.all(complete_identity)):
        raw_gate_failures.append("one_or_more_identities_missing_in_at_least_one_frame")
    if not bool(np.all(on_object[accepted])):
        raw_gate_failures.append("one_or_more_accepted_points_off_object")
    border_minimum = np.asarray(
        [
            np.nanmin(border_distance_px[:, identity])
            if np.any(accepted[:, identity])
            else -np.inf
            for identity in range(10)
        ],
        dtype=np.float64,
    )
    if not bool(
        np.all(border_minimum >= thresholds["minimum_image_border_distance_px"])
    ):
        raw_gate_failures.append("one_or_more_identities_enter_calibrated_border_band")
    minimum_pairwise = float(np.nanmin(pairwise_min_px))
    if minimum_pairwise < thresholds["minimum_fixed_identity_pair_distance_px"]:
        raw_gate_failures.append("one_or_more_frames_violate_calibrated_identity_separation")

    identity_decisions = []
    for identity in range(10):
        reasons: list[str] = []
        reference_radius_px = float(np.linalg.norm(reference_canonical[identity]) * 255.5)
        if reference_radius_px < thresholds["minimum_reference_orbit_radius_px"]:
            reasons.append("reference_point_inside_static_centre_escape_radius")
        if border_minimum[identity] < thresholds["minimum_image_border_distance_px"]:
            reasons.append("border_distance_below_calibrated_minimum")
        for partition in partitions:
            row = partition["identities"][identity]
            prefix = partition["name"]
            if row["coverage"] < thresholds["required_coverage"]:
                reasons.append(f"{prefix}:incomplete_independent_redetection")
            if row["on_object_all_frame_rate"] < thresholds["required_on_object_rate"]:
                reasons.append(f"{prefix}:grounding_failure")
            material_max = row["material_error_px"]["max"]
            if material_max is None or material_max > thresholds["maximum_material_error_px"]:
                reasons.append(f"{prefix}:material_error_exceeds_full_resolution_oracle")
            canonical_rms = row["canonical_rms_about_mean_px"]
            if (
                canonical_rms is None
                or canonical_rms > thresholds["maximum_canonical_rms_about_mean_px"]
            ):
                reasons.append(f"{prefix}:canonical_rms_exceeds_full_resolution_oracle")
            canonical_radius = row["canonical_radius_about_mean_px"]
            if (
                canonical_radius is None
                or canonical_radius > thresholds["maximum_canonical_radius_about_mean_px"]
            ):
                reasons.append(f"{prefix}:canonical_radius_exceeds_full_resolution_oracle")
            adjacent_max = row["adjacent_canonical_step_px"]["max"]
            if (
                adjacent_max is None
                or adjacent_max > thresholds["maximum_adjacent_canonical_step_px"]
            ):
                reasons.append(f"{prefix}:adjacent_wobble_exceeds_full_resolution_oracle")
            second_max = row["canonical_second_difference_px"]["max"]
            if (
                second_max is None
                or second_max > thresholds["maximum_canonical_second_difference_px"]
            ):
                reasons.append(f"{prefix}:second_difference_exceeds_full_resolution_oracle")
        seam = seam_rows[identity]
        if not seam["valid"]:
            reasons.append("seam:missing_identity")
        elif seam["canonical_step_px"] > thresholds["maximum_seam_canonical_step_px"]:
            reasons.append("seam:canonical_step_exceeds_full_resolution_oracle")
        identity_decisions.append(
            {
                "identity": identity,
                "reference_orbit_radius_px": reference_radius_px,
                "pass": len(reasons) == 0,
                "reasons": reasons,
            }
        )

    if minimum_pairwise < thresholds["minimum_fixed_identity_pair_distance_px"]:
        frame = int(np.nanargmin(pairwise_min_px))
        for identity in pairwise_argmin[frame].tolist():
            if identity >= 0:
                identity_decisions[identity]["pass"] = False
                identity_decisions[identity]["reasons"].append(
                    f"frame_{frame}:identity_separation_below_calibrated_minimum"
                )
    numerical_contract_pass = bool(
        not raw_gate_failures and all(row["pass"] for row in identity_decisions)
    )

    visuals = create_visualizations(
        output_root,
        images,
        coordinates_px,
        expected_raw_px,
        canonical,
        reference_canonical,
        material_error_px,
        accepted,
    )
    metrics = {
        "schema": "sift_bridge_evaluation_metrics.r1",
        "raw_prediction_sha256": raw_hash,
        "calibration_binding": {
            "path": str(calibration_path),
            "file_sha256": calibration_binding["file_sha256"],
            "content_hash_sha256": calibration_binding["content_hash_sha256"],
            "thresholds": thresholds,
        },
        "evaluation_only_inputs_loaded_after_raw_hash": [
            "physical theta metadata",
            "object masks",
            "dataset and split bindings",
        ],
        "coordinate_convention": {
            "raw": "OpenCV pixel x,y",
            "normalized": "endpoint-aligned [-1,1]",
            "canonical": "physical R(-theta) around normalized pivot (0,0)",
            "pixel_scale_for_normalized_distance": 255.5,
        },
        "partition_metrics": partitions,
        "seam_179_to_0": seam_rows,
        "full_orbit": {
            "accepted_per_identity": np.sum(accepted, axis=0).astype(int).tolist(),
            "missing_per_identity": np.sum(~accepted, axis=0).astype(int).tolist(),
            "on_object_accepted_rate_per_identity": [
                float(np.mean(on_object[:, identity][accepted[:, identity]]))
                if np.any(accepted[:, identity])
                else None
                for identity in range(10)
            ],
            "minimum_border_distance_px_per_identity": border_minimum.tolist(),
            "minimum_pairwise_identity_distance_px": minimum_pairwise,
            "minimum_pairwise_frame": int(np.nanargmin(pairwise_min_px)),
            "minimum_pairwise_identity_pair": pairwise_argmin[int(np.nanargmin(pairwise_min_px))].tolist(),
            "maximum_material_error_px_per_identity": [
                float(np.nanmax(material_error_px[:, identity]))
                if np.any(accepted[:, identity])
                else None
                for identity in range(10)
            ],
        },
        "raw_categorical_failures_before_scientific_tolerances": raw_gate_failures,
        "identity_decisions": identity_decisions,
        "numerical_contract_pass": numerical_contract_pass,
        "scientific_threshold_status": lock["scientific_threshold_binding"],
        "visualizations": visuals,
        "statistical_language": "descriptive; frames and overlapping differences are correlated; no SEM or population CI",
    }
    metrics_path = output_root / "SIFT_BRIDGE_EVALUATION_METRICS.json"
    write_json_no_overwrite(metrics_path, metrics)

    outcome = {
        "schema": "sift_bridge_result.r1",
        "raw_prediction_sha256": raw_hash,
        "metrics_sha256": sha256_file(metrics_path)[0],
        "bridge_numerical_contract_pass": numerical_contract_pass,
        "bridge_contract_pass": False,
        "bridge_contract_status": (
            "numerical_pass_pending_required_visual_audit"
            if numerical_contract_pass
            else "failed_frozen_numerical_contract"
        ),
        "raw_categorical_failures": raw_gate_failures,
        "failed_identity_count": int(
            sum(not row["pass"] for row in identity_decisions)
        ),
        "training_performed": False,
        "gpu_used": False,
        "operator_experiment_performed": False,
        "scientific_claim": "descriptive bridge feasibility on the already-studied hammer orbit only",
    }
    write_json_no_overwrite(output_root / "RESULT.json", outcome)
    print(json.dumps(outcome, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "evaluate"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    config_path = args.config_lock.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    if args.stage == "infer":
        infer_stage(dataset_root, config_path, output_root)
    else:
        require(output_root.is_dir(), f"output root does not exist: {output_root}")
        evaluate_stage(dataset_root, config_path, output_root)


if __name__ == "__main__":
    try:
        main()
    except (RunnerError, SiftBridgeError) as exc:
        raise SystemExit(f"SIFT bridge gate failed: {exc}") from exc
