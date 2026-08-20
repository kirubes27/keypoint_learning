#!/usr/bin/env python3
"""Create post-hash train-only visual diagnostics for ALIKED smoke output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_PACKAGES = REPO_ROOT / "third_party" / "python_pkgs"
if str(ISOLATED_PACKAGES) not in sys.path:
    sys.path.insert(0, str(ISOLATED_PACKAGES))

import matplotlib  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from keypoint_net.run_sift_bridge import (  # noqa: E402
    IDENTITY_COLORS,
    OBJECT_ID,
    TRAIN_FRAMES,
    frame_paths,
    load_meta,
    load_rgb,
    pixel_to_normalized,
    rotate_normalized,
    save_figure_no_overwrite,
    sha256_file,
    strict_json,
    write_json_no_overwrite,
)


class TrainVisualError(RuntimeError):
    """Raised when a smoke visual would not bind to the frozen raw output."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainVisualError(message)


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    config_path = args.config_lock.resolve(strict=True)
    smoke_root = args.smoke_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    require(not output_root.exists(), f"refusing to overwrite output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    config = strict_json(config_path)
    receipt_path = smoke_root / "SMOKE_RECEIPT.json"
    raw_path = smoke_root / "TRAIN_ONLY_SMOKE_RAW.npz"
    receipt = strict_json(receipt_path)
    raw_hash, raw_size = sha256_file(raw_path)
    require(raw_hash == receipt["raw"]["sha256"], "smoke raw hash differs")
    require(raw_size == int(receipt["raw"]["size_bytes"]), "smoke raw size differs")
    require(receipt["forbidden_inputs_loaded"] == [], "smoke loaded a forbidden input")
    require(receipt["holdout_or_guard_rgb_opened"] is False, "smoke opened holdout/guard")
    require(receipt["missing_predictions_filled"] is False, "smoke filled a missing prediction")
    require(receipt["frame_order_invariance_exact"] is True, "smoke order invariance failed")

    calibration_binding = config["scientific_threshold_binding"]
    calibration_path = Path(calibration_binding["path"]).resolve(strict=True)
    require(sha256_file(calibration_path)[0] == calibration_binding["file_sha256"], "calibration hash differs")
    calibration = strict_json(calibration_path)
    require(calibration["content_hash_sha256"] == calibration_binding["content_hash_sha256"], "calibration content differs")
    thresholds = calibration["thresholds"]

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    frames = np.asarray(raw["train_frame_index"], dtype=np.int64)
    coordinates_px = np.asarray(raw["coordinate_px"], dtype=np.float64)
    accepted = np.asarray(raw["accepted"], dtype=bool)
    require(np.array_equal(frames, np.asarray(TRAIN_FRAMES)), "train frame membership differs")
    require(coordinates_px.shape == (len(TRAIN_FRAMES), 10, 2), "coordinate shape differs")
    require(accepted.shape == (len(TRAIN_FRAMES), 10), "accepted shape differs")
    require(np.array_equal(np.isfinite(coordinates_px).all(axis=-1), accepted), "NaN/accepted semantics differ")

    meta = load_meta(dataset_root)
    theta = np.asarray([float(meta[int(frame)]["theta_deg"]) for frame in frames])
    require(np.array_equal(theta, frames.astype(np.float64) * 2.0), "theta differs")
    normalized = pixel_to_normalized(coordinates_px, 512, 512)
    canonical = np.full_like(normalized, np.nan)
    canonical_safe = rotate_normalized(np.where(accepted[..., None], normalized, 0.0), -theta[:, None])
    canonical[accepted] = canonical_safe[accepted]
    seed_xy = np.asarray(raw["seed_xy_px"], dtype=np.float64)
    seed_normalized = pixel_to_normalized(seed_xy, 512, 512)
    reference_canonical = rotate_normalized(seed_normalized, np.full(10, -54.0))
    reference_full = np.broadcast_to(reference_canonical, canonical.shape)
    material_error_px = np.full((len(TRAIN_FRAMES), 10), np.nan, dtype=np.float64)
    material_error_px[accepted] = np.linalg.norm(canonical[accepted] - reference_full[accepted], axis=-1) * 255.5
    expected_normalized = rotate_normalized(reference_full, theta[:, None])
    expected_raw_px = np.empty_like(expected_normalized)
    expected_raw_px[..., 0] = (expected_normalized[..., 0] + 1.0) * 255.5
    expected_raw_px[..., 1] = (expected_normalized[..., 1] + 1.0) * 255.5

    paths = frame_paths(dataset_root)
    images = {int(frame): load_rgb(paths[int(frame)]) for frame in frames}
    on_object = np.zeros_like(accepted)
    mask_root = dataset_root / "train" / OBJECT_ID / "masks" / "a"
    for row, frame in enumerate(frames.tolist()):
        with Image.open(mask_root / f"mask_{frame:04d}.png") as image:
            mask = np.asarray(image.convert("L")) > 0
        for identity in range(10):
            if accepted[row, identity]:
                x, y = coordinates_px[row, identity]
                on_object[row, identity] = bool(mask[int(np.clip(np.rint(y), 0, 511)), int(np.clip(np.rint(x), 0, 511))])

    artifacts: list[dict[str, Any]] = []
    representative = (27, 45, 60, 90, 120, 150, 176)
    figure, axes = plt.subplots(2, 4, figsize=(16, 9))
    for axis, frame in zip(axes.flat, representative):
        row = int(frame - 27)
        axis.imshow(images[frame])
        for identity in range(10):
            ex, ey = expected_raw_px[row, identity]
            axis.scatter(ex, ey, marker="x", s=30, linewidths=1.0, color=IDENTITY_COLORS[identity], alpha=0.45)
            if accepted[row, identity]:
                x, y = coordinates_px[row, identity]
                axis.scatter(x, y, s=38, color=IDENTITY_COLORS[identity], edgecolors="black", linewidths=0.5)
                axis.text(x + 3, y - 3, str(identity), color="white", fontsize=7, bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})
        axis.set_title(f"frame {frame}: {int(np.sum(accepted[row]))}/10 matched")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.suptitle("Train-only direct ALIKED + LightGlue matches; x marks expected material locations")
    path = output_root / "train_frame_overlays.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "actual versus expected material locations on train frames"})

    figure, axis = plt.subplots(figsize=(9, 9))
    for identity in range(10):
        axis.plot(canonical[:, identity, 0], canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=1.0, label=f"ALIKED{identity}")
        axis.scatter(reference_canonical[identity, 0], reference_canonical[identity, 1], marker="*", s=100, color=IDENTITY_COLORS[identity], edgecolors="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("canonical x")
    axis.set_ylabel("canonical y")
    axis.set_title("Train-only physically de-rotated trajectories; gaps are rejected matches")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    path = output_root / "train_canonical_trajectories.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "visible accepted-match material drift and gaps"})

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for identity in range(10):
        axes[0].plot(frames, canonical[:, identity, 0], color=IDENTITY_COLORS[identity], linewidth=0.9, label=f"ALIKED{identity}")
        axes[1].plot(frames, canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=0.9)
    axes[0].set_ylabel("canonical x")
    axes[1].set_ylabel("canonical y")
    axes[1].set_xlabel("frame")
    axes[0].set_title("Train-only canonical coordinates over time")
    axes[0].legend(ncol=5, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    path = output_root / "train_canonical_coordinates_over_time.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "visual sliding and adjacent zig-zag among accepted matches"})

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for identity in range(10):
        axes[0].plot(frames, material_error_px[:, identity], color=IDENTITY_COLORS[identity], linewidth=0.9, label=f"ALIKED{identity}")
    axes[0].axhline(thresholds["maximum_material_error_px"], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("material error (px)")
    axes[0].set_title("Accepted-match material error and frame match count")
    axes[0].legend(ncol=5, fontsize=7)
    axes[1].plot(frames, np.sum(accepted, axis=1), color="tab:red")
    axes[1].axhline(10, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("matched identities / 10")
    axes[1].set_xlabel("frame")
    for axis in axes:
        axis.grid(alpha=0.25)
    path = output_root / "train_material_error_and_coverage.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "material violations and catastrophic coverage collapse"})

    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    worst_rows = []
    for identity, axis in enumerate(axes.flat):
        missing_rows = np.flatnonzero(~accepted[:, identity])
        if missing_rows.size:
            row = int(missing_rows[0])
            reason = "first missing"
            error = None
        else:
            row = int(np.nanargmax(material_error_px[:, identity]))
            reason = "max material error"
            error = float(material_error_px[row, identity])
        frame = int(frames[row])
        axis.imshow(images[frame])
        ex, ey = expected_raw_px[row, identity]
        axis.scatter(ex, ey, marker="x", s=100, linewidths=2.0, color="lime", label="expected")
        if accepted[row, identity]:
            x, y = coordinates_px[row, identity]
            axis.scatter(x, y, marker="o", s=70, facecolors="none", edgecolors="red", linewidths=2.0, label="matched")
        axis.set_title(f"ALIKED{identity} f{frame}\n{reason}; error={error}", fontsize=9)
        axis.axis("off")
        worst_rows.append({"identity": identity, "frame": frame, "reason": reason, "material_error_px": error})
    axes.flat[0].legend(loc="lower left", fontsize=7)
    path = output_root / "train_worst_identity_events.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "first missing or worst accepted event for every identity", "selection": worst_rows})

    rows = []
    for identity in range(10):
        valid = accepted[:, identity]
        adjacent_values = []
        second_values = []
        for row in range(len(TRAIN_FRAMES) - 1):
            if valid[row] and valid[row + 1]:
                adjacent_values.append(np.linalg.norm(canonical[row + 1, identity] - canonical[row, identity]) * 255.5)
        for row in range(len(TRAIN_FRAMES) - 2):
            if valid[row] and valid[row + 1] and valid[row + 2]:
                second_values.append(np.linalg.norm(canonical[row + 2, identity] - 2.0 * canonical[row + 1, identity] + canonical[row, identity]) * 255.5)
        rows.append({
            "identity": identity,
            "accepted_count": int(np.count_nonzero(valid)),
            "missing_count": int(np.count_nonzero(~valid)),
            "coverage": float(np.mean(valid)),
            "on_object_accepted_rate": float(np.mean(on_object[:, identity][valid])) if np.any(valid) else None,
            "material_error_px": summary(material_error_px[:, identity]),
            "adjacent_canonical_step_px": summary(np.asarray(adjacent_values)),
            "canonical_second_difference_px": summary(np.asarray(second_values)),
        })
    metrics = {
        "schema": "aliked_lightglue_train_visual_diagnostic.r1",
        "raw_prediction_sha256": raw_hash,
        "raw_hashed_before_evaluation_inputs": True,
        "evaluation_only_inputs": ["train physical theta", "train masks"],
        "holdout_or_guard_opened": False,
        "identity_rows": rows,
        "strict_train_coverage_feasible": bool(np.all(accepted)),
        "visualizations": artifacts,
        "statistical_language": "descriptive only; frames and overlapping differences are correlated; no SEM or population CI",
        "training_performed": False,
        "gpu_used": False,
    }
    write_json_no_overwrite(output_root / "TRAIN_VISUAL_DIAGNOSTIC.json", metrics)
    print(json.dumps({"status": "complete", "strict_train_coverage_feasible": bool(np.all(accepted)), "raw_sha256": raw_hash}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except TrainVisualError as exc:
        raise SystemExit(f"train visual diagnostic failed: {exc}") from exc
