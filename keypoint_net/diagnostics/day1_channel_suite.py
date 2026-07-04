"""Day-1 per-channel diagnostic suite (no training).

Preregistered interpretations come from DIAGNOSTIC_WEEK_SPEC v1.2:
* duplicate channel: median nearest-neighbour distance < 1 CELL64;
* static channel: activity ratio < 0.2 only when expected motion > 0.5 CELL64;
* spread calibration usable only when both across-channel and median blocked
  within-channel Spearman rho >= 0.6; unusable if either is < 0.3.

All temporal summaries are descriptive for one 180-frame orbit. Frames are
correlated; no population-level inference is claimed. CSV rows retain the
per-channel sample unit (n=180 frames) and calibration uses 12 contiguous
15-frame blocks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from dxutils import (
    CELL64_NORM,
    CELL64_PX,
    COLORS,
    OUTPUTS,
    derotate,
    estimate_rotation_model,
    forward_sequence,
    highpass_residual,
    load_masks,
    load_run,
    overlay,
    run_directories,
    to_px,
    transport,
    write_rotation_report,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    assert rows, path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_masks(masks: np.ndarray, coords_px: np.ndarray) -> np.ndarray:
    xy = np.rint(coords_px).astype(int)
    xy[..., 0] = np.clip(xy[..., 0], 0, masks.shape[2] - 1)
    xy[..., 1] = np.clip(xy[..., 1], 0, masks.shape[1] - 1)
    frames = np.arange(len(masks))[:, None]
    return masks[frames, xy[..., 1], xy[..., 0]]


def _dilate_masks(masks: np.ndarray, radius_px: int = 8) -> np.ndarray:
    tensor = torch.from_numpy(masks.astype(np.float32))[:, None]
    dilated = F.max_pool2d(
        tensor, kernel_size=2 * radius_px + 1, stride=1, padding=radius_px
    )
    return dilated[:, 0].numpy() > 0.5


def _equivariance_errors(
    coords: np.ndarray, rotation, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    expected = transport(coords, hop, rotation)
    observed = np.roll(coords, -hop, axis=0)
    vector_px = to_px(observed) - to_px(expected)
    error_cells = np.linalg.norm(vector_px, axis=-1) / CELL64_PX
    return error_cells, vector_px / CELL64_PX


def _pair_metrics(coords_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = coords_px[:, :, None, :] - coords_px[:, None, :, :]
    distances = np.linalg.norm(difference, axis=-1) / CELL64_PX
    keypoints = distances.shape[1]
    distances[:, np.arange(keypoints), np.arange(keypoints)] = np.inf
    nearest_distance = distances.min(axis=2)
    nearest_index = distances.argmin(axis=2)
    persistent_pair_distance = np.median(distances, axis=0)
    return nearest_distance, nearest_index, persistent_pair_distance


def _component_count(pair_distance: np.ndarray, inactive_flags: np.ndarray) -> int:
    active = [index for index, flag in enumerate(inactive_flags) if not flag]
    parent = {index: index for index in active}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    for position, first in enumerate(active):
        for second in active[position + 1 :]:
            if pair_distance[first, second] < 1.0:
                union(first, second)
    return len({find(index) for index in active})


def _heatmap_frame_stats(
    logits: np.ndarray,
    masks: np.ndarray,
    temperature: float,
) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(logits.astype(np.float32))
    frames, keypoints, height, width = tensor.shape
    flat = tensor.view(frames, keypoints, -1) / float(temperature)
    probability = torch.softmax(flat, dim=-1).view_as(tensor)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=(-1, -2))
    max_probability = probability.amax(dim=(-1, -2))

    x_grid = torch.linspace(0.0, 511.0, width).view(1, 1, 1, width)
    y_grid = torch.linspace(0.0, 511.0, height).view(1, 1, height, 1)
    mean_x = (probability * x_grid).sum(dim=(-1, -2))
    mean_y = (probability * y_grid).sum(dim=(-1, -2))
    variance = (
        (probability * (x_grid - mean_x[:, :, None, None]) ** 2).sum(dim=(-1, -2))
        + (probability * (y_grid - mean_y[:, :, None, None]) ** 2).sum(dim=(-1, -2))
    )
    spatial_std = variance.sqrt()

    resized_masks = F.interpolate(
        torch.from_numpy(masks.astype(np.float32))[:, None],
        size=(height, width),
        mode="nearest",
    )[:, 0]
    mass_in_mask = (probability * resized_masks[:, None]).sum(dim=(-1, -2))

    # Peak ratio is evaluated outside a 15-input-pixel disk around argmax.
    probability_np = probability.numpy()
    peak_ratio = np.empty((frames, keypoints), dtype=np.float64)
    input_per_x = 511.0 / max(width - 1, 1)
    input_per_y = 511.0 / max(height - 1, 1)
    rows, cols = np.ogrid[:height, :width]
    for frame in range(frames):
        for keypoint in range(keypoints):
            current = probability_np[frame, keypoint]
            peak_flat = int(np.argmax(current))
            peak_row, peak_col = np.unravel_index(peak_flat, current.shape)
            outside = (
                ((rows - peak_row) * input_per_y) ** 2
                + ((cols - peak_col) * input_per_x) ** 2
            ) > 15.0**2
            outside_max = float(current[outside].max()) if np.any(outside) else 0.0
            peak_ratio[frame, keypoint] = float(current[peak_row, peak_col]) / max(
                outside_max, 1e-12
            )
    return {
        "entropy": entropy.numpy(),
        "std_px": spatial_std.numpy(),
        "maxprob": max_probability.numpy(),
        "peak_ratio": peak_ratio,
        "mass_in_mask": mass_in_mask.numpy(),
        "probability": probability_np,
    }


def _coverage(masks: np.ndarray, coords_px: np.ndarray, radius_px: float = 24.0) -> np.ndarray:
    result = []
    for frame, mask in enumerate(masks):
        seeds = np.ones_like(mask, dtype=bool)
        points = np.rint(coords_px[frame]).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)
        seeds[points[:, 1], points[:, 0]] = False
        near_keypoint = distance_transform_edt(seeds) <= radius_px
        result.append(float(np.logical_and(mask, near_keypoint).sum()) / max(mask.sum(), 1))
    return np.asarray(result)


def _hot_colormap(values: np.ndarray) -> np.ndarray:
    """Small NumPy equivalent of a black-red-yellow-white hot colour map."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(3.0 * values, 0.0, 1.0)
    green = np.clip(3.0 * values - 1.0, 0.0, 1.0)
    blue = np.clip(3.0 * values - 2.0, 0.0, 1.0)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255.0)


def _heatmap_grid(
    probability: np.ndarray,
    entropy: np.ndarray,
    mass_in_mask: np.ndarray,
    path: Path,
    frame: int = 45,
) -> None:
    """Shared absolute probability scale; flat/dead maps remain visibly weak."""
    keypoints, height, width = probability.shape[1:]
    panel_size = 180
    header = 31
    canvas = Image.new("RGB", (panel_size * 5, (panel_size + header) * 2), color=(0, 0, 0))
    global_max = max(float(probability[frame].max()), 1e-12)
    entropy_norm = entropy[frame] / np.log(height * width)
    for keypoint in range(keypoints):
        current = probability[frame, keypoint]
        scaled = current / global_max
        panel = Image.fromarray(_hot_colormap(scaled), mode="RGB")
        panel = panel.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(panel)
        peak_row, peak_col = np.unravel_index(int(np.argmax(current)), current.shape)
        x_coord = (peak_col + 0.5) / width * panel_size
        y_coord = (peak_row + 0.5) / height * panel_size
        color = COLORS[keypoint % len(COLORS)]
        draw.ellipse([x_coord - 5, y_coord - 5, x_coord + 5, y_coord + 5], outline=color, width=3)
        tile = Image.new("RGB", (panel_size, panel_size + header), color=(0, 0, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((5, 2), f"kp{keypoint}  maxP={current.max():.4f}", fill=color)
        tile_draw.text(
            (5, 15),
            f"Hnorm={entropy_norm[keypoint]:.2f}  mask={mass_in_mask[frame, keypoint]:.2f}",
            fill="white",
        )
        tile.paste(panel, (0, header))
        canvas.paste(
            tile,
            ((keypoint % 5) * panel_size, (keypoint // 5) * (panel_size + header)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _blocked_calibration(spread: np.ndarray, error: np.ndarray) -> list[dict]:
    rows = []
    for keypoint in range(spread.shape[1]):
        block_spread = []
        block_error = []
        for block in range(12):
            section = slice(block * 15, (block + 1) * 15)
            block_spread.append(float(np.median(spread[section, keypoint])))
            block_error.append(float(np.median(error[section, keypoint])))
        rho, p_value = spearmanr(block_spread, block_error)
        rows.append(
            {
                "channel": keypoint,
                "blocked_spearman_rho": float(rho),
                "blocked_spearman_p": float(p_value),
                "sample_unit": "15-frame contiguous block",
                "n": 12,
                "interpretation": "descriptive; temporal blocks are not independent sequences",
            }
        )
    return rows


def diagnose_model(name: str, run_dir: Path, masks: np.ndarray, rotation) -> tuple[list[dict], dict, list[dict]]:
    extractor, cfg = load_run(run_dir)
    coords, logits = forward_sequence(extractor, include_heatmaps=True)
    assert logits is not None
    np.savez_compressed(
        OUTPUTS / f"day1_cache_{name}.npz",
        coords=coords,
        logits=logits,
        temperature=float(cfg["temperature"]),
    )
    coords_px = to_px(coords)
    on_mask = _sample_masks(masks, coords_px)
    on_mask_dilated = _sample_masks(_dilate_masks(masks), coords_px)
    eq1, eq1_vector = _equivariance_errors(coords, rotation, 1)
    eq3, _ = _equivariance_errors(coords, rotation, 3)
    nearest_distance, nearest_index, persistent_distance = _pair_metrics(coords_px)

    next_coords = np.roll(coords, -1, axis=0)
    observed_motion = np.linalg.norm(to_px(next_coords) - coords_px, axis=-1) / CELL64_PX
    expected_motion = np.linalg.norm(to_px(transport(coords, 1, rotation)) - coords_px, axis=-1) / CELL64_PX
    observed_median = np.median(observed_motion, axis=0)
    expected_median = np.median(expected_motion, axis=0)
    activity_ratio = observed_median / np.maximum(expected_median, 1e-12)
    static_flags = np.logical_and(activity_ratio < 0.2, expected_median > 0.5)

    heatmap = _heatmap_frame_stats(logits, masks, float(cfg["temperature"]))
    height, width = logits.shape[-2:]
    median_entropy = np.median(heatmap["entropy"], axis=0)
    median_maxprob = np.median(heatmap["maxprob"], axis=0)
    dead_flags = np.logical_and(
        median_entropy > np.log(height * width) - 0.5,
        median_maxprob < 2.0 / (height * width),
    )
    derotated = derotate(coords, rotation)
    highpass = highpass_residual(derotated)
    sigma_axis = highpass.std(axis=0) / CELL64_NORM

    rows = []
    for keypoint in range(coords.shape[1]):
        counts = np.bincount(nearest_index[:, keypoint], minlength=coords.shape[1])
        partner = int(np.argmax(counts))
        partner_fraction = float(counts[partner] / len(coords))
        rows.append(
            {
                "model": name,
                "channel": keypoint,
                "on_mask_frac": float(on_mask[:, keypoint].mean()),
                "on_mask_frac_dilated": float(on_mask_dilated[:, keypoint].mean()),
                "eq_err_1_median_cells64": float(np.median(eq1[:, keypoint])),
                "eq_err_1_p90_cells64": float(np.quantile(eq1[:, keypoint], 0.9)),
                "eq_err_3_median_cells64": float(np.median(eq3[:, keypoint])),
                "nn_dist_median_cells64": float(np.median(nearest_distance[:, keypoint])),
                "dup_partner": partner if partner_fraction > 0.5 else -1,
                "dup_partner_fraction": partner_fraction,
                "dup_flag": bool(np.median(nearest_distance[:, keypoint]) < 1.0),
                "activity_ratio": float(activity_ratio[keypoint]),
                "expected_motion_median_cells64": float(expected_median[keypoint]),
                "observed_motion_median_cells64": float(observed_median[keypoint]),
                "static_flag": bool(static_flags[keypoint]),
                "dead_flag": bool(dead_flags[keypoint]),
                "hm_entropy_nats": float(np.median(heatmap["entropy"][:, keypoint])),
                "hm_std_px": float(np.median(heatmap["std_px"][:, keypoint])),
                "hm_maxprob": float(np.median(heatmap["maxprob"][:, keypoint])),
                "hm_peak_ratio": float(np.median(heatmap["peak_ratio"][:, keypoint])),
                "hm_mass_in_mask": float(np.median(heatmap["mass_in_mask"][:, keypoint])),
                "sig_x_cells64": float(sigma_axis[keypoint, 0]),
                "sig_y_cells64": float(sigma_axis[keypoint, 1]),
                "geometry_suspect": not rotation.geometry_ok,
                "sample_unit": "frame",
                "n_frames": 180,
                "uncertainty_scope": "descriptive single orbit; temporally correlated",
            }
        )

    spread_by_channel = np.array([row["hm_std_px"] for row in rows])
    error_by_channel = np.array([row["eq_err_1_median_cells64"] for row in rows])
    across_rho, across_p = spearmanr(spread_by_channel, error_by_channel)
    blocked_rows = _blocked_calibration(heatmap["std_px"], eq1)
    for row in blocked_rows:
        row["model"] = name
    blocked_rhos = np.array([row["blocked_spearman_rho"] for row in blocked_rows])
    median_blocked_rho = float(np.nanmedian(blocked_rhos))
    if across_rho >= 0.6 and median_blocked_rho >= 0.6:
        calibration = "usable"
    elif across_rho < 0.3 or median_blocked_rho < 0.3:
        calibration = "not_usable"
    else:
        calibration = "unresolved"

    coverage = _coverage(masks, coords_px)
    n_distinct = _component_count(
        persistent_distance, np.logical_or(static_flags, dead_flags)
    )
    summary = {
        "model": name,
        "run_dir": str(run_dir),
        "heatmap_res": int(cfg.get("heatmap_res", 64)),
        "temperature": float(cfg["temperature"]),
        "median_on_mask_frac": float(np.median([row["on_mask_frac"] for row in rows])),
        "min_on_mask_frac": float(np.min([row["on_mask_frac"] for row in rows])),
        "median_eq_err_1_cells64": float(np.median(eq1)),
        "median_eq_err_3_cells64": float(np.median(eq3)),
        "n_dup_flag_channels": int(sum(bool(row["dup_flag"]) for row in rows)),
        "n_static_flag_channels": int(static_flags.sum()),
        "n_dead_flag_channels": int(dead_flags.sum()),
        "n_distinct_active_channels": int(n_distinct),
        "coverage_median": float(np.median(coverage)),
        "spread_error_spearman_across_channels": float(across_rho),
        "spread_error_spearman_p_across_channels": float(across_p),
        "spread_error_blocked_rho_median": median_blocked_rho,
        "spread_calibration_verdict": calibration,
        "geometry_suspect": not rotation.geometry_ok,
        "sample_unit": "channel for calibration; frame for model metrics",
        "n_channels": 10,
        "n_frames": 180,
        "uncertainty_scope": "descriptive; no population inference",
    }

    for frame in (0, 45, 90, 135):
        overlay(frame, coords_px[frame], OUTPUTS / f"day1_overlay_{name}_frame{frame}.png")
    _heatmap_grid(
        heatmap["probability"],
        heatmap["entropy"],
        heatmap["mass_in_mask"],
        OUTPUTS / f"day1_heatmaps_{name}_frame45.png",
    )
    np.savez_compressed(
        OUTPUTS / f"day1_frame_metrics_{name}.npz",
        eq_err_1_cells64=eq1,
        eq_err_1_vector_cells64=eq1_vector,
        eq_err_3_cells64=eq3,
        hm_std_px=heatmap["std_px"],
        on_mask=on_mask,
        nearest_distance_cells64=nearest_distance,
        coverage=coverage,
    )
    return rows, summary, blocked_rows


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    masks = load_masks()
    rotation = estimate_rotation_model(masks)
    write_rotation_report(rotation)
    if not rotation.geometry_ok:
        print("GEOMETRY FLAG: metrics will be marked geometry_suspect")

    summaries = []
    for name, run_dir in run_directories().items():
        print(f"\nDiagnosing {name}: {run_dir.name}")
        rows, summary, blocked_rows = diagnose_model(name, run_dir, masks, rotation)
        _write_csv(OUTPUTS / f"day1_channels_{name}.csv", rows)
        _write_csv(OUTPUTS / f"day1_calibration_blocks_{name}.csv", blocked_rows)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    _write_csv(OUTPUTS / "day1_summary.csv", summaries)
    metadata = {
        "statistics": "medians and quantiles are descriptive",
        "frame_sample_n": 180,
        "channel_sample_n": 10,
        "temporal_dependence": "frames are one correlated cyclic sequence",
        "calibration_blocks": "12 contiguous blocks of 15 frames",
        "error_bar_definition": "none; no error bars or inferential confidence intervals reported",
    }
    (OUTPUTS / "day1_statistical_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote Day-1 outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
