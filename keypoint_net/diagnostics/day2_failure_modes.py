"""Day-2 hidden failure modes on frozen checkpoints (no training).

Preregistered thresholds:
* channel switching implicated when controlled Hungarian switch_gain > 0.30;
* hard readout clearly better when equivariance error is >=20% lower and true
  mask occupancy is >=0.05 higher;
* aliasing implicated for a 0.25/0.5-pixel shift when median residual exceeds
  0.5 CELL64 and twice the matched transform-inverse control median.

All results are descriptive for one correlated 180-frame orbit. Aliasing
summaries use n=12 evenly spaced frames; no inferential claim is made.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dxutils import (
    CELL64_PX,
    IMG_SIZE,
    OUTPUTS,
    estimate_rotation_model,
    frame_files,
    hungarian,
    load_masks,
    load_run,
    overlay,
    preprocess_image,
    run_directories,
    to_norm,
    to_px,
    transport,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    assert rows, path
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _day1_rows(name: str) -> list[dict]:
    with (OUTPUTS / f"day1_channels_{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def _sample_masks(masks: np.ndarray, coords: np.ndarray) -> np.ndarray:
    pixels = np.rint(to_px(coords)).astype(int)
    pixels[..., 0] = np.clip(pixels[..., 0], 0, masks.shape[2] - 1)
    pixels[..., 1] = np.clip(pixels[..., 1], 0, masks.shape[1] - 1)
    return masks[np.arange(len(masks))[:, None], pixels[..., 1], pixels[..., 0]]


def _eq_error(coords: np.ndarray, rotation, hop: int = 1) -> np.ndarray:
    expected = transport(coords, hop, rotation)
    observed = np.roll(coords, -hop, axis=0)
    return np.linalg.norm(to_px(observed) - to_px(expected), axis=-1) / CELL64_PX


def _hard_coords(logits: np.ndarray) -> np.ndarray:
    frames, keypoints, height, width = logits.shape
    indices = logits.reshape(frames, keypoints, -1).argmax(axis=-1)
    rows, cols = np.divmod(indices, width)
    x_coord = cols / max(width - 1, 1) * 2.0 - 1.0
    y_coord = rows / max(height - 1, 1) * 2.0 - 1.0
    return np.stack([x_coord, y_coord], axis=-1).astype(np.float32)


def _soft_coords(logits: np.ndarray, temperature: float) -> np.ndarray:
    tensor = torch.from_numpy(logits)
    frames, keypoints, height, width = tensor.shape
    probability = torch.softmax(
        tensor.view(frames, keypoints, -1) / temperature, dim=-1
    ).view_as(tensor)
    x_grid = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    y_grid = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    x_coord = (probability * x_grid).sum(dim=(-1, -2))
    y_coord = (probability * y_grid).sum(dim=(-1, -2))
    return torch.stack([x_coord, y_coord], dim=-1).numpy()


def switching_rows(name: str, coords: np.ndarray, rotation) -> tuple[list[dict], dict]:
    expected = transport(coords, 1, rotation)
    observed = np.roll(coords, -1, axis=0)
    expected_px, observed_px = to_px(expected), to_px(observed)
    preserved = np.linalg.norm(observed_px - expected_px, axis=-1) / CELL64_PX
    day1 = _day1_rows(name)
    nonduplicate = np.array(
        [row["dup_flag"] != "True" and row.get("dead_flag") != "True" for row in day1]
    )
    active_indices = np.flatnonzero(nonduplicate)
    matched_controlled = np.full_like(preserved, np.nan)
    identity_count = 0
    permutations = Counter()
    for frame in range(len(coords)):
        full_cost = np.linalg.norm(
            expected_px[frame, :, None, :] - observed_px[frame, None, :, :], axis=-1
        )
        rows, cols = hungarian(full_cost)
        permutation = tuple(int(value) for value in cols[np.argsort(rows)])
        permutations[permutation] += 1
        identity_count += int(permutation == tuple(range(coords.shape[1])))

        controlled_cost = full_cost[np.ix_(active_indices, active_indices)]
        sub_rows, sub_cols = hungarian(controlled_cost)
        for row_index, col_index in zip(sub_rows, sub_cols):
            source = active_indices[row_index]
            matched_controlled[frame, source] = controlled_cost[row_index, col_index] / CELL64_PX

    rows_out = []
    for channel in range(coords.shape[1]):
        preserved_median = float(np.median(preserved[:, channel]))
        if nonduplicate[channel]:
            matched_median = float(np.nanmedian(matched_controlled[:, channel]))
            gain = (preserved_median - matched_median) / max(preserved_median, 1e-12)
        else:
            matched_median, gain = float("nan"), float("nan")
        rows_out.append(
            {
                "model": name,
                "channel": channel,
                "nonduplicate_informative_control": bool(nonduplicate[channel]),
                "preserved_eq_err_median_cells64": preserved_median,
                "hungarian_eq_err_median_cells64": matched_median,
                "switch_gain": gain,
                "switching_implicated": bool(np.isfinite(gain) and gain > 0.30),
                "sample_unit": "frame pair",
                "n": 180,
                "uncertainty_scope": "descriptive single cyclic orbit",
            }
        )
    controlled_preserved = preserved[:, active_indices].ravel()
    controlled_matched = matched_controlled[:, active_indices].ravel()
    model_gain = (
        float(np.median(controlled_preserved)) - float(np.nanmedian(controlled_matched))
    ) / max(float(np.median(controlled_preserved)), 1e-12)
    summary = {
        "identity_assignment_fraction": identity_count / len(coords),
        "controlled_switch_gain": model_gain,
        "switching_implicated": model_gain > 0.30,
        "top_permutations": [
            {"permutation": list(perm), "count": count}
            for perm, count in permutations.most_common(10)
        ],
    }
    return rows_out, summary


def hard_soft_rows(
    name: str,
    soft: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    rotation,
) -> tuple[list[dict], dict]:
    hard = _hard_coords(logits)
    soft_error, hard_error = _eq_error(soft, rotation), _eq_error(hard, rotation)
    soft_mask, hard_mask = _sample_masks(masks, soft), _sample_masks(masks, hard)
    rows = []
    for channel in range(soft.shape[1]):
        soft_eq = float(np.median(soft_error[:, channel]))
        hard_eq = float(np.median(hard_error[:, channel]))
        soft_on = float(soft_mask[:, channel].mean())
        hard_on = float(hard_mask[:, channel].mean())
        improvement = (soft_eq - hard_eq) / max(soft_eq, 1e-12)
        clearly_better = improvement >= 0.20 and hard_on - soft_on >= 0.05
        rows.append(
            {
                "model": name,
                "channel": channel,
                "soft_eq_err_median_cells64": soft_eq,
                "hard_eq_err_median_cells64": hard_eq,
                "relative_eq_improvement": improvement,
                "soft_on_mask_frac": soft_on,
                "hard_on_mask_frac": hard_on,
                "on_mask_absolute_improvement": hard_on - soft_on,
                "hard_clearly_better_both": clearly_better,
                "sample_unit": "frame",
                "n": 180,
                "uncertainty_scope": "descriptive single cyclic orbit",
            }
        )
    soft_eq_model = float(np.median(soft_error))
    hard_eq_model = float(np.median(hard_error))
    soft_on_model = float(soft_mask.mean())
    hard_on_model = float(hard_mask.mean())
    summary = {
        "soft_eq_err_median_cells64": soft_eq_model,
        "hard_eq_err_median_cells64": hard_eq_model,
        "relative_eq_improvement": (soft_eq_model - hard_eq_model) / max(soft_eq_model, 1e-12),
        "soft_on_mask_frac": soft_on_model,
        "hard_on_mask_frac": hard_on_model,
        "on_mask_absolute_improvement": hard_on_model - soft_on_model,
    }
    summary["multimodal_readout_implicated"] = bool(
        summary["relative_eq_improvement"] >= 0.20
        and summary["on_mask_absolute_improvement"] >= 0.05
    )
    overlay(45, to_px(hard[45]), OUTPUTS / f"day2_hard_overlay_{name}_frame45.png")
    return rows, summary


def _pixel_grid(kind: str, value: float, axis: str | None, center: np.ndarray) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(IMG_SIZE, dtype=torch.float32),
        torch.arange(IMG_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    if kind == "shift":
        dx = value if axis == "x" else 0.0
        dy = value if axis == "y" else 0.0
        source_x, source_y = cols - dx, rows - dy
    elif kind == "rotation":
        angle = math.radians(value)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        qx, qy = cols - center[0], rows - center[1]
        source_x = cos_a * qx + sin_a * qy + center[0]
        source_y = -sin_a * qx + cos_a * qy + center[1]
    else:
        raise ValueError(kind)
    return torch.stack(
        [
            source_x / (IMG_SIZE - 1) * 2.0 - 1.0,
            source_y / (IMG_SIZE - 1) * 2.0 - 1.0,
        ],
        dim=-1,
    ).unsqueeze(0)


def _transform_images(
    images: torch.Tensor,
    kind: str,
    value: float,
    axis: str | None,
    center: np.ndarray,
) -> torch.Tensor:
    grid = _pixel_grid(kind, value, axis, center).expand(images.shape[0], -1, -1, -1)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _inverse_coords(
    coords: np.ndarray,
    kind: str,
    value: float,
    axis: str | None,
    center: np.ndarray,
) -> np.ndarray:
    pixels = to_px(coords)
    if kind == "shift":
        pixels[..., 0] -= value if axis == "x" else 0.0
        pixels[..., 1] -= value if axis == "y" else 0.0
    else:
        angle = math.radians(-value)
        relative = pixels - center
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = np.empty_like(relative)
        rotated[..., 0] = cos_a * relative[..., 0] - sin_a * relative[..., 1]
        rotated[..., 1] = sin_a * relative[..., 0] + cos_a * relative[..., 1]
        pixels = rotated + center
    return to_norm(pixels)


def _model_coords(extractor, images: torch.Tensor, batch_size: int = 4) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            flat, _ = extractor(images[start : start + batch_size])
            result.append(flat.view(-1, extractor.num_keypoints, 2).cpu())
    return torch.cat(result).numpy()


def aliasing_rows(name: str, extractor, base_coords: np.ndarray, rotation) -> tuple[list[dict], dict]:
    frame_indices = list(range(0, 180, 15))
    images = torch.cat([preprocess_image(frame_files()[index]) for index in frame_indices])
    baseline = base_coords[frame_indices]
    transforms = []
    for value in (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0):
        transforms.extend([("shift", value, "x"), ("shift", value, "y")])
    for value in (-0.5, -0.25, 0.25, 0.5):
        transforms.append(("rotation", value, None))

    rows = []
    implicated_any = False
    for kind, value, axis in transforms:
        transformed = _transform_images(images, kind, value, axis, rotation.center_px)
        predicted = _model_coords(extractor, transformed)
        recovered = _inverse_coords(predicted, kind, value, axis, rotation.center_px)
        residual = np.linalg.norm(to_px(recovered) - to_px(baseline), axis=-1) / CELL64_PX

        inverse_value = -value
        restored = _transform_images(transformed, kind, inverse_value, axis, rotation.center_px)
        control_prediction = _model_coords(extractor, restored)
        control = np.linalg.norm(
            to_px(control_prediction) - to_px(baseline), axis=-1
        ) / CELL64_PX
        for channel in range(baseline.shape[1]):
            median = float(np.median(residual[:, channel]))
            control_median = float(np.median(control[:, channel]))
            is_subpixel_shift = kind == "shift" and abs(value) in (0.25, 0.5)
            implicated = bool(
                is_subpixel_shift and median > 0.5 and median > 2.0 * control_median
            )
            implicated_any |= implicated
            rows.append(
                {
                    "model": name,
                    "transform": kind,
                    "axis": axis or "about_center",
                    "value_px_or_deg": value,
                    "channel": channel,
                    "residual_median_cells64": median,
                    "residual_p90_cells64": float(np.quantile(residual[:, channel], 0.9)),
                    "control_median_cells64": control_median,
                    "control_p90_cells64": float(np.quantile(control[:, channel], 0.9)),
                    "aliasing_implicated": implicated,
                    "sample_unit": "frame",
                    "n": 12,
                    "uncertainty_scope": "descriptive evenly spaced frames",
                }
            )
    return rows, {"aliasing_implicated_any_channel": implicated_any}


def temperature_rows(
    name: str, logits: np.ndarray, base_temperature: float, masks: np.ndarray, rotation
) -> list[dict]:
    baseline = _soft_coords(logits, base_temperature)
    baseline_error = _eq_error(baseline, rotation)
    rows = []
    for temperature in (0.25, 0.5, 1.0, 2.0):
        coords = _soft_coords(logits, temperature)
        error = _eq_error(coords, rotation)
        on_mask = _sample_masks(masks, coords)
        for channel in range(coords.shape[1]):
            base = float(np.median(baseline_error[:, channel]))
            current = float(np.median(error[:, channel]))
            rows.append(
                {
                    "model": name,
                    "lever": "temperature",
                    "value": temperature,
                    "channel": channel,
                    "eq_err_median_cells64": current,
                    "relative_eq_change_vs_checkpoint": (current - base) / max(base, 1e-12),
                    "on_mask_frac": float(on_mask[:, channel].mean()),
                    "sample_unit": "frame",
                    "n": 180,
                    "uncertainty_scope": "diagnostic-only, descriptive",
                }
            )
    return rows


def tta_coords(extractor, rotation) -> np.ndarray:
    shifts = [
        (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5),
        (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5),
    ]
    outputs = [[] for _ in shifts]
    files = frame_files()
    for start in range(0, len(files), 6):
        images = torch.cat([preprocess_image(path) for path in files[start : start + 6]])
        for shift_index, (dx, dy) in enumerate(shifts):
            shifted = images
            if dx:
                shifted = _transform_images(shifted, "shift", dx, "x", rotation.center_px)
            if dy:
                shifted = _transform_images(shifted, "shift", dy, "y", rotation.center_px)
            predicted = _model_coords(extractor, shifted, batch_size=3)
            pixels = to_px(predicted)
            pixels[..., 0] -= dx
            pixels[..., 1] -= dy
            outputs[shift_index].append(to_norm(pixels))
    trajectories = [np.concatenate(parts, axis=0) for parts in outputs]
    return np.mean(np.stack(trajectories, axis=0), axis=0)


def append_tta_rows(
    rows: list[dict], name: str, tta: np.ndarray, baseline: np.ndarray, masks: np.ndarray, rotation
) -> None:
    baseline_error, tta_error = _eq_error(baseline, rotation), _eq_error(tta, rotation)
    on_mask = _sample_masks(masks, tta)
    for channel in range(tta.shape[1]):
        base = float(np.median(baseline_error[:, channel]))
        current = float(np.median(tta_error[:, channel]))
        rows.append(
            {
                "model": name,
                "lever": "tta_8_shift",
                "value": 0.5,
                "channel": channel,
                "eq_err_median_cells64": current,
                "relative_eq_change_vs_checkpoint": (current - base) / max(base, 1e-12),
                "on_mask_frac": float(on_mask[:, channel].mean()),
                "sample_unit": "frame",
                "n": 180,
                "uncertainty_scope": "diagnostic-only, descriptive",
            }
        )


def main() -> None:
    masks = load_masks()
    rotation = estimate_rotation_model(masks)
    assert rotation.geometry_ok, "G0 must pass before Day-2 equivariance interpretation"
    all_switching, all_hard_soft, all_aliasing, all_levers = [], [], [], []
    summaries = {}
    for name, run_dir in run_directories().items():
        print(f"\nDay 2: {name}")
        cache = np.load(OUTPUTS / f"day1_cache_{name}.npz")
        coords, logits = cache["coords"], cache["logits"]
        temperature = float(cache["temperature"])
        extractor, _ = load_run(run_dir)

        switch, switch_summary = switching_rows(name, coords, rotation)
        hard_soft, hard_summary = hard_soft_rows(name, coords, logits, masks, rotation)
        aliasing, alias_summary = aliasing_rows(name, extractor, coords, rotation)
        levers = temperature_rows(name, logits, temperature, masks, rotation)
        print("  running 8-view test-time averaging")
        averaged = tta_coords(extractor, rotation)
        append_tta_rows(levers, name, averaged, coords, masks, rotation)

        all_switching.extend(switch)
        all_hard_soft.extend(hard_soft)
        all_aliasing.extend(aliasing)
        all_levers.extend(levers)
        summaries[name] = {
            "switching": switch_summary,
            "hard_vs_soft": hard_summary,
            "aliasing": alias_summary,
        }
        print(json.dumps(summaries[name], indent=2))

    _write_csv(OUTPUTS / "day2_switching.csv", all_switching)
    _write_csv(OUTPUTS / "day2_hard_soft.csv", all_hard_soft)
    _write_csv(OUTPUTS / "day2_aliasing.csv", all_aliasing)
    _write_csv(OUTPUTS / "day2_diagnostic_levers.csv", all_levers)
    (OUTPUTS / "day2_summary.json").write_text(json.dumps(summaries, indent=2))
    metadata = {
        "statistics": "median and p90, descriptive",
        "aliasing_sample_unit": "12 evenly spaced frames",
        "orbit_sample_unit": "180 temporally correlated frames",
        "error_bar_definition": "none; no inferential intervals reported",
    }
    (OUTPUTS / "day2_statistical_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote Day-2 outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
