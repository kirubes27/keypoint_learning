#!/usr/bin/env python3
"""Frozen-checkpoint Gate 0 representative-pilot readout replay.

This script performs inference only. It reconstructs the representative
pilot's unaugmented validation set and supervised targets, loads the frozen
seed-41 checkpoint, and compares the current global expectation readout with
the frozen local/windowed variants. Cell argmax is retained only as the
mode-correctness probe required by the v2.1 amendments.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PHD_ROOT = Path("/Users/kirubeso.r/Documents/PhD")
REPO_ROOT = PHD_ROOT / "keypoint_learning_fitted_operator"
KEYPOINT_ROOT = REPO_ROOT / "keypoint_net"
DATA_ROOT = REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2"
RUN_DIR = (
    PHD_ROOT
    / "cluster_downloads"
    / "stage_r2_representative_pilot_20260705_210253"
    / "keypoint_net"
    / "diagnostics"
    / "outputs"
    / "final_material_keypoints"
    / "stage_r2_representative_pilot"
    / "runs"
    / "coordinate_standard64_k10_seed41"
)
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"
CONFIG_PATH = RUN_DIR / "config.json"
ARCHIVED_METRICS_PATH = RUN_DIR / "best_validation_metrics.json"
MODEL_PATH = KEYPOINT_ROOT / "model.py"
TARGET_CODE_PATH = KEYPOINT_ROOT / "diagnostics" / "day45_supervised_control.py"
SPLIT_PATH = DATA_ROOT / "indices" / "split_phase_mod6.json"
OUTPUT_PATH = SCRIPT_DIR / "gate0_replay_metrics.json"

WINDOW_RADII = (2, 4, 8)
EXPECTED_SEED = 41
EXPECTED_VALIDATION_FRAMES = 60
EXPECTED_KEYPOINTS = 10
EXPECTED_HEATMAP_RESOLUTION = 64
TEMPERATURE = 1.0
CELL64_NORM = 2.0 / 64.0
IMAGE_CENTER_XY = (255.49998435893767, 255.50001568508694)
ROLL_SIGN = 1
BASELINE_REPRODUCTION_TOLERANCE_CELLS64 = 0.01

sys.path.insert(0, str(KEYPOINT_ROOT))
from model import KeypointExtractor, spatial_softmax  # noqa: E402
from diagnostics.day45_supervised_control import (  # noqa: E402
    SupervisedRollDataset,
    farthest_interior_points,
    load_arrays,
    target_mask_fraction,
    transported_targets,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")


def require_directory(path: Path, role: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {role}: {path}")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def local_window_expectation(
    heatmaps: torch.Tensor, radius: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the frozen clipped square-window expectation to BxKxHxW logits."""
    if radius < 0:
        raise ValueError(f"radius must be nonnegative, got {radius}")
    batch, channels, height, width = heatmaps.shape
    flat_peak = torch.argmax(heatmaps.reshape(batch, channels, -1), dim=-1)
    peak_y = torch.div(flat_peak, width, rounding_mode="floor")
    peak_x = flat_peak.remainder(width)

    row = torch.arange(height, device=heatmaps.device).view(1, 1, height, 1)
    col = torch.arange(width, device=heatmaps.device).view(1, 1, 1, width)
    window = (
        (row - peak_y[..., None, None]).abs() <= radius
    ) & ((col - peak_x[..., None, None]).abs() <= radius)

    restricted_logits = (heatmaps / temperature).masked_fill(~window, float("-inf"))
    weights = torch.softmax(restricted_logits.flatten(-2), dim=-1)
    grid_y = torch.linspace(
        -1.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype
    )
    grid_x = torch.linspace(
        -1.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype
    )
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    x = (weights * xx.reshape(-1)).sum(dim=-1)
    y = (weights * yy.reshape(-1)).sum(dim=-1)
    return torch.stack((x, y), dim=-1), peak_x, peak_y


def cell_argmax_coordinates(
    peak_x: torch.Tensor, peak_y: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    x = -1.0 + 2.0 * peak_x.to(torch.float32) / float(width - 1)
    y = -1.0 + 2.0 * peak_y.to(torch.float32) / float(height - 1)
    return torch.stack((x, y), dim=-1)


def self_test() -> None:
    """Smoke-test row-major ties, boundary clipping, and restricted softmax."""
    logits = torch.zeros((1, 1, 3, 4), dtype=torch.float64)
    logits[0, 0, 0, 3] = 5.0
    logits[0, 0, 1, 0] = 5.0
    local, peak_x, peak_y = local_window_expectation(logits, 1, 1.0)
    if (int(peak_x.item()), int(peak_y.item())) != (3, 0):
        raise AssertionError("argmax tie did not select the first row-major cell")

    explicit_logits = logits[0, 0, 0:2, 2:4].reshape(-1)
    explicit_weights = torch.softmax(explicit_logits, dim=0)
    explicit_x = torch.tensor(
        [1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0], dtype=logits.dtype
    )
    explicit_y = torch.tensor(
        [-1.0, -1.0, 0.0, 0.0], dtype=logits.dtype
    )
    expected = torch.stack(
        (
            (explicit_weights * explicit_x).sum(),
            (explicit_weights * explicit_y).sum(),
        )
    )
    if not torch.allclose(local[0, 0], expected, atol=1e-12, rtol=0.0):
        raise AssertionError("local expectation does not equal explicit clipped-window result")

    global_expected = spatial_softmax(logits, temperature=1.0).reshape(1, 1, 2)
    if not torch.isfinite(global_expected).all():
        raise AssertionError("global expectation self-test produced nonfinite coordinates")


def state_dict_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_dict_equal(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> bool:
    return before.keys() == after.keys() and all(
        torch.equal(before[key], after[key]) for key in before
    )


def summarize_errors(errors: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(errors, dtype=np.float64)[mask]
    if selected.size == 0:
        return {"count": 0, "median_cells64": None, "p90_cells64": None}
    return {
        "count": int(selected.size),
        "median_cells64": float(np.median(selected)),
        "p90_cells64": float(np.quantile(selected, 0.9)),
    }


def validate_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "seed": EXPECTED_SEED,
        "num_keypoints": EXPECTED_KEYPOINTS,
        "architecture": "standard64",
        "heatmap_res": EXPECTED_HEATMAP_RESOLUTION,
        "target_shift": 0,
        "supervision": "coordinate",
        "shape_constraint": "none",
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint configuration mismatch: {mismatches}")

    validation_frames = [int(frame) for frame in config["validation_frames"]]
    if len(validation_frames) != EXPECTED_VALIDATION_FRAMES:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_FRAMES} validation frames, "
            f"found {len(validation_frames)}"
        )
    if len(set(validation_frames)) != len(validation_frames):
        raise ValueError("checkpoint validation frame list contains duplicates")

    observed_split_hash = sha256_file(SPLIT_PATH)
    if observed_split_hash != config["split_sha256"]:
        raise ValueError(
            "local split hash does not match checkpoint: "
            f"{observed_split_hash} != {config['split_sha256']}"
        )
    split = json.loads(SPLIT_PATH.read_text())
    split_validation = sorted(
        int(row["frame_index"])
        for row in split["val"]
        if row.get("model_name") == config["object"]
    )
    if split_validation != validation_frames:
        raise ValueError(
            "local split validation IDs do not match checkpoint validation IDs"
        )
    if set(validation_frames).intersection(config["test_frames_committed_not_evaluated"]):
        raise ValueError("validation frame list overlaps committed test frames")
    return {
        "validation_frames": validation_frames,
        "split_sha256": observed_split_hash,
    }


def load_validation_problem(
    config: dict[str, Any], validation_frames: list[int]
) -> tuple[SupervisedRollDataset, list[Path], np.ndarray]:
    images, masks, frame_paths = load_arrays(DATA_ROOT, config["object"])
    if len(frame_paths) != 180:
        raise RuntimeError(f"expected 180 frame paths, found {len(frame_paths)}")

    frozen_frame0 = np.asarray(
        config["frame0_targets_px_in_channel_order"], dtype=np.float64
    )
    regenerated_frame0 = farthest_interior_points(
        masks[0], count=config["num_keypoints"]
    )
    if not np.array_equal(frozen_frame0, regenerated_frame0):
        maximum_difference = float(np.max(np.abs(frozen_frame0 - regenerated_frame0)))
        raise ValueError(
            "local mask does not reconstruct frozen frame-0 supervised targets; "
            f"maximum absolute pixel difference={maximum_difference}"
        )

    targets = transported_targets(
        frozen_frame0,
        len(images),
        center_xy=IMAGE_CENTER_XY,
        roll_sign=ROLL_SIGN,
    )
    grounding = target_mask_fraction(targets, masks)
    recorded_grounding = float(
        config["transported_target_on_mask_fraction_all_frames"]
    )
    if abs(grounding - recorded_grounding) > 1e-12:
        raise ValueError(
            "reconstructed target grounding does not match checkpoint: "
            f"{grounding} != {recorded_grounding}"
        )

    dataset = SupervisedRollDataset(
        images,
        masks,
        targets,
        validation_frames,
        augment=False,
        seed=2026070401,
        center_xy=IMAGE_CENTER_XY,
    )
    dataset.set_epoch(0)
    used_frame_paths = [frame_paths[frame] for frame in validation_frames]
    for path in used_frame_paths:
        require_file(path, "validation frame")
    return dataset, used_frame_paths, targets[validation_frames]


def run_replay() -> dict[str, Any]:
    for path, role in (
        (CHECKPOINT_PATH, "frozen best checkpoint"),
        (CONFIG_PATH, "frozen checkpoint config"),
        (ARCHIVED_METRICS_PATH, "archived validation metrics"),
        (MODEL_PATH, "model source"),
        (TARGET_CODE_PATH, "supervised-target source"),
        (SPLIT_PATH, "validation split"),
    ):
        require_file(path, role)
    require_directory(DATA_ROOT, "local representative-pilot data root")

    self_test()
    config = json.loads(CONFIG_PATH.read_text())
    frozen_validation = validate_frozen_inputs(config)
    validation_frames = frozen_validation["validation_frames"]
    dataset, used_frame_paths, expected_targets = load_validation_problem(
        config, validation_frames
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    extractor = KeypointExtractor(
        num_keypoints=int(config["num_keypoints"]),
        base_channels=int(config["base_channels"]),
        temperature=TEMPERATURE,
        padding_mode="reflect",
        heatmap_res=int(config["heatmap_res"]),
        true_quarter_res=bool(config.get("true_quarter_res", False)),
    ).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    extractor.load_state_dict(checkpoint["extractor_state_dict"], strict=True)
    extractor.eval()
    state_before = state_dict_snapshot(extractor)
    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH)

    frames_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    coords_parts: dict[str, list[np.ndarray]] = {
        "global_expectation": [],
        "local_window_r2": [],
        "local_window_r4": [],
        "local_window_r8": [],
        "cell_argmax_probe": [],
    }
    peak_x_parts: list[np.ndarray] = []
    peak_y_parts: list[np.ndarray] = []
    global_forward_max_abs_difference = 0.0
    observed_heatmap_shape: tuple[int, int] | None = None

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            forward_flat, heatmaps = extractor(images)
            batch_size, channels, height, width = heatmaps.shape
            if observed_heatmap_shape is None:
                observed_heatmap_shape = (height, width)
            elif observed_heatmap_shape != (height, width):
                raise RuntimeError("heatmap resolution changed across validation batches")

            global_coords = spatial_softmax(
                heatmaps, temperature=TEMPERATURE
            ).reshape(batch_size, channels, 2)
            forward_coords = forward_flat.reshape(batch_size, channels, 2)
            global_forward_max_abs_difference = max(
                global_forward_max_abs_difference,
                float(torch.max(torch.abs(global_coords - forward_coords)).cpu()),
            )
            coords_parts["global_expectation"].append(global_coords.cpu().numpy())

            reference_peak_x = None
            reference_peak_y = None
            for radius in WINDOW_RADII:
                local_coords, peak_x, peak_y = local_window_expectation(
                    heatmaps, radius, TEMPERATURE
                )
                coords_parts[f"local_window_r{radius}"].append(
                    local_coords.cpu().numpy()
                )
                if reference_peak_x is None:
                    reference_peak_x, reference_peak_y = peak_x, peak_y
                elif not (
                    torch.equal(reference_peak_x, peak_x)
                    and torch.equal(reference_peak_y, peak_y)
                ):
                    raise AssertionError("argmax changed across window-radius evaluations")

            assert reference_peak_x is not None and reference_peak_y is not None
            argmax_coords = cell_argmax_coordinates(
                reference_peak_x, reference_peak_y, height, width
            )
            coords_parts["cell_argmax_probe"].append(argmax_coords.cpu().numpy())
            peak_x_parts.append(reference_peak_x.cpu().numpy())
            peak_y_parts.append(reference_peak_y.cpu().numpy())
            frames_parts.append(batch["frame"].cpu().numpy())
            targets_parts.append(batch["target"].cpu().numpy())

    if observed_heatmap_shape != (
        EXPECTED_HEATMAP_RESOLUTION,
        EXPECTED_HEATMAP_RESOLUTION,
    ):
        raise RuntimeError(
            "checkpoint produced unexpected heatmap shape: "
            f"{observed_heatmap_shape}"
        )
    state_after = state_dict_snapshot(extractor)
    parameters_unchanged = state_dict_equal(state_before, state_after)
    checkpoint_hash_after = sha256_file(CHECKPOINT_PATH)
    if not parameters_unchanged:
        raise AssertionError("model state changed during inference-only replay")
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("checkpoint file changed during replay")

    frames = np.concatenate(frames_parts)
    targets = np.concatenate(targets_parts)
    peak_x = np.concatenate(peak_x_parts)
    peak_y = np.concatenate(peak_y_parts)
    coords = {
        name: np.concatenate(parts) for name, parts in coords_parts.items()
    }
    order = np.argsort(frames)
    frames = frames[order]
    targets = targets[order]
    peak_x = peak_x[order]
    peak_y = peak_y[order]
    coords = {name: values[order] for name, values in coords.items()}
    if frames.tolist() != validation_frames:
        raise ValueError("replay output frame order does not match frozen validation list")
    if not np.array_equal(targets, expected_targets.astype(np.float32)):
        raise ValueError("dataset targets do not equal independently reconstructed targets")

    errors = {
        name: np.linalg.norm(values - targets, axis=-1) / CELL64_NORM
        for name, values in coords.items()
    }
    global_errors = errors["global_expectation"]
    high_error_threshold = float(np.quantile(global_errors, 0.75))
    high_error = global_errors > high_error_threshold

    height, width = observed_heatmap_shape
    target_x_continuous = (targets[..., 0] + 1.0) * 0.5 * (width - 1)
    target_y_continuous = (targets[..., 1] + 1.0) * 0.5 * (height - 1)
    target_x_cell = np.rint(target_x_continuous).astype(np.int64)
    target_y_cell = np.rint(target_y_continuous).astype(np.int64)
    target_x_cell = np.clip(target_x_cell, 0, width - 1)
    target_y_cell = np.clip(target_y_cell, 0, height - 1)
    target_cell_tie_margin = float(
        min(
            np.min(np.abs((target_x_continuous % 1.0) - 0.5)),
            np.min(np.abs((target_y_continuous % 1.0) - 0.5)),
        )
    )
    mode_distance_cells = np.sqrt(
        (peak_x - target_x_cell) ** 2 + (peak_y - target_y_cell) ** 2
    )
    correct_dominant_mode = mode_distance_cells <= 1.0
    correct_high = high_error & correct_dominant_mode
    wrong_or_diffuse_high = high_error & ~correct_dominant_mode
    if int(correct_high.sum() + wrong_or_diffuse_high.sum()) != int(high_error.sum()):
        raise AssertionError("high-error strata do not partition the fixed denominator")

    total_pairs = int(global_errors.size)
    high_count = int(high_error.sum())
    correct_count = int(correct_high.sum())
    wrong_count = int(wrong_or_diffuse_high.sum())
    correct_fraction = float(correct_count / high_count)
    windowing_keeps_rank = correct_count * 2 >= high_count

    archived = json.loads(ARCHIVED_METRICS_PATH.read_text())["unaugmented"]
    replay_global = summarize_errors(
        global_errors, np.ones_like(global_errors, dtype=bool)
    )
    channel_medians = np.median(global_errors, axis=0)
    baseline_reproduction = {
        "tolerance_cells64": BASELINE_REPRODUCTION_TOLERANCE_CELLS64,
        "archived_device": config["device"],
        "replay_device": str(device),
        "archived_median_cells64": float(archived["median_error_cells64"]),
        "replay_median_cells64": replay_global["median_cells64"],
        "median_abs_difference_cells64": abs(
            replay_global["median_cells64"]
            - float(archived["median_error_cells64"])
        ),
        "archived_p90_cells64": float(archived["p90_error_cells64"]),
        "replay_p90_cells64": replay_global["p90_cells64"],
        "p90_abs_difference_cells64": abs(
            replay_global["p90_cells64"] - float(archived["p90_error_cells64"])
        ),
        "max_channel_median_abs_difference_cells64": float(
            np.max(
                np.abs(
                    channel_medians
                    - np.asarray(
                        archived["channel_median_error_cells64"], dtype=np.float64
                    )
                )
            )
        ),
    }
    baseline_reproduction["within_tolerance"] = all(
        float(baseline_reproduction[key])
        <= BASELINE_REPRODUCTION_TOLERANCE_CELLS64
        for key in (
            "median_abs_difference_cells64",
            "p90_abs_difference_cells64",
            "max_channel_median_abs_difference_cells64",
        )
    )
    if not baseline_reproduction["within_tolerance"]:
        raise ValueError(
            "local replay does not reproduce archived global baseline within "
            f"{BASELINE_REPRODUCTION_TOLERANCE_CELLS64} cell64: "
            f"{baseline_reproduction}"
        )

    all_pairs = np.ones_like(global_errors, dtype=bool)
    variant_metrics: dict[str, Any] = {}
    for name, variant_errors in errors.items():
        variant_metrics[name] = {
            "all_pairs": summarize_errors(variant_errors, all_pairs),
            "fixed_global_high_error_pairs": summarize_errors(
                variant_errors, high_error
            ),
            "correct_dominant_mode_high_error_pairs": summarize_errors(
                variant_errors, correct_high
            ),
            "wrong_or_diffuse_mode_high_error_pairs": summarize_errors(
                variant_errors, wrong_or_diffuse_high
            ),
        }

    pairs: list[dict[str, Any]] = []
    for frame_index in range(len(frames)):
        for channel in range(config["num_keypoints"]):
            pairs.append(
                {
                    "frame": int(frames[frame_index]),
                    "channel": int(channel),
                    "target_norm_xy": targets[frame_index, channel].tolist(),
                    "target_cell_xy": [
                        int(target_x_cell[frame_index, channel]),
                        int(target_y_cell[frame_index, channel]),
                    ],
                    "argmax_cell_xy": [
                        int(peak_x[frame_index, channel]),
                        int(peak_y[frame_index, channel]),
                    ],
                    "argmax_to_target_cell_distance_cells": float(
                        mode_distance_cells[frame_index, channel]
                    ),
                    "correct_dominant_mode": bool(
                        correct_dominant_mode[frame_index, channel]
                    ),
                    "global_high_error": bool(high_error[frame_index, channel]),
                    "errors_cells64": {
                        name: float(values[frame_index, channel])
                        for name, values in errors.items()
                    },
                }
            )

    result = {
        "schema_version": 1,
        "gate": "Gate 0 representative-replay pre-gate, v2.1 amendments 1-3",
        "execution_scope": {
            "seed": int(config["seed"]),
            "description": (
                "Descriptive one-seed replay over one object's correlated cyclic "
                "validation orbit; no inference and no pass/fail claim."
            ),
            "training_performed": False,
            "weight_updates_performed": False,
            "test_split_evaluated": False,
            "device": str(device),
        },
        "paths": {
            "checkpoint": CHECKPOINT_PATH,
            "checkpoint_config": CONFIG_PATH,
            "archived_validation_metrics": ARCHIVED_METRICS_PATH,
            "model_source": MODEL_PATH,
            "supervised_target_code": TARGET_CODE_PATH,
            "data_root": DATA_ROOT,
            "validation_split": SPLIT_PATH,
            "validation_frame_directory": (
                DATA_ROOT
                / "train"
                / config["object"]
                / "frames"
                / "a"
            ),
            "validation_frame_paths": used_frame_paths,
            "output": OUTPUT_PATH,
        },
        "hashes": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "model_source_sha256": sha256_file(MODEL_PATH),
            "split_sha256": frozen_validation["split_sha256"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "frozen_configuration": {
            "object": config["object"],
            "validation_frames": validation_frames,
            "number_of_validation_frames": len(validation_frames),
            "number_of_channels": int(config["num_keypoints"]),
            "sample_unit": "channel-frame pair over all validation frames",
            "total_pairs": total_pairs,
            "heatmap_shape_hw": observed_heatmap_shape,
            "temperature": TEMPERATURE,
            "cell_error_unit": (
                "Euclidean normalized-coordinate error divided by 2/64, "
                "matching the representative pilot's CELL64_NORM"
            ),
            "validation_augmentation": False,
            "target_source": (
                "frame0_targets_px_in_channel_order in frozen config, transported "
                "by the pilot's roll target function"
            ),
            "target_center_xy": IMAGE_CENTER_XY,
            "target_roll_sign": ROLL_SIGN,
            "target_grounding_fraction_all_180_frames": float(
                config["transported_target_on_mask_fraction_all_frames"]
            ),
        },
        "implementation_checks": {
            "self_test_passed": True,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "global_forward_max_abs_coordinate_difference": (
                global_forward_max_abs_difference
            ),
            "model_parameters_unchanged": parameters_unchanged,
            "checkpoint_file_unchanged": (
                checkpoint_hash_before == checkpoint_hash_after
            ),
            "target_cell_quantization": (
                "nearest heatmap cell by numpy.rint; no half-cell tie occurred"
            ),
            "minimum_target_quantization_half_cell_tie_margin": (
                target_cell_tie_margin
            ),
            "correct_mode_distance": (
                "Euclidean distance in integer (x,y) heatmap-cell indices <= 1"
            ),
            "argmax_tie_behavior": "first flattened index, row-major",
            "window_boundary_behavior": "clipped at edges, no padding",
            "wrong_or_diffuse_definition": (
                "logical complement of correct-dominant-mode within the fixed "
                "global high-error denominator; v2.1 freezes no separate diffuse cutoff"
            ),
        },
        "baseline_reproduction": baseline_reproduction,
        "high_error_definition": {
            "source_variant": "global_expectation",
            "quantile": 0.75,
            "numpy_quantile_method": "linear (NumPy default)",
            "comparison": "strictly greater than threshold",
            "threshold_cells64": high_error_threshold,
        },
        "strata": {
            "fixed_high_error_denominator": high_count,
            "fixed_high_error_fraction_of_all_pairs": float(
                high_count / total_pairs
            ),
            "correct_dominant_mode_high_error": {
                "count": correct_count,
                "fraction_of_fixed_high_error_denominator": correct_fraction,
                "fraction_of_all_pairs": float(correct_count / total_pairs),
            },
            "wrong_or_diffuse_mode_high_error": {
                "count": wrong_count,
                "fraction_of_fixed_high_error_denominator": float(
                    wrong_count / high_count
                ),
                "fraction_of_all_pairs": float(wrong_count / total_pairs),
            },
        },
        "variant_metrics": variant_metrics,
        "decision": {
            "rule": (
                "at least half of high-error pairs correct-dominant-mode means "
                "windowing keeps rank; otherwise it loses rank"
            ),
            "windowing_keeps_rank": windowing_keeps_rank,
            "correct_dominant_mode_high_error_fraction": correct_fraction,
            "outcome": (
                "Local/windowed readout keeps first rank for Action 2."
                if windowing_keeps_rank
                else (
                    "Local/windowed readout loses rank; the wrong-or-diffuse "
                    "diagnosis routes to the diagnostic head (Decision 2.3) "
                    "and re-planning."
                )
            ),
        },
        "pairs": pairs,
    }
    return result


def main() -> None:
    result = run_replay()
    OUTPUT_PATH.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(json_ready({
        "output": OUTPUT_PATH,
        "script_sha256": result["hashes"]["script_sha256"],
        "strata": result["strata"],
        "decision": result["decision"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
