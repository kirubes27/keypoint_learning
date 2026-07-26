"""Decision 2.3 direct-coordinate diagnostic panel.

This module implements the versioned specification at
``docs/decisions/2026-07-26/DECISION_2_3_DIAGNOSTIC_HEAD_SPEC_v1.md``.

Gate 0 is an immutable input to this experiment.  This module never imports,
executes, or regenerates the Gate 0 replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, __version__ as PILLOW_VERSION
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
REPOSITORY_ROOT = KEYPOINT_ROOT.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor, spatial_softmax  # noqa: E402
from diagnostics.day45_supervised_control import (  # noqa: E402
    CELL64_NORM,
    IMG_SIZE,
    farthest_interior_points,
    target_mask_fraction,
    to_px,
    transported_targets,
)
from diagnostics.stage_a_supervised_control import (  # noqa: E402
    DEFAULT_TEST_AUGMENT_SEED,
    DEFAULT_VALIDATION_AUGMENT_SEED,
    PhaseSplit,
    channel_to_target_indices,
    choose_device,
    json_ready,
    load_phase_split,
    make_dataset,
    make_eval_loader,
    read_history,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
    sha256_file,
    write_history,
)


ARMS = ("raw_linear", "probability_linear", "fixed_expectation")
LEARNED_ARMS = ("raw_linear", "probability_linear")
INITIAL_SEEDS = (42, 43, 44)
EXTENSION_SEEDS = (45, 46)
EXPECTED_DATASET_BASENAME = "_tdw_world_z_roll_base_panel_512_v2"
EXPECTED_OBJECT = "engineers_hammer_vray"
EXPECTED_SPLIT_SHA256 = (
    "49f9d2a34c352d3ebb84809ec36e0a46572b0cde6b7a6d357f317dc44e3da486"
)
EXPECTED_SEMANTIC_LOCK_SHA256 = (
    "c625cc590e8de42b6d8162b044ff6398c0f2e1c04cd6b5ce0cea6f6bccb152aa"
)
EXPECTED_DATASET_INDEX_SHA256 = (
    "719645aee0b092d647cbc29a4cd807d7cd7ca4da1ec2608baacf1c2a12224c6b"
)
EXPECTED_OPERATOR_REFERENCE_SHA256 = (
    "668c79a1a1f7ce789293b4355e3130517c567cd40181fbfd8e02237a151ca856"
)
EXPECTED_TRAINVAL_CONTENT_MANIFEST_SHA256 = (
    "59432dfbb3fdda10d796372024cc53e971ea6550aba280263d91a66fee5e86ce"
)
EXPECTED_PROBE_CHECKPOINT_SHA256 = (
    "d4777e3abfed3d81698ab07edf0563484b49ed314130aa8e925a0de2e1188c3d"
)
EXPECTED_PROBE_CONFIG_SHA256 = (
    "2df6bfd4d0d8487e6f74fb3e63b0eeaa4c2e5604c2ba845c1d7bd56253b9e750"
)
EXPECTED_CENTER_X = 255.49998435893767
EXPECTED_CENTER_Y = 255.50001568508694
EXPECTED_ROLL_SIGN = 1
EXPECTED_TARGET_SHIFT = 0
FULL_RECIPE = {
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "min_epochs": 1000,
    "max_epochs": 3000,
    "eval_every": 25,
    "plateau_patience": 400,
    "relative_improvement": 0.01,
}
SMOKE_RECIPE = {
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "min_epochs": 2,
    "max_epochs": 2,
    "eval_every": 1,
    "plateau_patience": 1,
    "relative_improvement": 0.01,
}
SPEC_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "decisions"
    / "2026-07-26"
    / "DECISION_2_3_DIAGNOSTIC_HEAD_SPEC_v1.md"
)
SOURCE_DEPENDENCIES = {
    "model": KEYPOINT_ROOT / "model.py",
    "day45_supervised_control": HERE / "day45_supervised_control.py",
    "stage_a_supervised_control": HERE / "stage_a_supervised_control.py",
}
SLURM_RUNFILES = {
    "d1_smoke": REPOSITORY_ROOT / "cluster" / "decision23_smoke.slurm",
    "d2_full": REPOSITORY_ROOT / "cluster" / "decision23_full.slurm",
    "d3_finalize": REPOSITORY_ROOT / "cluster" / "decision23_finalize.slurm",
    "extension": REPOSITORY_ROOT / "cluster" / "decision23_extension.slurm",
    "extension_finalize": (
        REPOSITORY_ROOT / "cluster" / "decision23_finalize_extension.slurm"
    ),
}


def source_dependency_hashes() -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in SOURCE_DEPENDENCIES.items()
    }


def slurm_runfile_hashes() -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in SLURM_RUNFILES.items()
    }


def assert_json_finite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_json_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_json_finite(item, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite JSON value at {path}: {value}")


def evaluation_epoch_due(current_epoch: int, eval_every: int) -> bool:
    if current_epoch <= 0 or eval_every <= 0:
        raise ValueError("evaluation epochs and intervals must be positive")
    return current_epoch % eval_every == 0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write auditable JSON while rejecting NaN and Infinity."""
    ready = json_ready(payload)
    assert_json_finite(ready)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ready, indent=2, sort_keys=True, allow_nan=False)
    )


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create JSON exactly once; never replace an existing artifact."""
    ready = json_ready(payload)
    assert_json_finite(ready)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(ready, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one durable, finite JSON record to an existing ledger."""
    ready = json_ready(payload)
    assert_json_finite(ready)
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                ready,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a ledger with one compact JSON record."""
    ready = json_ready(payload)
    assert_json_finite(ready)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(
            json.dumps(
                ready,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"non-object JSONL record at {path}:{line_number}"
                )
            assert_json_finite(value, path=f"{path}:{line_number}")
            records.append(value)
    if not records:
        raise RuntimeError(f"empty ledger: {path}")
    return records


def canonical_json_sha256(payload: Any) -> str:
    ready = json_ready(payload)
    assert_json_finite(ready)
    encoded = json.dumps(
        ready,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def coordinate_grid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return flattened (x, y) grid rows in production flatten order."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=0)


class SharedSpatialReadout(nn.Module):
    """Channel-shared readout for the three Decision 2.3 arms."""

    def __init__(self, arm: str, height: int = 64, width: int = 64):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown Decision 2.3 arm: {arm}")
        self.arm = arm
        self.height = int(height)
        self.width = int(width)
        if self.height <= 1 or self.width <= 1:
            raise ValueError("spatial readout requires height and width greater than one")
        grid = coordinate_grid(self.height, self.width)
        self.register_buffer("coordinate_grid", grid, persistent=True)
        if arm in LEARNED_ARMS:
            self.decoder = nn.Linear(self.height * self.width, 2)
            with torch.no_grad():
                if arm == "raw_linear":
                    self.decoder.weight.copy_(
                        grid / math.sqrt(self.height * self.width)
                    )
                else:
                    self.decoder.weight.copy_(grid)
                self.decoder.bias.zero_()
        else:
            self.decoder = None

    @property
    def added_parameter_count(self) -> int:
        if self.decoder is None:
            return 0
        return sum(parameter.numel() for parameter in self.decoder.parameters())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"expected BxKxHxW logits, got {tuple(logits.shape)}")
        batch, channels, height, width = logits.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(
                f"expected {self.height}x{self.width} logits, got {height}x{width}"
            )
        if self.arm == "fixed_expectation":
            return spatial_softmax(logits, temperature=1.0)

        flat = logits.reshape(batch, channels, -1)
        if self.arm == "raw_linear":
            readout_input = flat - flat.mean(dim=-1, keepdim=True)
        else:
            readout_input = torch.softmax(flat, dim=-1)
        assert self.decoder is not None
        return self.decoder(readout_input.reshape(batch * channels, -1)).reshape(
            batch, channels, 2
        )


class Decision23Extractor(nn.Module):
    """Existing encoder/heatmap head plus a frozen Decision 2.3 readout arm."""

    def __init__(
        self,
        *,
        arm: str,
        num_keypoints: int = 10,
        base_channels: int = 32,
    ):
        super().__init__()
        self.arm = arm
        self.num_keypoints = int(num_keypoints)
        self.backbone = KeypointExtractor(
            num_keypoints=self.num_keypoints,
            base_channels=base_channels,
            temperature=1.0,
            padding_mode="reflect",
            heatmap_res=64,
            true_quarter_res=False,
        )
        self.readout = SharedSpatialReadout(arm, height=64, width=64)

    def heatmap_logits(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encoder(image)
        if self.backbone.head_upsample is not None:
            features = self.backbone.head_upsample(features)
        return self.backbone.heatmap_head(features)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.heatmap_logits(image)
        coordinates = self.readout(logits)
        return coordinates.reshape(image.shape[0], -1), logits

    def get_keypoint_coords(self, image: torch.Tensor) -> torch.Tensor:
        flat, _ = self.forward(image)
        return flat.reshape(image.shape[0], self.num_keypoints, 2)

    def set_training_mode(self, *, freeze_backbone: bool) -> None:
        self.train()
        if freeze_backbone:
            self.backbone.eval()


def build_extractor(args: argparse.Namespace, device: torch.device) -> Decision23Extractor:
    return Decision23Extractor(
        arm=args.arm,
        num_keypoints=args.num_keypoints,
        base_channels=args.base_channels,
    ).to(device)


def _dataset_paths(
    data_root: Path,
    object_name: str,
) -> tuple[list[Path], list[Path]]:
    object_dir = data_root / "train" / object_name
    frame_paths = sorted((object_dir / "frames" / "a").glob("img_*.png"))
    mask_paths = sorted((object_dir / "masks" / "a").glob("mask_*.png"))
    expected_frames = [f"img_{index:04d}.png" for index in range(180)]
    expected_masks = [f"mask_{index:04d}.png" for index in range(180)]
    if [path.name for path in frame_paths] != expected_frames:
        raise RuntimeError("Decision 2.3 requires the exact 180 RGB frame names")
    if [path.name for path in mask_paths] != expected_masks:
        raise RuntimeError("Decision 2.3 requires the exact 180 mask frame names")
    return frame_paths, mask_paths


def _authorized_frames(split: PhaseSplit, mode: str) -> tuple[int, ...]:
    if mode in {"lock", "train", "probe"}:
        return tuple(sorted(set(split.train) | set(split.validation)))
    if mode in {"finalize", "finalize-extension"}:
        return tuple(split.test)
    raise ValueError(f"unknown Decision 2.3 mode: {mode}")


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_scoped_problem(
    args: argparse.Namespace,
    *,
    mode: str,
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load only the frames that the current command is authorized to observe."""
    split_path = args.split_json or (
        args.data_root / "indices" / "split_phase_mod6.json"
    )
    split = load_phase_split(split_path, args.object)
    frame_paths, mask_paths = _dataset_paths(args.data_root, args.object)
    loaded_frames = _authorized_frames(split, mode)
    images = {frame: _read_rgb(frame_paths[frame]) for frame in loaded_frames}
    masks = {frame: _read_mask(mask_paths[frame]) for frame in loaded_frames}
    content_manifest: dict[str, str] = {}
    for frame in loaded_frames:
        for path in (frame_paths[frame], mask_paths[frame]):
            relative = str(path.relative_to(args.data_root))
            content_manifest[relative] = sha256_file(path)
    content_manifest_sha256 = canonical_json_sha256(content_manifest)
    center = (args.center_x, args.center_y)
    if frozen_config is None:
        if 0 not in masks:
            raise RuntimeError("frame 0 is required to construct frozen targets")
        physical_frame0_points = farthest_interior_points(
            masks[0], count=args.num_keypoints
        )
        channel_to_target = channel_to_target_indices(
            args.num_keypoints, args.target_shift
        )
    else:
        physical_frame0_points = np.asarray(
            frozen_config["physical_frame0_targets_px"], dtype=np.float64
        )
        channel_to_target = [
            int(index) for index in frozen_config["channel_to_physical_target"]
        ]
        expected_mapping = channel_to_target_indices(
            args.num_keypoints, args.target_shift
        )
        if channel_to_target != expected_mapping:
            raise RuntimeError("frozen checkpoint changed the channel-target mapping")
        if physical_frame0_points.shape != (args.num_keypoints, 2):
            raise RuntimeError("invalid frozen frame-0 target shape")
    physical_targets = transported_targets(
        physical_frame0_points,
        180,
        center_xy=center,
        roll_sign=args.roll_sign,
    )
    targets = physical_targets[:, channel_to_target]
    frame0_points = physical_frame0_points[channel_to_target]
    loaded_targets = targets[list(loaded_frames)]
    loaded_masks = np.stack([masks[frame] for frame in loaded_frames])
    grounding = target_mask_fraction(loaded_targets, loaded_masks)
    if grounding < 0.98:
        raise RuntimeError(
            "transported target grounding failed on authorized frames: "
            f"{grounding:.6f} < 0.98"
        )
    return {
        "split": split,
        "split_path": split_path,
        "images": images,
        "masks": masks,
        "frame_paths": frame_paths,
        "mask_paths": mask_paths,
        "targets": targets,
        "physical_targets": physical_targets,
        "frame0_points": frame0_points,
        "physical_frame0_points": physical_frame0_points,
        "channel_to_physical_target": channel_to_target,
        "target_grounding": grounding,
        "target_grounding_frame_indices": list(loaded_frames),
        "loaded_frame_indices": list(loaded_frames),
        "loaded_content_sha256": content_manifest,
        "loaded_content_manifest_sha256": content_manifest_sha256,
        "discovered_frame_count": len(frame_paths),
        "discovered_mask_count": len(mask_paths),
        "center": center,
    }


def assert_semantic_scope(
    args: argparse.Namespace,
    problem: dict[str, Any],
) -> None:
    split: PhaseSplit = problem["split"]
    failures: list[str] = []
    if args.data_root.name != EXPECTED_DATASET_BASENAME:
        failures.append(
            f"dataset basename {args.data_root.name!r} != "
            f"{EXPECTED_DATASET_BASENAME!r}"
        )
    if args.object != EXPECTED_OBJECT:
        failures.append(f"object {args.object!r} != {EXPECTED_OBJECT!r}")
    if split.sha256 != EXPECTED_SPLIT_SHA256:
        failures.append(f"split SHA-256 {split.sha256} != {EXPECTED_SPLIT_SHA256}")
    if (
        problem["discovered_frame_count"] != 180
        or problem["discovered_mask_count"] != 180
    ):
        failures.append("dataset must contain exactly 180 RGB frames and masks")
    expected_loaded = set(_authorized_frames(split, args.mode))
    if set(problem["loaded_frame_indices"]) != expected_loaded:
        failures.append("loaded frame set differs from the mode-authorized split set")
    forbidden = (
        set(split.test)
        if args.mode in {"lock", "train", "probe"}
        else set(split.train) | set(split.validation)
    )
    if set(problem["loaded_frame_indices"]) & forbidden:
        failures.append("a forbidden split frame was loaded")
    if (
        args.mode in {"lock", "train", "probe"}
        and problem["loaded_content_manifest_sha256"]
        != EXPECTED_TRAINVAL_CONTENT_MANIFEST_SHA256
    ):
        failures.append(
            "train/validation RGB-mask content manifest differs from the "
            "frozen dataset"
        )
    if args.num_keypoints != 10:
        failures.append("Decision 2.3 is frozen at ten keypoints")
    if args.base_channels != 32:
        failures.append("Decision 2.3 is frozen at base_channels=32")
    if args.seed not in (*INITIAL_SEEDS, *EXTENSION_SEEDS):
        failures.append("Decision 2.3 seed must be one of 42, 43, 44, 45, or 46")
    if args.mode == "probe" and args.seed not in INITIAL_SEEDS:
        failures.append("frozen probe seed must be one of 42, 43, or 44")
    if args.mode == "train" and args.run_scope == "smoke" and args.seed != 42:
        failures.append("D1 smoke seed must be 42")
    if args.target_shift != EXPECTED_TARGET_SHIFT:
        failures.append("Decision 2.3 target_shift is frozen at zero")
    if args.roll_sign != EXPECTED_ROLL_SIGN:
        failures.append("Decision 2.3 roll_sign is frozen at +1")
    if not math.isclose(args.center_x, EXPECTED_CENTER_X, rel_tol=0.0, abs_tol=1e-12):
        failures.append("Decision 2.3 center_x differs from the frozen pivot")
    if not math.isclose(args.center_y, EXPECTED_CENTER_Y, rel_tol=0.0, abs_tol=1e-12):
        failures.append("Decision 2.3 center_y differs from the frozen pivot")
    semantic_lock = args.data_root / "semantic_lock.json"
    dataset_index = args.data_root / "dataset_index.json"
    operator_reference = args.data_root / "operator_reference.json"
    if not semantic_lock.is_file():
        failures.append(f"missing dataset semantic lock: {semantic_lock}")
    else:
        semantic_lock_sha256 = sha256_file(semantic_lock)
        if semantic_lock_sha256 != EXPECTED_SEMANTIC_LOCK_SHA256:
            failures.append(
                "semantic-lock SHA-256 "
                f"{semantic_lock_sha256} != {EXPECTED_SEMANTIC_LOCK_SHA256}"
            )
        lock_text = semantic_lock.read_text()
        required = (
            "axis='roll'",
            "is_world=True",
            "use_centroid=True",
            "reset to base pose",
        )
        for phrase in required:
            if phrase not in lock_text:
                failures.append(f"semantic lock lacks {phrase!r}")
    metadata_hashes = (
        (dataset_index, EXPECTED_DATASET_INDEX_SHA256),
        (operator_reference, EXPECTED_OPERATOR_REFERENCE_SHA256),
    )
    for path, expected_sha256 in metadata_hashes:
        if not path.is_file():
            failures.append(f"missing dataset metadata: {path}")
        else:
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                failures.append(
                    f"{path.name} SHA-256 {observed_sha256} != {expected_sha256}"
                )
    if failures:
        raise RuntimeError("Decision 2.3 semantic-scope failure: " + "; ".join(failures))


def assert_split_access(mode: str, split_name: str) -> None:
    """Fail closed if a command tries to use a split outside its authority."""
    allowed = {
        "train": {"train", "validation"},
        "probe": {"train", "validation"},
        "finalize": {"test"},
        "finalize-extension": {"test"},
    }
    if mode not in allowed:
        raise ValueError(f"unknown split-access mode: {mode}")
    if split_name not in allowed[mode]:
        raise RuntimeError(
            f"{mode} is not authorized to construct the {split_name} dataset"
        )


def assert_recipe_scope(args: argparse.Namespace) -> None:
    """Fail closed on every optimization value in the frozen recipe."""
    if args.mode == "train" and args.run_scope == "smoke":
        expected = SMOKE_RECIPE
        if args.seed != 42:
            raise RuntimeError("D1 smoke is frozen at seed 42")
    else:
        expected = FULL_RECIPE
    observed = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "eval_every": args.eval_every,
        "plateau_patience": args.plateau_patience,
        "relative_improvement": args.relative_improvement,
    }
    mismatch = {
        key: (expected_value, observed[key])
        for key, expected_value in expected.items()
        if (
            not math.isclose(
                float(observed[key]),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            if isinstance(expected_value, float)
            else observed[key] != expected_value
        )
    }
    if mismatch:
        raise RuntimeError(f"Decision 2.3 recipe mismatch: {mismatch}")


def assert_frozen_config_recipe(config: dict[str, Any]) -> None:
    if config.get("run_scope") != "full":
        raise RuntimeError("scientific finalization accepts full runs only")
    mapping = {
        "batch_size": "batch_size",
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "min_epochs": "min_epochs",
        "max_epochs": "max_epochs",
        "eval_every": "eval_every",
        "plateau_patience": "plateau_patience_epochs",
        "relative_improvement": "relative_improvement",
    }
    mismatch = {
        config_key: (expected, config.get(config_key))
        for recipe_key, config_key in mapping.items()
        for expected in (FULL_RECIPE[recipe_key],)
        if (
            not math.isclose(
                float(config.get(config_key, float("nan"))),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            if isinstance(expected, float)
            else config.get(config_key) != expected
        )
    }
    if mismatch:
        raise RuntimeError(f"frozen run recipe mismatch: {mismatch}")


def extension_authority(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate the predeclared authority for optimization seeds 45 and 46."""
    if args.mode == "probe":
        if args.seed not in INITIAL_SEEDS:
            raise RuntimeError("frozen probes are authorized only for seeds 42-44")
        if args.initial_report is not None:
            raise RuntimeError("frozen probes cannot consume an extension report")
        return None
    if args.mode != "train":
        return None
    if args.seed in INITIAL_SEEDS:
        if args.initial_report is not None:
            raise RuntimeError("initial seeds must not consume extension authority")
        return None
    if args.seed not in EXTENSION_SEEDS:
        raise RuntimeError("end-to-end training seed is outside the frozen seed set")
    if args.run_scope != "full":
        raise RuntimeError("extension seeds are not authorized for smoke runs")
    if args.initial_report is None or not args.initial_report.is_file():
        raise RuntimeError(
            "seeds 45/46 require --initial-report from the immutable D3 result"
        )
    report = validate_initial_report_for_extension(
        args.initial_report,
        expected_output_root=args.output_root,
    )
    arm_payload = report.get("arms", {}).get(args.arm)
    if arm_payload is None or arm_payload.get("status") != "provisional":
        raise RuntimeError(f"initial report does not authorize extension for {args.arm}")
    if (
        int(arm_payload.get("pass_count", -1)) != 2
        or int(arm_payload.get("total_seeds", -1)) != 3
    ):
        raise RuntimeError("provisional extension requires exactly 2/3 initial passes")
    return {
        "path": str(args.initial_report.resolve()),
        "sha256": sha256_file(args.initial_report),
        "arm": args.arm,
        "initial_pass_count": 2,
        "initial_total_seeds": 3,
    }


def assert_probe_checkpoint(path: Path) -> Path:
    """Bind the optional probe to the exact frozen representative seed-41 run."""
    checkpoint_sha256 = sha256_file(path)
    if checkpoint_sha256 != EXPECTED_PROBE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "probe checkpoint SHA-256 differs from the frozen seed-41 checkpoint"
        )
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"probe checkpoint lacks sibling config: {config_path}")
    config_sha256 = sha256_file(config_path)
    if config_sha256 != EXPECTED_PROBE_CONFIG_SHA256:
        raise RuntimeError(
            "probe config SHA-256 differs from the frozen seed-41 config"
        )
    config = json.loads(config_path.read_text())
    required = {
        "seed": 41,
        "object": EXPECTED_OBJECT,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "supervision": "coordinate",
    }
    mismatch = {
        key: (required_value, config.get(key))
        for key, required_value in required.items()
        if config.get(key) != required_value
    }
    if mismatch:
        raise RuntimeError(f"probe checkpoint config mismatch: {mismatch}")
    return config_path


def validate_prelaunch_lock(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / "DECISION23_PRELAUNCH_LOCK.json"
    if not path.is_file():
        raise RuntimeError(f"missing Decision 2.3 prelaunch lock: {path}")
    payload = json.loads(path.read_text())
    git = git_identity()
    expected = {
        "created_before_decision23_runs": True,
        "decision_spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "source_dependencies_sha256": source_dependency_hashes(),
        "slurm_runfiles_sha256": slurm_runfile_hashes(),
        "dataset_basename": EXPECTED_DATASET_BASENAME,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "object": EXPECTED_OBJECT,
        "arms": list(ARMS),
        "initial_seeds": list(INITIAL_SEEDS),
        "extension_seeds": list(EXTENSION_SEEDS),
        "train_validation_content_manifest_sha256": (
            EXPECTED_TRAINVAL_CONTENT_MANIFEST_SHA256
        ),
    }
    mismatch = {
        key: (expected_value, payload.get(key))
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if payload.get("git", {}).get("commit") != git["commit"]:
        mismatch["git.commit"] = (
            git["commit"],
            payload.get("git", {}).get("commit"),
        )
    if git["status_porcelain"]:
        mismatch["git.status_porcelain"] = ("", git["status_porcelain"])
    if mismatch:
        raise RuntimeError(f"prelaunch lock mismatch: {mismatch}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def validate_d1_report(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    requires_d1 = (
        args.mode == "train"
        and not args.freeze_backbone
        and args.run_scope == "full"
    )
    if not requires_d1:
        if args.d1_report is not None:
            raise RuntimeError("D1 report is accepted only for full end-to-end runs")
        return None
    if args.d1_report is None or not args.d1_report.is_file():
        raise RuntimeError("full D2/extension runs require --d1-report")
    path = args.d1_report.resolve()
    if path.parent != args.output_root.resolve():
        raise RuntimeError("D1 report must belong to the current output root")
    report = json.loads(path.read_text())
    if (
        report.get("gate") != "Decision_2_3_D1_smoke"
        or report.get("status") != "pass"
        or report.get("scientific_result") is not False
    ):
        raise RuntimeError("D1 report does not record a passing wiring-only smoke")
    git = git_identity()
    expected_bindings = {
        "git_commit": git["commit"],
        "source_sha256": sha256_file(Path(__file__)),
        "decision_spec_sha256": sha256_file(SPEC_PATH),
        "source_dependencies_sha256": source_dependency_hashes(),
        "slurm_runfiles_sha256": slurm_runfile_hashes(),
    }
    mismatch = {
        key: (expected, report.get(key))
        for key, expected in expected_bindings.items()
        if report.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"D1 report binding mismatch: {mismatch}")
    rows = report.get("arms", [])
    identities = {(row.get("arm"), int(row.get("seed", -1))) for row in rows}
    if identities != {(arm, 42) for arm in ARMS}:
        raise RuntimeError("D1 report lacks the exact three-arm seed-42 matrix")
    if not report.get("slurm_job_id"):
        raise RuntimeError("D1 report lacks a Slurm job ID")
    gpu_names = set()
    for row in rows:
        config_path = Path(row["run_dir"]) / "config.json"
        checkpoint_path = Path(row["run_dir"]) / "best_model.pt"
        if sha256_file(config_path) != row["config_sha256"]:
            raise RuntimeError("D1 config hash changed")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise RuntimeError("D1 checkpoint hash changed")
        config = json.loads(config_path.read_text())
        if config.get("git_commit") != git["commit"]:
            raise RuntimeError("D1 arm config commit mismatch")
        for key, expected in expected_bindings.items():
            if key == "git_commit":
                continue
            if config.get(key) != expected:
                raise RuntimeError(f"D1 arm config {key} binding mismatch")
        gpu_name = config.get("runtime", {}).get("cuda_device_name")
        if not gpu_name:
            raise RuntimeError("D1 arm config lacks the exact CUDA device")
        gpu_names.add(gpu_name)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "slurm_job_id": str(report["slurm_job_id"]),
        "gpu_device_names": sorted(gpu_names),
    }


def make_scoped_dataset(
    problem: dict[str, Any],
    *,
    mode: str,
    split_name: str,
    augment: bool,
    seed: int,
    access_log: list[str] | None = None,
):
    """Construct a dataset only after enforcing and recording split authority."""
    assert_split_access(mode, split_name)
    split: PhaseSplit = problem["split"]
    if split_name == "train":
        indices = split.train
    elif split_name == "validation":
        indices = split.validation
    elif split_name == "test":
        indices = split.test
    else:
        raise ValueError(f"unknown split name: {split_name}")
    missing = set(indices) - set(problem["loaded_frame_indices"])
    if missing:
        raise RuntimeError(
            f"{split_name} dataset requested frames that were not authorized to load: "
            f"{sorted(missing)}"
        )
    if access_log is not None:
        access_log.append(split_name)
    return make_dataset(problem, indices, augment=augment, seed=seed)


def git_identity() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def runtime_identity(device: torch.device | None = None) -> dict[str, Any]:
    cuda_index: int | None = None
    cuda_name: str | None = None
    cuda_capability: list[int] | None = None
    cuda_total_memory: int | None = None
    if device is not None and device.type == "cuda":
        cuda_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        properties = torch.cuda.get_device_properties(cuda_index)
        cuda_name = properties.name
        cuda_capability = [int(properties.major), int(properties.minor)]
        cuda_total_memory = int(properties.total_memory)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": PILLOW_VERSION,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_index": cuda_index,
        "cuda_device_name": cuda_name,
        "cuda_device_capability": cuda_capability,
        "cuda_device_total_memory_bytes": cuda_total_memory,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


RUNTIME_SOFTWARE_KEYS = (
    "python",
    "torch",
    "numpy",
    "pillow",
    "cuda_runtime",
    "cudnn_version",
    "deterministic_algorithms_enabled",
    "cudnn_deterministic",
    "cudnn_benchmark",
)


def runtime_software_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    """Project a recorded runtime onto cross-job comparability requirements."""
    missing = [key for key in RUNTIME_SOFTWARE_KEYS if key not in runtime]
    if missing:
        raise RuntimeError(f"runtime identity lacks software fields: {missing}")
    return {key: runtime[key] for key in RUNTIME_SOFTWARE_KEYS}


def matched_config_value(config: dict[str, Any], key: str) -> Any:
    if key == "runtime":
        return runtime_software_identity(config[key])
    return config[key]


def slurm_identity() -> dict[str, Any]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURMD_NODENAME",
    )
    return {key.lower(): os.environ.get(key) for key in keys}


def run_name(args: argparse.Namespace) -> str:
    prefix = "probe" if args.freeze_backbone else "e2e"
    return f"{prefix}_{args.arm}_standard64_k10_seed{args.seed}"


def make_config(
    args: argparse.Namespace,
    problem: dict[str, Any],
    device: torch.device,
    extractor: Decision23Extractor,
    *,
    seed_authority: dict[str, Any] | None,
    prelaunch_lock: dict[str, Any],
    d1_report: dict[str, Any] | None,
) -> dict[str, Any]:
    split: PhaseSplit = problem["split"]
    git = git_identity()
    config = {
        "schema_version": 1,
        "semantic_role": "Decision_2_3_direct_coordinate_diagnostic_only",
        "decision_spec": str(SPEC_PATH),
        "decision_spec_sha256": sha256_file(SPEC_PATH),
        "gate0_policy": "complete; never rerun by this module",
        "arm": args.arm,
        "freeze_backbone": bool(args.freeze_backbone),
        "frozen_checkpoint": (
            str(args.frozen_checkpoint.resolve())
            if args.frozen_checkpoint is not None
            else None
        ),
        "frozen_checkpoint_sha256": (
            sha256_file(args.frozen_checkpoint)
            if args.frozen_checkpoint is not None
            else None
        ),
        "frozen_checkpoint_config": (
            str((args.frozen_checkpoint.parent / "config.json").resolve())
            if args.frozen_checkpoint is not None
            else None
        ),
        "frozen_checkpoint_config_sha256": (
            sha256_file(args.frozen_checkpoint.parent / "config.json")
            if args.frozen_checkpoint is not None
            else None
        ),
        "object": args.object,
        "data_root": str(args.data_root.resolve()),
        "dataset_basename": args.data_root.name,
        "transformation": "absolute TDW world-Z roll about centroid",
        "frames_per_object": 180,
        "angle_step_deg": 2,
        "cyclic": True,
        "dataset_semantic_lock_sha256": sha256_file(
            args.data_root / "semantic_lock.json"
        ),
        "dataset_index_sha256": sha256_file(args.data_root / "dataset_index.json"),
        "operator_reference_sha256": sha256_file(
            args.data_root / "operator_reference.json"
        ),
        "split_json": str(problem["split_path"].resolve()),
        "split_sha256": split.sha256,
        "train_frames": list(split.train),
        "validation_frames": list(split.validation),
        "test_frames_committed_not_evaluated": list(split.test),
        "seed": int(args.seed),
        "extension_authority": seed_authority,
        "prelaunch_lock": prelaunch_lock,
        "d1_report": d1_report,
        "target_shift": int(args.target_shift),
        "center_x": float(args.center_x),
        "center_y": float(args.center_y),
        "roll_sign": int(args.roll_sign),
        "num_keypoints": int(args.num_keypoints),
        "base_channels": int(args.base_channels),
        "architecture": "standard64",
        "heatmap_resolution": 64,
        "temperature": 1.0,
        "coordinate_range": "unbounded for learned arms; no clamp",
        "cell64_norm": CELL64_NORM,
        "readout": {
            "shared_across_channels": True,
            "cross_channel_mixing": False,
            "added_parameter_count": extractor.readout.added_parameter_count,
            "raw_linear_center_logits": args.arm == "raw_linear",
            "raw_linear_grid_scale": (
                "production coordinate grid / sqrt(4096)"
                if args.arm == "raw_linear"
                else None
            ),
            "probability_linear_init": (
                "exact production coordinate grid"
                if args.arm == "probability_linear"
                else None
            ),
        },
        "loss": "coordinate_MSE_only",
        "optimizer": "Adam",
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "min_epochs": int(args.min_epochs),
        "max_epochs": int(args.max_epochs),
        "eval_every": int(args.eval_every),
        "plateau_patience_epochs": int(args.plateau_patience),
        "relative_improvement": float(args.relative_improvement),
        "checkpoint_selection": (
            "minimum validation max(unaugmented,fixed-augmented) "
            "median-of-channel median error"
        ),
        "train_augmentation": {"rotation_deg": [-5, 5], "translation_px": [-8, 8]},
        "validation_augmentation_seed": DEFAULT_VALIDATION_AUGMENT_SEED,
        "test_augmentation_seed": DEFAULT_TEST_AUGMENT_SEED,
        "channel_to_physical_target": problem["channel_to_physical_target"],
        "frame0_targets_px_in_channel_order": problem["frame0_points"].tolist(),
        "physical_frame0_targets_px": problem["physical_frame0_points"].tolist(),
        "transported_target_on_mask_fraction_authorized_frames": problem[
            "target_grounding"
        ],
        "target_grounding_frame_indices": problem["target_grounding_frame_indices"],
        "loaded_frame_indices": problem["loaded_frame_indices"],
        "train_validation_content_sha256": problem["loaded_content_sha256"],
        "train_validation_content_manifest_sha256": problem[
            "loaded_content_manifest_sha256"
        ],
        "device": str(device),
        "run_scope": args.run_scope,
        "test_policy": "not loaded during train/probe; each checkpoint finalized once",
        "statistical_scope": (
            "one object and one correlated cyclic orbit; optimization seed is "
            "the replication unit; descriptive only"
        ),
        "git": git,
        "git_commit": git["commit"],
        "source_sha256": sha256_file(Path(__file__)),
        "source_dependencies_sha256": source_dependency_hashes(),
        "slurm_runfiles_sha256": slurm_runfile_hashes(),
        "runtime": runtime_identity(device),
        "slurm": slurm_identity(),
    }
    if args.run_scope == "full" and git["status_porcelain"]:
        raise RuntimeError("full Decision 2.3 runs require a clean Git worktree")
    return config


def in_range_mask_membership(
    predictions: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return in-range and on-mask flags without hiding out-of-range outputs."""
    if predictions.ndim != 3 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape frames x channels x 2")
    predictions64 = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(predictions64).all():
        raise FloatingPointError("non-finite coordinate prediction")
    in_range = np.logical_and(
        predictions64 >= -1.0, predictions64 <= 1.0
    ).all(axis=-1)
    indexing_copy = np.clip(predictions64, -1.0, 1.0)
    safe_pixels = np.rint(to_px(indexing_copy)).astype(np.int64)
    inside = np.stack(
        [
            masks[index, safe_pixels[index, :, 1], safe_pixels[index, :, 0]]
            for index in range(len(masks))
        ]
    ).astype(bool)
    return in_range, np.logical_and(inside, in_range)


@torch.no_grad()
def evaluate(
    extractor: Decision23Extractor,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_unit: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    extractor.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in loader:
        image = batch["image"].to(device)
        prediction = extractor.get_keypoint_coords(image)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("non-finite coordinate prediction")
        if not torch.isfinite(batch["target"]).all():
            raise FloatingPointError("non-finite coordinate target")
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].numpy())
        frames.append(batch["frame"].numpy())
        masks.append(batch["mask"].numpy())
    if not predictions:
        raise RuntimeError("evaluation loader yielded no batches")
    prediction_array = np.concatenate(predictions).astype(np.float64, copy=False)
    target_array = np.concatenate(targets).astype(np.float64, copy=False)
    frame_array = np.concatenate(frames)
    mask_array = np.concatenate(masks)
    order = np.argsort(frame_array)
    prediction_array = prediction_array[order]
    target_array = target_array[order]
    frame_array = frame_array[order]
    mask_array = mask_array[order]
    in_range, on_mask = in_range_mask_membership(prediction_array, mask_array)
    error = np.linalg.norm(prediction_array - target_array, axis=-1) / CELL64_NORM
    if not np.isfinite(error).all():
        raise FloatingPointError("non-finite coordinate error")
    channel_medians = np.median(error, axis=0)
    metrics = {
        "median_error_cells64": float(np.median(error)),
        "median_of_channel_medians_cells64": float(np.median(channel_medians)),
        "channel_median_error_cells64": channel_medians.tolist(),
        "p90_error_cells64": float(np.quantile(error, 0.9)),
        "on_mask_fraction": float(on_mask.mean()),
        "in_range_fraction": float(in_range.mean()),
        "out_of_range_count": int((~in_range).sum()),
        "n_frames": int(len(frame_array)),
        "n_channel_frame_pairs": int(error.size),
        "sample_unit": sample_unit,
        "uncertainty": (
            "descriptive only; frames are one correlated cyclic orbit and no "
            "population inference is made"
        ),
    }
    arrays = {
        "prediction": prediction_array,
        "target": target_array,
        "frame": frame_array,
        "in_range": in_range,
        "on_mask": on_mask,
        "error_cells64": error,
    }
    assert_json_finite(metrics)
    return metrics, arrays


def evaluate_pair(
    extractor: Decision23Extractor,
    plain_loader: DataLoader,
    augmented_loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    plain, plain_arrays = evaluate(
        extractor,
        plain_loader,
        device,
        sample_unit=f"{split_name} frame x supervised channel",
    )
    augmented, augmented_arrays = evaluate(
        extractor,
        augmented_loader,
        device,
        sample_unit=(
            f"fixed digitally augmented {split_name} frame x supervised channel"
        ),
    )
    score = max(
        plain["median_of_channel_medians_cells64"],
        augmented["median_of_channel_medians_cells64"],
    )
    metrics = {
        "selection_score_cells64": float(score),
        "selection_rule": (
            "max(unaugmented,fixed-augmented median-of-channel median error)"
        ),
        "unaugmented": plain,
        "fixed_augmented": augmented,
    }
    arrays: dict[str, np.ndarray] = {}
    for prefix, source in (
        ("plain", plain_arrays),
        ("augmented", augmented_arrays),
    ):
        arrays.update({f"{prefix}_{key}": value for key, value in source.items()})
    return metrics, arrays


def _module_modes(model: nn.Module) -> dict[nn.Module, bool]:
    return {module: module.training for module in model.modules()}


def _restore_module_modes(modes: dict[nn.Module, bool]) -> None:
    for module, training in modes.items():
        module.training = training


def _rng_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def _restore_rng(snapshot: dict[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if "cuda" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().double().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise FloatingPointError("tensor summary received empty or non-finite values")
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "exact_zero_count": int((values == 0).sum()),
        "n": int(values.numel()),
    }


def gradient_audit(
    extractor: Decision23Extractor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Observe the active readout's real gradient path without changing state."""
    modes = _module_modes(extractor)
    rng = _rng_snapshot()
    grad_buffers = {
        parameter: (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
        for parameter in extractor.parameters()
    }
    try:
        extractor.eval()
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        logits = extractor.heatmap_logits(image)
        backbone_frozen = not any(
            parameter.requires_grad for parameter in extractor.backbone.parameters()
        )
        audit_logits = (
            logits
            if logits.requires_grad
            else logits.detach().requires_grad_(True)
        )
        coordinates = extractor.readout(audit_logits)
        loss = F.mse_loss(coordinates, target)
        for name, tensor in (
            ("logits", logits),
            ("coordinates", coordinates),
            ("target", target),
            ("loss", loss),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"gradient audit has non-finite {name}")
        candidate_parameters: dict[str, nn.Parameter] = {
            "encoder_first_conv_weight": extractor.backbone.encoder[0].weight,
            "encoder_final_conv_weight": extractor.backbone.encoder[9].weight,
            "heatmap_head_weight": extractor.backbone.heatmap_head.weight,
            "heatmap_head_bias": extractor.backbone.heatmap_head.bias,
        }
        if extractor.readout.decoder is not None:
            candidate_parameters["decoder_weight"] = extractor.readout.decoder.weight
            candidate_parameters["decoder_bias"] = extractor.readout.decoder.bias
        named_parameters = {
            name: parameter
            for name, parameter in candidate_parameters.items()
            if parameter.requires_grad
        }
        requested: list[torch.Tensor] = [audit_logits, *named_parameters.values()]
        gradients = torch.autograd.grad(
            loss,
            requested,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        logit_gradient = gradients[0]
        if logit_gradient is None:
            raise RuntimeError("active readout produced no gradient to logits")
        if not torch.isfinite(logit_gradient).all():
            raise FloatingPointError("active readout produced non-finite logit gradient")
        parameter_gradients: dict[str, torch.Tensor | None] = {
            name: None for name in candidate_parameters
        }
        parameter_gradients.update(
            {
                name: gradient
                for name, gradient in zip(
                    named_parameters, gradients[1:], strict=True
                )
            }
        )
        for name, gradient in parameter_gradients.items():
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"gradient audit produced non-finite parameter gradient: {name}"
                )
        probability = torch.softmax(logits.detach().flatten(-2), dim=-1)
        per_pair_logit_norm = torch.linalg.vector_norm(
            logit_gradient.detach().flatten(-2), dim=-1
        )
        per_channel_logit_norm = torch.linalg.vector_norm(
            logit_gradient.detach().permute(1, 0, 2, 3).reshape(
                extractor.num_keypoints, -1
            ),
            dim=-1,
        )
        maximum_probability = probability.max(dim=-1).values
        effective_support = 1.0 / torch.sum(probability**2, dim=-1)
        parameter_gradient_norms = {
            name: (
                None
                if gradient is None
                else float(torch.linalg.vector_norm(gradient.detach()))
            )
            for name, gradient in parameter_gradients.items()
        }
        parameter_gradient_exact_zero_counts = {
            name: (
                None
                if gradient is None
                else int((gradient.detach() == 0).sum())
            )
            for name, gradient in parameter_gradients.items()
        }
        result = {
            "loss": float(loss.detach()),
            "pooled_logit_gradient_l2_norm": float(
                torch.linalg.vector_norm(logit_gradient.detach())
            ),
            "pooled_logit_gradient_exact_zero_count": int(
                (logit_gradient.detach() == 0).sum()
            ),
            "logit_gradient_norm_per_pair": _tensor_summary(per_pair_logit_norm),
            "per_channel_logit_gradient_l2_norm": per_channel_logit_norm.cpu().tolist(),
            "per_channel_logit_gradient_exact_zero_count": [
                int((logit_gradient.detach()[:, channel] == 0).sum())
                for channel in range(extractor.num_keypoints)
            ],
            "parameter_gradient_norms": parameter_gradient_norms,
            "parameter_gradient_exact_zero_counts": (
                parameter_gradient_exact_zero_counts
            ),
            "max_probability": _tensor_summary(maximum_probability),
            "effective_support_cells": _tensor_summary(effective_support),
            "coordinate_error_cells64": _tensor_summary(
                torch.linalg.vector_norm(coordinates.detach() - target, dim=-1)
                / CELL64_NORM
            ),
            "fixed_expectation_coordinate_error_cells64": _tensor_summary(
                torch.linalg.vector_norm(
                    spatial_softmax(logits.detach(), temperature=1.0) - target,
                    dim=-1,
                )
                / CELL64_NORM
            ),
            "decoder_weight_norm": (
                None
                if extractor.readout.decoder is None
                else float(
                    torch.linalg.vector_norm(extractor.readout.decoder.weight.detach())
                )
            ),
            "decoder_bias_norm": (
                None
                if extractor.readout.decoder is None
                else float(
                    torch.linalg.vector_norm(extractor.readout.decoder.bias.detach())
                )
            ),
            "sample_unit": "fixed unaugmented validation batch x channel",
            "backbone_frozen": backbone_frozen,
            "frozen_parameter_gradients_reported_as": "not_applicable",
            "interpretation": "descriptive active-readout autograd path; no gate",
        }
        assert_json_finite(result)
    finally:
        _restore_module_modes(modes)
        _restore_rng(rng)
    for parameter, before in grad_buffers.items():
        after = parameter.grad
        if before is None and after is not None:
            raise AssertionError("gradient audit populated a parameter .grad buffer")
        if before is not None:
            if after is None or not torch.equal(before, after):
                raise AssertionError("gradient audit changed a parameter .grad buffer")
    return result


def assert_expected_upstream_gradients(audit: dict[str, Any]) -> None:
    """Treat exact-zero expected paths as a wiring failure."""
    expected = []
    if audit["backbone_frozen"]:
        expected.append("decoder_weight")
    else:
        expected.extend(
            (
                "encoder_first_conv_weight",
                "encoder_final_conv_weight",
                "heatmap_head_weight",
            )
        )
        if "decoder_weight" in audit["parameter_gradient_norms"]:
            expected.append("decoder_weight")
    failures = [
        name
        for name in expected
        if audit["parameter_gradient_norms"].get(name) is None
        or not math.isfinite(float(audit["parameter_gradient_norms"][name]))
        or audit["parameter_gradient_norms"][name] <= 0.0
    ]
    pooled = float(audit["pooled_logit_gradient_l2_norm"])
    if not math.isfinite(pooled) or pooled <= 0.0:
        failures.append("logits")
    if failures:
        raise RuntimeError(
            "Decision 2.3 active readout has zero expected upstream gradient: "
            + ", ".join(failures)
        )


def verify_smoke_checkpoint_restore(
    args: argparse.Namespace,
    run_dir: Path,
    fixed_audit_batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Exercise the production restore and post-restore audit paths for D1."""
    restored = build_extractor(args, device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in restored.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = restore_checkpoint(
        checkpoint_path,
        restored,
        optimizer,
        loader_generator,
        device,
    )
    audit = gradient_audit(restored, fixed_audit_batch, device)
    assert_expected_upstream_gradients(audit)
    payload = {
        "status": "pass",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "restored_epoch": int(checkpoint["epoch"]),
        "restored_optimizer_parameter_groups": len(optimizer.param_groups),
        "restored_loader_generator_state": True,
        "post_restore_gradient_audit": audit,
    }
    write_json(run_dir / "smoke_checkpoint_restore.json", payload)
    return payload


def assert_finite_training_state(
    parameters: Iterable[nn.Parameter],
    *,
    require_gradients: bool,
) -> None:
    for index, parameter in enumerate(parameters):
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError(f"non-finite model parameter at index {index}")
        gradient = parameter.grad
        if require_gradients and gradient is None:
            raise RuntimeError(f"missing training gradient at parameter index {index}")
        if gradient is not None and not torch.isfinite(gradient.detach()).all():
            raise FloatingPointError(
                f"non-finite training gradient at parameter index {index}"
            )


def validate_resume_records(
    history: list[dict[str, Any]],
    audit_history: list[dict[str, Any]],
    *,
    checkpoint_epoch: int,
) -> None:
    history_epochs = [int(row["epoch"]) for row in history]
    audit_epochs = [int(row["epoch"]) for row in audit_history]
    if history_epochs != sorted(set(history_epochs)):
        raise RuntimeError("history has duplicate or non-monotone epochs")
    if audit_epochs != sorted(set(audit_epochs)):
        raise RuntimeError("gradient audit has duplicate or non-monotone epochs")
    if not audit_epochs or audit_epochs[0] != 0:
        raise RuntimeError("gradient audit lacks the epoch-0 record")
    if audit_epochs[1:] != history_epochs:
        raise RuntimeError("history and gradient-audit epochs are not aligned")
    expected_last = history_epochs[-1] if history_epochs else 0
    if expected_last != checkpoint_epoch:
        raise RuntimeError(
            "history/audit are not aligned with the resume checkpoint: "
            f"records={expected_last}, checkpoint={checkpoint_epoch}"
        )


def build_training_datasets(
    problem: dict[str, Any],
    args: argparse.Namespace,
    access_log: list[str],
):
    train_data = make_scoped_dataset(
        problem,
        mode=args.mode,
        split_name="train",
        augment=True,
        seed=args.seed,
        access_log=access_log,
    )
    validation_plain = make_scoped_dataset(
        problem,
        mode=args.mode,
        split_name="validation",
        augment=False,
        seed=DEFAULT_VALIDATION_AUGMENT_SEED,
        access_log=access_log,
    )
    validation_augmented = make_scoped_dataset(
        problem,
        mode=args.mode,
        split_name="validation",
        augment=True,
        seed=DEFAULT_VALIDATION_AUGMENT_SEED,
        access_log=access_log,
    )
    if (
        validation_plain.seed != DEFAULT_VALIDATION_AUGMENT_SEED
        or validation_augmented.seed != DEFAULT_VALIDATION_AUGMENT_SEED
        or validation_plain.epoch != 0
        or validation_augmented.epoch != 0
    ):
        raise RuntimeError("validation augmentation seed/epoch wiring changed")
    return train_data, validation_plain, validation_augmented


def _immutable_config_keys() -> tuple[str, ...]:
    return (
        "decision_spec_sha256",
        "arm",
        "freeze_backbone",
        "frozen_checkpoint_sha256",
        "frozen_checkpoint_config_sha256",
        "extension_authority",
        "prelaunch_lock",
        "d1_report",
        "object",
        "dataset_basename",
        "dataset_semantic_lock_sha256",
        "dataset_index_sha256",
        "operator_reference_sha256",
        "train_validation_content_manifest_sha256",
        "split_sha256",
        "seed",
        "target_shift",
        "center_x",
        "center_y",
        "roll_sign",
        "num_keypoints",
        "base_channels",
        "architecture",
        "heatmap_resolution",
        "temperature",
        "loss",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "min_epochs",
        "max_epochs",
        "eval_every",
        "plateau_patience_epochs",
        "relative_improvement",
        "run_scope",
        "git_commit",
        "source_sha256",
        "source_dependencies_sha256",
        "slurm_runfiles_sha256",
        "runtime",
    )


def train(args: argparse.Namespace) -> Path:
    assert_recipe_scope(args)
    seed_authority = extension_authority(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Decision 2.3 training/probe requires a CUDA allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Decision 2.3 training/probe must run inside Slurm")
    problem = load_scoped_problem(args, mode=args.mode)
    assert_semantic_scope(args, problem)
    prelaunch_lock = validate_prelaunch_lock(args)
    d1_report = validate_d1_report(args)
    split: PhaseSplit = problem["split"]
    namespace = "probe" if args.freeze_backbone else args.run_scope
    run_dir = args.output_root / namespace / "runs" / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(args, device)
    if args.freeze_backbone:
        if args.arm not in LEARNED_ARMS:
            raise ValueError("the frozen probe is defined only for learned arms A/B")
        if args.frozen_checkpoint is None:
            raise ValueError("--frozen-checkpoint is required for a frozen probe")
        assert_probe_checkpoint(args.frozen_checkpoint)
        checkpoint = torch.load(
            args.frozen_checkpoint, map_location=device, weights_only=True
        )
        extractor.backbone.load_state_dict(
            checkpoint["extractor_state_dict"], strict=True
        )
        for parameter in extractor.backbone.parameters():
            parameter.requires_grad_(False)
    elif args.frozen_checkpoint is not None:
        raise ValueError("--frozen-checkpoint is permitted only with --freeze-backbone")

    config = make_config(
        args,
        problem,
        device,
        extractor,
        seed_authority=seed_authority,
        prelaunch_lock=prelaunch_lock,
        d1_report=d1_report,
    )
    config_path = run_dir / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}; use --resume")
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        mismatch = {
            key: (
                (
                    runtime_software_identity(existing[key])
                    if key == "runtime" and key in existing
                    else existing.get(key)
                ),
                (
                    runtime_software_identity(config[key])
                    if key == "runtime" and key in config
                    else config.get(key)
                ),
            )
            for key in _immutable_config_keys()
            if (
                (
                    runtime_software_identity(existing[key])
                    if key == "runtime" and key in existing
                    else existing.get(key)
                )
                != (
                    runtime_software_identity(config[key])
                    if key == "runtime" and key in config
                    else config.get(key)
                )
            )
        }
        if mismatch:
            raise ValueError(f"resume configuration mismatch: {mismatch}")
        config = existing
    else:
        unexpected = [str(path) for path in run_dir.iterdir()]
        if unexpected:
            raise RuntimeError(
                f"new run directory contains unbound artifacts: {unexpected}"
            )
        write_json(config_path, config)

    split_access_log: list[str] = []
    train_data, validation_plain, validation_augmented = build_training_datasets(
        problem,
        args,
        split_access_log,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_plain_loader = make_eval_loader(validation_plain, args.batch_size)
    validation_augmented_loader = make_eval_loader(
        validation_augmented, args.batch_size
    )
    fixed_audit_batch = next(iter(validation_plain_loader))

    trainable = [parameter for parameter in extractor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_score = float("inf")
    significant_best = float("inf")
    last_significant_epoch = 0
    audit_path = run_dir / "gradient_audit_history.json"
    last_path = run_dir / "last_checkpoint.pt"
    initial_validation_path = run_dir / "initial_validation_metrics.json"
    if args.resume:
        if (run_dir / "training_summary.json").exists():
            raise RuntimeError("completed run cannot be resumed")
        if not last_path.exists():
            raise FileNotFoundError(f"missing resume checkpoint: {last_path}")
        checkpoint = restore_checkpoint(
            last_path, extractor, optimizer, loader_generator, device
        )
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        significant_best = float(checkpoint["significant_best"])
        last_significant_epoch = int(checkpoint["last_significant_epoch"])
        history = read_history(run_dir / "history.csv")
        if not audit_path.is_file():
            raise FileNotFoundError(f"missing resume gradient audit: {audit_path}")
        audit_history = json.loads(audit_path.read_text())["records"]
        validate_resume_records(
            history,
            audit_history,
            checkpoint_epoch=start_epoch,
        )
    else:
        history = []
        if (
            audit_path.exists()
            or last_path.exists()
            or initial_validation_path.exists()
        ):
            raise RuntimeError("new run directory contains stale training artifacts")
        initial_audit = gradient_audit(extractor, fixed_audit_batch, device)
        assert_expected_upstream_gradients(initial_audit)
        audit_history = [
            {
                "epoch": 0,
                "audit": initial_audit,
            }
        ]
        write_json(audit_path, {"records": audit_history})
        save_checkpoint(
            last_path,
            extractor=extractor,
            optimizer=optimizer,
            epoch=0,
            config=config,
            best_score=best_score,
            significant_best=significant_best,
            last_significant_epoch=last_significant_epoch,
            loader_generator=loader_generator,
        )

    if not initial_validation_path.exists():
        if start_epoch != 0:
            raise RuntimeError(
                "initial validation artifact is missing after training advanced"
            )
        initial_validation, _ = evaluate_pair(
            extractor,
            validation_plain_loader,
            validation_augmented_loader,
            device,
            split_name="validation",
        )
        write_json(initial_validation_path, initial_validation)

    start_time = time.perf_counter()
    stop_reason = "hard_cap_unconverged"
    for epoch in range(start_epoch, args.max_epochs):
        train_data.set_epoch(epoch)
        extractor.set_training_mode(freeze_backbone=args.freeze_backbone)
        losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            flat, _ = extractor(image)
            coordinates = flat.reshape(-1, args.num_keypoints, 2)
            loss = F.mse_loss(coordinates, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert_finite_training_state(trainable, require_gradients=True)
            optimizer.step()
            assert_finite_training_state(trainable, require_gradients=True)
            losses.append(float(loss.detach().cpu()))

        current_epoch = epoch + 1
        if not evaluation_epoch_due(current_epoch, args.eval_every):
            continue
        metrics, _ = evaluate_pair(
            extractor,
            validation_plain_loader,
            validation_augmented_loader,
            device,
            split_name="validation",
        )
        score = float(metrics["selection_score_cells64"])
        audit = gradient_audit(extractor, fixed_audit_batch, device)
        if args.run_scope == "smoke":
            assert_expected_upstream_gradients(audit)
        audit_history.append({"epoch": current_epoch, "audit": audit})
        write_json(audit_path, {"records": audit_history})
        row = {
            "epoch": current_epoch,
            "train_loss": float(np.mean(losses)),
            "val_score_cells64": score,
            "val_plain_median_cells64": metrics["unaugmented"][
                "median_of_channel_medians_cells64"
            ],
            "val_augmented_median_cells64": metrics["fixed_augmented"][
                "median_of_channel_medians_cells64"
            ],
        }
        assert_json_finite(row)
        history.append(row)
        write_history(run_dir / "history.csv", history)
        print(json.dumps(row), flush=True)

        if score < best_score:
            best_score = score
            save_checkpoint(
                run_dir / "best_model.pt",
                extractor=extractor,
                optimizer=optimizer,
                epoch=current_epoch,
                config=config,
                best_score=best_score,
                significant_best=significant_best,
                last_significant_epoch=last_significant_epoch,
                loader_generator=loader_generator,
            )
            write_json(run_dir / "best_validation_metrics.json", metrics)
            write_json(run_dir / "best_gradient_audit.json", audit)

        threshold = significant_best * (1.0 - args.relative_improvement)
        if not np.isfinite(significant_best) or score <= threshold:
            significant_best = score
            last_significant_epoch = current_epoch
        save_checkpoint(
            last_path,
            extractor=extractor,
            optimizer=optimizer,
            epoch=current_epoch,
            config=config,
            best_score=best_score,
            significant_best=significant_best,
            last_significant_epoch=last_significant_epoch,
            loader_generator=loader_generator,
        )
        if (
            current_epoch >= args.min_epochs
            and current_epoch - last_significant_epoch >= args.plateau_patience
        ):
            stop_reason = "validation_plateau"
            break

    if not history:
        raise RuntimeError("training produced no evaluation record")
    best_checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    smoke_restore = (
        verify_smoke_checkpoint_restore(
            args,
            run_dir,
            fixed_audit_batch,
            device,
        )
        if args.run_scope == "smoke"
        else None
    )
    summary = {
        "run_dir": str(run_dir.resolve()),
        "arm": args.arm,
        "seed": int(args.seed),
        "freeze_backbone": bool(args.freeze_backbone),
        "run_scope": args.run_scope,
        "completed_epoch": int(history[-1]["epoch"]),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_score_cells64": float(best_score),
        "stop_reason": stop_reason,
        "runtime_seconds_this_invocation": time.perf_counter() - start_time,
        "test_evaluated": False,
        "split_access_log": split_access_log,
        "split_access_assertion": (
            sorted(set(split_access_log)) == ["train", "validation"]
            and "test" not in split_access_log
        ),
        "smoke_checkpoint_restore": smoke_restore,
        "interpretation": (
            "validation-only descriptive frozen probe"
            if args.freeze_backbone
            else "Decision 2.3 end-to-end diagnostic; test remains untouched"
        ),
    }
    write_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


def instrument_pass(metrics: dict[str, Any]) -> bool:
    plain = metrics["unaugmented"]
    augmented = metrics["fixed_augmented"]
    return bool(
        max(
            plain["median_of_channel_medians_cells64"],
            augmented["median_of_channel_medians_cells64"],
        )
        <= 0.50
        and max(plain["p90_error_cells64"], augmented["p90_error_cells64"])
        <= 1.50
        and min(plain["on_mask_fraction"], augmented["on_mask_fraction"])
        >= 0.95
    )


def arm_status(pass_count: int, total: int) -> str:
    if total == 3:
        if pass_count == 3:
            return "pass"
        if pass_count == 2:
            return "provisional"
        return "fail"
    if total == 5:
        return "pass" if pass_count >= 4 else "fail"
    raise ValueError(f"unsupported seed total: {total}")


def assert_frozen_source_is_current(configs: Iterable[dict[str, Any]]) -> None:
    configs = list(configs)
    if not configs:
        raise ValueError("no frozen configs supplied")
    expected_script = sha256_file(Path(__file__))
    expected_dependencies = source_dependency_hashes()
    expected_slurm_runfiles = slurm_runfile_hashes()
    git = git_identity()
    if not torch.cuda.is_available():
        raise RuntimeError("test finalization requires CUDA")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("test finalization must run inside Slurm")
    current_runtime = runtime_identity(torch.device("cuda"))
    if git["status_porcelain"]:
        raise RuntimeError("test finalization requires a clean Git worktree")
    for config in configs:
        if config.get("source_sha256") != expected_script:
            raise RuntimeError(
                "finalizer source differs from the implementation frozen in runs"
            )
        if config.get("source_dependencies_sha256") != expected_dependencies:
            raise RuntimeError(
                "an imported model/data/evaluation dependency changed after training"
            )
        if config.get("slurm_runfiles_sha256") != expected_slurm_runfiles:
            raise RuntimeError(
                "a Decision 2.3 Slurm runfile changed after training"
            )
        if config.get("git_commit") != git["commit"]:
            raise RuntimeError("finalizer Git commit differs from the run commit")
        if runtime_software_identity(
            config.get("runtime", {})
        ) != runtime_software_identity(current_runtime):
            raise RuntimeError(
                "finalizer software/determinism runtime differs from the frozen runs"
            )
        if config.get("decision_spec_sha256") != sha256_file(SPEC_PATH):
            raise RuntimeError(
                "finalizer specification differs from the specification frozen in runs"
            )


def validate_frozen_run_artifacts(
    run_dir: Path,
    *,
    allow_bound_test_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate one scientific run before any test data can be constructed."""
    forbidden_test_artifacts = (
        run_dir / "test_metrics.json",
        run_dir / "test_predictions.npz",
    )
    existing = [str(path) for path in forbidden_test_artifacts if path.exists()]
    if existing and not allow_bound_test_artifacts:
        raise RuntimeError(f"test artifact already exists: {existing}")
    required = {
        "config": run_dir / "config.json",
        "summary": run_dir / "training_summary.json",
        "checkpoint": run_dir / "best_model.pt",
        "history": run_dir / "history.csv",
        "audit_history": run_dir / "gradient_audit_history.json",
        "best_validation": run_dir / "best_validation_metrics.json",
        "best_audit": run_dir / "best_gradient_audit.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete frozen run; missing={missing}")
    config = json.loads(required["config"].read_text())
    assert_frozen_config_recipe(config)
    summary = json.loads(required["summary"].read_text())
    if summary.get("test_evaluated"):
        raise RuntimeError(f"summary says test already evaluated: {run_dir}")
    if not summary.get("split_access_assertion"):
        raise RuntimeError(f"run did not prove split isolation: {run_dir}")
    if set(summary.get("split_access_log", [])) != {"train", "validation"}:
        raise RuntimeError(f"unexpected training split access: {run_dir}")
    checkpoint = torch.load(
        required["checkpoint"], map_location="cpu", weights_only=True
    )
    if json_ready(checkpoint.get("config")) != config:
        raise RuntimeError(f"checkpoint-embedded config mismatch: {run_dir}")
    if int(checkpoint.get("epoch", -1)) != int(summary.get("best_epoch", -2)):
        raise RuntimeError(f"best-checkpoint epoch mismatch: {run_dir}")
    history = read_history(required["history"])
    if not history:
        raise RuntimeError(f"empty training history: {run_dir}")
    history_epochs = {int(row["epoch"]) for row in history}
    if int(summary["best_epoch"]) not in history_epochs:
        raise RuntimeError(f"best epoch absent from history: {run_dir}")
    audit_history_payload = json.loads(required["audit_history"].read_text())
    audit_records = audit_history_payload.get("records", [])
    audit_by_epoch = {
        int(record["epoch"]): record["audit"] for record in audit_records
    }
    if 0 not in audit_by_epoch or int(summary["best_epoch"]) not in audit_by_epoch:
        raise RuntimeError(f"initial/best gradient audit missing: {run_dir}")
    best_audit = json.loads(required["best_audit"].read_text())
    if best_audit != audit_by_epoch[int(summary["best_epoch"])]:
        raise RuntimeError(f"selected-checkpoint audit mismatch: {run_dir}")
    best_validation = json.loads(required["best_validation"].read_text())
    if not math.isclose(
        float(best_validation["selection_score_cells64"]),
        float(summary["best_validation_score_cells64"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"selected validation metric mismatch: {run_dir}")
    return {
        "run_dir": run_dir,
        "config": config,
        "summary": summary,
        "checkpoint_path": required["checkpoint"],
        "config_sha256": sha256_file(required["config"]),
        "checkpoint_sha256": sha256_file(required["checkpoint"]),
    }


def _args_from_config(
    config: dict[str, Any],
    template: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(template).copy()
    values.update(
        {
            "data_root": Path(config["data_root"]),
            "split_json": Path(config["split_json"]),
            "object": config["object"],
            "arm": config["arm"],
            "seed": int(config["seed"]),
            "target_shift": int(config["target_shift"]),
            "center_x": float(config["center_x"]),
            "center_y": float(config["center_y"]),
            "roll_sign": int(config["roll_sign"]),
            "num_keypoints": int(config["num_keypoints"]),
            "base_channels": int(config["base_channels"]),
            "lr": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "batch_size": int(config["batch_size"]),
            "min_epochs": int(config["min_epochs"]),
            "max_epochs": int(config["max_epochs"]),
            "eval_every": int(config["eval_every"]),
            "plateau_patience": int(config["plateau_patience_epochs"]),
            "relative_improvement": float(config["relative_improvement"]),
            "freeze_backbone": bool(config["freeze_backbone"]),
            "frozen_checkpoint": (
                None
                if config["frozen_checkpoint"] is None
                else Path(config["frozen_checkpoint"])
            ),
            "run_scope": config["run_scope"],
        }
    )
    return argparse.Namespace(**values)


def _load_run_records(
    run_dirs: Iterable[str],
    *,
    expected_seeds: set[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in run_dirs:
        run_dir = Path(raw_path).resolve()
        record = validate_frozen_run_artifacts(
            run_dir, allow_bound_test_artifacts=True
        )
        config = record["config"]
        if config["freeze_backbone"]:
            raise ValueError("frozen probes cannot enter test finalization")
        if config["run_scope"] != "full":
            raise ValueError(f"non-full run cannot enter finalization: {run_dir}")
        if config["extension_authority"] is not None:
            raise ValueError("initial finalization cannot consume extension runs")
        records.append(record)
    combinations = {
        (record["config"]["arm"], int(record["config"]["seed"]))
        for record in records
    }
    expected = {(arm, seed) for arm in ARMS for seed in expected_seeds}
    if combinations != expected:
        missing = sorted(expected - combinations)
        extra = sorted(combinations - expected)
        raise ValueError(f"finalization matrix mismatch; missing={missing}, extra={extra}")
    source_hashes = {record["config"]["source_sha256"] for record in records}
    spec_hashes = {record["config"]["decision_spec_sha256"] for record in records}
    if len(source_hashes) != 1 or len(spec_hashes) != 1:
        raise ValueError("runs do not share one frozen implementation/specification")
    assert_frozen_source_is_current(record["config"] for record in records)
    comparable = (
        "decision_spec_sha256",
        "object",
        "dataset_basename",
        "dataset_semantic_lock_sha256",
        "dataset_index_sha256",
        "operator_reference_sha256",
        "train_validation_content_manifest_sha256",
        "split_sha256",
        "target_shift",
        "center_x",
        "center_y",
        "roll_sign",
        "num_keypoints",
        "base_channels",
        "architecture",
        "heatmap_resolution",
        "temperature",
        "loss",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "min_epochs",
        "max_epochs",
        "eval_every",
        "plateau_patience_epochs",
        "relative_improvement",
        "run_scope",
        "prelaunch_lock",
        "d1_report",
        "git_commit",
        "source_sha256",
        "source_dependencies_sha256",
        "slurm_runfiles_sha256",
        "runtime",
    )
    reference = records[0]["config"]
    for record in records[1:]:
        mismatch = {
            key: (
                matched_config_value(reference, key),
                matched_config_value(record["config"], key),
            )
            for key in comparable
            if matched_config_value(reference, key)
            != matched_config_value(record["config"], key)
        }
        if mismatch:
            raise ValueError(f"runs are not matched: {mismatch}")
    return records


def _strict_checkpoint_preflight(
    records: list[dict[str, Any]],
    template: argparse.Namespace,
) -> None:
    """Strict-load every checkpoint before reserving or reading any test byte."""
    for record in records:
        config = record["config"]
        run_args = _args_from_config(config, template)
        checkpoint = torch.load(
            record["checkpoint_path"], map_location="cpu", weights_only=True
        )
        state_dict = checkpoint.get("extractor_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError("checkpoint lacks an extractor_state_dict")
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"non-tensor checkpoint value: {name}")
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
                tensor
            ).all():
                raise FloatingPointError(f"non-finite checkpoint tensor: {name}")
        extractor = build_extractor(run_args, torch.device("cpu"))
        extractor.load_state_dict(state_dict, strict=True)
        if int(checkpoint.get("epoch", -1)) <= 0:
            raise RuntimeError("checkpoint epoch must be positive")
        record["checkpoint_epoch"] = int(checkpoint["epoch"])
        record["extractor_state_dict"] = state_dict


def _claim_path(output_root: Path, record: dict[str, Any]) -> Path:
    config = record["config"]
    return (
        output_root
        / "test_claims"
        / (
            f"{config['arm']}_seed{int(config['seed'])}_"
            f"{record['checkpoint_sha256']}.jsonl"
        )
    )


def _finalization_plan(
    records: list[dict[str, Any]],
    *,
    event_kind: str,
) -> dict[str, Any]:
    reference = records[0]["config"]
    return {
        "schema_version": 1,
        "event_kind": event_kind,
        "git_commit": reference["git_commit"],
        "source_sha256": reference["source_sha256"],
        "source_dependencies_sha256": reference["source_dependencies_sha256"],
        "slurm_runfiles_sha256": reference["slurm_runfiles_sha256"],
        "decision_spec_sha256": reference["decision_spec_sha256"],
        "prelaunch_lock": reference["prelaunch_lock"],
        "d1_report": reference["d1_report"],
        "targets": sorted(
            [
                {
                    "arm": record["config"]["arm"],
                    "seed": int(record["config"]["seed"]),
                    "config_sha256": record["config_sha256"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                }
                for record in records
            ],
            key=lambda row: (row["arm"], row["seed"]),
        ),
    }


def _validate_test_predictions_npz(
    path: Path,
    metrics: dict[str, Any],
) -> None:
    expected_keys = {
        f"{prefix}_{name}"
        for prefix in ("plain", "augmented")
        for name in (
            "prediction",
            "target",
            "frame",
            "in_range",
            "on_mask",
            "error_cells64",
        )
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise RuntimeError(
                    "test predictions NPZ keys differ from the frozen schema"
                )
            arrays = {key: payload[key] for key in expected_keys}
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"invalid test predictions NPZ: {path}") from exc

    frames_by_condition: dict[str, np.ndarray] = {}
    for prefix, metric_key in (
        ("plain", "unaugmented"),
        ("augmented", "fixed_augmented"),
    ):
        prediction = np.asarray(arrays[f"{prefix}_prediction"], dtype=np.float64)
        target = np.asarray(arrays[f"{prefix}_target"], dtype=np.float64)
        frame = np.asarray(arrays[f"{prefix}_frame"])
        in_range = np.asarray(arrays[f"{prefix}_in_range"])
        on_mask = np.asarray(arrays[f"{prefix}_on_mask"])
        error = np.asarray(arrays[f"{prefix}_error_cells64"], dtype=np.float64)
        if (
            prediction.ndim != 3
            or prediction.shape[-1] != 2
            or target.shape != prediction.shape
            or frame.shape != (prediction.shape[0],)
            or in_range.shape != prediction.shape[:2]
            or on_mask.shape != prediction.shape[:2]
            or error.shape != prediction.shape[:2]
        ):
            raise RuntimeError(
                f"test predictions NPZ shape mismatch for {prefix}: {path}"
            )
        if in_range.dtype != np.bool_ or on_mask.dtype != np.bool_:
            raise RuntimeError(
                f"test predictions NPZ mask dtype mismatch for {prefix}: {path}"
            )
        if (
            not np.isfinite(prediction).all()
            or not np.isfinite(target).all()
            or not np.isfinite(error).all()
            or np.any(error < 0)
        ):
            raise FloatingPointError(
                f"non-finite or negative test array for {prefix}: {path}"
            )
        if len(np.unique(frame)) != len(frame) or not np.array_equal(
            frame, np.sort(frame)
        ):
            raise RuntimeError(
                f"test frame IDs are duplicated or unsorted for {prefix}: {path}"
            )
        recomputed_in_range = np.logical_and(
            prediction >= -1.0, prediction <= 1.0
        ).all(axis=-1)
        recomputed_error = (
            np.linalg.norm(prediction - target, axis=-1) / CELL64_NORM
        )
        if not np.array_equal(in_range, recomputed_in_range):
            raise RuntimeError(
                f"test in-range flags differ from predictions for {prefix}: {path}"
            )
        if not np.allclose(
            error, recomputed_error, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"test errors differ from predictions/targets for {prefix}: {path}"
            )
        condition = metrics.get(metric_key, {})
        channel_medians = np.median(error, axis=0)
        expected_scalars = {
            "median_error_cells64": float(np.median(error)),
            "median_of_channel_medians_cells64": float(
                np.median(channel_medians)
            ),
            "p90_error_cells64": float(np.quantile(error, 0.9)),
            "on_mask_fraction": float(on_mask.mean()),
            "in_range_fraction": float(in_range.mean()),
        }
        for key, expected in expected_scalars.items():
            if not math.isclose(
                float(condition.get(key, math.nan)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"test metric {metric_key}.{key} differs from NPZ arrays"
                )
        if not np.allclose(
            np.asarray(
                condition.get("channel_median_error_cells64", []),
                dtype=np.float64,
            ),
            channel_medians,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"test channel medians differ from NPZ arrays for {prefix}"
            )
        if (
            int(condition.get("out_of_range_count", -1))
            != int((~in_range).sum())
            or int(condition.get("n_frames", -1)) != prediction.shape[0]
            or int(condition.get("n_channel_frame_pairs", -1)) != error.size
        ):
            raise RuntimeError(
                f"test counts differ from NPZ arrays for {prefix}: {path}"
            )
        frames_by_condition[prefix] = frame
    if not np.array_equal(
        frames_by_condition["plain"], frames_by_condition["augmented"]
    ):
        raise RuntimeError("test conditions contain different frame identities")
    selection_score = max(
        float(metrics["unaugmented"]["median_of_channel_medians_cells64"]),
        float(metrics["fixed_augmented"]["median_of_channel_medians_cells64"]),
    )
    if not math.isclose(
        float(metrics.get("selection_score_cells64", math.nan)),
        selection_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("test selection score differs from NPZ-derived metrics")


def _validate_test_claim_artifacts(
    record: dict[str, Any],
    metrics_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    assert_json_finite(metrics)
    if (
        metrics.get("arm") != record["config"]["arm"]
        or int(metrics.get("seed", -1)) != int(record["config"]["seed"])
        or metrics.get("checkpoint_sha256") != record["checkpoint_sha256"]
        or metrics.get("config_sha256") != record["config_sha256"]
        or bool(metrics.get("instrument_pass")) != instrument_pass(metrics)
    ):
        raise RuntimeError(f"completed claim metric identity mismatch: {metrics_path}")
    _validate_test_predictions_npz(predictions_path, metrics)
    return metrics


def _verify_claim_completion(
    record: dict[str, Any],
    *,
    event_id: str,
    claim_path: Path,
    allow_unobservable_retry: bool = False,
) -> dict[str, Any] | None:
    claim_records = read_jsonl(claim_path)
    reserved = claim_records[0]
    expected_reserved = {
        "status": "reserved",
        "event_id": event_id,
        "arm": record["config"]["arm"],
        "seed": int(record["config"]["seed"]),
        "config_sha256": record["config_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
    }
    mismatch = {
        key: (expected, reserved.get(key))
        for key, expected in expected_reserved.items()
        if reserved.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"test claim reservation mismatch: {mismatch}")
    metrics_path = record["run_dir"] / "test_metrics.json"
    predictions_path = record["run_dir"] / "test_predictions.npz"
    artifact_presence = (metrics_path.is_file(), predictions_path.is_file())
    if len(claim_records) == 1:
        if artifact_presence == (False, False) and allow_unobservable_retry:
            return None
        if artifact_presence == (True, True) and allow_unobservable_retry:
            metrics = _validate_test_claim_artifacts(
                record, metrics_path, predictions_path
            )
            recovery_lock = claim_path.with_suffix(".recovery.lock")
            write_jsonl_exclusive(
                recovery_lock,
                {
                    "status": "recovery_reserved",
                    "event_id": event_id,
                    "test_metrics_sha256": sha256_file(metrics_path),
                    "test_predictions_sha256": sha256_file(predictions_path),
                    "reserved_by_slurm": slurm_identity(),
                    "reserved_unix_time_ns": time.time_ns(),
                },
            )
            completion = {
                "status": "completed",
                "event_id": event_id,
                "test_metrics_sha256": sha256_file(metrics_path),
                "test_predictions_sha256": sha256_file(predictions_path),
                "completed_by_slurm": slurm_identity(),
                "completed_unix_time_ns": time.time_ns(),
                "recovered_existing_artifacts": True,
                "recovery_lock_sha256": sha256_file(recovery_lock),
            }
            append_jsonl(claim_path, completion)
            claim_records.append(completion)
        else:
            raise RuntimeError(
                "test claim is reserved with an observable or inconsistent "
                f"partial state; re-evaluation is forbidden: {claim_path}"
            )
    if len(claim_records) != 2 or claim_records[1].get("status") != "completed":
        raise RuntimeError(f"invalid test-claim ledger state: {claim_path}")
    completed = claim_records[1]
    if completed.get("recovered_existing_artifacts"):
        recovery_lock = claim_path.with_suffix(".recovery.lock")
        if (
            not recovery_lock.is_file()
            or sha256_file(recovery_lock)
            != completed.get("recovery_lock_sha256")
        ):
            raise RuntimeError(
                f"recovered claim lacks its immutable recovery lock: {claim_path}"
            )
    for key, path in (
        ("test_metrics_sha256", metrics_path),
        ("test_predictions_sha256", predictions_path),
    ):
        if not path.is_file() or sha256_file(path) != completed.get(key):
            raise RuntimeError(f"completed claim artifact mismatch: {path}")
    metrics = _validate_test_claim_artifacts(
        record, metrics_path, predictions_path
    )
    return {
        **record,
        "test_metrics": metrics,
        "arrays": None,
        "claim_path": claim_path,
        "claim_sha256": sha256_file(claim_path),
        "test_metrics_sha256": completed["test_metrics_sha256"],
        "test_predictions_sha256": completed["test_predictions_sha256"],
    }


def _prepare_finalization_event(
    records: list[dict[str, Any]],
    template: argparse.Namespace,
    *,
    event_kind: str,
    output_paths: tuple[Path, ...],
) -> dict[str, Any]:
    plan = _finalization_plan(records, event_kind=event_kind)
    event_id = canonical_json_sha256(plan)
    event_path = template.output_root / f"{event_kind}_TEST_LEDGER.jsonl"
    claim_paths = {
        (record["config"]["arm"], int(record["config"]["seed"])): _claim_path(
            template.output_root, record
        )
        for record in records
    }
    if not event_path.exists():
        preexisting_outputs = [str(path) for path in output_paths if path.exists()]
        preexisting_artifacts = [
            str(path)
            for record in records
            for path in (
                record["run_dir"] / "test_metrics.json",
                record["run_dir"] / "test_predictions.npz",
                claim_paths[
                    (record["config"]["arm"], int(record["config"]["seed"]))
                ],
                claim_paths[
                    (record["config"]["arm"], int(record["config"]["seed"]))
                ].with_suffix(".recovery.lock"),
            )
            if path.exists()
        ]
        if preexisting_outputs or preexisting_artifacts:
            raise RuntimeError(
                "unbound test-finalization artifacts exist before reservation: "
                f"{preexisting_outputs + preexisting_artifacts}"
            )
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("x") as handle:
            handle.write(
                json.dumps(
                    {
                        "status": "reserved",
                        "event_id": event_id,
                        "plan": plan,
                        "reserved_by_slurm": slurm_identity(),
                        "reserved_unix_time_ns": time.time_ns(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for record in records:
            identity = (
                record["config"]["arm"],
                int(record["config"]["seed"]),
            )
            claim_path = claim_paths[identity]
            write_jsonl_exclusive(
                claim_path,
                {
                    "status": "reserved",
                    "event_id": event_id,
                    "arm": identity[0],
                    "seed": identity[1],
                    "config_sha256": record["config_sha256"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "reserved_by_slurm": slurm_identity(),
                    "reserved_unix_time_ns": time.time_ns(),
                },
            )
        return {
            "state": "fresh",
            "event_id": event_id,
            "event_path": event_path,
            "claim_paths": claim_paths,
            "reserved_by_slurm": slurm_identity(),
            "staged": [],
            "pending_records": records,
        }

    event_records = read_jsonl(event_path)
    reserved = event_records[0]
    if (
        reserved.get("status") != "reserved"
        or reserved.get("event_id") != event_id
        or reserved.get("plan") != json_ready(plan)
    ):
        raise RuntimeError("existing finalization ledger belongs to a different plan")
    if len(event_records) == 2 and event_records[1].get("status") == "completed":
        completed = event_records[1]
        for path in output_paths:
            if (
                not path.is_file()
                or sha256_file(path)
                != completed.get("output_sha256", {}).get(path.name)
            ):
                raise RuntimeError(f"completed finalization output mismatch: {path}")
        return {
            "state": "completed",
            "event_id": event_id,
            "event_path": event_path,
            "claim_paths": claim_paths,
            "reserved_by_slurm": reserved.get("reserved_by_slurm"),
            "staged": [],
            "pending_records": [],
        }
    if len(event_records) != 1:
        raise RuntimeError(f"invalid finalization ledger state: {event_path}")
    preexisting_outputs = [str(path) for path in output_paths if path.exists()]
    if preexisting_outputs:
        raise RuntimeError(
            "aggregate exists before all test claims completed: "
            f"{preexisting_outputs}"
        )
    staged = []
    pending_records = []
    for record in records:
        identity = (record["config"]["arm"], int(record["config"]["seed"]))
        claim_path = claim_paths[identity]
        if not claim_path.is_file():
            raise RuntimeError(
                "finalization reservation is incomplete; exact-once policy "
                f"forbids proceeding: {claim_path}"
            )
        completed_record = _verify_claim_completion(
            record,
            event_id=event_id,
            claim_path=claim_path,
            allow_unobservable_retry=True,
        )
        if completed_record is None:
            pending_records.append(record)
        else:
            staged.append(completed_record)
    return {
        "state": (
            "recover_aggregate" if not pending_records else "resume_incomplete"
        ),
        "event_id": event_id,
        "event_path": event_path,
        "claim_paths": claim_paths,
        "reserved_by_slurm": reserved.get("reserved_by_slurm"),
        "staged": staged,
        "pending_records": pending_records,
    }


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_or_verify_json(path: Path, payload: dict[str, Any]) -> None:
    ready = json_ready(payload)
    assert_json_finite(ready)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != ready:
            raise RuntimeError(f"existing aggregate differs during recovery: {path}")
        return
    write_json_exclusive(path, payload)


def _complete_finalization_event(
    event: dict[str, Any],
    output_paths: tuple[Path, ...],
) -> None:
    event_records = read_jsonl(event["event_path"])
    if len(event_records) == 2:
        return
    append_jsonl(
        event["event_path"],
        {
            "status": "completed",
            "event_id": event["event_id"],
            "output_sha256": {
                path.name: sha256_file(path) for path in output_paths
            },
            "completed_by_slurm": slurm_identity(),
            "completed_unix_time_ns": time.time_ns(),
        },
    )


def _evaluate_frozen_records(
    records: list[dict[str, Any]],
    template: argparse.Namespace,
    *,
    event_kind: str,
    output_paths: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _strict_checkpoint_preflight(records, template)
    event = _prepare_finalization_event(
        records,
        template,
        event_kind=event_kind,
        output_paths=output_paths,
    )
    if event["state"] == "completed":
        return [], event
    if event["state"] == "recover_aggregate":
        return event["staged"], event

    reference_config = records[0]["config"]
    reference_args = _args_from_config(reference_config, template)
    problem = load_scoped_problem(
        reference_args,
        mode=template.mode,
        frozen_config=reference_config,
    )
    assert_semantic_scope(reference_args, problem)
    test_plain = make_scoped_dataset(
        problem,
        mode=template.mode,
        split_name="test",
        augment=False,
        seed=DEFAULT_TEST_AUGMENT_SEED,
    )
    test_augmented = make_scoped_dataset(
        problem,
        mode=template.mode,
        split_name="test",
        augment=True,
        seed=DEFAULT_TEST_AUGMENT_SEED,
    )
    device = choose_device(template.device)
    staged: list[dict[str, Any]] = list(event["staged"])
    for record in event["pending_records"]:
        config = record["config"]
        run_args = _args_from_config(config, template)
        extractor = build_extractor(run_args, device)
        extractor.load_state_dict(record["extractor_state_dict"], strict=True)
        metrics, arrays = evaluate_pair(
            extractor,
            make_eval_loader(test_plain, run_args.batch_size),
            make_eval_loader(test_augmented, run_args.batch_size),
            device,
            split_name="test",
        )
        metrics.update(
            {
                "arm": config["arm"],
                "seed": int(config["seed"]),
                "checkpoint_epoch": int(record["checkpoint_epoch"]),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "config_sha256": record["config_sha256"],
                "test_content_manifest_sha256": problem[
                    "loaded_content_manifest_sha256"
                ],
                "test_content_sha256": problem["loaded_content_sha256"],
                "instrument_pass": instrument_pass(metrics),
                "thresholds": {
                    "both_median_of_channel_medians_cells64_max": 0.50,
                    "both_p90_error_cells64_max": 1.50,
                    "both_on_mask_fraction_min": 0.95,
                },
                "statistical_scope": (
                    "one object and correlated orbit; optimization seed is the "
                    "replication unit; descriptive only"
                ),
            }
        )
        metrics_path = record["run_dir"] / "test_metrics.json"
        predictions_path = record["run_dir"] / "test_predictions.npz"
        write_json_exclusive(metrics_path, metrics)
        _write_npz_exclusive(predictions_path, arrays)
        identity = (config["arm"], int(config["seed"]))
        claim_path = event["claim_paths"][identity]
        completion = {
            "status": "completed",
            "event_id": event["event_id"],
            "test_metrics_sha256": sha256_file(metrics_path),
            "test_predictions_sha256": sha256_file(predictions_path),
            "completed_by_slurm": slurm_identity(),
            "completed_unix_time_ns": time.time_ns(),
        }
        append_jsonl(claim_path, completion)
        staged.append(
            {
                **record,
                "test_metrics": metrics,
                "arrays": None,
                "claim_path": claim_path,
                "claim_sha256": sha256_file(claim_path),
                "test_metrics_sha256": completion["test_metrics_sha256"],
                "test_predictions_sha256": completion[
                    "test_predictions_sha256"
                ],
            }
        )
    staged.sort(
        key=lambda row: (row["config"]["arm"], int(row["config"]["seed"]))
    )
    return staged, event


def _initial_interpretation(statuses: dict[str, str]) -> str:
    if "provisional" in statuses.values():
        return "At least one arm is provisional; no mechanism branch is authorized."
    a = statuses["raw_linear"] == "pass"
    b = statuses["probability_linear"] == "pass"
    c = statuses["fixed_expectation"] == "pass"
    if a and not b and not c:
        return "Raw-logit bypass passes while both softmax-path arms fail."
    if a and b and not c:
        return "Learned spatial decoders pass; softmax itself is not isolated."
    if a and b and c:
        return "All arms pass; matched seed variability dominates the seed-41 result."
    if not a and not b and not c:
        return "All logit-level readouts fail; a broader observability control needs a new spec."
    if not a and b:
        return "Raw-logit parameterization is suspect; do not claim upstream failure."
    if c:
        return "The matched fixed expectation passes; no learned-head repair is justified."
    return "Mixed arm outcome; use the frozen interpretation matrix without extrapolation."


def finalize_initial(args: argparse.Namespace) -> Path:
    expected_seeds = set(INITIAL_SEEDS)
    if len(args.run_dirs) != len(ARMS) * len(expected_seeds):
        raise ValueError("initial finalization requires exactly nine run directories")
    output = args.output_root / "DECISION23_INITIAL_TEST_REPORT.json"
    records = _load_run_records(args.run_dirs, expected_seeds=expected_seeds)
    staged, event = _evaluate_frozen_records(
        records,
        args,
        event_kind="DECISION23_INITIAL",
        output_paths=(output,),
    )
    if event["state"] == "completed":
        return output
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [
            row["test_metrics"] for row in staged if row["config"]["arm"] == arm
        ]
        pass_count = sum(int(row["instrument_pass"]) for row in arm_rows)
        per_arm[arm] = {
            "pass_count": pass_count,
            "total_seeds": 3,
            "status": arm_status(pass_count, 3),
            "per_seed": sorted(arm_rows, key=lambda row: row["seed"]),
        }
    statuses = {arm: payload["status"] for arm, payload in per_arm.items()}
    reference = records[0]["config"]
    content_manifests = {
        row["test_metrics"]["test_content_manifest_sha256"] for row in staged
    }
    if len(content_manifests) != 1:
        raise RuntimeError("initial runs did not evaluate one identical test dataset")
    aggregate = {
        "schema_version": 1,
        "gate": "Decision_2_3_initial_three_seed_test",
        "git_commit": reference["git_commit"],
        "source_sha256": reference["source_sha256"],
        "source_dependencies_sha256": reference["source_dependencies_sha256"],
        "slurm_runfiles_sha256": reference["slurm_runfiles_sha256"],
        "decision_spec_sha256": reference["decision_spec_sha256"],
        "prelaunch_lock": reference["prelaunch_lock"],
        "d1_report": reference["d1_report"],
        "runtime": reference["runtime"],
        "finalizer_slurm": event["reserved_by_slurm"],
        "test_content_manifest_sha256": next(iter(content_manifests)),
        "test_ledger": str(event["event_path"].resolve()),
        "test_ledger_event_id": event["event_id"],
        "test_finalization_event": (
            "one event containing unaugmented and fixed-augmented conditions"
        ),
        "arms": per_arm,
        "seed_rule": {
            "3_of_3": "pass",
            "2_of_3": "provisional; add frozen seeds 45 and 46",
            "0_or_1_of_3": "fail",
            "extension": "both added seeds must pass, yielding 4/5",
        },
        "interpretation": _initial_interpretation(statuses),
        "statistical_reporting": {
            "quantity": "raw per-seed descriptive localization metrics; no error bars",
            "sample_unit": "optimization seed",
            "n": 3,
            "frame_dependence": "60 test frames per seed are one correlated orbit",
            "scope": "one-object instrument capability; no population inference",
        },
        "frozen_runs": [
            {
                "arm": row["config"]["arm"],
                "seed": int(row["config"]["seed"]),
                "run_dir": str(row["run_dir"]),
                "config_sha256": row["config_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_metrics_sha256": row["test_metrics_sha256"],
                "test_predictions_sha256": row["test_predictions_sha256"],
                "test_claim": str(row["claim_path"].resolve()),
                "test_claim_sha256": row["claim_sha256"],
            }
            for row in staged
        ],
    }
    _write_or_verify_json(output, aggregate)
    _complete_finalization_event(event, (output,))
    print(json.dumps(aggregate, indent=2), flush=True)
    return output


def validate_initial_report_for_extension(
    path: Path,
    *,
    expected_output_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute initial authority from immutable run and one-shot artifacts."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != "DECISION23_INITIAL_TEST_REPORT.json":
        raise RuntimeError("extension authority must use the canonical initial report")
    if expected_output_root is not None and path.parent != expected_output_root.resolve():
        raise RuntimeError("initial report belongs to a different output root")
    report = json.loads(path.read_text())
    assert_json_finite(report)
    git = git_identity()
    expected_bindings = {
        "schema_version": 1,
        "gate": "Decision_2_3_initial_three_seed_test",
        "git_commit": git["commit"],
        "source_sha256": sha256_file(Path(__file__)),
        "source_dependencies_sha256": source_dependency_hashes(),
        "slurm_runfiles_sha256": slurm_runfile_hashes(),
        "decision_spec_sha256": sha256_file(SPEC_PATH),
    }
    mismatch = {
        key: (expected, report.get(key))
        for key, expected in expected_bindings.items()
        if report.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"initial report binding mismatch: {mismatch}")
    if git["status_porcelain"]:
        raise RuntimeError("initial report validation requires a clean Git worktree")

    frozen_rows = report.get("frozen_runs", [])
    identities = {
        (row.get("arm"), int(row.get("seed", -1))) for row in frozen_rows
    }
    expected_identities = {(arm, seed) for arm in ARMS for seed in INITIAL_SEEDS}
    if len(frozen_rows) != 9 or identities != expected_identities:
        raise RuntimeError("initial report does not bind the exact 3x3 matrix")

    event_path = path.parent / "DECISION23_INITIAL_TEST_LEDGER.jsonl"
    if Path(report.get("test_ledger", "")).resolve() != event_path:
        raise RuntimeError("initial report points to a non-canonical test ledger")
    event_records = read_jsonl(event_path)
    if (
        len(event_records) != 2
        or event_records[0].get("status") != "reserved"
        or event_records[1].get("status") != "completed"
        or event_records[0].get("event_id") != report.get("test_ledger_event_id")
        or event_records[1].get("event_id") != report.get("test_ledger_event_id")
        or event_records[1].get("output_sha256", {}).get(path.name)
        != sha256_file(path)
    ):
        raise RuntimeError("initial test ledger is incomplete or report hash changed")

    metrics_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    content_manifests: set[str] = set()
    verified_records: list[dict[str, Any]] = []
    for row in frozen_rows:
        arm = row["arm"]
        seed = int(row["seed"])
        run_dir = Path(row["run_dir"]).resolve()
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "test_metrics.json"
        predictions_path = run_dir / "test_predictions.npz"
        claim_path = _claim_path(
            path.parent,
            {
                "config": {"arm": arm, "seed": seed},
                "checkpoint_sha256": row["checkpoint_sha256"],
            },
        )
        expected_paths = {
            "config_sha256": config_path,
            "checkpoint_sha256": checkpoint_path,
            "test_metrics_sha256": metrics_path,
            "test_predictions_sha256": predictions_path,
            "test_claim_sha256": claim_path,
        }
        for key, artifact_path in expected_paths.items():
            if not artifact_path.is_file() or sha256_file(artifact_path) != row.get(key):
                raise RuntimeError(f"initial frozen artifact changed: {artifact_path}")
        if Path(row.get("test_claim", "")).resolve() != claim_path:
            raise RuntimeError("initial report points to a non-canonical claim ledger")
        config = json.loads(config_path.read_text())
        if (
            config.get("arm") != arm
            or int(config.get("seed", -1)) != seed
            or config.get("git_commit") != report["git_commit"]
            or config.get("source_sha256") != report["source_sha256"]
            or config.get("source_dependencies_sha256")
            != report["source_dependencies_sha256"]
            or config.get("slurm_runfiles_sha256")
            != report["slurm_runfiles_sha256"]
            or config.get("decision_spec_sha256") != report["decision_spec_sha256"]
            or config.get("prelaunch_lock") != report["prelaunch_lock"]
            or config.get("d1_report") != report["d1_report"]
        ):
            raise RuntimeError(f"initial frozen config binding mismatch: {config_path}")
        recovered = _verify_claim_completion(
            {
                "run_dir": run_dir,
                "config": config,
                "config_sha256": row["config_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            },
            event_id=report["test_ledger_event_id"],
            claim_path=claim_path,
        )
        metrics = recovered["test_metrics"]
        metrics_by_arm[arm].append(metrics)
        content_manifests.add(metrics["test_content_manifest_sha256"])
        verified_records.append(
            {
                "config": config,
                "config_sha256": row["config_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )

    if content_manifests != {report.get("test_content_manifest_sha256")}:
        raise RuntimeError("initial report test-content binding mismatch")
    expected_plan = _finalization_plan(
        verified_records,
        event_kind="DECISION23_INITIAL",
    )
    if (
        event_records[0].get("plan") != json_ready(expected_plan)
        or canonical_json_sha256(expected_plan) != report["test_ledger_event_id"]
    ):
        raise RuntimeError("initial event plan differs from the frozen run matrix")
    for arm in ARMS:
        metrics = sorted(metrics_by_arm[arm], key=lambda row: int(row["seed"]))
        pass_count = sum(int(instrument_pass(row)) for row in metrics)
        payload = report.get("arms", {}).get(arm, {})
        if (
            payload.get("per_seed") != metrics
            or int(payload.get("pass_count", -1)) != pass_count
            or int(payload.get("total_seeds", -1)) != 3
            or payload.get("status") != arm_status(pass_count, 3)
        ):
            raise RuntimeError(f"initial report arm result is not reproducible: {arm}")
    statuses = {arm: report["arms"][arm]["status"] for arm in ARMS}
    if report.get("interpretation") != _initial_interpretation(statuses):
        raise RuntimeError("initial report interpretation differs from frozen matrix")
    return report


def finalize_extension(args: argparse.Namespace) -> Path:
    if args.initial_report is None:
        raise ValueError("--initial-report is required")
    initial_report = validate_initial_report_for_extension(
        args.initial_report,
        expected_output_root=args.output_root,
    )
    provisional = {
        arm
        for arm, payload in initial_report["arms"].items()
        if payload["status"] == "provisional"
    }
    if not provisional:
        raise ValueError("initial report authorizes no provisional extension")
    expected_seeds = set(EXTENSION_SEEDS)
    expected_count = len(provisional) * len(expected_seeds)
    if len(args.run_dirs) != expected_count:
        raise ValueError(
            f"extension requires {expected_count} run directories for "
            f"{sorted(provisional)}"
        )
    extension_output = args.output_root / "DECISION23_EXTENSION_TEST_REPORT.json"
    combined_output = (
        args.output_root / "DECISION23_COMBINED_FIVE_SEED_TEST_REPORT.json"
    )
    records = _load_run_records_for_extension(
        args.run_dirs,
        provisional_arms=provisional,
        initial_report=initial_report,
        initial_report_path=args.initial_report.resolve(),
        initial_report_sha256=sha256_file(args.initial_report),
    )
    staged, event = _evaluate_frozen_records(
        records,
        args,
        event_kind="DECISION23_EXTENSION",
        output_paths=(extension_output, combined_output),
    )
    if event["state"] == "completed":
        return combined_output
    extension_content_manifests = {
        row["test_metrics"]["test_content_manifest_sha256"] for row in staged
    }
    if extension_content_manifests != {
        initial_report["test_content_manifest_sha256"]
    }:
        raise RuntimeError("extension used different test bytes from the initial event")
    extension_arms: dict[str, Any] = {}
    combined_arms: dict[str, Any] = {
        arm: {
            "initial_pass_count": int(payload["pass_count"]),
            "extension_pass_count": 0,
            "combined_pass_count": int(payload["pass_count"]),
            "total_seeds": 3,
            "status": payload["status"],
            "initial_per_seed": payload["per_seed"],
            "extension_per_seed": [],
        }
        for arm, payload in initial_report["arms"].items()
        if arm not in provisional
    }
    for arm in sorted(provisional):
        extension_rows = [
            row["test_metrics"] for row in staged if row["config"]["arm"] == arm
        ]
        extension_passes = sum(
            int(row["instrument_pass"]) for row in extension_rows
        )
        initial_payload = initial_report["arms"][arm]
        combined_passes = int(initial_payload["pass_count"]) + extension_passes
        extension_arms[arm] = {
            "extension_pass_count": extension_passes,
            "extension_total_seeds": 2,
            "both_added_seeds_pass": extension_passes == 2,
            "per_seed": sorted(extension_rows, key=lambda row: row["seed"]),
        }
        combined_arms[arm] = {
            "initial_pass_count": int(initial_payload["pass_count"]),
            "extension_pass_count": extension_passes,
            "combined_pass_count": combined_passes,
            "total_seeds": 5,
            "status": arm_status(combined_passes, 5),
            "initial_per_seed": initial_payload["per_seed"],
            "extension_per_seed": sorted(
                extension_rows, key=lambda row: row["seed"]
            ),
        }
    extension_aggregate = {
        "schema_version": 1,
        "gate": "Decision_2_3_predeclared_two_seed_extension",
        "initial_report": str(args.initial_report.resolve()),
        "initial_report_sha256": sha256_file(args.initial_report),
        "git_commit": records[0]["config"]["git_commit"],
        "source_sha256": records[0]["config"]["source_sha256"],
        "source_dependencies_sha256": records[0]["config"][
            "source_dependencies_sha256"
        ],
        "slurm_runfiles_sha256": records[0]["config"][
            "slurm_runfiles_sha256"
        ],
        "decision_spec_sha256": records[0]["config"]["decision_spec_sha256"],
        "prelaunch_lock": records[0]["config"]["prelaunch_lock"],
        "d1_report": records[0]["config"]["d1_report"],
        "runtime": records[0]["config"]["runtime"],
        "finalizer_slurm": event["reserved_by_slurm"],
        "test_content_manifest_sha256": next(iter(extension_content_manifests)),
        "test_ledger": str(event["event_path"].resolve()),
        "test_ledger_event_id": event["event_id"],
        "arms": extension_arms,
        "test_finalization_event": (
            "one event containing unaugmented and fixed-augmented conditions"
        ),
        "initial_report_immutable": True,
        "statistical_reporting": {
            "quantity": "raw per-seed descriptive localization metrics; no error bars",
            "sample_unit": "optimization seed",
            "n": 2,
            "scope": "predeclared extension only; no population inference",
        },
        "frozen_runs": [
            {
                "arm": row["config"]["arm"],
                "seed": int(row["config"]["seed"]),
                "run_dir": str(row["run_dir"]),
                "config_sha256": row["config_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_metrics_sha256": row["test_metrics_sha256"],
                "test_predictions_sha256": row["test_predictions_sha256"],
                "test_claim": str(row["claim_path"].resolve()),
                "test_claim_sha256": row["claim_sha256"],
            }
            for row in staged
        ],
    }
    combined_statuses = {
        arm: payload["status"] for arm, payload in combined_arms.items()
    }
    combined_aggregate = {
        "schema_version": 1,
        "gate": "Decision_2_3_combined_five_seed_result",
        "initial_report": str(args.initial_report.resolve()),
        "initial_report_sha256": sha256_file(args.initial_report),
        "extension_report": str(extension_output.resolve()),
        "git_commit": records[0]["config"]["git_commit"],
        "source_sha256": records[0]["config"]["source_sha256"],
        "source_dependencies_sha256": records[0]["config"][
            "source_dependencies_sha256"
        ],
        "slurm_runfiles_sha256": records[0]["config"][
            "slurm_runfiles_sha256"
        ],
        "decision_spec_sha256": records[0]["config"]["decision_spec_sha256"],
        "arms": combined_arms,
        "rule": "both seeds 45 and 46 must pass, giving at least 4/5",
        "interpretation": _initial_interpretation(combined_statuses),
        "initial_report_immutable": True,
        "statistical_reporting": {
            "quantity": "raw per-seed descriptive localization metrics; no error bars",
            "sample_unit": "optimization seed",
            "n": 5,
            "scope": "one-object instrument capability; no population inference",
        },
    }
    _write_or_verify_json(extension_output, extension_aggregate)
    combined_aggregate["extension_report_sha256"] = sha256_file(extension_output)
    _write_or_verify_json(combined_output, combined_aggregate)
    _complete_finalization_event(event, (extension_output, combined_output))
    print(
        json.dumps(
            {
                "extension": extension_aggregate,
                "combined": combined_aggregate,
            },
            indent=2,
        ),
        flush=True,
    )
    return combined_output


def _load_run_records_for_extension(
    run_dirs: Iterable[str],
    *,
    provisional_arms: set[str],
    initial_report: dict[str, Any],
    initial_report_path: Path,
    initial_report_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    initial_frozen = initial_report["frozen_runs"]
    if not any(row["arm"] in provisional_arms for row in initial_frozen):
        raise ValueError("initial report lacks frozen provisional-arm records")
    for raw_path in run_dirs:
        run_dir = Path(raw_path).resolve()
        record = validate_frozen_run_artifacts(
            run_dir, allow_bound_test_artifacts=True
        )
        config = record["config"]
        if config["arm"] not in provisional_arms:
            raise ValueError(f"arm {config['arm']} was not provisional")
        if int(config["seed"]) not in EXTENSION_SEEDS:
            raise ValueError("extension seeds are frozen at 45 and 46")
        if config["freeze_backbone"] or config["run_scope"] != "full":
            raise ValueError("extension accepts end-to-end full runs only")
        authority = config.get("extension_authority")
        if authority is None:
            raise ValueError("extension run lacks frozen initial-report authority")
        if authority.get("sha256") != initial_report_sha256:
            raise ValueError("extension run used a different initial report")
        if Path(authority.get("path", "")).resolve() != initial_report_path:
            raise ValueError("extension authority path differs from finalizer input")
        if authority.get("arm") != config["arm"]:
            raise ValueError("extension authority arm mismatch")
        records.append(record)
    combinations = {
        (row["config"]["arm"], int(row["config"]["seed"])) for row in records
    }
    expected = {
        (arm, seed) for arm in provisional_arms for seed in EXTENSION_SEEDS
    }
    if combinations != expected:
        raise ValueError(
            f"extension matrix mismatch; missing={sorted(expected-combinations)}, "
            f"extra={sorted(combinations-expected)}"
        )
    # Every extension config must match one initial config for its arm on all
    # non-seed recipe fields.  The initial report keeps config hashes and test
    # metrics immutable; configs are loaded from their recorded run directories.
    initial_by_arm: dict[str, dict[str, Any]] = {}
    for row in initial_frozen:
        arm = row["arm"]
        if arm not in provisional_arms or arm in initial_by_arm:
            continue
        initial_config_path = Path(row["run_dir"]) / "config.json"
        if sha256_file(initial_config_path) != row["config_sha256"]:
            raise RuntimeError("initial frozen config hash changed")
        initial_by_arm[arm] = json.loads(initial_config_path.read_text())
    comparable = (
        "decision_spec_sha256",
        "arm",
        "object",
        "dataset_basename",
        "dataset_semantic_lock_sha256",
        "dataset_index_sha256",
        "operator_reference_sha256",
        "train_validation_content_manifest_sha256",
        "split_sha256",
        "target_shift",
        "center_x",
        "center_y",
        "roll_sign",
        "num_keypoints",
        "base_channels",
        "architecture",
        "heatmap_resolution",
        "temperature",
        "loss",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "min_epochs",
        "max_epochs",
        "eval_every",
        "plateau_patience_epochs",
        "relative_improvement",
        "run_scope",
        "prelaunch_lock",
        "d1_report",
        "git_commit",
        "source_sha256",
        "source_dependencies_sha256",
        "slurm_runfiles_sha256",
        "runtime",
    )
    for row in records:
        reference = initial_by_arm[row["config"]["arm"]]
        mismatch = {
            key: (
                matched_config_value(reference, key),
                matched_config_value(row["config"], key),
            )
            for key in comparable
            if matched_config_value(reference, key)
            != matched_config_value(row["config"], key)
        }
        if mismatch:
            raise ValueError(f"extension changed frozen recipe: {mismatch}")
    source_hashes = {row["config"]["source_sha256"] for row in records}
    if source_hashes != {sha256_file(Path(__file__))}:
        raise RuntimeError(
            "extension finalizer source differs from frozen run implementation"
        )
    spec_hashes = {row["config"]["decision_spec_sha256"] for row in records}
    if spec_hashes != {sha256_file(SPEC_PATH)}:
        raise RuntimeError(
            "extension finalizer specification differs from frozen runs"
        )
    assert_frozen_source_is_current(row["config"] for row in records)
    return records


def write_lock(args: argparse.Namespace) -> Path:
    assert_recipe_scope(args)
    problem = load_scoped_problem(args, mode=args.mode)
    assert_semantic_scope(args, problem)
    git = git_identity()
    if git["status_porcelain"]:
        raise RuntimeError("prelaunch lock requires a clean Git worktree")
    payload = {
        "schema_version": 1,
        "created_before_decision23_runs": True,
        "decision_spec": str(SPEC_PATH.resolve()),
        "decision_spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "source_dependencies_sha256": source_dependency_hashes(),
        "slurm_runfiles_sha256": slurm_runfile_hashes(),
        "dataset_basename": args.data_root.name,
        "dataset_semantic_lock_sha256": sha256_file(
            args.data_root / "semantic_lock.json"
        ),
        "dataset_index_sha256": sha256_file(args.data_root / "dataset_index.json"),
        "operator_reference_sha256": sha256_file(
            args.data_root / "operator_reference.json"
        ),
        "train_validation_content_sha256": problem["loaded_content_sha256"],
        "train_validation_content_manifest_sha256": problem[
            "loaded_content_manifest_sha256"
        ],
        "split_json": str(problem["split_path"].resolve()),
        "split_sha256": problem["split"].sha256,
        "object": args.object,
        "target_shift": int(args.target_shift),
        "center_x": float(args.center_x),
        "center_y": float(args.center_y),
        "roll_sign": int(args.roll_sign),
        "arms": list(ARMS),
        "initial_seeds": list(INITIAL_SEEDS),
        "extension_seeds": list(EXTENSION_SEEDS),
        "seed_rule": {
            "3_of_3": "pass",
            "2_of_3": "provisional; run both extension seeds unchanged",
            "0_or_1_of_3": "fail",
            "4_of_5_after_extension": "pass",
            "3_or_fewer_of_5_after_extension": "fail",
        },
        "thresholds": {
            "both_median_of_channel_medians_cells64_max": 0.50,
            "both_p90_error_cells64_max": 1.50,
            "both_on_mask_fraction_min": 0.95,
        },
        "full_recipe": FULL_RECIPE,
        "smoke_recipe": SMOKE_RECIPE,
        "slurm_resources": {
            "d1": "1 GPU, 8 CPUs, 5000 MB per CPU, 30 minutes, sequential arms",
            "d2": "array 0-8%2; 1 GPU, 8 CPUs, 5000 MB per CPU, 1 hour per task",
        },
        "test_policy": (
            "training/probe never load test; every frozen checkpoint is "
            "evaluated on test exactly once"
        ),
        "statistical_scope": (
            "one object and correlated orbit; seed is optimization replicate; "
            "descriptive only"
        ),
        "git": git,
        "runtime": runtime_identity(None),
    }
    output = args.output_root / "DECISION23_PRELAUNCH_LOCK.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != json_ready(payload):
            raise RuntimeError(f"prelaunch lock differs: {output}")
    else:
        write_json(output, payload)
    print(json.dumps(json_ready(payload), indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("lock", "train", "probe", "finalize", "finalize-extension"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--object", default=EXPECTED_OBJECT)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs" / "decision23")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-keypoints", type=int, default=10)
    parser.add_argument("--target-shift", type=int, default=EXPECTED_TARGET_SHIFT)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-epochs", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--plateau-patience", type=int, default=400)
    parser.add_argument("--relative-improvement", type=float, default=0.01)
    parser.add_argument("--center-x", type=float, default=EXPECTED_CENTER_X)
    parser.add_argument("--center-y", type=float, default=EXPECTED_CENTER_Y)
    parser.add_argument(
        "--roll-sign", type=int, choices=(-1, 1), default=EXPECTED_ROLL_SIGN
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-scope", choices=("smoke", "full"), default="full")
    parser.add_argument("--frozen-checkpoint", type=Path)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--initial-report", type=Path)
    parser.add_argument("--d1-report", type=Path)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.freeze_backbone = args.mode == "probe"
    if args.mode in {"lock", "train", "probe"} and args.data_root is None:
        parser.error("--data-root is required")
    if args.mode in {"train", "probe"} and args.arm is None:
        parser.error("--arm is required")
    if args.mode == "probe" and args.arm not in LEARNED_ARMS:
        parser.error("probe arm must be raw_linear or probability_linear")
    if args.mode == "probe" and args.frozen_checkpoint is None:
        parser.error("--frozen-checkpoint is required for probe")
    if args.mode == "finalize" and len(args.run_dirs) != 9:
        parser.error("finalize requires nine --run-dirs")
    if args.mode == "finalize-extension" and not args.run_dirs:
        parser.error("finalize-extension requires --run-dirs")
    if args.eval_every <= 0 or args.plateau_patience % args.eval_every != 0:
        parser.error("plateau patience must be a positive multiple of eval-every")
    if args.min_epochs > args.max_epochs:
        parser.error("min-epochs cannot exceed max-epochs")
    if args.mode == "train" and args.seed not in (*INITIAL_SEEDS, *EXTENSION_SEEDS):
        parser.error("training seed must be one of 42,43,44,45,46")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "lock":
        write_lock(args)
    elif args.mode in {"train", "probe"}:
        train(args)
    elif args.mode == "finalize":
        finalize_initial(args)
    else:
        finalize_extension(args)


if __name__ == "__main__":
    main()
