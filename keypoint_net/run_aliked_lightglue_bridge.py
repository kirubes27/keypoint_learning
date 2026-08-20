#!/usr/bin/env python3
"""Run the frozen two-stage ALIKED + LightGlue material-point bridge.

The ``infer`` stage reads RGB only.  It directly matches seed frame 27 to
every target frame, selects ten identities using frames 27..176 only, writes
and hashes raw predictions, and never loads geometry or masks.  The separate
``evaluate`` stage may then load physical theta and masks to score the frozen
raw coordinates under the externally calibrated eradication contract.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_PACKAGES = REPO_ROOT / "third_party" / "python_pkgs"
LIGHTGLUE_ROOT = REPO_ROOT / "third_party" / "LightGlue"
MODEL_CACHE_ROOT = REPO_ROOT / "third_party" / "model_cache"
for dependency_path in (ISOLATED_PACKAGES, LIGHTGLUE_ROOT):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))
os.environ["TORCH_HOME"] = str(MODEL_CACHE_ROOT)

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from keypoint_net.aliked_lightglue_bridge import (  # noqa: E402
    AlikedBridgeError,
    AlikedLightGlueConfig,
    DirectMatches,
    config_as_dict,
    predict_selected_identities,
    select_train_identities,
)
from keypoint_net.run_sift_bridge import (  # noqa: E402
    FRAME_COUNT,
    GUARD_FRAMES,
    HOLDOUT_FRAMES,
    IDENTITY_COLORS,
    OBJECT_ID,
    TRAIN_FRAMES,
    canonical_json_bytes,
    deterministic_npz_bytes,
    frame_paths,
    git,
    load_meta,
    load_rgb,
    partition_metrics,
    pixel_to_normalized,
    require,
    rotate_normalized,
    save_figure_no_overwrite,
    sha256_file,
    strict_json,
    write_json_no_overwrite,
)


class AlikedRunnerError(RuntimeError):
    """Raised when provenance or stage ordering invalidates the bridge."""


def local_require(condition: bool, message: str) -> None:
    if not condition:
        raise AlikedRunnerError(message)


def exact_equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def config_from_lock(lock: Mapping[str, Any], config_path: Path) -> AlikedLightGlueConfig:
    local_require(
        lock.get("schema") == "frozen_aliked_lightglue_bridge_config.r1",
        "wrong ALIKED + LightGlue config schema",
    )
    bridge = lock["bridge"]
    detector = bridge["detector"]
    matcher = bridge["matcher"]
    runtime = bridge["runtime"]
    config = AlikedLightGlueConfig(
        n_identities=int(bridge["n_identities"]),
        seed_frame_index=int(bridge["seed_frame_index"]),
        seed_separation_px=float(bridge["seed_separation_px"]),
        model_name=str(detector["model_name"]),
        max_num_keypoints=int(detector["max_num_keypoints"]),
        detection_threshold=float(detector["detection_threshold"]),
        nms_radius=int(detector["nms_radius"]),
        n_layers=int(matcher["n_layers"]),
        depth_confidence=float(matcher["depth_confidence"]),
        width_confidence=float(matcher["width_confidence"]),
        filter_threshold=float(matcher["filter_threshold"]),
        flash=bool(matcher["flash"]),
        mixed_precision=bool(matcher["mp"]),
    )
    config.validate()
    local_require(config == AlikedLightGlueConfig(), "config differs from frozen R1 defaults")
    local_require(detector["resize"] is None, "R1 must preserve native 512 x 512 resolution")
    local_require(runtime == {
        "device": "cpu",
        "eval_mode": True,
        "inference_mode": True,
        "gradients": False,
        "training": False,
        "compilation": False,
    }, "runtime lock differs")

    semantic_path = config_path.parent / "00_SEMANTIC_LOCK.md"
    local_require(semantic_path.is_file(), "semantic lock is missing")
    local_require(
        sha256_file(semantic_path)[0] == lock["semantic_lock_sha256"],
        "semantic lock hash differs",
    )
    return config


def git_at(root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AlikedRunnerError(f"git command failed in {root}: {' '.join(args)}") from exc


def verify_external_runtime(lock: Mapping[str, Any]) -> dict[str, Any]:
    upstream = lock["upstream"]
    upstream_root = Path(upstream["local_path"]).resolve(strict=True)
    local_require(upstream_root == LIGHTGLUE_ROOT.resolve(), "unexpected LightGlue root")
    local_require(git_at(upstream_root, ["rev-parse", "HEAD"]) == upstream["commit"], "LightGlue commit differs")
    local_require(git_at(upstream_root, ["status", "--porcelain"]) == "", "LightGlue checkout is dirty")
    for path, expected in (
        (upstream_root / "LICENSE", upstream["license_sha256"]),
        (upstream_root / "lightglue" / "aliked.py", upstream["aliked_source_sha256"]),
        (upstream_root / "lightglue" / "lightglue.py", upstream["lightglue_source_sha256"]),
    ):
        local_require(sha256_file(path)[0] == expected, f"upstream file hash differs: {path}")

    verified_weights: dict[str, Any] = {}
    for name in ("aliked", "lightglue"):
        row = lock["weights"][name]
        path = Path(row["path"]).resolve(strict=True)
        digest, size = sha256_file(path)
        local_require(digest == row["sha256"], f"{name} weight hash differs")
        local_require(size == int(row["size_bytes"]), f"{name} weight size differs")
        verified_weights[name] = {"path": str(path), "sha256": digest, "size_bytes": size, "url": row["url"]}

    dependencies = lock["runtime_dependencies"]
    for name, relative in (
        ("kornia", "kornia-0.6.12.dist-info/METADATA"),
        ("opencv_python_headless", "opencv_python_headless-4.12.0.88.dist-info/METADATA"),
    ):
        row = dependencies[name]
        root = Path(row["isolated_target"]).resolve(strict=True)
        local_require(root == ISOLATED_PACKAGES.resolve(), f"unexpected {name} target")
        local_require(
            sha256_file(root / relative)[0] == row["metadata_sha256"],
            f"{name} metadata hash differs",
        )
    local_require(torch.__version__ == dependencies["torch"], "torch version differs")
    local_require(np.__version__ == dependencies["numpy"], "numpy version differs")
    local_require(cv2.__version__ == "4.12.0", "OpenCV runtime version differs")
    return {
        "upstream_root": str(upstream_root),
        "upstream_commit": upstream["commit"],
        "upstream_clean": True,
        "weights": verified_weights,
    }


def create_models(config: AlikedLightGlueConfig):
    from lightglue import ALIKED, LightGlue

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    extractor = ALIKED(
        model_name=config.model_name,
        max_num_keypoints=config.max_num_keypoints,
        detection_threshold=config.detection_threshold,
        nms_radius=config.nms_radius,
    ).eval().cpu()
    matcher = LightGlue(
        features="aliked",
        n_layers=config.n_layers,
        depth_confidence=config.depth_confidence,
        width_confidence=config.width_confidence,
        filter_threshold=config.filter_threshold,
        flash=config.flash,
        mp=config.mixed_precision,
    ).eval().cpu()
    extractor.requires_grad_(False)
    matcher.requires_grad_(False)
    local_require(not any(parameter.requires_grad for parameter in extractor.parameters()), "extractor is not frozen")
    local_require(not any(parameter.requires_grad for parameter in matcher.parameters()), "matcher is not frozen")
    return extractor, matcher


def image_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = np.asarray(image_rgb)
    local_require(image.shape == (512, 512, 3) and image.dtype == np.uint8, "unexpected RGB input")
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)


def extract_features(extractor, image_rgb: np.ndarray) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        features = extractor.extract(image_tensor(image_rgb), resize=None)
    required = {"keypoints", "descriptors", "keypoint_scores", "image_size"}
    local_require(required <= set(features), "ALIKED feature output is incomplete")
    for name in required:
        local_require(isinstance(features[name], torch.Tensor), f"feature {name} is not a tensor")
        local_require(features[name].device.type == "cpu", f"feature {name} left CPU")
        local_require(torch.isfinite(features[name]).all().item(), f"feature {name} is non-finite")
    local_require(features["keypoints"].shape[0] == 1, "ALIKED batch size differs")
    local_require(features["keypoints"].shape[-1] == 2, "ALIKED keypoint shape differs")
    local_require(features["descriptors"].shape[:2] == features["keypoints"].shape[:2], "descriptor count differs")
    local_require(features["keypoint_scores"].shape == features["keypoints"].shape[:2], "score count differs")
    return features


def feature_arrays(features: Mapping[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    xy = features["keypoints"][0].detach().cpu().numpy().astype(np.float64, copy=True)
    score = features["keypoint_scores"][0].detach().cpu().numpy().astype(np.float64, copy=True)
    return xy, score


def match_direct(matcher, seed_features: Mapping[str, torch.Tensor], target_features: Mapping[str, torch.Tensor]) -> DirectMatches:
    with torch.inference_mode():
        result = matcher({"image0": seed_features, "image1": target_features})
    target = result["matches0"][0].detach().cpu().numpy().astype(np.int64, copy=True)
    score = result["matching_scores0"][0].detach().cpu().numpy().astype(np.float64, copy=True)
    direct = DirectMatches(target_index=target, score=score)
    direct.validate(
        seed_count=int(seed_features["keypoints"].shape[1]),
        target_count=int(target_features["keypoints"].shape[1]),
    )
    return direct


def write_npz_no_overwrite(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    local_require(not path.exists(), f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(deterministic_npz_bytes(arrays))


def smoke_stage(dataset_root: Path, config_path: Path, output_root: Path) -> None:
    """Run a train-only operational smoke without opening holdout/guard RGB."""

    local_require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    lock = strict_json(config_path)
    config = config_from_lock(lock, config_path)
    external = verify_external_runtime(lock)
    paths = frame_paths(dataset_root)
    started = time.time()
    extractor, matcher = create_models(config)
    train_images = {frame: load_rgb(paths[frame]) for frame in TRAIN_FRAMES}
    seed_features = extract_features(extractor, train_images[config.seed_frame_index])
    repeated_seed = extract_features(extractor, train_images[config.seed_frame_index])
    seed_repeatable = all(
        torch.equal(seed_features[name], repeated_seed[name])
        for name in ("keypoints", "descriptors", "keypoint_scores", "image_size")
    )
    local_require(seed_repeatable, "train smoke seed extraction is not exactly repeatable")
    seed_xy, seed_score = feature_arrays(seed_features)
    seed_count = seed_xy.shape[0]
    local_require(seed_count >= config.n_identities, "too few ALIKED seed detections")
    print(json.dumps({"event": "smoke_seed_extracted", "seed_count": seed_count}), flush=True)

    features_by_frame: dict[int, dict[str, torch.Tensor]] = {}
    xy_by_frame: dict[int, np.ndarray] = {}
    forward: dict[int, DirectMatches] = {}
    for completed, frame in enumerate(TRAIN_FRAMES, start=1):
        features = seed_features if frame == config.seed_frame_index else extract_features(extractor, train_images[frame])
        xy, _ = feature_arrays(features)
        direct = match_direct(matcher, seed_features, features)
        features_by_frame[frame] = features
        xy_by_frame[frame] = xy
        forward[frame] = direct
        if completed % 10 == 0 or completed == len(TRAIN_FRAMES):
            print(
                json.dumps(
                    {
                        "event": "smoke_forward_progress",
                        "completed": completed,
                        "frame": frame,
                        "detections": int(xy.shape[0]),
                        "direct_matches": int(np.count_nonzero(direct.target_index >= 0)),
                    }
                ),
                flush=True,
            )

    reverse_target = np.full((len(TRAIN_FRAMES), seed_count), -1, dtype=np.int64)
    reverse_score = np.zeros((len(TRAIN_FRAMES), seed_count), dtype=np.float64)
    frame_to_row = {frame: row for row, frame in enumerate(TRAIN_FRAMES)}
    for completed, frame in enumerate(reversed(TRAIN_FRAMES), start=1):
        direct = match_direct(matcher, seed_features, features_by_frame[frame])
        row = frame_to_row[frame]
        reverse_target[row] = direct.target_index
        reverse_score[row] = direct.score
        if completed % 20 == 0 or completed == len(TRAIN_FRAMES):
            print(json.dumps({"event": "smoke_reverse_progress", "completed": completed}), flush=True)
    forward_target = np.stack([forward[frame].target_index for frame in TRAIN_FRAMES], axis=0)
    forward_score = np.stack([forward[frame].score for frame in TRAIN_FRAMES], axis=0)
    order_invariant = bool(
        np.array_equal(forward_target, reverse_target)
        and np.array_equal(forward_score, reverse_score)
    )
    local_require(order_invariant, "train-only matching depends on frame processing order")

    identities = select_train_identities(
        seed_xy,
        seed_score,
        forward,
        {frame: int(xy_by_frame[frame].shape[0]) for frame in TRAIN_FRAMES},
        TRAIN_FRAMES,
        config,
    )
    accepted = np.zeros((len(TRAIN_FRAMES), config.n_identities), dtype=bool)
    coordinate = np.full((len(TRAIN_FRAMES), config.n_identities, 2), np.nan, dtype=np.float64)
    target_index = np.full((len(TRAIN_FRAMES), config.n_identities), -1, dtype=np.int64)
    score = np.zeros((len(TRAIN_FRAMES), config.n_identities), dtype=np.float64)
    for row, frame in enumerate(TRAIN_FRAMES):
        coordinate[row], accepted[row], target_index[row], score[row] = predict_selected_identities(
            identities, forward[frame], xy_by_frame[frame]
        )
    local_require(np.array_equal(np.isfinite(coordinate).all(axis=-1), accepted), "smoke missing-data representation differs")

    raw_path = output_root / "TRAIN_ONLY_SMOKE_RAW.npz"
    write_npz_no_overwrite(
        raw_path,
        {
            "train_frame_index": np.asarray(TRAIN_FRAMES, dtype=np.int64),
            "coordinate_px": coordinate,
            "accepted": accepted,
            "target_index": target_index,
            "match_score": score,
            "seed_xy_px_all": seed_xy,
            "seed_detector_score_all": seed_score,
            "selected_seed_index": identities.selected_seed_indices,
            "seed_xy_px": identities.seed_xy_px,
            "seed_detector_score": identities.seed_detector_score,
            "train_coverage": identities.train_coverage,
            "train_median_match_score": identities.train_median_match_score,
            "complete_seed_ranking": identities.complete_seed_ranking,
        },
    )
    raw_hash, raw_size = sha256_file(raw_path)
    identity_lock = {
        "schema": "aliked_lightglue_train_smoke_identity_lock.r1",
        "config": config_as_dict(config),
        "train_frame_indices": list(TRAIN_FRAMES),
        "selected_identity_rows": [
            {
                "identity": identity,
                "seed_candidate_index": int(identities.selected_seed_indices[identity]),
                "seed_xy_px": identities.seed_xy_px[identity].tolist(),
                "seed_detector_score": float(identities.seed_detector_score[identity]),
                "train_coverage": float(identities.train_coverage[identity]),
                "train_median_match_score": float(identities.train_median_match_score[identity]),
            }
            for identity in range(config.n_identities)
        ],
        "selection_used_holdout_or_guard": False,
        "global_transform_used": False,
        "temporal_state_used": False,
    }
    write_json_no_overwrite(output_root / "TRAIN_ONLY_SMOKE_IDENTITY_LOCK.json", identity_lock)
    receipt = {
        "schema": "aliked_lightglue_train_only_smoke_receipt.r1",
        "status": "operational_smoke_pass",
        "completed_unix_seconds": time.time(),
        "duration_seconds": time.time() - started,
        "external_runtime": external,
        "source": {
            "git_head": git(["rev-parse", "HEAD"]),
            "git_branch": git(["branch", "--show-current"]),
            "runner_sha256": sha256_file(REPO_ROOT / "keypoint_net" / "run_aliked_lightglue_bridge.py")[0],
            "core_sha256": sha256_file(REPO_ROOT / "keypoint_net" / "aliked_lightglue_bridge.py")[0],
            "config_sha256": sha256_file(config_path)[0],
        },
        "raw": {"path": str(raw_path), "sha256": raw_hash, "size_bytes": raw_size},
        "seed_extraction_repeatable_exact": seed_repeatable,
        "frame_order_invariance_exact": order_invariant,
        "ten_identities_frozen_from_train_only": identities.selected_seed_indices.size == config.n_identities,
        "accepted_per_identity_on_train": np.sum(accepted, axis=0).astype(int).tolist(),
        "missing_per_identity_on_train": np.sum(~accepted, axis=0).astype(int).tolist(),
        "strict_train_coverage_feasible": bool(np.all(accepted)),
        "missing_predictions_filled": False,
        "global_transform_used": False,
        "temporal_state_used": False,
        "training_performed": False,
        "gpu_used": False,
        "forbidden_inputs_loaded": [],
        "opened_rgb_frames": list(TRAIN_FRAMES),
        "holdout_or_guard_rgb_opened": False,
    }
    write_json_no_overwrite(output_root / "SMOKE_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


def infer_stage(dataset_root: Path, config_path: Path, output_root: Path) -> None:
    local_require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    lock = strict_json(config_path)
    config = config_from_lock(lock, config_path)
    external = verify_external_runtime(lock)
    paths = frame_paths(dataset_root)
    images = [load_rgb(path) for path in paths]

    started = time.time()
    extractor, matcher = create_models(config)
    seed_features = extract_features(extractor, images[config.seed_frame_index])
    repeated_seed = extract_features(extractor, images[config.seed_frame_index])
    seed_repeatable = all(
        torch.equal(seed_features[name], repeated_seed[name])
        for name in ("keypoints", "descriptors", "keypoint_scores", "image_size")
    )
    local_require(seed_repeatable, "fresh ALIKED seed extraction is not exactly repeatable")
    seed_xy, seed_score = feature_arrays(seed_features)
    seed_count = seed_xy.shape[0]
    local_require(seed_count >= config.n_identities, "too few ALIKED seed detections")
    print(json.dumps({"event": "seed_extracted", "seed_count": seed_count}), flush=True)

    target_features: list[dict[str, torch.Tensor]] = []
    target_xy: list[np.ndarray] = []
    target_score: list[np.ndarray] = []
    forward_matches: list[DirectMatches] = []
    for frame in range(FRAME_COUNT):
        features = seed_features if frame == config.seed_frame_index else extract_features(extractor, images[frame])
        xy, score = feature_arrays(features)
        direct = match_direct(matcher, seed_features, features)
        target_features.append(features)
        target_xy.append(xy)
        target_score.append(score)
        forward_matches.append(direct)
        if frame % 10 == 0 or frame == FRAME_COUNT - 1:
            print(
                json.dumps(
                    {
                        "event": "forward_progress",
                        "frame": frame,
                        "detections": int(xy.shape[0]),
                        "direct_matches": int(np.count_nonzero(direct.target_index >= 0)),
                    }
                ),
                flush=True,
            )

    reverse_target = np.full((FRAME_COUNT, seed_count), -1, dtype=np.int64)
    reverse_score = np.zeros((FRAME_COUNT, seed_count), dtype=np.float64)
    for completed, frame in enumerate(reversed(range(FRAME_COUNT)), start=1):
        direct = match_direct(matcher, seed_features, target_features[frame])
        reverse_target[frame] = direct.target_index
        reverse_score[frame] = direct.score
        if completed % 20 == 0 or completed == FRAME_COUNT:
            print(json.dumps({"event": "reverse_progress", "completed": completed}), flush=True)

    all_target_index = np.stack([value.target_index for value in forward_matches], axis=0)
    all_match_score = np.stack([value.score for value in forward_matches], axis=0)
    frame_order_invariant = bool(
        np.array_equal(all_target_index, reverse_target)
        and np.array_equal(all_match_score, reverse_score)
    )
    local_require(frame_order_invariant, "direct matching depends on frame processing order")

    train_matches = {frame: forward_matches[frame] for frame in TRAIN_FRAMES}
    train_counts = {frame: int(target_xy[frame].shape[0]) for frame in TRAIN_FRAMES}
    identities = select_train_identities(
        seed_xy,
        seed_score,
        train_matches,
        train_counts,
        TRAIN_FRAMES,
        config,
    )

    coordinates = np.full((FRAME_COUNT, config.n_identities, 2), np.nan, dtype=np.float64)
    accepted = np.zeros((FRAME_COUNT, config.n_identities), dtype=bool)
    selected_target_index = np.full((FRAME_COUNT, config.n_identities), -1, dtype=np.int64)
    selected_match_score = np.zeros((FRAME_COUNT, config.n_identities), dtype=np.float64)
    for frame in range(FRAME_COUNT):
        values = predict_selected_identities(identities, forward_matches[frame], target_xy[frame])
        coordinates[frame], accepted[frame], selected_target_index[frame], selected_match_score[frame] = values

    offsets = [0]
    for xy in target_xy:
        offsets.append(offsets[-1] + xy.shape[0])
    arrays = {
        "coordinate_px": coordinates,
        "accepted": accepted,
        "target_index": selected_target_index,
        "match_score": selected_match_score,
        "frame_index": np.arange(FRAME_COUNT, dtype=np.int64),
        "detection_count": np.asarray([xy.shape[0] for xy in target_xy], dtype=np.int64),
        "target_offset": np.asarray(offsets, dtype=np.int64),
        "target_xy_px": np.concatenate(target_xy, axis=0),
        "target_detector_score": np.concatenate(target_score, axis=0),
        "all_seed_match_target_index": all_target_index,
        "all_seed_match_score": all_match_score,
        "seed_xy_px_all": seed_xy,
        "seed_detector_score_all": seed_score,
        "selected_seed_index": identities.selected_seed_indices,
        "seed_xy_px": identities.seed_xy_px,
        "seed_detector_score": identities.seed_detector_score,
        "train_coverage": identities.train_coverage,
        "train_median_match_score": identities.train_median_match_score,
        "complete_seed_ranking": identities.complete_seed_ranking,
    }
    raw_path = output_root / "RAW_ALIKED_LIGHTGLUE_PREDICTIONS.npz"
    raw_payload = deterministic_npz_bytes(arrays)
    local_require(raw_payload == deterministic_npz_bytes(arrays), "deterministic NPZ encoding failed")
    with raw_path.open("xb") as handle:
        handle.write(raw_payload)
    raw_hash, raw_size = sha256_file(raw_path)

    template_lock = {
        "schema": "frozen_aliked_lightglue_identity_lock.r1",
        "config": config_as_dict(config),
        "train_frame_indices": list(TRAIN_FRAMES),
        "identity_rows": [
            {
                "identity": identity,
                "label": f"ALIKED{identity}",
                "seed_candidate_index": int(identities.selected_seed_indices[identity]),
                "seed_xy_px": identities.seed_xy_px[identity].tolist(),
                "seed_detector_score": float(identities.seed_detector_score[identity]),
                "train_coverage": float(identities.train_coverage[identity]),
                "train_median_match_score": float(identities.train_median_match_score[identity]),
            }
            for identity in range(config.n_identities)
        ],
        "selection_information": "train RGB and direct seed-to-train LightGlue matches only",
        "templates_frozen_before_evaluation": True,
        "global_transform_used": False,
        "temporal_state_used": False,
    }
    write_json_no_overwrite(output_root / "RAW_IDENTITY_LOCK.json", template_lock)

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
    source_files = [
        REPO_ROOT / "keypoint_net" / "aliked_lightglue_bridge.py",
        REPO_ROOT / "keypoint_net" / "run_aliked_lightglue_bridge.py",
        REPO_ROOT / "keypoint_net" / "run_sift_bridge.py",
        config_path,
    ]
    receipt = {
        "schema": "raw_aliked_lightglue_inference_receipt.r1",
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
        "external_runtime": external,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "raw_predictions": {
            "path": str(raw_path),
            "sha256": raw_hash,
            "size_bytes": raw_size,
            "shape": [FRAME_COUNT, config.n_identities, 2],
            "accepted_per_identity": np.sum(accepted, axis=0).astype(int).tolist(),
            "missing_per_identity": np.sum(~accepted, axis=0).astype(int).tolist(),
        },
        "seed_extraction_repeatable_exact": seed_repeatable,
        "frame_order_invariance_exact": frame_order_invariant,
        "missing_predictions_filled": False,
        "global_transform_used": False,
        "temporal_state_used": False,
        "model_parameters_updated": False,
        "training_performed": False,
        "forbidden_inputs_loaded": [],
        "rgb_inputs": input_hashes,
    }
    write_json_no_overwrite(output_root / "RAW_INFERENCE_RECEIPT.json", receipt)
    print(json.dumps({"status": "inference_complete", "raw_sha256": raw_hash}, sort_keys=True), flush=True)


def create_visualizations(
    output_root: Path,
    images: Sequence[np.ndarray],
    coordinates_px: np.ndarray,
    expected_raw_px: np.ndarray,
    canonical: np.ndarray,
    reference_canonical: np.ndarray,
    material_error_px: np.ndarray,
    accepted: np.ndarray,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    visual_root = output_root / "visualizations"
    visual_root.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, Any]] = []

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
    figure.suptitle("Frozen ALIKED + LightGlue identities: independent direct seed matching")
    path = visual_root / "frame_overlays.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "actual target detections on fixed representative frames"})

    figure, axis = plt.subplots(figsize=(9, 9))
    for identity in range(10):
        axis.plot(canonical[:, identity, 0], canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=1.0, alpha=0.8, label=f"ALIKED{identity}")
        axis.scatter(reference_canonical[identity, 0], reference_canonical[identity, 1], marker="*", s=100, color=IDENTITY_COLORS[identity], edgecolors="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("canonical x (endpoint-normalized)")
    axis.set_ylabel("canonical y (endpoint-normalized)")
    axis.set_title("Physically de-rotated material trajectories; gaps are missing matches")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    path = visual_root / "canonical_trajectories.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "visual material sliding and wobble after exact physical de-rotation"})

    figure, axis = plt.subplots(figsize=(14, 6))
    for identity in range(10):
        axis.plot(material_error_px[:, identity], color=IDENTITY_COLORS[identity], linewidth=1.0, label=f"ALIKED{identity}")
    axis.axhline(thresholds["maximum_material_error_px"], color="black", linestyle="--", linewidth=1.0, label="frozen material bound")
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
        axes[0].plot(canonical[:, identity, 0], color=IDENTITY_COLORS[identity], linewidth=0.9, label=f"ALIKED{identity}")
        axes[1].plot(canonical[:, identity, 1], color=IDENTITY_COLORS[identity], linewidth=0.9)
    axes[0].set_ylabel("canonical x")
    axes[1].set_ylabel("canonical y")
    axes[1].set_xlabel("frame")
    axes[0].set_title("Canonical coordinates over all 180 frames")
    axes[0].legend(ncol=5, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    path = visual_root / "canonical_coordinates_over_time.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "visual adjacent zig-zag and slow drift in x and y"})

    adjacent = np.linalg.norm(np.diff(canonical, axis=0), axis=-1) * 255.5
    second = np.linalg.norm(canonical[2:] - 2.0 * canonical[1:-1] + canonical[:-2], axis=-1) * 255.5
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for identity in range(10):
        axes[0].plot(np.arange(1, FRAME_COUNT), adjacent[:, identity], color=IDENTITY_COLORS[identity], linewidth=0.8, label=f"ALIKED{identity}")
        axes[1].plot(np.arange(1, FRAME_COUNT - 1), second[:, identity], color=IDENTITY_COLORS[identity], linewidth=0.8)
    axes[0].axhline(thresholds["maximum_adjacent_canonical_step_px"], color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(thresholds["maximum_canonical_second_difference_px"], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("adjacent canonical step (px)")
    axes[1].set_ylabel("canonical second difference (px)")
    axes[1].set_xlabel("centre/end frame")
    axes[0].set_title("Frame-to-frame wobble with frozen full-resolution bounds")
    axes[0].legend(ncol=5, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    path = visual_root / "frame_to_frame_wobble.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "hard visual trace of adjacent and second-difference wobble"})

    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    worst_rows = []
    for identity, axis in enumerate(axes.flat):
        missing = np.flatnonzero(~accepted[:, identity])
        if missing.size:
            frame = int(missing[0])
            reason = "first missing"
            error = None
        else:
            ratios = material_error_px[:, identity] / thresholds["maximum_material_error_px"]
            frame = int(np.nanargmax(ratios))
            reason = "maximum normalized material violation"
            error = float(material_error_px[frame, identity])
        axis.imshow(images[frame])
        expected_x, expected_y = expected_raw_px[frame, identity]
        axis.scatter(expected_x, expected_y, marker="x", s=100, linewidths=2.0, color="lime", label="expected material")
        if accepted[frame, identity]:
            x, y = coordinates_px[frame, identity]
            axis.scatter(x, y, marker="o", s=70, facecolors="none", edgecolors="red", linewidths=2.0, label="ALIKED match")
        axis.set_title(f"ALIKED{identity} f{frame}\n{reason}; error={error}", fontsize=9)
        axis.axis("off")
        worst_rows.append({"identity": identity, "frame": frame, "reason": reason, "material_error_px": error})
    axes.flat[0].legend(loc="lower left", fontsize=7)
    path = visual_root / "worst_identity_events.png"
    save_figure_no_overwrite(path, figure)
    artifacts.append({"path": str(path), "meaning": "first missing or worst material event for every identity", "selection": worst_rows})
    return artifacts


def evaluate_stage(dataset_root: Path, config_path: Path, output_root: Path) -> None:
    lock = strict_json(config_path)
    _ = config_from_lock(lock, config_path)
    _ = verify_external_runtime(lock)
    calibration_binding = lock["scientific_threshold_binding"]
    calibration_path = Path(calibration_binding["path"]).resolve(strict=True)
    local_require(sha256_file(calibration_path)[0] == calibration_binding["file_sha256"], "calibration hash differs")
    calibration = strict_json(calibration_path)
    local_require(calibration["schema"] == calibration_binding["schema"], "calibration schema differs")
    local_require(calibration["content_hash_sha256"] == calibration_binding["content_hash_sha256"], "calibration content hash differs")
    local_require(calibration["all_semantic_assertions_pass"] is True, "calibration failed")
    thresholds = calibration["thresholds"]

    raw_path = output_root / "RAW_ALIKED_LIGHTGLUE_PREDICTIONS.npz"
    receipt_path = output_root / "RAW_INFERENCE_RECEIPT.json"
    local_require(raw_path.is_file() and receipt_path.is_file(), "raw inference stage is incomplete")
    receipt = strict_json(receipt_path)
    raw_hash, raw_size = sha256_file(raw_path)
    local_require(raw_hash == receipt["raw_predictions"]["sha256"], "raw prediction hash changed")
    local_require(raw_size == int(receipt["raw_predictions"]["size_bytes"]), "raw prediction size changed")
    local_require(receipt["forbidden_inputs_loaded"] == [], "raw stage loaded a forbidden input")
    local_require(receipt["missing_predictions_filled"] is False, "raw stage filled missing predictions")
    local_require(receipt["frame_order_invariance_exact"] is True, "frame-order invariance failed")
    local_require(receipt["global_transform_used"] is False, "raw stage used a global transform")
    local_require(receipt["temporal_state_used"] is False, "raw stage used temporal state")
    local_require(receipt["training_performed"] is False, "raw stage reports training")

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    coordinates_px = np.asarray(raw["coordinate_px"], dtype=np.float64)
    accepted = np.asarray(raw["accepted"], dtype=bool)
    local_require(coordinates_px.shape == (FRAME_COUNT, 10, 2), "raw coordinate shape differs")
    local_require(accepted.shape == (FRAME_COUNT, 10), "accepted shape differs")
    local_require(np.array_equal(np.isfinite(coordinates_px).all(axis=-1), accepted), "NaN/accepted semantics differ")

    dataset_index_path = dataset_root / "dataset_index.json"
    train_pair_path = REPO_ROOT / "docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__train.json"
    validation_pair_path = REPO_ROOT / "docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__validation.json"
    for path, expected in (
        (dataset_index_path, lock["dataset"]["dataset_index_sha256"]),
        (train_pair_path, lock["dataset"]["train_pair_index_sha256"]),
        (validation_pair_path, lock["dataset"]["validation_pair_index_sha256"]),
    ):
        local_require(sha256_file(path)[0] == expected, f"evaluation binding differs: {path}")

    meta = load_meta(dataset_root)
    theta = np.asarray([float(row["theta_deg"]) for row in meta], dtype=np.float64)
    local_require(np.array_equal(theta, np.arange(FRAME_COUNT, dtype=np.float64) * 2.0), "theta metadata differs")
    image_paths = frame_paths(dataset_root)
    images = [load_rgb(path) for path in image_paths]

    normalized = pixel_to_normalized(coordinates_px, 512, 512)
    canonical = np.full_like(normalized, np.nan)
    safe = np.where(accepted[..., None], normalized, 0.0)
    canonical_safe = rotate_normalized(safe, -theta[:, None])
    canonical[accepted] = canonical_safe[accepted]
    seed_xy = np.asarray(raw["seed_xy_px"], dtype=np.float64)
    seed_normalized = pixel_to_normalized(seed_xy, 512, 512)
    reference_canonical = rotate_normalized(seed_normalized, np.full(10, -theta[27], dtype=np.float64))
    reference_full = np.broadcast_to(reference_canonical, canonical.shape)
    material_error_px = np.full((FRAME_COUNT, 10), np.nan, dtype=np.float64)
    material_error_px[accepted] = np.linalg.norm(canonical[accepted] - reference_full[accepted], axis=-1) * 255.5
    expected_normalized = rotate_normalized(np.broadcast_to(reference_canonical, (FRAME_COUNT, 10, 2)), theta[:, None])
    expected_raw_px = np.empty_like(expected_normalized)
    expected_raw_px[..., 0] = (expected_normalized[..., 0] + 1.0) * 255.5
    expected_raw_px[..., 1] = (expected_normalized[..., 1] + 1.0) * 255.5

    mask_root = dataset_root / "train" / OBJECT_ID / "masks" / "a"
    on_object = np.zeros((FRAME_COUNT, 10), dtype=bool)
    border_distance_px = np.full((FRAME_COUNT, 10), np.nan, dtype=np.float64)
    for frame in range(FRAME_COUNT):
        with Image.open(mask_root / f"mask_{frame:04d}.png") as image:
            mask = np.asarray(image.convert("L")) > 0
        local_require(mask.shape == (512, 512), f"mask shape differs at frame {frame}")
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
        first, second = np.unravel_index(int(np.argmin(distances)), distances.shape)
        pairwise_min_px[frame] = float(distances[first, second])
        pairwise_argmin[frame] = [int(valid_identities[first]), int(valid_identities[second])]

    partitions = [
        partition_metrics("train", TRAIN_FRAMES, accepted, on_object, material_error_px, canonical),
        partition_metrics("holdout", HOLDOUT_FRAMES, accepted, on_object, material_error_px, canonical),
        partition_metrics("guard", GUARD_FRAMES, accepted, on_object, material_error_px, canonical),
    ]
    seam_rows = []
    for identity in range(10):
        valid = bool(accepted[179, identity] and accepted[0, identity])
        seam_rows.append({
            "identity": identity,
            "valid": valid,
            "canonical_step_px": float(np.linalg.norm(canonical[0, identity] - canonical[179, identity]) * 255.5) if valid else None,
        })

    border_minimum = np.asarray([
        np.nanmin(border_distance_px[:, identity]) if np.any(accepted[:, identity]) else -np.inf
        for identity in range(10)
    ])
    finite_pairwise = pairwise_min_px[np.isfinite(pairwise_min_px)]
    minimum_pairwise = float(np.min(finite_pairwise)) if finite_pairwise.size else None
    raw_gate_failures: list[str] = []
    if not bool(np.all(accepted)):
        raw_gate_failures.append("one_or_more_identities_missing_in_at_least_one_frame")
    if not bool(np.all(on_object[accepted])):
        raw_gate_failures.append("one_or_more_accepted_points_off_object")
    if not bool(np.all(border_minimum >= thresholds["minimum_image_border_distance_px"])):
        raw_gate_failures.append("one_or_more_identities_enter_calibrated_border_band")
    if minimum_pairwise is None or minimum_pairwise < thresholds["minimum_fixed_identity_pair_distance_px"]:
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
            checks = (
                (row["coverage"] >= thresholds["required_coverage"], "incomplete_independent_redetection"),
                (row["on_object_all_frame_rate"] >= thresholds["required_on_object_rate"], "grounding_failure"),
                (row["material_error_px"]["max"] is not None and row["material_error_px"]["max"] <= thresholds["maximum_material_error_px"], "material_error_exceeds_full_resolution_oracle"),
                (row["canonical_rms_about_mean_px"] is not None and row["canonical_rms_about_mean_px"] <= thresholds["maximum_canonical_rms_about_mean_px"], "canonical_rms_exceeds_full_resolution_oracle"),
                (row["canonical_radius_about_mean_px"] is not None and row["canonical_radius_about_mean_px"] <= thresholds["maximum_canonical_radius_about_mean_px"], "canonical_radius_exceeds_full_resolution_oracle"),
                (row["adjacent_canonical_step_px"]["max"] is not None and row["adjacent_canonical_step_px"]["max"] <= thresholds["maximum_adjacent_canonical_step_px"], "adjacent_wobble_exceeds_full_resolution_oracle"),
                (row["canonical_second_difference_px"]["max"] is not None and row["canonical_second_difference_px"]["max"] <= thresholds["maximum_canonical_second_difference_px"], "second_difference_exceeds_full_resolution_oracle"),
            )
            reasons.extend(f"{prefix}:{name}" for passed, name in checks if not passed)
        seam = seam_rows[identity]
        if not seam["valid"]:
            reasons.append("seam:missing_identity")
        elif seam["canonical_step_px"] > thresholds["maximum_seam_canonical_step_px"]:
            reasons.append("seam:canonical_step_exceeds_full_resolution_oracle")
        identity_decisions.append({
            "identity": identity,
            "reference_orbit_radius_px": reference_radius_px,
            "pass": len(reasons) == 0,
            "reasons": reasons,
        })

    if minimum_pairwise is not None and minimum_pairwise < thresholds["minimum_fixed_identity_pair_distance_px"]:
        frame = int(np.nanargmin(pairwise_min_px))
        for identity in pairwise_argmin[frame].tolist():
            if identity >= 0:
                identity_decisions[identity]["pass"] = False
                identity_decisions[identity]["reasons"].append(f"frame_{frame}:identity_separation_below_calibrated_minimum")
    numerical_contract_pass = bool(not raw_gate_failures and all(row["pass"] for row in identity_decisions))

    visuals = create_visualizations(
        output_root,
        images,
        coordinates_px,
        expected_raw_px,
        canonical,
        reference_canonical,
        material_error_px,
        accepted,
        thresholds,
    )
    minimum_pairwise_frame = int(np.nanargmin(pairwise_min_px)) if finite_pairwise.size else None
    metrics = {
        "schema": "aliked_lightglue_bridge_evaluation_metrics.r1",
        "raw_prediction_sha256": raw_hash,
        "calibration_binding": {
            "path": str(calibration_path),
            "file_sha256": calibration_binding["file_sha256"],
            "content_hash_sha256": calibration_binding["content_hash_sha256"],
            "thresholds": thresholds,
        },
        "evaluation_only_inputs_loaded_after_raw_hash": ["physical theta metadata", "object masks", "dataset and split bindings"],
        "coordinate_convention": {
            "raw": "actual ALIKED target pixel x,y from direct LightGlue seed match",
            "normalized": "endpoint-aligned [-1,1]",
            "canonical": "physical R(-theta) around normalized pivot (0,0)",
            "pixel_scale_for_normalized_distance": 255.5,
        },
        "partition_metrics": partitions,
        "seam_179_to_0": seam_rows,
        "full_orbit": {
            "accepted_per_identity": np.sum(accepted, axis=0).astype(int).tolist(),
            "missing_per_identity": np.sum(~accepted, axis=0).astype(int).tolist(),
            "on_object_accepted_rate_per_identity": [float(np.mean(on_object[:, identity][accepted[:, identity]])) if np.any(accepted[:, identity]) else None for identity in range(10)],
            "minimum_border_distance_px_per_identity": border_minimum.tolist(),
            "minimum_pairwise_identity_distance_px": minimum_pairwise,
            "minimum_pairwise_frame": minimum_pairwise_frame,
            "minimum_pairwise_identity_pair": pairwise_argmin[minimum_pairwise_frame].tolist() if minimum_pairwise_frame is not None else None,
            "maximum_material_error_px_per_identity": [float(np.nanmax(material_error_px[:, identity])) if np.any(accepted[:, identity]) else None for identity in range(10)],
        },
        "raw_categorical_failures_before_scientific_tolerances": raw_gate_failures,
        "identity_decisions": identity_decisions,
        "numerical_contract_pass": numerical_contract_pass,
        "visualizations": visuals,
        "statistical_language": "descriptive; frames and overlapping differences are correlated; no SEM or population CI",
    }
    metrics_path = output_root / "ALIKED_LIGHTGLUE_EVALUATION_METRICS.json"
    write_json_no_overwrite(metrics_path, metrics)
    outcome = {
        "schema": "aliked_lightglue_bridge_result.r1",
        "raw_prediction_sha256": raw_hash,
        "metrics_sha256": sha256_file(metrics_path)[0],
        "bridge_numerical_contract_pass": numerical_contract_pass,
        "bridge_contract_pass": False,
        "bridge_contract_status": "numerical_pass_pending_required_visual_audit" if numerical_contract_pass else "failed_frozen_numerical_contract",
        "raw_categorical_failures": raw_gate_failures,
        "failed_identity_count": int(sum(not row["pass"] for row in identity_decisions)),
        "required_visual_audit_complete": False,
        "training_performed": False,
        "gpu_used": False,
        "global_transform_used": False,
        "temporal_state_used": False,
        "operator_experiment_performed": False,
        "scientific_claim": "descriptive bridge feasibility on the already-studied hammer orbit only",
    }
    write_json_no_overwrite(output_root / "RESULT.json", outcome)
    print(json.dumps(outcome, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "infer", "evaluate"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    config_path = args.config_lock.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    if args.stage == "smoke":
        smoke_stage(dataset_root, config_path, output_root)
    elif args.stage == "infer":
        infer_stage(dataset_root, config_path, output_root)
    else:
        local_require(output_root.is_dir(), f"output root does not exist: {output_root}")
        evaluate_stage(dataset_root, config_path, output_root)


if __name__ == "__main__":
    try:
        main()
    except (AlikedRunnerError, AlikedBridgeError) as exc:
        raise SystemExit(f"ALIKED + LightGlue bridge gate failed: {exc}") from exc
