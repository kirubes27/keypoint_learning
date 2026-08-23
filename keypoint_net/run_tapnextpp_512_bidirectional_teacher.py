"""Run the frozen official TAPNext++ 512 support-point bridge on local CPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image
import torch

try:
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_LOCK_SHA256 = (
    "5f438c7b9958ec64dd25d3344321eccb7e9b13a41e586ef5198fa8036990b78f"
)
EXPECTED_MANIFEST_SHA256 = (
    "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
)
EXPECTED_INITIALS_SHA256 = (
    "d5bffc4651347eb76556000ba92ac8f3a82e324f310bda37f84b8c5b789b8a34"
)
EXPECTED_TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
EXPECTED_CHECKPOINT_SHA256 = (
    "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f"
)
EXPECTED_CHECKPOINT_SIZE_BYTES = 2_532_283_010
EXPECTED_FRAMES = 180
EXPECTED_WITNESSES = 10
IMAGE_SIZE = 512
MODEL_INPUT_RESOLUTION = 512
MODEL_COORDINATE_RESOLUTION = 256
SUPPORT_POINTS_PER_QUERY = 64
SUPPORT_RADIUS_MODEL_INPUT_PX = 32.0
SUPPORT_MODE = "local"
SUPPORT_RADIUS_SPACE = "model"
EXPECTED_SUPPORT_POINTS = EXPECTED_WITNESSES * SUPPORT_POINTS_PER_QUERY
EXPECTED_INTERNAL_QUERIES = EXPECTED_WITNESSES + EXPECTED_SUPPORT_POINTS
DETERMINISM_PREFIX_FRAMES = 5
FULL_RUN_MODEL_CALLS = 2 * DETERMINISM_PREFIX_FRAMES + 2 * EXPECTED_FRAMES
CPU_TIMING_SAFETY_FACTOR = 1.25
CPU_MAX_PROJECTED_SECONDS = 7_200.0


SOURCE_RELPATHS = (
    "tapnet/tapnextpp/votsp2026/README.md",
    "tapnet/tapnextpp/votsp2026/model.py",
    "tapnet/tapnextpp/votsp2026/tracker.py",
    "tapnet/tapnextpp/votsp2026/utils.py",
    "tapnet/tapnext/tapnext_torch.py",
    "tapnet/tapnext/tapnext_lru_modules.py",
    "tapnet/tapnext/pscan.py",
)


def _grid_support_points(n: int, width: float, height: float) -> np.ndarray:
    """Mirror the pinned official tracker grid construction exactly."""
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    columns = max(1, round(float(np.sqrt(n * width / height))))
    rows = max(1, int(np.ceil(n / columns)))
    xs = (np.arange(columns) + 0.5) * (width / columns)
    ys = (np.arange(rows) + 0.5) * (height / rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)
    return points[:n]


def _local_support_points(query_xy: np.ndarray) -> np.ndarray:
    """Construct 64 local support points per real query as in tracker.py."""
    require(
        query_xy.shape == (EXPECTED_WITNESSES, 2),
        "real query shape differs before support construction",
    )
    radius = SUPPORT_RADIUS_MODEL_INPUT_PX * (
        IMAGE_SIZE / float(MODEL_INPUT_RESOLUTION)
    )
    all_points: list[np.ndarray] = []
    for query_x, query_y in query_xy:
        local = _grid_support_points(
            SUPPORT_POINTS_PER_QUERY,
            2.0 * radius,
            2.0 * radius,
        )
        local -= np.asarray([radius, radius], dtype=np.float32)
        local += np.asarray([query_x, query_y], dtype=np.float32)
        local[:, 0] = np.clip(local[:, 0], 0, IMAGE_SIZE - 1)
        local[:, 1] = np.clip(local[:, 1], 0, IMAGE_SIZE - 1)
        all_points.append(local)
    support = np.concatenate(all_points, axis=0).astype(np.float32)
    require(
        support.shape == (EXPECTED_SUPPORT_POINTS, 2),
        "official support-point shape differs",
    )
    return support


def _build_internal_queries(
    initial_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = _local_support_points(initial_coordinate)
    internal = np.concatenate([initial_coordinate, support], axis=0).astype(np.float32)
    require(
        internal.shape == (EXPECTED_INTERNAL_QUERIES, 2),
        "internal query shape differs",
    )
    require(bool(np.isfinite(internal).all()), "internal query is non-finite")
    require(
        bool(((internal >= 0.0) & (internal < IMAGE_SIZE)).all()),
        "internal query is outside the image",
    )
    return internal, support


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    require(rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"RGB shape differs: {path}")
    return np.ascontiguousarray(rgb[..., ::-1])


def _run_traversal(
    model: Any,
    frames_bgr: list[np.ndarray],
    query_xy: np.ndarray,
    traversal: Iterable[int],
    *,
    autocast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    order = [int(value) for value in traversal]
    require(order and order[0] == 0, "TAPNext++ traversal must start at frame zero")
    positions: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    state = None
    for step, frame_index in enumerate(order):
        if step == 0:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index],
                query_points_xy=query_xy,
                autocast=autocast,
            )
        else:
            current_xy, current_visible, state = model.track_frame(
                frames_bgr[frame_index],
                state=state,
                autocast=autocast,
            )
        current_xy = np.asarray(current_xy, dtype=np.float32)
        current_visible = np.asarray(current_visible, dtype=bool)
        require(
            current_xy.shape == (EXPECTED_INTERNAL_QUERIES, 2),
            "TAPNext++ internal query output shape differs",
        )
        require(
            current_visible.shape == (EXPECTED_INTERNAL_QUERIES,),
            "TAPNext++ internal visibility output shape differs",
        )
        # The official VOT configuration co-tracks supports but discards them.
        # Only the first ten real queries enter scientific outputs/evaluation.
        positions.append(current_xy[:EXPECTED_WITNESSES].copy())
        visibility.append(current_visible[:EXPECTED_WITNESSES].copy())
    return np.stack(positions), np.stack(visibility)


def _project_cpu_full_seconds(
    model_load_seconds: float,
    prefix_seconds_a: float,
    prefix_seconds_b: float,
) -> float:
    """Conservatively project the complete 370-call CPU execution."""
    values = np.asarray(
        [model_load_seconds, prefix_seconds_a, prefix_seconds_b], dtype=np.float64
    )
    require(bool(np.isfinite(values).all()), "CPU timing contains a non-finite value")
    require(bool((values >= 0.0).all()), "CPU timing contains a negative value")
    slow_prefix_seconds_per_call = max(prefix_seconds_a, prefix_seconds_b) / float(
        DETERMINISM_PREFIX_FRAMES
    )
    return float(
        CPU_TIMING_SAFETY_FACTOR
        * (model_load_seconds + FULL_RUN_MODEL_CALLS * slow_prefix_seconds_per_call)
    )


def _peak_rss_record() -> dict[str, Any]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {
        "ru_maxrss_raw": raw,
        "ru_maxrss_platform_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
        "peak_resident_memory_bytes": raw * multiplier,
    }


def _environment_record(device: str, torchvision: Any, einops: Any) -> dict[str, Any]:
    mps_backend = getattr(torch.backends, "mps", None)
    record: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "einops": einops.__version__,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "execution_device": device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "model_input_resolution": MODEL_INPUT_RESOLUTION,
        "model_coordinate_resolution": MODEL_COORDINATE_RESOLUTION,
        "half_precision": False,
        "torch_compile": False,
        "autocast_enabled": device == "cuda",
        "allow_tf32": False,
    }
    if device == "cuda":
        record.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_device_count_visible": torch.cuda.device_count(),
            }
        )
    else:
        record.update(
            {
                "cuda_runtime": None,
                "cuda_device_name": None,
                "cuda_device_capability": None,
                "cuda_device_count_visible": 0,
            }
        )
    return record


def _write_pip_freeze(output_dir: Path) -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = output_dir / "TAPNEXTPP_EXECUTION_PIP_FREEZE.txt"
    path.write_text(freeze)
    return file_record(path)


def _canonicalize(
    traversal: Iterable[int],
    positions: np.ndarray,
    visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray([int(value) for value in traversal], dtype=np.int64)
    require(
        order.ndim == 1 and len(order) == len(positions),
        "traversal output length differs",
    )
    require(
        np.array_equal(np.sort(order), np.arange(len(order))),
        "traversal is not a permutation",
    )
    canonical_positions = np.empty_like(positions)
    canonical_visibility = np.empty_like(visibility)
    canonical_positions[order] = positions
    canonical_visibility[order] = visibility
    return canonical_positions, canonical_visibility


def _validate_prediction_arrays(positions: np.ndarray, visibility: np.ndarray) -> None:
    require(
        positions.shape[1:] == (EXPECTED_WITNESSES, 2),
        "TAPNext++ prediction query shape differs",
    )
    require(
        visibility.shape == positions.shape[:2], "TAPNext++ visibility shape differs"
    )
    require(
        bool(np.isfinite(positions).all()), "TAPNext++ produced non-finite coordinates"
    )
    require(
        bool(
            (
                (positions[..., 0] >= 0.0)
                & (positions[..., 0] < IMAGE_SIZE)
                & (positions[..., 1] >= 0.0)
                & (positions[..., 1] < IMAGE_SIZE)
            ).all()
        ),
        "TAPNext++ produced an out-of-image coordinate",
    )


def _coordinate_mapping_smoke(
    initial_coordinate: np.ndarray,
    display_to_model: Callable[..., np.ndarray],
    model_to_display: Callable[..., np.ndarray],
) -> dict[str, Any]:
    """Prove the official 512 display -> 256 coordinate -> 512 convention."""
    model_coordinate = np.asarray(
        display_to_model(
            initial_coordinate,
            IMAGE_SIZE,
            IMAGE_SIZE,
            MODEL_COORDINATE_RESOLUTION,
        ),
        dtype=np.float32,
    )
    roundtrip_coordinate = np.asarray(
        model_to_display(
            model_coordinate,
            IMAGE_SIZE,
            IMAGE_SIZE,
            MODEL_COORDINATE_RESOLUTION,
        ),
        dtype=np.float32,
    )
    require(
        model_coordinate.shape == initial_coordinate.shape, "model query shape differs"
    )
    require(
        roundtrip_coordinate.shape == initial_coordinate.shape,
        "display round-trip shape differs",
    )
    maximum_absolute_error = float(
        np.max(np.abs(roundtrip_coordinate.astype(np.float64) - initial_coordinate))
    )
    require(
        maximum_absolute_error <= 1e-6,
        "official 512-to-256-to-512 coordinate round-trip exceeds 1e-6 pixels",
    )
    return {
        "display_resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "model_input_resolution": [MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION],
        "model_coordinate_resolution": [
            MODEL_COORDINATE_RESOLUTION,
            MODEL_COORDINATE_RESOLUTION,
        ],
        "display_input_order": "x_y",
        "inner_query_order": "t_y_x",
        "inner_track_order": "y_x",
        "display_output_order": "x_y",
        "display_to_model_scale_xy": [0.5, 0.5],
        "model_to_display_scale_xy": [2.0, 2.0],
        "resize_mode": "bilinear_align_corners_false",
        "custom_half_pixel_correction_applied": False,
        "coordinate_roundtrip_query_count": int(initial_coordinate.shape[0]),
        "maximum_absolute_roundtrip_error_px": maximum_absolute_error,
        "roundtrip_tolerance_px": 1e-6,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    require(args.device == "cpu", "TAPNext++ 512 local bridge is CPU only")
    require(not torch.cuda.is_available(), "CPU profile refuses a CUDA-visible runtime")
    require(
        not args.output_dir.exists(),
        "output directory already exists; use a fresh attempt",
    )
    require(
        _git(args.repo_root, "status", "--porcelain") == "",
        "implementation repository is dirty",
    )
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")

    require(args.tapnet_root.is_dir(), "TAPNet checkout is missing")
    require(
        _git(args.tapnet_root, "status", "--porcelain") == "",
        "TAPNet checkout is dirty",
    )
    tapnet_commit = _git(args.tapnet_root, "rev-parse", "HEAD")
    require(tapnet_commit == EXPECTED_TAPNET_COMMIT, "TAPNet commit differs")
    tapnet_remote = _git(args.tapnet_root, "remote", "get-url", "origin")
    require(
        tapnet_remote
        in {
            "https://github.com/google-deepmind/tapnet.git",
            "https://github.com/deepmind/tapnet.git",
        },
        "TAPNet origin differs",
    )

    lock_record = file_record(args.semantic_lock)
    require(
        lock_record["sha256"] == EXPECTED_LOCK_SHA256,
        "TAPNext++ 512 semantic-lock SHA-256 differs",
    )
    manifest_record = file_record(args.manifest)
    require(
        manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256,
        "manifest SHA-256 differs",
    )
    initials_record = file_record(args.initials)
    require(
        initials_record["sha256"] == EXPECTED_INITIALS_SHA256,
        "initials SHA-256 differs",
    )
    checkpoint_record = file_record(args.checkpoint)
    require(
        checkpoint_record["sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "TAPNext++ checkpoint SHA-256 differs",
    )
    require(
        checkpoint_record["size_bytes"] == EXPECTED_CHECKPOINT_SIZE_BYTES,
        "TAPNext++ checkpoint size differs",
    )

    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)
    require(len(rgb_paths) == EXPECTED_FRAMES, "RGB frame count differs")
    with np.load(args.initials) as archive:
        require(
            set(archive.files)
            == {"witness_id", "initial_frame_index", "initial_coordinate_px"},
            "initials expose unexpected arrays",
        )
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        initial_frame = int(archive["initial_frame_index"])
        initial_coordinate = np.asarray(
            archive["initial_coordinate_px"], dtype=np.float32
        )
    require(witness_id.shape == (EXPECTED_WITNESSES,), "witness IDs differ")
    require(initial_frame == 0, "initial frame differs")
    require(
        initial_coordinate.shape == (EXPECTED_WITNESSES, 2),
        "initial coordinates differ",
    )
    internal_query_xy, support_xy = _build_internal_queries(initial_coordinate)
    opened_rgb_paths = (
        rgb_paths[:DETERMINISM_PREFIX_FRAMES] if args.timing_smoke_only else rgb_paths
    )
    frames_bgr = [_load_bgr(path) for path in opened_rgb_paths]

    sys.path.insert(0, str(args.tapnet_root.resolve()))
    import einops  # pylint: disable=import-outside-toplevel
    import torchvision  # pylint: disable=import-outside-toplevel
    from tapnet.tapnextpp.votsp2026 import (
        utils as tapnextpp_utils,
    )  # pylint: disable=import-outside-toplevel
    from tapnet.tapnextpp.votsp2026.model import (  # pylint: disable=import-outside-toplevel
        TAPNextPP,
    )

    coordinate_mapping = _coordinate_mapping_smoke(
        internal_query_xy,
        tapnextpp_utils.display_to_model,
        tapnextpp_utils.model_to_display,
    )
    require(
        TAPNextPP.MODEL_SIZE == MODEL_COORDINATE_RESOLUTION,
        "official model coordinate size differs",
    )

    torch.manual_seed(0)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.set_num_threads(4)

    model_load_started = time.perf_counter()
    model = TAPNextPP.from_checkpoint(
        args.checkpoint,
        device=args.device,
        half_precision=False,
        compile_model=False,
        input_resolution=MODEL_INPUT_RESOLUTION,
    )
    model_load_seconds = float(time.perf_counter() - model_load_started)
    require(
        model.input_resolution == MODEL_INPUT_RESOLUTION,
        "official input resolution differs",
    )
    require(model.device.type == args.device, "official model device differs")
    autocast = args.device == "cuda"
    prefix_order = list(range(DETERMINISM_PREFIX_FRAMES))
    prefix_started = time.perf_counter()
    prefix_positions_a, prefix_visibility_a = _run_traversal(
        model, frames_bgr, internal_query_xy, prefix_order, autocast=autocast
    )
    prefix_seconds_a = float(time.perf_counter() - prefix_started)
    prefix_started = time.perf_counter()
    prefix_positions_b, prefix_visibility_b = _run_traversal(
        model, frames_bgr, internal_query_xy, prefix_order, autocast=autocast
    )
    prefix_seconds_b = float(time.perf_counter() - prefix_started)
    prefix_positions_exact = bool(
        np.array_equal(prefix_positions_a, prefix_positions_b)
    )
    prefix_visibility_exact = bool(
        np.array_equal(prefix_visibility_a, prefix_visibility_b)
    )
    require(prefix_positions_exact, "repeated TAPNext++ prefix positions differ")
    require(prefix_visibility_exact, "repeated TAPNext++ prefix visibility differs")
    _validate_prediction_arrays(prefix_positions_a, prefix_visibility_a)

    source_records = {
        relpath: file_record(args.tapnet_root / relpath) for relpath in SOURCE_RELPATHS
    }
    environment = _environment_record(args.device, torchvision, einops)

    if args.timing_smoke_only:
        projected_full_seconds = _project_cpu_full_seconds(
            model_load_seconds,
            prefix_seconds_a,
            prefix_seconds_b,
        )
        full_run_authorized = projected_full_seconds <= CPU_MAX_PROJECTED_SECONDS
        args.output_dir.mkdir(parents=True)
        arrays_path = args.output_dir / "TAPNEXTPP_512_CPU_TIMING_SMOKE_ARRAYS.npz"
        np.savez_compressed(
            arrays_path,
            witness_id=witness_id,
            initial_frame_index=np.asarray(initial_frame, dtype=np.int64),
            initial_coordinate_px=initial_coordinate,
            prefix_prediction_xy_a=prefix_positions_a,
            prefix_visible_a=prefix_visibility_a,
            prefix_prediction_xy_b=prefix_positions_b,
            prefix_visible_b=prefix_visibility_b,
            traversal=np.asarray(prefix_order, dtype=np.int64),
        )
        pip_freeze_record = _write_pip_freeze(args.output_dir)
        smoke_receipt = {
            "schema_version": "tapnextpp_512_support64_local_cpu_timing_smoke.v1",
            "artifact_type": "prediction_only_local_cpu_runtime_and_determinism_smoke",
            "decision": {
                "full_run_authorized": full_run_authorized,
                "branch": (
                    "authorize_full_local_cpu_raw_gate"
                    if full_run_authorized
                    else "stop_local_cpu_runtime_exceeds_two_hours"
                ),
            },
            "sources": {
                "semantic_lock": lock_record,
                "sanitized_manifest": manifest_record,
                "frame_zero_initials": initials_record,
                "tapnextpp_checkpoint": checkpoint_record,
                "tapnet": {
                    "origin": tapnet_remote,
                    "commit": tapnet_commit,
                    "source_files": source_records,
                },
            },
            "implementation_head": implementation_head,
            "implementation_sources": {
                "runner": file_record(Path(__file__)),
                "gate_io": file_record(
                    Path(__file__).with_name("material_transport_gate_io.py")
                ),
            },
            "environment": environment,
            "pip_freeze": pip_freeze_record,
            "coordinate_mapping": coordinate_mapping,
            "support_configuration": {
                "mode": SUPPORT_MODE,
                "support_radius_space": SUPPORT_RADIUS_SPACE,
                "support_radius_model_input_px": SUPPORT_RADIUS_MODEL_INPUT_PX,
                "support_points_per_real_query": SUPPORT_POINTS_PER_QUERY,
                "real_query_count": EXPECTED_WITNESSES,
                "support_query_count": int(support_xy.shape[0]),
                "internal_query_count": int(internal_query_xy.shape[0]),
                "support_trajectories_saved": False,
                "source": "tapnet/tapnextpp/votsp2026/tracker.py",
            },
            "timing_seconds": {
                "model_load": model_load_seconds,
                "five_frame_prefix_a": prefix_seconds_a,
                "five_frame_prefix_b": prefix_seconds_b,
                "projection_safety_factor": CPU_TIMING_SAFETY_FACTOR,
                "projected_full_370_call_execution": projected_full_seconds,
                "maximum_authorized_projection": CPU_MAX_PROJECTED_SECONDS,
            },
            "process_memory": _peak_rss_record(),
            "arrays": file_record(arrays_path),
            "controls": {
                "all_180_rgb_hashes_rechecked_before_open": True,
                "opened_rgb_frame_count": DETERMINISM_PREFIX_FRAMES,
                "only_frame_zero_initial_coordinates_opened": True,
                "supplied_masks_opened": False,
                "non_frame_zero_truth_opened": False,
                "renderer_angle_or_pivot_opened": False,
                "learned_keypoint_checkpoint_or_features_opened": False,
                "local_laptop_gpu_used": False,
                "local_laptop_cpu_only": True,
                "cluster_cuda_only": False,
                "autocast_enabled": False,
                "repeated_prefix_positions_exact": prefix_positions_exact,
                "repeated_prefix_visibility_exact": prefix_visibility_exact,
                "coordinates_finite_and_in_image": True,
                "official_512_input_256_coordinate_roundtrip_pass": True,
                "official_local_support_construction_pass": True,
                "only_ten_real_query_trajectories_saved": True,
            },
            "frame_calls_executed": 2 * DETERMINISM_PREFIX_FRAMES,
            "command_argv": list(sys.argv),
            "runtime_seconds": float(time.time() - started),
            "privileged_evaluation_authorized": False,
            "training_or_weight_update_performed": False,
        }
        receipt_path = args.output_dir / "TAPNEXTPP_512_CPU_TIMING_SMOKE_RECEIPT.json"
        write_json(receipt_path, smoke_receipt)
        del model
        gc.collect()
        return {**smoke_receipt, "receipt": file_record(receipt_path)}

    forward_order = list(range(EXPECTED_FRAMES))
    reverse_order = [0, *range(EXPECTED_FRAMES - 1, 0, -1)]
    forward_positions, forward_visibility = _run_traversal(
        model, frames_bgr, internal_query_xy, forward_order, autocast=autocast
    )
    reverse_positions_ordered, reverse_visibility_ordered = _run_traversal(
        model, frames_bgr, internal_query_xy, reverse_order, autocast=autocast
    )
    reverse_positions, reverse_visibility = _canonicalize(
        reverse_order, reverse_positions_ordered, reverse_visibility_ordered
    )
    _validate_prediction_arrays(forward_positions, forward_visibility)
    _validate_prediction_arrays(reverse_positions, reverse_visibility)
    require(
        forward_positions.shape[0] == EXPECTED_FRAMES, "forward frame count differs"
    )
    require(
        reverse_positions.shape[0] == EXPECTED_FRAMES, "reverse frame count differs"
    )

    direction_difference = np.linalg.norm(
        forward_positions.astype(np.float64) - reverse_positions.astype(np.float64),
        axis=-1,
    )
    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "RAW_TAPNEXTPP_512_BIDIRECTIONAL_TEACHER.npz"
    np.savez_compressed(
        arrays_path,
        witness_id=witness_id,
        initial_frame_index=np.asarray(initial_frame, dtype=np.int64),
        initial_coordinate_px=initial_coordinate,
        forward_prediction_xy=forward_positions,
        forward_visible=forward_visibility,
        reverse_prediction_xy=reverse_positions,
        reverse_visible=reverse_visibility,
        forward_reverse_difference_px=direction_difference,
        forward_traversal=np.asarray(forward_order, dtype=np.int64),
        reverse_traversal=np.asarray(reverse_order, dtype=np.int64),
        repeated_prefix_prediction_xy=prefix_positions_a,
        repeated_prefix_visible=prefix_visibility_a,
    )

    pip_freeze_record = _write_pip_freeze(args.output_dir)
    receipt = {
        "schema_version": "raw_tapnextpp_512_support64_bidirectional_teacher.v1",
        "artifact_type": "prediction_only_pretrained_bidirectional_material_tracks",
        "execution_profile": args.device,
        "decision": {
            "raw_tracker_semantic_pass": True,
            "branch": "authorize_privileged_posthash_evaluation",
        },
        "sources": {
            "semantic_lock": lock_record,
            "sanitized_manifest": manifest_record,
            "frame_zero_initials": initials_record,
            "tapnextpp_checkpoint": checkpoint_record,
            "tapnet": {
                "origin": tapnet_remote,
                "commit": tapnet_commit,
                "source_files": source_records,
            },
        },
        "implementation_head": implementation_head,
        "implementation_sources": {
            "runner": file_record(Path(__file__)),
            "gate_io": file_record(
                Path(__file__).with_name("material_transport_gate_io.py")
            ),
        },
        "environment": environment,
        "pip_freeze": pip_freeze_record,
        "coordinate_mapping": coordinate_mapping,
        "support_configuration": {
            "mode": SUPPORT_MODE,
            "support_radius_space": SUPPORT_RADIUS_SPACE,
            "support_radius_model_input_px": SUPPORT_RADIUS_MODEL_INPUT_PX,
            "support_points_per_real_query": SUPPORT_POINTS_PER_QUERY,
            "real_query_count": EXPECTED_WITNESSES,
            "support_query_count": int(support_xy.shape[0]),
            "internal_query_count": int(internal_query_xy.shape[0]),
            "support_trajectories_saved": False,
            "source": "tapnet/tapnextpp/votsp2026/tracker.py",
        },
        "raw_predictions": file_record(arrays_path),
        "controls": {
            "all_rgb_hashes_rechecked_before_open": True,
            "only_frame_zero_initial_coordinates_opened": True,
            "supplied_masks_opened": False,
            "non_frame_zero_truth_opened": False,
            "renderer_angle_or_pivot_opened": False,
            "learned_keypoint_checkpoint_or_features_opened": False,
            "local_laptop_gpu_used": False,
            "local_laptop_cpu_only": args.device == "cpu",
            "cluster_cuda_only": args.device == "cuda",
            "autocast_enabled": autocast,
            "repeated_prefix_frame_count": DETERMINISM_PREFIX_FRAMES,
            "repeated_prefix_positions_exact": prefix_positions_exact,
            "repeated_prefix_visibility_exact": prefix_visibility_exact,
            "all_forward_coordinates_finite_and_in_image": True,
            "all_reverse_coordinates_finite_and_in_image": True,
            "official_512_input_256_coordinate_roundtrip_pass": True,
            "official_local_support_construction_pass": True,
            "only_ten_real_query_trajectories_saved": True,
            "forward_visible_count_diagnostic": int(np.sum(forward_visibility)),
            "reverse_visible_count_diagnostic": int(np.sum(reverse_visibility)),
        },
        "direction_disagreement_diagnostic_px": {
            "n": int(direction_difference.size),
            "mean": float(np.mean(direction_difference)),
            "median": float(np.median(direction_difference)),
            "q90": float(np.quantile(direction_difference, 0.90)),
            "maximum": float(np.max(direction_difference)),
        },
        "frame_count": EXPECTED_FRAMES,
        "model_frame_call_count": FULL_RUN_MODEL_CALLS,
        "witness_count": EXPECTED_WITNESSES,
        "command_argv": list(sys.argv),
        "runtime_seconds": float(time.time() - started),
        "privileged_evaluation_authorized": True,
        "training_or_weight_update_performed": False,
        "statistical_scope": "ten fixed witnesses over one correlated 180-frame orbit; descriptive only",
    }
    receipt_path = (
        args.output_dir / "RAW_TAPNEXTPP_512_BIDIRECTIONAL_TEACHER_RECEIPT.json"
    )
    write_json(receipt_path, receipt)

    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--initials", required=True, type=Path)
    parser.add_argument("--semantic-lock", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--tapnet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--timing-smoke-only", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
