"""TDW affine-style dataset generator (separate from Phase-A yaw pipeline).

Generates single-object image datasets that vary:
1) image-plane translation (x/y offsets),
2) object scale,
3) either in-plane/world-roll rotation or true object yaw at fixed degree increments.

Design choices for this pipeline:
- Standalone script; does NOT modify or depend on the existing Phase-A generator.
- One TDW launch for the full run (stability on macOS).
- Fixed camera/background policy for uniform rendering.
- Per-frame validation via `_id` mask to reject clipped / badly framed samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.version import __version__ as tdw_version
from tdw.backend.update import Update
from tdw.output_data import Bounds, OutputData
from tdw.tdw_utils import TDWUtils


@dataclass(frozen=True)
class Transform:
    x_offset: float
    y_offset: float
    scale_rel: float
    theta_deg: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _read_models(models_file: Path) -> List[str]:
    lines = [line.strip() for line in models_file.read_text().splitlines()]
    models = [line for line in lines if line and not line.startswith("#")]
    if not models:
        raise ValueError(f"No models found in {models_file}")
    return models


def _hash_split(name: str, test_frac: float) -> str:
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / float(4294967296)
    if u < test_frac:
        return "test"
    return "train"


def _make_angles(rotation_step_deg: float) -> List[float]:
    if rotation_step_deg <= 0:
        raise ValueError("rotation_step_deg must be > 0")
    angles = []
    angle = 0.0
    while angle < 359.999999999:
        angles.append(float(angle))
        angle += rotation_step_deg
    return angles


def _make_grid(
    x_offsets: List[float],
    y_offsets: List[float],
    scale_factors: List[float],
    angles: List[float],
) -> List[Transform]:
    transforms = []
    for scale_rel in scale_factors:
        for y_offset in y_offsets:
            for x_offset in x_offsets:
                for theta_deg in angles:
                    transforms.append(
                        Transform(
                            x_offset=float(x_offset),
                            y_offset=float(y_offset),
                            scale_rel=float(scale_rel),
                            theta_deg=float(theta_deg),
                        )
                    )
    return transforms


def _pick_nearest(values: List[float], targets: List[float]) -> List[float]:
    if not values:
        return []
    uniq = sorted({float(v) for v in values})
    picked = []
    for t in targets:
        picked.append(min(uniq, key=lambda v: abs(v - float(t))))
    return sorted({float(v) for v in picked})


def _make_tuning_subset(transforms: List[Transform]) -> List[Transform]:
    xs = [t.x_offset for t in transforms]
    ys = [t.y_offset for t in transforms]
    scales = [t.scale_rel for t in transforms]
    thetas = [t.theta_deg for t in transforms]
    x_samples = _pick_nearest(xs, [min(xs), 0.0, max(xs)])
    y_samples = _pick_nearest(ys, [min(ys), 0.0, max(ys)])
    s_samples = _pick_nearest(scales, [min(scales), 1.0, max(scales)])
    theta_samples = _pick_nearest(thetas, [0.0, 90.0, 180.0, 270.0])
    subset = []
    for s in s_samples:
        for y in y_samples:
            for x in x_samples:
                for th in theta_samples:
                    subset.append(
                        Transform(
                            x_offset=float(x),
                            y_offset=float(y),
                            scale_rel=float(s),
                            theta_deg=float(th),
                        )
                    )
    return subset


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--uniform_bg_rgb must be 'R,G,B'")
    rgb = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError("--uniform_bg_rgb channels must be in [0, 255]")
    return rgb


def _id_mask(id_image: Image.Image) -> np.ndarray:
    arr = np.array(id_image.convert("RGB"))
    border = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
        axis=0,
    )
    colors, counts = np.unique(border, axis=0, return_counts=True)
    bg = colors[int(np.argmax(counts))]
    return np.any(arr != bg, axis=2)


def _apply_uniform_background(
    rgb_image: Image.Image,
    id_image: Image.Image,
    bg_rgb: Tuple[int, int, int],
) -> Image.Image:
    arr = np.array(rgb_image.convert("RGB"))
    mask = _id_mask(id_image)
    arr[~mask] = np.array(bg_rgb, dtype=np.uint8)
    return Image.fromarray(arr)


def _make_controller(port: int, launch_build: bool) -> Controller:
    if launch_build:
        Update.get_pypi_version = staticmethod(lambda: tdw_version)
        return Controller(port=port, check_version=True, launch_build=True)
    return Controller(port=port, check_version=False, launch_build=False)


def _get_bounds(
    controller: Controller,
    object_id: int,
    *,
    max_attempts: int = 8,
) -> Dict[str, np.ndarray]:
    for attempt in range(max_attempts):
        resp = controller.communicate(
            [{"$type": "send_bounds", "ids": [object_id], "frequency": "once"}]
        )
        for r in resp:
            if OutputData.get_data_type_id(r) != "boun":
                continue
            b = Bounds(r)
            for i in range(b.get_num()):
                if b.get_id(i) != object_id:
                    continue
                return {
                    "left": np.array(b.get_left(i), dtype=float),
                    "right": np.array(b.get_right(i), dtype=float),
                    "top": np.array(b.get_top(i), dtype=float),
                    "bottom": np.array(b.get_bottom(i), dtype=float),
                    "front": np.array(b.get_front(i), dtype=float),
                    "back": np.array(b.get_back(i), dtype=float),
                    "center": np.array(b.get_center(i), dtype=float),
                }
        if attempt < max_attempts - 1:
            controller.communicate([])
    raise RuntimeError(f"Bounds not found for object_id={object_id} after {max_attempts} attempts")


def _max_extent(bounds: Dict[str, np.ndarray]) -> float:
    x = float(np.linalg.norm(bounds["right"] - bounds["left"]))
    y = float(np.linalg.norm(bounds["top"] - bounds["bottom"]))
    z = float(np.linalg.norm(bounds["front"] - bounds["back"]))
    return max(x, y, z)


def _capture_pil(
    controller: Controller,
    cap: ImageCapture,
    commands: List[dict],
) -> Tuple[Image.Image, Image.Image]:
    cap.set(frequency="once", avatar_ids=["a"], pass_masks=["_img", "_id"], save=False)
    controller.communicate(commands)
    images = cap.get_pil_images()
    if "a" not in images:
        raise RuntimeError("Avatar 'a' did not return image data.")
    if "_img" not in images["a"] or "_id" not in images["a"]:
        raise RuntimeError("Missing required passes ('_img', '_id').")
    return images["a"]["_img"], images["a"]["_id"]


def _apply_centered_pose(
    controller: Controller,
    object_id: int,
    *,
    target_center_x: float,
    target_center_y: float,
    target_center_z: float,
    scale_abs: float,
    yaw_deg: float,
    theta_deg: float,
    rotation_axis: str,
    object_base_y: float,
    current_scale_abs: float,
) -> float:
    rel_scale = float(scale_abs) / float(max(current_scale_abs, 1e-8))
    if rotation_axis not in {"yaw", "roll"}:
        raise ValueError("rotation_axis must be 'roll' or 'yaw'")
    commands = [
        {
            "$type": "scale_object",
            "id": object_id,
            "scale_factor": {"x": rel_scale, "y": rel_scale, "z": rel_scale},
        },
        {
            "$type": "rotate_object_to_euler_angles",
            "id": object_id,
            "euler_angles": {
                "x": 0,
                "y": float(yaw_deg + theta_deg if rotation_axis == "yaw" else yaw_deg),
                "z": 0,
            },
        },
    ]
    if rotation_axis == "roll":
        commands.append(
            {
                "$type": "rotate_object_by",
                "id": object_id,
                "angle": float(theta_deg),
                "axis": "roll",
                "is_world": True,
                "use_centroid": True,
            }
        )
    commands.append(
        {
            "$type": "teleport_object",
            "id": object_id,
            "position": {
                "x": float(target_center_x),
                "y": float(object_base_y),
                "z": float(target_center_z),
            },
        }
    )
    controller.communicate(commands)

    position = np.array([target_center_x, object_base_y, target_center_z], dtype=float)
    target = np.array([target_center_x, target_center_y, target_center_z], dtype=float)
    for _ in range(2):
        bounds = _get_bounds(controller, object_id)
        center = bounds["center"]
        delta = target - center
        if np.linalg.norm(delta) < 0.0001:
            break
        position = position + delta
        controller.communicate(
            [
                {
                    "$type": "teleport_object",
                    "id": object_id,
                    "position": {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                    },
                }
            ]
        )
    return float(scale_abs)


def _mask_stats(id_image: Image.Image) -> Dict[str, float] | None:
    mask = _id_mask(id_image)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width_frac = (x1 - x0 + 1) / float(w)
    height_frac = (y1 - y0 + 1) / float(h)
    area_frac = float(mask.sum()) / float(w * h)
    margin_frac = min(
        x0 / float(w),
        (w - 1 - x1) / float(w),
        y0 / float(h),
        (h - 1 - y1) / float(h),
    )
    return {
        "bbox_width_frac": float(width_frac),
        "bbox_height_frac": float(height_frac),
        "area_frac": float(area_frac),
        "margin_frac": float(margin_frac),
    }


def _is_valid_frame(
    stats: Dict[str, float] | None,
    min_margin: float,
    min_area: float,
    max_size: float,
) -> bool:
    if stats is None:
        return False
    size = max(stats["bbox_width_frac"], stats["bbox_height_frac"])
    return stats["margin_frac"] >= min_margin and stats["area_frac"] >= min_area and size <= max_size


def _invalid_reason(
    stats: Dict[str, float] | None,
    min_margin: float,
    min_area: float,
    max_size: float,
) -> str:
    if stats is None:
        return "missing"
    size = max(stats["bbox_width_frac"], stats["bbox_height_frac"])
    if stats["area_frac"] < min_area:
        return "too_small"
    if size > max_size:
        return "too_large"
    if stats["margin_frac"] < min_margin:
        return "touching_border"
    return "valid"


def _choose_base_yaw(
    controller: Controller,
    cap: ImageCapture,
    object_id: int,
    base_scale: float,
    object_base_y: float,
    min_margin: float,
    min_area: float,
) -> Tuple[float, float]:
    best_score = None
    best_yaw = 0.0
    current_scale = 1.0
    yaw_candidates = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    for yaw in yaw_candidates:
        current_scale = _apply_centered_pose(
            controller=controller,
            object_id=object_id,
            target_center_x=0.0,
            target_center_y=float(object_base_y),
            target_center_z=0.0,
            scale_abs=float(base_scale),
            yaw_deg=float(yaw),
            theta_deg=0.0,
            rotation_axis="roll",
            object_base_y=float(object_base_y),
            current_scale_abs=current_scale,
        )
        _, id_img = _capture_pil(controller=controller, cap=cap, commands=[])
        s = _mask_stats(id_img)
        if s is None:
            continue
        penalty = max(0.0, min_margin - s["margin_frac"]) * 2.0
        penalty += max(0.0, min_area - s["area_frac"]) * 2.0
        score = s["area_frac"] + 0.35 * s["bbox_width_frac"] - penalty
        if best_score is None or score > best_score:
            best_score = score
            best_yaw = yaw
    return float(best_yaw), float(current_scale)


def _choose_camera_distance(
    controller: Controller,
    cap: ImageCapture,
    cam: ThirdPersonCamera,
    object_id: int,
    base_yaw: float,
    base_scale: float,
    object_base_y: float,
    camera_height: float,
    look_at_height: float,
    x_offsets: List[float],
    y_offsets: List[float],
    scale_factors: List[float],
    rotation_axis: str,
    min_margin: float,
    min_area: float,
    max_size: float,
    current_scale_abs: float,
) -> Tuple[float, float]:
    dist_candidates = [0.45, 0.54, 0.64, 0.76, 0.9, 1.06, 1.24]
    theta_samples = [0.0, 90.0, 180.0, 270.0]
    x_samples = [float(min(x_offsets)), 0.0, float(max(x_offsets))]
    y_samples = [float(min(y_offsets)), 0.0, float(max(y_offsets))]
    scale_abs = base_scale * max(scale_factors)
    best_full = None
    best_any = None
    current_scale = float(current_scale_abs)
    for dist in dist_candidates:
        cam.teleport(
            {"x": 0, "y": float(object_base_y + camera_height), "z": float(dist)}
        )
        cam.look_at({"x": 0, "y": float(object_base_y + look_at_height), "z": 0})
        controller.communicate([])
        valid_count = 0
        total = 0
        area_sum = 0.0
        for x in x_samples:
            for y in y_samples:
                for theta in theta_samples:
                    total += 1
                    current_scale = _apply_centered_pose(
                        controller=controller,
                        object_id=object_id,
                        target_center_x=float(x),
                        target_center_y=float(object_base_y + y),
                        target_center_z=0.0,
                        scale_abs=float(scale_abs),
                        yaw_deg=float(base_yaw),
                        theta_deg=float(theta),
                        rotation_axis=rotation_axis,
                        object_base_y=float(object_base_y),
                        current_scale_abs=current_scale,
                    )
                    _, id_img = _capture_pil(controller=controller, cap=cap, commands=[])
                    s = _mask_stats(id_img)
                    if _is_valid_frame(s, min_margin=min_margin, min_area=0.0, max_size=max_size):
                        valid_count += 1
                        area_sum += 0.0 if s is None else s["area_frac"]
        avg_area = area_sum / float(max(valid_count, 1))
        if valid_count == total:
            if (
                best_full is None
                or avg_area > best_full[1]
                or (abs(avg_area - best_full[1]) < 1e-9 and dist < best_full[0])
            ):
                best_full = (dist, avg_area)
        if (
            best_any is None
            or valid_count > best_any[0]
            or (valid_count == best_any[0] and avg_area > best_any[2])
            or (valid_count == best_any[0] and abs(avg_area - best_any[2]) < 1e-9 and dist > best_any[1])
        ):
            best_any = (valid_count, dist, avg_area)
    if best_full is not None:
        return float(best_full[0]), float(current_scale)
    if best_any is not None:
        return float(best_any[1]), float(current_scale)
    return float(max(dist_candidates)), float(current_scale)


def _evaluate_transforms_validity(
    *,
    controller: Controller,
    cap: ImageCapture,
    object_id: int,
    transforms: List[Transform],
    base_scale: float,
    base_yaw: float,
    object_base_y: float,
    min_margin: float,
    min_area: float,
    max_size: float,
    rotation_axis: str,
    debug_dir: Path | None = None,
    current_scale_abs: float = 1.0,
) -> Tuple[int, float, List[Dict], Dict[str, int], float]:
    valid_count = 0
    area_sum = 0.0
    invalid_examples = []
    current_scale = float(current_scale_abs)
    reason_counts = {"missing": 0, "too_small": 0, "too_large": 0, "touching_border": 0}
    for t_idx, tr in enumerate(transforms):
        scale_abs = base_scale * tr.scale_rel
        current_scale = _apply_centered_pose(
            controller=controller,
            object_id=object_id,
            target_center_x=float(tr.x_offset),
            target_center_y=float(object_base_y + tr.y_offset),
            target_center_z=0.0,
            scale_abs=float(scale_abs),
            yaw_deg=float(base_yaw),
            theta_deg=float(tr.theta_deg),
            rotation_axis=rotation_axis,
            object_base_y=float(object_base_y),
            current_scale_abs=current_scale,
        )
        rgb_img, id_img = _capture_pil(controller=controller, cap=cap, commands=[])
        stats = _mask_stats(id_img)
        reason = _invalid_reason(stats=stats, min_margin=min_margin, min_area=min_area, max_size=max_size)
        if reason == "valid":
            valid_count += 1
            area_sum += 0.0 if stats is None else float(stats["area_frac"])
            continue
        reason_counts[reason] += 1
        if stats is None and debug_dir is not None and len(invalid_examples) < 3:
            _ensure_dir(debug_dir)
            rgb_img.save(debug_dir / f"eval_t{t_idx:04d}_rgb.png")
            id_img.save(debug_dir / f"eval_t{t_idx:04d}_id.png")
        bounds_center = None
        bounds_extent = None
        if stats is not None and len(invalid_examples) < 3:
            b = _get_bounds(controller, object_id)
            bounds_center = b["center"].tolist()
            bounds_extent = {
                "x": float(np.linalg.norm(b["right"] - b["left"])),
                "y": float(np.linalg.norm(b["top"] - b["bottom"])),
                "z": float(np.linalg.norm(b["front"] - b["back"])),
            }
        if len(invalid_examples) < 10:
            invalid_examples.append(
                {
                    "transform_index": t_idx,
                    "x_offset": tr.x_offset,
                    "y_offset": tr.y_offset,
                    "scale_rel": tr.scale_rel,
                    "theta_deg": tr.theta_deg,
                    "stats": stats,
                    "reason": reason,
                    "bounds_center": bounds_center,
                    "bounds_extent": bounds_extent,
                }
            )
    avg_area = area_sum / float(max(valid_count, 1))
    return valid_count, avg_area, invalid_examples, reason_counts, float(current_scale)


def render_object_dataset(
    *,
    controller: Controller,
    model_name: str,
    object_dir: Path,
    transforms: List[Transform],
    img_size: int,
    object_base_y: float,
    x_offsets: List[float],
    y_offsets: List[float],
    scale_factors: List[float],
    camera_height: float,
    look_at_height: float,
    uniform_bg_rgb: Tuple[int, int, int],
    target_max_dim: float,
    min_margin: float,
    min_area: float,
    max_size: float,
    rotation_axis: str,
) -> Dict:
    _ensure_dir(object_dir)
    frames_dir = object_dir / "frames" / "a"
    _ensure_dir(frames_dir)
    for old_png in frames_dir.glob("*.png"):
        old_png.unlink()
    meta_path = object_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    c = controller
    c.add_ons.clear()
    cam = ThirdPersonCamera(
        avatar_id="a",
        position={"x": 0, "y": float(object_base_y + camera_height), "z": 0.9},
        look_at={"x": 0, "y": float(object_base_y + look_at_height), "z": 0},
    )
    cap = ImageCapture(
        avatar_ids=["a"],
        path=str((object_dir / "frames").resolve()),
        png=True,
        pass_masks=["_img", "_id"],
    )
    c.add_ons.extend([cam, cap])
    cap.set(frequency="never", avatar_ids=["a"], pass_masks=["_img", "_id"], save=False)

    cmds = [
        {"$type": "load_scene", "scene_name": "ProcGenScene"},
        TDWUtils.create_empty_room(12, 12),
        {"$type": "set_post_process", "value": False},
        {"$type": "set_screen_size", "width": int(img_size), "height": int(img_size)},
        {"$type": "set_target_framerate", "framerate": 30},
        {"$type": "set_render_quality", "render_quality": 5},
    ]
    obj_id = c.get_unique_id()
    cmds.append(
        c.get_add_object(
            model_name=model_name,
            object_id=obj_id,
            position={"x": 0, "y": float(object_base_y), "z": 0},
        )
    )
    cmds.append({"$type": "set_kinematic_state", "id": obj_id, "is_kinematic": True, "use_gravity": False})
    c.communicate(cmds)

    bounds = _get_bounds(c, obj_id)
    base_scale = float(np.clip(target_max_dim / max(_max_extent(bounds), 1e-6), 0.2, 10.0))
    current_scale = 1.0
    current_scale = _apply_centered_pose(
        controller=c,
        object_id=obj_id,
        target_center_x=0.0,
        target_center_y=float(object_base_y),
        target_center_z=0.0,
        scale_abs=float(base_scale),
        yaw_deg=0.0,
        theta_deg=0.0,
        rotation_axis=rotation_axis,
        object_base_y=float(object_base_y),
        current_scale_abs=current_scale,
    )
    base_yaw, current_scale = _choose_base_yaw(
        controller=c,
        cap=cap,
        object_id=obj_id,
        base_scale=base_scale,
        object_base_y=object_base_y,
        min_margin=min_margin,
        min_area=min_area,
    )
    camera_dist, current_scale = _choose_camera_distance(
        controller=c,
        cap=cap,
        cam=cam,
        object_id=obj_id,
        base_yaw=base_yaw,
        base_scale=base_scale,
        object_base_y=object_base_y,
        camera_height=camera_height,
        look_at_height=look_at_height,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        scale_factors=scale_factors,
        rotation_axis=rotation_axis,
        min_margin=min_margin,
        min_area=min_area,
        max_size=max_size,
        current_scale_abs=current_scale,
    )
    cam.teleport({"x": 0, "y": float(object_base_y + camera_height), "z": float(camera_dist)})
    cam.look_at({"x": 0, "y": float(object_base_y + look_at_height), "z": 0})
    c.communicate([])

    n_transforms = len(transforms)
    tuning_transforms = _make_tuning_subset(transforms)
    n_tuning = len(tuning_transforms)
    tuning_log = []
    tuned_scale = base_scale
    tuned_dist = camera_dist
    max_tune_attempts = 12
    for attempt in range(max_tune_attempts):
        cam.teleport({"x": 0, "y": float(object_base_y + camera_height), "z": float(tuned_dist)})
        cam.look_at({"x": 0, "y": float(object_base_y + look_at_height), "z": 0})
        c.communicate([])
        valid_count, avg_area, invalid_preview, reason_counts, current_scale = _evaluate_transforms_validity(
            controller=c,
            cap=cap,
            object_id=obj_id,
            transforms=tuning_transforms,
            base_scale=tuned_scale,
            base_yaw=base_yaw,
            object_base_y=object_base_y,
            min_margin=min_margin,
            min_area=min_area,
            max_size=max_size,
            rotation_axis=rotation_axis,
            debug_dir=object_dir / "_debug_invalid" if attempt == 0 else None,
            current_scale_abs=current_scale,
        )
        tuning_log.append(
            {
                "attempt": attempt,
                "base_yaw": base_yaw,
                "base_scale": tuned_scale,
                "camera_dist": tuned_dist,
                "valid_count": valid_count,
                "n_transforms": n_tuning,
                "avg_area": avg_area,
                "reason_counts": reason_counts,
                "invalid_examples": invalid_preview,
            }
        )
        if valid_count == n_tuning:
            break
        n_small = reason_counts["too_small"] + reason_counts["missing"]
        n_large = reason_counts["too_large"] + reason_counts["touching_border"]
        if n_small >= n_large:
            tuned_scale = min(tuned_scale * 1.25, 10.0)
            tuned_dist = max(tuned_dist * 0.93, 0.28)
        else:
            tuned_scale = max(tuned_scale * 0.85, 0.08)
            tuned_dist = min(tuned_dist * 1.08, 2.4)
    else:
        best = max(tuning_log, key=lambda r: (r["valid_count"], r["avg_area"]))
        _write_json(object_dir / "tuning_failed.json", {"tuning_log": tuning_log, "best": best})
        raise RuntimeError(
            f"{model_name}: couldn't find fully valid config after {max_tune_attempts} attempts; "
            f"best={best['valid_count']}/{n_tuning}; sample_invalid={best['invalid_examples'][:1]}"
        )

    base_scale = tuned_scale
    camera_dist = tuned_dist
    cam.teleport({"x": 0, "y": float(object_base_y + camera_height), "z": float(camera_dist)})
    cam.look_at({"x": 0, "y": float(object_base_y + look_at_height), "z": 0})
    c.communicate([])

    render_attempt_log = []
    max_render_attempts = 4
    invalid_examples = []
    frame_idx = 0
    n_invalid = 0
    for render_attempt in range(max_render_attempts):
        for old_png in frames_dir.glob("*.png"):
            old_png.unlink()
        if meta_path.exists():
            meta_path.unlink()
        cam.teleport({"x": 0, "y": float(object_base_y + camera_height), "z": float(camera_dist)})
        cam.look_at({"x": 0, "y": float(object_base_y + look_at_height), "z": 0})
        c.communicate([])
        invalid_examples = []
        frame_idx = 0
        n_invalid = 0
        reason_counts = {"missing": 0, "too_small": 0, "too_large": 0, "touching_border": 0}
        for t_idx, tr in enumerate(transforms):
            scale_abs = base_scale * tr.scale_rel
            current_scale = _apply_centered_pose(
                controller=c,
                object_id=obj_id,
                target_center_x=float(tr.x_offset),
                target_center_y=float(object_base_y + tr.y_offset),
                target_center_z=0.0,
                scale_abs=float(scale_abs),
                yaw_deg=float(base_yaw),
                theta_deg=float(tr.theta_deg),
                rotation_axis=rotation_axis,
                object_base_y=float(object_base_y),
                current_scale_abs=current_scale,
            )
            rgb_img, id_img = _capture_pil(controller=c, cap=cap, commands=[])
            stats = _mask_stats(id_img)
            reason = _invalid_reason(stats=stats, min_margin=min_margin, min_area=min_area, max_size=max_size)
            valid = reason == "valid"
            if not valid:
                n_invalid += 1
                reason_counts[reason] += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        {
                            "transform_index": t_idx,
                            "x_offset": tr.x_offset,
                            "y_offset": tr.y_offset,
                            "scale_rel": tr.scale_rel,
                            "theta_deg": tr.theta_deg,
                            "stats": stats,
                            "reason": reason,
                        }
                    )
                continue

            composed_img = _apply_uniform_background(rgb_img, id_img, uniform_bg_rgb)
            filename = f"img_{frame_idx:04d}.png"
            image_path = frames_dir / filename
            composed_img.save(image_path)
            image_relpath = f"frames/a/{filename}"
            _append_jsonl(
                meta_path,
                {
                    "t": frame_idx,
                    "transform_index": t_idx,
                    "image_index": frame_idx,
                    "image_relpath": image_relpath,
                    "model_name": model_name,
                    "object_id": obj_id,
                    "x_offset": tr.x_offset,
                    "y_offset": tr.y_offset,
                    "scale_rel": tr.scale_rel,
                    "scale_abs": scale_abs,
                    "theta_deg": tr.theta_deg,
                    "rotation_axis": rotation_axis,
                    "yaw_delta_deg": tr.theta_deg if rotation_axis == "yaw" else 0.0,
                    "yaw_deg": (base_yaw + tr.theta_deg) % 360.0 if rotation_axis == "yaw" else base_yaw,
                    "roll_delta_deg": tr.theta_deg if rotation_axis == "roll" else 0.0,
                    "camera_dist": camera_dist,
                    "camera_y": float(object_base_y + camera_height),
                    "look_at_y": float(object_base_y + look_at_height),
                    "bbox_width_frac": stats["bbox_width_frac"],
                    "bbox_height_frac": stats["bbox_height_frac"],
                    "area_frac": stats["area_frac"],
                    "margin_frac": stats["margin_frac"],
                },
            )
            frame_idx += 1

        render_attempt_log.append(
            {
                "attempt": render_attempt,
                "base_scale": base_scale,
                "camera_dist": camera_dist,
                "valid_count": frame_idx,
                "invalid_count": n_invalid,
                "reason_counts": reason_counts,
                "invalid_examples": invalid_examples,
            }
        )
        if n_invalid == 0:
            break
        n_small = reason_counts["too_small"] + reason_counts["missing"]
        n_large = reason_counts["too_large"] + reason_counts["touching_border"]
        if n_small >= n_large:
            base_scale = min(base_scale * 1.08, 10.0)
            camera_dist = max(camera_dist * 0.96, 0.24)
        else:
            base_scale = max(base_scale * 0.92, 0.06)
            camera_dist = min(camera_dist * 1.05, 2.4)
    else:
        _write_json(
            object_dir / "render_failed.json",
            {"render_attempt_log": render_attempt_log, "n_requested_transforms": len(transforms)},
        )
        raise RuntimeError(
            f"{model_name}: render stage failed after {max_render_attempts} attempts; "
            f"last_invalid={n_invalid}"
        )

    summary = {
        "model_name": model_name,
        "object_id": obj_id,
        "n_requested_transforms": len(transforms),
        "n_valid_frames": frame_idx,
        "n_invalid_frames": n_invalid,
        "base_yaw_deg": base_yaw,
        "rotation_axis": rotation_axis,
        "base_scale": base_scale,
        "camera_dist": camera_dist,
        "object_base_y": object_base_y,
        "camera_height": camera_height,
        "look_at_height": look_at_height,
        "uniform_bg_rgb": list(uniform_bg_rgb),
        "tuning_log": tuning_log,
        "render_attempt_log": render_attempt_log,
        "frames_dir": str(frames_dir),
        "meta_jsonl": str(meta_path),
        "invalid_examples": invalid_examples,
    }
    _write_json(object_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--models_file", type=str, required=True)
    parser.add_argument("--port", type=int, default=1071)
    parser.add_argument("--launch_build", action="store_true")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--test_frac", type=float, default=0.0)
    parser.add_argument("--rotation_step_deg", type=float, default=6.0)
    parser.add_argument("--x_offsets", type=float, nargs="+", default=[-0.04, 0.0, 0.04])
    parser.add_argument("--y_offsets", type=float, nargs="+", default=[-0.04, 0.0, 0.04])
    parser.add_argument("--scale_factors", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--rotation_axis", type=str, default="roll", choices=["roll", "yaw"])
    parser.add_argument("--object_base_y", type=float, default=1.0)
    parser.add_argument("--camera_height", type=float, default=0.0)
    parser.add_argument("--look_at_height", type=float, default=0.0)
    parser.add_argument("--uniform_bg_rgb", type=str, default="166,166,166")
    parser.add_argument("--target_max_dim", type=float, default=0.24)
    parser.add_argument("--min_margin", type=float, default=0.02)
    parser.add_argument("--min_area", type=float, default=0.015)
    parser.add_argument("--max_size", type=float, default=0.8)
    args = parser.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_root)
    models = _read_models(Path(args.models_file))
    uniform_bg_rgb = _parse_rgb(args.uniform_bg_rgb)
    angles = _make_angles(rotation_step_deg=args.rotation_step_deg)
    transforms = _make_grid(
        x_offsets=list(args.x_offsets),
        y_offsets=list(args.y_offsets),
        scale_factors=list(args.scale_factors),
        angles=angles,
    )
    dataset_index = {
        "config": {
            "img_size": args.img_size,
            "test_frac": args.test_frac,
            "rotation_step_deg": args.rotation_step_deg,
            "x_offsets": list(args.x_offsets),
            "y_offsets": list(args.y_offsets),
            "scale_factors": list(args.scale_factors),
            "rotation_axis": args.rotation_axis,
            "object_base_y": args.object_base_y,
            "camera_height": args.camera_height,
            "look_at_height": args.look_at_height,
            "uniform_bg_rgb": list(uniform_bg_rgb),
            "target_max_dim": args.target_max_dim,
            "min_margin": args.min_margin,
            "min_area": args.min_area,
            "max_size": args.max_size,
        },
        "angles_deg": angles,
        "n_transforms_per_object": len(transforms),
        "splits": {"train": [], "test": []},
        "objects": {},
    }

    controller = _make_controller(port=int(args.port), launch_build=bool(args.launch_build))
    for idx, model_name in enumerate(models):
        split = _hash_split(model_name, args.test_frac) if args.test_frac > 0 else "train"
        object_dir = out_root / split / model_name
        print(f"[{idx + 1}/{len(models)}] {model_name} -> {object_dir}")
        summary = render_object_dataset(
            controller=controller,
            model_name=model_name,
            object_dir=object_dir,
            transforms=transforms,
            img_size=args.img_size,
            object_base_y=float(args.object_base_y),
            x_offsets=list(args.x_offsets),
            y_offsets=list(args.y_offsets),
            scale_factors=list(args.scale_factors),
            camera_height=float(args.camera_height),
            look_at_height=float(args.look_at_height),
            uniform_bg_rgb=uniform_bg_rgb,
            target_max_dim=float(args.target_max_dim),
            min_margin=float(args.min_margin),
            min_area=float(args.min_area),
            max_size=float(args.max_size),
            rotation_axis=str(args.rotation_axis),
        )
        dataset_index["splits"][split].append(model_name)
        dataset_index["objects"][model_name] = summary

    _write_json(out_root / "dataset_index.json", dataset_index)
    controller.communicate([{"$type": "terminate"}])
    print(f"Done. Wrote {out_root / 'dataset_index.json'}")


if __name__ == "__main__":
    main()
