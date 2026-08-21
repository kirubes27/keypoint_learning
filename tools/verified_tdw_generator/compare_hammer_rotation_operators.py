"""Hammer-only sanity check for three rotation operators.

This script is deliberately small and sanity-only. It compares:

1. TDW world-Y yaw: change Euler y before rendering.
2. TDW camera-axis/world-roll rotation: start from the base pose, rotate around
   TDW world roll/Z before rendering.
3. Post-render 2D affine rotation: rotate the saved base RGBA cutout with PIL.

No full dataset is generated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from create_true_yaw_from_base_panel_sanity import (
    ObjectSpec,
    RenderConfig,
    _append_jsonl,
    _apply_uniform_background,
    _binary_mask_image,
    _capture_pil,
    _contact_sheet,
    _ensure_dir,
    _get_bounds,
    _id_mask,
    _load_source_specs,
    _make_controller,
    _mask_stats,
    _setup_scene,
    _source_match_stats,
    _valid_stats,
    _write_json,
)


ANGLES = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
HAMMER = "engineers_hammer_vray"


def _pad_to_rotation_safe(rgba: Image.Image, margin_px: int = 8) -> Image.Image:
    w, h = rgba.size
    side = int(math.ceil(math.sqrt(float(w * w + h * h)))) + 2 * int(margin_px)
    side = max(side, w + 2 * margin_px, h + 2 * margin_px)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2), rgba)
    return canvas


def _apply_pil_inplane(
    base_rgba_padded: Image.Image,
    *,
    theta_deg: float,
    canvas_size: int,
    bg_rgb: Tuple[int, int, int],
) -> Tuple[Image.Image, np.ndarray]:
    rotated = base_rgba_padded.rotate(
        angle=float(theta_deg),
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )
    layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    cx = (canvas_size - 1) / 2.0
    cy = (canvas_size - 1) / 2.0
    left = int(round(cx - rotated.width / 2.0))
    top = int(round(cy - rotated.height / 2.0))
    layer.paste(rotated, (left, top), rotated)
    alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
    mask = alpha > 0
    bg = Image.new("RGB", (canvas_size, canvas_size), color=bg_rgb)
    rgb = Image.alpha_composite(bg.convert("RGBA"), layer).convert("RGB")
    return rgb, mask


def _recenter_object(controller, object_id: int, cfg: RenderConfig) -> float:
    target = np.array([0.0, cfg.object_base_y, 0.0], dtype=float)
    position = target.copy()
    center_error = float("inf")
    for _ in range(4):
        bounds = _get_bounds(controller, object_id)
        delta = target - bounds["center"]
        center_error = float(np.linalg.norm(delta))
        if center_error <= cfg.center_tolerance_world:
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
    return center_error


def _apply_tdw_operator(
    controller,
    *,
    object_id: int,
    spec: ObjectSpec,
    theta_deg: float,
    operator_name: str,
    scale_abs: float,
    current_scale_abs: float,
    cfg: RenderConfig,
) -> Tuple[float, float, Dict]:
    rel_scale = float(scale_abs) / float(max(current_scale_abs, 1e-8))
    command_log: Dict = {
        "operator_name": operator_name,
        "base_euler_xyz_deg": [spec.base_pitch_deg, spec.base_yaw_deg, spec.base_roll_deg],
        "theta_deg": float(theta_deg),
    }

    if operator_name == "tdw_world_y_yaw":
        euler = {
            "x": float(spec.base_pitch_deg),
            "y": float(spec.base_yaw_deg + theta_deg),
            "z": float(spec.base_roll_deg),
        }
        command_log["tdw_command_semantics"] = "rotate_object_to_euler_angles with y = base_yaw + theta"
        command_log["final_euler_xyz_deg"] = [euler["x"], euler["y"] % 360.0, euler["z"]]
    elif operator_name == "tdw_absolute_euler_z_roll":
        euler = {
            "x": float(spec.base_pitch_deg),
            "y": float(spec.base_yaw_deg),
            "z": float(spec.base_roll_deg + theta_deg),
        }
        command_log["tdw_command_semantics"] = "rotate_object_to_euler_angles with z = base_roll + theta"
        command_log["final_euler_xyz_deg"] = [euler["x"], euler["y"], euler["z"] % 360.0]
    elif operator_name in {
        "tdw_camera_axis_roll",
        "tdw_world_z_roll",
        "tdw_world_x_pitch",
        "tdw_world_y_yaw_by_axis",
    }:
        euler = {
            "x": float(spec.base_pitch_deg),
            "y": float(spec.base_yaw_deg),
            "z": float(spec.base_roll_deg),
        }
        axis_by_operator = {
            "tdw_camera_axis_roll": "roll",
            "tdw_world_z_roll": "roll",
            "tdw_world_x_pitch": "pitch",
            "tdw_world_y_yaw_by_axis": "yaw",
        }
        axis = axis_by_operator[operator_name]
        command_log["tdw_command_semantics"] = (
            "rotate_object_to_euler_angles to base pose, then rotate_object_by "
            f"axis='{axis}', is_world=True, use_centroid=True"
        )
        command_log["final_euler_xyz_deg"] = None
        command_log["extra_rotation"] = {
            "axis": axis,
            "is_world": True,
            "use_centroid": True,
            "angle_deg": float(theta_deg),
        }
    else:
        raise ValueError(f"Unknown TDW operator: {operator_name}")

    controller.communicate(
        [
            {
                "$type": "scale_object",
                "id": object_id,
                "scale_factor": {"x": rel_scale, "y": rel_scale, "z": rel_scale},
            },
            {
                "$type": "rotate_object_to_euler_angles",
                "id": object_id,
                "euler_angles": euler,
            },
            {
                "$type": "teleport_object",
                "id": object_id,
                "position": {"x": 0.0, "y": float(cfg.object_base_y), "z": 0.0},
            },
        ]
    )

    if operator_name in {
        "tdw_camera_axis_roll",
        "tdw_world_z_roll",
        "tdw_world_x_pitch",
        "tdw_world_y_yaw_by_axis",
    } and abs(float(theta_deg)) > 1e-9:
        controller.communicate(
            [
                {
                    "$type": "rotate_object_by",
                    "id": object_id,
                    "angle": float(theta_deg),
                    "axis": command_log["extra_rotation"]["axis"],
                    "is_world": True,
                    "use_centroid": True,
                }
            ]
        )

    center_error = _recenter_object(controller, object_id, cfg)
    return float(scale_abs), center_error, command_log


def _render_tdw_operator(
    controller,
    *,
    spec: ObjectSpec,
    cfg: RenderConfig,
    operator_name: str,
    object_dir: Path,
) -> Dict:
    obj_id, cap = _setup_scene(controller, spec, object_dir, cfg)
    frames_dir = object_dir / "frames" / "a"
    masks_dir = object_dir / "masks" / "a"
    ids_dir = object_dir / "id_passes" / "a"
    for path in [frames_dir, masks_dir, ids_dir]:
        _ensure_dir(path)

    meta_path = object_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    current_scale = 1.0
    records = []
    rgb_items = []
    mask_items = []
    for frame_idx, theta in enumerate(ANGLES):
        current_scale, center_error, command_log = _apply_tdw_operator(
            controller,
            object_id=obj_id,
            spec=spec,
            theta_deg=float(theta),
            operator_name=operator_name,
            scale_abs=spec.base_scale_tdw,
            current_scale_abs=current_scale,
            cfg=cfg,
        )
        rgb_img, id_img = _capture_pil(controller=controller, cap=cap, commands=[])
        mask = _id_mask(id_img)
        stats = _mask_stats(mask)
        valid, invalid_reason = _valid_stats(stats, cfg)
        rgb_out = _apply_uniform_background(rgb_img, mask, cfg.uniform_bg_rgb)
        mask_img = _binary_mask_image(mask)

        image_relpath = f"frames/a/img_{frame_idx:04d}.png"
        mask_relpath = f"masks/a/mask_{frame_idx:04d}.png"
        id_relpath = f"id_passes/a/id_{frame_idx:04d}.png"
        rgb_out.save(object_dir / image_relpath)
        mask_img.save(object_dir / mask_relpath)
        id_img.save(object_dir / id_relpath)

        rec = {
            "t": int(frame_idx),
            "image_index": int(frame_idx),
            "image_relpath": image_relpath,
            "mask_relpath": mask_relpath,
            "id_relpath": id_relpath,
            "model_name": spec.model_name,
            "operator_name": operator_name,
            "theta_deg": float(theta),
            "rotation_is_tdw_rerendered": True,
            "command_log": command_log,
            "base_pitch_deg": spec.base_pitch_deg,
            "base_yaw_deg": spec.base_yaw_deg,
            "base_roll_deg": spec.base_roll_deg,
            "scale_abs": spec.base_scale_tdw,
            "camera_dist": spec.camera_dist,
            "object_base_y": cfg.object_base_y,
            "camera_height": cfg.camera_height,
            "look_at_height": cfg.look_at_height,
            "center_error_world": float(center_error),
            "valid": bool(valid),
            "invalid_reason": invalid_reason,
            "stats": stats,
        }
        if abs(float(theta)) < 1e-9:
            rec["source_match"] = _source_match_stats(rgb_out, spec.source_base_preview)
        _append_jsonl(meta_path, rec)
        records.append(rec)
        rgb_items.append((rgb_out, f"{operator_name}\nTDW render\ntheta {theta:g}"))
        mask_items.append((mask_img.convert("RGB"), f"{operator_name}\nmask\ntheta {theta:g}"))

    summary = {
        "operator_name": operator_name,
        "model_name": spec.model_name,
        "n_frames": len(records),
        "angles_deg": ANGLES,
        "pass": all(r["valid"] for r in records),
        "yaw0_source_match": records[0].get("source_match"),
        "records": records,
    }
    _write_json(object_dir / "summary.json", summary)
    return {"summary": summary, "rgb_items": rgb_items, "mask_items": mask_items}


def _render_pil_operator(*, spec: ObjectSpec, cfg: RenderConfig, object_dir: Path) -> Dict:
    frames_dir = object_dir / "frames" / "a"
    masks_dir = object_dir / "masks" / "a"
    _ensure_dir(frames_dir)
    _ensure_dir(masks_dir)
    meta_path = object_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    base_rgba = Image.open(spec.source_object_dir / "base_rgba.png").convert("RGBA")
    base_rgba_padded = _pad_to_rotation_safe(base_rgba)
    source_preview = Image.open(spec.source_base_preview).convert("RGB")

    records = []
    rgb_items = []
    mask_items = []
    for frame_idx, theta in enumerate(ANGLES):
        rgb_out, mask = _apply_pil_inplane(
            base_rgba_padded,
            theta_deg=float(theta),
            canvas_size=cfg.img_size,
            bg_rgb=cfg.uniform_bg_rgb,
        )
        mask_img = _binary_mask_image(mask)
        stats = _mask_stats(mask)
        valid, invalid_reason = _valid_stats(stats, cfg)

        image_relpath = f"frames/a/img_{frame_idx:04d}.png"
        mask_relpath = f"masks/a/mask_{frame_idx:04d}.png"
        rgb_out.save(object_dir / image_relpath)
        mask_img.save(object_dir / mask_relpath)

        rec = {
            "t": int(frame_idx),
            "image_index": int(frame_idx),
            "image_relpath": image_relpath,
            "mask_relpath": mask_relpath,
            "model_name": spec.model_name,
            "operator_name": "post_render_2d_affine",
            "theta_deg": float(theta),
            "rotation_is_tdw_rerendered": False,
            "implementation": "PIL rotate() on saved base_rgba cutout",
            "base_pitch_deg": spec.base_pitch_deg,
            "base_yaw_deg": spec.base_yaw_deg,
            "base_roll_deg": spec.base_roll_deg,
            "valid": bool(valid),
            "invalid_reason": invalid_reason,
            "stats": stats,
        }
        if abs(float(theta)) < 1e-9:
            diff = ImageChops.difference(source_preview, rgb_out)
            arr = np.asarray(diff, dtype=np.float32)
            rec["source_match"] = {
                "source_path": str(spec.source_base_preview),
                "mean_abs_rgb_diff": float(arr.mean()),
                "max_abs_rgb_diff": float(arr.max()),
            }
        _append_jsonl(meta_path, rec)
        records.append(rec)
        rgb_items.append((rgb_out, f"post_render_2d_affine\nPIL rotate\ntheta {theta:g}"))
        mask_items.append((mask_img.convert("RGB"), f"post_render_2d_affine\nmask\ntheta {theta:g}"))

    summary = {
        "operator_name": "post_render_2d_affine",
        "model_name": spec.model_name,
        "n_frames": len(records),
        "angles_deg": ANGLES,
        "pass": all(r["valid"] for r in records),
        "yaw0_source_match": records[0].get("source_match"),
        "records": records,
    }
    _write_json(object_dir / "summary.json", summary)
    return {"summary": summary, "rgb_items": rgb_items, "mask_items": mask_items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_index", type=str, default="./_full_2d_affine_selected_final_512/dataset_index.json")
    parser.add_argument("--models_file", type=str, default="./models_affine_final_6.txt")
    parser.add_argument("--out_dir", type=str, default="./_sanity_hammer_three_rotation_operators_512")
    parser.add_argument("--port", type=int, default=1084)
    parser.add_argument("--launch_build", action="store_true")
    parser.add_argument(
        "--operator_set",
        type=str,
        default="comparison",
        choices=["comparison", "world_axes"],
        help="comparison keeps the previous diagnostic rows; world_axes renders only world-Y/X/Z rotate_by rows.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Refusing to mix outputs into nonempty folder: {out_root}")
    _ensure_dir(out_root)

    specs, cfg, _ = _load_source_specs(Path(args.source_index).resolve(), Path(args.models_file).resolve())
    hammer_specs = [s for s in specs if s.model_name == HAMMER]
    if not hammer_specs:
        raise ValueError(f"Could not find {HAMMER} in source specs.")
    spec = hammer_specs[0]

    semantic_lock = {
        "must_be_true": [
            "Hammer starts from the exact base-panel pose, scale, and camera.",
            "All three operators use the same six theta values.",
            "Only hammer sanity frames are generated.",
        ],
        "operators": {
            "tdw_world_y_yaw": "TDW rerender after Euler y = base_yaw + theta.",
            "tdw_world_y_yaw_by_axis": "TDW rerender after rotating around world yaw/Y axis from the base pose.",
            "tdw_camera_axis_roll": "TDW rerender after rotating around world/camera roll axis from the base pose.",
            "tdw_world_z_roll": "TDW rerender after rotating around world roll/Z axis from the base pose.",
            "tdw_world_x_pitch": "TDW rerender after rotating around world pitch axis from the base pose.",
            "tdw_absolute_euler_z_roll": "TDW rerender after absolute Euler z = base_roll + theta.",
            "post_render_2d_affine": "No TDW rerender; PIL rotates the saved base RGBA cutout.",
        },
        "camera": {
            "position": [0.0, cfg.object_base_y + cfg.camera_height, spec.camera_dist],
            "look_at": [0.0, cfg.object_base_y + cfg.look_at_height, 0.0],
        },
        "base_pose_xyz_deg": [spec.base_pitch_deg, spec.base_yaw_deg, spec.base_roll_deg],
        "angles_deg": ANGLES,
    }
    _write_json(out_root / "semantic_lock.json", semantic_lock)

    summaries: Dict[str, Dict] = {}
    rgb_items: List[Tuple[Image.Image, str]] = []
    mask_items: List[Tuple[Image.Image, str]] = []

    controller = _make_controller(port=int(args.port), launch_build=bool(args.launch_build))
    try:
        if args.operator_set == "world_axes":
            tdw_operators = ["tdw_world_y_yaw_by_axis", "tdw_world_x_pitch", "tdw_world_z_roll"]
        else:
            tdw_operators = [
                "tdw_world_y_yaw",
                "tdw_camera_axis_roll",
                "tdw_world_x_pitch",
                "tdw_absolute_euler_z_roll",
            ]
        for operator_name in tdw_operators:
            result = _render_tdw_operator(
                controller,
                spec=spec,
                cfg=cfg,
                operator_name=operator_name,
                object_dir=out_root / operator_name / "train" / spec.model_name,
            )
            summaries[operator_name] = result["summary"]
            rgb_items.extend(result["rgb_items"])
            mask_items.extend(result["mask_items"])
    finally:
        controller.communicate([{"$type": "terminate"}])

    if args.operator_set != "world_axes":
        pil_result = _render_pil_operator(
            spec=spec,
            cfg=cfg,
            object_dir=out_root / "post_render_2d_affine" / "train" / spec.model_name,
        )
        summaries["post_render_2d_affine"] = pil_result["summary"]
        rgb_items.extend(pil_result["rgb_items"])
        mask_items.extend(pil_result["mask_items"])

    _contact_sheet(rgb_items, cols=len(ANGLES)).save(out_root / "hammer_three_operators_rgb_contact_sheet.png")
    _contact_sheet(mask_items, cols=len(ANGLES)).save(out_root / "hammer_three_operators_mask_contact_sheet.png")
    _write_json(
        out_root / "sanity_index.json",
        {
            "semantic_lock": semantic_lock,
            "out_root": str(out_root),
            "rgb_contact_sheet": str(out_root / "hammer_three_operators_rgb_contact_sheet.png"),
            "mask_contact_sheet": str(out_root / "hammer_three_operators_mask_contact_sheet.png"),
            "summaries": summaries,
        },
    )
    print(json.dumps({"out_root": str(out_root)}, indent=2))


if __name__ == "__main__":
    main()
