"""Sanity renders for true yaw starting from the existing base panel poses.

This is a sanity-check-only script. It renders sparse yaw deltas from the exact
base poses/scales/camera stored in the current Angela-style 2D-affine dataset.
It does not generate the full 2-degree dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.tdw_utils import TDWUtils

from create_dataset_affine_xy_scale_rotation import (
    _capture_pil,
    _ensure_dir,
    _get_bounds,
    _id_mask,
    _make_controller,
    _write_json,
)


SANITY_YAWS = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
SAFE_SCALE_YAWS = [float(v) for v in range(0, 360, 30)]


@dataclass(frozen=True)
class ObjectSpec:
    model_name: str
    split: str
    source_object_dir: Path
    base_pitch_deg: float
    base_yaw_deg: float
    base_roll_deg: float
    base_scale_tdw: float
    camera_dist: float
    source_base_preview: Path


@dataclass(frozen=True)
class RenderConfig:
    img_size: int
    object_base_y: float
    camera_height: float
    look_at_height: float
    uniform_bg_rgb: Tuple[int, int, int]
    min_margin: float
    max_size: float
    center_tolerance_world: float


def _append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _signed_yaw(delta: float) -> float:
    value = float(delta) % 360.0
    return value if value <= 180.0 else value - 360.0


def _mask_stats(mask: np.ndarray) -> Dict[str, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width_frac = (x1 - x0 + 1) / float(w)
    height_frac = (y1 - y0 + 1) / float(h)
    return {
        "bbox_width_frac": float(width_frac),
        "bbox_height_frac": float(height_frac),
        "bbox_max_size_frac": float(max(width_frac, height_frac)),
        "area_frac": float(mask.sum()) / float(w * h),
        "margin_frac": float(
            min(
                x0 / float(w),
                (w - 1 - x1) / float(w),
                y0 / float(h),
                (h - 1 - y1) / float(h),
            )
        ),
        "bbox_center_x_offset_frac": float(((x0 + x1) / 2.0 - (w - 1) / 2.0) / float(w)),
        "bbox_center_y_offset_frac": float(((y0 + y1) / 2.0 - (h - 1) / 2.0) / float(h)),
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1),
    }


def _valid_stats(stats: Dict[str, float] | None, cfg: RenderConfig) -> Tuple[bool, str]:
    if stats is None:
        return False, "empty_mask"
    if stats["margin_frac"] < cfg.min_margin:
        return False, "low_margin"
    if stats["bbox_max_size_frac"] > cfg.max_size:
        return False, "too_large"
    return True, "valid"


def _binary_mask_image(mask: np.ndarray) -> Image.Image:
    arr = np.zeros(mask.shape, dtype=np.uint8)
    arr[mask] = 255
    return Image.fromarray(arr, mode="L")


def _apply_uniform_background(
    rgb_image: Image.Image,
    mask: np.ndarray,
    bg_rgb: Tuple[int, int, int],
) -> Image.Image:
    arr = np.array(rgb_image.convert("RGB"), dtype=np.uint8)
    arr[~mask] = np.array(bg_rgb, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _camera_position(spec: ObjectSpec, cfg: RenderConfig) -> Dict[str, float]:
    return {
        "x": 0.0,
        "y": float(cfg.object_base_y + cfg.camera_height),
        "z": float(spec.camera_dist),
    }


def _look_at(cfg: RenderConfig) -> Dict[str, float]:
    return {"x": 0.0, "y": float(cfg.object_base_y + cfg.look_at_height), "z": 0.0}


def _load_source_specs(index_path: Path, models_file: Path) -> Tuple[List[ObjectSpec], RenderConfig, Dict]:
    idx = json.loads(index_path.read_text())
    desired_order = [
        line.strip()
        for line in models_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    records = {str(rec["model_name"]): rec for rec in idx["objects"]}
    missing = [name for name in desired_order if name not in records]
    if missing:
        raise ValueError(f"Missing source records for: {missing}")

    cfg_raw = idx["config"]
    specs: List[ObjectSpec] = []
    for model_name in desired_order:
        rec = records[model_name]
        base = rec["base_info"]
        object_dir = Path(str(rec["object_dir"]))
        specs.append(
            ObjectSpec(
                model_name=model_name,
                split=str(rec["split"]),
                source_object_dir=object_dir,
                base_pitch_deg=float(base["base_pitch_deg"]),
                base_yaw_deg=float(base["base_yaw_deg"]),
                base_roll_deg=float(base["base_roll_deg"]),
                base_scale_tdw=float(base["base_scale_tdw"]),
                camera_dist=float(base["camera_dist"]),
                source_base_preview=object_dir / "base_preview.png",
            )
        )

    cfg = RenderConfig(
        img_size=int(cfg_raw["img_size"]),
        object_base_y=float(cfg_raw["object_base_y"]),
        camera_height=float(cfg_raw["camera_height"]),
        look_at_height=float(cfg_raw["look_at_height"]),
        uniform_bg_rgb=tuple(int(v) for v in cfg_raw["uniform_bg_rgb"]),
        min_margin=0.01,
        max_size=0.82,
        center_tolerance_world=0.005,
    )
    return specs, cfg, idx


def _setup_scene(controller, spec: ObjectSpec, object_dir: Path, cfg: RenderConfig):
    _ensure_dir(object_dir)
    c = controller
    c.add_ons.clear()
    cam = ThirdPersonCamera(avatar_id="a", position=_camera_position(spec, cfg), look_at=_look_at(cfg))
    cap = ImageCapture(
        avatar_ids=["a"],
        path=str((object_dir / "capture_unused").resolve()),
        png=True,
        pass_masks=["_img", "_id"],
    )
    c.add_ons.extend([cam, cap])
    cap.set(frequency="never", avatar_ids=["a"], pass_masks=["_img", "_id"], save=False)

    commands = [
        {"$type": "load_scene", "scene_name": "ProcGenScene"},
        TDWUtils.create_empty_room(12, 12),
        {"$type": "set_post_process", "value": False},
        {"$type": "set_screen_size", "width": cfg.img_size, "height": cfg.img_size},
        {"$type": "set_target_framerate", "framerate": 30},
        {"$type": "set_render_quality", "render_quality": 5},
    ]
    obj_id = c.get_unique_id()
    commands.append(
        c.get_add_object(
            model_name=spec.model_name,
            object_id=obj_id,
            position={"x": 0.0, "y": float(cfg.object_base_y), "z": 0.0},
        )
    )
    commands.append({"$type": "set_kinematic_state", "id": obj_id, "is_kinematic": True, "use_gravity": False})
    commands.append({"$type": "send_segmentation_colors", "frequency": "once"})
    c.communicate(commands)
    cam.teleport(_camera_position(spec, cfg))
    cam.look_at(_look_at(cfg))
    c.communicate([])
    return obj_id, cap


def _apply_base_plus_yaw(
    controller,
    *,
    object_id: int,
    spec: ObjectSpec,
    yaw_delta: float,
    scale_abs: float,
    current_scale_abs: float,
    cfg: RenderConfig,
) -> Tuple[float, float]:
    rel_scale = float(scale_abs) / float(max(current_scale_abs, 1e-8))
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
                "x": float(spec.base_pitch_deg),
                "y": float(spec.base_yaw_deg + yaw_delta),
                "z": float(spec.base_roll_deg),
            },
        },
        {
            "$type": "teleport_object",
            "id": object_id,
            "position": {"x": 0.0, "y": float(cfg.object_base_y), "z": 0.0},
        },
    ]
    controller.communicate(commands)

    position = np.array([0.0, cfg.object_base_y, 0.0], dtype=float)
    target = np.array([0.0, cfg.object_base_y, 0.0], dtype=float)
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
    return float(scale_abs), center_error


def _source_match_stats(rendered: Image.Image, source_path: Path) -> Dict | None:
    if not source_path.exists():
        return None
    source = Image.open(source_path).convert("RGB")
    rendered = rendered.convert("RGB")
    if source.size != rendered.size:
        rendered = rendered.resize(source.size, resample=Image.BILINEAR)
    diff = ImageChops.difference(source, rendered)
    arr = np.array(diff, dtype=np.float32)
    return {
        "source_path": str(source_path),
        "mean_abs_rgb_diff": float(arr.mean()),
        "max_abs_rgb_diff": float(arr.max()),
    }


def _capture_record(
    controller,
    cap: ImageCapture,
    *,
    object_id: int,
    spec: ObjectSpec,
    yaw_delta: float,
    scale_abs: float,
    current_scale_abs: float,
    frame_idx: int,
    object_dir: Path,
    cfg: RenderConfig,
    save_images: bool,
) -> Tuple[float, Dict, Image.Image | None, Image.Image | None]:
    current_scale_abs, center_error_world = _apply_base_plus_yaw(
        controller,
        object_id=object_id,
        spec=spec,
        yaw_delta=yaw_delta,
        scale_abs=scale_abs,
        current_scale_abs=current_scale_abs,
        cfg=cfg,
    )
    rgb_img, id_img = _capture_pil(controller=controller, cap=cap, commands=[])
    mask = _id_mask(id_img)
    stats = _mask_stats(mask)
    valid, invalid_reason = _valid_stats(stats, cfg)
    rgb_out = _apply_uniform_background(rgb_img, mask, cfg.uniform_bg_rgb)
    mask_img = _binary_mask_image(mask)

    image_relpath = None
    id_relpath = None
    mask_relpath = None
    if save_images:
        frames_dir = object_dir / "frames" / "a"
        masks_dir = object_dir / "masks" / "a"
        ids_dir = object_dir / "id_passes" / "a"
        _ensure_dir(frames_dir)
        _ensure_dir(masks_dir)
        _ensure_dir(ids_dir)
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
        "tdw_object_id": int(object_id),
        "base_pitch_deg": spec.base_pitch_deg,
        "base_yaw_deg": spec.base_yaw_deg,
        "base_roll_deg": spec.base_roll_deg,
        "yaw_delta_deg_unsigned": float(yaw_delta % 360.0),
        "yaw_delta_deg_signed": float(_signed_yaw(yaw_delta)),
        "tdw_euler_x_deg": spec.base_pitch_deg,
        "tdw_euler_y_deg": float((spec.base_yaw_deg + yaw_delta) % 360.0),
        "tdw_euler_z_deg": spec.base_roll_deg,
        "scale_abs": float(scale_abs),
        "source_base_scale_tdw": spec.base_scale_tdw,
        "camera_dist": spec.camera_dist,
        "object_base_y": cfg.object_base_y,
        "camera_height": cfg.camera_height,
        "look_at_height": cfg.look_at_height,
        "center_error_world": float(center_error_world),
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
        "stats": stats,
    }
    if abs(float(yaw_delta)) < 1e-9:
        rec["source_match"] = _source_match_stats(rgb_out, spec.source_base_preview)
    return current_scale_abs, rec, rgb_out if save_images else None, mask_img.convert("RGB") if save_images else None


def _render_sequence(
    controller,
    *,
    spec: ObjectSpec,
    yaw_deltas: List[float],
    scale_abs: float,
    object_dir: Path,
    cfg: RenderConfig,
    label_prefix: str,
) -> Dict:
    obj_id, cap = _setup_scene(controller, spec, object_dir, cfg)
    meta_path = object_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()
    current_scale = 1.0
    records = []
    rgb_items = []
    mask_items = []
    for frame_idx, yaw_delta in enumerate(yaw_deltas):
        current_scale, rec, rgb, mask_rgb = _capture_record(
            controller,
            cap,
            object_id=obj_id,
            spec=spec,
            yaw_delta=float(yaw_delta),
            scale_abs=float(scale_abs),
            current_scale_abs=current_scale,
            frame_idx=frame_idx,
            object_dir=object_dir,
            cfg=cfg,
            save_images=True,
        )
        _append_jsonl(meta_path, rec)
        records.append(rec)
        rgb_items.append((rgb, f"{label_prefix}\n{spec.model_name}\nyaw {yaw_delta:g}"))
        mask_items.append((mask_rgb, f"{label_prefix}\n{spec.model_name}\nyaw {yaw_delta:g}"))

    margins = [r["stats"]["margin_frac"] for r in records if r["stats"]]
    sizes = [r["stats"]["bbox_max_size_frac"] for r in records if r["stats"]]
    areas = [r["stats"]["area_frac"] for r in records if r["stats"]]
    summary = {
        "model_name": spec.model_name,
        "n_frames": len(records),
        "yaw_deltas": yaw_deltas,
        "scale_abs": float(scale_abs),
        "scale_multiplier_from_base": float(scale_abs / spec.base_scale_tdw),
        "pass": all(r["valid"] for r in records),
        "min_margin_frac": min(margins) if margins else None,
        "max_bbox_size_frac": max(sizes) if sizes else None,
        "min_area_frac": min(areas) if areas else None,
        "max_center_error_world": max(r["center_error_world"] for r in records) if records else None,
        "yaw0_source_match": records[0].get("source_match") if records else None,
        "records": records,
    }
    _write_json(object_dir / "summary.json", summary)
    return {"summary": summary, "rgb_items": rgb_items, "mask_items": mask_items}


def _evaluate_scale(
    controller,
    *,
    spec: ObjectSpec,
    scale_abs: float,
    cfg: RenderConfig,
    scratch_dir: Path,
) -> Dict:
    obj_id, cap = _setup_scene(controller, spec, scratch_dir, cfg)
    current_scale = 1.0
    records = []
    for frame_idx, yaw_delta in enumerate(SAFE_SCALE_YAWS):
        current_scale, rec, _, _ = _capture_record(
            controller,
            cap,
            object_id=obj_id,
            spec=spec,
            yaw_delta=float(yaw_delta),
            scale_abs=float(scale_abs),
            current_scale_abs=current_scale,
            frame_idx=frame_idx,
            object_dir=scratch_dir,
            cfg=cfg,
            save_images=False,
        )
        records.append(rec)
    return {
        "scale_abs": float(scale_abs),
        "scale_multiplier_from_base": float(scale_abs / spec.base_scale_tdw),
        "pass": all(r["valid"] for r in records),
        "min_margin_frac": min((r["stats"]["margin_frac"] for r in records if r["stats"]), default=None),
        "max_bbox_size_frac": max((r["stats"]["bbox_max_size_frac"] for r in records if r["stats"]), default=None),
        "min_area_frac": min((r["stats"]["area_frac"] for r in records if r["stats"]), default=None),
        "invalid_reasons": sorted({r["invalid_reason"] for r in records if not r["valid"]}),
    }


def _choose_safe_scale(
    controller,
    *,
    spec: ObjectSpec,
    multipliers: List[float],
    probe_dir: Path,
    cfg: RenderConfig,
) -> Dict:
    _ensure_dir(probe_dir)
    attempts = []
    best = None
    for multiplier in sorted(float(v) for v in multipliers):
        result = _evaluate_scale(
            controller,
            spec=spec,
            scale_abs=float(spec.base_scale_tdw * multiplier),
            cfg=cfg,
            scratch_dir=probe_dir / f"_scratch_{multiplier:.3f}",
        )
        attempts.append(result)
        if result["pass"]:
            best = result
    summary = {
        "model_name": spec.model_name,
        "source_base_scale_tdw": spec.base_scale_tdw,
        "safe_scale_probe_yaws": SAFE_SCALE_YAWS,
        "chosen_scale_abs": best["scale_abs"] if best else None,
        "chosen_multiplier_from_base": best["scale_multiplier_from_base"] if best else None,
        "pass": best is not None,
        "attempts": attempts,
    }
    _write_json(probe_dir / "safe_scale_summary.json", summary)
    return summary


def _contact_sheet(items: List[Tuple[Image.Image, str]], cols: int, cell: int = 160) -> Image.Image:
    rows = int(np.ceil(len(items) / float(cols)))
    label_h = 44
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label) in enumerate(items):
        row, col = divmod(idx, cols)
        x = col * cell
        y = row * (cell + label_h)
        thumb = img.convert("RGB").resize((cell, cell), resample=Image.BILINEAR)
        sheet.paste(thumb, (x, y))
        draw.multiline_text((x + 4, y + cell + 3), label, fill=(0, 0, 0), spacing=1)
    return sheet


def _write_run_summary(
    out_root: Path,
    *,
    source_index_path: Path,
    source_panel_path: Path,
    specs: List[ObjectSpec],
    cfg: RenderConfig,
    exact_summaries: Dict,
    safe_scale_summaries: Dict,
    proposed_summaries: Dict,
) -> None:
    summary = {
        "semantic_lock": {
            "must_be_true": [
                "This is a sanity-check-only run, not the full 2-degree dataset.",
                "All base poses/scales/camera values come from the existing base-panel dataset index.",
                "True yaw is applied as TDW Euler y = base_yaw + yaw_delta.",
                "The exact-base-scale yaw_delta=0 frames are compared against the existing base previews.",
            ],
            "must_not_happen": [
                "Do not replace base poses with guessed pitch=0 poses.",
                "Do not generate 180 frames per object.",
                "Do not modify keypoint_net training or L_loc.",
            ],
        },
        "source_index_path": str(source_index_path),
        "source_panel_path": str(source_panel_path),
        "object_order": [s.model_name for s in specs],
        "config": {
            "img_size": cfg.img_size,
            "object_base_y": cfg.object_base_y,
            "camera_height": cfg.camera_height,
            "look_at_height": cfg.look_at_height,
            "uniform_bg_rgb": list(cfg.uniform_bg_rgb),
            "min_margin": cfg.min_margin,
            "max_size": cfg.max_size,
            "center_tolerance_world": cfg.center_tolerance_world,
            "sanity_yaws": SANITY_YAWS,
            "safe_scale_yaws": SAFE_SCALE_YAWS,
        },
        "base_specs": [
            {
                "model_name": s.model_name,
                "base_pitch_deg": s.base_pitch_deg,
                "base_yaw_deg": s.base_yaw_deg,
                "base_roll_deg": s.base_roll_deg,
                "base_scale_tdw": s.base_scale_tdw,
                "camera_dist": s.camera_dist,
                "source_base_preview": str(s.source_base_preview),
            }
            for s in specs
        ],
        "exact_base_scale": exact_summaries,
        "safe_scale_probe": safe_scale_summaries,
        "proposed_safe_scale": proposed_summaries,
        "full_dataset_spec_after_approval": {
            "yaw_range": "0..358",
            "yaw_step_deg": 2,
            "frames_per_object": 180,
            "total_rgb_frames": 1080,
            "total_mask_frames": 1080,
            "indices": {"skip1": "2 deg", "skip3": "6 deg", "skip5": "10 deg"},
            "note": "Existing keypoint_net/dataset.py does not consume JSON index files yet.",
        },
    }
    _write_json(out_root / "sanity_index.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_index", type=str, default="./_full_2d_affine_selected_final_512/dataset_index.json")
    parser.add_argument("--models_file", type=str, default="./models_affine_final_6.txt")
    parser.add_argument("--out_dir", type=str, default="./_sanity_true_yaw_from_base_panel_512")
    parser.add_argument("--port", type=int, default=1082)
    parser.add_argument("--launch_build", action="store_true")
    parser.add_argument("--min_margin", type=float, default=0.01)
    parser.add_argument("--max_size", type=float, default=0.82)
    parser.add_argument("--center_tolerance_world", type=float, default=0.005)
    parser.add_argument(
        "--safe_scale_multipliers",
        type=float,
        nargs="+",
        default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0],
    )
    args = parser.parse_args()

    source_index_path = Path(args.source_index).expanduser().resolve()
    models_file = Path(args.models_file).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Refusing to mix outputs into nonempty folder: {out_root}")
    _ensure_dir(out_root)

    specs, cfg, source_index = _load_source_specs(source_index_path, models_file)
    cfg = RenderConfig(
        img_size=cfg.img_size,
        object_base_y=cfg.object_base_y,
        camera_height=cfg.camera_height,
        look_at_height=cfg.look_at_height,
        uniform_bg_rgb=cfg.uniform_bg_rgb,
        min_margin=float(args.min_margin),
        max_size=float(args.max_size),
        center_tolerance_world=float(args.center_tolerance_world),
    )
    source_panel_path = Path(str(source_index["base_panel_path"]))

    _write_json(
        out_root / "semantic_lock.json",
        {
            "source_index": str(source_index_path),
            "source_panel": str(source_panel_path),
            "rule": "Use exact base-panel poses/scales/camera; add only TDW yaw deltas for sanity frames.",
        },
    )

    controller = _make_controller(port=int(args.port), launch_build=bool(args.launch_build))
    try:
        exact_rgb_items = []
        exact_mask_items = []
        exact_summaries = {}
        for spec in specs:
            rendered = _render_sequence(
                controller,
                spec=spec,
                yaw_deltas=SANITY_YAWS,
                scale_abs=spec.base_scale_tdw,
                object_dir=out_root / "exact_base_scale" / "train" / spec.model_name,
                cfg=cfg,
                label_prefix="base-scale",
            )
            exact_summaries[spec.model_name] = rendered["summary"]
            exact_rgb_items.extend(rendered["rgb_items"])
            exact_mask_items.extend(rendered["mask_items"])
        _contact_sheet(exact_rgb_items, cols=len(SANITY_YAWS)).save(out_root / "exact_base_scale_rgb_contact_sheet.png")
        _contact_sheet(exact_mask_items, cols=len(SANITY_YAWS)).save(out_root / "exact_base_scale_mask_contact_sheet.png")

        safe_scale_summaries = {}
        proposed_rgb_items = []
        proposed_mask_items = []
        proposed_summaries = {}
        for spec in specs:
            safe = _choose_safe_scale(
                controller,
                spec=spec,
                multipliers=[float(v) for v in args.safe_scale_multipliers],
                probe_dir=out_root / "safe_scale_probe" / spec.model_name,
                cfg=cfg,
            )
            safe_scale_summaries[spec.model_name] = safe
            if not safe["pass"]:
                proposed_summaries[spec.model_name] = {"pass": False, "reason": "no_safe_scale"}
                continue
            rendered = _render_sequence(
                controller,
                spec=spec,
                yaw_deltas=SANITY_YAWS,
                scale_abs=float(safe["chosen_scale_abs"]),
                object_dir=out_root / "proposed_safe_scale" / "train" / spec.model_name,
                cfg=cfg,
                label_prefix=f"safe-scale x{safe['chosen_multiplier_from_base']:.2f}",
            )
            proposed_summaries[spec.model_name] = rendered["summary"]
            proposed_rgb_items.extend(rendered["rgb_items"])
            proposed_mask_items.extend(rendered["mask_items"])
        _contact_sheet(proposed_rgb_items, cols=len(SANITY_YAWS)).save(out_root / "proposed_safe_scale_rgb_contact_sheet.png")
        _contact_sheet(proposed_mask_items, cols=len(SANITY_YAWS)).save(out_root / "proposed_safe_scale_mask_contact_sheet.png")

        _write_run_summary(
            out_root,
            source_index_path=source_index_path,
            source_panel_path=source_panel_path,
            specs=specs,
            cfg=cfg,
            exact_summaries=exact_summaries,
            safe_scale_summaries=safe_scale_summaries,
            proposed_summaries=proposed_summaries,
        )
        print(json.dumps({"out_root": str(out_root), "sanity_index": str(out_root / "sanity_index.json")}, indent=2))
    finally:
        controller.communicate([{"$type": "terminate"}])


if __name__ == "__main__":
    main()

