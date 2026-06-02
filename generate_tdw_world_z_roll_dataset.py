"""Generate the Phase-A TDW World-Z roll dataset.

This script stores the dataset-generation code only. It does not belong in the
training package because its job is to render TDW image/mask assets.

Semantic lock:
- Render the object in TDW for every frame; do not use PIL/post-render rotation.
- Start from the selected six-object base pose, then apply
  rotate_object_by(axis="roll", is_world=True, use_centroid=True).
- Reset to the base pose before each theta, so frame t is an absolute pose.
- Save RGB frames, raw TDW _id passes, binary masks, metadata, and cyclic pair
  indices for skip 1/3/5.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ObjectSpec:
    model_name: str
    base_pitch_deg: float = 0.0
    base_yaw_deg: float = 0.0
    base_roll_deg: float = 0.0
    source_base_scale_tdw: float = 1.0
    scale_abs: float = 1.0
    split: str = "train"


DEFAULT_OBJECT_SPECS: Dict[str, ObjectSpec] = {
    "engineers_hammer_vray": ObjectSpec("engineers_hammer_vray", base_yaw_deg=0.0),
    "b03_banana_01_high": ObjectSpec("b03_banana_01_high", base_yaw_deg=180.0),
    "kettle": ObjectSpec("kettle", base_yaw_deg=0.0),
    "dewalt_compact_drill_vray": ObjectSpec("dewalt_compact_drill_vray", base_yaw_deg=90.0),
    "toy_monkey_medium": ObjectSpec("toy_monkey_medium", base_yaw_deg=0.0),
    "b03_trumpet_vray": ObjectSpec("b03_trumpet_vray", base_yaw_deg=90.0),
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _read_model_names(path: Optional[Path]) -> List[str]:
    if path is None:
        return list(DEFAULT_OBJECT_SPECS)
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if not names:
        raise ValueError(f"No model names found in {path}")
    return names


def _maybe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_spec(model_name: str, raw: Optional[Dict[str, Any]]) -> ObjectSpec:
    base = DEFAULT_OBJECT_SPECS.get(model_name, ObjectSpec(model_name=model_name))
    if not raw:
        return base

    pose = raw.get("base_pose", raw)
    scale = raw.get("full_dataset_scale_recommendation", raw.get("scale", raw))
    return ObjectSpec(
        model_name=model_name,
        base_pitch_deg=_maybe_float(
            pose.get("base_pitch_deg", pose.get("pitch_deg", pose.get("pitch", None))),
            base.base_pitch_deg,
        ),
        base_yaw_deg=_maybe_float(
            pose.get("base_yaw_deg", pose.get("yaw_deg", pose.get("yaw", None))),
            base.base_yaw_deg,
        ),
        base_roll_deg=_maybe_float(
            pose.get("base_roll_deg", pose.get("roll_deg", pose.get("roll", None))),
            base.base_roll_deg,
        ),
        source_base_scale_tdw=_maybe_float(
            raw.get("source_base_scale_tdw", raw.get("base_scale_tdw", None)),
            base.source_base_scale_tdw,
        ),
        scale_abs=_maybe_float(
            scale.get("scale_abs", raw.get("chosen_scale_abs", raw.get("scale_abs", None)))
            if isinstance(scale, dict)
            else raw.get("scale_abs", None),
            base.scale_abs,
        ),
        split=str(raw.get("split", base.split)),
    )


def _load_specs(
    *,
    model_names: Sequence[str],
    specs_json: Optional[Path],
    scale_index: Optional[Path],
) -> Dict[str, ObjectSpec]:
    """Load object base poses/scales from flexible JSON, falling back to defaults.

    Accepted JSON shapes:
    - {"objects": {"model": {...}}}
    - {"base_specs": {"model": {...}}}
    - {"scale_summaries": {"model": {...}}}
    - {"scale_summaries": [{... "model_name": "..."}]}
    """

    merged: Dict[str, Dict[str, Any]] = {}
    for path in [specs_json, scale_index]:
        if path is None:
            continue
        data = json.loads(path.read_text())
        for key in ["objects", "base_specs", "scale_summaries", "safe_scale_summaries"]:
            block = data.get(key)
            if isinstance(block, dict):
                for model, raw in block.items():
                    merged.setdefault(model, {}).update(raw if isinstance(raw, dict) else {})
            elif isinstance(block, list):
                for raw in block:
                    if isinstance(raw, dict):
                        model = raw.get("model_name") or raw.get("object") or raw.get("name")
                        if model:
                            merged.setdefault(model, {}).update(raw)

    return {name: _coerce_spec(name, merged.get(name)) for name in model_names}


def _camera_position(camera_dist: float, object_base_y: float, camera_height: float) -> Dict[str, float]:
    return {"x": 0.0, "y": float(object_base_y + camera_height), "z": float(camera_dist)}


def _look_at(object_base_y: float, look_at_height: float) -> Dict[str, float]:
    return {"x": 0.0, "y": float(object_base_y + look_at_height), "z": 0.0}


def _find_capture(path: Path, prefix: str, frame_idx: int) -> Path:
    candidates = [
        path / "frames" / "a" / f"{prefix}_{frame_idx:04d}.png",
        path / "frames" / "a" / f"{prefix}_{frame_idx:04d}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted((path / "frames" / "a").glob(f"{prefix}_*.png"))
    if frame_idx < len(matches):
        return matches[frame_idx]
    raise FileNotFoundError(f"Could not find {prefix} capture for frame {frame_idx} in {path}")


def _id_to_binary_mask(id_path: Path, out_path: Path) -> Dict[str, Any]:
    """Convert TDW _id pass to a binary foreground mask.

    With one object in a fixed room, the background color in the _id pass is the
    dominant color. Foreground is every pixel not equal to that dominant color.
    This avoids assuming a particular segmentation color.
    """

    arr = np.asarray(Image.open(id_path).convert("RGB"))
    flat = arr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[int(np.argmax(counts))]
    mask = np.any(arr != bg.reshape(1, 1, 3), axis=-1)
    mask_u8 = (mask.astype(np.uint8) * 255)
    _ensure_dir(out_path.parent)
    Image.fromarray(mask_u8, mode="L").save(out_path)
    return _mask_stats(mask_u8)


def _mask_stats(mask_u8: np.ndarray) -> Dict[str, Any]:
    ys, xs = np.where(mask_u8 > 0)
    h, w = mask_u8.shape[:2]
    if len(xs) == 0:
        return {
            "mask_nonempty": False,
            "bbox": None,
            "area_frac": 0.0,
            "margin_frac": 0.0,
            "centroid_xy": None,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    margin = min(x0, y0, w - 1 - x1, h - 1 - y1) / float(min(w, h))
    return {
        "mask_nonempty": True,
        "bbox": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "area_frac": float(len(xs) / (w * h)),
        "margin_frac": float(margin),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _draw_contact_sheet(items: List[Tuple[str, Path]], out_path: Path, thumb: int = 160) -> None:
    if not items:
        return
    cols = min(6, len(items))
    rows = int(math.ceil(len(items) / cols))
    label_h = 28
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(items):
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb, thumb))
        x = (i % cols) * thumb + (thumb - im.width) // 2
        y = (i // cols) * (thumb + label_h)
        sheet.paste(im, (x, y))
        draw.text(((i % cols) * thumb + 4, y + thumb + 4), label[:24], fill=(0, 0, 0))
    _ensure_dir(out_path.parent)
    sheet.save(out_path)


def _make_controller(port: int):
    from tdw.controller import Controller

    return Controller(port=port)


def _setup_scene(
    *,
    controller: Any,
    object_spec: ObjectSpec,
    object_base_y: float,
    camera_dist: float,
    camera_height: float,
    look_at_height: float,
    img_size: int,
    frames_dir: Path,
) -> Tuple[Any, Any, int]:
    from tdw.add_ons.image_capture import ImageCapture
    from tdw.add_ons.third_person_camera import ThirdPersonCamera
    from tdw.tdw_utils import TDWUtils

    controller.add_ons.clear()
    cam = ThirdPersonCamera(
        avatar_id="a",
        position=_camera_position(camera_dist, object_base_y, camera_height),
        look_at=_look_at(object_base_y, look_at_height),
    )
    cap = ImageCapture(avatar_ids=["a"], path=str(frames_dir.resolve()), png=True, pass_masks=["_img", "_id"])
    controller.add_ons.extend([cam, cap])
    cap.set(frequency="never", avatar_ids=["a"], save=False)

    obj_id = controller.get_unique_id()
    commands = [
        {"$type": "load_scene", "scene_name": "ProcGenScene"},
        TDWUtils.create_empty_room(12, 12),
        {"$type": "set_post_process", "value": False},
        {"$type": "set_screen_size", "width": img_size, "height": img_size},
        {"$type": "set_target_framerate", "framerate": 30},
        {"$type": "set_render_quality", "render_quality": 5},
        controller.get_add_object(
            model_name=object_spec.model_name,
            object_id=obj_id,
            position={"x": 0.0, "y": float(object_base_y), "z": 0.0},
        ),
    ]
    controller.communicate(commands)

    scale_factor = object_spec.scale_abs / max(object_spec.source_base_scale_tdw, 1e-9)
    controller.communicate(
        [
            {
                "$type": "scale_object",
                "id": obj_id,
                "scale_factor": {"x": scale_factor, "y": scale_factor, "z": scale_factor},
            }
        ]
    )
    return cam, cap, obj_id


def _render_object(
    *,
    controller: Any,
    object_spec: ObjectSpec,
    object_dir: Path,
    theta_values: Sequence[float],
    img_size: int,
    object_base_y: float,
    camera_dist: float,
    camera_height: float,
    look_at_height: float,
) -> Dict[str, Any]:
    if object_dir.exists():
        shutil.rmtree(object_dir)
    frames_dir = object_dir / "frames"
    masks_dir = object_dir / "masks" / "a"
    _ensure_dir(frames_dir)
    _ensure_dir(masks_dir)
    meta_path = object_dir / "meta.jsonl"

    _, cap, obj_id = _setup_scene(
        controller=controller,
        object_spec=object_spec,
        object_base_y=object_base_y,
        camera_dist=camera_dist,
        camera_height=camera_height,
        look_at_height=look_at_height,
        img_size=img_size,
        frames_dir=frames_dir,
    )

    rows: List[Dict[str, Any]] = []
    rgb_sheet_items: List[Tuple[str, Path]] = []
    mask_sheet_items: List[Tuple[str, Path]] = []
    stats: List[Dict[str, Any]] = []

    for frame_idx, theta in enumerate(theta_values):
        cap.set(frequency="once", avatar_ids=["a"], save=True)
        controller.communicate(
            [
                {
                    "$type": "rotate_object_to_euler_angles",
                    "id": obj_id,
                    "euler_angles": {
                        "x": object_spec.base_pitch_deg,
                        "y": object_spec.base_yaw_deg,
                        "z": object_spec.base_roll_deg,
                    },
                },
                {
                    "$type": "rotate_object_by",
                    "id": obj_id,
                    "angle": float(theta),
                    "axis": "roll",
                    "is_world": True,
                    "use_centroid": True,
                },
            ]
        )

        rgb_path = _find_capture(object_dir, "img", frame_idx)
        id_path = _find_capture(object_dir, "id", frame_idx)
        mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
        stat = _id_to_binary_mask(id_path, mask_path)
        stats.append(stat)

        rel = lambda p: str(p.relative_to(object_dir.parent.parent))
        row = {
            "frame_index": frame_idx,
            "image_relpath": rel(rgb_path),
            "id_relpath": rel(id_path),
            "mask_relpath": rel(mask_path),
            "model_name": object_spec.model_name,
            "tdw_object_id": obj_id,
            "operator_name": "tdw_world_z_roll",
            "tdw_axis": "roll",
            "is_world": True,
            "use_centroid": True,
            "rotation_is_tdw_rerendered": True,
            "theta_deg": float(theta % 360.0),
            "base_pitch_deg": object_spec.base_pitch_deg,
            "base_yaw_deg": object_spec.base_yaw_deg,
            "base_roll_deg": object_spec.base_roll_deg,
            "source_base_scale_tdw": object_spec.source_base_scale_tdw,
            "scale_abs": object_spec.scale_abs,
            "camera": {
                "position": _camera_position(camera_dist, object_base_y, camera_height),
                "look_at": _look_at(object_base_y, look_at_height),
                "img_size": img_size,
            },
            **stat,
        }
        rows.append(row)
        _append_jsonl(meta_path, row)

        if frame_idx in {0, 30, 60, 90, 120, 150} or len(theta_values) <= 12:
            rgb_sheet_items.append((f"{object_spec.model_name} {theta:g}", rgb_path))
            mask_sheet_items.append((f"{object_spec.model_name} {theta:g}", mask_path))

    _draw_contact_sheet(rgb_sheet_items, object_dir / "rgb_contact_sheet.png")
    _draw_contact_sheet(mask_sheet_items, object_dir / "mask_contact_sheet.png")

    margins = [s["margin_frac"] for s in stats if s["mask_nonempty"]]
    areas = [s["area_frac"] for s in stats if s["mask_nonempty"]]
    summary = {
        "model_name": object_spec.model_name,
        "n_frames": len(theta_values),
        "object_spec": asdict(object_spec),
        "meta_jsonl": str(meta_path),
        "all_masks_nonempty": all(s["mask_nonempty"] for s in stats),
        "min_margin_frac": min(margins) if margins else 0.0,
        "max_area_frac": max(areas) if areas else 0.0,
        "mean_area_frac": float(np.mean(areas)) if areas else 0.0,
    }
    _write_json(object_dir / "summary.json", summary)
    return {"summary": summary, "frames": rows}


def _write_pair_indices(
    *,
    out_root: Path,
    object_frames: Dict[str, List[Dict[str, Any]]],
    frame_skips: Iterable[int],
    theta_step_deg: float,
) -> None:
    indices_dir = out_root / "indices"
    _ensure_dir(indices_dir)
    for skip in frame_skips:
        pairs = []
        for model_name, frames in object_frames.items():
            n = len(frames)
            for src_idx in range(n):
                dst_idx = (src_idx + skip) % n
                src = frames[src_idx]
                dst = frames[dst_idx]
                pairs.append(
                    {
                        "model_name": model_name,
                        "src_frame_index": src_idx,
                        "dst_frame_index": dst_idx,
                        "src_theta_deg": src["theta_deg"],
                        "dst_theta_deg": dst["theta_deg"],
                        "src_image_relpath": src["image_relpath"],
                        "dst_image_relpath": dst["image_relpath"],
                        "src_mask_relpath": src["mask_relpath"],
                        "dst_mask_relpath": dst["mask_relpath"],
                    }
                )
        _write_json(
            indices_dir / f"pairs_skip{skip}_cyclic.json",
            {
                "cyclic": True,
                "frame_skip": skip,
                "theta_step_deg": theta_step_deg,
                "delta_theta_deg": skip * theta_step_deg,
                "pairs": pairs,
            },
        )


def _theta_values(full: bool, theta_step_deg: float, sanity_yaws: Sequence[float]) -> List[float]:
    if full:
        n = int(round(360.0 / theta_step_deg))
        return [i * theta_step_deg for i in range(n)]
    return [float(v) for v in sanity_yaws]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--models_file", type=Path, default=Path("models_affine_final_6.txt"))
    ap.add_argument("--object_specs_json", type=Path, default=None)
    ap.add_argument("--scale_index", type=Path, default=None)
    ap.add_argument("--port", type=int, default=1071)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--camera_dist", type=float, default=0.56)
    ap.add_argument("--object_base_y", type=float, default=1.0)
    ap.add_argument("--camera_height", type=float, default=0.0)
    ap.add_argument("--look_at_height", type=float, default=0.0)
    ap.add_argument("--theta_step_deg", type=float, default=2.0)
    ap.add_argument("--full", action="store_true", help="Render 0..358 by 2 deg. Omit for sparse sanity render.")
    ap.add_argument("--sanity_yaws", type=float, nargs="+", default=[0, 60, 120, 180, 240, 300])
    ap.add_argument("--frame_skips", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    model_names = _read_model_names(args.models_file if args.models_file.exists() else None)
    specs = _load_specs(
        model_names=model_names,
        specs_json=args.object_specs_json,
        scale_index=args.scale_index,
    )

    out_root = args.out_dir.expanduser().resolve()
    _ensure_dir(out_root)

    theta_values = _theta_values(args.full, args.theta_step_deg, args.sanity_yaws)
    controller = _make_controller(args.port)
    object_frames: Dict[str, List[Dict[str, Any]]] = {}
    dataset_index: Dict[str, Any] = {
        "semantic_lock": [
            "TDW rerendered World-Z roll, not PIL rotation.",
            "Base pose is reset before every absolute theta.",
            "rotate_object_by(axis='roll', is_world=True, use_centroid=True).",
            "Masks come from TDW _id pass and are evaluation artifacts.",
            "No post-render 2D recentering/cropping.",
        ],
        "config": {
            "operator_name": "tdw_world_z_roll",
            "theta_step_deg": args.theta_step_deg,
            "full": bool(args.full),
            "img_size": args.img_size,
            "camera_position": _camera_position(args.camera_dist, args.object_base_y, args.camera_height),
            "look_at": _look_at(args.object_base_y, args.look_at_height),
        },
        "splits": {"train": [], "test": []},
        "objects": {},
    }

    try:
        for i, model_name in enumerate(model_names, start=1):
            spec = specs[model_name]
            split = spec.split
            dataset_index["splits"].setdefault(split, []).append(model_name)
            obj_dir = out_root / split / model_name
            print(f"[{i}/{len(model_names)}] Rendering {model_name} -> {obj_dir}")
            rendered = _render_object(
                controller=controller,
                object_spec=spec,
                object_dir=obj_dir,
                theta_values=theta_values,
                img_size=args.img_size,
                object_base_y=args.object_base_y,
                camera_dist=args.camera_dist,
                camera_height=args.camera_height,
                look_at_height=args.look_at_height,
            )
            object_frames[model_name] = rendered["frames"]
            dataset_index["objects"][model_name] = rendered["summary"]
    finally:
        controller.communicate([{"$type": "terminate"}])

    _write_pair_indices(
        out_root=out_root,
        object_frames=object_frames,
        frame_skips=args.frame_skips,
        theta_step_deg=args.theta_step_deg,
    )
    _write_json(out_root / "dataset_index.json", dataset_index)
    print(f"Wrote dataset index: {out_root / 'dataset_index.json'}")


if __name__ == "__main__":
    main()
