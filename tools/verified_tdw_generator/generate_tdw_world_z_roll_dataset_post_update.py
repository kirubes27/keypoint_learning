"""Generate the approved TDW world-Z roll dataset.

This full generator uses the scale sanity output from
`_sanity_tdw_world_z_roll_scale_probe_512_v2` and renders:

    base pose -> rotate_object_by(axis="roll", is_world=True, use_centroid=True)

for theta = 0, 2, ..., 358.

It saves RGB frames, raw TDW _id passes, binary masks, per-object metadata, pair
indices, and dataset-level summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from compare_hammer_rotation_operators import _apply_tdw_operator
from create_true_yaw_from_base_panel_sanity import (
    ObjectSpec,
    RenderConfig,
    _append_jsonl,
    _apply_uniform_background,
    _binary_mask_image,
    _capture_pil,
    _contact_sheet,
    _ensure_dir,
    _id_mask,
    _load_source_specs,
    _make_controller,
    _mask_stats,
    _source_match_stats,
    _setup_scene,
    _valid_stats,
    _write_json,
)


OPERATOR_NAME = "tdw_world_z_roll"
FULL_THETAS = [float(v) for v in range(0, 360, 2)]
SAMPLE_THETAS = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
PAIR_SKIPS = [1, 3, 5]


def _load_scale_recommendations(scale_index_path: Path) -> Dict[str, Dict[str, float]]:
    idx = json.loads(scale_index_path.read_text())
    recommendations = idx.get("full_dataset_scale_recommendation")
    if not isinstance(recommendations, dict):
        raise ValueError(f"No full_dataset_scale_recommendation in {scale_index_path}")
    return {
        str(name): {
            "scale_abs": float(rec["scale_abs"]),
            "scale_multiplier_from_base": float(rec["scale_multiplier_from_base"]),
        }
        for name, rec in recommendations.items()
    }


def _summarize_records(records: List[Dict]) -> Dict:
    valid_records = [r for r in records if r["stats"] is not None]
    invalid = [r for r in records if not r["valid"]]
    return {
        "pass": len(invalid) == 0,
        "n_frames": len(records),
        "n_invalid": len(invalid),
        "invalid_examples": invalid[:10],
        "invalid_reasons": sorted({r["invalid_reason"] for r in invalid}),
        "min_margin_frac": min((r["stats"]["margin_frac"] for r in valid_records), default=None),
        "max_bbox_size_frac": max((r["stats"]["bbox_max_size_frac"] for r in valid_records), default=None),
        "min_area_frac": min((r["stats"]["area_frac"] for r in valid_records), default=None),
        "max_center_error_world": max((r["center_error_world"] for r in records), default=None),
        "max_abs_bbox_center_x_offset_frac": max(
            (abs(r["stats"]["bbox_center_x_offset_frac"]) for r in valid_records),
            default=None,
        ),
        "max_abs_bbox_center_y_offset_frac": max(
            (abs(r["stats"]["bbox_center_y_offset_frac"]) for r in valid_records),
            default=None,
        ),
    }


def _render_object_full(
    controller,
    *,
    spec: ObjectSpec,
    cfg: RenderConfig,
    scale_abs: float,
    out_root: Path,
) -> Dict:
    object_dir = out_root / "train" / spec.model_name
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
    sample_rgb_items: List[Tuple[Image.Image, str]] = []
    sample_mask_items: List[Tuple[Image.Image, str]] = []
    sample_theta_set = set(SAMPLE_THETAS)
    scale_multiplier = float(scale_abs / spec.base_scale_tdw)

    for frame_idx, theta in enumerate(FULL_THETAS):
        current_scale, center_error, command_log = _apply_tdw_operator(
            controller,
            object_id=obj_id,
            spec=spec,
            theta_deg=float(theta),
            operator_name=OPERATOR_NAME,
            scale_abs=float(scale_abs),
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
            "frame_index": int(frame_idx),
            "image_index": int(frame_idx),
            "image_relpath": image_relpath,
            "mask_relpath": mask_relpath,
            "id_relpath": id_relpath,
            "model_name": spec.model_name,
            "operator_name": OPERATOR_NAME,
            "theta_deg": float(theta),
            "theta_step_deg": 2.0,
            "rotation_is_tdw_rerendered": True,
            "tdw_axis": "roll",
            "is_world": True,
            "use_centroid": True,
            "base_pitch_deg": spec.base_pitch_deg,
            "base_yaw_deg": spec.base_yaw_deg,
            "base_roll_deg": spec.base_roll_deg,
            "source_base_scale_tdw": spec.base_scale_tdw,
            "scale_abs": float(scale_abs),
            "scale_multiplier_from_base": scale_multiplier,
            "camera_position": [0.0, cfg.object_base_y + cfg.camera_height, spec.camera_dist],
            "look_at": [0.0, cfg.object_base_y + cfg.look_at_height, 0.0],
            "camera_dist": spec.camera_dist,
            "object_base_y": cfg.object_base_y,
            "camera_height": cfg.camera_height,
            "look_at_height": cfg.look_at_height,
            "uniform_bg_rgb": list(cfg.uniform_bg_rgb),
            "center_error_world": float(center_error),
            "valid": bool(valid),
            "invalid_reason": invalid_reason,
            "stats": stats,
            "command_log": command_log,
        }
        if abs(float(theta)) < 1e-9:
            rec["source_match"] = _source_match_stats(rgb_out, spec.source_base_preview)
        _append_jsonl(meta_path, rec)
        records.append(rec)

        if float(theta) in sample_theta_set:
            sample_rgb_items.append((rgb_out, f"x{scale_multiplier:.2f}\n{spec.model_name}\ntheta {theta:g}"))
            sample_mask_items.append((mask_img.convert("RGB"), f"x{scale_multiplier:.2f}\n{spec.model_name}\ntheta {theta:g}"))

    summary = {
        "model_name": spec.model_name,
        "operator_name": OPERATOR_NAME,
        "scale_abs": float(scale_abs),
        "scale_multiplier_from_base": scale_multiplier,
        "theta_range": "0..358",
        "theta_step_deg": 2.0,
        "records": records,
        **_summarize_records(records),
    }
    _write_json(object_dir / "summary.json", summary)
    return {
        "summary": summary,
        "sample_rgb_items": sample_rgb_items,
        "sample_mask_items": sample_mask_items,
    }


def _make_pair_records(specs: List[ObjectSpec], skip: int, *, cyclic: bool) -> List[Dict]:
    pairs: List[Dict] = []
    n = len(FULL_THETAS)
    for spec in specs:
        for src_idx, src_theta in enumerate(FULL_THETAS):
            dst_idx = src_idx + int(skip)
            if cyclic:
                dst_idx = dst_idx % n
            elif dst_idx >= n:
                continue
            dst_theta = FULL_THETAS[dst_idx]
            pairs.append(
                {
                    "model_name": spec.model_name,
                    "operator_name": OPERATOR_NAME,
                    "skip_frames": int(skip),
                    "delta_theta_deg": float(2 * skip),
                    "cyclic": bool(cyclic),
                    "src_frame_index": int(src_idx),
                    "dst_frame_index": int(dst_idx),
                    "src_theta_deg": float(src_theta),
                    "dst_theta_deg": float(dst_theta),
                    "src_image_relpath": f"train/{spec.model_name}/frames/a/img_{src_idx:04d}.png",
                    "dst_image_relpath": f"train/{spec.model_name}/frames/a/img_{dst_idx:04d}.png",
                    "src_mask_relpath": f"train/{spec.model_name}/masks/a/mask_{src_idx:04d}.png",
                    "dst_mask_relpath": f"train/{spec.model_name}/masks/a/mask_{dst_idx:04d}.png",
                }
            )
    return pairs


def _write_indices(out_root: Path, specs: List[ObjectSpec]) -> Dict:
    index_dir = out_root / "indices"
    _ensure_dir(index_dir)
    info = {}
    for skip in PAIR_SKIPS:
        cyclic_pairs = _make_pair_records(specs, skip, cyclic=True)
        nocycle_pairs = _make_pair_records(specs, skip, cyclic=False)
        cyclic_path = index_dir / f"pairs_skip{skip}_cyclic.json"
        nocycle_path = index_dir / f"pairs_skip{skip}_nocycle.json"
        _write_json(
            cyclic_path,
            {
                "operator_name": OPERATOR_NAME,
                "skip_frames": int(skip),
                "delta_theta_deg": float(2 * skip),
                "cyclic": True,
                "pairs": cyclic_pairs,
            },
        )
        _write_json(
            nocycle_path,
            {
                "operator_name": OPERATOR_NAME,
                "skip_frames": int(skip),
                "delta_theta_deg": float(2 * skip),
                "cyclic": False,
                "pairs": nocycle_pairs,
            },
        )
        info[f"skip{skip}"] = {
            "delta_theta_deg": float(2 * skip),
            "cyclic_path": str(cyclic_path),
            "cyclic_n_pairs": len(cyclic_pairs),
            "nocycle_path": str(nocycle_path),
            "nocycle_n_pairs": len(nocycle_pairs),
        }

    phase_split = {"train": [], "val": [], "test": []}
    arc_holdout = {"train": [], "test": []}
    for spec in specs:
        for frame_idx, theta in enumerate(FULL_THETAS):
            rec = {
                "model_name": spec.model_name,
                "frame_index": int(frame_idx),
                "theta_deg": float(theta),
                "image_relpath": f"train/{spec.model_name}/frames/a/img_{frame_idx:04d}.png",
            }
            phase = int(theta) % 6
            if phase == 0:
                phase_split["train"].append(rec)
            elif phase == 2:
                phase_split["val"].append(rec)
            else:
                phase_split["test"].append(rec)

            if theta < 300.0:
                arc_holdout["train"].append(rec)
            else:
                arc_holdout["test"].append(rec)

    phase_path = index_dir / "split_phase_mod6.json"
    arc_path = index_dir / "split_arc_holdout_300_360.json"
    _write_json(phase_path, phase_split)
    _write_json(arc_path, arc_holdout)
    info["phase_split_path"] = str(phase_path)
    info["arc_holdout_path"] = str(arc_path)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_index", type=str, default="./_full_2d_affine_selected_final_512/dataset_index.json")
    parser.add_argument("--models_file", type=str, default="./models_affine_final_6.txt")
    parser.add_argument(
        "--scale_index",
        type=str,
        default="./_sanity_tdw_world_z_roll_scale_probe_512_v2/sanity_index.json",
    )
    parser.add_argument("--out_dir", type=str, default="./_tdw_world_z_roll_base_panel_512")
    parser.add_argument("--port", type=int, default=1090)
    parser.add_argument("--launch_build", action="store_true")
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--max_size", type=float, default=0.82)
    parser.add_argument("--center_tolerance_world", type=float, default=0.005)
    parser.add_argument(
        "--scale_override",
        action="append",
        default=[],
        help="Optional per-object multiplier override as model_name=multiplier_from_base.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Refusing to mix outputs into nonempty folder: {out_root}")
    _ensure_dir(out_root)

    source_index_path = Path(args.source_index).expanduser().resolve()
    models_file = Path(args.models_file).expanduser().resolve()
    scale_index_path = Path(args.scale_index).expanduser().resolve()
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
    scale_recommendations = _load_scale_recommendations(scale_index_path)
    missing_scales = [s.model_name for s in specs if s.model_name not in scale_recommendations]
    if missing_scales:
        raise ValueError(f"Missing scale recommendations for: {missing_scales}")
    specs_by_name = {s.model_name: s for s in specs}
    for override in args.scale_override:
        if "=" not in override:
            raise ValueError(f"Expected --scale_override model_name=multiplier, got: {override}")
        model_name, multiplier_text = override.split("=", 1)
        if model_name not in specs_by_name:
            raise ValueError(f"Unknown --scale_override model_name: {model_name}")
        multiplier = float(multiplier_text)
        spec = specs_by_name[model_name]
        scale_recommendations[model_name] = {
            "scale_abs": float(spec.base_scale_tdw * multiplier),
            "scale_multiplier_from_base": multiplier,
        }

    semantic_lock = {
        "must_be_true": [
            "Full dataset uses TDW rerendering, not PIL rotation.",
            "Operator is base pose, then rotate_object_by(axis='roll', is_world=True, use_centroid=True).",
            "Each frame is reset to base pose before applying the absolute theta.",
            "World lighting remains fixed.",
            "Manual TDW _id capture is saved along with binary masks.",
            "Scale values are loaded from the approved world-Z-roll scale probe.",
            "Cyclic pair indices are included because 358 + 2 wraps to 0.",
        ],
        "must_not_happen": [
            "Do not use base_2d_scale as a TDW scale.",
            "Do not chain incremental 2-degree rotations.",
            "Do not rotate lights with the object.",
            "Do not use post-render 2D recentering/cropping.",
        ],
        "source_index": str(source_index_path),
        "source_base_panel": str(Path(str(source_index["base_panel_path"]))),
        "scale_index": str(scale_index_path),
        "operator_reference_image": str(
            Path("_sanity_hammer_tdw_world_axes_yaw_pitch_roll_512/hammer_three_operators_rgb_contact_sheet.png").resolve()
        ),
        "scale_sanity_image": str(
            Path("_sanity_tdw_world_z_roll_scale_probe_512_v2/proposed_scale_rgb_contact_sheet.png").resolve()
        ),
    }
    _write_json(out_root / "semantic_lock.json", semantic_lock)
    _write_json(
        out_root / "operator_reference.json",
        {
            "current_operator": OPERATOR_NAME,
            "current_operator_command": {
                "reset_to_base_pose": True,
                "rotate_object_by": {
                    "axis": "roll",
                    "is_world": True,
                    "use_centroid": True,
                    "angle": "theta_deg",
                },
            },
            "future_supported_operators": {
                "tdw_world_y_yaw_by_axis": {"axis": "yaw", "is_world": True, "use_centroid": True},
                "tdw_world_x_pitch": {"axis": "pitch", "is_world": True, "use_centroid": True},
            },
        },
    )
    _write_json(out_root / "scale_summary.json", scale_recommendations)

    controller = _make_controller(port=int(args.port), launch_build=bool(args.launch_build))
    try:
        object_summaries = {}
        sample_rgb_items: List[Tuple[Image.Image, str]] = []
        sample_mask_items: List[Tuple[Image.Image, str]] = []
        for spec in specs:
            scale_abs = float(scale_recommendations[spec.model_name]["scale_abs"])
            rendered = _render_object_full(
                controller,
                spec=spec,
                cfg=cfg,
                scale_abs=scale_abs,
                out_root=out_root,
            )
            object_summaries[spec.model_name] = rendered["summary"]
            sample_rgb_items.extend(rendered["sample_rgb_items"])
            sample_mask_items.extend(rendered["sample_mask_items"])

        rgb_sheet = out_root / "sample_rgb_contact_sheet.png"
        mask_sheet = out_root / "sample_mask_contact_sheet.png"
        _contact_sheet(sample_rgb_items, cols=len(SAMPLE_THETAS)).save(rgb_sheet)
        _contact_sheet(sample_mask_items, cols=len(SAMPLE_THETAS)).save(mask_sheet)

        index_info = _write_indices(out_root, specs)
        dataset_index = {
            "dataset_name": out_root.name,
            "operator_name": OPERATOR_NAME,
            "out_root": str(out_root),
            "img_size": cfg.img_size,
            "theta_range": "0..358",
            "theta_step_deg": 2.0,
            "frames_per_object": len(FULL_THETAS),
            "n_objects": len(specs),
            "total_rgb_frames": len(specs) * len(FULL_THETAS),
            "total_id_frames": len(specs) * len(FULL_THETAS),
            "total_mask_frames": len(specs) * len(FULL_THETAS),
            "camera": {
                "object_base_y": cfg.object_base_y,
                "camera_height": cfg.camera_height,
                "look_at_height": cfg.look_at_height,
                "camera_position_rule": "(0, object_base_y + camera_height, camera_dist)",
                "look_at_rule": "(0, object_base_y + look_at_height, 0)",
            },
            "uniform_bg_rgb": list(cfg.uniform_bg_rgb),
            "lighting": "fixed_world_lighting",
            "objects": [
                {
                    "model_name": spec.model_name,
                    "source_split": spec.split,
                    "object_dir": str(out_root / "train" / spec.model_name),
                    "base_pitch_deg": spec.base_pitch_deg,
                    "base_yaw_deg": spec.base_yaw_deg,
                    "base_roll_deg": spec.base_roll_deg,
                    "source_base_scale_tdw": spec.base_scale_tdw,
                    "scale_abs": scale_recommendations[spec.model_name]["scale_abs"],
                    "scale_multiplier_from_base": scale_recommendations[spec.model_name]["scale_multiplier_from_base"],
                    "camera_dist": spec.camera_dist,
                    "summary": {
                        k: v
                        for k, v in object_summaries[spec.model_name].items()
                        if k != "records"
                    },
                }
                for spec in specs
            ],
            "indices": index_info,
            "sample_rgb_contact_sheet": str(rgb_sheet),
            "sample_mask_contact_sheet": str(mask_sheet),
        }
        _write_json(out_root / "dataset_index.json", dataset_index)
        print(json.dumps({"out_root": str(out_root), "dataset_index": str(out_root / "dataset_index.json")}, indent=2))
    finally:
        controller.communicate([{"$type": "terminate"}])


if __name__ == "__main__":
    main()
