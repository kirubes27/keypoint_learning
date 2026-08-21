"""Generate the four next datasets (yaw, pitch, scale, translation) per
docs/PLAN_NEXT_DATASETS_YAW_PITCH_SCALE_TRANSLATION_2026-07-10.md.

LIVES IN THE PROJECT (recovered_sources/verified_generator/) deliberately —
an earlier copy lived in the session scratchpad and was lost when the session
restarted mid-generation (2026-07-10). Keep it here.

Modes:
  --gate-only   Render ONLY endpoints/corners for ALL 6 objects and evaluate
                every gate. No dataset files written. Exit code 1 on any failure
                (stop-rule: reject the spec, never drop frames silently).
  (default)     Full generation of all four datasets. Every frame is gate-checked
                during rendering; a violation aborts the dataset with a report.

Gate rules (user decision 2026-07-10):
- yaw/pitch: REAL-FRAME rule — the silhouette must never touch the physical image
  edge (margin >= 2px); the roll-era 0.82-bbox/0.05-margin policy does NOT apply.
  Plus visibility: mask area >= 0.60 * base-pose area.
- scale/translation: inherit the roll policy (margin >= 0.05, bbox <= 0.82).

Zoom rule (user decision 2026-07-10): yaw/pitch render at each object's own
approved roll scale S_o EXCEPT the toy monkey at 0.96*S_o — the only object whose
silhouette physically clips the frame inside +/-60 deg (verified by render, see
docs/probes_yaw_pitch_2026-07-10/).

Semantic locks (mirroring the verified roll generator):
- absolute reset to base pose before every frame (no chained increments);
- rotations: rotate_object_by(theta, axis, is_world=True, use_centroid=True);
- scale: absolute scale s*S_o, bounds-recentred to (0, y0, 0); geometric ladder;
- translation: fixed scale 0.60*S_o, bounds-centre placed at (x, y0+y, 0);
- TDW re-render for every frame; no post-render 2D tricks; fixed lighting/bg.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

VG = Path(__file__).resolve().parent
sys.path.insert(0, str(VG))

SRC_DS = Path(
    "/Users/kirubeso.r/Documents/PhD/data/active/_tdw_world_z_roll_base_panel_512_v2"
)
OUT_BASE = Path("/Users/kirubeso.r/Documents/PhD/data/active")

# ---- design constants (from the reconciled plan) ----
ARC_DEG = 60.0
ARC_STEP = 1.0
ROT_SKIPS = [2, 6, 10]
# Per-object yaw/pitch zoom factors on top of approved roll scale S_o.
# Real-frame rule: only the monkey physically clips the frame at full scale
# inside +/-60 deg (arm exits the left edge at yaw -60); everyone else fits.
ROT_RENDER_SCALE = {
    "engineers_hammer_vray": 1.0,
    "b03_banana_01_high": 1.0,
    "kettle": 1.0,
    "dewalt_compact_drill_vray": 1.0,
    "toy_monkey_medium": 0.96,
    "b03_trumpet_vray": 1.0,
}
SCALE_MIN = 0.5
SCALE_N = 121  # i=0..120
SCALE_R = 2 ** (1.0 / 120.0)
SCALE_SKIPS = [4, 12, 20]
EVAL_N = 31  # j=0..30, s = 0.5 * r^(4j+1/2)
T_SCALE = 0.60
T_EXTENT = 0.08
T_GRID = 11
T_SKIPS = [1, 3, 5]
GATE_MARGIN = 0.05
GATE_BBOX = 0.82
GATE_EDGE = 2.0 / 512  # real-frame rule for rotations
GATE_AREA_RATIO = 0.60  # rotations, vs base pose
GATE_MIN_AREA = 0.015  # scale floor

DATASET_ROOTS = {
    "yaw": "_tdw_world_y_yaw_arc60_step1_base_panel_512_v1",
    "pitch": "_tdw_world_x_pitch_arc60_step1_base_panel_512_v1",
    "scale": "_tdw_uniform_scale_loghalf_to_one_base_panel_512_v1",
    "translation": "_tdw_camera_plane_xy_grid11_scale060_base_panel_512_v1",
}
ROT_OPS = {"yaw": "tdw_world_y_yaw_by_axis", "pitch": "tdw_world_x_pitch"}


def t_offsets() -> List[float]:
    step = 2 * T_EXTENT / (T_GRID - 1)
    return [round(-T_EXTENT + k * step, 6) for k in range(T_GRID)]


def build_specs_and_cfg():
    from create_true_yaw_from_base_panel_sanity import ObjectSpec, RenderConfig

    idx = json.loads((SRC_DS / "dataset_index.json").read_text())
    cfg = RenderConfig(
        img_size=int(idx["img_size"]),
        object_base_y=float(idx["camera"]["object_base_y"]),
        camera_height=float(idx["camera"]["camera_height"]),
        look_at_height=float(idx["camera"]["look_at_height"]),
        uniform_bg_rgb=tuple(int(v) for v in idx["uniform_bg_rgb"]),
        min_margin=GATE_MARGIN,
        max_size=GATE_BBOX,
        center_tolerance_world=0.005,
    )
    specs = []
    for rec in idx["objects"]:
        specs.append(
            (
                ObjectSpec(
                    model_name=rec["model_name"],
                    split=rec["source_split"],
                    source_object_dir=Path(rec["object_dir"]),
                    base_pitch_deg=float(rec["base_pitch_deg"]),
                    base_yaw_deg=float(rec["base_yaw_deg"]),
                    base_roll_deg=float(rec["base_roll_deg"]),
                    base_scale_tdw=float(rec["source_base_scale_tdw"]),
                    camera_dist=float(rec["camera_dist"]),
                    source_base_preview=Path("/nonexistent"),
                ),
                float(rec["scale_abs"]),
            )
        )
    return specs, cfg


def recenter_to(controller, obj_id, target_xyz, tolerance=1e-4, max_iters=6):
    """Place the object's BOUNDS CENTRE at target (generalized _recenter_object).

    scale_object/teleport_object act on the PIVOT, whose offset from the visual
    centre is object-dependent (kettle!) — bounds-centring is what the original
    roll dataset used (recorded max_center_error_world ~ 8e-08)."""
    from create_dataset_affine_xy_scale_rotation import _get_bounds
    from tdw.output_data import OutputData, Transforms

    target = np.array(target_xyz, dtype=float)
    err = float("inf")
    for _ in range(max_iters):
        delta = target - _get_bounds(controller, obj_id)["center"]
        err = float(np.linalg.norm(delta))
        if err <= tolerance:
            break
        resp = controller.communicate(
            [{"$type": "send_transforms", "ids": [obj_id], "frequency": "once"}]
        )
        pivot = None
        for r in resp:
            if OutputData.get_data_type_id(r) == "tran":
                t = Transforms(r)
                for i in range(t.get_num()):
                    if t.get_id(i) == obj_id:
                        pivot = np.array(t.get_position(i), dtype=float)
        if pivot is None:
            raise RuntimeError("no transform for object")
        p = pivot + delta
        controller.communicate(
            [{"$type": "teleport_object", "id": obj_id,
              "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}}]
        )
    return err


class ObjectSession:
    """One object loaded in the scene, with absolute-scale + pose-reset helpers."""

    def __init__(self, controller, spec, S_o, cfg, capture_dir: Path):
        from create_true_yaw_from_base_panel_sanity import _setup_scene

        self.controller = controller
        self.spec = spec
        self.S_o = float(S_o)
        self.cfg = cfg
        self.obj_id, self.cap = _setup_scene(controller, spec, capture_dir, cfg)
        self.current_abs = 1.0  # _setup_scene leaves object at TDW absolute scale 1.0

    def set_abs_scale(self, target_abs: float):
        factor = target_abs / self.current_abs
        self.controller.communicate(
            [{"$type": "scale_object", "id": self.obj_id,
              "scale_factor": {"x": factor, "y": factor, "z": factor}}]
        )
        self.current_abs = target_abs

    def reset_pose(self):
        self.controller.communicate(
            [{
                "$type": "rotate_object_to_euler_angles",
                "id": self.obj_id,
                "euler_angles": {
                    "x": self.spec.base_pitch_deg,
                    "y": self.spec.base_yaw_deg,
                    "z": self.spec.base_roll_deg,
                },
            }]
        )

    def capture(self):
        from create_true_yaw_from_base_panel_sanity import (
            _apply_uniform_background, _capture_pil, _id_mask, _mask_stats,
        )

        rgb, idp = _capture_pil(controller=self.controller, cap=self.cap, commands=[])
        mask = _id_mask(idp)
        st = _mask_stats(mask)
        out = _apply_uniform_background(rgb, mask, self.cfg.uniform_bg_rgb)
        return out, idp, mask, st


def check_gates(st: Dict, *, base_area: Optional[float], dataset: str) -> Optional[str]:
    if st.get("area_frac", 0.0) <= 0.0:
        return "empty mask"
    if dataset in ("yaw", "pitch"):
        # user rule 2026-07-10: rotations must only stay inside the PHYSICAL frame
        if st["margin_frac"] < GATE_EDGE:
            return f"touches frame edge (margin {st['margin_frac']:.4f})"
    else:
        if st["margin_frac"] < GATE_MARGIN:
            return f"margin {st['margin_frac']:.3f} < {GATE_MARGIN}"
        if st["bbox_max_size_frac"] > GATE_BBOX:
            return f"bbox {st['bbox_max_size_frac']:.3f} > {GATE_BBOX}"
    if dataset in ("yaw", "pitch") and base_area:
        ratio = st["area_frac"] / base_area
        if ratio < GATE_AREA_RATIO:
            return f"area ratio {ratio:.3f} < {GATE_AREA_RATIO}"
    if dataset == "scale" and st["area_frac"] < GATE_MIN_AREA:
        return f"area {st['area_frac']:.4f} < {GATE_MIN_AREA}"
    return None


# ---------- frame value schedules ----------

def rot_thetas() -> List[float]:
    n = int(round(2 * ARC_DEG / ARC_STEP)) + 1
    return [-ARC_DEG + i * ARC_STEP for i in range(n)]


def scale_ladder() -> List[float]:
    return [SCALE_MIN * (SCALE_R ** i) for i in range(SCALE_N)]


def eval_ladder() -> List[float]:
    return [SCALE_MIN * (SCALE_R ** (4 * j + 0.5)) for j in range(EVAL_N)]


# ---------- rendering per dataset ----------

def render_rotation(controller, specs, cfg, axis: str, out_root: Path, writer) -> None:
    from compare_hammer_rotation_operators import _apply_tdw_operator

    op = ROT_OPS[axis]
    thetas = rot_thetas()
    for spec, S_o in specs:
        sess = ObjectSession(controller, spec, S_o, cfg, out_root / "train" / spec.model_name)
        current = 1.0
        base_area = None
        render_scale = ROT_RENDER_SCALE[spec.model_name]
        for pass_thetas in ([0.0], thetas):
            for theta in pass_thetas:
                current, cerr, _ = _apply_tdw_operator(
                    controller, object_id=sess.obj_id, spec=spec, theta_deg=float(theta),
                    operator_name=op, scale_abs=render_scale * S_o,
                    current_scale_abs=current, cfg=cfg,
                )
                out, idp, mask, st = sess.capture()
                if base_area is None:
                    base_area = st["area_frac"]
                    continue  # reference render, not stored
                fail = check_gates(st, base_area=base_area, dataset=axis)
                writer(spec, dict(theta_deg=theta, operator_name=op,
                                  scale_abs=render_scale * S_o,
                                  render_scale_rel=render_scale,
                                  center_error_world=cerr, base_area_frac=base_area, **st),
                       out, idp, mask, fail)


def render_scale(controller, specs, cfg, out_root: Path, writer) -> None:
    y0 = cfg.object_base_y
    for spec, S_o in specs:
        sess = ObjectSession(controller, spec, S_o, cfg, out_root / "train" / spec.model_name)
        for split_name, ladder in [("train", scale_ladder()), ("eval_abs_holdout", eval_ladder())]:
            for i, s_rel in enumerate(ladder):
                sess.reset_pose()
                sess.set_abs_scale(s_rel * S_o)
                cerr = recenter_to(controller, sess.obj_id, (0.0, y0, 0.0))
                out, idp, mask, st = sess.capture()
                fail = check_gates(st, base_area=None, dataset="scale")
                writer(spec, dict(s_rel=s_rel, scale_abs=s_rel * S_o, ladder=split_name,
                                  ladder_index=i, center_error_world=cerr,
                                  operator_name="tdw_uniform_scale", **st),
                       out, idp, mask, fail, split=split_name)


def render_translation(controller, specs, cfg, out_root: Path, writer) -> None:
    y0 = cfg.object_base_y
    offs = t_offsets()
    for spec, S_o in specs:
        sess = ObjectSession(controller, spec, S_o, cfg, out_root / "train" / spec.model_name)
        sess.set_abs_scale(T_SCALE * S_o)
        for gy, dy in enumerate(offs):
            for gx, dx in enumerate(offs):
                sess.reset_pose()
                cerr = recenter_to(controller, sess.obj_id, (dx, y0 + dy, 0.0))
                out, idp, mask, st = sess.capture()
                fail = check_gates(st, base_area=None, dataset="translation")
                writer(spec, dict(grid_x=gx, grid_y=gy, dx_world=dx, dy_world=dy,
                                  scale_rel=T_SCALE, scale_abs=T_SCALE * S_o,
                                  center_error_world=cerr,
                                  operator_name="tdw_camera_plane_translation", **st),
                       out, idp, mask, fail)


# ---------- index writers ----------

def rot_indices(out_root: Path, model_names: List[str]) -> None:
    thetas = rot_thetas()
    n = len(thetas)
    idx_dir = out_root / "indices"
    idx_dir.mkdir(exist_ok=True)
    for skip in ROT_SKIPS:
        for reverse in (False, True):
            pairs = []
            for m in model_names:
                for i in range(n - skip):
                    a, b = (i + skip, i) if reverse else (i, i + skip)
                    pairs.append({
                        "model_name": m, "skip_frames": skip,
                        "delta_theta_deg": (-ARC_STEP if reverse else ARC_STEP) * skip,
                        "src_frame_index": a, "dst_frame_index": b,
                        "src_theta_deg": thetas[a], "dst_theta_deg": thetas[b],
                        "src_image_relpath": f"train/{m}/frames/a/img_{a:04d}.png",
                        "dst_image_relpath": f"train/{m}/frames/a/img_{b:04d}.png",
                        "src_mask_relpath": f"train/{m}/masks/a/mask_{a:04d}.png",
                        "dst_mask_relpath": f"train/{m}/masks/a/mask_{b:04d}.png",
                    })
            tag = "reverse" if reverse else "forward"
            (idx_dir / f"pairs_skip{skip}_{tag}_nocycle.json").write_text(json.dumps(
                {"skip_frames": skip, "delta_theta_deg": (-ARC_STEP if reverse else ARC_STEP) * skip,
                 "cyclic": False, "direction": tag, "pairs": pairs}, indent=1))


def scale_indices(out_root: Path, model_names: List[str]) -> None:
    ladder = scale_ladder()
    n = len(ladder)
    idx_dir = out_root / "indices"
    idx_dir.mkdir(exist_ok=True)
    for skip in SCALE_SKIPS:
        for reverse in (False, True):
            ratio = SCALE_R ** (-skip if reverse else skip)
            pairs = []
            for m in model_names:
                for i in range(n - skip):
                    a, b = (i + skip, i) if reverse else (i, i + skip)
                    pairs.append({
                        "model_name": m, "skip_frames": skip, "scale_ratio": ratio,
                        "delta_log_scale": math.log(ratio),
                        "src_frame_index": a, "dst_frame_index": b,
                        "src_s_rel": ladder[a], "dst_s_rel": ladder[b],
                        "src_image_relpath": f"train/{m}/frames/a/img_{a:04d}.png",
                        "dst_image_relpath": f"train/{m}/frames/a/img_{b:04d}.png",
                        "src_mask_relpath": f"train/{m}/masks/a/mask_{a:04d}.png",
                        "dst_mask_relpath": f"train/{m}/masks/a/mask_{b:04d}.png",
                    })
            tag = "reverse" if reverse else "forward"
            (idx_dir / f"pairs_skip{skip}_{tag}_nocycle.json").write_text(json.dumps(
                {"skip_frames": skip, "scale_ratio": ratio, "cyclic": False,
                 "direction": tag, "pairs": pairs}, indent=1))
    # eval ladder: skip-1 pairs have ratio r^4 == training skip-4 group element
    ev = eval_ladder()
    pairs = []
    for m in model_names:
        for j in range(len(ev) - 1):
            pairs.append({
                "model_name": m, "skip_frames": 1, "scale_ratio": SCALE_R ** 4,
                "src_frame_index": j, "dst_frame_index": j + 1,
                "src_s_rel": ev[j], "dst_s_rel": ev[j + 1],
                "src_image_relpath": f"eval_abs_holdout/{m}/frames/a/img_{j:04d}.png",
                "dst_image_relpath": f"eval_abs_holdout/{m}/frames/a/img_{j+1:04d}.png",
                "src_mask_relpath": f"eval_abs_holdout/{m}/masks/a/mask_{j:04d}.png",
                "dst_mask_relpath": f"eval_abs_holdout/{m}/masks/a/mask_{j+1:04d}.png",
            })
    (idx_dir / "pairs_eval_abs_holdout_ratio_r4.json").write_text(json.dumps(
        {"scale_ratio": SCALE_R ** 4, "note": "held-out absolute sizes, training skip-4 group element",
         "pairs": pairs}, indent=1))


def translation_indices(out_root: Path, model_names: List[str]) -> None:
    offs = t_offsets()
    n = T_GRID
    frame_idx = lambda gx, gy: gy * n + gx
    idx_dir = out_root / "indices"
    idx_dir.mkdir(exist_ok=True)
    step = offs[1] - offs[0]
    for direction in ("x", "y"):
        for q in T_SKIPS:
            for reverse in (False, True):
                pairs = []
                d_world = (-step * q) if reverse else (step * q)
                for m in model_names:
                    for gy in range(n):
                        for gx in range(n):
                            if direction == "x":
                                if gx + q >= n:
                                    continue
                                a, b = frame_idx(gx, gy), frame_idx(gx + q, gy)
                            else:
                                if gy + q >= n:
                                    continue
                                a, b = frame_idx(gx, gy), frame_idx(gx, gy + q)
                            if reverse:
                                a, b = b, a
                            pairs.append({
                                "model_name": m, "direction": direction, "skip_steps": q,
                                "delta_world": d_world,
                                "src_frame_index": a, "dst_frame_index": b,
                                "src_image_relpath": f"train/{m}/frames/a/img_{a:04d}.png",
                                "dst_image_relpath": f"train/{m}/frames/a/img_{b:04d}.png",
                                "src_mask_relpath": f"train/{m}/masks/a/mask_{a:04d}.png",
                                "dst_mask_relpath": f"train/{m}/masks/a/mask_{b:04d}.png",
                            })
                tag = "reverse" if reverse else "forward"
                (idx_dir / f"pairs_d{direction}{q}_{tag}_nocycle.json").write_text(json.dumps(
                    {"direction": direction, "skip_steps": q, "delta_world": d_world,
                     "grid": n, "world_step": step, "cyclic": False, "pairs": pairs}, indent=1))


# ---------- output plumbing ----------

class DatasetWriter:
    def __init__(self, out_root: Path, dataset: str):
        self.root = out_root
        self.dataset = dataset
        self.counters: Dict[Tuple[str, str], int] = {}
        self.failures: List[Dict] = []
        self.sheet_items = []

    def __call__(self, spec, meta: Dict, rgb, idp, mask_bool, fail: Optional[str], split="train"):
        from create_true_yaw_from_base_panel_sanity import _append_jsonl, _binary_mask_image

        m = spec.model_name
        k = (split, m)
        i = self.counters.get(k, 0)
        self.counters[k] = i + 1
        obj = self.root / split / m
        for sub in ["frames/a", "masks/a", "id_passes/a"]:
            (obj / sub).mkdir(parents=True, exist_ok=True)
        rgb.save(obj / f"frames/a/img_{i:04d}.png")
        _binary_mask_image(mask_bool).save(obj / f"masks/a/mask_{i:04d}.png")
        idp.save(obj / f"id_passes/a/id_{i:04d}.png")
        row = {"frame_index": i, "model_name": m, "split": split,
               "gate_failure": fail, **meta}
        _append_jsonl(obj / "meta.jsonl", row)
        if fail:
            self.failures.append(row)
        if i % max(1, (121 // 6)) == 0:
            self.sheet_items.append((rgb, f"{m[:12]}\n#{i}"))


def run_full(specs, cfg, only: Optional[List[str]] = None):
    from create_true_yaw_from_base_panel_sanity import _contact_sheet, _make_controller, _write_json

    model_names = [s.model_name for s, _ in specs]
    controller = _make_controller(port=1096, launch_build=True)
    summary = {}
    try:
        for dataset in (only or ["yaw", "pitch", "scale", "translation"]):
            out_root = OUT_BASE / DATASET_ROOTS[dataset]
            if out_root.exists() and any(out_root.iterdir()):
                raise RuntimeError(f"refusing to write into nonempty {out_root}")
            out_root.mkdir(exist_ok=True)
            w = DatasetWriter(out_root, dataset)
            print(f"[dataset] {dataset} -> {out_root}")
            if dataset in ("yaw", "pitch"):
                render_rotation(controller, specs, cfg, dataset, out_root, w)
                rot_indices(out_root, model_names)
            elif dataset == "scale":
                render_scale(controller, specs, cfg, out_root, w)
                scale_indices(out_root, model_names)
            else:
                render_translation(controller, specs, cfg, out_root, w)
                translation_indices(out_root, model_names)
            _contact_sheet(w.sheet_items, cols=7).save(out_root / "sample_rgb_contact_sheet.png")
            lock = {
                "dataset": dataset,
                "plan": "docs/PLAN_NEXT_DATASETS_YAW_PITCH_SCALE_TRANSLATION_2026-07-10.md",
                "must_be_true": [
                    "TDW rerender per frame; absolute reset to base pose before every frame.",
                    "Gates: rotations = real-frame rule (no physical edge contact, margin>=2px)"
                    " + area>=0.60*base; scale/translation = margin>=0.05, bbox<=0.82.",
                    "Bounds-centre (not pivot) is the positional reference.",
                    "One pair file = one group element; forward and reverse are separate files.",
                    "Yaw/pitch zoom: approved roll scale per object; toy_monkey_medium at 0.96.",
                ],
                "generator": "/Users/kirubeso.r/Documents/PhD/code/keypoint-learning/tools/verified_tdw_generator/generate_next_datasets.py",
                "rot_render_scale": ROT_RENDER_SCALE,
                "n_gate_failures": len(w.failures),
            }
            _write_json(out_root / "semantic_lock.json", lock)
            _write_json(out_root / "gate_failures.json", w.failures)
            n_frames = sum(v for (sp, m), v in w.counters.items())
            summary[dataset] = {"frames": n_frames, "gate_failures": len(w.failures)}
            print(f"[done] {dataset}: {n_frames} frames, {len(w.failures)} gate failures")
            if w.failures:
                print(f"[STOP] gate failures in {dataset} — see {out_root/'gate_failures.json'}")
                break
    finally:
        controller.communicate([{"$type": "terminate"}])
    print(json.dumps(summary, indent=1))


def run_gate(specs, cfg):
    """Endpoints/corners for ALL objects; no dataset files written."""
    from create_true_yaw_from_base_panel_sanity import _contact_sheet, _make_controller
    from compare_hammer_rotation_operators import _apply_tdw_operator

    out = Path("/private/tmp/claude-501/-Users-kirubeso-r-Documents-PhD/f93714c7-69bb-44c1-975c-72a6b753f580/scratchpad/_gate_renders")
    out.mkdir(parents=True, exist_ok=True)
    controller = _make_controller(port=1096, launch_build=True)
    rows, items = [], []
    ok = True
    y0 = cfg.object_base_y
    try:
        for spec, S_o in specs:
            sess = ObjectSession(controller, spec, S_o, cfg, out / spec.model_name)
            render_scale = ROT_RENDER_SCALE[spec.model_name]
            for axis, op in ROT_OPS.items():
                current = sess.current_abs
                base_area = None
                for theta in [0.0, -ARC_DEG, ARC_DEG]:
                    current, cerr, _ = _apply_tdw_operator(
                        controller, object_id=sess.obj_id, spec=spec, theta_deg=theta,
                        operator_name=op, scale_abs=render_scale * S_o,
                        current_scale_abs=current, cfg=cfg)
                    outim, _, _, st = sess.capture()
                    if theta == 0.0:
                        base_area = st["area_frac"]
                        continue
                    fail = check_gates(st, base_area=base_area, dataset=axis)
                    rows.append({"object": spec.model_name, "check": f"{axis} {theta:+g}", "fail": fail, **st})
                    items.append((outim, f"{spec.model_name[:10]}\n{axis}{theta:+g} x{render_scale:g}"))
                    ok &= fail is None
                sess.current_abs = current
            for s in [SCALE_MIN, 1.0]:
                sess.reset_pose()
                sess.set_abs_scale(s * S_o)
                recenter_to(controller, sess.obj_id, (0.0, y0, 0.0))
                outim, _, _, st = sess.capture()
                fail = check_gates(st, base_area=None, dataset="scale")
                rows.append({"object": spec.model_name, "check": f"scale {s:g}", "fail": fail, **st})
                items.append((outim, f"{spec.model_name[:12]}\ns={s:g}"))
                ok &= fail is None
            sess.set_abs_scale(T_SCALE * S_o)
            for dx in (-T_EXTENT, T_EXTENT):
                for dy in (-T_EXTENT, T_EXTENT):
                    sess.reset_pose()
                    recenter_to(controller, sess.obj_id, (dx, y0 + dy, 0.0))
                    outim, _, _, st = sess.capture()
                    fail = check_gates(st, base_area=None, dataset="translation")
                    rows.append({"object": spec.model_name,
                                 "check": f"corner ({dx:+g},{dy:+g})", "fail": fail, **st})
                    items.append((outim, f"{spec.model_name[:10]}\n({dx:+g},{dy:+g})"))
                    ok &= fail is None
    finally:
        controller.communicate([{"$type": "terminate"}])

    (out / "gate_report.json").write_text(json.dumps(rows, indent=1))
    _contact_sheet(items, cols=10).save(out / "gate_sheet.png")
    print(f"{'PASS' if ok else 'FAIL'}: {sum(1 for r in rows if not r['fail'])}/{len(rows)} checks ok")
    for r in rows:
        if r["fail"]:
            print(f"  FAIL {r['object']} {r['check']}: {r['fail']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(DATASET_ROOTS))
    args = ap.parse_args()
    specs, cfg = build_specs_and_cfg()
    if args.gate_only:
        run_gate(specs, cfg)
    else:
        run_full(specs, cfg, only=args.only)
