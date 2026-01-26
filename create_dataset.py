"""TDW pose dataset generator (fine angular steps, Stage-A friendly).

This is a *drop-in replacement* for the original create_dataset.py you shared.
Your original script is great for recreating the older Joseph-style grid dataset,
but it has a few issues for *your* Stage-A goal:

1) It renders MANY objects in one scene (4x3 grid) -> background distractors.
2) It uses 10° steps -> too coarse for compositional/locally-linear operator fits.
3) It mixes train/test angle sets but not as temporal sequences.
4) A couple bugs/rough edges (typos like `wseaaaaaaaaaangles`, hard-coded paths).

This script:
- renders ONE object per scene (clean background).
- uses 1° steps by default.
- writes ordered frames + metadata (so training can use (t,t+1,t+2) triples).
- deterministic train/test split by object identity.
- supports either (A) camera orbit around a fixed object OR (B) object rotation
  under a fixed camera (you can choose with --mode).

Usage example (yaw-only, 1° step):
  python create_dataset.py \
      --out_dir ./tdw_pose_1deg \
      --models_file ./models_12.txt \
      --mode orbit_camera \
      --yaw_min -90 --yaw_max 90 --yaw_step 1 \
      --pitch_list 0 \
      --dist 1.5 \
      --img_size 128

A models_file is a newline-separated list of TDW model names (e.g. 'iron_box').
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# TDW imports (must be installed in your env)
from tdw.controller import Controller
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.tdw_utils import TDWUtils


@dataclass(frozen=True)
class Pose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0


def _read_models(models_file: Path) -> List[str]:
    lines = [ln.strip() for ln in models_file.read_text().splitlines()]
    models = [ln for ln in lines if ln and not ln.startswith("#")]
    if not models:
        raise ValueError(f"No models found in {models_file}")
    return models


def _hash_split(name: str, test_frac: float) -> str:
    """Deterministic split by object name."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    # map first 8 hex digits to [0,1)
    u = int(h[:8], 16) / float(16**8)
    return "test" if u < test_frac else "train"


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _camera_pos_from_yaw_pitch(dist: float, yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """Spherical coordinates around the origin.

    Convention:
      - yaw rotates around +Y (azimuth)
      - pitch is elevation (0 = horizon, + up)

    Returns (x,y,z) camera position.
    """
    yaw = _deg2rad(yaw_deg)
    pitch = _deg2rad(pitch_deg)
    x = dist * math.cos(pitch) * math.sin(yaw)
    z = dist * math.cos(pitch) * math.cos(yaw)
    y = dist * math.sin(pitch)
    return (x, y, z)


def _make_pose_grid(yaw_min: float, yaw_max: float, yaw_step: float, pitch_list: List[float], roll: float = 0.0) -> List[Pose]:
    if yaw_step <= 0:
        raise ValueError("yaw_step must be > 0")
    poses: List[Pose] = []
    yaw = yaw_min
    # include yaw_max (within floating error)
    while yaw <= yaw_max + 1e-9:
        for p in pitch_list:
            poses.append(Pose(yaw_deg=float(yaw), pitch_deg=float(p), roll_deg=float(roll)))
        yaw += yaw_step
    return poses


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def render_object_sequence(
    *,
    model_name: str,
    out_dir: Path,
    poses: List[Pose],
    img_size: int,
    dist: float,
    mode: str,
    seed: int,
    port: int,
    camera_height: float,
    look_at_height: float,
) -> Dict:
    """Render a sequence for one object. Returns a summary dict."""

    _ensure_dir(out_dir)
    frames_dir = out_dir / "frames"
    _ensure_dir(frames_dir)

    meta_jsonl = out_dir / "meta.jsonl"
    if meta_jsonl.exists():
        meta_jsonl.unlink()  # avoid accidental append from prior runs

    # TDW controller
    c = Controller(port=int(port))

    # Camera + capture. TDW ImageCapture writes images named by frame count.
    # We keep a single avatar id and save only RGB.
    cam = ThirdPersonCamera(avatar_id="a", position={"x": 0, "y": 0, "z": 0}, look_at={"x": 0, "y": 0, "z": 0})
    # Use lossless PNG for Phase-A (avoid compression artifacts in predictability loss).
    cap = ImageCapture(avatar_ids=["a"], path=str(frames_dir.resolve()), png=True, pass_masks=["_img"])
    c.add_ons.extend([cam, cap])

    # IMPORTANT: ImageCapture always requests one image on initialization.
    # We disable saving until we start the pose loop so that:
    # - warmup/scene setup doesn't create a file
    # - the first saved frame is img_0000.png and corresponds to t=0
    cap.set(frequency="never", avatar_ids=["a"], save=False)

    # clean empty room
    commands = [
        TDWUtils.create_empty_room(12, 12),
        # Disable post-processing to avoid depth-of-field blur.
        {"$type": "set_post_process", "value": False},
        {"$type": "set_screen_size", "width": img_size, "height": img_size},
        {"$type": "set_target_framerate", "framerate": 30},
        {"$type": "set_render_quality", "render_quality": 5},
    ]

    # add object at origin
    obj_id = c.get_unique_id()
    commands.append(
        c.get_add_object(model_name=model_name, object_id=obj_id, position={"x": 0, "y": 0, "z": 0})
    )

    # warmup 1 frame
    c.communicate(commands)

    # fixed camera if rotating object
    if mode == "rotate_object":
        # place camera at yaw=0,pitch=0 by default
        x, y, z = _camera_pos_from_yaw_pitch(dist=dist, yaw_deg=0.0, pitch_deg=0.0)
        cam.teleport({"x": x, "y": float(camera_height), "z": z})
        cam.look_at({"x": 0, "y": float(look_at_height), "z": 0})
        # Apply camera placement, but still don't save.
        c.communicate([])

    # render poses
    for t, pose in enumerate(poses):
        # Request exactly one image for this pose.
        # ImageCapture filenames are based on its internal frame counter (0000, 0001, ...).
        image_index = cap.frame
        image_relpath = f"frames/a/img_{image_index:04d}.png"
        cap.set(frequency="once", avatar_ids=["a"], save=True)

        cmds: List[dict] = []
        if mode == "orbit_camera":
            x, y, z = _camera_pos_from_yaw_pitch(dist=dist, yaw_deg=pose.yaw_deg, pitch_deg=pose.pitch_deg)
            cam.teleport({"x": x, "y": float(y) + float(camera_height), "z": z})
            cam.look_at({"x": 0, "y": float(look_at_height), "z": 0})
            # Apply camera move and capture on this same frame.
        elif mode == "rotate_object":
            # rotate object around Y (yaw) and X (pitch) in-place.
            # TDW uses Euler angles in degrees for 'rotation'.
            cmds.append(
                {"$type": "rotate_object_to_euler_angles", "id": obj_id, "euler_angles": {"x": pose.pitch_deg, "y": pose.yaw_deg, "z": pose.roll_deg}}
            )
        else:
            raise ValueError("mode must be orbit_camera or rotate_object")

        # One communicate() per pose => one saved image per pose.
        c.communicate(cmds)

        # Record both pose index and expected filename for alignment verification.
        _append_jsonl(
            meta_jsonl,
            {
                "t": t,
                "image_index": image_index,
                "image_relpath": image_relpath,
                "model_name": model_name,
                "object_id": obj_id,
                "yaw_deg": pose.yaw_deg,
                "pitch_deg": pose.pitch_deg,
                "roll_deg": pose.roll_deg,
                "mode": mode,
                "dist": dist,
            },
        )

    # done
    c.communicate([{ "$type": "terminate" }])

    summary = {
        "model_name": model_name,
        "object_id": obj_id,
        "n_frames": len(poses),
        "img_size": img_size,
        "dist": dist,
        "mode": mode,
        "poses": {"yaw_min": poses[0].yaw_deg if poses else None, "yaw_max": poses[-1].yaw_deg if poses else None, "pitch_list": sorted({p.pitch_deg for p in poses})},
        "frames_dir": str(frames_dir),
        "meta_jsonl": str(meta_jsonl),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--models_file", type=str, required=True, help="newline-separated TDW model names")
    ap.add_argument("--port", type=int, default=1071, help="ZMQ port for TDW build (use a different one if a prior run crashed).")

    ap.add_argument("--mode", type=str, default="orbit_camera", choices=["orbit_camera", "rotate_object"])

    ap.add_argument("--yaw_min", type=float, default=-90)
    ap.add_argument("--yaw_max", type=float, default=90)
    ap.add_argument("--yaw_step", type=float, default=1)
    ap.add_argument("--pitch_list", type=float, nargs="+", default=[0.0], help="e.g. 0 10 -10")
    ap.add_argument("--roll", type=float, default=0.0)

    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera_height", type=float, default=0.0, help="Raise camera (meters).")
    ap.add_argument("--look_at_height", type=float, default=0.0, help="Look-at Y target (meters).")

    args = ap.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_root)

    models = _read_models(Path(args.models_file))

    poses = _make_pose_grid(
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        yaw_step=args.yaw_step,
        pitch_list=list(args.pitch_list),
        roll=args.roll,
    )

    dataset_index: Dict[str, Dict] = {
        "config": {
            "mode": args.mode,
            "yaw_min": args.yaw_min,
            "yaw_max": args.yaw_max,
            "yaw_step": args.yaw_step,
            "pitch_list": list(args.pitch_list),
            "roll": args.roll,
            "dist": args.dist,
            "img_size": args.img_size,
            "test_frac": args.test_frac,
            "seed": args.seed,
        },
        "splits": {"train": [], "test": []},
        "objects": {},
    }

    for i, model_name in enumerate(models):
        split = _hash_split(model_name, test_frac=args.test_frac)
        obj_out = out_root / split / model_name
        print(f"[{i+1}/{len(models)}] Rendering {model_name} -> {obj_out}")
        summary = render_object_sequence(
            model_name=model_name,
            out_dir=obj_out,
            poses=poses,
            img_size=args.img_size,
            dist=args.dist,
            mode=args.mode,
            seed=args.seed,
            port=args.port,
            camera_height=args.camera_height,
            look_at_height=args.look_at_height,
        )
        dataset_index["splits"][split].append(model_name)
        dataset_index["objects"][model_name] = summary

    _write_json(out_root / "dataset_index.json", dataset_index)

    # convenience: precompute (t,t+1,t+2) triples indices once (same for all objects)
    triples = [{"t0": t, "t1": t + 1, "t2": t + 2} for t in range(len(poses) - 2)]
    _write_json(out_root / "triples.json", {"n_frames": len(poses), "triples": triples})

    print("Done.")
    print(f"Index: {out_root/'dataset_index.json'}")


if __name__ == "__main__":
    main()
