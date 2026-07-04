"""
Semantic verification of the 64-vs-128 smoke comparison (rule: a passing
metric is not a result until the output is checked against intent).

Questions:
1. Are the keypoints ON THE OBJECT in both runs, or did the 128 run's large
   min-pairdist come from scattering onto the background?
2. What does the 64 run's total duplication (min-pairdist 0.0004) look like —
   collapapsed on-object or parked off-object?

Method: motion support mask = pixels whose grayscale std over the sequence
exceeds a threshold (background is static by construction). On-object
fraction = fraction of (frame, keypoint) samples inside the (dilated) motion
support. Also saves keypoint overlays on two frames per run for eyeballing.

Run: /opt/anaconda3/envs/phd/bin/python check_smoke_semantics.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from compare_res_smoke import RUNS_DIR, FRAMES, OUT, load_run, trajectories

COLORS = ["red", "lime", "blue", "yellow", "magenta", "cyan", "orange",
          "white", "pink", "springgreen"]


def motion_mask(stride=6, thresh=8.0, dilate=9):
    files = sorted(FRAMES.glob("img_*.png"))[::stride]
    stack = np.stack([np.asarray(Image.open(f).convert("L").resize((512, 512)),
                                 dtype=np.float32) for f in files])
    m = (stack.std(axis=0) > thresh).astype(np.float32)
    t = torch.from_numpy(m)[None, None]
    m = torch.nn.functional.max_pool2d(t, dilate, stride=1,
                                       padding=dilate // 2)[0, 0].numpy()
    return m


def to_pix(coords):
    return np.clip((coords + 1.0) / 2.0 * 511.0, 0, 511)


def overlay(run_name, frame_idx, coords_px, tag):
    f = sorted(FRAMES.glob("img_*.png"))[frame_idx]
    im = Image.open(f).convert("RGB").resize((512, 512))
    dr = ImageDraw.Draw(im)
    for k in range(coords_px.shape[0]):
        x, y = coords_px[k]
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COLORS[k], width=3)
        dr.text((x + 8, y - 6), str(k), fill=COLORS[k])
    p = OUT / f"smoke_overlay_{tag}_frame{frame_idx}.png"
    im.save(p)
    return p


def main():
    mask = motion_mask()
    print(f"motion support covers {100*mask.mean():.1f}% of image")
    runs = sorted(RUNS_DIR.glob("phase_a_*"))
    summary = {}
    for r in runs:
        cfg = json.loads((r / "config.json").read_text())
        hr = cfg.get("heatmap_res", 64)
        ext, _ = load_run(r)
        coords = trajectories(ext)                    # (180, 10, 2)
        px = to_pix(coords)
        on = mask[px[..., 1].astype(int), px[..., 0].astype(int)]
        per_ch = on.mean(axis=0)
        summary[f"res{hr}"] = {
            "on_object_frac_overall": float(on.mean()),
            "on_object_frac_per_channel": per_ch.round(3).tolist(),
            "channels_mostly_on_object(>80%)": int((per_ch > 0.8).sum()),
        }
        for fi in [0, 45]:
            p = overlay(r.name, fi, px[fi], f"res{hr}")
            print(f"  saved {p.name}")
        print(f"res{hr}: on-object {100*on.mean():.1f}% overall; "
              f"per-channel {per_ch.round(2).tolist()}")
    (OUT / "smoke_semantics.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n\n## Semantic check of the smoke comparison "
                "(check_smoke_semantics.py)\n\n" + json.dumps(summary,
                                                              indent=2))
    print("Appended to GATE_REPORT.md")


if __name__ == "__main__":
    main()
