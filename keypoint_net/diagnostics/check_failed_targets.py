"""Semantic check of the persistently failing supervised targets (3, 6, 9).

Reproduces the exact deterministic FPS targets, then measures what those
image locations actually ARE: local gradient energy at readout scale,
distance to the nearest strong edge (vs the ~35 px encoder receptive field),
distance to the mask boundary. Saves a labeled patch contact sheet.
Zero training. Read-only.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DS = ("/Users/kirubeso.r/Documents/PhD/data/active/"
      "_tdw_world_z_roll_base_panel_512_v2/train/engineers_hammer_vray")
OUT = ("/private/tmp/claude-501/-Users-kirubeso-r-Documents-PhD/"
       "7563635a-4a3d-4d83-a4d9-383617a7c7ba/scratchpad")
FAILING = {3, 6, 9}
ALSO_FLAKY = {1}
K = 10


def eroded_interior(mask, radius=8):
    fg = torch.tensor(mask.astype(np.float32))[None, None]
    bg = F.max_pool2d(1.0 - fg, kernel_size=2 * radius + 1, stride=1,
                      padding=radius)
    return ((fg > 0.5) & (bg < 0.5))[0, 0].numpy()


def fps(mask, count=K):
    rows, cols = np.nonzero(eroded_interior(mask))
    cand = np.stack([cols, rows], axis=1).astype(np.float64)
    centroid = cand.mean(axis=0)
    first = int(np.argmin(np.sum((cand - centroid) ** 2, axis=1)))
    sel = [first]
    mn = np.sum((cand - cand[first]) ** 2, axis=1)
    for _ in range(1, count):
        nxt = int(np.argmax(mn))
        sel.append(nxt)
        mn = np.minimum(mn, np.sum((cand - cand[nxt]) ** 2, axis=1))
    return cand[sel]


img = np.asarray(Image.open(f"{DS}/frames/a/img_0000.png").convert("L"),
                 dtype=np.float32)
rgb = Image.open(f"{DS}/frames/a/img_0000.png").convert("RGB")
mask = np.asarray(Image.open(f"{DS}/masks/a/mask_0000.png")) > 0

targets = fps(mask)
print("reproduced targets (x, y):")
for i, (x, y) in enumerate(targets):
    print(f"  t{i}: ({x:.0f}, {y:.0f})")
match = (np.allclose(targets[3], [175, 441], atol=1)
         and np.allclose(targets[6], [241, 450], atol=1)
         and np.allclose(targets[9], [279, 418], atol=1))
print(f"doc says t3=(175,441) t6=(241,450) t9=(279,418)  -> match: {match}")

# image-gradient (edge) map
gy, gx = np.gradient(img)
ge = np.sqrt(gx ** 2 + gy ** 2)
edge_ys, edge_xs = np.where(ge > 8.0)

# mask boundary
fg = torch.tensor(mask.astype(np.float32))[None, None]
inner = F.max_pool2d(1.0 - fg, 3, 1, 1)[0, 0].numpy()
b_ys, b_xs = np.where((mask) & (inner > 0.5))

print(f"\n{'tgt':>4} {'fail?':>6} {'gradE(9px)':>11} {'gradE(33px)':>12} "
      f"{'d_edge(px)':>11} {'d_maskbnd(px)':>14}")
rows = []
for i, (x, y) in enumerate(targets):
    xi, yi = int(round(x)), int(round(y))
    w9 = ge[yi - 4:yi + 5, xi - 4:xi + 5]
    w33 = ge[yi - 16:yi + 17, xi - 16:xi + 17]
    d_edge = np.sqrt((edge_ys - yi) ** 2 + (edge_xs - xi) ** 2).min()
    d_bnd = np.sqrt((b_ys - yi) ** 2 + (b_xs - xi) ** 2).min()
    tag = "FAIL" if i in FAILING else ("flaky" if i in ALSO_FLAKY else "ok")
    print(f"  t{i}: {tag:>6} {np.median(w9):11.2f} {np.median(w33):12.2f} "
          f"{d_edge:11.1f} {d_bnd:14.1f}")
    rows.append((i, tag, float(np.median(w9)), float(np.median(w33)),
                 float(d_edge), float(d_bnd)))

# contact sheet: 96x96 patches, red border = persistent failure
P = 48
sheet = Image.new("RGB", (10 * (2 * P + 4), 2 * P + 30), "white")
dr_s = ImageDraw.Draw(sheet)
for i, (x, y) in enumerate(targets):
    xi, yi = int(round(x)), int(round(y))
    patch = rgb.crop((xi - P, yi - P, xi + P, yi + P))
    d = ImageDraw.Draw(patch)
    d.ellipse([P - 5, P - 5, P + 5, P + 5], outline="yellow", width=2)
    col = ("red" if i in FAILING else
           "orange" if i in ALSO_FLAKY else "green")
    d.rectangle([0, 0, 2 * P - 1, 2 * P - 1], outline=col, width=4)
    sheet.paste(patch, (i * (2 * P + 4) + 2, 26))
    dr_s.text((i * (2 * P + 4) + 8, 6), f"t{i} ({xi},{yi})", fill=col)
sheet.save(f"{OUT}/failed_targets_contact_sheet.png")

# full-frame overlay
ov = rgb.copy()
d = ImageDraw.Draw(ov)
for i, (x, y) in enumerate(targets):
    col = ("red" if i in FAILING else
           "orange" if i in ALSO_FLAKY else "lime")
    d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col, width=3)
    d.text((x + 9, y - 7), str(i), fill=col)
ov.save(f"{OUT}/failed_targets_frame0_overlay.png")

# transported targets on training frames stay on-mask?
c = (255.5, 255.5)
print("\ntransported-target mask check (training frames 0/3/6/9):")
for frame in [0, 3, 6, 9]:
    m = np.asarray(Image.open(f"{DS}/masks/a/mask_{frame:04d}.png")) > 0
    for sign in [+1]:
        th = np.deg2rad(sign * 2.0 * frame)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        tp = (targets - c) @ R.T + c
        on = [m[int(round(yy)), int(round(xx))] for xx, yy in tp]
        bad = [i for i, o in enumerate(on) if not o]
        print(f"  frame {frame}: on-mask {sum(on)}/10"
              + (f"  OFF-MASK: {bad}" if bad else ""))
print("\nsaved: failed_targets_contact_sheet.png, "
      "failed_targets_frame0_overlay.png")
