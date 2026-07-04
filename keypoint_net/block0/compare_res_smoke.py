"""
64-vs-128 resolution smoke comparison (review item 3).

PROSPECTIVE CRITERIA (preregistered here before results are seen):
- Same training budget (300 epochs), same losses (Task-55 clean: disp=0.1,
  ent=0.01, smooth=0, inv=cyc=0), same seed (42), same object (hammer),
  single run each — a SMOKE test, not an inferential comparison.
- Jitter: median per-channel high-pass residual sigma in NORMALIZED coords
  (resolution-independent), same derotation pipeline as
  extract_empirical_jitter.py.
- Duplicate separation: median over frames of the minimum pairwise keypoint
  distance; and fraction of frames with any pair closer than one 64-res cell.
- 128 is a GO signal iff median normalized jitter drops >= 25% vs the 64
  twin, OR median min pairwise distance rises >= 25%. Anything less =
  NO-GO on the "128 rescues the affine rung" hypothesis at smoke scale.
Caveats recorded: 300-epoch single-seed smoke; empirical-jitter caveat (the
E7b noise model came from a 1000-epoch Task-80 model with a different
objective).

Run after both trainings finish:
    /opt/anaconda3/envs/phd/bin/python compare_res_smoke.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model import KeypointExtractor  # noqa: E402

RUNS_DIR = HERE.parent / "runs_res_smoke"
FRAMES = Path("/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /"
              "_tdw_world_z_roll_base_panel_512_v2/train/"
              "engineers_hammer_vray/frames/a")
OUT = HERE / "outputs"
CELL64 = 2.0 / 64.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_run(run_dir):
    ck = torch.load(run_dir / "best_model.pt", map_location="cpu",
                    weights_only=False)
    # config.json is authoritative: the checkpoint-embedded config dict is
    # built from a fixed key list in train.py and lacks heatmap_res
    cfg = json.loads((run_dir / "config.json").read_text())
    ext = KeypointExtractor(
        in_channels=3, num_keypoints=cfg["num_keypoints"],
        base_channels=cfg["base_channels"], temperature=cfg["temperature"],
        padding_mode=cfg["padding_mode"],
        heatmap_res=cfg.get("heatmap_res", 64))
    sd = {k[len("extractor."):]: v for k, v in ck["model_state_dict"].items()
          if k.startswith("extractor.")}
    ext.load_state_dict(sd, strict=True)
    ext.eval()
    return ext, cfg


def trajectories(ext):
    files = sorted(FRAMES.glob("img_*.png"))
    coords = []
    with torch.no_grad():
        for i in range(0, len(files), 12):
            batch = []
            for f in files[i:i + 12]:
                im = Image.open(f).convert("RGB").resize((512, 512))
                x = torch.from_numpy(
                    np.asarray(im, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                batch.append((x - MEAN) / STD)
            kp, _ = ext(torch.cat(batch, 0))
            coords.append(kp.view(-1, ext.num_keypoints, 2))
    return torch.cat(coords, 0).numpy()


def derotate_highpass(coords, w=9):
    T = coords.shape[0]
    best = None
    for sign in (+1.0, -1.0):
        th = sign * np.deg2rad(2.0) * np.arange(T)
        c, s = np.cos(-th), np.sin(-th)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        de = np.einsum("tij,tkj->tki", R, coords)
        var = de.var(axis=0).mean()
        if best is None or var < best[0]:
            best = (var, de)
    de = best[1]
    pad = w // 2
    padded = np.concatenate([de[:pad][::-1], de, de[-pad:][::-1]], 0)
    kernel = np.ones(w) / w
    smooth = np.stack([np.stack([
        np.convolve(padded[:, k, d], kernel, mode="valid")
        for d in range(2)], -1) for k in range(de.shape[1])], 1)
    return de - smooth


def metrics(coords):
    resid = derotate_highpass(coords)
    sig = resid.std(axis=0).mean(axis=1)             # per-channel, normalized
    D = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)
    K = coords.shape[1]
    D = D + np.eye(K)[None] * 1e9
    mind = D.min(axis=(1, 2))                        # per-frame min pair dist
    return {
        "jitter_norm_median": float(np.median(sig)),
        "jitter_norm_per_channel": sig.round(5).tolist(),
        "jitter_cells64_median": float(np.median(sig) / CELL64),
        "min_pairdist_median": float(np.median(mind)),
        "frac_frames_pair_lt_1cell64": float((mind < CELL64).mean()),
    }


def main():
    runs = sorted(RUNS_DIR.glob("phase_a_*"))
    by_res = {}
    for r in runs:
        cfg = json.loads((r / "config.json").read_text())
        by_res[cfg.get("heatmap_res", 64)] = r
    assert 64 in by_res and 128 in by_res, f"need both runs, have {by_res}"
    res = {}
    for hr, run_dir in sorted(by_res.items()):
        ext, cfg = load_run(run_dir)
        m = metrics(trajectories(ext))
        m["run"] = run_dir.name
        res[hr] = m
        print(f"heatmap_res={hr}: jitter median {m['jitter_cells64_median']:.3f} "
              f"cells64 ({m['jitter_norm_median']:.5f} norm); "
              f"min-pairdist median {m['min_pairdist_median']:.4f}; "
              f"frames with pair<1cell: {100*m['frac_frames_pair_lt_1cell64']:.0f}%")
    jr = 1 - res[128]["jitter_norm_median"] / res[64]["jitter_norm_median"]
    dr = res[128]["min_pairdist_median"] / max(res[64]["min_pairdist_median"],
                                               1e-12) - 1
    go = jr >= 0.25 or dr >= 0.25
    verdict = {"jitter_reduction": round(float(jr), 3),
               "min_pairdist_increase": round(float(dr), 3),
               "GO_128": bool(go)}
    print(json.dumps(verdict, indent=2))
    (OUT / "res_smoke_comparison.json").write_text(
        json.dumps({"res64": res[64], "res128": res[128],
                    "verdict": verdict}, indent=2))
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n\n---\n\n# 64-vs-128 RESOLUTION SMOKE (review item 3; "
                "prospective criteria in compare_res_smoke.py)\n\n"
                + json.dumps({"res64": res[64], "res128": res[128],
                              "verdict": verdict}, indent=2))
    print("Appended to GATE_REPORT.md")


if __name__ == "__main__":
    main()
