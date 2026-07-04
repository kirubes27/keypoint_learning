"""
Extract the EMPIRICAL keypoint-jitter distribution from a real trained
checkpoint (Task 80, seed 42) on the real 180-frame roll dataset.

Needed by E7b (review decision 2026-07-03): the jitter model must include
temporal and channel correlations, not only iid Gaussian noise.

Method:
1. Load the extractor weights from best_model.pt (extractor.* keys only).
2. Run all 180 frames (2 deg/frame world-z roll) -> coords (180, 10, 2).
3. Derotate by the known frame angle about the image centre (sign chosen by
   whichever minimizes derotated variance); an attached keypoint is constant
   after derotation.
4. High-pass: residual = derotated - centered moving average (w=9). The slow
   component is drift (a separate phenomenon); the fast residual is jitter.
5. Save per-channel sigma (in 64-res cells), lag-1 autocorrelation, cross-
   channel correlation, and the raw residual array for bootstrap use in E7b.

Run: /opt/anaconda3/envs/phd/bin/python extract_empirical_jitter.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # keypoint_net/
from model import KeypointExtractor          # noqa: E402

RUN = Path("/Users/kirubeso.r/Documents/PhD/cluster_downloads/"
           "hammer_full360_shared_complete/keypoint_net/"
           "runs_hammer_full360_shared/"
           "phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_"
           "pid3289780")
FRAMES = Path("/Users/kirubeso.r/Documents/PhD/tdw_phase_a_starter /"
              "_tdw_world_z_roll_base_panel_512_v2/train/"
              "engineers_hammer_vray/frames/a")
OUT = HERE / "outputs"
CELL64 = 2.0 / 64.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_extractor():
    ck = torch.load(RUN / "best_model.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    ext = KeypointExtractor(
        in_channels=3, num_keypoints=cfg["num_keypoints"],
        base_channels=cfg["base_channels"], temperature=cfg["temperature"],
        padding_mode=cfg["padding_mode"])
    sd = {k[len("extractor."):]: v for k, v in ck["model_state_dict"].items()
          if k.startswith("extractor.")}
    ext.load_state_dict(sd, strict=True)
    ext.eval()
    return ext, cfg


def run_frames(ext):
    files = sorted(FRAMES.glob("img_*.png"))
    assert len(files) == 180, f"expected 180 frames, got {len(files)}"
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
    return torch.cat(coords, 0).numpy()          # (180, 10, 2), [-1,1]


def derotate(coords):
    T = coords.shape[0]
    best = None
    for sign in (+1.0, -1.0):
        th = sign * np.deg2rad(2.0) * np.arange(T)
        c, s = np.cos(-th), np.sin(-th)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        de = np.einsum("tij,tkj->tki", R, coords)
        var = de.var(axis=0).mean()
        if best is None or var < best[0]:
            best = (var, sign, de)
    return best[2], best[1]


def highpass(de, w=9):
    pad = w // 2
    padded = np.concatenate([de[:pad][::-1], de, de[-pad:][::-1]], 0)
    kernel = np.ones(w) / w
    smooth = np.stack([np.stack([
        np.convolve(padded[:, k, d], kernel, mode="valid")
        for d in range(2)], -1) for k in range(de.shape[1])], 1)
    return de - smooth


def main():
    ext, cfg = load_extractor()
    print(f"loaded extractor from {RUN.name} "
          f"(padding={cfg['padding_mode']}, epoch={cfg.get('epochs')})")
    coords = run_frames(ext)
    de, sign = derotate(coords)
    resid = highpass(de)
    sig_ch = resid.std(axis=0)                       # (10, 2)
    sig_cells = sig_ch / CELL64
    # lag-1 autocorrelation per channel/dim
    r = resid - resid.mean(0)
    rho = np.array([[np.corrcoef(r[:-1, k, d], r[1:, k, d])[0, 1]
                     for d in range(2)] for k in range(r.shape[1])])
    # cross-channel correlation (pooled x/y): mean off-diagonal
    flat = r.reshape(r.shape[0], -1)                 # (T, 20)
    C = np.corrcoef(flat.T)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    summary = {
        "run": RUN.name, "derotation_sign": sign,
        "sigma_cells_per_channel": sig_cells.mean(axis=1).round(3).tolist(),
        "sigma_cells_median": float(np.median(sig_cells)),
        "sigma_cells_mean": float(sig_cells.mean()),
        "lag1_autocorr_mean": float(np.nanmean(rho)),
        "cross_channel_corr_mean": float(off.mean()),
        "cross_channel_corr_p90": float(np.quantile(np.abs(off), 0.9)),
        "note": ("residuals in normalized [-1,1] coords; CELL64=0.03125; "
                 "high-pass window 9 frames; drift excluded by design"),
    }
    np.savez(OUT / "empirical_jitter_residuals.npz",
             residuals=resid, derotated=de, coords=coords)
    (OUT / "empirical_jitter_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
