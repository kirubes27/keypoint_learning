"""
E7b — PROSPECTIVE confirmation gate (review decision 2026-07-03).
E7 remains FAIL on the record; this is a NEW criterion, preregistered here
BEFORE running, on fresh held-out seeds.

PREREGISTERED CRITERIA (do not edit after first run):
- Seed base 20260703 (never used in Block 0, which used base 0).
- Batch size B = 16 (intended training batch size); N = 150 batch
  comparisons per cell (>= 100 required by review).
- Cells: {affine, similarity} x K in {6, 10}.
- Noise conditions:
    (a) EMPIRICAL bootstrap  [PRIMARY]: contiguous 3-frame windows of real
        high-pass residuals from Task 80 (extract_empirical_jitter.py),
        sampled per-triplet across channels -> preserves temporal and
        cross-channel correlation and per-channel heterogeneity
        (mean sigma = 0.97 cells, includes the dead channel).
    (b) iid Gaussian sigma = 1 cell (replication reference, informative).
    (c) empirical scaled x 0.5 (128-resolution preview, informative).
- E7b PASSES iff, under EMPIRICAL noise (a), for ALL four cells:
    C1 (ordering): batch-mean loss of dup_exact > healthy in >= 95% of the
        150 paired comparisons; and
    C2 (gradient): for near-duplicates at eps = 0.5 cell (K = 6 cells only),
        the batch gradient separates the duplicate pair (separating
        component of -grad positive) in >= 95% of 150 batches.
Conditions (b) and (c) are reported but not gating.

Run: /opt/anaconda3/envs/phd/bin/python e7b_confirmation.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from block0 import (OUT, CELL, sample_G, subset_consistency, _jittered_world,
                    _distinct3, _track, U)

SEED_BASE = 20260703
B = 16
N_COMP = 150
RESID = np.load(OUT / "empirical_jitter_residuals.npz")["residuals"]  # (180,10,2)


def _noise(kind, K, rng):
    """Return (3, K, 2) noise for one triplet, in normalized coords."""
    if kind == "iid1":
        return torch.tensor(rng.standard_normal((3, K, 2))) * CELL
    t0 = rng.integers(0, RESID.shape[0] - 3)
    chans = rng.choice(RESID.shape[1], size=K, replace=(K > RESID.shape[1]))
    win = torch.tensor(RESID[t0:t0 + 3][:, chans, :].astype(np.float64))
    if kind == "emp":
        return win
    if kind == "emp_half":
        return 0.5 * win
    raise ValueError(kind)


def _noisy_triplet(A, t, kind_world, K, g, rng, noise_kind):
    P0, P1, P2 = _jittered_world(A, t, kind_world, K, g, 0.0)
    n = _noise(noise_kind, K, rng)
    return P0 + n[0], P1 + n[1], P2 + n[2]


def ordering_cell(fam, K, noise_kind):
    wins = 0
    for comp in range(N_COMP):
        rng = np.random.default_rng(SEED_BASE + comp * 7 + hashseed(fam, K,
                                                                    noise_kind))
        h_acc = d_acc = 0.0
        for b in range(B):
            g = torch.Generator().manual_seed(
                SEED_BASE + 7919 * (comp * B + b) + hashseed(fam, K,
                                                             noise_kind))
            A, t = sample_G(fam, g)
            gh = torch.Generator().manual_seed(
                SEED_BASE + 104729 * (comp * B + b) + 1)
            gd = torch.Generator().manual_seed(
                SEED_BASE + 104729 * (comp * B + b) + 2)
            wh = _noisy_triplet(A, t, "healthy", K, gh, rng, noise_kind)
            wd = _noisy_triplet(A, t, "dup_exact", K, gd, rng, noise_kind)
            h_acc += subset_consistency(*wh, fam)["mean"].item()
            d_acc += subset_consistency(*wd, fam)["mean"].item()
        wins += int(d_acc > h_acc)
    return wins / N_COMP


def hashseed(*args):
    import zlib
    return zlib.crc32("|".join(map(str, args)).encode()) % 99991


def gradient_cell(fam, noise_kind, K=6, eps_cells=0.5):
    """Fraction of batches whose batch gradient separates a near-dup pair."""
    sep_wins = 0
    for comp in range(N_COMP):
        rng = np.random.default_rng(SEED_BASE + 31 * comp
                                    + hashseed("grad", fam, noise_kind))
        sep_scores = []
        for b in range(B):
            g = torch.Generator().manual_seed(
                SEED_BASE + 6007 * (comp * B + b)
                + hashseed("grad", fam, noise_kind))
            A, t = sample_G(fam, g)
            base = _distinct3(g)
            d = U(g, 0.0, 2 * torch.pi, 1)[0]
            off = eps_cells * CELL * torch.stack(
                [torch.cos(d), torch.sin(d)])
            extra = U(g, 0.15, 0.55, 2)  # radii for two more distinct anchors
            ang = U(g, 0.0, 2 * torch.pi, 2)
            more = torch.stack([extra * torch.cos(ang),
                                extra * torch.sin(ang)], -1)
            anch = torch.cat([base, (base[0] + off)[None, :], more], 0
                             ).clone().requires_grad_(True)   # K=6, dup pair (0,3)
            P0, P1, P2 = _track(A, t, anch)
            n = _noise(noise_kind, 6, rng)
            loss = subset_consistency(P0 + n[0], P1 + n[1], P2 + n[2],
                                      fam)["mean"]
            gr, = torch.autograd.grad(loss, anch)
            s = (anch[0] - anch[3]).detach()
            u = s / s.norm().clamp_min(1e-12)
            # d|separation|/dstep under gradient DESCENT:
            sep_scores.append(float(u @ (-gr[0] + gr[3])))
        sep_wins += int(sum(sep_scores) / len(sep_scores) > 0)
    return sep_wins / N_COMP


def main():
    rows = []
    print(f"E7b: B={B}, N={N_COMP}, seed base {SEED_BASE}")
    for noise_kind in ["emp", "iid1", "emp_half"]:
        for fam in ["affine", "similarity"]:
            for K in [6, 10]:
                frac = ordering_cell(fam, K, noise_kind)
                rows.append({"test": "ordering", "family": fam, "K": K,
                             "noise": noise_kind, "frac": frac})
                print(f"  ordering {fam} K={K} {noise_kind}: {frac:.3f}")
            gfrac = gradient_cell(fam, noise_kind)
            rows.append({"test": "gradient_sep", "family": fam, "K": 6,
                         "noise": noise_kind, "frac": gfrac})
            print(f"  gradient {fam} K=6 {noise_kind}: {gfrac:.3f}")
    with open(OUT / "e7b_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    emp = [r for r in rows if r["noise"] == "emp"]
    c1 = all(r["frac"] >= 0.95 for r in emp if r["test"] == "ordering")
    c2 = all(r["frac"] >= 0.95 for r in emp if r["test"] == "gradient_sep")
    verdict = "PASS" if (c1 and c2) else "FAIL"
    lines = ["\n\n---\n\n# E7b PROSPECTIVE CONFIRMATION (preregistered in "
             "e7b_confirmation.py; fresh seeds; empirical jitter)\n",
             f"Empirical noise source: Task-80 residuals "
             f"(mean sigma 0.97 cells, per-channel 0.66-1.56 + dead channel, "
             f"temporal/channel correlations preserved by 3-frame bootstrap "
             f"windows).\n"]
    for r in rows:
        lines.append(f"- {r['test']} {r['family']} K={r['K']} "
                     f"noise={r['noise']}: {100*r['frac']:.1f}%")
    lines.append(f"\nC1 ordering >=95% (empirical, all cells): {c1}")
    lines.append(f"C2 gradient separation >=95% (empirical): {c2}")
    lines.append(f"\n## E7b VERDICT: {verdict}")
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n".join(lines))
    print(f"\nE7b VERDICT: {verdict} (C1={c1}, C2={c2}) — appended to "
          "GATE_REPORT.md")


if __name__ == "__main__":
    main()
