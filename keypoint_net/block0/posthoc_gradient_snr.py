"""
POST-HOC diagnostic (NOT preregistered; does not change E7b = FAIL).

E7b's C2 asked for per-batch gradient separation in >=95% of batches and got
~coin-flip. Per-batch sign is the wrong long-run quantity for SGD, which
accumulates gradients over many steps: what matters is whether the EXPECTED
separating component is positive and how many batches are needed before the
accumulated signal dominates the noise. This script measures, under the
EMPIRICAL noise model, the mean separating gradient m, its std s across
batches, and the implied number of batches n* ~ (s/m)^2 for ~2-sigma
reliable separation. It informs (but cannot substitute for) the review
decision on whether C2's per-batch-sign criterion should be revised in a
future prospective gate.

Run: /opt/anaconda3/envs/phd/bin/python posthoc_gradient_snr.py
"""

import json

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from block0 import CELL, OUT, sample_G, subset_consistency, _distinct3, \
    _track, U
from e7b_confirmation import _noise, SEED_BASE, hashseed

N_BATCH = 150
B = 16


def sep_components(fam, noise_kind, eps_cells=0.5):
    vals = []
    for comp in range(N_BATCH):
        rng = np.random.default_rng(SEED_BASE + 31 * comp
                                    + hashseed("grad", fam, noise_kind))
        for b in range(B):
            g = torch.Generator().manual_seed(
                SEED_BASE + 6007 * (comp * B + b)
                + hashseed("grad", fam, noise_kind))
            A, t = sample_G(fam, g)
            base = _distinct3(g)
            d = U(g, 0.0, 2 * torch.pi, 1)[0]
            off = eps_cells * CELL * torch.stack([torch.cos(d), torch.sin(d)])
            extra = U(g, 0.15, 0.55, 2)
            ang = U(g, 0.0, 2 * torch.pi, 2)
            more = torch.stack([extra * torch.cos(ang),
                                extra * torch.sin(ang)], -1)
            anch = torch.cat([base, (base[0] + off)[None, :], more], 0
                             ).clone().requires_grad_(True)
            P0, P1, P2 = _track(A, t, anch)
            n = _noise(noise_kind, 6, rng)
            loss = subset_consistency(P0 + n[0], P1 + n[1], P2 + n[2],
                                      fam)["mean"]
            gr, = torch.autograd.grad(loss, anch)
            s = (anch[0] - anch[3]).detach()
            u = s / s.norm().clamp_min(1e-12)
            vals.append(float(u @ (-gr[0] + gr[3])))
    return np.array(vals)


def main():
    out = {}
    for fam in ["affine", "similarity"]:
        for nk in ["emp", "emp_half"]:
            v = sep_components(fam, nk)
            m, s = v.mean(), v.std()
            bm = v.reshape(N_BATCH, B).mean(1)      # per-batch means (B=16)
            n_star = (s / m) ** 2 / B if m > 0 else float("inf")
            out[f"{fam}_{nk}"] = {
                "mean_sep_grad": float(m), "std_across_triplets": float(s),
                "frac_positive_triplet": float((v > 0).mean()),
                "frac_positive_batch": float((bm > 0).mean()),
                "t_stat": float(m / (s / np.sqrt(len(v)))),
                "batches_for_2sigma": float(4 * n_star)
                if np.isfinite(n_star) else None}
            print(f"{fam} {nk}: mean={m:.3e} (t={out[f'{fam}_{nk}']['t_stat']:.1f}), "
                  f"pos-batch={100*(bm>0).mean():.0f}%, "
                  f"~batches for 2-sigma={out[f'{fam}_{nk}']['batches_for_2sigma']:.0f}"
                  if m > 0 else f"{fam} {nk}: mean={m:.3e} NEGATIVE")
    (OUT / "posthoc_gradient_snr.json").write_text(json.dumps(out, indent=2))
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n\n---\n\n# POST-HOC GRADIENT-SNR DIAGNOSTIC "
                "(not preregistered; E7b remains FAIL)\n\n"
                + json.dumps(out, indent=2)
                + "\n\nInterpretation: mean separating gradient and the "
                "approximate number of B=16 batches for the accumulated "
                "separation signal to reach 2-sigma reliability. SGD "
                "integrates over hundreds of batches per epoch-scale window; "
                "a positive mean with n* << typical training length means "
                "separation still occurs in expectation despite per-batch "
                "sign noise. Any revision of the C2 criterion is a REVIEW "
                "decision for a future prospective gate.")
    print("Appended to GATE_REPORT.md")


if __name__ == "__main__":
    main()
