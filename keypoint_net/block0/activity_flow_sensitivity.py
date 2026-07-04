"""
Activity-signal sensitivity to flow-estimation error (review item 4:
"validate a non-oracle activity signal against the known synthetic flow
before using it in real training").

The Block-0 A3/A4 candidates passed with ORACLE flow. A real system gets an
ESTIMATED flow. This script corrupts the oracle flow with
  flow_est = R(beta) @ flow + eta * |flow|_mean * N(0, I)
(angular bias beta, relative endpoint noise eta) and re-runs the amendment-4
conditions c1-c4 for A3/A4 at each corruption level. The output is the
tolerance envelope: the (eta, beta) region where the candidate still passes.
An image-derived flow (optical flow, feature flow, mask transport) is
admissible for training only if its measured error at keypoint locations
falls inside this envelope. The c5 ascent probe is run at the boundary of
the passing region.

Score logic mirrors block0 phase 6 with injectable flow (kept separate so
the gated block0.py stays frozen).

Run: /opt/anaconda3/envs/phd/bin/python activity_flow_sensitivity.py
"""

import csv
import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from block0 import (OUT, CELL, make_world, sample_G, apply_G, oracle_flow,
                    _track, _anchors, N_TRIALS)

ETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
BETAS = [0.0, 10.0, 20.0, 30.0]
SEED = 20260703


def corrupt(flow, eta, beta_deg, g):
    b = math.radians(beta_deg)
    R = torch.tensor([[math.cos(b), -math.sin(b)],
                      [math.sin(b), math.cos(b)]])
    scale = flow.norm(dim=-1).mean().clamp_min(1e-9)
    noise = eta * scale * torch.randn(flow.shape, generator=g)
    return flow @ R.T + noise


def scores(P0, P1, flow_est):
    v = P1 - P0
    fl = flow_est.norm(dim=-1)
    sref = torch.clamp(0.5 * fl, min=CELL)
    a3 = torch.exp(-((v - flow_est) ** 2).sum(-1) / (2 * sref ** 2))
    gate = torch.clamp(v.norm(dim=-1) / fl.clamp_min(CELL), max=1.0)
    a4 = torch.where(fl < CELL,
                     torch.exp(-(v ** 2).sum(-1) / (2 * CELL ** 2)),
                     gate * a3)
    return {"A3_flowref": a3, "A4_gated": a4}


def cell(eta, beta, fam="affine", K=6):
    acc = {c: {w: [] for w in ["static", "jitter", "tracking"]}
           for c in ["A3_flowref", "A4_gated"]}
    for label, wname, sig in [("static", "static_on", 0.0),
                              ("jitter", "static_on", 1.0),
                              ("tracking", "healthy", 0.5)]:
        for tr in range(N_TRIALS):
            g = torch.Generator().manual_seed(SEED + 13 * tr)
            w = make_world(wname, fam, K, g, sigma=sig)
            fl = corrupt(oracle_flow(w["A"], w["t"], w["P0"]), eta, beta, g)
            sc = scores(w["P0"], w["P1"], fl)
            for c in acc:
                acc[c][label].append(sc[c].mean().item())
    cen, trk = {c: [] for c in acc}, {c: [] for c in acc}
    for tr in range(N_TRIALS):
        g = torch.Generator().manual_seed(SEED + 29 * tr)
        A, t = sample_G("rotation", g)
        anch = torch.cat([torch.zeros(1, 2), _anchors(g, K - 1)], 0)
        P0, P1, P2 = _track(A, t, anch)
        s = 0.5 * CELL
        P0n = P0 + s * torch.randn(K, 2, generator=g)
        P1n = P1 + s * torch.randn(K, 2, generator=g)
        fl = corrupt(oracle_flow(A, t, P0n), eta, beta, g)
        sc = scores(P0n, P1n, fl)
        for c in acc:
            cen[c].append(sc[c][0].item())
            trk[c].append(sc[c][1:].mean().item())
    rows = []
    for c in acc:
        st = float(np.mean(acc[c]["static"]))
        ji = float(np.mean(acc[c]["jitter"]))
        tk = float(np.mean(acc[c]["tracking"]))
        rng = max(tk - st, 1e-12)
        c1 = st < tk - 0.5 * rng + 1e-12
        c2 = (tk - ji) > 0.2 * rng
        c3 = tk > st + 0.5 * rng - 1e-12
        c4 = float(np.mean(cen[c])) >= 0.5 * float(np.mean(trk[c]))
        rows.append({"candidate": c, "eta": eta, "beta_deg": beta,
                     "static": st, "jitter": ji, "tracking": tk,
                     "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                     "pass_c1_c4": all([c1, c2, c3, c4])})
    return rows


def ascent_probe(eta, beta, cand="A4_gated", fam="affine", K=6):
    g = torch.Generator().manual_seed(SEED + 71)
    w = make_world("static_on", fam, K, g)
    P1v = (w["P0"] + 0.1 * CELL * torch.randn(K, 2, generator=g)
           ).clone().requires_grad_(True)
    opt = torch.optim.Adam([P1v], lr=0.01)
    for _ in range(150):
        fl = corrupt(oracle_flow(w["A"], w["t"], w["P0"]), eta, beta, g)
        sc = scores(w["P0"], P1v, fl)[cand].mean()
        opt.zero_grad(); (-sc).backward(); opt.step()
    d1 = (P1v - w["P0"]).detach().norm(dim=-1).mean().item()
    fm = oracle_flow(w["A"], w["t"], w["P0"]).norm(dim=-1).mean().item()
    return d1 > 0.5 * fm


def main():
    rows = []
    for eta in ETAS:
        for beta in BETAS:
            rows.extend(cell(eta, beta))
    with open(OUT / "activity_flow_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    env = {}
    for c in ["A3_flowref", "A4_gated"]:
        passing = [(r["eta"], r["beta_deg"]) for r in rows
                   if r["candidate"] == c and r["pass_c1_c4"]]
        max_eta = max([e for e, b in passing if b == 0.0], default=None)
        max_beta = max([b for e, b in passing if e == 0.0], default=None)
        env[c] = {"passing_cells": passing, "max_eta_at_beta0": max_eta,
                  "max_beta_at_eta0": max_beta}
        print(f"{c}: max eta (beta=0) = {max_eta}; "
              f"max beta (eta=0) = {max_beta}; "
              f"passing cells = {len(passing)}/{len(ETAS)*len(BETAS)}")
    bd_eta = env["A4_gated"]["max_eta_at_beta0"]
    c5 = ascent_probe(bd_eta if bd_eta is not None else 0.0, 0.0)
    lines = ["\n\n---\n\n# ACTIVITY FLOW-SENSITIVITY (review item 4; "
             "non-oracle validation on synthetic flow)\n",
             "flow_est = R(beta) flow + eta*|flow|*noise; conditions c1-c4 "
             "re-tested per corruption level:\n"]
    for c, e in env.items():
        lines.append(f"- {c}: passes c1-c4 up to eta={e['max_eta_at_beta0']} "
                     f"(beta=0) and beta={e['max_beta_at_eta0']} deg (eta=0); "
                     f"{len(e['passing_cells'])}/{len(ETAS)*len(BETAS)} "
                     "corruption cells pass.")
    lines.append(f"- c5 ascent probe at A4's eta boundary (beta=0): "
                 f"escaped_static={c5}")
    lines.append("\nADMISSIBILITY REQUIREMENT for any image-derived flow "
                 "(optical flow / feature flow / mask transport): measured "
                 "relative endpoint error and angular bias AT KEYPOINT "
                 "LOCATIONS must fall inside the passing envelope above, "
                 "verified against ground-truth G on the synthetic dataset "
                 "before entering any training loss.")
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n".join(lines))
    print("Appended to GATE_REPORT.md")


if __name__ == "__main__":
    main()
