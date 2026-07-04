"""
POST-HOC SUPPLEMENT to Block 0 (NOT preregistered — labeled per plan rules).

Motivation: E7's preregistered unit was single-triplet paired ordering
(dup > healthy per individual triplet). Training, however, optimizes the
batch-averaged loss: the gradient signal SGD sees is the ordering of
BATCH MEANS, not of single triplets. This supplement measures the same
duplicate-vs-healthy ordering at batch sizes B in {1, 8, 32} (B=1 replicates
the preregistered test). It does not overwrite E7; it informs the review
decision on whether to amend the criterion's unit.

Run: /opt/anaconda3/envs/phd/bin/python supplementary_e7_batch.py
Appends its section to outputs/GATE_REPORT.md and writes
outputs/supplement_e7_batch.csv.
"""

import csv
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

from block0 import OUT, sample_G, subset_consistency, _jittered_world

N_COMPARISONS = 20


def batch_ordering(fam, K, sig, B, seed=0):
    wins = 0
    gaps = []
    for comp in range(N_COMPARISONS):
        h_acc, d_acc = 0.0, 0.0
        for b in range(B):
            g = torch.Generator().manual_seed(seed + 7919 * (comp * 1000 + b))
            A, t = sample_G(fam, g)
            gh = torch.Generator().manual_seed(
                seed + 104729 * (comp * 1000 + b) + 1)
            gd = torch.Generator().manual_seed(
                seed + 104729 * (comp * 1000 + b) + 2)
            wh = _jittered_world(A, t, "healthy", K, gh, sig)
            wd = _jittered_world(A, t, "dup_exact", K, gd, sig)
            h_acc += subset_consistency(*wh, fam)["mean"].item()
            d_acc += subset_consistency(*wd, fam)["mean"].item()
        wins += int(d_acc > h_acc)
        gaps.append(d_acc / max(h_acc, 1e-300))
    return wins / N_COMPARISONS, sum(gaps) / len(gaps)


def main():
    rows = []
    for fam in ["affine", "similarity"]:
        for K in [6, 10]:
            for sig in [1.0, 2.0]:
                for B in [1, 8, 32]:
                    frac, gap = batch_ordering(fam, K, sig, B)
                    rows.append({"family": fam, "K": K, "sigma_cells": sig,
                                 "batch_size": B,
                                 "frac_dup_gt_healthy": frac,
                                 "mean_gap_ratio": gap})
                    print(f"{fam} K={K} sigma={sig} B={B}: "
                          f"frac={frac:.2f} gap={gap:.2f}")
    with open(OUT / "supplement_e7_batch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["\n\n---\n\n# POST-HOC SUPPLEMENT (not preregistered): "
             "E7 at training-relevant batch sizes\n",
             "E7's preregistered unit was single-triplet ordering; SGD sees "
             "batch means. Same worlds, same losses, ordering of batch-mean "
             "losses at B in {1, 8, 32} (B=1 = preregistered test):\n"]
    for r in rows:
        lines.append(f"- {r['family']} K={r['K']} sigma={r['sigma_cells']} "
                     f"B={r['batch_size']}: dup>healthy in "
                     f"{100*r['frac_dup_gt_healthy']:.0f}% of comparisons "
                     f"(mean gap x{r['mean_gap_ratio']:.2f})")
    lines.append(
        "\nProposed (subject to review, NOT self-approved): amend E7's unit "
        "to batch-averaged ordering at a realistic training batch size "
        "(B >= 8), keeping sigma = 1 cell. Rationale: the loss is only ever "
        "optimized as a batch mean; single-triplet ordering measures noise "
        "SGD averages out. If review rejects the amendment, E7 stands as "
        "FAIL and the mechanism must be hardened (e.g., conditioning-aware "
        "subset weighting) before the image stage.")
    with open(OUT / "GATE_REPORT.md", "a") as f:
        f.write("\n".join(lines))
    print("\nAppended to GATE_REPORT.md")


if __name__ == "__main__":
    main()
