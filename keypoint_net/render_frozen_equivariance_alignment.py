"""Render the candidate-loss versus planted-wobble relationship."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--ocr-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("output directory already exists; use a fresh path")

    roles = []
    for label, receipt_path in (
        ("Control", args.control_receipt),
        ("OCR-ZNCC", args.ocr_receipt),
    ):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        arrays_path = Path(receipt["arrays"]["absolute_path"])
        with np.load(arrays_path, allow_pickle=False) as arrays:
            roles.append(
                (
                    label,
                    receipt,
                    [str(value) for value in arrays["nuisance_name"].tolist()],
                    np.asarray(arrays["jensen_shannon"], dtype=np.float64),
                    np.asarray(arrays["soft_residual_pixels"], dtype=np.float64),
                    np.asarray(arrays["hard_peak_changed"], dtype=np.bool_),
                )
            )

    transform_order = (
        "translation_plus_quarter_pixel_xy",
        "rotation_plus_quarter_degree_clockwise_image",
    )
    transform_labels = ("+0.25 px x/y translation", "0.25 degree clockwise rotation")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
    for row, (role_label, receipt, names, js, residual, changed) in enumerate(roles):
        for column, (name, transform_label) in enumerate(zip(transform_order, transform_labels)):
            index = names.index(name)
            x = js[index].reshape(-1)
            y = residual[index].reshape(-1)
            hard = changed[index].reshape(-1)
            ax = axes[row, column]
            ax.scatter(
                x[~hard],
                y[~hard],
                s=10,
                alpha=0.28,
                color="#3478bf",
                linewidths=0,
                label="hard cell unchanged",
            )
            ax.scatter(
                x[hard],
                y[hard],
                s=13,
                alpha=0.42,
                color="#d64545",
                linewidths=0,
                label="hard cell changed",
            )
            worst = int(np.argmax(y))
            ax.scatter(
                [x[worst]],
                [y[worst]],
                s=105,
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                label="worst coordinate jump",
            )
            summary = receipt["summary"][name]
            rho = summary["spearman_rank_correlation_js_vs_soft_residual"]
            rank = summary["worst_soft_residual_event"]["jensen_shannon_rank_percentile"]
            ax.set_title(
                f"{role_label}: {transform_label}\n"
                f"descriptive Spearman rho={rho:.3f}; worst jump loss rank={100*rank:.1f}%"
            )
            ax.set_xlabel("candidate full-heatmap Jensen-Shannon penalty")
            ax.set_ylabel("soft-coordinate residual after undoing planted motion (input px)")
            ax.grid(alpha=0.18)
            if row == 0 and column == 0:
                ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        "Does the proposed loss target the observed frozen wobble?\n"
        "Each panel: 180 frames x 10 fixed channels = 1,800 correlated observations; descriptive only",
        fontsize=15,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "equivariance_loss_vs_wobble.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
