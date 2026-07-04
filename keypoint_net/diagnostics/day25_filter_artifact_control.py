"""Control whether the width-9 high-pass creates the apparent temporal failure.

Semantic lock (frozen before execution):
* Synthetic input is temporally and cross-channel independent Gaussian noise.
* The exact Block-0 width-9 high-pass is applied.
* The primary control is post-filter variance-matched per channel and axis, so
  it changes dependence without confounding the empirical noise magnitude.
* Five independently generated banks are evaluated with the existing affine
  ordering cell (K={6,10}, B=16, N=150).
* Both K cells closer to empirical => filter-artifact-like; both closer to the
  iid-heterogeneous reference => network-residual-like; split => unresolved.
  Filter-artifact-like or unresolved conservatively selects a cubic detrended
  bank and reruns the two key cells before similarity optimization.

All intervals are descriptive Monte-Carlo summaries conditional on this one
checkpoint/orbit. They are not checkpoint, object, or population inference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import day25_noise_ladder as ladder
from dxutils import OUTPUTS, highpass_residual


SEED = 20260705
N_BANKS = 5
N_COMPARISONS = 150
BATCH_SIZE = 16
REFERENCE = {
    6: {"empirical": 0.660, "iid_heterogeneous": 0.907},
    10: {"empirical": 0.373, "iid_heterogeneous": 0.807},
}
ORIGINAL_BANK_PATH = ladder.RESIDUAL_PATH
PARAMETRIC_BANK_PATH = OUTPUTS / "day25_parametric_detrend_residuals.npz"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lag1(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape[1:], dtype=np.float64)
    for channel in range(values.shape[1]):
        for axis in range(values.shape[2]):
            output[channel, axis] = np.corrcoef(
                values[:-1, channel, axis], values[1:, channel, axis]
            )[0, 1]
    return np.nan_to_num(output)


def make_filtered_white_bank(seed: int) -> tuple[np.ndarray, dict]:
    empirical = ladder.RESIDUALS
    target_sigma = empirical.std(axis=0, ddof=0)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(empirical.shape) * target_sigma[None]
    filtered = highpass_residual(raw, window=9)
    filtered_sigma = filtered.std(axis=0, ddof=0)
    matched = filtered * (target_sigma / np.maximum(filtered_sigma, 1e-15))[None]
    matched -= matched.mean(axis=0, keepdims=True)
    metrics = {
        "seed": seed,
        "raw_lag1_mean": float(_lag1(raw).mean()),
        "filtered_lag1_mean": float(_lag1(filtered).mean()),
        "matched_lag1_mean": float(_lag1(matched).mean()),
        "matched_sigma_max_abs_error": float(
            np.max(np.abs(matched.std(axis=0, ddof=0) - target_sigma))
        ),
    }
    return matched, metrics


def cubic_detrend_bank() -> tuple[np.ndarray, dict]:
    source = np.load(ORIGINAL_BANK_PATH)["derotated"].astype(np.float64)
    time = np.linspace(-1.0, 1.0, source.shape[0])
    trend = np.empty_like(source)
    for channel in range(source.shape[1]):
        for axis in range(source.shape[2]):
            coefficients = np.polyfit(time, source[:, channel, axis], deg=3)
            trend[:, channel, axis] = np.polyval(coefficients, time)
    residuals = source - trend
    residuals -= residuals.mean(axis=0, keepdims=True)
    np.savez_compressed(
        PARAMETRIC_BANK_PATH,
        residuals=residuals,
        derotated=source,
        fitted_trend=trend,
        detrend_degree=np.array(3),
    )
    metrics = {
        "path": str(PARAMETRIC_BANK_PATH),
        "detrend": "independent cubic least-squares fit per channel and axis",
        "lag1_mean": float(_lag1(residuals).mean()),
        "sigma_channel_cells64": (
            np.sqrt(np.mean(residuals.std(axis=0, ddof=0) ** 2, axis=1))
            / ladder.CELL64_NORM
        ).tolist(),
    }
    return residuals, metrics


def run_ordering_for_bank(
    residuals: np.ndarray, *, bank_index: int | str, label: str
) -> list[dict]:
    original = ladder.BANK
    ladder.BANK = ladder.NoiseBank(residuals)
    rows = []
    try:
        for keypoints in (6, 10):
            result = ladder.ordering_cell(
                "empirical",
                keypoints,
                n_comparisons=N_COMPARISONS,
                batch_size=BATCH_SIZE,
            )
            result["bank_index"] = bank_index
            result["bank_label"] = label
            rows.append(result)
            print(
                f"{label} bank={bank_index} K={keypoints}: "
                f"{100 * result['ordering_fraction']:.1f}%"
            )
    finally:
        ladder.BANK = original
    return rows


def summarize_control(rows: list[dict]) -> tuple[dict, str]:
    summary: dict[str, dict] = {}
    per_k_decisions = []
    for keypoints in (6, 10):
        selected = [row for row in rows if int(row["K"]) == keypoints]
        values = np.array([row["ordering_fraction"] for row in selected])
        wins = sum(int(row["wins"]) for row in selected)
        n = sum(int(row["n_batch_comparisons"]) for row in selected)
        low, high = ladder.wilson_interval(wins, n)
        mean = float(values.mean())
        distances = {
            name: abs(mean - value) for name, value in REFERENCE[keypoints].items()
        }
        closer_to = min(distances, key=distances.get)
        per_k_decisions.append(closer_to)
        summary[str(keypoints)] = {
            "individual_bank_fractions": values.tolist(),
            "across_bank_mean": mean,
            "across_bank_sample_std_ddof1": float(values.std(ddof=1)),
            "pooled_wins": wins,
            "pooled_n": n,
            "pooled_wilson95": [low, high],
            "reference": REFERENCE[keypoints],
            "absolute_distance": distances,
            "closer_to": closer_to,
        }
    if all(item == "empirical" for item in per_k_decisions):
        verdict = "filter_artifact_like"
    elif all(item == "iid_heterogeneous" for item in per_k_decisions):
        verdict = "network_residual_like"
    else:
        verdict = "unresolved"
    return summary, verdict


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    empirical_rho = _lag1(ladder.RESIDUALS)
    rows: list[dict] = []
    bank_metrics = []
    for bank_index in range(N_BANKS):
        residuals, metrics = make_filtered_white_bank(SEED + 1009 * bank_index)
        metrics["bank_index"] = bank_index
        bank_metrics.append(metrics)
        rows.extend(
            run_ordering_for_bank(
                residuals, bank_index=bank_index, label="variance_matched_filtered_white"
            )
        )

    control_summary, verdict = summarize_control(rows)
    selected_path: Path
    parametric_metrics = None
    parametric_rows: list[dict] = []
    if verdict == "network_residual_like":
        selected_path = ORIGINAL_BANK_PATH
    else:
        parametric, parametric_metrics = cubic_detrend_bank()
        parametric_rows = run_ordering_for_bank(
            parametric, bank_index="parametric", label="cubic_detrended_empirical"
        )
        selected_path = PARAMETRIC_BANK_PATH

    all_rows = rows + parametric_rows
    _write_csv(OUTPUTS / "day25_filter_artifact_control.csv", all_rows)
    artifact = {
        "semantic_lock": {
            "primary_control": "width-9-highpass filtered white noise, post-filter variance matched per channel/axis",
            "n_independent_synthetic_banks": N_BANKS,
            "ordering": f"K=6,10; B={BATCH_SIZE}; N={N_COMPARISONS} per bank",
            "decision_rule": "both K closer to empirical => filter_artifact_like; both closer to iid_heterogeneous => network_residual_like; split => unresolved",
        },
        "empirical_bank_lag1_mean": float(empirical_rho.mean()),
        "synthetic_bank_metrics": bank_metrics,
        "control_summary": control_summary,
        "verdict": verdict,
        "parametric_rebuild": parametric_metrics,
        "parametric_ordering_cells": parametric_rows,
        "selected_residual_bank": str(selected_path),
        "similarity_run_authorized": True,
        "statistics": {
            "individual_unit": "one simulated B=16 batch comparison",
            "across_bank_std": "sample standard deviation across five independently generated synthetic banks, ddof=1",
            "pooled_interval": "Wilson 95% interval over 750 simulated comparisons per K; descriptive Monte-Carlo uncertainty",
            "generalization": "conditional on one Task-80 checkpoint/orbit; not population inference",
        },
    }
    (OUTPUTS / "day25_filter_artifact_control.json").write_text(
        json.dumps(artifact, indent=2)
    )
    (OUTPUTS / "day25_selected_residual_bank.json").write_text(
        json.dumps(
            {
                "path": str(selected_path),
                "selection_basis": verdict,
                "similarity_run_authorized": True,
            },
            indent=2,
        )
    )
    print(json.dumps({"verdict": verdict, "selected_residual_bank": str(selected_path)}, indent=2))


if __name__ == "__main__":
    main()
