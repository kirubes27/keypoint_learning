"""Day 2.5: causal noise ladder and multi-step similarity separation.

Noise ordering gate:
* affine duplicate-vs-healthy batch-mean ordering, B=16, N=150;
* pass threshold is >=95% for both K in {6,10};
* attribution requires agreement between cumulative addition and a
  remove-from-full ablation.

Similarity gate:
* three healthy anchors are fixed; only their three near-duplicate partners
  are optimized;
* LR is selected by a preregistered 200-step stability test;
* full optimization is 2000 steps, B=16, five seeds at full and half noise;
* success requires paired separation >2 CELL64, distance from every other
  anchor >1 CELL64, radius <=0.8, fixed healthy anchors, held-out loss
  decrease on the same 100 batches, and finite values, in >=4/5 seeds.

Statistics: ordering fractions use 150 independently seeded simulated batch
comparisons conditional on one fixed 180-frame empirical residual bank.
Wilson 95% intervals describe Monte-Carlo uncertainty conditional on that
bank; they do not quantify checkpoint/object/sequence generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, rankdata

from dxutils import CELL64_NORM, KEYPOINT_ROOT, OUTPUTS


sys.path.insert(0, str(KEYPOINT_ROOT / "block0"))
from block0 import (  # noqa: E402
    _distinct3,
    _jittered_world,
    _track,
    sample_G,
    subset_consistency,
)


torch.set_default_dtype(torch.float64)

SEED_BASE = 20260704
BATCH_SIZE = 16
N_COMPARISONS = 150
FULL_STEPS = 2000
RESIDUAL_PATH = KEYPOINT_ROOT / "block0" / "outputs" / "empirical_jitter_residuals.npz"
RESIDUALS = np.load(RESIDUAL_PATH)["residuals"].astype(np.float64)
RESIDUALS = RESIDUALS - RESIDUALS.mean(axis=0, keepdims=True)

CUMULATIVE = (
    "iid_homogeneous",
    "iid_heterogeneous",
    "iid_anisotropic",
    "ar1_independent",
    "joint_gaussian",
    "empirical",
)
REMOVALS = (
    "emp_equalized",
    "emp_temporal_shuffled",
    "emp_isotropized",
    "emp_gaussianized",
    "emp_crosschannel_shuffled",
)
ALL_NOISE = CUMULATIVE + REMOVALS


def hashseed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % 99991


def _write_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    radius = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    )
    return centre - radius, centre + radius


@dataclass
class NoiseBank:
    residuals: np.ndarray

    def __post_init__(self) -> None:
        self.frames, self.channels, self.axes = self.residuals.shape
        self.sigma_axis = self.residuals.std(axis=0, ddof=0)
        self.sigma_channel = np.sqrt(np.mean(self.sigma_axis**2, axis=1))
        self.global_sigma = float(np.mean(self.sigma_channel))
        self.rho = np.zeros_like(self.sigma_axis)
        for channel in range(self.channels):
            for axis in range(self.axes):
                first = self.residuals[:-1, channel, axis]
                second = self.residuals[1:, channel, axis]
                self.rho[channel, axis] = np.corrcoef(first, second)[0, 1]
        self.rho = np.nan_to_num(self.rho).clip(-0.95, 0.95)

        channel_scale = self.global_sigma / np.maximum(self.sigma_channel, 1e-12)
        self.equalized = self.residuals * channel_scale[None, :, None]
        self.isotropized = self._make_isotropic()
        self.gaussianized = self._make_gaussian_marginals()

        windows = np.stack(
            [self.residuals[start : start + 3].reshape(-1) for start in range(self.frames - 2)]
        )
        self.window_mean = windows.mean(axis=0)
        covariance = np.cov(windows, rowvar=False, ddof=1)
        # Mild shrinkage makes the 60-D covariance stable with 178 windows.
        diagonal = np.diag(np.diag(covariance))
        covariance = 0.9 * covariance + 0.1 * diagonal
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(np.median(np.diag(covariance))) * 1e-6, 1e-14)
        eigenvalues = np.clip(eigenvalues, floor, None)
        # V @ diag(sqrt(lambda)) without a BLAS matmul. The local NumPy/BLAS
        # build emits spurious overflow warnings for the equivalent tiny
        # 60x60 matmul even though its result is finite.
        self.window_factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
        assert np.isfinite(self.window_factor).all()

    def _make_isotropic(self) -> np.ndarray:
        output = np.empty_like(self.residuals)
        for channel in range(self.channels):
            current = self.residuals[:, channel]
            covariance = np.cov(current, rowvar=False, ddof=0)
            values, vectors = np.linalg.eigh(covariance)
            target = math.sqrt(max(float(values.mean()), 1e-14))
            transform = vectors @ np.diag(target / np.sqrt(np.clip(values, 1e-14, None))) @ vectors.T
            output[:, channel] = current @ transform.T
        return output

    def _make_gaussian_marginals(self) -> np.ndarray:
        output = np.empty_like(self.residuals)
        for channel in range(self.channels):
            for axis in range(self.axes):
                values = self.residuals[:, channel, axis]
                quantiles = (rankdata(values, method="average") - 0.5) / len(values)
                transformed = norm.ppf(quantiles)
                output[:, channel, axis] = transformed * self.sigma_axis[channel, axis]
        return output

    def _channels(self, keypoints: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.channels, size=keypoints, replace=keypoints > self.channels)

    def _window(self, source: np.ndarray, selected: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        start = int(rng.integers(0, self.frames - 2))
        return source[start : start + 3][:, selected, :].copy()

    def sample(self, kind: str, keypoints: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        selected = self._channels(keypoints, rng)
        if kind == "iid_homogeneous":
            result = rng.standard_normal((3, keypoints, 2)) * self.global_sigma
        elif kind == "iid_heterogeneous":
            sigma = self.sigma_channel[selected]
            result = rng.standard_normal((3, keypoints, 2)) * sigma[None, :, None]
        elif kind == "iid_anisotropic":
            result = rng.standard_normal((3, keypoints, 2)) * self.sigma_axis[selected][None]
        elif kind == "ar1_independent":
            sigma = self.sigma_axis[selected]
            rho = self.rho[selected]
            result = np.empty((3, keypoints, 2), dtype=np.float64)
            result[0] = rng.standard_normal((keypoints, 2)) * sigma
            innovation = sigma * np.sqrt(np.maximum(1.0 - rho**2, 0.0))
            for frame in (1, 2):
                result[frame] = rho * result[frame - 1] + rng.standard_normal((keypoints, 2)) * innovation
        elif kind == "joint_gaussian":
            full = self.window_mean + np.einsum(
                "ij,j->i",
                self.window_factor,
                rng.standard_normal(self.window_factor.shape[1]),
            )
            assert np.isfinite(full).all()
            result = full.reshape(3, self.channels, 2)[:, selected]
        elif kind == "empirical":
            result = self._window(self.residuals, selected, rng)
        elif kind == "emp_equalized":
            result = self._window(self.equalized, selected, rng)
        elif kind == "emp_temporal_shuffled":
            frames = rng.integers(0, self.frames, size=3)
            result = self.residuals[frames][:, selected, :].copy()
        elif kind == "emp_isotropized":
            result = self._window(self.isotropized, selected, rng)
        elif kind == "emp_gaussianized":
            result = self._window(self.gaussianized, selected, rng)
        elif kind == "emp_crosschannel_shuffled":
            result = np.empty((3, keypoints, 2), dtype=np.float64)
            for output_channel, source_channel in enumerate(selected):
                start = int(rng.integers(0, self.frames - 2))
                result[:, output_channel] = self.residuals[start : start + 3, source_channel]
        else:
            raise ValueError(kind)
        return scale * result


BANK = NoiseBank(RESIDUALS)


def noisy_world(
    transform_a: torch.Tensor,
    transform_t: torch.Tensor,
    world: str,
    keypoints: int,
    generator: torch.Generator,
    rng: np.random.Generator,
    noise_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = _jittered_world(transform_a, transform_t, world, keypoints, generator, 0.0)
    noise = torch.tensor(BANK.sample(noise_kind, keypoints, rng))
    return tuple(point + noise[index] for index, point in enumerate(points))


def ordering_cell(
    noise_kind: str,
    keypoints: int,
    *,
    n_comparisons: int = N_COMPARISONS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    wins = 0
    for comparison in range(n_comparisons):
        rng = np.random.default_rng(
            SEED_BASE + 7 * comparison + hashseed("ordering", noise_kind, keypoints)
        )
        healthy_total = 0.0
        duplicate_total = 0.0
        for batch_index in range(batch_size):
            index = comparison * batch_size + batch_index
            generator = torch.Generator().manual_seed(
                SEED_BASE + 7919 * index + hashseed(noise_kind, keypoints)
            )
            transform_a, transform_t = sample_G("affine", generator)
            healthy_generator = torch.Generator().manual_seed(SEED_BASE + 104729 * index + 1)
            duplicate_generator = torch.Generator().manual_seed(SEED_BASE + 104729 * index + 2)
            healthy = noisy_world(
                transform_a, transform_t, "healthy", keypoints,
                healthy_generator, rng, noise_kind,
            )
            duplicate = noisy_world(
                transform_a, transform_t, "dup_exact", keypoints,
                duplicate_generator, rng, noise_kind,
            )
            healthy_total += float(subset_consistency(*healthy, "affine")["mean"])
            duplicate_total += float(subset_consistency(*duplicate, "affine")["mean"])
        wins += int(duplicate_total > healthy_total)
    lower, upper = wilson_interval(wins, n_comparisons)
    return {
        "noise": noise_kind,
        "K": keypoints,
        "wins": wins,
        "n_batch_comparisons": n_comparisons,
        "ordering_fraction": wins / n_comparisons,
        "wilson95_low": lower,
        "wilson95_high": upper,
        "batch_size": batch_size,
        "sample_unit": "simulated B-sized batch comparison",
        "uncertainty_scope": "Wilson 95% Monte-Carlo interval conditional on fixed residual bank",
    }


def analyze_attribution(rows: list[dict]) -> dict:
    fraction = {(row["noise"], int(row["K"])): row["ordering_fraction"] for row in rows}

    def passes(kind: str) -> bool:
        return all(fraction[(kind, keypoints)] >= 0.95 for keypoints in (6, 10))

    tests = {
        "heterogeneity": passes("iid_homogeneous") and not passes("iid_heterogeneous") and passes("emp_equalized"),
        "anisotropy": passes("iid_heterogeneous") and not passes("iid_anisotropic") and passes("emp_isotropized"),
        "temporal_dependence": passes("iid_anisotropic") and not passes("ar1_independent") and passes("emp_temporal_shuffled"),
        "joint_crosschannel_or_higher_temporal": passes("ar1_independent") and not passes("joint_gaussian") and passes("emp_crosschannel_shuffled"),
        "non_gaussian_marginals": passes("joint_gaussian") and not passes("empirical") and passes("emp_gaussianized"),
    }
    named = [name for name, value in tests.items() if value]
    return {
        "passes_by_condition": {kind: passes(kind) for kind in ALL_NOISE},
        "agreement_tests": tests,
        "attributed_properties": named,
        "verdict": "identified" if len(named) == 1 else "interaction_or_unresolved",
    }


def run_noise_ladder(
    *, n_comparisons: int = N_COMPARISONS, batch_size: int = BATCH_SIZE
) -> tuple[list[dict], dict]:
    rows = []
    for noise_kind in ALL_NOISE:
        for keypoints in (6, 10):
            result = ordering_cell(
                noise_kind, keypoints,
                n_comparisons=n_comparisons,
                batch_size=batch_size,
            )
            rows.append(result)
            print(
                f"noise {noise_kind:28s} K={keypoints}: "
                f"{100*result['ordering_fraction']:.1f}% "
                f"[{100*result['wilson95_low']:.1f}, {100*result['wilson95_high']:.1f}]"
            )
    return rows, analyze_attribution(rows)


def _initial_anchors(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    healthy = _distinct3(generator).detach()
    directions = torch.rand(3, generator=generator) * (2.0 * math.pi)
    offset = 0.5 * CELL64_NORM * torch.stack(
        [torch.cos(directions), torch.sin(directions)], dim=-1
    )
    duplicate = (healthy + offset).detach()
    return healthy, duplicate


def _training_loss(
    healthy: torch.Tensor,
    duplicate: torch.Tensor,
    *,
    seed: int,
    step: int,
    batch_size: int,
    noise_scale: float,
) -> torch.Tensor:
    losses = []
    rng = np.random.default_rng(seed + 1000003 * step)
    anchors = torch.cat([healthy, duplicate], dim=0)
    for batch_index in range(batch_size):
        generator = torch.Generator().manual_seed(seed + 6007 * (step * batch_size + batch_index))
        transform_a, transform_t = sample_G("similarity", generator)
        points = _track(transform_a, transform_t, anchors)
        noise = torch.tensor(BANK.sample("empirical", 6, rng, scale=noise_scale))
        noisy = tuple(point + noise[index] for index, point in enumerate(points))
        losses.append(subset_consistency(*noisy, "similarity")["mean"])
    return torch.stack(losses).mean()


def _heldout_loss(
    healthy: torch.Tensor,
    duplicate: torch.Tensor,
    *,
    seed: int,
    noise_scale: float,
    n_batches: int,
    batch_size: int,
) -> float:
    values = []
    anchors = torch.cat([healthy, duplicate], dim=0)
    rng = np.random.default_rng(seed + 888888)
    with torch.no_grad():
        for batch in range(n_batches):
            batch_losses = []
            for item in range(batch_size):
                generator = torch.Generator().manual_seed(seed + 17011 * (batch * batch_size + item))
                transform_a, transform_t = sample_G("similarity", generator)
                points = _track(transform_a, transform_t, anchors)
                noise = torch.tensor(BANK.sample("empirical", 6, rng, scale=noise_scale))
                noisy = tuple(point + noise[index] for index, point in enumerate(points))
                batch_losses.append(subset_consistency(*noisy, "similarity")["mean"])
            values.append(float(torch.stack(batch_losses).mean()))
    return float(np.mean(values))


def optimize_similarity(
    *,
    seed: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
    noise_scale: float,
    heldout_batches: int = 100,
) -> tuple[dict, list[dict]]:
    healthy, duplicate_initial = _initial_anchors(seed)
    duplicate = duplicate_initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([duplicate], lr=learning_rate)
    trajectory = []
    finite = True
    losses = []
    max_radius_seen = 0.0
    initial_eval = _heldout_loss(
        healthy, duplicate.detach(), seed=seed + 300000, noise_scale=noise_scale,
        n_batches=heldout_batches, batch_size=batch_size,
    )
    for step in range(steps):
        loss = _training_loss(
            healthy, duplicate, seed=seed, step=step,
            batch_size=batch_size, noise_scale=noise_scale,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_finite = bool(torch.isfinite(duplicate).all() and torch.isfinite(loss))
        finite &= current_finite
        radius = float(duplicate.detach().norm(dim=-1).max())
        max_radius_seen = max(max_radius_seen, radius)
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            trajectory.append(
                {
                    "seed": seed,
                    "noise_scale": noise_scale,
                    "learning_rate": learning_rate,
                    "step": step + 1,
                    "training_loss": float(loss.detach()),
                    "max_duplicate_radius": radius,
                    "finite": current_finite,
                }
            )
        if not current_finite:
            break

    duplicate_final = duplicate.detach()
    final_eval = _heldout_loss(
        healthy, duplicate_final, seed=seed + 300000, noise_scale=noise_scale,
        n_batches=heldout_batches, batch_size=batch_size,
    ) if finite else float("nan")
    paired_separation = torch.linalg.norm(duplicate_final - healthy, dim=-1) / CELL64_NORM
    all_anchors = torch.cat([healthy, duplicate_final], dim=0)
    distance = torch.cdist(duplicate_final, all_anchors) / CELL64_NORM
    for duplicate_index in range(3):
        distance[duplicate_index, duplicate_index + 3] = float("inf")
    other_distance = float(distance.min())
    max_radius = float(all_anchors.norm(dim=-1).max())
    success_conditions = {
        "paired_separation_gt_2": float(paired_separation.min()) > 2.0,
        "all_other_distances_gt_1": other_distance > 1.0,
        "radius_le_0_8": max_radius <= 0.8,
        "healthy_displacement_lt_1": True,  # fixed by construction
        "heldout_loss_decreased": final_eval < initial_eval,
        "finite": finite,
    }
    result = {
        "seed": seed,
        "noise_scale": noise_scale,
        "learning_rate": learning_rate,
        "steps_requested": steps,
        "steps_completed": len(losses),
        "initial_heldout_loss": initial_eval,
        "final_heldout_loss": final_eval,
        "heldout_loss_ratio": final_eval / max(initial_eval, 1e-15),
        "paired_separation_min_cells64": float(paired_separation.min()),
        "other_distance_min_cells64": other_distance,
        "max_anchor_radius": max_radius,
        "max_duplicate_radius_seen": max_radius_seen,
        "healthy_displacement_max_cells64": 0.0,
        **success_conditions,
        "success": all(success_conditions.values()),
        "sample_unit": "optimization seed",
        "heldout_evaluation": f"same fixed {heldout_batches}x{batch_size} simulated triplets before/after",
    }
    return result, trajectory


def learning_rate_precheck(
    *, steps: int = 200, batch_size: int = BATCH_SIZE
) -> tuple[float, list[dict]]:
    rows = []
    for learning_rate in (0.02, 0.005, 0.001):
        result, trajectory = optimize_similarity(
            seed=SEED_BASE,
            learning_rate=learning_rate,
            steps=steps,
            batch_size=batch_size,
            noise_scale=1.0,
            heldout_batches=10,
        )
        losses = np.array([row["training_loss"] for row in trajectory], dtype=float)
        stable = bool(
            result["finite"]
            and result["max_duplicate_radius_seen"] <= 1.0
            and len(losses) >= 2
            and np.median(losses[len(losses)//2:]) <= np.median(losses[: max(1, len(losses)//2)])
        )
        rows.append(
            {
                "learning_rate": learning_rate,
                "stable": stable,
                "max_duplicate_radius_seen": result["max_duplicate_radius_seen"],
                "heldout_loss_ratio": result["heldout_loss_ratio"],
                "finite": result["finite"],
            }
        )
        print(f"LR {learning_rate}: stable={stable}, radius={result['max_duplicate_radius_seen']:.3f}")
    stable_rates = [row["learning_rate"] for row in rows if row["stable"]]
    if not stable_rates:
        raise RuntimeError("no stable learning rate; stop before full similarity run")
    return max(stable_rates), rows


def run_similarity(
    *, steps: int = FULL_STEPS, batch_size: int = BATCH_SIZE, seeds: int = 5
) -> tuple[list[dict], list[dict], dict]:
    learning_rate, precheck = learning_rate_precheck(batch_size=batch_size)
    results, trajectories = [], []
    for noise_scale in (1.0, 0.5):
        for seed_index in range(seeds):
            seed = SEED_BASE + 1009 * seed_index
            result, trajectory = optimize_similarity(
                seed=seed,
                learning_rate=learning_rate,
                steps=steps,
                batch_size=batch_size,
                noise_scale=noise_scale,
                heldout_batches=100,
            )
            results.append(result)
            trajectories.extend(trajectory)
            print(
                f"similarity scale={noise_scale} seed={seed}: success={result['success']} "
                f"paired={result['paired_separation_min_cells64']:.2f} "
                f"other={result['other_distance_min_cells64']:.2f} "
                f"loss_ratio={result['heldout_loss_ratio']:.3f}"
            )
    verdict = {}
    for noise_scale in (1.0, 0.5):
        selected = [row for row in results if row["noise_scale"] == noise_scale]
        successes = sum(bool(row["success"]) for row in selected)
        verdict[str(noise_scale)] = {
            "successes": successes,
            "n_seeds": len(selected),
            "pass": successes >= 4,
        }
    return results, trajectories, {
        "selected_learning_rate": learning_rate,
        "precheck": precheck,
        "verdict_by_noise_scale": verdict,
        "full_noise_gate_pass": verdict["1.0"]["pass"],
    }


def benchmark() -> dict:
    start = time.perf_counter()
    run_noise_ladder(n_comparisons=3, batch_size=4)
    noise_seconds = time.perf_counter() - start
    noise_projection = noise_seconds * (N_COMPARISONS / 3.0) * (BATCH_SIZE / 4.0)

    start = time.perf_counter()
    optimize_similarity(
        seed=SEED_BASE,
        learning_rate=0.001,
        steps=25,
        batch_size=4,
        noise_scale=1.0,
        heldout_batches=3,
    )
    similarity_seconds = time.perf_counter() - start
    # 600 LR-precheck steps + 10*2000 full steps, all at B=16, plus heldout.
    similarity_projection = similarity_seconds * ((600 + 10 * FULL_STEPS) / 25.0) * (BATCH_SIZE / 4.0)
    result = {
        "noise_benchmark_seconds": noise_seconds,
        "noise_projected_full_seconds": noise_projection,
        "similarity_benchmark_seconds": similarity_seconds,
        "similarity_projected_full_seconds_conservative": similarity_projection,
        "combined_projected_seconds": noise_projection + similarity_projection,
    }
    (OUTPUTS / "day25_runtime_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("benchmark", "noise", "precheck", "similarity", "all"),
        default="benchmark",
    )
    parser.add_argument(
        "--residual-bank",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing a (frames,channels,2) 'residuals' array. "
            "Used by the similarity run after the filter-artifact control."
        ),
    )
    args = parser.parse_args()
    if args.residual_bank is not None:
        global BANK
        loaded = np.load(args.residual_bank)["residuals"].astype(np.float64)
        loaded = loaded - loaded.mean(axis=0, keepdims=True)
        BANK = NoiseBank(loaded)
        print(f"using residual bank: {args.residual_bank}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        benchmark()
        return
    if args.mode == "precheck":
        learning_rate, rows = learning_rate_precheck()
        _write_csv(OUTPUTS / "day25_similarity_lr_precheck.csv", rows)
        (OUTPUTS / "day25_similarity_lr_precheck.json").write_text(
            json.dumps({"selected_learning_rate": learning_rate, "rows": rows}, indent=2)
        )
        print(f"selected learning rate: {learning_rate}")
        return

    metadata = {
        "ordering_statistic": "fraction of B=16 simulated batch comparisons with duplicate loss > healthy loss",
        "ordering_n": N_COMPARISONS,
        "ordering_interval": "Wilson 95% interval, conditional Monte-Carlo uncertainty",
        "residual_sample_unit": "one fixed Task-80 180-frame orbit",
        "similarity_sample_unit": "one deterministic optimization seed",
        "similarity_n": "5 seeds per noise scale",
        "similarity_uncertainty": "no error bars or inferential test; pass count is a preregistered descriptive replication gate",
        "generalization": "descriptive/mechanistic; not population inference",
    }
    (OUTPUTS / "day25_statistical_metadata.json").write_text(json.dumps(metadata, indent=2))

    summary_path = OUTPUTS / "day25_summary.json"
    combined = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if args.mode in ("noise", "all"):
        rows, attribution = run_noise_ladder()
        _write_csv(OUTPUTS / "day25_noise_ladder.csv", rows)
        (OUTPUTS / "day25_noise_attribution.json").write_text(json.dumps(attribution, indent=2))
        combined["noise_attribution"] = attribution
    if args.mode in ("similarity", "all"):
        results, trajectories, verdict = run_similarity()
        _write_csv(OUTPUTS / "day25_similarity_results.csv", results)
        _write_csv(OUTPUTS / "day25_similarity_trajectories.csv", trajectories)
        (OUTPUTS / "day25_similarity_verdict.json").write_text(json.dumps(verdict, indent=2))
        combined["similarity"] = verdict
    summary_path.write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
