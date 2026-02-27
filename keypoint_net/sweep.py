"""
Hyperparameter sweep orchestrator for Phase A.

Two-stage sweep:
  Stage 1: Train all configs + collect metrics only (eval_rollout_viz --metrics_only)
  Stage 2: Generate visualizations for promoted configs only

Self-contained. Deleting this file leaves the codebase untouched.

Usage:
    # Full sweep (stage 1 + stage 2)
    python keypoint_net/sweep.py --data_root /path/to/data --object engineers_hammer_vray

    # Stage 1 only
    python keypoint_net/sweep.py ... --stage 1

    # Collect from existing runs (no training)
    python keypoint_net/sweep.py --collect_only --runs_dir keypoint_net/runs

    # Preview
    python keypoint_net/sweep.py --dry_run --data_root /path --object hammer
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_GRID = {
    "lambda_ent":    [0.0, 0.005, 0.01, 0.05, 0.1],
    "lambda_act":    [0.0, 0.1, 0.5, 1.0],
    "lambda_smooth": [0.0, 0.001, 0.01],
}

# Fixed across all configs
FIXED_PARAMS = {
    "lambda_disp": 0.1,
    "sigma": 0.1,
    "frame_skip": 3,
    "num_keypoints": 10,
    "epochs": 1000,
    "seed": 42,
    "batch_size": 16,
    "lr": 1e-4,
    "num_action_classes": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Promotion gates
# ─────────────────────────────────────────────────────────────────────────────

MANDATORY_GATES = {
    "identity_ratio": (">", 2.0),
    "on_foreground_pct":  (">", 0.5),
}

SOFT_GATES = {
    "val_act_acc":    (">", 0.7),
    "active_kp_frac": (">", 0.3),
    "k10_k1_ratio":   ("<", 5.0),
}


def _passes_gate(value, op, threshold) -> bool:
    if value is None:
        return False
    if op == ">":
        return value > threshold
    elif op == "<":
        return value < threshold
    return False


def evaluate_promotion(metrics: dict) -> str:
    """
    Determine promotion tier for a config.

    Returns: 'full', 'soft', or 'none'
    """
    # Must pass ALL mandatory gates
    for key, (op, thresh) in MANDATORY_GATES.items():
        if not _passes_gate(metrics.get(key), op, thresh):
            return "none"

    # Count soft gates passed
    soft_passed = sum(
        1 for key, (op, thresh) in SOFT_GATES.items()
        if _passes_gate(metrics.get(key), op, thresh)
    )

    if soft_passed >= len(SOFT_GATES):
        return "full"
    elif soft_passed >= len(SOFT_GATES) - 1:  # pass n-1 of n
        return "soft"
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# Grid generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_configs() -> list[dict]:
    """Generate all sweep configurations from the grid."""
    keys = sorted(SWEEP_GRID.keys())
    values = [SWEEP_GRID[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED_PARAMS)
        configs.append(cfg)
    return configs


def config_tag(cfg: dict) -> str:
    """Short string identifying a config, for matching and display."""
    return (
        f"ent{cfg['lambda_ent']}_act{cfg['lambda_act']}_sm{cfg['lambda_smooth']}"
    )


def parse_filter(filter_str: str) -> dict:
    """Parse filter like 'lambda_act=0.1,lambda_ent=0.01' into a dict."""
    filters = {}
    for part in filter_str.split(","):
        key, val = part.strip().split("=")
        filters[key.strip()] = float(val.strip())
    return filters


def apply_filter(configs: list[dict], filters: dict) -> list[dict]:
    """Keep only configs matching all filter key=value pairs."""
    result = []
    for cfg in configs:
        if all(cfg.get(k) == v for k, v in filters.items()):
            result.append(cfg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Run matching (for --collect_only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_run_config(run_dir: Path) -> dict | None:
    """Load config.json from a run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def _configs_match(run_cfg: dict, target: dict) -> bool:
    """
    Check if a run config matches a target sweep config.

    Compares all swept params AND key fixed params (num_keypoints,
    epochs, seed, frame_skip) to avoid matching the wrong run.
    """
    # Swept params
    for key in SWEEP_GRID:
        if abs(run_cfg.get(key, -999) - target[key]) > 1e-8:
            return False
    # Fixed params that affect results
    for key in ("num_keypoints", "epochs", "seed", "frame_skip"):
        if run_cfg.get(key) != target.get(key):
            return False
    return True


def match_existing_runs(
    runs_dir: Path, target_configs: list[dict], object_name: str
) -> dict[str, Path]:
    """
    Match target configs to existing run directories.

    For each target config, finds ALL matching runs, then picks the
    most recent one that has a best_model.pt (i.e., completed training).
    Falls back to most recent match if none have best_model.pt.

    Returns dict mapping config_tag -> run_dir Path.
    """
    matches = {}
    if not runs_dir.exists():
        return matches

    # Collect all run configs
    all_runs = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        run_cfg = _load_run_config(run_path)
        if run_cfg is None:
            continue
        if run_cfg.get("object") != object_name:
            continue
        all_runs.append((run_path, run_cfg))

    # For each target, find the best matching run
    for target in target_configs:
        tag = config_tag(target)
        candidates = []
        for run_path, run_cfg in all_runs:
            if _configs_match(run_cfg, target):
                has_checkpoint = (run_path / "best_model.pt").exists()
                candidates.append((run_path, has_checkpoint))

        if not candidates:
            continue

        # Prefer completed runs (have best_model.pt), then most recent
        completed = [(p, c) for p, c in candidates if c]
        if completed:
            # Most recent by mtime
            best = max(completed, key=lambda x: x[0].stat().st_mtime)
        else:
            best = max(candidates, key=lambda x: x[0].stat().st_mtime)

        matches[tag] = best[0]

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Metric collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_metrics(run_dir: Path) -> dict:
    """
    Collect all scorecard metrics from a run's JSON artifacts.

    Returns flat dict of metric_name -> value (or None if missing).
    """
    metrics = {}

    # --- ablations.json ---
    ablation_path = run_dir / "ablations" / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            abl = json.load(f)
        baseline = abl.get("baseline", None)
        identity = abl.get("identity_operator", None)
        if baseline is not None and baseline > 0:
            metrics["baseline_MSE"] = baseline
            if identity is not None:
                metrics["identity_ratio"] = identity / baseline
        metrics["identity_MSE"] = identity

    # --- compositionality_results.json ---
    # Try compositionality_v2, then compositionality_sem, then compositionality
    comp_path = run_dir / "compositionality_v2" / "compositionality_results.json"
    if not comp_path.exists():
        comp_path = run_dir / "compositionality_sem" / "compositionality_results.json"
    if not comp_path.exists():
        comp_path = run_dir / "compositionality" / "compositionality_results.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        errors = comp.get("errors", {})
        k1 = errors.get("1", errors.get(1, {}))
        k10 = errors.get("10", errors.get(10, {}))
        k1_mean = k1.get("mean") if isinstance(k1, dict) else None
        k10_mean = k10.get("mean") if isinstance(k10, dict) else None
        if k1_mean and k10_mean and k1_mean > 0:
            metrics["k10_k1_ratio"] = k10_mean / k1_mean

    # --- history.json (best val_act_acc) ---
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        # Find best val_act_acc
        best_acc = 0.0
        for entry in hist:
            acc = entry.get("val_act_acc", 0.0)
            if acc is not None and acc > best_acc:
                best_acc = acc
        metrics["val_act_acc"] = best_acc

    # --- rollout_metrics.json ---
    rollout_path = run_dir / "rollout" / "rollout_metrics.json"
    if rollout_path.exists():
        with open(rollout_path) as f:
            rm = json.load(f)

        # Localization (backward compat: old artifacts used "on_object_pct")
        loc = rm.get("localization", {})
        metrics["on_foreground_pct"] = loc.get("on_foreground_pct") or loc.get("on_object_pct")

        # Participation
        part = rm.get("participation", {})
        metrics["active_kp_frac"] = part.get("active_kp_frac")
        metrics["top1_energy_frac"] = part.get("top1_energy_frac")

        # Stability
        stab = rm.get("stability", {})
        metrics["mean_speed"] = stab.get("mean_speed")
        metrics["mean_accel"] = stab.get("mean_accel")

        # Trajectory geometry
        tg = rm.get("trajectory_geometry", {})
        metrics["dim2_frac"] = tg.get("dim2_frac")

        # Operator spectrum
        spec = rm.get("operator_spectrum", {})
        metrics["sv_min"] = spec.get("sv_min")
        metrics["sv_max"] = spec.get("sv_max")
        metrics["spectral_radius"] = spec.get("spectral_radius")
        metrics["orth_err"] = spec.get("orth_err")

        # Inverse
        metrics["inverse_k1_MSE"] = rm.get("inverse", {}).get("k1_mse")
        metrics["inverse_condition_number"] = rm.get("inverse_condition_number")

        # Forward rollout
        fwd = rm.get("forward", {})
        metrics["forward_k1_mse"] = fwd.get("k1_mse")
        metrics["forward_k10_mse"] = fwd.get("k10_mse")

    # --- operator_metrics.json ---
    op_path = run_dir / "visualizations" / "operator_metrics.json"
    if not op_path.exists():
        op_path = run_dir / "visualizations_v2" / "operator_metrics.json"
    if op_path.exists():
        with open(op_path) as f:
            op = json.load(f)
        metrics["W_minus_I_fro"] = op.get("fro_W_minus_I")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training + eval execution
# ─────────────────────────────────────────────────────────────────────────────

def _find_run_dir(output_dir: Path, cfg: dict, object_name: str) -> Path | None:
    """
    Check if a run matching this config already exists.

    Uses config_tag matching against existing run configs.
    """
    tag = config_tag(cfg)
    matched = match_existing_runs(output_dir, [cfg], object_name)
    return matched.get(tag)


def run_training(
    cfg: dict,
    data_root: str,
    object_name: str,
    output_dir: Path,
) -> Path:
    """
    Run train.py with the given config via subprocess.

    Returns the run directory Path.
    """
    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    cmd = [
        py,
        str(script_dir / "train.py"),
        "--data_root", str(data_root),
        "--object", object_name,
        "--output_dir", str(output_dir),
        "--auto_eval",
        # Swept params
        "--lambda_ent", str(cfg["lambda_ent"]),
        "--lambda_act", str(cfg["lambda_act"]),
        "--lambda_smooth", str(cfg["lambda_smooth"]),
        # Fixed params
        "--lambda_disp", str(cfg["lambda_disp"]),
        "--sigma", str(cfg["sigma"]),
        "--frame_skip", str(cfg["frame_skip"]),
        "--num_keypoints", str(cfg["num_keypoints"]),
        "--epochs", str(cfg["epochs"]),
        "--seed", str(cfg["seed"]),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--num_action_classes", str(cfg["num_action_classes"]),
    ]

    print(f"\n{'='*60}")
    print(f"[Sweep] Training: {config_tag(cfg)}")
    print(f"[Sweep] Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

    # Find the run directory that matches this config
    # (prefer config-matched lookup; fall back to most recent if not found,
    # e.g. if train.py stored config keys differently)
    matched = _find_run_dir(output_dir, cfg, object_name)
    if matched is not None:
        return matched
    return _find_most_recent_run(output_dir, object_name)


def _find_most_recent_run(output_dir: Path, object_name: str) -> Path | None:
    """Find the most recently modified run directory for an object."""
    if not output_dir.exists():
        return None
    candidates = []
    for p in output_dir.iterdir():
        if p.is_dir() and object_name in p.name:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_rollout_eval(
    run_dir: Path,
    frames_dir: str,
    metrics_only: bool = True,
    forward_only: bool = False,
) -> None:
    """Run eval_rollout_viz.py on a run's best checkpoint."""
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        print(f"[Sweep] WARNING: No best_model.pt in {run_dir}, skipping rollout eval")
        return

    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    cmd = [
        py,
        str(script_dir / "eval_rollout_viz.py"),
        "--checkpoint", str(checkpoint),
        "--frames_dir", str(frames_dir),
        "--output_dir", str(run_dir / "rollout"),
    ]
    if metrics_only:
        cmd.append("--metrics_only")
    if forward_only:
        cmd.append("--forward_only")

    mode = "metrics_only" if metrics_only else ("forward_only" if forward_only else "full")
    print(f"[Sweep] Rollout eval: {run_dir.name} ({mode})")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Sweep] WARNING: rollout eval failed for {run_dir.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary output
# ─────────────────────────────────────────────────────────────────────────────

SCORECARD_COLUMNS = [
    # config
    "tag", "lambda_ent", "lambda_act", "lambda_smooth",
    # mandatory gates
    "identity_ratio", "on_foreground_pct",
    # soft gates
    "val_act_acc", "active_kp_frac", "k10_k1_ratio",
    # reported
    "baseline_MSE", "top1_energy_frac", "mean_speed", "mean_accel",
    # diagnostics
    "dim2_frac", "sv_min", "sv_max", "spectral_radius", "orth_err",
    "W_minus_I_fro", "inverse_k1_MSE",
    # promotion
    "promotion_tier",
]


def write_summary_csv(results: list[dict], output_path: Path):
    """Write sweep summary as CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved: {output_path}")


def write_summary_json(results: list[dict], output_path: Path,
                       object_name: str = None, data_root: str = None):
    """Write full sweep results as JSON."""
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "object": object_name,
        "data_root": data_root,
        "n_configs": len(results),
        "grid": SWEEP_GRID,
        "fixed_params": FIXED_PARAMS,
        "mandatory_gates": {k: list(v) for k, v in MANDATORY_GATES.items()},
        "soft_gates": {k: list(v) for k, v in SOFT_GATES.items()},
        "single_seed_caveat": (
            "This sweep uses seed=42 for screening. Top configs must be "
            "re-run with 2-3 seeds before making claims."
        ),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {output_path}")


def plot_summary(results: list[dict], output_path: Path):
    """Visual comparison of sweep results."""
    if not results:
        return

    # Filter to configs with at least identity_ratio
    plotable = [r for r in results if r.get("identity_ratio") is not None]
    if not plotable:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sort by identity_ratio descending
    plotable.sort(key=lambda r: r.get("identity_ratio", 0), reverse=True)
    tags = [r["tag"] for r in plotable]
    x = np.arange(len(tags))

    def _bar(ax, key, title, threshold=None, threshold_label=None):
        vals = [r.get(key) for r in plotable]
        vals = [v if v is not None else 0 for v in vals]
        colors = []
        for r in plotable:
            tier = r.get("promotion_tier", "none")
            if tier == "full":
                colors.append("#2ecc71")
            elif tier == "soft":
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=90, fontsize=6)
        if threshold is not None:
            ax.axhline(threshold, color="black", linestyle="--", alpha=0.5,
                       label=threshold_label or f"threshold={threshold}")
            ax.legend(fontsize=8)

    _bar(axes[0, 0], "identity_ratio", "Identity Ratio (>2.0 = mandatory gate)", 2.0)
    _bar(axes[0, 1], "on_foreground_pct", "On-Foreground % (>0.5 = mandatory gate)", 0.5)
    _bar(axes[0, 2], "val_act_acc", "Action Accuracy (>0.7 = soft gate)", 0.7)
    _bar(axes[1, 0], "active_kp_frac", "Active KP Fraction (>0.3 = soft gate)", 0.3)
    _bar(axes[1, 1], "k10_k1_ratio", "k10/k1 Ratio (<5.0 = soft gate)", 5.0)

    # Legend for promotion tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Full promotion"),
        Patch(facecolor="#f39c12", label="Soft promotion"),
        Patch(facecolor="#e74c3c", label="No promotion"),
    ]
    axes[1, 2].legend(handles=legend_elements, loc="center", fontsize=12)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Promotion Legend")

    fig.suptitle(
        f"Sweep Summary ({len(plotable)} configs)  |  "
        f"Green = full, Orange = soft, Red = none",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase A hyperparameter sweep")
    parser.add_argument("--data_root", type=str, help="Path to dataset root")
    parser.add_argument("--object", type=str, default="engineers_hammer_vray")
    parser.add_argument("--output_dir", type=str, default="keypoint_net/runs",
                        help="Directory for run outputs")
    parser.add_argument("--sweep_output_dir", type=str, default=None,
                        help="Directory for sweep summary (default: output_dir/sweep)")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run only stage 1 or stage 2. Default: both.")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter configs, e.g., 'lambda_act=0.1,lambda_ent=0.01'")
    parser.add_argument("--collect_only", action="store_true",
                        help="Collect metrics from existing runs, no training.")
    parser.add_argument("--runs_dir", type=str, default=None,
                        help="Directory with existing runs (for --collect_only)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configs without running anything.")
    args = parser.parse_args()

    # ── Generate configs ──
    configs = generate_configs()
    if args.filter:
        filters = parse_filter(args.filter)
        configs = apply_filter(configs, filters)
        print(f"Filter applied: {filters} -> {len(configs)} configs")

    print(f"Sweep grid: {len(configs)} total configs")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1:3d}] {config_tag(cfg)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would run {len(configs)} configs. Exiting.")
        return

    # ── Resolve directories ──
    output_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir) if args.runs_dir else output_dir
    sweep_out = Path(args.sweep_output_dir) if args.sweep_output_dir else output_dir / "sweep"
    sweep_out.mkdir(parents=True, exist_ok=True)

    # ── Infer frames directory ──
    if args.data_root:
        data_root = Path(args.data_root)
        frames_dir = data_root / "train" / args.object / "frames" / "a"
        if not frames_dir.exists():
            frames_dir = data_root / "test" / args.object / "frames" / "a"
        frames_dir_str = str(frames_dir)
    else:
        frames_dir_str = None

    # ── Stage 1: Train + collect metrics ──
    all_results = []

    if args.collect_only:
        print(f"\n[Sweep] Collecting from existing runs in {runs_dir}...")
        matched = match_existing_runs(runs_dir, configs, args.object)
        print(f"[Sweep] Matched {len(matched)}/{len(configs)} configs to existing runs")

        for cfg in configs:
            tag = config_tag(cfg)
            run_dir = matched.get(tag)
            row = {"tag": tag, **{k: cfg[k] for k in SWEEP_GRID}}

            if run_dir is None:
                print(f"  [{tag}] No matching run found")
                row["promotion_tier"] = "none"
                all_results.append(row)
                continue

            # Run rollout eval if not already done
            if frames_dir_str and not (run_dir / "rollout" / "rollout_metrics.json").exists():
                run_rollout_eval(run_dir, frames_dir_str, metrics_only=True)

            metrics = collect_metrics(run_dir)
            row.update(metrics)
            row["run_dir"] = str(run_dir)
            row["promotion_tier"] = evaluate_promotion(metrics)
            all_results.append(row)
            print(f"  [{tag}] {row['promotion_tier']}  identity_ratio={metrics.get('identity_ratio', '?')}")

    else:
        # Run stage 1 (or both)
        if not args.data_root:
            parser.error("--data_root is required when not using --collect_only")

        run_stage_1 = args.stage is None or args.stage == 1
        run_stage_2 = args.stage is None or args.stage == 2

        if run_stage_1:
            print(f"\n{'='*60}")
            print(f"[Sweep] STAGE 1: Train + metrics for {len(configs)} configs")
            print(f"{'='*60}")

            for i, cfg in enumerate(configs):
                tag = config_tag(cfg)
                print(f"\n[Sweep] Config {i+1}/{len(configs)}: {tag}")

                # Resume safety: check if run already exists
                existing = _find_run_dir(runs_dir, cfg, args.object)
                if existing and (existing / "best_model.pt").exists():
                    print(f"[Sweep] Found existing run: {existing.name} — skipping training")
                    run_dir = existing
                else:
                    run_dir = run_training(cfg, args.data_root, args.object, output_dir)

                if run_dir is None:
                    print(f"[Sweep] WARNING: Could not find run dir after training {tag}")
                    all_results.append({
                        "tag": tag, **{k: cfg[k] for k in SWEEP_GRID},
                        "promotion_tier": "none",
                    })
                    continue

                # Run rollout eval (metrics only for stage 1)
                if not (run_dir / "rollout" / "rollout_metrics.json").exists():
                    run_rollout_eval(run_dir, frames_dir_str, metrics_only=True)

                metrics = collect_metrics(run_dir)
                row = {"tag": tag, **{k: cfg[k] for k in SWEEP_GRID}}
                row.update(metrics)
                row["run_dir"] = str(run_dir)
                row["promotion_tier"] = evaluate_promotion(metrics)
                all_results.append(row)

        if run_stage_2:
            # ── Stage 2: Full viz for promoted configs ──
            # If all_results is empty (e.g. --stage 2 standalone), load from prior sweep
            if not all_results:
                prior_path = sweep_out / "sweep_results.json"
                if prior_path.exists():
                    with open(prior_path) as f:
                        prior = json.load(f)
                    all_results = prior.get("results", [])
                    # Respect --filter: keep only results whose tag matches a config in the filtered set
                    allowed_tags = {config_tag(c) for c in configs}
                    all_results = [r for r in all_results if r["tag"] in allowed_tags]
                    print(f"[Sweep] Loaded {len(all_results)} results from {prior_path} (filtered to {len(allowed_tags)} tags)")
                else:
                    print(f"[Sweep] No prior sweep_results.json found at {prior_path}.")
                    print(f"[Sweep] Run --stage 1 or --collect_only first.")

            promoted = [r for r in all_results if r.get("promotion_tier") in ("full", "soft")]
            print(f"\n{'='*60}")
            print(f"[Sweep] STAGE 2: Visualizations for {len(promoted)} promoted configs")
            print(f"{'='*60}")

            for row in promoted:
                run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
                if run_dir is None:
                    continue

                tier = row["promotion_tier"]
                if tier == "full":
                    # Full viz (forward + inverse)
                    run_rollout_eval(run_dir, frames_dir_str, metrics_only=False)
                elif tier == "soft":
                    # Forward viz only (no inverse)
                    run_rollout_eval(run_dir, frames_dir_str,
                                     metrics_only=False, forward_only=True)

    # ── Write outputs ──
    print(f"\n{'='*60}")
    print(f"[Sweep] Writing summary ({len(all_results)} configs)")
    print(f"{'='*60}")

    write_summary_json(all_results, sweep_out / "sweep_results.json",
                       object_name=args.object, data_root=args.data_root)
    write_summary_csv(all_results, sweep_out / "sweep_summary.csv")
    plot_summary(all_results, sweep_out / "sweep_summary.png")

    # Print promotion summary
    n_full = sum(1 for r in all_results if r.get("promotion_tier") == "full")
    n_soft = sum(1 for r in all_results if r.get("promotion_tier") == "soft")
    n_none = sum(1 for r in all_results if r.get("promotion_tier") == "none")
    print(f"\nPromotion tiers: {n_full} full, {n_soft} soft, {n_none} none")

    # Write promoted configs
    promoted = [r for r in all_results if r.get("promotion_tier") in ("full", "soft")]
    if promoted:
        promoted_path = sweep_out / "promoted_configs.json"
        with open(promoted_path, "w") as f:
            json.dump(promoted, f, indent=2, default=str)
        print(f"Saved: {promoted_path}")

    print(f"\nAll sweep outputs saved to: {sweep_out}")


if __name__ == "__main__":
    main()
