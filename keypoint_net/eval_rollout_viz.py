"""
Per-run diagnostics: rollout visualization, inverse test, localization,
participation, stability, trajectory geometry, and operator spectrum.

Self-contained evaluation script. Deleting this file leaves the rest of
the codebase untouched.

Usage:
    # Full eval (metrics + visualizations)
    python keypoint_net/eval_rollout_viz.py \
        --checkpoint ./runs/run_dir/best_model.pt \
        --frames_dir /path/to/frames/a \
        --output_dir ./runs/run_dir/rollout

    # Metrics only (for sweep Stage 1)
    python keypoint_net/eval_rollout_viz.py ... --metrics_only
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Local imports (same package)
from visualize import (
    load_model,
    plot_keypoints_on_image,
    keypoints_to_pixels,
    tensor_to_image,
    COLORS,
)
from dataset import SingleObjectDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_all_keypoints(
    model, frames, transform, device: str
) -> torch.Tensor:
    """
    Extract keypoints for every frame in the sequence.

    Args:
        model: PhaseAModel (eval mode)
        frames: list of Path objects (sorted frame files)
        transform: torchvision Transform (PIL -> tensor)
        device: torch device string

    Returns:
        Tensor of shape (T, 2N) with all keypoints.
    """
    all_kp = []
    for frame_path in frames:
        img = Image.open(frame_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p, _ = model.extractor(x)
        all_kp.append(p.cpu())
    return torch.cat(all_kp, dim=0)  # (T, 2N)


# ─────────────────────────────────────────────────────────────────────────────
# A) Forward k-step rollout
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 10]


@torch.no_grad()
def compute_forward_rollout(
    model, all_keypoints: torch.Tensor, start: int, frame_skip: int, device: str
) -> dict:
    """
    Compute W^k predictions vs actual keypoints for each k in K_VALUES.

    Returns dict mapping k -> {predicted: (2N,), actual: (2N,), mse: float}
    Only includes k values where the target frame exists.
    """
    n_frames = all_keypoints.shape[0]
    p_0 = all_keypoints[start : start + 1].to(device)  # (1, 2N)
    results = {}

    for k in K_VALUES:
        target_idx = start + k * frame_skip
        if target_idx >= n_frames:
            warnings.warn(
                f"Skipping forward k={k}: target frame {target_idx} >= {n_frames}"
            )
            continue
        p_hat = model.multi_step_predict(p_0, k)  # (1, 2N)
        p_actual = all_keypoints[target_idx : target_idx + 1]  # (1, 2N)
        mse = float(torch.mean((p_hat.cpu() - p_actual) ** 2))
        results[k] = {
            "predicted": p_hat[0].cpu(),
            "actual": p_actual[0],
            "mse": mse,
            "target_frame_idx": target_idx,
        }
    return results


def plot_forward_rollout(
    model,
    forward_results: dict,
    start: int,
    frames: list,
    transform,
    img_size: int,
    output_path: Path,
):
    """Overlay actual vs predicted keypoints on real frames for each k."""
    valid_ks = sorted(forward_results.keys())
    n_plots = len(valid_ks) + 1  # +1 for start frame
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    N = model.num_keypoints

    # Start frame
    img = Image.open(frames[start]).convert("RGB")
    x = transform(img)
    img_np = tensor_to_image(x)
    p_start = forward_results[valid_ks[0]]["actual"]  # just need start kp for display
    # Actually re-extract start keypoints from the actual start
    # (forward_results stores actual at target, not start)
    # We'll use the first k's predicted to get p_0 indirectly — but cleaner to
    # just overlay a note. For start frame, show actual keypoints at start.
    # Pull from all_keypoints via the predicted computation (p_0).
    # Simpler: just show the frame with a "Start" label.
    axes[0].imshow(img_np)
    axes[0].set_title(f"Start (frame {start})")
    axes[0].axis("off")

    for ax_idx, k in enumerate(valid_ks, start=1):
        res = forward_results[k]
        target_idx = res["target_frame_idx"]

        # Load target frame image
        img = Image.open(frames[target_idx]).convert("RGB")
        x = transform(img)
        img_np = tensor_to_image(x)

        ax = axes[ax_idx]
        ax.imshow(img_np)

        # Actual keypoints (solid circles)
        actual_kp = res["actual"].view(N, 2)
        actual_px = keypoints_to_pixels(actual_kp, img_size)
        for i, (px, py) in enumerate(actual_px):
            color = COLORS[i % len(COLORS)]
            ax.plot(px, py, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2)

        # Predicted keypoints (X markers)
        pred_kp = res["predicted"].view(N, 2)
        pred_px = keypoints_to_pixels(pred_kp, img_size)
        for i, (px, py) in enumerate(pred_px):
            color = COLORS[i % len(COLORS)]
            ax.plot(px, py, "x", color=color, markersize=9, markeredgewidth=2.5)

        ax.set_title(f"k={k}  MSE={res['mse']:.5f}")
        ax.axis("off")

    fig.suptitle(
        "Forward Rollout: ● actual  ✕ W^k predicted",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# B) W^{-1} reversibility diagnostic
# ─────────────────────────────────────────────────────────────────────────────

COND_THRESHOLD = 100.0


@torch.no_grad()
def compute_inverse_rollout(
    model, all_keypoints: torch.Tensor, start: int, frame_skip: int, device: str
) -> dict:
    """
    Apply W^{-1} iteratively to predict backward frames.

    Uses torch.linalg.solve for numerical stability.
    Returns dict with per-k results + condition number + reliability flag.
    """
    W = model.operator.W.detach().cpu().double()  # (2N, 2N) in float64
    b = model.operator.b.detach().cpu().double()  # (2N,)

    # Condition number in float64
    try:
        cond = float(torch.linalg.cond(W).item())
    except Exception:
        cond = float("inf")

    reliable = cond < COND_THRESHOLD

    results = {
        "condition_number": cond,
        "reliable": reliable,
        "per_k": {},
    }

    if not reliable:
        warnings.warn(
            f"cond(W) = {cond:.1f} > {COND_THRESHOLD}. "
            f"Inverse rollout is numerically unreliable — skipping."
        )
        return results

    p_current = all_keypoints[start].double()  # (2N,)

    for k in K_VALUES:
        target_idx = start - k * frame_skip
        if target_idx < 0:
            warnings.warn(
                f"Skipping inverse k={k}: target frame {target_idx} < 0"
            )
            continue

        # Apply inverse k times from scratch (not accumulating from previous k)
        p_inv = all_keypoints[start].double()
        for _ in range(k):
            # Forward: p_next = W @ p + b  =>  Inverse: p = W^{-1} (p_next - b)
            rhs = p_inv - b
            p_inv = torch.linalg.solve(W, rhs)

        p_actual = all_keypoints[target_idx].double()
        mse = float(torch.mean((p_inv - p_actual) ** 2))

        results["per_k"][k] = {
            "predicted": p_inv.float().cpu(),
            "actual": p_actual.float().cpu(),
            "mse": mse,
            "target_frame_idx": target_idx,
        }

    return results


def plot_inverse_rollout(
    model,
    inverse_results: dict,
    start: int,
    frames: list,
    transform,
    img_size: int,
    output_path: Path,
):
    """Overlay actual vs W^{-k} predicted keypoints on real frames."""
    per_k = inverse_results["per_k"]
    if not per_k:
        print("No valid inverse rollout steps — skipping inverse visualization.")
        return

    valid_ks = sorted(per_k.keys())
    n_plots = len(valid_ks) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    N = model.num_keypoints

    # Start frame
    img = Image.open(frames[start]).convert("RGB")
    x = transform(img)
    img_np = tensor_to_image(x)
    axes[0].imshow(img_np)
    axes[0].set_title(f"Start (frame {start})")
    axes[0].axis("off")

    for ax_idx, k in enumerate(valid_ks, start=1):
        res = per_k[k]
        target_idx = res["target_frame_idx"]

        img = Image.open(frames[target_idx]).convert("RGB")
        x = transform(img)
        img_np = tensor_to_image(x)

        ax = axes[ax_idx]
        ax.imshow(img_np)

        # Actual keypoints
        actual_kp = res["actual"].view(N, 2)
        actual_px = keypoints_to_pixels(actual_kp, img_size)
        for i, (px, py) in enumerate(actual_px):
            color = COLORS[i % len(COLORS)]
            ax.plot(px, py, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2)

        # Predicted (inverse)
        pred_kp = res["predicted"].view(N, 2)
        pred_px = keypoints_to_pixels(pred_kp, img_size)
        for i, (px, py) in enumerate(pred_px):
            color = COLORS[i % len(COLORS)]
            ax.plot(px, py, "x", color=color, markersize=9, markeredgewidth=2.5)

        ax.set_title(f"k=-{k}  MSE={res['mse']:.5f}")
        ax.axis("off")

    cond = inverse_results["condition_number"]
    fig.suptitle(
        f"Inverse Rollout (stress test): ● actual  ✕ W^{{-k}} predicted  |  cond(W)={cond:.1f}",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# C) Localization metric
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_background_color(img_np: np.ndarray) -> np.ndarray:
    """
    Estimate background color from the corners of the image.

    TDW renders have a uniform background. Sample corner regions
    and take the median — robust to small object overlap at edges.
    """
    h, w = img_np.shape[:2]
    margin = max(1, min(h, w) // 8)
    corners = np.concatenate([
        img_np[:margin, :margin].reshape(-1, 3),        # top-left
        img_np[:margin, -margin:].reshape(-1, 3),       # top-right
        img_np[-margin:, :margin].reshape(-1, 3),       # bottom-left
        img_np[-margin:, -margin:].reshape(-1, 3),      # bottom-right
    ], axis=0)
    return np.median(corners, axis=0)  # (3,)


def _background_subtraction_mask(
    frame_path: Path, img_size: int, bg_threshold: float = 30.0
) -> np.ndarray:
    """
    Create object mask via background subtraction.

    TDW renders have uniform backgrounds. Pixels that differ from
    the estimated background color by more than bg_threshold (in L2
    across RGB channels) are considered "object".

    Returns:
        (H, W) boolean mask at img_size resolution.
    """
    img = Image.open(frame_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)  # (H, W, 3)

    bg_color = _estimate_background_color(img_np)  # (3,)
    diff = np.linalg.norm(img_np - bg_color[None, None, :], axis=2)  # (H, W)
    mask = diff > bg_threshold
    return mask


def compute_localization(
    all_keypoints: torch.Tensor,
    frames: list,
    img_size: int,
    n_kp: int,
    bg_threshold: float = 30.0,
) -> dict:
    """
    Compute localization metric: % keypoints on foreground (non-background).

    Uses background subtraction on TDW renders (uniform background).
    Works with any TDW dataset (yaw-only, 2D affine, etc.) without
    requiring metadata or base_rgba.png.

    NOTE: This is 'on foreground' not strict 'on object surface'.
    Shadows and anti-aliased edges near the object count as foreground.
    Useful for screening (catches keypoints in empty background) but
    may be slightly optimistic compared to a true object mask.
    """
    T = all_keypoints.shape[0]
    n_eval = min(T, len(frames))
    per_frame_pct = []

    for t in range(n_eval):
        mask = _background_subtraction_mask(frames[t], img_size, bg_threshold)
        kp = all_keypoints[t].view(n_kp, 2).numpy()  # (N, 2) in [-1, 1]
        kp_px = ((kp + 1) / 2) * (img_size - 1)  # (N, 2) pixel coords

        inside_count = 0
        for i in range(n_kp):
            px_x = int(round(kp_px[i, 0]))
            px_y = int(round(kp_px[i, 1]))
            if 0 <= px_x < img_size and 0 <= px_y < img_size:
                if mask[px_y, px_x]:  # y=row, x=col
                    inside_count += 1
        per_frame_pct.append(inside_count / n_kp)

    return {
        "on_foreground_pct": float(np.mean(per_frame_pct)),
        "method": "background_subtraction",
        "bg_threshold": bg_threshold,
        "per_frame_pct": per_frame_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# D) Trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_participation(
    all_keypoints: torch.Tensor, n_kp: int
) -> dict:
    """
    Participation metrics: do multiple keypoints contribute, or does one dominate?

    - per_kp_energy: E_i = mean_t(||delta_p_i(t)||^2) for each keypoint
    - active_kp_frac: fraction with energy > 10% of max energy
    - top1_energy_frac: max single-keypoint energy / total energy
    """
    T = all_keypoints.shape[0]
    kp = all_keypoints.view(T, n_kp, 2).numpy()  # (T, N, 2)

    # Frame-to-frame deltas
    delta = np.diff(kp, axis=0)  # (T-1, N, 2)
    # Per-keypoint squared displacement per step
    sq_disp = np.sum(delta ** 2, axis=2)  # (T-1, N)
    # Mean energy per keypoint
    per_kp_energy = np.mean(sq_disp, axis=0).tolist()  # (N,)

    total_energy = sum(per_kp_energy)
    max_energy = max(per_kp_energy) if per_kp_energy else 0.0

    # Active: energy > 10% of max
    threshold = 0.1 * max_energy if max_energy > 0 else 0.0
    active = sum(1 for e in per_kp_energy if e > threshold)
    active_frac = active / n_kp if n_kp > 0 else 0.0

    # Top-1 fraction
    top1_frac = max_energy / total_energy if total_energy > 0 else 1.0

    return {
        "active_kp_frac": float(active_frac),
        "top1_energy_frac": float(top1_frac),
        "per_kp_energy": per_kp_energy,
    }


def compute_stability(
    all_keypoints: torch.Tensor, n_kp: int
) -> dict:
    """
    Stability metrics: speed, acceleration, total motion energy.

    Smoothness alone can be gamed by static keypoints, so we also
    report total_motion_energy as a motion floor.
    """
    T = all_keypoints.shape[0]
    kp = all_keypoints.view(T, n_kp, 2).numpy()  # (T, N, 2)

    # Velocity: first difference
    vel = np.diff(kp, axis=0)  # (T-1, N, 2)
    speed = np.linalg.norm(vel, axis=2)  # (T-1, N)
    mean_speed = float(np.mean(speed))

    # Acceleration: second difference
    if T >= 3:
        accel = np.diff(vel, axis=0)  # (T-2, N, 2)
        accel_mag = np.linalg.norm(accel, axis=2)  # (T-2, N)
        mean_accel = float(np.mean(accel_mag))
    else:
        mean_accel = 0.0

    total_motion_energy = float(np.sum(speed ** 2))

    return {
        "mean_speed": mean_speed,
        "mean_accel": mean_accel,
        "total_motion_energy": total_motion_energy,
    }


def compute_trajectory_geometry(
    all_keypoints: torch.Tensor, n_kp: int, lambda_ratio_threshold: float = 0.1
) -> dict:
    """
    Trajectory geometry: do keypoints move in 2D or just along 1D lines?

    For each keypoint, compute 2D covariance of (x_i(t), y_i(t)) over time,
    then eigenvalue ratio lambda2/lambda1. Ratio near 0 = 1D motion, near 1 = 2D.

    dim2_frac = fraction of keypoints with ratio > threshold.

    NOTE: Diagnostic only. Some valid keypoints can be ~1D under projected yaw
    on symmetric/elongated object parts.
    """
    T = all_keypoints.shape[0]
    kp = all_keypoints.view(T, n_kp, 2).numpy()  # (T, N, 2)

    per_kp_ratio = []
    for i in range(n_kp):
        traj = kp[:, i, :]  # (T, 2)
        if T < 3:
            per_kp_ratio.append(0.0)
            continue
        # Center
        traj_centered = traj - traj.mean(axis=0, keepdims=True)
        # 2x2 covariance
        cov = np.cov(traj_centered.T)  # (2, 2)
        eigvals = np.linalg.eigvalsh(cov)  # sorted ascending
        lam1 = max(eigvals[-1], 1e-12)
        lam2 = eigvals[0]
        per_kp_ratio.append(float(lam2 / lam1))

    dim2_frac = sum(1 for r in per_kp_ratio if r > lambda_ratio_threshold) / n_kp

    return {
        "dim2_frac": float(dim2_frac),
        "per_kp_lambda_ratio": per_kp_ratio,
        "lambda_ratio_threshold": lambda_ratio_threshold,
    }


def compute_operator_spectrum(model) -> dict:
    """
    Operator spectrum diagnostics (all in float64).

    - sv_min, sv_max: singular value range
    - spectral_radius: max |eigenvalue|
    - orth_err: ||W^T W - I||_F (how far from orthogonal)
    """
    W = model.operator.W.detach().cpu().double()  # (2N, 2N)
    D = W.shape[0]

    try:
        sv = torch.linalg.svdvals(W)
        sv_min = float(sv.min())
        sv_max = float(sv.max())
    except Exception:
        sv_min = float("nan")
        sv_max = float("nan")

    try:
        eigvals = torch.linalg.eigvals(W)
        spectral_radius = float(torch.max(torch.abs(eigvals)).item())
    except Exception:
        spectral_radius = float("nan")

    try:
        I = torch.eye(D, dtype=W.dtype)
        orth_err = float(torch.linalg.norm(W.T @ W - I, ord="fro").item())
    except Exception:
        orth_err = float("nan")

    return {
        "sv_min": sv_min,
        "sv_max": sv_max,
        "spectral_radius": spectral_radius,
        "orth_err": orth_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-run diagnostics: rollout viz, inverse test, localization, "
        "participation, stability, trajectory geometry, operator spectrum."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./rollout_output")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--start_frame",
        type=str,
        default="mid",
        help="Starting frame for rollout. 'mid' = middle of sequence, or integer.",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=None,
        help="Frame skip (overrides checkpoint config).",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Skip heavy visualizations, only compute metrics.",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Generate forward rollout viz but skip inverse viz. "
        "Used for soft-promoted configs in sweep.",
    )

    args = parser.parse_args()

    device = _select_device()
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device=device)
    frame_skip = args.frame_skip or config.get("frame_skip", 1)
    n_kp = model.num_keypoints

    # Load dataset (for transform + frame list)
    frames_dir = Path(args.frames_dir)
    dataset = SingleObjectDataset(
        str(frames_dir), img_size=args.img_size, frame_skip=frame_skip
    )
    frames = dataset.frames  # sorted list of Path objects
    n_frames = len(frames)

    # Determine start frame
    if args.start_frame == "mid":
        start = n_frames // 2
    else:
        start = int(args.start_frame)
    start = max(0, min(start, n_frames - 1))
    print(f"Start frame: {start} / {n_frames - 1}, frame_skip: {frame_skip}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract all keypoints (needed for all metrics) ──
    print("Extracting keypoints from all frames...")
    all_kp = _extract_all_keypoints(model, frames, dataset.transform, device)
    print(f"Extracted keypoints: shape {all_kp.shape}")

    results = {
        "start_frame": start,
        "frame_skip": frame_skip,
        "n_frames": n_frames,
        "num_keypoints": n_kp,
        "checkpoint": str(args.checkpoint),
    }

    # ── A) Forward rollout ──
    print("\n--- Forward rollout ---")
    forward = compute_forward_rollout(model, all_kp, start, frame_skip, device)
    results["forward"] = {
        f"k{k}_mse": res["mse"] for k, res in forward.items()
    }
    for k, res in sorted(forward.items()):
        print(f"  k={k}: MSE = {res['mse']:.6f}")

    if not args.metrics_only and forward:
        plot_forward_rollout(
            model, forward, start, frames, dataset.transform,
            args.img_size, output_dir / "rollout_visualization.png",
        )

    # ── B) Inverse rollout ──
    print("\n--- Inverse rollout (diagnostic) ---")
    inverse = compute_inverse_rollout(model, all_kp, start, frame_skip, device)
    results["inverse"] = {
        f"k{k}_mse": res["mse"] for k, res in inverse["per_k"].items()
    }
    results["inverse_condition_number"] = inverse["condition_number"]
    results["inverse_reliable"] = inverse["reliable"]
    print(f"  cond(W) = {inverse['condition_number']:.2f}  reliable={inverse['reliable']}")
    for k, res in sorted(inverse["per_k"].items()):
        print(f"  k=-{k}: MSE = {res['mse']:.6f}")

    if (not args.metrics_only and not args.forward_only
            and inverse["reliable"] and inverse["per_k"]):
        plot_inverse_rollout(
            model, inverse, start, frames, dataset.transform,
            args.img_size, output_dir / "inverse_visualization.png",
        )

    # ── C) Localization ──
    print("\n--- Localization ---")
    localization = compute_localization(all_kp, frames, args.img_size, n_kp)
    results["localization"] = {
        "on_foreground_pct": localization["on_foreground_pct"],
        "method": localization["method"],
    }
    # Store per-frame in a separate key (can be large)
    results["localization_per_frame"] = localization["per_frame_pct"]
    print(f"  on_foreground_pct = {localization['on_foreground_pct']:.3f}  method={localization['method']}")

    # ── D) Participation ──
    print("\n--- Participation ---")
    participation = compute_participation(all_kp, n_kp)
    results["participation"] = {
        "active_kp_frac": participation["active_kp_frac"],
        "top1_energy_frac": participation["top1_energy_frac"],
        "per_kp_energy": participation["per_kp_energy"],
    }
    print(f"  active_kp_frac = {participation['active_kp_frac']:.3f}")
    print(f"  top1_energy_frac = {participation['top1_energy_frac']:.3f}")

    # ── Stability ──
    print("\n--- Stability ---")
    stability = compute_stability(all_kp, n_kp)
    results["stability"] = stability
    print(f"  mean_speed = {stability['mean_speed']:.6f}")
    print(f"  mean_accel = {stability['mean_accel']:.6f}")
    print(f"  total_motion_energy = {stability['total_motion_energy']:.4f}")

    # ── Trajectory geometry ──
    print("\n--- Trajectory geometry (diagnostic) ---")
    traj_geom = compute_trajectory_geometry(all_kp, n_kp)
    results["trajectory_geometry"] = {
        "dim2_frac": traj_geom["dim2_frac"],
        "per_kp_lambda_ratio": traj_geom["per_kp_lambda_ratio"],
    }
    print(f"  dim2_frac = {traj_geom['dim2_frac']:.3f}")

    # ── Operator spectrum ──
    print("\n--- Operator spectrum (diagnostic) ---")
    spectrum = compute_operator_spectrum(model)
    results["operator_spectrum"] = spectrum
    print(f"  sv_min={spectrum['sv_min']:.4f}  sv_max={spectrum['sv_max']:.4f}")
    print(f"  spectral_radius={spectrum['spectral_radius']:.4f}")
    print(f"  orth_err={spectrum['orth_err']:.4f}")

    # ── Save ──
    metrics_path = output_dir / "rollout_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
