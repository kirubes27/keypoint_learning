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
from typing import Optional

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


def _object_dir_from_frames_dir(frames_dir: Path) -> Path:
    """Return the object directory for a TDW path like .../object/frames/a."""
    # frames_dir is usually .../train/<object>/frames/a.
    if frames_dir.name and frames_dir.parent.name == "frames":
        return frames_dir.parent.parent
    if frames_dir.name == "frames":
        return frames_dir.parent
    return frames_dir.parent


def _load_frame_metadata(frames_dir: Path, frames: list[Path]) -> list[Optional[dict]]:
    """
    Load per-frame metadata aligned to `frames`.

    For the v2 roll dataset, theta_deg lives in <object>/meta.jsonl. We do not
    recompute angles from frame index because that is exactly where 2deg-vs-6deg
    mistakes enter the analysis.
    """
    object_dir = _object_dir_from_frames_dir(frames_dir)
    meta_path = object_dir / "meta.jsonl"
    if not meta_path.exists():
        return [None for _ in frames]

    by_relpath = {}
    with open(meta_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "image_relpath" in row:
                by_relpath[row["image_relpath"]] = row

    aligned = []
    for frame in frames:
        try:
            rel = frame.relative_to(object_dir).as_posix()
        except ValueError:
            rel = frame.as_posix()
        aligned.append(by_relpath.get(rel))
    return aligned


def _mask_path_for_frame(frame_path: Path) -> Optional[Path]:
    """
    Infer TDW binary-mask path from frame path.

    Expected layout:
        <object>/frames/a/img_0000.png
        <object>/masks/a/mask_0000.png
    """
    if frame_path.parent.parent.name != "frames":
        return None
    object_dir = frame_path.parent.parent.parent
    camera_dir = frame_path.parent.name
    suffix = frame_path.stem.replace("img_", "")
    mask_path = object_dir / "masks" / camera_dir / f"mask_{suffix}.png"
    return mask_path if mask_path.exists() else None


def _load_binary_mask(mask_path: Path, img_size: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    return np.array(mask) > 0


def _keypoints_to_pixel_array(kp_norm: np.ndarray, img_size: int) -> np.ndarray:
    """Convert keypoints from [-1, 1] normalized coordinates to image pixels."""
    return ((kp_norm + 1.0) / 2.0) * (img_size - 1)


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def compute_mask_localization(
    all_keypoints: torch.Tensor,
    frames: list[Path],
    img_size: int,
    n_kp: int,
    border_frac: float = 0.05,
) -> Optional[dict]:
    """
    Compute strict object localization from TDW binary masks.

    This is the preferred metric for the v2 roll dataset. It is intentionally
    evaluation-only: masks are not fed into the training loss here.
    """
    mask_paths = [_mask_path_for_frame(Path(f)) for f in frames]
    missing = sum(p is None for p in mask_paths)
    if missing:
        return None

    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        distance_transform_edt = None

    kp = all_keypoints.view(all_keypoints.shape[0], n_kp, 2).numpy()
    T = min(kp.shape[0], len(frames))
    border_px = max(1, int(round(img_size * border_frac)))

    per_frame_pct = []
    per_frame_mean_dist_px = []
    per_frame_mean_dist_norm = []
    per_kp_inside = np.zeros((T, n_kp), dtype=np.float32)
    per_kp_dist_px = np.zeros((T, n_kp), dtype=np.float32)
    border_hits = np.zeros((T, n_kp), dtype=np.float32)
    out_of_bounds = np.zeros((T, n_kp), dtype=np.float32)
    centroid_offsets_norm = []
    bbox_diags = []
    sqrt_areas = []

    for t in range(T):
        mask = _load_binary_mask(mask_paths[t], img_size)
        area = int(mask.sum())
        sqrt_area = float(np.sqrt(max(area, 1)))
        sqrt_areas.append(sqrt_area)

        bbox = _mask_bbox(mask)
        if bbox is None:
            bbox_diag = float("nan")
            centroid_offsets_norm.append(float("nan"))
        else:
            x0, y0, x1, y1 = bbox
            bbox_diag = float(np.sqrt((x1 - x0 + 1) ** 2 + (y1 - y0 + 1) ** 2))
            ys, xs = np.where(mask)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            center = (img_size - 1) / 2.0
            centroid_offsets_norm.append(
                float(np.sqrt((centroid_x - center) ** 2 + (centroid_y - center) ** 2) / max(bbox_diag, 1.0))
            )
        bbox_diags.append(bbox_diag)

        dist_map = None
        if distance_transform_edt is not None:
            # Distance is 0 on-object and positive in background.
            dist_map = distance_transform_edt(~mask)

        kp_px = _keypoints_to_pixel_array(kp[t], img_size)
        inside_count = 0
        dists = []
        norm_dists = []
        for i in range(n_kp):
            x = kp_px[i, 0]
            y = kp_px[i, 1]
            border_hits[t, i] = (
                x < border_px or x > (img_size - 1 - border_px)
                or y < border_px or y > (img_size - 1 - border_px)
            )
            if x < 0 or x >= img_size or y < 0 or y >= img_size:
                out_of_bounds[t, i] = 1.0
                dist_px = float(img_size)
            else:
                px_x = int(round(x))
                px_y = int(round(y))
                px_x = min(max(px_x, 0), img_size - 1)
                px_y = min(max(px_y, 0), img_size - 1)
                if mask[px_y, px_x]:
                    inside_count += 1
                    per_kp_inside[t, i] = 1.0
                if dist_map is not None:
                    dist_px = float(dist_map[px_y, px_x])
                else:
                    dist_px = 0.0 if mask[px_y, px_x] else float("nan")
            per_kp_dist_px[t, i] = dist_px
            dists.append(dist_px)
            norm_dists.append(dist_px / max(sqrt_area, 1.0))

        per_frame_pct.append(inside_count / n_kp)
        per_frame_mean_dist_px.append(float(np.nanmean(dists)))
        per_frame_mean_dist_norm.append(float(np.nanmean(norm_dists)))

    return {
        "on_object_pct": float(np.mean(per_frame_pct)),
        "method": "tdw_binary_mask",
        "mask_distance_method": (
            "scipy.ndimage.distance_transform_edt" if distance_transform_edt is not None
            else "inside_only_no_distance_transform"
        ),
        "border_frac": float(border_frac),
        "border_px": int(border_px),
        "border_occupancy_pct": float(np.mean(border_hits)),
        "out_of_bounds_pct": float(np.mean(out_of_bounds)),
        "mean_dist_to_object_px": float(np.nanmean(per_frame_mean_dist_px)),
        "mean_dist_to_object_norm_sqrt_area": float(np.nanmean(per_frame_mean_dist_norm)),
        "per_kp_on_object_pct": np.mean(per_kp_inside, axis=0).astype(float).tolist(),
        "per_kp_mean_dist_to_object_px": np.nanmean(per_kp_dist_px, axis=0).astype(float).tolist(),
        "per_frame_pct": [float(x) for x in per_frame_pct],
        "per_frame_mean_dist_px": [float(x) for x in per_frame_mean_dist_px],
        "mask_centroid_mean_offset_norm_bboxdiag": float(np.nanmean(centroid_offsets_norm)),
        "mask_centroid_max_offset_norm_bboxdiag": float(np.nanmax(centroid_offsets_norm)),
        "mean_mask_sqrt_area_px": float(np.nanmean(sqrt_areas)),
        "mean_mask_bbox_diag_px": float(np.nanmean(bbox_diags)),
        "n_frames": int(T),
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

    Note: active_kp_frac is a coarse participation metric, not a semantic
    quality metric. A jittering keypoint can be active but still slide across
    object parts. Combine it with mask localization and canonical RMS.
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
    per_kp_active = [bool(e > threshold) for e in per_kp_energy]
    active = sum(per_kp_active)
    active_frac = active / n_kp if n_kp > 0 else 0.0

    # Top-1 fraction
    top1_frac = max_energy / total_energy if total_energy > 0 else 1.0

    return {
        "active_kp_frac": float(active_frac),
        "active_count": int(active),
        "active_energy_threshold": float(threshold),
        "active_energy_rule": "per_kp_energy > 0.1 * max(per_kp_energy)",
        "per_kp_active": per_kp_active,
        "top1_energy_frac": float(top1_frac),
        "per_kp_energy": per_kp_energy,
    }


def compute_keypoint_quality(
    participation: dict,
    localization: dict,
    canonical: Optional[dict],
    n_kp: int,
    on_object_threshold: float = 0.5,
    canonical_rms_threshold: float = 0.20,
) -> dict:
    """
    Combine per-keypoint diagnostics into stricter quality summaries.

    This first partitions keypoints by motion and object-mask status:
    - active_on_object: moving and inside the TDW mask;
    - active_off_object: moving in the background;
    - static_on_object: not moving much but inside the TDW mask;
    - dead_off_object: not moving much and outside the TDW mask.

    If canonical roll metadata is available, active_on_object keypoints are
    further split into clean vs sliding by canonical RMS.
    """
    energies = participation.get("per_kp_energy") or [None] * n_kp
    active_flags = participation.get("per_kp_active")
    if active_flags is None:
        threshold = participation.get("active_energy_threshold", 0.0)
        active_flags = [
            bool(e is not None and e > threshold)
            for e in energies
        ]

    per_kp_on_object = (
        localization.get("per_kp_on_object_pct")
        or localization.get("per_kp_on_foreground_pct")
        or [None] * n_kp
    )
    if len(per_kp_on_object) < n_kp:
        per_kp_on_object = list(per_kp_on_object) + [None] * (n_kp - len(per_kp_on_object))
    on_object_flags = [
        bool(v is not None and v >= on_object_threshold)
        for v in per_kp_on_object[:n_kp]
    ]

    canonical_available = bool(canonical and canonical.get("available", False))
    canonical_rms = [None] * n_kp
    canonical_flags = [None] * n_kp
    if canonical_available:
        locked = canonical.get("locked_minus", {})
        canonical_rms = locked.get("per_kp_canonical_rms") or canonical_rms
        canonical_flags = [
            bool(v is not None and v <= canonical_rms_threshold)
            for v in canonical_rms[:n_kp]
        ]

    rows = []
    active_on_object = 0
    clean = 0
    dead_off_object = 0
    active_off_object = 0
    static_on_object = 0
    sliding_on_object = 0

    for i in range(n_kp):
        active = bool(active_flags[i]) if i < len(active_flags) else False
        on_object = bool(on_object_flags[i]) if i < len(on_object_flags) else False
        rms = canonical_rms[i] if i < len(canonical_rms) else None
        canon_ok = canonical_flags[i] if i < len(canonical_flags) else None

        if active and on_object:
            active_on_object += 1
        if active and (not on_object):
            active_off_object += 1
        if (not active) and on_object:
            static_on_object += 1
        if (not active) and (not on_object):
            dead_off_object += 1
        if canonical_available:
            if active and on_object and (canon_ok is True):
                clean += 1
            if active and on_object and canon_ok is False:
                sliding_on_object += 1

        if active and on_object:
            motion_object_bucket = "active_on_object"
        elif active and not on_object:
            motion_object_bucket = "active_off_object"
        elif (not active) and on_object:
            motion_object_bucket = "static_on_object"
        else:
            motion_object_bucket = "dead_off_object"

        rows.append({
            "kp": int(i),
            "on_object_pct": per_kp_on_object[i] if i < len(per_kp_on_object) else None,
            "energy": energies[i] if i < len(energies) else None,
            "active": active,
            "on_object": on_object,
            "canonical_rms": rms,
            "canonical_ok": canon_ok,
            "motion_object_bucket": motion_object_bucket,
            "active_on_object": bool(active and on_object),
            "active_off_object": bool(active and (not on_object)),
            "static_on_object": bool((not active) and on_object),
            "dead_off_object": bool((not active) and (not on_object)),
            "clean": bool(active and on_object and (canon_ok is True)),
        })

    return {
        "on_object_threshold": float(on_object_threshold),
        "canonical_rms_threshold": float(canonical_rms_threshold),
        "canonical_available": bool(canonical_available),
        "active_on_object_frac": float(active_on_object / n_kp) if n_kp else 0.0,
        "active_off_object_frac": float(active_off_object / n_kp) if n_kp else 0.0,
        "static_on_object_frac": float(static_on_object / n_kp) if n_kp else 0.0,
        "dead_off_object_frac": float(dead_off_object / n_kp) if n_kp else 0.0,
        "motion_object_partition_sum": float(
            (active_on_object + active_off_object + static_on_object + dead_off_object) / n_kp
        ) if n_kp else 0.0,
        "clean_kp_frac": float(clean / n_kp) if (n_kp and canonical_available) else None,
        "sliding_on_object_frac": (
            float(sliding_on_object / n_kp) if (n_kp and canonical_available) else None
        ),
        "per_kp": rows,
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


@torch.no_grad()
def compute_cyclic_compositionality(
    model,
    all_keypoints: torch.Tensor,
    frame_skip: int,
    max_k: int,
    device: str,
    target_step_deg: Optional[float] = None,
) -> dict:
    """
    Evaluate modular k-step rollouts on a closed orbit.

    Target index is (t + k * frame_skip) mod T. This is required for the
    360-degree roll dataset because the most important check is the closed
    orbit: with skip=3 and 2deg source frames, k=60 should return to frame t.
    """
    model.eval()
    T = all_keypoints.shape[0]
    errors_by_k = {}

    for k in range(1, max_k + 1):
        errs = []
        for t in range(T):
            target_idx = (t + k * frame_skip) % T
            p_t = all_keypoints[t : t + 1].to(device)
            p_actual = all_keypoints[target_idx : target_idx + 1].to(device)
            p_hat = model.multi_step_predict(p_t, k)
            errs.append(float(torch.mean((p_hat - p_actual) ** 2).item()))
        arr = np.array(errs, dtype=np.float64)
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        errors_by_k[str(k)] = {
            "mean": float(np.mean(arr)),
            "std": std,
            "sem": float(std / np.sqrt(len(arr))) if len(arr) else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_samples": int(len(arr)),
            "target_offset_frames_mod": int((k * frame_skip) % T),
        }

    closed_orbit_k = None
    if target_step_deg is not None and target_step_deg > 0:
        maybe_k = 360.0 / float(target_step_deg)
        if abs(maybe_k - round(maybe_k)) < 1e-6:
            closed_orbit_k = int(round(maybe_k))
    if closed_orbit_k is None:
        for k in range(1, max_k + 1):
            if (k * frame_skip) % T == 0:
                closed_orbit_k = k
                break

    result = {
        "method": "cyclic_modular_target_indexing",
        "frame_skip": int(frame_skip),
        "max_k": int(max_k),
        "n_frames": int(T),
        "target_step_deg": float(target_step_deg) if target_step_deg is not None else None,
        "closed_orbit_k": closed_orbit_k,
        "error_bar_definition": (
            "mean +/- 1 s.e.m. across cyclic start frames t; "
            "descriptive within-sequence variability, not independent samples"
        ),
        "errors": errors_by_k,
    }
    if closed_orbit_k is not None and str(closed_orbit_k) in errors_by_k:
        result["closed_orbit"] = errors_by_k[str(closed_orbit_k)]
    return result


def plot_cyclic_compositionality(results: dict, output_path: Path):
    ks = [int(k) for k in results["errors"].keys()]
    ks.sort()
    means = [results["errors"][str(k)]["mean"] for k in ks]
    sems = [results["errors"][str(k)]["sem"] for k in ks]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(ks, means, yerr=sems, marker="o", markersize=3, linewidth=1.5, capsize=2)
    ax.set_xlabel("k operator applications")
    ax.set_ylabel("MSE (mean +/- 1 s.e.m.)")
    ax.set_title("Cyclic Compositionality: modular target indexing")
    ax.grid(True, alpha=0.3)
    closed_k = results.get("closed_orbit_k")
    if closed_k is not None and closed_k in ks:
        ax.axvline(closed_k, color="red", linestyle="--", linewidth=1.2, label=f"closed orbit k={closed_k}")
        ax.legend()
    ax.text(
        0.02, 0.98,
        "Error bars: +/- 1 s.e.m. over cyclic start frames t\n(descriptive within-sequence variability)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _rotate_points_standard(points: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """
    Rotate row-vector points by a per-frame standard 2D rotation matrix.

    points: (T, N, 2), theta_rad: (T,)
    keypoints are in normalized image coordinates [-1, 1], so the verified
    image center is the origin and no pixel-space pivot is used.
    """
    c = np.cos(theta_rad)[:, None]
    s = np.sin(theta_rad)[:, None]
    x = points[:, :, 0]
    y = points[:, :, 1]
    x_new = c * x - s * y
    y_new = s * x + c * y
    return np.stack([x_new, y_new], axis=-1)


def _canonical_rms(canonical: np.ndarray) -> tuple[list[float], float, float, float]:
    centered = canonical - canonical.mean(axis=0, keepdims=True)
    sq = np.sum(centered ** 2, axis=-1)  # (T, N)
    per_kp_rms = np.sqrt(np.mean(sq, axis=0))
    return (
        per_kp_rms.astype(float).tolist(),
        float(np.mean(per_kp_rms)),
        float(np.median(per_kp_rms)),
        float(np.max(per_kp_rms)),
    )


def _pairwise_distance_drift(kp: np.ndarray) -> dict:
    T, N, _ = kp.shape
    if N < 2:
        return {"mean_pairwise_dist_std": 0.0, "max_pairwise_dist_std": 0.0}
    dists = []
    for t in range(T):
        diff = kp[t, :, None, :] - kp[t, None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        dists.append(d[np.triu_indices(N, k=1)])
    dists = np.stack(dists, axis=0)
    stds = np.std(dists, axis=0, ddof=1) if T > 1 else np.zeros(dists.shape[1])
    return {
        "mean_pairwise_dist_std": float(np.mean(stds)),
        "max_pairwise_dist_std": float(np.max(stds)),
    }


def compute_canonical_stability(
    all_keypoints: torch.Tensor,
    frame_metadata: list[Optional[dict]],
    n_kp: int,
) -> Optional[dict]:
    """
    Object-relative stability for World-Z roll.

    The sign is locked to -theta from the independent mask IoU smoke check for
    this TDW World-Z roll dataset. We also compute +theta as audit-only and flag
    if it appears lower-variance; we never choose the sign by minimizing the
    reported metric.
    """
    if not frame_metadata or any(m is None or "theta_deg" not in m for m in frame_metadata):
        return None

    kp = all_keypoints.view(all_keypoints.shape[0], n_kp, 2).numpy()
    theta_deg = np.array([float(m["theta_deg"]) for m in frame_metadata[: kp.shape[0]]], dtype=np.float64)
    theta_rad = np.deg2rad(theta_deg)

    canonical_plus = _rotate_points_standard(kp, theta_rad)
    canonical_minus = _rotate_points_standard(kp, -theta_rad)

    plus_per, plus_mean, plus_median, plus_max = _canonical_rms(canonical_plus)
    minus_per, minus_mean, minus_median, minus_max = _canonical_rms(canonical_minus)

    raw_centered = kp - kp.mean(axis=0, keepdims=True)
    raw_per_kp_rms = np.sqrt(np.mean(np.sum(raw_centered ** 2, axis=-1), axis=0))
    probe_idx = int(np.argmax(raw_per_kp_rms))
    plus_probe = float(plus_per[probe_idx])
    minus_probe = float(minus_per[probe_idx])

    locked_not_lower = minus_mean > plus_mean
    pairwise = _pairwise_distance_drift(kp)

    return {
        "method": "canonical_unrotation_world_z_roll",
        "coordinate_space": "normalized_keypoints_minus1_to_1",
        "pivot": [0.0, 0.0],
        "angle_source": "frame_metadata.theta_deg",
        "locked_sign": "-theta",
        "locked_sign_provenance": (
            "Independent mask IoU smoke verification showed -theta de-rotation "
            "recovers frame 0 under the eval image-coordinate convention; sign "
            "is not selected by canonical variance."
        ),
        "audit_sign": "+theta",
        "warning_locked_sign_not_lower_variance": bool(locked_not_lower),
        "locked_minus": {
            "per_kp_canonical_rms": minus_per,
            "mean_canonical_rms": minus_mean,
            "median_canonical_rms": minus_median,
            "max_canonical_rms": minus_max,
        },
        "audit_plus": {
            "per_kp_canonical_rms": plus_per,
            "mean_canonical_rms": plus_mean,
            "median_canonical_rms": plus_median,
            "max_canonical_rms": plus_max,
        },
        "flatness_probe": {
            "keypoint_index": probe_idx,
            "selection_rule": "largest raw trajectory RMS; not used to choose sign",
            "raw_rms": float(raw_per_kp_rms[probe_idx]),
            "locked_minus_canonical_rms": minus_probe,
            "audit_plus_canonical_rms": plus_probe,
        },
        "pairwise_distance_drift": pairwise,
        "theta_deg_first": float(theta_deg[0]),
        "theta_deg_last": float(theta_deg[-1]),
        "n_frames": int(kp.shape[0]),
    }


def plot_canonical_flatness_probe(stability: dict, all_keypoints: torch.Tensor,
                                  frame_metadata: list[dict], n_kp: int,
                                  output_path: Path):
    kp = all_keypoints.view(all_keypoints.shape[0], n_kp, 2).numpy()
    theta_deg = np.array([float(m["theta_deg"]) for m in frame_metadata[: kp.shape[0]]], dtype=np.float64)
    theta_rad = np.deg2rad(theta_deg)
    idx = stability["flatness_probe"]["keypoint_index"]
    raw = kp[:, idx, :]
    plus = _rotate_points_standard(kp[:, idx:idx + 1, :], theta_rad)[:, 0, :]
    minus = _rotate_points_standard(kp[:, idx:idx + 1, :], -theta_rad)[:, 0, :]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, arr, title in [
        (axes[0], raw, "raw keypoint coordinates"),
        (axes[1], minus, "canonical -theta (locked sign)"),
        (axes[2], plus, "canonical +theta (audit only)"),
    ]:
        ax.plot(theta_deg, arr[:, 0], label="x", linewidth=1.5)
        ax.plot(theta_deg, arr[:, 1], label="y", linewidth=1.5)
        ax.set_ylabel("coord")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("theta_deg from metadata")
    fig.suptitle(f"Canonical flatness probe: keypoint {idx}", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compute_operator_spectrum(model, target_step_deg: float = None) -> dict:
    """
    Operator spectrum diagnostics (all in float64).

    - sv_min, sv_max: singular value range
    - spectral_radius: max |eigenvalue|
    - orth_err: ||W^T W - I||_F (how far from orthogonal)
    """
    W = model.operator.W.detach().cpu().double()  # (2N, 2N)
    D = W.shape[0]
    N = model.num_keypoints

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

    result = {
        "operator_type": type(model.operator).__name__,
        "operator_trainable_parameters": int(sum(p.numel() for p in model.operator.parameters())),
        "sv_min": sv_min,
        "sv_max": sv_max,
        "spectral_radius": spectral_radius,
        "orth_err": orth_err,
    }

    if hasattr(model.operator, "A") and hasattr(model.operator, "bias"):
        A = model.operator.A.detach().cpu().double()
        bias = model.operator.bias.detach().cpu().double()
        try:
            s = torch.linalg.svdvals(A)
            U, _, Vh = torch.linalg.svd(A)
            R = U @ Vh
            angle_rad = float(torch.atan2(R[1, 0], R[0, 0]).item())
            singular_values = [float(s[0]), float(s[1])]
        except Exception:
            R = torch.full((2, 2), float("nan"), dtype=A.dtype)
            angle_rad = float("nan")
            singular_values = [float("nan"), float("nan")]

        shared = {
            "A": A.tolist(),
            "bias": bias.tolist(),
            "det_A": float(torch.linalg.det(A).item()),
            "singular_values": singular_values,
            "orthogonality_error_fro": float(torch.linalg.norm(A.T @ A - torch.eye(2, dtype=A.dtype), ord="fro").item()),
            "closest_rotation_matrix": R.tolist(),
            "closest_rotation_angle_rad": angle_rad,
            "closest_rotation_angle_deg": float(np.rad2deg(angle_rad)),
            "bias_norm": float(torch.linalg.norm(bias).item()),
        }
        if target_step_deg is not None:
            theta = np.deg2rad(float(target_step_deg))
            R_pos = torch.tensor(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                dtype=A.dtype,
            )
            R_neg = torch.tensor(
                [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
                dtype=A.dtype,
            )
            err_pos = float(torch.linalg.norm(A - R_pos, ord="fro").item())
            err_neg = float(torch.linalg.norm(A - R_neg, ord="fro").item())
            shared.update(
                {
                    "target_step_deg": float(target_step_deg),
                    "fro_error_to_R_pos": err_pos,
                    "fro_error_to_R_neg": err_neg,
                    "best_signed_target": "+target" if err_pos <= err_neg else "-target",
                    "best_fro_error_to_target_rotation": min(err_pos, err_neg),
                }
            )
        result["shared_affine"] = shared

    if target_step_deg is not None:
        theta = np.deg2rad(float(target_step_deg))
        R_pos = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=W.dtype,
        )
        R_neg = torch.tensor(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
            dtype=W.dtype,
        )
        pos_errors = []
        neg_errors = []
        for i in range(N):
            A_i = W[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
            pos_errors.append(float(torch.linalg.norm(A_i - R_pos, ord="fro").item()))
            neg_errors.append(float(torch.linalg.norm(A_i - R_neg, ord="fro").item()))
        result["target_step_deg"] = float(target_step_deg)
        result["diag_block_mean_fro_error_to_R_pos"] = float(np.mean(pos_errors))
        result["diag_block_mean_fro_error_to_R_neg"] = float(np.mean(neg_errors))
        result["diag_block_best_signed_target"] = (
            "+target" if result["diag_block_mean_fro_error_to_R_pos"] <= result["diag_block_mean_fro_error_to_R_neg"] else "-target"
        )
        result["diag_block_best_mean_fro_error_to_target_rotation"] = min(
            result["diag_block_mean_fro_error_to_R_pos"],
            result["diag_block_mean_fro_error_to_R_neg"],
        )

    return result


def compute_known_success_gates(results: dict) -> dict:
    """
    Gate known metrics with AND semantics.

    This per-run evaluator does not know identity-ratio or validation action
    accuracy; sweep.py adds those. Here we enforce that on-object can veto a
    run even if rollout/operator metrics look good.
    """
    loc = results.get("localization", {})
    part = results.get("participation", {})
    gates = {
        "on_object_pct": {
            "value": loc.get("on_object_pct"),
            "op": ">",
            "threshold": 0.5,
        },
        "active_kp_frac": {
            "value": part.get("active_kp_frac"),
            "op": ">",
            "threshold": 0.3,
        },
    }
    quality = results.get("keypoint_quality", {})
    if quality:
        gates["active_on_object_frac"] = {
            "value": quality.get("active_on_object_frac"),
            "op": ">",
            "threshold": 0.5,
        }
    for gate in gates.values():
        value = gate["value"]
        if value is None:
            gate["pass"] = False
        elif gate["op"] == ">":
            gate["pass"] = bool(value > gate["threshold"])
        elif gate["op"] == "<":
            gate["pass"] = bool(value < gate["threshold"])
        else:
            gate["pass"] = False

    failed = [name for name, gate in gates.items() if not gate["pass"]]
    warnings_list = []
    if "on_object_pct" in failed:
        warnings_list.append(
            "on_object_pct failed; this is a standalone veto for border/background shortcut runs."
        )
    if loc.get("border_occupancy_pct", 0.0) > 0.2:
        warnings_list.append("High border_occupancy_pct; inspect for border shortcut.")
    if quality.get("dead_off_object_frac", 0.0) > 0.0:
        warnings_list.append(
            "Some keypoints are both inactive and off-object; inspect per-kp quality table."
        )
    if quality.get("active_off_object_frac", 0.0) > 0.0:
        warnings_list.append(
            "Some keypoints are active but off-object; inspect for moving background shortcuts."
        )
    if (quality.get("sliding_on_object_frac") or 0.0) > 0.0:
        warnings_list.append(
            "Some keypoints are active/on-object but have high canonical RMS; inspect for sliding."
        )
    if results.get("canonical_stability", {}).get("warning_locked_sign_not_lower_variance"):
        warnings_list.append(
            "Canonical sign audit: locked -theta has higher variance than +theta; check sign/pivot/metadata handling."
        )

    return {
        "partial_gates_pass": len(failed) == 0,
        "known_gates_pass": len(failed) == 0,
        "gate_scope": (
            "partial rollout-only gates; identity_ratio, val_act_acc, and sweep-level "
            "promotion logic are not available inside this evaluator"
        ),
        "and_gate_semantics": "all known gates must pass; any single failed gate blocks promotion",
        "gates": gates,
        "failed_gates": failed,
        "warnings": warnings_list,
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
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Eval image size. Defaults to checkpoint config img_size, or 256 for legacy checkpoints.",
    )
    parser.add_argument(
        "--padding_mode_override",
        type=str,
        default=None,
        choices=["zeros", "reflect", "replicate", "circular"],
        help="Override checkpoint padding mode for legacy compatibility checks.",
    )
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
    parser.add_argument(
        "--cyclic_max_k",
        type=int,
        default=None,
        help=(
            "Maximum k for cyclic modular rollout. Defaults to the closed-orbit "
            "k when target_step_deg divides 360, otherwise 60."
        ),
    )
    parser.add_argument(
        "--border_frac",
        type=float,
        default=0.05,
        help="Border width as a fraction of image size for border-shortcut metric.",
    )

    args = parser.parse_args()

    device = _select_device()
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(
        args.checkpoint,
        device=device,
        padding_mode_override=args.padding_mode_override,
    )
    frame_skip = args.frame_skip or config.get("frame_skip", 1)
    args.img_size = args.img_size or config.get("img_size", 256)
    yaw_step_deg = config.get("yaw_step_deg", None)
    target_step_deg = frame_skip * yaw_step_deg if yaw_step_deg is not None else None
    n_kp = model.num_keypoints

    # Load dataset (for transform + frame list)
    frames_dir = Path(args.frames_dir)
    dataset = SingleObjectDataset(
        str(frames_dir), img_size=args.img_size, frame_skip=frame_skip
    )
    frames = dataset.frames  # sorted list of Path objects
    n_frames = len(frames)
    frame_metadata = _load_frame_metadata(frames_dir, frames)

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
        "yaw_step_deg": yaw_step_deg,
        "target_step_deg": target_step_deg,
        "frame_metadata_available": all(m is not None for m in frame_metadata),
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
    localization = compute_mask_localization(
        all_kp, frames, args.img_size, n_kp, border_frac=args.border_frac
    )
    if localization is None:
        localization = compute_localization(all_kp, frames, args.img_size, n_kp)
        results["localization"] = {
            "on_foreground_pct": localization["on_foreground_pct"],
            "method": localization["method"],
        }
        print(
            f"  on_foreground_pct = {localization['on_foreground_pct']:.3f}  "
            f"method={localization['method']}"
        )
    else:
        results["localization"] = {
            k: v for k, v in localization.items()
            if not k.startswith("per_frame")
        }
        print(
            f"  on_object_pct = {localization['on_object_pct']:.3f}  "
            f"mean_dist_px={localization['mean_dist_to_object_px']:.3f}  "
            f"border_pct={localization['border_occupancy_pct']:.3f}  "
            f"method={localization['method']}"
        )
    # Store per-frame in a separate key (can be large)
    results["localization_per_frame"] = localization["per_frame_pct"]
    if "per_frame_mean_dist_px" in localization:
        results["localization_per_frame_mean_dist_px"] = localization["per_frame_mean_dist_px"]

    # ── D) Participation ──
    print("\n--- Participation ---")
    participation = compute_participation(all_kp, n_kp)
    results["participation"] = participation
    print(f"  active_kp_frac = {participation['active_kp_frac']:.3f}")
    print(
        f"  active_count = {participation['active_count']}/{n_kp}  "
        f"threshold={participation['active_energy_threshold']:.6f}"
    )
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

    # ── Cyclic compositionality ──
    print("\n--- Cyclic compositionality ---")
    if args.cyclic_max_k is not None:
        cyclic_max_k = args.cyclic_max_k
    elif target_step_deg is not None and target_step_deg > 0:
        cyclic_max_k = int(round(360.0 / target_step_deg))
    else:
        cyclic_max_k = 60
    cyclic = compute_cyclic_compositionality(
        model, all_kp, frame_skip=frame_skip, max_k=cyclic_max_k,
        device=device, target_step_deg=target_step_deg,
    )
    results["cyclic_compositionality"] = cyclic
    closed = cyclic.get("closed_orbit")
    if closed is not None:
        print(
            f"  closed_orbit k={cyclic['closed_orbit_k']}: "
            f"MSE = {closed['mean']:.6f} +/- {closed['sem']:.6f}"
        )
    else:
        k1 = cyclic["errors"].get("1", {})
        print(f"  k=1 cyclic MSE = {k1.get('mean', float('nan')):.6f}")
    if not args.metrics_only:
        plot_cyclic_compositionality(cyclic, output_dir / "cyclic_compositionality.png")

    # ── Canonical object-relative stability ──
    print("\n--- Canonical object-relative stability ---")
    canonical = compute_canonical_stability(all_kp, frame_metadata, n_kp)
    if canonical is None:
        results["canonical_stability"] = {
            "available": False,
            "reason": "frame metadata with theta_deg not available for every frame",
        }
        print("  skipped: frame metadata with theta_deg is unavailable")
    else:
        canonical["available"] = True
        results["canonical_stability"] = canonical
        locked = canonical["locked_minus"]
        audit = canonical["audit_plus"]
        print(
            f"  locked -theta mean_rms={locked['mean_canonical_rms']:.6f}; "
            f"audit +theta mean_rms={audit['mean_canonical_rms']:.6f}; "
            f"warning={canonical['warning_locked_sign_not_lower_variance']}"
        )
        if not args.metrics_only:
            plot_canonical_flatness_probe(
                canonical, all_kp, frame_metadata, n_kp,
                output_dir / "canonical_flatness_probe.png",
            )

    # ── Per-keypoint quality table ──
    print("\n--- Per-keypoint quality ---")
    canonical_for_quality = results.get("canonical_stability", {})
    quality = compute_keypoint_quality(
        participation=participation,
        localization=localization,
        canonical=canonical_for_quality if canonical_for_quality.get("available", False) else None,
        n_kp=n_kp,
    )
    results["keypoint_quality"] = quality
    print(f"  active_on_object_frac = {quality['active_on_object_frac']:.3f}")
    print(f"  active_off_object_frac = {quality['active_off_object_frac']:.3f}")
    print(f"  static_on_object_frac = {quality['static_on_object_frac']:.3f}")
    print(f"  dead_off_object_frac = {quality['dead_off_object_frac']:.3f}")
    print(f"  motion_object_partition_sum = {quality['motion_object_partition_sum']:.3f}")
    if quality["canonical_available"]:
        print(f"  clean_kp_frac = {quality['clean_kp_frac']:.3f}")
        print(f"  sliding_on_object_frac = {quality['sliding_on_object_frac']:.3f}")
    else:
        print("  clean_kp_frac = n/a (canonical metadata unavailable)")
        print("  sliding_on_object_frac = n/a (canonical metadata unavailable)")
    print("  kp  on_obj  energy      canon_rms  active  bucket              clean")
    for row in quality["per_kp"]:
        on_obj = row["on_object_pct"]
        energy = row["energy"]
        rms = row["canonical_rms"]
        print(
            f"  {row['kp']:2d}  "
            f"{on_obj if on_obj is not None else float('nan'):6.3f}  "
            f"{energy if energy is not None else float('nan'):10.6f}  "
            f"{rms if rms is not None else float('nan'):9.6f}  "
            f"{str(row['active']):>6}  "
            f"{row['motion_object_bucket']:<18}  {str(row['clean']):>5}"
        )

    # ── Operator spectrum ──
    print("\n--- Operator spectrum (diagnostic) ---")
    spectrum = compute_operator_spectrum(model, target_step_deg=target_step_deg)
    results["operator_spectrum"] = spectrum
    print(f"  sv_min={spectrum['sv_min']:.4f}  sv_max={spectrum['sv_max']:.4f}")
    print(f"  spectral_radius={spectrum['spectral_radius']:.4f}")
    print(f"  orth_err={spectrum['orth_err']:.4f}")
    if "shared_affine" in spectrum:
        shared = spectrum["shared_affine"]
        print(f"  shared angle={shared['closest_rotation_angle_deg']:.3f} deg")
        if "best_fro_error_to_target_rotation" in shared:
            print(
                f"  shared target error={shared['best_fro_error_to_target_rotation']:.6f} "
                f"({shared['best_signed_target']})"
            )

    # ── Success gates available in this evaluator ──
    print("\n--- Known success gates ---")
    gates = compute_known_success_gates(results)
    results["success_gates"] = gates
    print(f"  known_gates_pass = {gates['known_gates_pass']}")
    if gates["failed_gates"]:
        print(f"  failed_gates = {', '.join(gates['failed_gates'])}")
    for warning in gates["warnings"]:
        print(f"  warning: {warning}")

    # ── Save ──
    metrics_path = output_dir / "rollout_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
