"""
Visualization utilities for Phase A.

Success criteria from Action Plan:
- Visualization makes sense
- Stable, smooth keypoints
- Linear map predicts K_{t+1} well
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from torchvision import transforms

from model import PhaseAModel
from dataset import SingleObjectDataset, get_inverse_transform


# Color palette for keypoints
COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
    '#00FFFF', '#FFA500', '#800080', '#008000', '#000080',
    '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#D2691E',
]


def load_model(checkpoint_path: str, num_keypoints: int = None, device: str = "cpu") -> tuple:
    """Load trained model from checkpoint.
    
    If checkpoint contains 'config', uses those hyperparameters.
    Otherwise falls back to num_keypoints argument (default 10).
    
    Returns:
        (model, config_dict) -- config_dict may be empty if not in checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to load config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = PhaseAModel(
            num_keypoints=config.get('num_keypoints', 10),
            base_channels=config.get('base_channels', 32),
            temperature=config.get('temperature', 1.0),
            num_action_classes=config.get('num_action_classes', 0),
        )
        print(f"Loaded config from checkpoint: {config}")
    else:
        # Fall back to argument or default
        n_kp = num_keypoints if num_keypoints is not None else 10
        model = PhaseAModel(num_keypoints=n_kp, num_action_classes=0)
        config = {}
        print(f"No config in checkpoint, using num_keypoints={n_kp}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model, config


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable numpy image."""
    # Denormalize
    inv_transform = get_inverse_transform()
    img = inv_transform(tensor)
    
    # Clip and convert to numpy
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def keypoints_to_pixels(keypoints: torch.Tensor, img_size: int) -> np.ndarray:
    """
    Convert keypoint coordinates from [-1, 1] to pixel coordinates.
    
    Args:
        keypoints: (N, 2) tensor of (x, y) coordinates in [-1, 1]
        img_size: Image dimension (assuming square)
    
    Returns:
        (N, 2) numpy array of (x, y) pixel coordinates
    """
    # Convert from [-1, 1] to [0, img_size-1]
    pixels = ((keypoints.numpy() + 1) / 2) * (img_size - 1)
    return pixels


def plot_keypoints_on_image(
    ax: plt.Axes,
    image: np.ndarray,
    keypoints: np.ndarray,
    title: str = "",
    show_numbers: bool = True,
    marker_size: int = 8,
):
    """Plot keypoints overlaid on image."""
    ax.imshow(image)
    
    for i, (x, y) in enumerate(keypoints):
        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, 'o', color=color, markersize=marker_size, markeredgecolor='white', markeredgewidth=1.5)
        if show_numbers:
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom', color='white',
                       xytext=(0, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    ax.set_title(title)
    ax.axis('off')


def visualize_sequence(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    frame_indices: list,
    output_path: Optional[str] = None,
    device: str = "cpu",
):
    """
    Visualize keypoints on a sequence of frames.
    
    Shows how keypoints move across the sequence (should be stable and smooth).
    """
    n_frames = len(frame_indices)
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))
    if n_frames == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, frame_indices):
        # Load frame
        sample = dataset[idx]
        x_t = sample['x_t'].unsqueeze(0).to(device)
        img_size = sample['x_t'].shape[-1]  # infer from tensor
        
        # Extract keypoints
        with torch.no_grad():
            keypoints, heatmaps = model.extractor(x_t)
        
        # Convert to (N, 2) pixel coordinates
        kp_coords = keypoints[0].view(model.num_keypoints, 2).cpu()
        kp_pixels = keypoints_to_pixels(kp_coords, img_size)
        
        # Convert tensor to displayable image
        img = tensor_to_image(sample['x_t'])
        
        # Plot
        plot_keypoints_on_image(ax, img, kp_pixels, title=f"Frame {idx}")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    frame_idx: int,
    output_path: Optional[str] = None,
    device: str = "cpu",
    yaw_step_deg: float = None,
):
    """
    Visualize keypoint prediction: p_t, p_{t+delta} (actual), p_hat_{t+delta} (predicted).
    
    This shows whether the linear operator predicts well.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    frame_skip = dataset.frame_skip
    # Build label that shows effective degrees if known
    if yaw_step_deg is not None:
        eff_deg = frame_skip * yaw_step_deg
        step_label = f"+{eff_deg:.0f} deg"
    else:
        step_label = f"+{frame_skip} frames"
    
    # Load pair (uses dataset's frame_skip)
    sample = dataset[frame_idx]
    img_size = sample['x_t'].shape[-1]  # infer from tensor
    t_idx = sample['t']
    t1_idx = sample['t1']
    x_t = sample['x_t'].unsqueeze(0).to(device)
    x_t1 = sample['x_t1'].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x_t, x_t1)
    
    # Get coordinates
    p_t = outputs['p_t'][0].view(model.num_keypoints, 2).cpu()
    p_t1 = outputs['p_t1'][0].view(model.num_keypoints, 2).cpu()
    p_hat_t1 = outputs['p_hat_t1'][0].view(model.num_keypoints, 2).cpu()
    
    # Convert to pixels
    kp_t_pixels = keypoints_to_pixels(p_t, img_size)
    kp_t1_pixels = keypoints_to_pixels(p_t1, img_size)
    kp_hat_t1_pixels = keypoints_to_pixels(p_hat_t1, img_size)
    
    # Convert images
    img_t = tensor_to_image(sample['x_t'])
    img_t1 = tensor_to_image(sample['x_t1'])
    
    # Plot with correct frame indices and step label
    plot_keypoints_on_image(axes[0], img_t, kp_t_pixels, title=f"p_t (frame {t_idx})")
    plot_keypoints_on_image(axes[1], img_t1, kp_t1_pixels, title=f"p_{{t+d}} actual (frame {t1_idx}, {step_label})")
    plot_keypoints_on_image(axes[2], img_t1, kp_hat_t1_pixels, title=f"p_hat predicted ({step_label})")
    
    # Compute prediction error
    error = torch.norm(p_t1 - p_hat_t1).item()
    fig.suptitle(f"Prediction Error: {error:.4f}", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_keypoint_trajectories(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    output_path: Optional[str] = None,
    device: str = "cpu",
):
    """
    Plot keypoint trajectories across the entire sequence.
    
    This shows whether keypoints are stable and move smoothly.
    """
    n_frames = len(dataset) + 1  # +1 because dataset gives pairs
    
    # Extract keypoints for all frames
    all_keypoints = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x_t = sample['x_t'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            keypoints, _ = model.extractor(x_t)
        
        kp = keypoints[0].view(model.num_keypoints, 2).cpu().numpy()
        all_keypoints.append(kp)
    
    # Also get last frame
    last_sample = dataset[len(dataset) - 1]
    x_last = last_sample['x_t1'].unsqueeze(0).to(device)
    with torch.no_grad():
        keypoints, _ = model.extractor(x_last)
    all_keypoints.append(keypoints[0].view(model.num_keypoints, 2).cpu().numpy())
    
    all_keypoints = np.stack(all_keypoints)  # (T, N, 2)
    
    # Plot trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: trajectories in normalized space
    ax = axes[0]
    for i in range(model.num_keypoints):
        traj = all_keypoints[:, i, :]  # (T, 2)
        color = COLORS[i % len(COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=color, markersize=2, linewidth=1, label=f'KP {i}')
        # Mark start and end
        ax.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=8)  # Start
        ax.plot(traj[-1, 0], traj[-1, 1], '^', color=color, markersize=8)  # End
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Keypoint Trajectories (normalized coords)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Right: x-coordinate over time
    ax = axes[1]
    frames = np.arange(len(all_keypoints))
    for i in range(model.num_keypoints):
        color = COLORS[i % len(COLORS)]
        ax.plot(frames, all_keypoints[:, i, 0], '-', color=color, linewidth=1.5, label=f'KP {i}')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('x coordinate')
    ax.set_title('Keypoint x-coordinate over time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_heatmaps(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    frame_idx: int,
    output_path: Optional[str] = None,
    device: str = "cpu",
):
    """Visualize the N keypoint heatmaps."""
    sample = dataset[frame_idx]
    x_t = sample['x_t'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        keypoints, heatmaps = model.extractor(x_t)
    
    heatmaps = heatmaps[0].cpu().numpy()  # (N, H', W')
    N = heatmaps.shape[0]
    
    # Determine grid size
    cols = min(5, N)
    rows = (N + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if N > 1 else [axes]
    
    for i in range(N):
        ax = axes[i]
        im = ax.imshow(heatmaps[i], cmap='hot')
        ax.set_title(f'Keypoint {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused axes
    for i in range(N, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Keypoint Heatmaps (Frame {frame_idx})')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_operator(
    model: PhaseAModel,
    output_dir: Path,
):
    """
    Visualize the learned linear operator p̂_{t+1} = W p_t + b.

    This does not "prove" semantic keypoints, but it helps debug whether the
    learned operator is:
    - close to identity (degenerate),
    - mostly per-keypoint (block-diagonal),
    - strongly mixing keypoints (off-diagonal energy),
    - roughly norm-preserving (rotation-like) vs scaling/shearing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    W = model.operator.W.detach().cpu().numpy()  # (2N, 2N)
    b = model.operator.b.detach().cpu().numpy()  # (2N,)
    N = model.num_keypoints
    D = 2 * N

    # Basic metrics
    I = np.eye(D, dtype=W.dtype)
    fro_W = float(np.linalg.norm(W, ord="fro"))
    fro_WmI = float(np.linalg.norm(W - I, ord="fro"))
    off_diag_mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(off_diag_mask, False)

    # Block (2x2) structure: block[i,j] maps keypoint j -> keypoint i
    block_norms = np.zeros((N, N), dtype=np.float64)
    diag_block_angles = []
    diag_block_det = []
    diag_block_svals = []

    for i in range(N):
        for j in range(N):
            blk = W[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            block_norms[i, j] = np.linalg.norm(blk, ord="fro")
        # Diagonal 2x2 block diagnostics
        A = W[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        try:
            U, s, Vt = np.linalg.svd(A)
            R = U @ Vt  # closest rotation (polar decomposition)
            angle = float(np.arctan2(R[1, 0], R[0, 0]))
            diag_block_angles.append(angle)
            diag_block_svals.append([float(s[0]), float(s[1])])
        except np.linalg.LinAlgError:
            diag_block_angles.append(float("nan"))
            diag_block_svals.append([float("nan"), float("nan")])
        diag_block_det.append(float(np.linalg.det(A)))

    # Energy split: diagonal blocks vs off-diagonal blocks (squared Frobenius)
    block_energy = block_norms ** 2
    diag_energy = float(np.trace(block_energy))
    off_energy = float(block_energy[off_diag_mask].sum())
    mix_ratio = off_energy / (diag_energy + off_energy + 1e-12)

    # Spectral diagnostics
    try:
        eigvals = np.linalg.eigvals(W)
    except np.linalg.LinAlgError:
        eigvals = np.array([], dtype=np.complex128)

    # Save metrics
    metrics = {
        "dim": int(D),
        "num_keypoints": int(N),
        "fro_W": fro_W,
        "fro_W_minus_I": fro_WmI,
        "block_diag_energy": diag_energy,
        "block_offdiag_energy": off_energy,
        "block_mix_ratio_offdiag": float(mix_ratio),
        "diag_block_angles_rad": [float(x) for x in diag_block_angles],
        "diag_block_det": [float(x) for x in diag_block_det],
        "diag_block_singular_values": diag_block_svals,
        "bias_norm": float(np.linalg.norm(b)),
        "eig_abs_mean": float(np.mean(np.abs(eigvals))) if eigvals.size else None,
        "eig_abs_max": float(np.max(np.abs(eigvals))) if eigvals.size else None,
    }
    (output_dir / "operator_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Build plots
    vmax = float(np.quantile(np.abs(W), 0.99)) if W.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Raw W heatmap
    ax = axes[0, 0]
    im = ax.imshow(W, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Operator W (2N x 2N)")
    ax.set_xlabel("Input dims")
    ax.set_ylabel("Output dims")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2) 2x2 block norm heatmap (N x N)
    ax = axes[0, 1]
    im = ax.imshow(block_norms, cmap="viridis")
    ax.set_title("Block norms ||W_ij||_F (each block is 2x2)")
    ax.set_xlabel("Input keypoint j")
    ax.set_ylabel("Output keypoint i")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.text(
        0.02,
        0.98,
        f"Off-diagonal mix ratio: {mix_ratio:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # 3) Eigenvalues (complex plane)
    ax = axes[1, 0]
    if eigvals.size:
        ax.scatter(eigvals.real, eigvals.imag, s=30, alpha=0.8)
        # Unit circle for reference
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, alpha=0.5)
        ax.set_aspect("equal", "box")
    ax.set_title("eig(W) in complex plane (unit circle dashed)")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, alpha=0.3)

    # 4) Diagonal block "rotation-like" angles (polar decomposition)
    ax = axes[1, 1]
    ax.plot(range(N), diag_block_angles, marker="o", linewidth=2)
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.3)
    ax.set_title("Angles of diagonal 2x2 blocks (closest rotation), radians")
    ax.set_xlabel("Keypoint i")
    ax.set_ylabel("angle (rad)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Operator diagnostics: ||W||_F={fro_W:.2f}, ||W-I||_F={fro_WmI:.2f}, ||b||={np.linalg.norm(b):.2f}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    out_path = output_dir / "operator_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Phase A results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--output_dir", type=str, default="./viz_output")
    parser.add_argument("--num_keypoints", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load model and config from checkpoint
    model, ckpt_config = load_model(args.checkpoint, args.num_keypoints, device)
    
    # Use frame_skip and yaw_step_deg from checkpoint so viz matches training
    frame_skip = ckpt_config.get('frame_skip', 1)
    yaw_step_deg = ckpt_config.get('yaw_step_deg', None)
    
    dataset = SingleObjectDataset(args.frames_dir, img_size=args.img_size, frame_skip=frame_skip)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Sequence visualization (sample frames)
    n_frames = len(dataset)
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    visualize_sequence(model, dataset, sample_indices, output_dir / "sequence.png", device)
    
    # 2. Prediction visualization (middle frame)
    visualize_prediction(model, dataset, n_frames // 2, output_dir / "prediction.png", device,
                         yaw_step_deg=yaw_step_deg)
    
    # 3. Keypoint trajectories
    visualize_keypoint_trajectories(model, dataset, output_dir / "trajectories.png", device)
    
    # 4. Heatmaps
    visualize_heatmaps(model, dataset, n_frames // 2, output_dir / "heatmaps.png", device)

    # 5. Operator diagnostics
    visualize_operator(model, output_dir)
    
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
