"""
Compositionality / Multi-step Rollout Evaluation (10.5A)

Tests "extent of linearity" by applying the operator k times and measuring
how prediction error grows with k.

If the representation is truly flat (globally linear), error should grow
slowly/linearly with k. If it's only locally linear, error will explode.

Usage:
    python eval_compositionality.py \
        --checkpoint ./runs/phase_a_xxx/best_model.pt \
        --frames_dir /path/to/frames/a \
        --max_k 10
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import PhaseAModel
from dataset import SingleObjectDataset


@torch.no_grad()
def evaluate_compositionality(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    max_k: int = 10,
    frame_skip: int = 1,
    device: str = "cpu",
) -> dict:
    """
    Evaluate k-step prediction error for k = 1, 2, ..., max_k.
    
    One operator step corresponds to `frame_skip` frames in the dataset.
    So after k operator steps we compare against frame t + k*frame_skip.
    
    For each starting frame t, we:
    1. Extract p_t = K(x_t)
    2. Predict p̂_{t+k*s} = W^k p_t + (sum of bias terms)
    3. Extract actual p_{t+k*s} = K(x_{t+k*s})
    4. Compute error ||p̂ - p||²
    
    Returns dict with errors per k.
    """
    model.eval()
    model.to(device)
    
    # We need access to individual frames, not pairs
    # The dataset has self.frames which is the list of frame paths
    n_frames = len(dataset.frames)
    
    # Pre-extract all keypoints
    from PIL import Image
    all_keypoints = []
    for i in range(n_frames):
        img = Image.open(dataset.frames[i]).convert('RGB')
        x = dataset.transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
        p, _ = model.extractor(x)  # (1, 2N)
        all_keypoints.append(p.cpu())
    
    all_keypoints = torch.cat(all_keypoints, dim=0)  # (T, 2N)
    print(f"Extracted keypoints from {n_frames} frames (frame_skip={frame_skip})")
    
    # Evaluate for each k (k operator steps = k*frame_skip frames)
    errors_by_k = {k: [] for k in range(1, max_k + 1)}
    
    for k in range(1, max_k + 1):
        target_offset = k * frame_skip
        # For each valid starting point
        for t in range(n_frames - target_offset):
            p_t = all_keypoints[t:t+1].to(device)  # (1, 2N)
            p_actual = all_keypoints[t + target_offset:t + target_offset + 1].to(device)
            
            # Multi-step prediction (k operator applications)
            p_hat = model.multi_step_predict(p_t, k)  # (1, 2N)
            
            # Compute MSE
            error = torch.mean((p_hat - p_actual) ** 2).item()
            errors_by_k[k].append(error)
    
    # Compute statistics
    results = {
        'max_k': max_k,
        'frame_skip': frame_skip,
        'n_frames': n_frames,
        'errors': {},
    }
    
    for k in range(1, max_k + 1):
        errs = errors_by_k[k]
        results['errors'][k] = {
            'mean': float(np.mean(errs)),
            'std': float(np.std(errs)),
            'min': float(np.min(errs)),
            'max': float(np.max(errs)),
            'n_samples': len(errs),
        }
    
    return results


def plot_compositionality(results: dict, output_path: str = None, yaw_step_deg: float = None):
    """Plot error vs k to visualize extent of linearity."""
    ks = list(range(1, results['max_k'] + 1))
    means = [results['errors'][k]['mean'] for k in ks]
    stds = [results['errors'][k]['std'] for k in ks]
    frame_skip = results.get('frame_skip', 1)
    
    # Build x-axis label that reports effective degrees if known
    if yaw_step_deg is not None:
        eff_deg = frame_skip * yaw_step_deg
        x_label = f'k operator steps ({eff_deg:.0f} deg each)'
    else:
        x_label = f'k operator steps (frame_skip={frame_skip})'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Error vs k
    ax = axes[0]
    ax.errorbar(ks, means, yerr=stds, marker='o', capsize=3, linewidth=2, markersize=8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('MSE (prediction error)', fontsize=12)
    ax.set_title('Multi-step Prediction Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    
    # Right: Log scale to see growth rate
    ax = axes[1]
    ax.semilogy(ks, means, marker='o', linewidth=2, markersize=8)
    ax.fill_between(ks, 
                    np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    alpha=0.3)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Error Growth Rate (log scale)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(ks)
    
    # Add annotation about linearity
    growth_ratio = means[-1] / means[0] if means[0] > 0 else float('inf')
    fig.suptitle(f'Compositionality Test: Error growth ratio (k={results["max_k"]}/k=1) = {growth_ratio:.2f}x', 
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def _select_device() -> str:
    """Prefer cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Compositionality Evaluation (10.5A)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum k for k-step prediction")
    parser.add_argument("--num_keypoints", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--frame_skip", type=int, default=None,
                        help="Frame skip (overrides checkpoint config if set)")
    
    args = parser.parse_args()
    
    device = _select_device()
    print(f"Using device: {device}")
    
    # Load checkpoint and config
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get model config from checkpoint or use args
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_keypoints = config.get('num_keypoints', args.num_keypoints)
        base_channels = config.get('base_channels', 32)
        temperature = config.get('temperature', 1.0)
        frame_skip = args.frame_skip if args.frame_skip is not None else config.get('frame_skip', 1)
        yaw_step_deg = config.get('yaw_step_deg', None)
        print(f"Loaded config from checkpoint: {config}")
    else:
        num_keypoints = args.num_keypoints
        base_channels = 32
        temperature = 1.0
        frame_skip = args.frame_skip if args.frame_skip is not None else 1
        yaw_step_deg = None
        print(f"No config in checkpoint, using num_keypoints={num_keypoints}")
    
    # Load model with correct config
    model = PhaseAModel(
        num_keypoints=num_keypoints,
        base_channels=base_channels,
        temperature=temperature,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    
    # Load dataset (for transforms and frame list)
    dataset = SingleObjectDataset(
        args.frames_dir, 
        img_size=args.img_size,
        frame_skip=frame_skip,
    )
    
    # Evaluate
    eff_label = f" ({frame_skip * yaw_step_deg:.0f} deg/step)" if yaw_step_deg else ""
    print(f"\nEvaluating compositionality for k=1..{args.max_k}, frame_skip={frame_skip}{eff_label}...")
    results = evaluate_compositionality(
        model, dataset, max_k=args.max_k, frame_skip=frame_skip, device=device,
    )
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for k in range(1, args.max_k + 1):
        err = results['errors'][k]
        frame_label = f" (frames +{k * frame_skip})" if frame_skip > 1 else ""
        print(f"  k={k:2d}{frame_label}: MSE = {err['mean']:.6f} +/- {err['std']:.6f}")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "compositionality_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_compositionality(results, output_dir / "compositionality_plot.png", yaw_step_deg=yaw_step_deg)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
