"""
Ablation Studies (10.5B)

Control experiments to verify that the linear operator is doing the work:

1. Remove operator (identity): Replace W with identity matrix
   → Prediction should collapse (just predicts p_t = p_{t+1})

2. Random operator: Replace W with random matrix
   → Prediction should be terrible

3. Randomize keypoints: Shuffle keypoint indices at test time
   → Predictability should collapse

4. Random keypoints (frozen): Replace extractor output with random
   → Predictability should collapse

These ablations support the claim that keypoints emerge BECAUSE they
make the generator action globally linear.

Usage:
    python ablations.py \
        --checkpoint ./runs/phase_a_xxx/best_model.pt \
        --frames_dir /path/to/frames/a
"""

import argparse
import json
from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import PhaseAModel, compute_losses
from dataset import SingleObjectDataset


def _seed_everything(seed: int) -> None:
    """Set seeds for reproducible ablations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_operator_identity(operator: nn.Module, dim: int, device: str) -> None:
    """Set dense or shared-affine operator to identity without assuming `.linear`."""
    if hasattr(operator, "linear"):
        operator.linear.weight.copy_(torch.eye(dim, device=device))
        operator.linear.bias.zero_()
    elif hasattr(operator, "A") and hasattr(operator, "bias"):
        operator.A.copy_(torch.eye(2, device=device))
        operator.bias.zero_()
    else:
        raise TypeError(f"Unsupported operator type for identity ablation: {type(operator).__name__}")


def _set_operator_random(operator: nn.Module, dim: int, device: str) -> None:
    """Set dense or shared-affine operator to a small random map."""
    if hasattr(operator, "linear"):
        operator.linear.weight.copy_(torch.randn(dim, dim, device=device) * 0.1)
        operator.linear.bias.copy_(torch.randn(dim, device=device) * 0.1)
    elif hasattr(operator, "A") and hasattr(operator, "bias"):
        operator.A.copy_(torch.randn(2, 2, device=device) * 0.1)
        operator.bias.copy_(torch.randn(2, device=device) * 0.1)
    else:
        raise TypeError(f"Unsupported operator type for random ablation: {type(operator).__name__}")


@torch.no_grad()
def evaluate_prediction_error(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    device: str = "cpu",
    num_keypoints: int = 10,
) -> float:
    """Compute average prediction error ||p̂_{t+1} - p_{t+1}||²."""
    model.eval()
    
    total_error = 0.0
    n_samples = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        x_t = sample['x_t'].unsqueeze(0).to(device)
        x_t1 = sample['x_t1'].unsqueeze(0).to(device)
        
        outputs = model(x_t, x_t1)
        error = F.mse_loss(outputs['p_hat_t1'], outputs['p_t1']).item()
        
        total_error += error
        n_samples += 1
    
    return total_error / n_samples


@torch.no_grad()
def ablation_identity_operator(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    device: str = "cpu",
    num_keypoints: int = 10,
) -> float:
    """
    Ablation 1: Replace W with identity, b with zero.
    
    This tests: does the linear operator actually learn something,
    or is p_t ≈ p_{t+1} already (trivial solution)?
    """
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    
    # Set W = I, b = 0
    dim = 2 * num_keypoints
    with torch.no_grad():
        _set_operator_identity(model_copy.operator, dim, device)
    
    return evaluate_prediction_error(model_copy, dataset, device, num_keypoints)


@torch.no_grad()
def ablation_random_operator(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    device: str = "cpu",
    num_keypoints: int = 10,
) -> float:
    """
    Ablation 2: Replace W with random matrix.
    
    This tests: is the learned W special, or would any matrix work?
    """
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    
    # Set W = random, b = random
    dim = 2 * num_keypoints
    with torch.no_grad():
        _set_operator_random(model_copy.operator, dim, device)
    
    return evaluate_prediction_error(model_copy, dataset, device, num_keypoints)


@torch.no_grad()
def ablation_shuffled_keypoints(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    device: str = "cpu",
    num_keypoints: int = 10,
) -> float:
    """
    Ablation 3: Shuffle keypoint indices at test time.
    
    This tests: does the correspondence between keypoints matter?
    If keypoints are semantically meaningful, shuffling should break prediction.
    """
    model.eval()
    model.to(device)
    
    # Create random permutation
    perm = torch.randperm(num_keypoints)
    
    total_error = 0.0
    n_samples = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        x_t = sample['x_t'].unsqueeze(0).to(device)
        x_t1 = sample['x_t1'].unsqueeze(0).to(device)
        
        # Extract keypoints
        p_t, _ = model.extractor(x_t)      # (1, 2N)
        p_t1, _ = model.extractor(x_t1)    # (1, 2N)
        
        # Shuffle p_t keypoints
        p_t_reshaped = p_t.view(1, num_keypoints, 2)  # (1, N, 2)
        p_t_shuffled = p_t_reshaped[:, perm, :]       # (1, N, 2)
        p_t_shuffled = p_t_shuffled.view(1, -1)       # (1, 2N)
        
        # Predict with shuffled input
        p_hat_t1 = model.operator(p_t_shuffled)
        
        # Error against unshuffled p_t1
        error = F.mse_loss(p_hat_t1, p_t1).item()
        
        total_error += error
        n_samples += 1
    
    return total_error / n_samples


@torch.no_grad()
def ablation_random_keypoints(
    model: PhaseAModel,
    dataset: SingleObjectDataset,
    device: str = "cpu",
    num_keypoints: int = 10,
) -> float:
    """
    Ablation 4: Replace extracted keypoints with random values.
    
    This tests: do the learned keypoint locations matter?
    """
    model.eval()
    model.to(device)
    
    total_error = 0.0
    n_samples = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        x_t1 = sample['x_t1'].unsqueeze(0).to(device)
        
        # Random keypoints for p_t (in [-1, 1] range)
        p_t_random = torch.rand(1, 2 * num_keypoints, device=device) * 2 - 1
        
        # Get actual p_t1
        p_t1, _ = model.extractor(x_t1)
        
        # Predict from random
        p_hat_t1 = model.operator(p_t_random)
        
        error = F.mse_loss(p_hat_t1, p_t1).item()
        
        total_error += error
        n_samples += 1
    
    return total_error / n_samples


def main():
    parser = argparse.ArgumentParser(description="Ablation Studies (10.5B)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ablation_output")
    parser.add_argument("--num_keypoints", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument(
        "--padding_mode_override",
        type=str,
        default=None,
        choices=["zeros", "reflect", "replicate", "circular"],
        help="Override checkpoint padding mode for legacy compatibility checks",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible ablation perturbations")
    
    args = parser.parse_args()

    _seed_everything(args.seed)
    print(f"Seed: {args.seed}")
    
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint and config
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get config from checkpoint or fall back to CLI args
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_keypoints = config.get('num_keypoints', args.num_keypoints)
        base_channels = config.get('base_channels', 32)
        temperature = config.get('temperature', 1.0)
        num_action_classes = config.get('num_action_classes', 0)
        operator_type = config.get('operator_type', 'dense')
        learn_inverse_operator = config.get('learn_inverse_operator', False)
        padding_mode = args.padding_mode_override if args.padding_mode_override is not None else config.get('padding_mode')
        frame_skip = config.get('frame_skip', args.frame_skip)
        if padding_mode is None:
            raise ValueError(
                "Checkpoint config is missing 'padding_mode'. "
                "Pass --padding_mode_override (usually 'zeros' for early Feb 2026 runs)."
            )
        print(f"Loaded config from checkpoint: {config}")
        print(f"Using padding_mode={padding_mode}")
    else:
        num_keypoints = args.num_keypoints
        base_channels = 32
        temperature = 1.0
        num_action_classes = 0
        operator_type = 'dense'
        learn_inverse_operator = False
        padding_mode = "reflect" if args.padding_mode_override is None else args.padding_mode_override
        frame_skip = args.frame_skip
        print(f"No config in checkpoint, using num_keypoints={num_keypoints}")
        print(f"Using padding_mode={padding_mode}")
    
    # Load model with correct config
    model = PhaseAModel(
        num_keypoints=num_keypoints,
        base_channels=base_channels,
        temperature=temperature,
        num_action_classes=num_action_classes,
        padding_mode=padding_mode,
        operator_type=operator_type,
        learn_inverse_operator=learn_inverse_operator,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    
    # Load dataset
    dataset = SingleObjectDataset(
        args.frames_dir,
        img_size=args.img_size,
        frame_skip=frame_skip,
    )
    
    # Run ablations
    print("\nRunning ablations...")
    print("-" * 60)
    
    results = {}
    
    # Baseline (trained model)
    print("1. Baseline (trained model)...")
    results['baseline'] = evaluate_prediction_error(model, dataset, device, num_keypoints)
    print(f"   MSE = {results['baseline']:.6f}")
    
    # Ablation 1: Identity operator
    print("2. Identity operator (W=I, b=0)...")
    results['identity_operator'] = ablation_identity_operator(model, dataset, device, num_keypoints)
    print(f"   MSE = {results['identity_operator']:.6f}")
    
    # Ablation 2: Random operator
    print("3. Random operator...")
    results['random_operator'] = ablation_random_operator(model, dataset, device, num_keypoints)
    print(f"   MSE = {results['random_operator']:.6f}")
    
    # Ablation 3: Shuffled keypoints
    print("4. Shuffled keypoint indices...")
    results['shuffled_keypoints'] = ablation_shuffled_keypoints(model, dataset, device, num_keypoints)
    print(f"   MSE = {results['shuffled_keypoints']:.6f}")
    
    # Ablation 4: Random keypoints
    print("5. Random keypoints (replacing extractor output)...")
    results['random_keypoints'] = ablation_random_keypoints(model, dataset, device, num_keypoints)
    print(f"   MSE = {results['random_keypoints']:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (trained):     {results['baseline']:.6f}")
    print(f"Identity operator:      {results['identity_operator']:.6f} ({results['identity_operator']/results['baseline']:.2f}x baseline)")
    print(f"Random operator:        {results['random_operator']:.6f} ({results['random_operator']/results['baseline']:.2f}x baseline)")
    print(f"Shuffled keypoints:     {results['shuffled_keypoints']:.6f} ({results['shuffled_keypoints']/results['baseline']:.2f}x baseline)")
    print(f"Random keypoints:       {results['random_keypoints']:.6f} ({results['random_keypoints']/results['baseline']:.2f}x baseline)")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if results['identity_operator'] > results['baseline'] * 1.5:
        print("✓ Identity operator is worse → W learns non-trivial transformation")
    else:
        print("⚠ Identity operator similar to baseline → might be trivial solution (p_t ≈ p_{t+1})")
    
    if results['random_operator'] > results['baseline'] * 2:
        print("✓ Random operator is much worse → learned W is special")
    else:
        print("⚠ Random operator not much worse → W might not be important")
    
    if results['shuffled_keypoints'] > results['baseline'] * 1.5:
        print("✓ Shuffling breaks prediction → keypoint correspondence matters")
    else:
        print("⚠ Shuffling doesn't hurt much → keypoints might be interchangeable")
    
    if results['random_keypoints'] > results['baseline'] * 2:
        print("✓ Random keypoints fail → learned keypoint locations matter")
    else:
        print("⚠ Random keypoints work → keypoint locations might not matter")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
