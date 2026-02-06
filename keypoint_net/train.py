"""
Phase A Training Script

Trains the keypoint extractor + linear operator as specified in the Action Plan.

Usage:
    # Single object (recommended for debugging)
    python train.py --data_root ./phase_a_yaw_only --object coffeemug --epochs 1000
    
    # All objects in train split
    python train.py --data_root ./phase_a_yaw_only --epochs 1000
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import PhaseAModel, compute_losses
from dataset import PoseSequenceDataset, SingleObjectDataset


def _select_device() -> torch.device:
    """Prefer cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_smooth: float,
    lambda_disp: float,
    lambda_ent: float,
    sigma: float,
    num_keypoints: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_pred = 0.0
    total_smooth = 0.0
    total_disp = 0.0
    total_ent = 0.0
    n_batches = 0
    
    for batch in loader:
        x_t = batch['x_t'].to(device)
        x_t1 = batch['x_t1'].to(device)
        
        # Forward pass
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=lambda_smooth,
            lambda_disp=lambda_disp,
            lambda_ent=lambda_ent,
            sigma=sigma,
            num_keypoints=num_keypoints,
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += losses['loss'].item()
        total_pred += losses['l_pred'].item()
        total_smooth += losses['l_smooth'].item()
        total_disp += losses['l_disp'].item()
        total_ent += losses['l_ent'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'l_pred': total_pred / n_batches,
        'l_smooth': total_smooth / n_batches,
        'l_disp': total_disp / n_batches,
        'l_ent': total_ent / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_smooth: float,
    lambda_disp: float,
    lambda_ent: float,
    sigma: float,
    num_keypoints: int,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_pred = 0.0
    n_batches = 0
    
    for batch in loader:
        x_t = batch['x_t'].to(device)
        x_t1 = batch['x_t1'].to(device)
        
        outputs = model(x_t, x_t1)
        losses = compute_losses(
            outputs,
            lambda_smooth=lambda_smooth,
            lambda_disp=lambda_disp,
            lambda_ent=lambda_ent,
            sigma=sigma,
            num_keypoints=num_keypoints,
        )
        
        total_loss += losses['loss'].item()
        total_pred += losses['l_pred'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'l_pred': total_pred / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase A Training")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--object", type=str, default=None, help="Single object name (e.g., coffeemug)")
    parser.add_argument("--img_size", type=int, default=256)
    
    # Model
    parser.add_argument("--num_keypoints", type=int, default=10, help="N keypoints")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0, help="Soft-argmax temperature")
    
    # Loss weights -- Run 0 defaults: only L_pred + L_disp
    parser.add_argument("--lambda_smooth", type=float, default=0.0, help="λ_s for L_smooth (0 for Run 0)")
    parser.add_argument("--lambda_disp", type=float, default=0.1, help="λ_d for L_disp")
    parser.add_argument("--lambda_ent", type=float, default=0.0, help="λ_e for L_ent (0 for Run 0)")
    parser.add_argument("--sigma", type=float, default=0.1, help="σ for L_disp length scale")
    
    # Dataset (10.4)
    parser.add_argument("--frame_skip", type=int, default=3, help="Frame skip for angular step (1=2 deg, 3=6 deg if yaw_step=2)")
    parser.add_argument("--yaw_step_deg", type=float, default=2.0, help="Yaw step in degrees used when generating the dataset")
    parser.add_argument("--center_crop", type=int, default=None, help="Center crop size before resize (reduces background)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10, help="Log metrics every N epochs")
    
    args = parser.parse_args()
    
    # Reproducibility
    _seed_everything(args.seed)
    print(f"Seed: {args.seed}")
    
    # Setup device
    device = _select_device()
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_name = args.object or "all"
    run_name = f"phase_a_{obj_name}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Output directory: {run_dir}")
    
    # Create dataset
    if args.object:
        # Single object mode - look for frames directly
        # Try to find the frames directory
        train_frames = Path(args.data_root) / "train" / args.object / "frames" / "a"
        test_frames = Path(args.data_root) / "test" / args.object / "frames" / "a"
        
        if train_frames.exists():
            train_dataset = SingleObjectDataset(
                str(train_frames), 
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
            )
            # Use same data for validation (small dataset)
            val_dataset = train_dataset
        elif test_frames.exists():
            train_dataset = SingleObjectDataset(
                str(test_frames), 
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
            )
            val_dataset = train_dataset
        else:
            # Fall back to PoseSequenceDataset with object filter
            train_dataset = PoseSequenceDataset(
                args.data_root, 
                split="train", 
                object_name=args.object, 
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
            )
            val_dataset = train_dataset
    else:
        train_dataset = PoseSequenceDataset(
            args.data_root, 
            split="train", 
            img_size=args.img_size,
            frame_skip=args.frame_skip,
            center_crop=args.center_crop,
        )
        try:
            val_dataset = PoseSequenceDataset(
                args.data_root, 
                split="test", 
                img_size=args.img_size,
                frame_skip=args.frame_skip,
                center_crop=args.center_crop,
            )
        except:
            print("No test split found, using train for validation")
            val_dataset = train_dataset
    
    print(f"Train samples: {len(train_dataset)}")
    
    # pin_memory only helps on CUDA; num_workers=0 is safest on MPS/CPU
    use_cuda = device.type == "cuda"
    n_workers = 4 if use_cuda else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=use_cuda,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=use_cuda,
    )
    
    # Create model
    model = PhaseAModel(
        num_keypoints=args.num_keypoints,
        base_channels=args.base_channels,
        temperature=args.temperature,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    history = []
    best_loss = float('inf')
    
    # Full config dict saved into every checkpoint for reproducibility
    ckpt_config = {
        'num_keypoints': args.num_keypoints,
        'base_channels': args.base_channels,
        'temperature': args.temperature,
        'frame_skip': args.frame_skip,
        'yaw_step_deg': args.yaw_step_deg,
        'lambda_smooth': args.lambda_smooth,
        'lambda_disp': args.lambda_disp,
        'lambda_ent': args.lambda_ent,
        'sigma': args.sigma,
        'img_size': args.img_size,
        'seed': args.seed,
    }
    
    eff_deg = args.frame_skip * args.yaw_step_deg
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Loss weights: lambda_smooth={args.lambda_smooth}, lambda_disp={args.lambda_disp}, "
          f"lambda_ent={args.lambda_ent}, sigma={args.sigma}")
    print(f"Frame skip: {args.frame_skip} (effective step: {eff_deg:.0f} deg)")
    print("-" * 70)
    
    # ---- First-batch diagnostic (one-time) ----
    with torch.no_grad():
        diag_batch = next(iter(train_loader))
        diag_x_t = diag_batch['x_t'].to(device)
        diag_x_t1 = diag_batch['x_t1'].to(device)
        diag_out = model(diag_x_t, diag_x_t1)
        diag_losses = compute_losses(
            diag_out, lambda_smooth=args.lambda_smooth, lambda_disp=args.lambda_disp,
            lambda_ent=args.lambda_ent, sigma=args.sigma, num_keypoints=args.num_keypoints,
        )
        print(f"\n[Diagnostic] First batch (before training):")
        print(f"  p_t  range: [{diag_out['p_t'].min().item():.4f}, {diag_out['p_t'].max().item():.4f}]")
        print(f"  l_pred:  {diag_losses['l_pred'].item():.6f}")
        print(f"  l_disp:  {diag_losses['l_disp'].item():.6f}")
        print(f"  l_smooth:{diag_losses['l_smooth'].item():.6f}")
        print(f"  l_ent:   {diag_losses['l_ent'].item():.6f}")
        print(f"  total:   {diag_losses['loss'].item():.6f}")
        print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda_smooth, args.lambda_disp, args.lambda_ent, args.sigma, args.num_keypoints
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if epoch % args.log_every == 0 or epoch == 1:
            val_metrics = evaluate(
                model, val_loader, device,
                args.lambda_smooth, args.lambda_disp, args.lambda_ent, args.sigma, args.num_keypoints
            )
            
            print(f"Epoch {epoch:4d} | "
                  f"Loss: {train_metrics['loss']:.5f} | "
                  f"L_pred: {train_metrics['l_pred']:.5f} | "
                  f"L_smooth: {train_metrics['l_smooth']:.5f} | "
                  f"L_disp: {train_metrics['l_disp']:.5f} | "
                  f"L_ent: {train_metrics['l_ent']:.5f} | "
                  f"Val: {val_metrics['loss']:.5f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_pred': train_metrics['l_pred'],
                'train_smooth': train_metrics['l_smooth'],
                'train_disp': train_metrics['l_disp'],
                'train_ent': train_metrics['l_ent'],
                'val_loss': val_metrics['loss'],
                'val_pred': val_metrics['l_pred'],
                'lr': scheduler.get_last_lr()[0],
            })
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': ckpt_config,
                }, run_dir / "best_model.pt")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'config': ckpt_config,
            }, run_dir / f"checkpoint_{epoch:05d}.pt")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_metrics['loss'],
        'config': ckpt_config,
    }, run_dir / "final_model.pt")
    
    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("-" * 60)
    print(f"Training complete! Best validation loss: {best_loss:.6f}")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
