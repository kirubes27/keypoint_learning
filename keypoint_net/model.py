"""
Phase A Models: Keypoint Extractor + Linear Operator

From the Modeling Plan (10.2, 10.3):
- Keypoint extractor K(x): Small CNN (4-5 conv layers) → N heatmaps → soft-argmax → K(x) ∈ ℝ^(2N)
- Operator/pose predictor: Single linear layer p̂_{t+1} = W·p_t + b (no nonlinearity)
- Losses: L_pred + λ_s·L_smooth + λ_d·L_disp + λ_e·L_ent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_softmax(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Soft-argmax over spatial dimensions to extract (x, y) coordinates from heatmaps.
    
    Args:
        heatmaps: (B, N, H, W) tensor of N keypoint heatmaps
        temperature: Lower = sharper peaks, higher = smoother
    
    Returns:
        coords: (B, N, 2) tensor of (x, y) coordinates in [-1, 1] range
    """
    B, N, H, W = heatmaps.shape
    
    # Flatten spatial dimensions and apply softmax
    flat = heatmaps.view(B, N, -1)  # (B, N, H*W)
    weights = F.softmax(flat / temperature, dim=-1)  # (B, N, H*W)
    
    # Create coordinate grids (normalized to [-1, 1])
    device = heatmaps.device
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten coordinate grids
    xx_flat = xx.reshape(-1)  # (H*W,)
    yy_flat = yy.reshape(-1)  # (H*W,)
    
    # Compute expected coordinates (soft-argmax)
    x = (weights * xx_flat).sum(dim=-1)  # (B, N)
    y = (weights * yy_flat).sum(dim=-1)  # (B, N)
    
    coords = torch.stack([x, y], dim=-1)  # (B, N, 2)
    return coords


class KeypointExtractor(nn.Module):
    """
    Small CNN that outputs N keypoint heatmaps, then applies soft-argmax.
    
    Architecture (10.2): 4-5 conv blocks → N heatmaps H ∈ ℝ^{N×h×w} → soft-argmax → p_t
    Output: K(x) ∈ ℝ^(2N) — flattened (x,y) coordinates for N keypoints.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 10,
        base_channels: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        
        # 4-layer CNN encoder (using stride=2 for downsampling)
        self.encoder = nn.Sequential(
            # Layer 1: 256 -> 128
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2: 128 -> 64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 -> 32
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Layer 4: 32 -> 32 (preserve resolution)
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap head: 1x1 conv to N channels
        self.heatmap_head = nn.Conv2d(base_channels * 4, num_keypoints, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W) input image
        
        Returns:
            keypoints: (B, 2*N) flattened keypoint coordinates
            heatmaps: (B, N, H', W') intermediate heatmaps (for visualization/entropy)
        """
        features = self.encoder(x)  # (B, 128, H/8, W/8)
        heatmaps = self.heatmap_head(features)  # (B, N, H/8, W/8)
        
        # Soft-argmax to get coordinates
        coords = spatial_softmax(heatmaps, self.temperature)  # (B, N, 2)
        
        # Flatten to (B, 2*N) as specified: K(x) ∈ ℝ^(2N)
        keypoints = coords.view(coords.shape[0], -1)  # (B, 2*N)
        
        return keypoints, heatmaps
    
    def get_keypoint_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method to get (B, N, 2) coordinates."""
        keypoints, _ = self.forward(x)
        return keypoints.view(x.shape[0], self.num_keypoints, 2)


class LinearOperator(nn.Module):
    """
    Single linear layer (perceptron): p̂_{t+1} = W·p_t + b
    
    No nonlinearity, no hidden layers.
    Interpretable as globally linear operator in keypoint space.
    """
    
    def __init__(self, keypoint_dim: int):
        """
        Args:
            keypoint_dim: 2*N (flattened keypoint coordinates)
        """
        super().__init__()
        # Single linear layer: W is (2N x 2N), b is (2N,)
        self.linear = nn.Linear(keypoint_dim, keypoint_dim)
    
    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        """
        Predict next keypoints from current keypoints.
        
        Args:
            p_t: (B, 2*N) current keypoints
        
        Returns:
            p_hat_t1: (B, 2*N) predicted next keypoints
        """
        return self.linear(p_t)
    
    @property
    def W(self) -> torch.Tensor:
        """Get the weight matrix for analysis."""
        return self.linear.weight
    
    @property
    def b(self) -> torch.Tensor:
        """Get the bias vector for analysis."""
        return self.linear.bias


class PhaseAModel(nn.Module):
    """
    Combined model for Phase A training (10.1).
    
    Objective: Learn chart K(·) such that a single global operator predicts
    keypoint evolution under pose changes.
    
    Forward pass:
    1. Extract keypoints from x_t: p_t = K(x_t)
    2. Extract keypoints from x_{t+1}: p_{t+1} = K(x_{t+1})
    3. Predict p̂_{t+1} = W·p_t + b
    4. Compute losses
    """
    
    def __init__(
        self,
        num_keypoints: int = 10,
        in_channels: int = 3,
        base_channels: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoint_dim = 2 * num_keypoints
        
        self.extractor = KeypointExtractor(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            base_channels=base_channels,
            temperature=temperature,
        )
        
        self.operator = LinearOperator(self.keypoint_dim)
    
    def forward(self, x_t: torch.Tensor, x_t1: torch.Tensor) -> dict:
        """
        Forward pass for a training pair (x_t, x_{t+1}).
        
        Returns dict with all quantities needed for loss computation.
        """
        # Extract keypoints from both frames
        p_t, heatmaps_t = self.extractor(x_t)      # (B, 2N), (B, N, H', W')
        p_t1, heatmaps_t1 = self.extractor(x_t1)  # (B, 2N), (B, N, H', W')
        
        # Predict next keypoints using linear operator
        p_hat_t1 = self.operator(p_t)  # (B, 2N)
        
        return {
            'p_t': p_t,                # Current keypoints (B, 2N)
            'p_t1': p_t1,              # Next keypoints - ground truth (B, 2N)
            'p_hat_t1': p_hat_t1,      # Predicted next keypoints (B, 2N)
            'heatmaps_t': heatmaps_t,
            'heatmaps_t1': heatmaps_t1,
        }
    
    def multi_step_predict(self, p_0: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply operator k times: p̂_k = W^k p_0 + (W^{k-1} + ... + I)b
        
        Used for compositionality evaluation (10.5A).
        """
        p = p_0
        for _ in range(k):
            p = self.operator(p)
        return p


# =============================================================================
# Losses (10.3)
# =============================================================================

def compute_entropy_loss(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    L_ent: Heatmap sharpness / entropy loss.
    
    L_ent = Σ_i Entropy(H_i)
    
    Penalizes flat heatmaps — encourages sharp, localized peaks.
    Low entropy = sharp peak (good), High entropy = flat/diffuse (bad).
    
    Args:
        heatmaps: (B, N, H, W) raw heatmaps from CNN
    
    Returns:
        Mean entropy across all keypoints and batch
    """
    B, N, H, W = heatmaps.shape
    
    # Flatten spatial dimensions and convert to probabilities
    flat = heatmaps.view(B, N, -1)  # (B, N, H*W)
    probs = F.softmax(flat, dim=-1)  # (B, N, H*W)
    
    # Compute entropy: -Σ p log(p)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, N)
    
    return entropy.mean()


def compute_losses(
    outputs: dict,
    lambda_smooth: float = 0.1,
    lambda_disp: float = 0.1,
    lambda_ent: float = 0.1,
    sigma: float = 0.1,
    num_keypoints: int = 10,
) -> dict:
    """
    Compute all losses as specified in Phase A Modeling Plan (10.3).
    
    Main loss (predict next keypoints):
        L_pred = ||p̂_{t+1} - p_{t+1}||²
    
    Regularizers:
        L_smooth = ||p_{t+1} - p_t||²                    (temporal smoothness)
        L_disp = Σ_{i≠j} exp(-||p_i - p_j||² / σ²)       (dispersion/repulsion)
        L_ent = Σ_i Entropy(H_i)                          (heatmap sharpness)
    
    Total:
        L = L_pred + λ_s·L_smooth + λ_d·L_disp + λ_e·L_ent
    
    Args:
        outputs: dict from PhaseAModel.forward()
        lambda_smooth: weight for L_smooth
        lambda_disp: weight for L_disp  
        lambda_ent: weight for L_ent
        sigma: length scale for dispersion (10.3 specifies σ² denominator)
        num_keypoints: N
    
    Returns:
        dict with 'loss', 'l_pred', 'l_smooth', 'l_disp', 'l_ent'
    """
    p_t = outputs['p_t']              # (B, 2N)
    p_t1 = outputs['p_t1']            # (B, 2N)
    p_hat_t1 = outputs['p_hat_t1']    # (B, 2N)
    heatmaps_t1 = outputs['heatmaps_t1']  # (B, N, H', W')
    
    B = p_t.shape[0]
    
    # =========================================================================
    # L_pred: Main prediction loss
    # ||p̂_{t+1} - p_{t+1}||²
    # =========================================================================
    l_pred = F.mse_loss(p_hat_t1, p_t1)
    
    # =========================================================================
    # L_smooth: Temporal smoothness regularizer
    # ||p_{t+1} - p_t||²
    # Prevents temporal jitter and identity swapping
    # =========================================================================
    l_smooth = F.mse_loss(p_t1, p_t)
    
    # =========================================================================
    # L_disp: Dispersion regularizer (prevents keypoint collapse)
    # Σ_{i≠j} exp(-||p_i - p_j||² / σ²)
    # High when keypoints are close together → pushes them apart
    # σ controls the length scale of repulsion
    # =========================================================================
    p_t1_reshaped = p_t1.view(B, num_keypoints, 2)  # (B, N, 2)
    
    # Compute pairwise squared distances
    diff = p_t1_reshaped.unsqueeze(2) - p_t1_reshaped.unsqueeze(1)  # (B, N, N, 2)
    sq_dist = (diff ** 2).sum(dim=-1)  # (B, N, N)
    
    # exp(-||p_i - p_j||² / σ²) for i ≠ j
    sigma_sq = sigma ** 2
    exp_neg_dist = torch.exp(-sq_dist / sigma_sq)  # (B, N, N)
    
    # Mask out diagonal (i = j) and normalize by N*(N-1) so lambda_disp
    # is comparable across different num_keypoints values.
    mask = 1.0 - torch.eye(num_keypoints, device=p_t.device).unsqueeze(0)
    n_pairs = num_keypoints * (num_keypoints - 1)
    l_disp = (exp_neg_dist * mask).sum(dim=(1, 2)).mean() / n_pairs
    
    # =========================================================================
    # L_ent: Entropy loss (heatmap sharpness)
    # Σ_i Entropy(H_i)
    # Prevents flat heatmaps, encourages localized peaks
    # =========================================================================
    l_ent = compute_entropy_loss(heatmaps_t1)
    
    # =========================================================================
    # Total loss
    # L = L_pred + λ_s·L_smooth + λ_d·L_disp + λ_e·L_ent
    # =========================================================================
    l_total = l_pred + lambda_smooth * l_smooth + lambda_disp * l_disp + lambda_ent * l_ent
    
    return {
        'loss': l_total,
        'l_pred': l_pred,
        'l_smooth': l_smooth,
        'l_disp': l_disp,
        'l_ent': l_ent,
    }


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    B, C, H, W = 4, 3, 256, 256
    N = 10
    
    model = PhaseAModel(num_keypoints=N)
    x_t = torch.randn(B, C, H, W)
    x_t1 = torch.randn(B, C, H, W)
    
    outputs = model(x_t, x_t1)
    losses = compute_losses(outputs, num_keypoints=N, sigma=0.1)
    
    print("Model test passed!")
    print(f"  p_t shape: {outputs['p_t'].shape}")        # (B, 2N)
    print(f"  p_hat_t1 shape: {outputs['p_hat_t1'].shape}")  # (B, 2N)
    print(f"  heatmaps shape: {outputs['heatmaps_t'].shape}")  # (B, N, H', W')
    print(f"\nLosses:")
    print(f"  Total:    {losses['loss'].item():.4f}")
    print(f"  L_pred:   {losses['l_pred'].item():.4f}")
    print(f"  L_smooth: {losses['l_smooth'].item():.4f}")
    print(f"  L_disp:   {losses['l_disp'].item():.4f}")
    print(f"  L_ent:    {losses['l_ent'].item():.4f}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multi-step prediction
    p_0 = outputs['p_t']
    p_3 = model.multi_step_predict(p_0, k=3)
    print(f"\nMulti-step prediction (k=3): {p_3.shape}")
