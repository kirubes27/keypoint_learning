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
        padding_mode: str = "reflect",
        heatmap_res: int = 64,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        self.padding_mode = padding_mode
        self.heatmap_res = heatmap_res
        
        # 4-layer CNN encoder (using stride=2 for downsampling)
        self.encoder = nn.Sequential(
            # Layer 1: 256 -> 128
            # Use reflect padding to reduce border artifacts in heatmaps.
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2: 128 -> 64
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 -> 32
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Layer 4: 32 -> 32 (preserve resolution)
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap head. heatmap_res=64 (legacy): 1x1 conv on the /8 feature
        # map (64x64 at 512 input). heatmap_res=128: bilinear FEATURE
        # upsample + 3x3 conv computing new features at /4 resolution before
        # the 1x1 head — a real higher-resolution head, not a resized
        # 64-res heatmap (soft-argmax cell size halves: 0.03125 -> 0.015625).
        if heatmap_res == 128:
            self.head_upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(base_channels * 4, base_channels * 2,
                          kernel_size=3, padding=1,
                          padding_mode=padding_mode),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
            )
            self.heatmap_head = nn.Conv2d(base_channels * 2, num_keypoints,
                                          kernel_size=1)
        elif heatmap_res == 64:
            self.head_upsample = None
            self.heatmap_head = nn.Conv2d(base_channels * 4, num_keypoints,
                                          kernel_size=1)
        else:
            raise ValueError(f"heatmap_res must be 64 or 128, got {heatmap_res}")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W) input image
        
        Returns:
            keypoints: (B, 2*N) flattened keypoint coordinates
            heatmaps: (B, N, H', W') intermediate heatmaps (for visualization/entropy)
        """
        features = self.encoder(x)  # (B, 128, H/8, W/8)
        if self.head_upsample is not None:
            features = self.head_upsample(features)  # (B, 64, H/4, W/4)
        heatmaps = self.heatmap_head(features)  # (B, N, H/8 or H/4, ...)
        
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


class SharedAffineOperator(nn.Module):
    """
    Shared 2D affine operator applied independently to every keypoint:

        p'_{i} = A p_i + b

    This is the strict "same rule across keypoints" baseline. It prevents the
    operator from mixing keypoint identities or using one keypoint coordinate to
    predict another keypoint coordinate.
    """

    def __init__(self, num_keypoints: int):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.A = nn.Parameter(torch.eye(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        B = p_t.shape[0]
        pts = p_t.reshape(B, self.num_keypoints, 2)
        pred = torch.matmul(pts, self.A.t()) + self.bias.view(1, 1, 2)
        return pred.reshape(B, 2 * self.num_keypoints)

    @property
    def W(self) -> torch.Tensor:
        """Return the equivalent 2N x 2N block-diagonal matrix for diagnostics."""
        eye = torch.eye(self.num_keypoints, device=self.A.device, dtype=self.A.dtype)
        return torch.kron(eye, self.A)

    @property
    def b(self) -> torch.Tensor:
        """Return the equivalent flattened bias vector for diagnostics."""
        return self.bias.repeat(self.num_keypoints)


def make_operator(operator_type: str, keypoint_dim: int, num_keypoints: int) -> nn.Module:
    """Construct an operator by name while keeping dense as the legacy default."""
    if operator_type == "dense":
        return LinearOperator(keypoint_dim)
    if operator_type == "shared_affine":
        return SharedAffineOperator(num_keypoints)
    raise ValueError(f"Unknown operator_type='{operator_type}'")


class ActionClassifier(nn.Module):
    """
    Linear action head over keypoint deltas.

    Given delta_k = p_{t+1} - p_t, predicts action class logits.
    Keeping this linear tests whether action information is linearly decodable
    from keypoint motion.
    """

    def __init__(self, keypoint_dim: int, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.linear = nn.Linear(keypoint_dim, num_classes)

    def forward(self, delta_k: torch.Tensor) -> torch.Tensor:
        return self.linear(delta_k)


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
        num_action_classes: int = 0,
        padding_mode: str = "reflect",
        operator_type: str = "dense",
        learn_inverse_operator: bool = False,
        heatmap_res: int = 64,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoint_dim = 2 * num_keypoints
        self.num_action_classes = num_action_classes
        self.operator_type = operator_type
        self.learn_inverse_operator = learn_inverse_operator

        self.extractor = KeypointExtractor(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            base_channels=base_channels,
            temperature=temperature,
            padding_mode=padding_mode,
            heatmap_res=heatmap_res,
        )

        self.operator = make_operator(operator_type, self.keypoint_dim, num_keypoints)
        if learn_inverse_operator:
            self.inverse_operator = make_operator(operator_type, self.keypoint_dim, num_keypoints)
        else:
            self.inverse_operator = None
        if num_action_classes > 0:
            self.action_classifier = ActionClassifier(self.keypoint_dim, num_action_classes)
        else:
            self.action_classifier = None
    
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

        outputs = {
            'p_t': p_t,                # Current keypoints (B, 2N)
            'p_t1': p_t1,              # Next keypoints - ground truth (B, 2N)
            'p_hat_t1': p_hat_t1,      # Predicted next keypoints (B, 2N)
            'heatmaps_t': heatmaps_t,
            'heatmaps_t1': heatmaps_t1,
        }

        if self.action_classifier is not None:
            delta_k = p_t1 - p_t
            outputs['delta_k'] = delta_k
            outputs['action_logits'] = self.action_classifier(delta_k)

        if self.inverse_operator is not None:
            # Direct inverse prediction and one-step cycle diagnostics:
            # W- K_{t+1} ~= K_t, W- W+ K_t ~= K_t, W+ W- K_{t+1} ~= K_{t+1}.
            p_hat_t0 = self.inverse_operator(p_t1)
            outputs['p_hat_t0'] = p_hat_t0
            outputs['p_cycle_t'] = self.inverse_operator(p_hat_t1)
            outputs['p_cycle_t1'] = self.operator(p_hat_t0)

        return outputs
    
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


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization used by dataset transforms."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _background_subtraction_mask_from_normalized(
    x_norm: torch.Tensor,
    bg_threshold: float = 30.0,
    corner_frac: float = 0.125,
) -> torch.Tensor:
    """
    Build a per-image foreground mask from normalized images using corner-color
    background subtraction, matching eval_rollout_viz.py semantics.

    Args:
        x_norm: (B,3,H,W) normalized image tensor (ImageNet normalization)
        bg_threshold: pixel-distance threshold on 0-255 scale
        corner_frac: fraction of image side used for corner patch size

    Returns:
        mask: (B,H,W) float tensor in {0,1}
    """
    x = _denormalize_imagenet(x_norm)  # (B,3,H,W) in [0,1]
    B, _, H, W = x.shape
    m = max(1, int(min(H, W) * corner_frac))

    c1 = x[:, :, :m, :m].mean(dim=(2, 3))
    c2 = x[:, :, :m, -m:].mean(dim=(2, 3))
    c3 = x[:, :, -m:, :m].mean(dim=(2, 3))
    c4 = x[:, :, -m:, -m:].mean(dim=(2, 3))
    bg = (c1 + c2 + c3 + c4) / 4.0  # (B,3)

    diff = torch.linalg.norm(x - bg.unsqueeze(-1).unsqueeze(-1), dim=1)  # (B,H,W), [0, sqrt(3)]
    thr = bg_threshold / 255.0
    mask = (diff > thr).to(x.dtype)
    return mask


def compute_localization_loss(
    p_t: torch.Tensor,
    p_t1: torch.Tensor,
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    num_keypoints: int,
    bg_threshold: float = 30.0,
) -> torch.Tensor:
    """
    Penalize keypoints landing on background using differentiable mask sampling.

    Loss:
        L_loc = -log(M_t(p_t)) - log(M_{t+1}(p_{t+1}))
    where M(.) is foreground probability map from background subtraction.
    """
    B = p_t.shape[0]
    p_t_reshaped = p_t.view(B, num_keypoints, 2)    # (B,N,2), [-1,1]
    p_t1_reshaped = p_t1.view(B, num_keypoints, 2)  # (B,N,2), [-1,1]

    m_t = _background_subtraction_mask_from_normalized(x_t, bg_threshold=bg_threshold).unsqueeze(1)   # (B,1,H,W)
    m_t1 = _background_subtraction_mask_from_normalized(x_t1, bg_threshold=bg_threshold).unsqueeze(1)  # (B,1,H,W)

    # grid_sample expects grid (B, out_h, out_w, 2); use out_h=N, out_w=1.
    g_t = p_t_reshaped.unsqueeze(2)    # (B,N,1,2)
    g_t1 = p_t1_reshaped.unsqueeze(2)  # (B,N,1,2)

    s_t = F.grid_sample(m_t, g_t, mode="bilinear", align_corners=True).squeeze(1).squeeze(-1)    # (B,N)
    s_t1 = F.grid_sample(m_t1, g_t1, mode="bilinear", align_corners=True).squeeze(1).squeeze(-1)  # (B,N)

    eps = 1e-6
    l_t = -torch.log(s_t.clamp_min(eps)).mean()
    l_t1 = -torch.log(s_t1.clamp_min(eps)).mean()
    return 0.5 * (l_t + l_t1)


def compute_losses(
    outputs: dict,
    lambda_smooth: float = 0.1,
    lambda_disp: float = 0.1,
    lambda_ent: float = 0.1,
    lambda_act: float = 0.0,
    lambda_loc: float = 0.0,
    lambda_inv: float = 0.0,
    lambda_cycle: float = 0.0,
    sigma: float = 0.1,
    num_keypoints: int = 10,
    action_labels: torch.Tensor = None,
    x_t: torch.Tensor = None,
    x_t1: torch.Tensor = None,
    loc_bg_threshold: float = 30.0,
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
        L = L_pred + λ_s·L_smooth + λ_d·L_disp + λ_e·L_ent + λ_a·L_act
            + λ_l·L_loc + λ_i·L_inv + λ_c·L_cycle
    
    Args:
        outputs: dict from PhaseAModel.forward()
        lambda_smooth: weight for L_smooth
        lambda_disp: weight for L_disp  
        lambda_ent: weight for L_ent
        lambda_act: weight for L_act
        lambda_inv: weight for inverse prediction W- K_{t+1} ~= K_t
        lambda_cycle: weight for one-step cycle consistency
        sigma: length scale for dispersion (10.3 specifies σ² denominator)
        num_keypoints: N
        action_labels: (B,) class labels for action prediction, if using action head
        x_t, x_t1: normalized input frames for localization loss (optional)
        loc_bg_threshold: threshold for foreground mask in localization loss

    Returns:
        dict with loss and component metrics.
    """
    p_t = outputs['p_t']              # (B, 2N)
    p_t1 = outputs['p_t1']            # (B, 2N)
    p_hat_t1 = outputs['p_hat_t1']    # (B, 2N)
    heatmaps_t1 = outputs['heatmaps_t1']  # (B, N, H', W')

    if lambda_inv > 0.0 and 'p_hat_t0' not in outputs:
        raise ValueError("lambda_inv > 0 requires PhaseAModel(learn_inverse_operator=True)")
    if lambda_cycle > 0.0 and ('p_cycle_t' not in outputs or 'p_cycle_t1' not in outputs):
        raise ValueError("lambda_cycle > 0 requires PhaseAModel(learn_inverse_operator=True)")
    
    B = p_t.shape[0]

    # When backward pairs are included for action supervision, keep the
    # operator objective tied to a single fixed action (forward/yaw+).
    if action_labels is None:
        forward_mask = torch.ones(B, dtype=torch.bool, device=p_t.device)
    else:
        forward_mask = action_labels.long() == 0
    
    # =========================================================================
    # L_pred: Main prediction loss
    # ||p̂_{t+1} - p_{t+1}||²
    # =========================================================================
    if forward_mask.any():
        l_pred = F.mse_loss(p_hat_t1[forward_mask], p_t1[forward_mask])
    else:
        l_pred = p_t.new_tensor(0.0)
    
    # =========================================================================
    # L_smooth: Temporal smoothness regularizer
    # ||p_{t+1} - p_t||²
    # Prevents temporal jitter and identity swapping
    # =========================================================================
    if forward_mask.any():
        l_smooth = F.mse_loss(p_t1[forward_mask], p_t[forward_mask])
    else:
        l_smooth = p_t.new_tensor(0.0)
    
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
    # L_act: Action classification loss on keypoint deltas
    # CE(h(delta_k), action_label)
    # =========================================================================
    l_act = p_t.new_tensor(0.0)
    act_acc = p_t.new_tensor(0.0)
    if 'action_logits' in outputs and action_labels is not None:
        l_act = F.cross_entropy(outputs['action_logits'], action_labels.long())
        preds = outputs['action_logits'].argmax(dim=-1)
        act_acc = (preds == action_labels.long()).float().mean()

    # =========================================================================
    # L_loc: localization loss (foreground consistency for keypoint coordinates)
    # =========================================================================
    l_loc = p_t.new_tensor(0.0)
    if lambda_loc > 0.0 and x_t is not None and x_t1 is not None:
        l_loc = compute_localization_loss(
            p_t=p_t,
            p_t1=p_t1,
            x_t=x_t,
            x_t1=x_t1,
            num_keypoints=num_keypoints,
            bg_threshold=loc_bg_threshold,
        )

    # =========================================================================
    # L_inv / L_cycle: inverse and one-step cycle consistency
    # =========================================================================
    l_inv = p_t.new_tensor(0.0)
    if 'p_hat_t0' in outputs and forward_mask.any():
        l_inv = F.mse_loss(outputs['p_hat_t0'][forward_mask], p_t[forward_mask])

    l_cycle = p_t.new_tensor(0.0)
    if 'p_cycle_t' in outputs and 'p_cycle_t1' in outputs and forward_mask.any():
        l_cycle_t = F.mse_loss(outputs['p_cycle_t'][forward_mask], p_t[forward_mask])
        l_cycle_t1 = F.mse_loss(outputs['p_cycle_t1'][forward_mask], p_t1[forward_mask])
        l_cycle = 0.5 * (l_cycle_t + l_cycle_t1)

    # =========================================================================
    # Total loss
    # L = L_pred + λ_s·L_smooth + λ_d·L_disp + λ_e·L_ent + λ_a·L_act
    #     + λ_l·L_loc + λ_i·L_inv + λ_c·L_cycle
    # =========================================================================
    l_total = (
        l_pred
        + lambda_smooth * l_smooth
        + lambda_disp * l_disp
        + lambda_ent * l_ent
        + lambda_act * l_act
        + lambda_loc * l_loc
        + lambda_inv * l_inv
        + lambda_cycle * l_cycle
    )

    return {
        'loss': l_total,
        'l_pred': l_pred,
        'l_smooth': l_smooth,
        'l_disp': l_disp,
        'l_ent': l_ent,
        'l_act': l_act,
        'l_loc': l_loc,
        'l_inv': l_inv,
        'l_cycle': l_cycle,
        'act_acc': act_acc,
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
    print(f"  L_act:    {losses['l_act'].item():.4f}")
    print(f"  L_inv:    {losses['l_inv'].item():.4f}")
    print(f"  L_cycle:  {losses['l_cycle'].item():.4f}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test multi-step prediction
    p_0 = outputs['p_t']
    p_3 = model.multi_step_predict(p_0, k=3)
    print(f"\nMulti-step prediction (k=3): {p_3.shape}")

    # Test optional action head
    model_act = PhaseAModel(num_keypoints=N, num_action_classes=2)
    outputs_act = model_act(x_t, x_t1)
    action_labels = torch.randint(0, 2, (B,))
    losses_act = compute_losses(
        outputs_act,
        lambda_act=0.1,
        num_keypoints=N,
        sigma=0.1,
        action_labels=action_labels,
    )
    print(f"\nAction head test:")
    print(f"  action_logits shape: {outputs_act['action_logits'].shape}")
    print(f"  L_act: {losses_act['l_act'].item():.4f}")

    # Test shared operator + inverse/cycle path.
    model_shared = PhaseAModel(
        num_keypoints=N,
        operator_type="shared_affine",
        learn_inverse_operator=True,
    )
    outputs_shared = model_shared(x_t, x_t1)
    losses_shared = compute_losses(
        outputs_shared,
        lambda_inv=0.1,
        lambda_cycle=0.1,
        num_keypoints=N,
        sigma=0.1,
    )
    print(f"\nShared affine + inverse test:")
    print(f"  p_hat_t1 shape: {outputs_shared['p_hat_t1'].shape}")
    print(f"  W shape: {model_shared.operator.W.shape}")
    print(f"  L_inv: {losses_shared['l_inv'].item():.4f}")
    print(f"  L_cycle: {losses_shared['l_cycle'].item():.4f}")
