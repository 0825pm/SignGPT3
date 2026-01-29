"""
Part Mamba Denoiser for SignGPT3 Stage 2
========================================

Light-T2M의 Mamba 기반 Denoiser를 VAE latent space에 적용
Part axis (body, lhand, rhand)를 시퀀스로 처리

Input:  z_noisy [B, 3, 256] + text_emb [B, 512] + timestep [B]
Output: z_clean [B, 3, 256]

Light-T2M에서 차용:
- StageBlock (Mamba + LocalConv + Text injection)
- Bidirectional Mamba
- DDPM with sample prediction

복사 위치: motGPT/archs/part_mamba_denoiser.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from functools import partial
from einops import rearrange, repeat

# Mamba import
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[PartMambaDenoiser] Warning: mamba_ssm not available")


# =============================================================================
# Embedding Utilities (from Light-T2M)
# =============================================================================

def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Sinusoidal timestep embedding (from Light-T2M)
    
    Args:
        timesteps: [B] tensor of timesteps
        dim: embedding dimension
        
    Returns:
        emb: [B, dim] tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PositionEmbedding(nn.Module):
    """
    Learnable or sinusoidal position embedding
    """
    def __init__(self, max_len: int, dim: int, dropout: float = 0.1, learnable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        else:
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x: [B, T, D] with position embedding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# Mamba Block (from Light-T2M)
# =============================================================================

def create_mamba_block(
    d_model: int,
    ssm_cfg: Optional[Dict] = None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = True,
    pre_norm: bool = False,
):
    """Create a Mamba block with given config"""
    if ssm_cfg is None:
        ssm_cfg = {}
    
    mamba = Mamba(
        d_model=d_model,
        d_state=ssm_cfg.get('d_state', 16),
        d_conv=ssm_cfg.get('d_conv', 4),
        expand=ssm_cfg.get('expand', 2),
    )
    return mamba


# =============================================================================
# Stage Block (from Light-T2M, adapted for part-wise processing)
# =============================================================================

class PartStageBlock(nn.Module):
    """
    Light-T2M StageBlock adapted for part-wise latent processing
    
    원본 Light-T2M:
    - Local Conv (시간축 local 정보)
    - Text injection (gating)
    - Bidirectional Mamba (global)
    
    SignGPT3 적용:
    - Part axis가 3으로 짧아서 Local Conv 대신 Linear 사용
    - Text injection 유지
    - Bidirectional Mamba 유지
    """
    
    def __init__(
        self,
        dim: int,
        text_dim: int,
        ssm_cfg: Optional[Dict] = None,
        num_groups: int = 8,
        use_local_conv: bool = False,  # Part axis가 짧아서 optional
    ):
        super().__init__()
        
        self.dim = dim
        self.use_local_conv = use_local_conv
        
        # Local processing (optional for short sequences)
        if use_local_conv:
            self.local_conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=min(num_groups, dim)),
                nn.SiLU(),
            )
        else:
            self.local_fc = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
            )
        
        # Text injection (gating mechanism from Light-T2M)
        self.f_func = nn.Linear(dim * 2, dim)
        self.fuse_fn = nn.Linear(dim * 2, dim)
        
        # Bidirectional Mamba
        if MAMBA_AVAILABLE:
            self.mamba_fwd = create_mamba_block(dim, ssm_cfg)
            self.mamba_bwd = create_mamba_block(dim, ssm_cfg)
        else:
            # Fallback: simple MLP
            self.mamba_fwd = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )
            self.mamba_bwd = self.mamba_fwd
        
        # Final projection
        self.final_fc = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
    
    def inject_text(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Text injection with gating (from Light-T2M)
        
        Args:
            x: [B, L, D] motion/latent features
            y: [B, D] text embedding (already projected)
            
        Returns:
            x_hat: [B, L, D] text-injected features
        """
        # y: [B, D] → [B, L, D]
        y_repeat = y.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        # Gating
        y_hat = self.f_func(torch.cat([x, y_repeat], dim=-1))
        _y_hat = y_repeat * torch.sigmoid(y_hat)
        
        # Fuse
        x_hat = self.fuse_fn(torch.cat([x, _y_hat], dim=-1))
        return x_hat
    
    def forward(self, x: Tensor, text_emb: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, D] latent features (L=3 for parts, or L=5 with time+text tokens)
            text_emb: [B, D] projected text embedding
            
        Returns:
            x: [B, L, D] processed features
        """
        residual = x
        
        # 1. Local processing
        if self.use_local_conv:
            x_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            x_local = self.local_fc(x)
        
        # 2. Text injection
        x_text = self.inject_text(x_local, text_emb)
        
        # 3. Bidirectional Mamba
        if MAMBA_AVAILABLE:
            x_fwd = self.mamba_fwd(x_text)
            x_bwd = self.mamba_bwd(x_text.flip(1)).flip(1)
        else:
            x_fwd = self.mamba_fwd(x_text)
            x_bwd = self.mamba_bwd(x_text.flip(1)).flip(1)
        
        # 4. Combine forward and backward
        x_out = self.final_fc(torch.cat([x_fwd, x_bwd], dim=-1))
        
        # 5. Residual + Norm
        x_out = self.norm(x_out + residual)
        
        return x_out


# =============================================================================
# Part Mamba Denoiser (Main Class)
# =============================================================================

class PartMambaDenoiser(nn.Module):
    """
    Part-aware Mamba Denoiser for SignGPT3 Stage 2
    
    Architecture:
        z_noisy [B, 3, 256] (body, lhand, rhand)
              ↓
        [z_input_proj] → [B, 3, dim]
              ↓
        + timestep_emb [B, 1, dim]
        + text_emb [B, 1, dim]
              ↓
        concat → [B, 5, dim] (time, text, body, lhand, rhand)
              ↓
        [PartStageBlock] × num_layers
              ↓
        extract z tokens → [B, 3, dim]
              ↓
        [z_output_proj] → z_clean [B, 3, 256]
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_parts: int = 3,
        dropout: float = 0.1,
        ssm_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_parts = num_parts
        
        # Default SSM config (from Light-T2M)
        if ssm_cfg is None:
            ssm_cfg = {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
            }
        
        # Input projections
        self.z_input_proj = nn.Linear(latent_dim, hidden_dim)
        self.t_input_proj = nn.Linear(text_dim, hidden_dim)
        self.time_emb_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Part embedding (learnable tokens for body, lhand, rhand)
        self.part_emb = nn.Parameter(torch.randn(1, num_parts, hidden_dim) * 0.02)
        
        # Position embedding for the full sequence (time + text + 3 parts = 5)
        self.pos_emb = PositionEmbedding(max_len=10, dim=hidden_dim, dropout=dropout)
        
        # Mamba StageBlocks
        self.layers = nn.ModuleList([
            PartStageBlock(
                dim=hidden_dim,
                text_dim=hidden_dim,
                ssm_cfg=ssm_cfg,
                use_local_conv=False,  # Short sequence, use Linear
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.z_output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        z_noisy: Tensor,
        timestep: Tensor,
        text_emb: Tensor,
    ) -> Tensor:
        """
        Denoise z_noisy conditioned on text and timestep
        
        Args:
            z_noisy: [B, 3, latent_dim] noisy latent (body, lhand, rhand)
            timestep: [B] diffusion timestep
            text_emb: [B, text_dim] text embedding from M-CLIP
            
        Returns:
            z_pred: [B, 3, latent_dim] predicted clean latent (sample prediction)
        """
        B = z_noisy.shape[0]
        
        # 1. Project inputs
        z_feat = self.z_input_proj(z_noisy)  # [B, 3, hidden_dim]
        z_feat = z_feat + self.part_emb  # Add part embedding
        
        text_feat = self.t_input_proj(text_emb)  # [B, hidden_dim]
        text_feat = text_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        
        time_feat = timestep_embedding(timestep, self.hidden_dim)  # [B, hidden_dim]
        time_feat = self.time_emb_proj(time_feat)
        time_feat = time_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 2. Concat: [time, text, body, lhand, rhand] → [B, 5, hidden_dim]
        x = torch.cat([time_feat, text_feat, z_feat], dim=1)  # [B, 5, hidden_dim]
        
        # 3. Position embedding
        x = self.pos_emb(x)
        x = self.dropout(x)
        
        # 4. Extract text for conditioning (after pos_emb)
        text_cond = x[:, 1, :]  # [B, hidden_dim]
        
        # 5. Pass through Mamba StageBlocks
        for layer in self.layers:
            x = layer(x, text_cond)
        
        # 6. Extract z tokens (skip time and text tokens)
        z_out = x[:, 2:, :]  # [B, 3, hidden_dim]
        
        # 7. Project to latent dim
        z_pred = self.z_output_proj(z_out)  # [B, 3, latent_dim]
        
        return z_pred


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PartMambaDenoiser(
        latent_dim=256,
        text_dim=512,
        hidden_dim=256,
        num_layers=4,
    ).to(device)
    
    # Test input
    B = 4
    z_noisy = torch.randn(B, 3, 256).to(device)
    timestep = torch.randint(0, 1000, (B,)).to(device)
    text_emb = torch.randn(B, 512).to(device)
    
    # Forward
    z_pred = model(z_noisy, timestep, text_emb)
    
    print(f"Input z_noisy: {z_noisy.shape}")
    print(f"Input timestep: {timestep.shape}")
    print(f"Input text_emb: {text_emb.shape}")
    print(f"Output z_pred: {z_pred.shape}")
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params / 1e6:.2f}M")
