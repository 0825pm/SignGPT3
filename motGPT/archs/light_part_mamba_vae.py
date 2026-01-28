"""
Lightweight Part-aware MambaVae
===============================

Shared encoder/decoder backbone + Part tokens
파라미터 수 ~1/3 감소 (vs 독립 Part encoders)

Latent: [B, 3, 256] - body, lhand, rhand

Key idea:
- 하나의 shared encoder가 전체 motion 처리
- 3개의 learnable part tokens가 각 파트 정보 수집
- Cross-attention으로 part tokens ↔ motion features 상호작용

Author: SignGPT3 Team
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from typing import List, Optional, Tuple

# Mamba import
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[LightPartMambaVae] mamba_ssm not found, using fallback")


# =============================================================================
# Constants
# =============================================================================
PART_SLICES = {
    'body': (0, 30),
    'lhand': (30, 75),
    'rhand': (75, 120),
}


def lengths_to_mask(lengths: List[int], device, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
    return mask


# =============================================================================
# Building Blocks
# =============================================================================
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.combine = nn.Linear(dim * 2, dim)
            self._is_mamba = True
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.SiLU(),
                nn.Linear(dim * expand, dim),
            )
            self._is_mamba = False
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        if self._is_mamba:
            fwd = self.mamba_fwd(x)
            bwd = self.mamba_bwd(x.flip(1)).flip(1)
            x = self.combine(torch.cat([fwd, bwd], dim=-1))
        else:
            x = self.mlp(x)
        return x + residual


class LocalModule(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        return x + residual


# =============================================================================
# Lightweight Part MambaVae
# =============================================================================
class LightPartMambaVae(nn.Module):
    """
    Lightweight Part-aware MambaVae
    
    Shared backbone + learnable part tokens
    파라미터: 단일 MambaVae의 ~1.1배 (vs Part MambaVae의 ~3배)
    
    Latent: [B, 3, 256]
    """
    
    def __init__(
        self,
        ablation,
        nfeats: int = 120,
        latent_dim: list = [3, 256],
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "encoder_decoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        datatype: str = 'h2s',
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        
        self.latent_size = latent_dim[0]  # 3
        self.latent_dim = latent_dim[-1]  # 256
        self.nfeats = nfeats
        
        # Ablation
        self.ablation = ablation
        self.pe_type = ablation.get('PE_TYPE', 'mld')
        self.mlp_dist = ablation.get('MLP_DIST', False)
        
        # Stats
        self.mean_std_inv = 0.8457
        self.mean_std_inv_2 = self.mean_std_inv ** 2
        self.mean_mean = -0.1379
        
        # =====================================================================
        # Encoder
        # =====================================================================
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(nfeats, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 512, self.latent_dim) * 0.02)
        
        # Part tokens (learnable) - 핵심!
        self.part_tokens = nn.Parameter(torch.randn(1, 3, self.latent_dim) * 0.02)
        
        # Part-specific input projection (lightweight)
        self.part_input_proj = nn.ModuleList([
            nn.Linear(PART_SLICES['body'][1] - PART_SLICES['body'][0], self.latent_dim),
            nn.Linear(PART_SLICES['lhand'][1] - PART_SLICES['lhand'][0], self.latent_dim),
            nn.Linear(PART_SLICES['rhand'][1] - PART_SLICES['rhand'][0], self.latent_dim),
        ])
        
        # Shared encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'local': LocalModule(self.latent_dim),
                'mamba': MambaBlock(self.latent_dim, d_state, d_conv, expand),
                'cross_attn': nn.MultiheadAttention(self.latent_dim, num_heads, dropout=dropout, batch_first=True),
                'cross_norm': nn.LayerNorm(self.latent_dim),
            }))
        
        self.encoder_norm = nn.LayerNorm(self.latent_dim)
        
        # Distribution
        self.to_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.to_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        
        # =====================================================================
        # Decoder
        # =====================================================================
        # Shared decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(self.latent_dim, num_heads, dropout=dropout, batch_first=True),
                'self_norm': nn.LayerNorm(self.latent_dim),
                'cross_attn': nn.MultiheadAttention(self.latent_dim, num_heads, dropout=dropout, batch_first=True),
                'cross_norm': nn.LayerNorm(self.latent_dim),
                'mamba': MambaBlock(self.latent_dim, d_state, d_conv, expand),
                'local': LocalModule(self.latent_dim),
            }))
        
        self.decoder_norm = nn.LayerNorm(self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, nfeats)
        
        # Part-specific output projection (lightweight)
        self.part_output_proj = nn.ModuleList([
            nn.Linear(self.latent_dim, PART_SLICES['body'][1] - PART_SLICES['body'][0]),
            nn.Linear(self.latent_dim, PART_SLICES['lhand'][1] - PART_SLICES['lhand'][0]),
            nn.Linear(self.latent_dim, PART_SLICES['rhand'][1] - PART_SLICES['rhand'][0]),
        ])
        
        self._print_info()
    
    def _print_info(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[LightPartMambaVae] Initialized:")
        print(f"  - latent: [{self.latent_size}, {self.latent_dim}]")
        print(f"  - params: {total/1e6:.2f}M")
        print(f"  - Mamba: {MAMBA_AVAILABLE}")
    
    def encode(self, features: Tensor, lengths: Optional[List[int]] = None) -> Tuple[Tensor, Normal]:
        """
        features: [B, T, 120]
        Returns: z [B, 3, 256], dist
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        
        B, T, _ = features.shape
        device = features.device
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Input projection
        x = self.input_proj(features)  # [B, T, D]
        x = x + self.pos_embed[:, :T, :]
        
        # Part tokens: [1, 3, D] → [B, 3, D]
        part_tokens = self.part_tokens.expand(B, -1, -1)
        
        # Add part-specific info to part tokens
        body = features[:, :, PART_SLICES['body'][0]:PART_SLICES['body'][1]]
        lhand = features[:, :, PART_SLICES['lhand'][0]:PART_SLICES['lhand'][1]]
        rhand = features[:, :, PART_SLICES['rhand'][0]:PART_SLICES['rhand'][1]]
        
        # Mean pooling per part → add to part tokens
        body_feat = (self.part_input_proj[0](body) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        lhand_feat = (self.part_input_proj[1](lhand) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        rhand_feat = (self.part_input_proj[2](rhand) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        
        part_tokens = part_tokens + torch.stack([body_feat, lhand_feat, rhand_feat], dim=1)
        
        # Encoder layers
        for layer in self.encoder_layers:
            # Motion features: local + mamba
            x = layer['local'](x)
            x = layer['mamba'](x)
            
            # Part tokens attend to motion features
            part_tokens_normed = layer['cross_norm'](part_tokens)
            attn_out, _ = layer['cross_attn'](part_tokens_normed, x, x, key_padding_mask=~mask)
            part_tokens = part_tokens + attn_out
        
        part_tokens = self.encoder_norm(part_tokens)
        
        # Distribution
        mu = self.to_mu(part_tokens)
        logvar = self.to_logvar(part_tokens)
        
        std = (logvar * 0.5).exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        
        return z, dist  # [B, 3, 256]
    
    def decode(self, z: Tensor, lengths: List[int]) -> Tensor:
        """
        z: [B, 3, 256]
        Returns: [B, T, 120]
        """
        B = z.shape[0]
        T = max(lengths)
        device = z.device
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Initialize output sequence
        x = torch.zeros(B, T, self.latent_dim, device=device)
        x = x + self.pos_embed[:, :T, :]
        
        # Decoder layers
        for layer in self.decoder_layers:
            # Self attention on sequence
            x_normed = layer['self_norm'](x)
            self_attn_out, _ = layer['self_attn'](x_normed, x_normed, x_normed, key_padding_mask=~mask)
            x = x + self_attn_out
            
            # Cross attention: sequence attends to part tokens
            x_normed = layer['cross_norm'](x)
            cross_attn_out, _ = layer['cross_attn'](x_normed, z, z)
            x = x + cross_attn_out
            
            # Mamba + Local
            x = layer['mamba'](x)
            x = layer['local'](x)
        
        x = self.decoder_norm(x)
        
        # Output projection
        # Option 1: Single projection
        # output = self.output_proj(x)
        
        # Option 2: Part-aware projection (slightly better)
        # 각 part token의 정보를 해당 feature 영역에 반영
        body_out = self.part_output_proj[0](x + z[:, 0:1, :].expand(-1, T, -1))
        lhand_out = self.part_output_proj[1](x + z[:, 1:2, :].expand(-1, T, -1))
        rhand_out = self.part_output_proj[2](x + z[:, 2:3, :].expand(-1, T, -1))
        
        output = torch.cat([body_out, lhand_out, rhand_out], dim=-1)
        
        # Mask padding
        output = output * mask.unsqueeze(-1).float()
        
        return output
    
    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        z, dist = self.encode(features, lengths)
        recon = self.decode(z, lengths)
        return recon, z, dist


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing LightPartMambaVae...")
    
    ablation = {'PE_TYPE': 'mld', 'MLP_DIST': False, 'SKIP_CONNECT': True}
    
    model = LightPartMambaVae(
        ablation=ablation,
        nfeats=120,
        latent_dim=[3, 256],
        num_layers=4,
    )
    
    B, T = 4, 100
    x = torch.randn(B, T, 120)
    lengths = [100, 80, 60, 40]
    
    z, dist = model.encode(x, lengths)
    print(f"z: {z.shape}")  # [4, 3, 256]
    
    recon = model.decode(z, lengths)
    print(f"recon: {recon.shape}")  # [4, 100, 120]
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.2f}M")
    
    print("✓ Done!")
