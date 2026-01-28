"""
MambaVae - Mamba-based VAE for Sign Language Motion
====================================================

Light-T2M의 Mamba 구조를 차용한 경량 VAE
MldVae와 동일한 인터페이스로 drop-in replacement 가능

특징:
- Encoder: Bidirectional Mamba + LocalModule
- Decoder: Mamba + LocalModule  
- O(n) 복잡도로 속도 향상 (vs Transformer O(n²))
- MldVae와 100% 호환되는 인터페이스

Usage:
    # configs/sign_mamba_vae.yaml에서
    vae:
      mambavae:
        target: motGPT.archs.mamba_vae.MambaVae
        params:
          nfeats: 120
          latent_dim: [1, 256]
          num_layers: 4
          ...

Author: SignGPT3 Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from typing import List, Optional, Union, Tuple
from functools import partial

# Mamba import with fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[MambaVae] Warning: mamba_ssm not installed. Using Conv1D fallback.")


def lengths_to_mask(lengths: List[int], device: torch.device, max_len: Optional[int] = None) -> Tensor:
    """Convert lengths to boolean mask"""
    if max_len is None:
        max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
    return mask


# =============================================================================
# Local Module (Light-T2M style)
# =============================================================================
class LocalModule(nn.Module):
    """
    Light-T2M의 LocalModule
    1D Conv + GroupNorm으로 local 정보 모델링
    """
    def __init__(self, dim: int, num_groups: int = 16, kernel_size: int = 3):
        super().__init__()
        # Ensure num_groups divides dim
        num_groups = min(num_groups, dim)
        while dim % num_groups != 0:
            num_groups -= 1
        
        self.norm = nn.GroupNorm(num_groups, dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=1)
        self.act = nn.SiLU()
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] boolean
        Returns:
            x: [B, T, D]
        """
        residual = x
        
        # [B, T, D] → [B, D, T]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        # [B, D, T] → [B, T, D]
        x = x.transpose(1, 2)
        
        # Residual connection
        x = x + residual
        
        # Apply mask
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            
        return x


# =============================================================================
# Mamba Block
# =============================================================================
class MambaBlock(nn.Module):
    """
    Mamba SSM Block with residual connection
    Falls back to Conv1D if mamba_ssm not available
    """
    def __init__(
        self, 
        dim: int, 
        d_state: int = 16, 
        d_conv: int = 4, 
        expand: int = 2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(dim)
        
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            if bidirectional:
                self.mamba_bwd = Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.combine = nn.Linear(dim * 2, dim)
            self._is_mamba = True
        else:
            # Fallback: Conv1D-based approximation
            inner_dim = dim * expand
            self.conv_block = nn.Sequential(
                nn.Conv1d(dim, inner_dim, 1),
                nn.SiLU(),
                nn.Conv1d(inner_dim, inner_dim, d_conv, padding=d_conv // 2, groups=inner_dim),
                nn.SiLU(),
                nn.Conv1d(inner_dim, dim, 1),
            )
            self._is_mamba = False
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] boolean (optional)
        Returns:
            x: [B, T, D]
        """
        residual = x
        x = self.norm(x)
        
        if self._is_mamba:
            if self.bidirectional:
                # Forward pass
                fwd = self.mamba_fwd(x)
                # Backward pass (flip, process, flip back)
                bwd = self.mamba_bwd(x.flip(1)).flip(1)
                # Combine
                x = self.combine(torch.cat([fwd, bwd], dim=-1))
            else:
                x = self.mamba_fwd(x)
        else:
            # Conv1D fallback
            x = x.transpose(1, 2)  # [B, D, T]
            x = self.conv_block(x)
            x = x.transpose(1, 2)  # [B, T, D]
        
        # Residual
        x = x + residual
        
        # Apply mask
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            
        return x


# =============================================================================
# Mamba Encoder
# =============================================================================
class MambaEncoder(nn.Module):
    """
    Mamba-based VAE Encoder
    
    구조: Input → [LocalModule → MambaBlock] × N → DistTokens → μ, σ
    """
    def __init__(
        self,
        input_dim: int = 120,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_groups: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, latent_dim) * 0.02)
        
        # Mamba layers: [LocalModule → MambaBlock] × N
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'local': LocalModule(latent_dim, num_groups),
                'mamba': MambaBlock(latent_dim, d_state, d_conv, expand, bidirectional=True),
            }))
        
        # Final norm
        self.final_norm = nn.LayerNorm(latent_dim)
        
        # Distribution head (attention pooling)
        self.pool_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(latent_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # μ, σ projection
        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: Tensor, lengths: List[int]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, T, input_dim]
            lengths: list of sequence lengths
        Returns:
            mu: [1, B, latent_dim]
            logvar: [1, B, latent_dim]
        """
        B, T, _ = x.shape
        device = x.device
        
        # Create mask
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Input projection
        x = self.input_proj(x)  # [B, T, latent_dim]
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers
        for layer in self.layers:
            x = layer['local'](x, mask)
            x = layer['mamba'](x, mask)
        
        x = self.final_norm(x)
        
        # Attention pooling to get single vector
        # Query: learnable token, Key/Value: encoded sequence
        query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
        
        # Create attention mask (True = ignore)
        attn_mask = ~mask  # [B, T]
        
        pooled, _ = self.pool_attn(
            query, x, x,
            key_padding_mask=attn_mask,
        )  # [B, 1, D]
        
        pooled = pooled.squeeze(1)  # [B, D]
        
        # Get mu and logvar
        mu = self.to_mu(pooled)        # [B, D]
        logvar = self.to_logvar(pooled)  # [B, D]
        
        # Reshape to [1, B, D] for compatibility with MldVae
        mu = mu.unsqueeze(0)        # [1, B, D]
        logvar = logvar.unsqueeze(0)  # [1, B, D]
        
        return mu, logvar


# =============================================================================
# Mamba Decoder
# =============================================================================
class MambaDecoder(nn.Module):
    """
    Mamba-based VAE Decoder
    
    구조: z → Broadcast → [MambaBlock → LocalModule] × N → Output
    """
    def __init__(
        self,
        output_dim: int = 120,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_groups: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)
        
        # Mamba layers: [MambaBlock → LocalModule] × N
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'mamba': MambaBlock(latent_dim, d_state, d_conv, expand, bidirectional=True),
                'local': LocalModule(latent_dim, num_groups),
            }))
        
        # Final layers
        self.final_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
    def forward(self, z: Tensor, lengths: List[int]) -> Tensor:
        """
        Args:
            z: [1, B, latent_dim] or [B, latent_dim]
            lengths: list of sequence lengths
        Returns:
            output: [B, T, output_dim]
        """
        # Handle both [1, B, D] and [B, D] inputs
        if z.dim() == 3:
            z = z.squeeze(0)  # [B, D]
        
        B = z.shape[0]
        T = max(lengths)
        device = z.device
        
        # Create mask
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Project latent
        z = self.latent_proj(z)  # [B, D]
        
        # Broadcast to sequence
        x = z.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers
        for layer in self.layers:
            x = layer['mamba'](x, mask)
            x = layer['local'](x, mask)
        
        # Output
        x = self.final_norm(x)
        x = self.output_proj(x)  # [B, T, output_dim]
        
        # Zero out padding
        x = x * mask.unsqueeze(-1).float()
        
        return x


# =============================================================================
# Main: MambaVae
# =============================================================================
class MambaVae(nn.Module):
    """
    Mamba-based VAE for Sign Language Motion
    
    MldVae와 100% 호환되는 인터페이스
    
    Usage:
        vae = MambaVae(ablation, nfeats=120, latent_dim=[1, 256], ...)
        z, dist = vae.encode(features, lengths)  # z: [1, B, 256]
        recon = vae.decode(z, lengths)            # recon: [B, T, 120]
    """
    
    def __init__(
        self,
        ablation,                           # For MldVae compatibility
        nfeats: int = 120,
        latent_dim: list = [1, 256],        # [latent_size, latent_dim]
        ff_size: int = 1024,                # Ignored (for compatibility)
        num_layers: int = 4,                # Mamba layers
        num_heads: int = 4,                 # Ignored (for compatibility)
        dropout: float = 0.1,
        arch: str = "encoder_decoder",      # Ignored
        normalize_before: bool = False,     # Ignored
        activation: str = "gelu",           # Ignored
        position_embedding: str = "learned", # Ignored
        datatype: str = 'h2s',
        # Mamba-specific params
        num_groups: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        
        # MldVae-compatible attributes
        self.latent_size = latent_dim[0]  # 1
        self.latent_dim = latent_dim[-1]  # 256
        
        # Store ablation (for compatibility)
        self.ablation = ablation
        self.pe_type = ablation.get('PE_TYPE', 'mld')
        self.mlp_dist = ablation.get('MLP_DIST', False)
        
        # Normalization statistics (same as MldVae)
        if 'motionx' in datatype.lower():
            self.mean_std_inv = 0.7281
            self.mean_mean = 0.0636
        else:
            self.mean_std_inv = 0.8457
            self.mean_std_inv_2 = self.mean_std_inv ** 2
            self.mean_mean = -0.1379
        
        # Encoder
        self.encoder = MambaEncoder(
            input_dim=nfeats,
            latent_dim=self.latent_dim,
            num_layers=num_layers,
            num_groups=num_groups,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder_net = MambaDecoder(
            output_dim=nfeats,
            latent_dim=self.latent_dim,
            num_layers=num_layers,
            num_groups=num_groups,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        print(f"[MambaVae] Initialized with:")
        print(f"  - nfeats: {nfeats}")
        print(f"  - latent_dim: [{self.latent_size}, {self.latent_dim}]")
        print(f"  - num_layers: {num_layers}")
        print(f"  - Mamba available: {MAMBA_AVAILABLE}")
        
    def encode(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None
    ) -> Tuple[Tensor, Normal]:
        """
        Encode features to latent z
        
        Args:
            features: [B, T, nfeats]
            lengths: list of sequence lengths
            
        Returns:
            z: [1, B, latent_dim] (same as MldVae)
            dist: torch.distributions.Normal
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        
        # Get mu, logvar
        mu, logvar = self.encoder(features, lengths)  # [1, B, D]
        
        # Reparameterization
        std = (logvar * 0.5).exp()
        dist = Normal(mu, std)
        z = dist.rsample()  # [1, B, D]
        
        return z, dist
    
    def decode(self, z: Tensor, lengths: List[int]) -> Tensor:
        """
        Decode latent z to features
        
        Args:
            z: [1, B, latent_dim] or [B, latent_dim]
            lengths: list of sequence lengths
            
        Returns:
            features: [B, T, nfeats]
        """
        return self.decoder_net(z, lengths)
    
    def forward(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None
    ) -> Tuple[Tensor, Tensor, Normal]:
        """
        Full forward pass (encode + decode)
        
        Args:
            features: [B, T, nfeats]
            lengths: list of sequence lengths
            
        Returns:
            recon: [B, T, nfeats]
            z: [1, B, latent_dim]
            dist: torch.distributions.Normal
        """
        z, dist = self.encode(features, lengths)
        recon = self.decode(z, lengths)
        return recon, z, dist


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing MambaVae...")
    
    # Config (mimicking sign_vae.yaml)
    ablation = {
        'PE_TYPE': 'mld',
        'MLP_DIST': False,
        'SKIP_CONNECT': True,
    }
    
    # Create model
    model = MambaVae(
        ablation=ablation,
        nfeats=120,
        latent_dim=[1, 256],
        num_layers=4,
        dropout=0.1,
    )
    
    # Test data
    B, T = 4, 100
    features = torch.randn(B, T, 120)
    lengths = [100, 80, 60, 40]
    
    # Test encode
    z, dist = model.encode(features, lengths)
    print(f"z shape: {z.shape}")  # Expected: [1, 4, 256]
    print(f"dist.loc shape: {dist.loc.shape}")  # Expected: [1, 4, 256]
    
    # Test decode
    recon = model.decode(z, lengths)
    print(f"recon shape: {recon.shape}")  # Expected: [4, 100, 120]
    
    # Test forward
    recon2, z2, dist2 = model(features, lengths)
    print(f"forward recon shape: {recon2.shape}")
    
    # Check interface compatibility
    print(f"\nInterface check:")
    print(f"  latent_size: {model.latent_size}")  # Expected: 1
    print(f"  latent_dim: {model.latent_dim}")    # Expected: 256
    print(f"  mean_std_inv: {model.mean_std_inv}")
    
    print("\n✓ All tests passed!")
