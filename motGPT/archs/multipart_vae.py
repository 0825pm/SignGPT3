"""
Multi-Part VAE with Cross-Attention for Sign Language

각 신체 파트(body, lhand, rhand)를 독립적인 latent space로 인코딩하고,
Cross-Attention으로 파트 간 정보를 교환합니다.

SOKE 133-dim structure:
  [0:30]    upper_body (10 joints × 3)
  [30:75]   lhand (15 joints × 3)  
  [75:120]  rhand (15 joints × 3)
  [120:133] face (jaw + expr)

Usage:
  이 파일을 SignGPT3/motGPT/archs/multipart_vae.py 로 복사하세요.
"""

from typing import List, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

from motGPT.archs.operator import PositionalEncoding
from motGPT.archs.operator.cross_attention import (
    SkipTransformerEncoder,
    TransformerEncoderLayer,
)
from motGPT.models.utils.position_encoding import build_position_encoding
from motGPT.utils.temos_utils import lengths_to_mask


# =============================================================================
# Part Indices (SOKE 133-dim)
# =============================================================================
# SOKE 구조: body(30) + lhand(45) + rhand(45) + face(13) = 133
# face는 body와 함께 처리 (jaw+expr은 상체와 연관)
#
# 실제 분할:
#   body:  [0:30] + [120:133] = 43 dims (upper_body + face)
#   lhand: [30:75]            = 45 dims
#   rhand: [75:120]           = 45 dims

# 변경 (120-dim)
PART_SLICES = {
    'body':  [(0, 30)],     # upper_body only
    'lhand': [(30, 75)],
    'rhand': [(75, 120)],
}

PART_DIMS = {
    'body':  30,   # upper_body only
    'lhand': 45,
    'rhand': 45,
}

# Total: 43 + 45 + 45 = 133 ✓


# =============================================================================
# Cross-Part Attention Module
# =============================================================================
class CrossPartAttention(nn.Module):
    """
    파트 간 Cross-Attention
    각 파트가 다른 파트의 정보를 참조할 수 있게 함
    """
    def __init__(
        self,
        latent_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Cross-attention: query=self, key/value=others
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # (seq, batch, dim)
        )
        
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None):
        """
        Args:
            x: query tensor (T, B, D) - 현재 파트
            context: key/value tensor (T*num_other_parts, B, D) - 다른 파트들
            mask: attention mask
        Returns:
            output: (T, B, D)
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=mask,
        )
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


# =============================================================================
# Part Encoder
# =============================================================================
class PartEncoder(nn.Module):
    """
    단일 파트를 위한 Encoder
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        part_latent_dim: int = 64,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.part_latent_dim = part_latent_dim
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, latent_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            latent_dim, num_heads, ff_size, dropout, activation, normalize_before=False
        )
        encoder_norm = nn.LayerNorm(latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        # Distribution token (learnable)
        self.dist_token = nn.Parameter(torch.randn(2, latent_dim))  # 2 for mu, logvar
        
        # Output projection to part latent
        self.to_dist = nn.Linear(latent_dim, part_latent_dim * 2)  # mu + logvar
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, input_dim)
            mask: (B, T) boolean mask
        Returns:
            dist_tokens: (2, B, latent_dim) - for cross-attention
        """
        B, T, _ = x.shape
        
        # Embed
        x = self.input_embed(x)  # (B, T, latent_dim)
        x = x.permute(1, 0, 2)   # (T, B, latent_dim)
        
        # Add distribution tokens
        dist = self.dist_token.unsqueeze(1).expand(-1, B, -1)  # (2, B, latent_dim)
        x_seq = torch.cat([dist, x], dim=0)  # (2+T, B, latent_dim)
        
        # Create augmented mask
        dist_mask = torch.ones(B, 2, dtype=torch.bool, device=x.device)
        aug_mask = torch.cat([dist_mask, mask], dim=1)  # (B, 2+T)
        
        # Positional encoding
        x_seq = self.pos_encoder(x_seq)
        
        # Encode
        encoded = self.encoder(x_seq, src_key_padding_mask=~aug_mask)
        
        # Extract distribution tokens
        dist_tokens = encoded[:2]  # (2, B, latent_dim)
        
        return dist_tokens
    
    def get_distribution(self, dist_tokens: Tensor):
        """
        dist_tokens에서 mu, logvar 추출
        Args:
            dist_tokens: (2, B, latent_dim)
        Returns:
            mu: (1, B, part_latent_dim)
            logvar: (1, B, part_latent_dim)
        """
        # Pool the two tokens
        pooled = dist_tokens.mean(dim=0)  # (B, latent_dim)
        
        # Project to mu and logvar
        dist_params = self.to_dist(pooled)  # (B, part_latent_dim * 2)
        mu, logvar = dist_params.chunk(2, dim=-1)  # each (B, part_latent_dim)
        
        return mu.unsqueeze(0), logvar.unsqueeze(0)  # (1, B, part_latent_dim)


# =============================================================================
# Part Decoder
# =============================================================================
class PartDecoder(nn.Module):
    """
    단일 파트를 위한 Decoder
    """
    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 256,
        part_latent_dim: int = 64,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.part_latent_dim = part_latent_dim
        
        # Latent projection
        self.latent_proj = nn.Linear(part_latent_dim, latent_dim)
        
        # Positional encoding
        self.pos_decoder = PositionalEncoding(latent_dim, dropout)
        
        # Transformer decoder layers (using encoder architecture for simplicity)
        decoder_layer = TransformerEncoderLayer(
            latent_dim, num_heads, ff_size, dropout, activation, normalize_before=False
        )
        decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = SkipTransformerEncoder(decoder_layer, num_layers, decoder_norm)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
    
    def forward(self, z: Tensor, lengths: List[int], mask: Tensor) -> Tensor:
        """
        Args:
            z: (1, B, part_latent_dim)
            lengths: list of sequence lengths
            mask: (B, T) boolean mask
        Returns:
            output: (B, T, output_dim)
        """
        B = z.shape[1]
        T = mask.shape[1]
        device = z.device
        
        # Project latent
        z_proj = self.latent_proj(z)  # (1, B, latent_dim)
        
        # Create query tokens
        queries = torch.zeros(T, B, self.latent_dim, device=device)
        
        # Concatenate latent with queries
        x_seq = torch.cat([z_proj, queries], dim=0)  # (1+T, B, latent_dim)
        
        # Create augmented mask  
        z_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        aug_mask = torch.cat([z_mask, mask], dim=1)  # (B, 1+T)
        
        # Positional encoding
        x_seq = self.pos_decoder(x_seq)
        
        # Decode
        decoded = self.decoder(x_seq, src_key_padding_mask=~aug_mask)
        
        # Extract output (skip the latent token)
        output = decoded[1:]  # (T, B, latent_dim)
        output = output.permute(1, 0, 2)  # (B, T, latent_dim)
        
        # Project to output dimension
        output = self.output_proj(output)  # (B, T, output_dim)
        
        # Zero out padded positions
        output = output * mask.unsqueeze(-1)
        
        return output


# =============================================================================
# Multi-Part VAE
# =============================================================================
class MultiPartVae(nn.Module):
    """
    Multi-Part VAE with Cross-Attention for Sign Language
    
    3개 파트로 분리, 각각 독립적인 256-dim latent:
      - body:  upper_body + face (43 dims) → z_body (256 dim)
      - lhand: 왼손 (45 dims) → z_lhand (256 dim)
      - rhand: 오른손 (45 dims) → z_rhand (256 dim)
    
    각 파트는 독립적으로 mu, sigma를 학습하고 샘플링됩니다.
    Cross-Attention으로 파트 간 정보를 교환합니다.
    """
    
    def __init__(
        self,
        nfeats: int = 133,
        latent_dim: int = 256,        # 각 파트의 latent dimension (동일)
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        ablation: dict = None,
        # Cross-attention settings
        num_cross_layers: int = 2,
        # Loss weights
        body_weight: float = 1.0,
        lhand_weight: float = 2.0,
        rhand_weight: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        
        self.nfeats = nfeats
        self.latent_dim = latent_dim  # 각 파트의 latent dim (256)
        
        # 3개 파트 (face는 body에 포함)
        self.parts = ['body', 'lhand', 'rhand']
        
        # 모든 파트가 동일한 latent_dim (256) 사용
        self.part_latent_dims = {
            'body': latent_dim,   # 256
            'lhand': latent_dim,  # 256
            'rhand': latent_dim,  # 256
        }
        
        self.part_weights = {
            'body': body_weight,
            'lhand': lhand_weight,
            'rhand': rhand_weight,
        }
        
        # Total latent dimension
        self.total_latent_dim = sum(self.part_latent_dims[p] for p in self.parts)
        
        # Part Encoders
        self.encoders = nn.ModuleDict()
        for part in self.parts:
            self.encoders[part] = PartEncoder(
                input_dim=PART_DIMS[part],
                latent_dim=latent_dim,
                part_latent_dim=self.part_latent_dims[part],
                ff_size=ff_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
        
        # Cross-Part Attention (Encoder side)
        self.cross_attn_enc = nn.ModuleDict()
        for part in self.parts:
            self.cross_attn_enc[part] = nn.ModuleList([
                CrossPartAttention(latent_dim, num_heads, dropout)
                for _ in range(num_cross_layers)
            ])
        
        # Part Decoders
        self.decoders = nn.ModuleDict()
        for part in self.parts:
            self.decoders[part] = PartDecoder(
                output_dim=PART_DIMS[part],
                latent_dim=latent_dim,
                part_latent_dim=self.part_latent_dims[part],
                ff_size=ff_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
        
        # Cross-Part Attention (Decoder side) - operates on decoded features
        self.cross_attn_dec = nn.ModuleDict()
        for part in self.parts:
            self.cross_attn_dec[part] = nn.ModuleList([
                CrossPartAttention(latent_dim, num_heads, dropout)
                for _ in range(num_cross_layers)
            ])
        
        # Decoder feature projection (for cross-attention)
        self.dec_proj = nn.ModuleDict()
        self.dec_out_proj = nn.ModuleDict()
        for part in self.parts:
            self.dec_proj[part] = nn.Linear(PART_DIMS[part], latent_dim)
            self.dec_out_proj[part] = nn.Linear(latent_dim, PART_DIMS[part])

    def split_features(self, features: Tensor) -> Dict[str, Tensor]:
        parts = {}
        
        # body: upper_body only (face 제거)
        parts['body'] = features[:, :, 0:30]      # (B, T, 30)
        parts['lhand'] = features[:, :, 30:75]    # (B, T, 45)
        parts['rhand'] = features[:, :, 75:120]   # (B, T, 45)
        
        return parts
    
    def merge_features(self, parts: Dict[str, Tensor]) -> Tensor:
        # 120-dim: upper_body + lhand + rhand
        features = torch.cat([
            parts['body'],        # [0:30]   upper_body
            parts['lhand'],       # [30:75]
            parts['rhand'],       # [75:120]
        ], dim=-1)
        
        return features
    
    def encode_dist(self, features: Tensor, lengths: List[int]) -> Dict[str, Tensor]:
        """
        Encode features to distribution tokens (before sampling)
        """
        device = features.device
        B, T, _ = features.shape
        
        # Create mask
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Split features
        part_features = self.split_features(features)
        
        # Encode each part
        dist_tokens = {}
        for part in self.parts:
            dist_tokens[part] = self.encoders[part](part_features[part], mask)
        
        # Cross-attention between parts
        for layer_idx in range(len(self.cross_attn_enc['body'])):
            new_dist_tokens = {}
            for part in self.parts:
                # Get context from other parts
                other_parts = [p for p in self.parts if p != part]
                context = torch.cat([dist_tokens[p] for p in other_parts], dim=0)
                
                # Apply cross-attention
                new_dist_tokens[part] = self.cross_attn_enc[part][layer_idx](
                    dist_tokens[part], context
                )
            dist_tokens = new_dist_tokens
        
        return dist_tokens
    
    def encode(self, features: Tensor, lengths: List[int]):
        """
        Encode features to latent z
        
        Returns:
            z_dict: dict of latent for each part
                - 'body':  (1, B, 256)
                - 'lhand': (1, B, 256)
                - 'rhand': (1, B, 256)
            dist_dict: dict of distributions for KL loss
        """
        dist_tokens = self.encode_dist(features, lengths)
        
        z_dict = {}
        dist_dict = {}
        
        for part in self.parts:
            mu, logvar = self.encoders[part].get_distribution(dist_tokens[part])
            
            # Reparameterization (독립 샘플링)
            std = (logvar * 0.5).exp()
            eps = torch.randn_like(std)
            z_part = mu + std * eps  # (1, B, 256)
            
            z_dict[part] = z_part
            dist_dict[part] = torch.distributions.Normal(mu, std)
        
        return z_dict, dist_dict
    
    def encode_concat(self, features: Tensor, lengths: List[int]):
        """
        Encode and return concatenated z (for backward compatibility)
        
        Returns:
            z: (1, B, 768) - concatenated [z_body, z_lhand, z_rhand]
            dist: combined distribution
        """
        z_dict, dist_dict = self.encode(features, lengths)
        
        # Concatenate z
        z = torch.cat([z_dict[p] for p in self.parts], dim=-1)  # (1, B, 768)
        
        # Concatenate distributions
        mus = torch.cat([dist_dict[p].loc for p in self.parts], dim=-1)
        stds = torch.cat([dist_dict[p].scale for p in self.parts], dim=-1)
        dist = torch.distributions.Normal(mus, stds)
        
        return z, dist
    
    def decode(self, z_input, lengths: List[int]) -> Tensor:
        """
        Decode latent z to features
        
        Args:
            z_input: either dict {'body': ..., 'lhand': ..., 'rhand': ...}
                     or concatenated tensor (1, B, 768)
            lengths: list of sequence lengths
        Returns:
            features: (B, T, 133)
        """
        # Handle both dict and tensor input
        if isinstance(z_input, dict):
            z_parts = z_input
        else:
            # Split concatenated z
            z_parts = {}
            start = 0
            for part in self.parts:
                z_parts[part] = z_input[:, :, start:start+self.latent_dim]
                start += self.latent_dim
        
        device = z_parts['body'].device
        B = z_parts['body'].shape[1]
        T = max(lengths)
        
        # Create mask
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Decode each part
        decoded_parts = {}
        for part in self.parts:
            decoded_parts[part] = self.decoders[part](z_parts[part], lengths, mask)
        
        # Cross-attention between decoded parts
        for layer_idx in range(len(self.cross_attn_dec['body'])):
            # Project to latent dim for cross-attention
            projected = {}
            for part in self.parts:
                proj = self.dec_proj[part](decoded_parts[part])  # (B, T, latent_dim)
                projected[part] = proj.permute(1, 0, 2)  # (T, B, latent_dim)
            
            # Apply cross-attention
            new_projected = {}
            for part in self.parts:
                other_parts = [p for p in self.parts if p != part]
                context = torch.cat([projected[p] for p in other_parts], dim=0)
                
                new_projected[part] = self.cross_attn_dec[part][layer_idx](
                    projected[part], context
                )
            
            # Project back and update
            for part in self.parts:
                out = new_projected[part].permute(1, 0, 2)  # (B, T, latent_dim)
                decoded_parts[part] = self.dec_out_proj[part](out)
        
        # Merge parts
        features = self.merge_features(decoded_parts)
        
        # Zero out padded positions
        features = features * mask.unsqueeze(-1)
        
        return features
    
    def forward(self, features: Tensor, lengths: List[int]):
        """
        Full forward pass
        
        Returns:
            recon: (B, T, 133)
            z_dict: dict of latent {'body': (1,B,256), 'lhand': (1,B,256), 'rhand': (1,B,256)}
            dist_dict: dict of distributions
        """
        z_dict, dist_dict = self.encode(features, lengths)
        recon = self.decode(z_dict, lengths)
        return recon, z_dict, dist_dict
    
    def compute_loss(
        self, 
        features: Tensor, 
        recon: Tensor, 
        dist_dict: Dict[str, Distribution],
        lengths: List[int],
    ) -> Dict[str, Tensor]:
        """
        Compute reconstruction and KL losses with part-specific weights
        
        Args:
            features: (B, T, 133) ground truth
            recon: (B, T, 133) reconstructed
            dist_dict: dict of distributions for each part
            lengths: sequence lengths
        """
        device = features.device
        B, T, _ = features.shape
        mask = lengths_to_mask(lengths, device, max_len=T)
        
        # Split features
        feat_parts = self.split_features(features)
        recon_parts = self.split_features(recon)
        
        # Part-wise reconstruction loss
        recon_losses = {}
        total_recon = 0.0
        
        for part in self.parts:
            # MSE loss per part
            diff = (feat_parts[part] - recon_parts[part]) ** 2
            diff = diff * mask.unsqueeze(-1)  # Mask padding
            
            # Mean over valid positions
            loss = diff.sum() / (mask.sum() * PART_DIMS[part])
            recon_losses[part] = loss
            total_recon += self.part_weights[part] * loss
        
        # Part-wise KL loss (각 파트 독립)
        kl_losses = {}
        total_kl = 0.0
        
        for part in self.parts:
            dist = dist_dict[part]
            kl = -0.5 * torch.sum(
                1 + dist.scale.log() * 2 - dist.loc.pow(2) - dist.scale.pow(2)
            ) / B
            kl_losses[part] = kl
            total_kl += kl
        
        return {
            'recon_total': total_recon,
            'recon_body': recon_losses.get('body', 0),
            'recon_lhand': recon_losses.get('lhand', 0),
            'recon_rhand': recon_losses.get('rhand', 0),
            'kl_total': total_kl,
            'kl_body': kl_losses.get('body', 0),
            'kl_lhand': kl_losses.get('lhand', 0),
            'kl_rhand': kl_losses.get('rhand', 0),
        }


# =============================================================================
# Config-compatible wrapper (optional)
# =============================================================================
def build_multipart_vae(cfg) -> MultiPartVae:
    """
    Config에서 MultiPartVae 생성
    """
    return MultiPartVae(
        nfeats=cfg.DATASET.NFEATS,
        latent_dim=cfg.model.params.vae.params.get('latent_dim', [1, 256])[-1],
        ff_size=cfg.model.params.vae.params.get('ff_size', 1024),
        num_layers=cfg.model.params.vae.params.get('num_layers', 6),
        num_heads=cfg.model.params.vae.params.get('num_heads', 4),
        dropout=cfg.model.params.vae.params.get('dropout', 0.1),
        activation=cfg.model.params.vae.params.get('activation', 'gelu'),
    )