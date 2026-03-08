"""
motGPT/archs/light_denoiser.py
===============================
SignGPT3 Light Latent Denoiser

Light-T2M의 Mamba 구조 대신 MLP + AdaLN(Adaptive Layer Norm)을 사용하는
단순하고 효율적인 latent diffusion denoiser.

입출력:
    z_noisy : [B, latent_dim]       ← VAE encoded latent (noisy)
    timestep : [B]                  ← diffusion timestep
    text_embed : [B, text_dim]      ← CLIP text embedding
    → z_pred : [B, latent_dim]      ← denoised latent prediction

아키텍처:
    Condition = MLP(concat(timestep_sinusoidal, text_proj))  → [B, hidden_dim]
    Input     = Linear(z_noisy)                              → [B, hidden_dim]
    N × AdaLN ResBlock(input, condition)
    Output    = Linear(x)                                    → [B, latent_dim]

AdaLN ResBlock:
    scale, shift = chunk(Linear(condition), 2)
    x = x + Linear(SiLU(Linear(LayerNorm(x) * (1+scale) + shift)))

Reference:
    - DiT (Peebles & Xie, 2023): Adaptive Layer Norm conditioning
    - Light-T2M (Zeng et al., AAAI 2025): timestep_embedding 방식 차용
    - MotionGPT3 diffusion/diffloss.py: TimestepEmbedder 참고

Usage:
    denoiser = SignLatentDenoiser(
        latent_dim=256,
        text_dim=512,
        hidden_dim=512,
        num_blocks=8,
        dropout=0.1,
    )
    z_pred = denoiser(z_noisy, timestep, text_embed)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Timestep sinusoidal embedding (Light-T2M / DDPM 방식)
# =============================================================================

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (DDPM 방식).
    
    Args:
        timesteps: [B] LongTensor
        dim: embedding dimension
        max_period: controls minimum frequency
    Returns:
        [B, dim] float embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# =============================================================================
# Timestep MLP (sinusoidal → hidden_dim)
# =============================================================================

class TimestepMLP(nn.Module):
    """
    Sinusoidal timestep → hidden_dim via 2-layer MLP.
    Light-T2M의 TimestepEmbedder와 동일한 역할.
    """
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] LongTensor
        Returns:
            [B, hidden_dim]
        """
        emb = timestep_embedding(timesteps, self.freq_dim)
        return self.mlp(emb)


# =============================================================================
# AdaLN ResBlock (DiT-style)
# =============================================================================

class AdaLNResBlock(nn.Module):
    """
    Residual MLP block with Adaptive Layer Norm conditioning.

    AdaLN: scale and shift of LayerNorm are predicted from condition.
    
    Structure:
        condition → Linear → (scale, shift)  ← 조건부 스케일/시프트
        x_norm = LayerNorm(x) * (1 + scale) + shift
        x = x + dropout(Linear(SiLU(Linear(x_norm))))
    
    Args:
        hidden_dim: feature dimension
        cond_dim:   condition dimension (timestep + text)
        dropout:    dropout rate (0 = disabled)
        expand:     inner expansion factor for FFN (default 4)
    """
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
        expand: int = 4,
    ):
        super().__init__()
        inner_dim = hidden_dim * expand

        # LayerNorm (no affine: AdaLN이 대신함)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # AdaLN modulation: condition → (scale, shift)
        # zero-init output: 학습 초기에 residual이 identity가 되도록
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim, bias=True),
        )
        # zero-init for stable training (DiT 방식)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, hidden_dim),
        )
        # zero-init final linear for stable training
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, hidden_dim]
            cond: [B, cond_dim]
        Returns:
            [B, hidden_dim]
        """
        # AdaLN modulation
        scale, shift = self.adaLN_modulation(cond).chunk(2, dim=-1)  # [B, hidden_dim] each

        # Normalize + modulate
        x_norm = self.norm(x) * (1.0 + scale) + shift  # [B, hidden_dim]

        # FFN + residual
        return x + self.ffn(x_norm)


# =============================================================================
# SignLatentDenoiser (메인 모델)
# =============================================================================

class SignLatentDenoiser(nn.Module):
    """
    MLP-based latent denoiser for Sign Language motion generation.
    
    Light-T2M의 Mamba sequence denoiser 대신 사용하는 단순하고 빠른 구조.
    VAE가 이미 motion을 single vector로 압축했으므로 sequence modeling 불필요.
    
    Flow:
        z_noisy [B, latent_dim]
            ↓ input_proj
        x [B, hidden_dim]
        
        timestep [B] → TimestepMLP → t_emb [B, hidden_dim]
        text_embed [B, text_dim] → text_proj → tx_emb [B, hidden_dim]
        cond = t_emb + tx_emb  [B, hidden_dim]   ← 합산 (concat도 가능)
        
        N × AdaLNResBlock(x, cond)
            ↓ output_proj
        z_pred [B, latent_dim]
    
    Args:
        latent_dim: VAE latent dimension (default 256, MldVae의 [1, 256])
        text_dim:   text encoder output dim (CLIP: 512, mCLIP: 512)
        hidden_dim: internal feature dim
        num_blocks: number of AdaLN ResBlocks
        dropout:    dropout rate in FFN
        ffn_expand: expansion factor in FFN (hidden_dim × ffn_expand)
        freq_dim:   sinusoidal timestep freq dimension
    """

    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_blocks: int = 8,
        dropout: float = 0.1,
        ffn_expand: int = 4,
        freq_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Condition: timestep ───────────────────────────────────────────
        self.timestep_mlp = TimestepMLP(hidden_dim, freq_dim=freq_dim)

        # ── Condition: text ───────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── AdaLN ResBlocks ───────────────────────────────────────────────
        # condition dim = hidden_dim (timestep + text는 합산)
        self.blocks = nn.ModuleList([
            AdaLNResBlock(
                hidden_dim=hidden_dim,
                cond_dim=hidden_dim,
                dropout=dropout,
                expand=ffn_expand,
            )
            for _ in range(num_blocks)
        ])

        # ── Output projection ─────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # zero-init: 학습 초기에 identity prediction에 가깝게
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._print_params()

    def _print_params(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SignLatentDenoiser] "
              f"latent={self.latent_dim}, hidden={self.hidden_dim}, "
              f"blocks={len(self.blocks)}, "
              f"params={n_params/1e6:.2f}M")

    def forward(
        self,
        z_noisy: torch.Tensor,
        timestep: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_noisy:    [B, latent_dim]  noisy VAE latent
            timestep:   [B]              LongTensor diffusion step
            text_embed: [B, text_dim]    CLIP / mCLIP embedding
        Returns:
            z_pred: [B, latent_dim]      predicted clean latent (or noise)
        """
        # ── Input projection
        x = self.input_proj(z_noisy)          # [B, hidden_dim]

        # ── Condition: timestep + text
        t_emb = self.timestep_mlp(timestep)   # [B, hidden_dim]
        tx_emb = self.text_proj(text_embed)   # [B, hidden_dim]
        cond = t_emb + tx_emb                 # [B, hidden_dim]  (합산)

        # ── AdaLN ResBlocks
        for block in self.blocks:
            x = block(x, cond)                # [B, hidden_dim]

        # ── Output
        x = self.output_norm(x)               # [B, hidden_dim]
        z_pred = self.output_proj(x)          # [B, latent_dim]

        return z_pred

    # -------------------------------------------------------------------------
    # Classifier-Free Guidance (CFG) inference
    # -------------------------------------------------------------------------
    def forward_with_cfg(
        self,
        z_noisy: torch.Tensor,
        timestep: torch.Tensor,
        text_embed: torch.Tensor,
        guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        """
        CFG inference: concat(cond_batch, uncond_batch) → split → weighted sum
        
        사용 방법:
            doubled_z = torch.cat([z_noisy, z_noisy], dim=0)
            doubled_t = torch.cat([timestep, timestep], dim=0)
            null_text = torch.zeros_like(text_embed)
            doubled_text = torch.cat([text_embed, null_text], dim=0)
            
            pred = denoiser.forward_with_cfg(
                doubled_z, doubled_t, doubled_text, guidance_scale=4.0
            )
        
        Args:
            z_noisy:       [2B, latent_dim]  (cond + uncond)
            timestep:      [2B]
            text_embed:    [2B, text_dim]    (cond + zeros)
            guidance_scale: CFG weight
        Returns:
            [B, latent_dim]  guided prediction
        """
        pred = self.forward(z_noisy, timestep, text_embed)  # [2B, latent_dim]
        pred_cond, pred_uncond = pred.chunk(2, dim=0)       # [B, latent_dim] each
        guided = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        return guided


# =============================================================================
# Quick unit test
# =============================================================================

if __name__ == "__main__":
    import torch

    B = 4
    LATENT_DIM = 256
    TEXT_DIM = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- 기본 forward test ---
    denoiser = SignLatentDenoiser(
        latent_dim=LATENT_DIM,
        text_dim=TEXT_DIM,
        hidden_dim=512,
        num_blocks=8,
        dropout=0.1,
    ).to(DEVICE)

    z_noisy = torch.randn(B, LATENT_DIM, device=DEVICE)
    timestep = torch.randint(0, 1000, (B,), device=DEVICE)
    text_embed = torch.randn(B, TEXT_DIM, device=DEVICE)

    z_pred = denoiser(z_noisy, timestep, text_embed)
    print(f"z_noisy shape:  {z_noisy.shape}")
    print(f"timestep shape: {timestep.shape}")
    print(f"text_embed shape: {text_embed.shape}")
    print(f"z_pred shape:   {z_pred.shape}")
    assert z_pred.shape == (B, LATENT_DIM), f"Shape mismatch: {z_pred.shape}"
    print("✅ Basic forward test PASSED")

    # --- CFG forward test ---
    z_doubled = z_noisy.repeat(2, 1)
    t_doubled = timestep.repeat(2)
    tx_doubled = torch.cat([text_embed, torch.zeros_like(text_embed)], dim=0)
    z_guided = denoiser.forward_with_cfg(z_doubled, t_doubled, tx_doubled, guidance_scale=4.0)
    assert z_guided.shape == (B, LATENT_DIM), f"CFG shape mismatch: {z_guided.shape}"
    print("✅ CFG forward test PASSED")

    # --- Gradient flow test ---
    loss = z_pred.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in denoiser.parameters() if p.grad is not None]
    print(f"✅ Gradient flow test PASSED ({len(grads)} param groups with grad)")

    # --- Parameter count by component ---
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print("\n--- Parameter breakdown ---")
    print(f"  input_proj:    {count_params(denoiser.input_proj):,}")
    print(f"  timestep_mlp:  {count_params(denoiser.timestep_mlp):,}")
    print(f"  text_proj:     {count_params(denoiser.text_proj):,}")
    print(f"  blocks (x{len(denoiser.blocks)}): {count_params(denoiser.blocks):,}")
    print(f"  output:        {count_params(denoiser.output_norm) + count_params(denoiser.output_proj):,}")
    print(f"  TOTAL:         {count_params(denoiser):,}")

    # --- Small vs Large config 비교 ---
    print("\n--- Config comparison ---")
    for h, nb in [(256, 4), (512, 8), (512, 12), (1024, 8)]:
        m = SignLatentDenoiser(hidden_dim=h, num_blocks=nb)
        n = sum(p.numel() for p in m.parameters())
        print(f"  hidden={h}, blocks={nb}: {n/1e6:.2f}M params")
