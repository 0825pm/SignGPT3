"""
MT5 Sign Language LM v2 - with Light-T2M Optimizations
=======================================================

mT5 Encoder + Light-T2M 스타일 최적화 + Cycle Consistency

★ Light-T2M에서 가져온 구조:
1. LIMM (Local Information Modeling Module): 1D Conv로 local smoothness
2. ATII (Adaptive Textual Information Injector): 적응적 텍스트 주입
3. LGL 패턴: Local-Global-Local 구조

★ 기대 효과:
- 파라미터 30-50% 감소
- 추론 속도 향상
- 손 움직임 smoothness 향상 (수어에 중요!)

복사 위치: motGPT/archs/mt5_sign_lm_v2.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Tuple

try:
    from transformers import MT5EncoderModel, MT5Tokenizer
except ImportError:
    raise ImportError("transformers 라이브러리가 필요합니다: pip install transformers")

from ..config import instantiate_from_config


# ============================================================================
# Light-T2M 스타일 모듈
# ============================================================================

class LIMM(nn.Module):
    """
    Local Information Modeling Module (Light-T2M)
    
    1D Convolution으로 인접 프레임 간 local 정보 모델링
    Transformer의 global attention 대비 파라미터 90% 절약
    
    수어에 특히 유용: 손 움직임의 smooth transition 보장
    """
    
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        
        # Depthwise separable conv for efficiency
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)  # Pointwise
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            x: [B, T, D]
        """
        # Conv block
        residual = x
        x = self.ln1(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.act(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T, D]
        x = x + residual
        
        # FFN block
        x = x + self.ffn(self.ln2(x))
        
        return x


class ATII(nn.Module):
    """
    Adaptive Textual Information Injector (Light-T2M)
    
    텍스트 정보를 모션에 적응적으로 주입
    Gating mechanism으로 주입량 조절
    
    수어에 특히 유용: 문장의 각 부분이 다른 동작에 대응
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Motion query projection
        self.motion_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Text key-value projection
        self.text_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Gating mechanism: 얼마나 text 정보를 주입할지 결정
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, motion: Tensor, text_pool: Tensor) -> Tensor:
        """
        Args:
            motion: [B, D] or [B, T, D] - Motion features
            text_pool: [B, D] - Pooled text embedding
        Returns:
            motion: [B, D] or [B, T, D] - Text-injected motion
        """
        is_3d = motion.dim() == 3
        if is_3d:
            B, T, D = motion.shape
            motion = motion.reshape(B * T, D)
            text_pool = text_pool.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
        
        # Project
        motion_q = self.motion_proj(motion)
        text_kv = self.text_proj(text_pool)
        text_k, text_v = text_kv.chunk(2, dim=-1)
        
        # Scaled dot-product attention (simplified)
        scale = motion_q.size(-1) ** 0.5
        attn_weight = torch.sigmoid(torch.sum(motion_q * text_k, dim=-1, keepdim=True) / scale)
        text_info = attn_weight * text_v
        
        # Gating: 필요한 만큼만 text 정보 주입
        gate = self.gate(torch.cat([motion, text_info], dim=-1))
        gated_text = gate * text_info
        
        # Fusion
        fused = self.fusion(torch.cat([motion, gated_text], dim=-1))
        output = self.ln(fused + motion)
        
        if is_3d:
            output = output.reshape(B, T, D)
        
        return output


class SharedAttentionBlock(nn.Module):
    """MotionGPT3 스타일 Shared Attention Block (Global)"""
    
    def __init__(
        self, 
        hidden_dim: int = 768, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        # Motion ← Text cross attention
        self.motion_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.motion_cross_ln = nn.LayerNorm(hidden_dim)
        
        # Text ← Motion (bidirectional)
        if bidirectional:
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.text_cross_ln = nn.LayerNorm(hidden_dim)
        
        # Motion self attention
        self.motion_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.motion_self_ln = nn.LayerNorm(hidden_dim)
        
        # Motion FFN
        self.motion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.motion_ffn_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, text_hidden, motion_hidden, text_mask=None):
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        
        # Motion ← Text
        motion_norm = self.motion_cross_ln(motion_hidden)
        motion_cross_out, _ = self.motion_cross_attn(
            query=motion_norm, key=text_hidden, value=text_hidden, 
            key_padding_mask=key_padding_mask
        )
        motion_hidden = motion_hidden + motion_cross_out
        
        # Text ← Motion (bidirectional)
        if self.bidirectional:
            text_norm = self.text_cross_ln(text_hidden)
            text_cross_out, _ = self.text_cross_attn(
                query=text_norm, key=motion_hidden, value=motion_hidden
            )
            text_hidden = text_hidden + text_cross_out
        
        # Motion self attention
        motion_norm = self.motion_self_ln(motion_hidden)
        motion_self_out, _ = self.motion_self_attn(
            query=motion_norm, key=motion_norm, value=motion_norm
        )
        motion_hidden = motion_hidden + motion_self_out
        
        # Motion FFN
        motion_hidden = motion_hidden + self.motion_ffn(self.motion_ffn_ln(motion_hidden))
        
        return text_hidden, motion_hidden


class MotionToTextDecoder(nn.Module):
    """Motion → Text Decoder for Cycle Consistency"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 250112,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_length, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, motion_hidden, target_ids=None, target_mask=None):
        B = motion_hidden.shape[0]
        device = motion_hidden.device
        
        if target_ids is None:
            raise ValueError("target_ids required for training")
        
        seq_len = target_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        decoder_input = self.token_embed(target_ids) + self.pos_embed(pos_ids)
        
        # Causal mask - use float('-inf') for compatibility with all PyTorch versions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        
        # Decode (without tgt_is_causal for PyTorch < 2.0 compatibility)
        decoder_out = self.decoder(
            tgt=decoder_input,
            memory=motion_hidden,
            tgt_mask=causal_mask,
        )
        
        decoder_out = self.ln(decoder_out)
        logits = self.output_proj(decoder_out)
        
        return logits


# ============================================================================
# Main Model: MT5SignLM v2 (with Light-T2M optimizations)
# ============================================================================

class MT5SignLMv2(nn.Module):
    """
    mT5 기반 다국어 수어 생성 모델 v2
    
    ★ Light-T2M 최적화 포함:
    ┌─────────────────────────────────────────────────────────────┐
    │                     MT5SignLM v2                            │
    ├─────────────────────────────────────────────────────────────┤
    │  [T2M Path]                                                 │
    │  Text → mT5 Encoder → text_hidden                          │
    │                           ↓                                 │
    │  ★ ATII: Adaptive text injection to motion                 │
    │                           ↓                                 │
    │  ★ LGL Pattern:                                            │
    │     LIMM (Local) → Shared Attn (Global) → LIMM (Local)     │
    │                           ↓                                 │
    │                    Diffusion Head                           │
    │                           ↓                                 │
    │                       motion_z                              │
    ├─────────────────────────────────────────────────────────────┤
    │  [M2T Path - Cycle Consistency]                             │
    │  target_z → M2T Decoder → text_logits                      │
    │                    ↓                                        │
    │          L_cycle = CE(text_logits, original_text)           │
    └─────────────────────────────────────────────────────────────┘
    
    ★ 기대 효과:
    - 파라미터 30-50% 감소 (LIMM으로 Self-Attn 대체)
    - 손 움직임 smoothness 향상 (Local conv)
    - Text-Motion alignment 개선 (ATII)
    """
    
    def __init__(
        self,
        # mT5 config
        model_name: str = 'google/mt5-base',
        freeze_encoder: bool = False,
        
        # Motion branch config
        motion_branch_layers: int = 6,
        motion_branch_heads: int = 8,
        hidden_dim: int = 768,
        num_shared_layers: int = 2,  # ★ 줄임 (LIMM이 대체)
        bidirectional_attention: bool = True,
        
        # ★ Light-T2M config
        use_limm: bool = True,
        limm_kernel_size: int = 3,
        use_atii: bool = True,
        
        # Diffusion config
        diffhead: Optional[Dict] = None,
        vae_latent_channels: int = 256,
        
        # CFG config
        guidance_uncondp: float = 0.1,
        guidance_scale: float = 1.0,
        motion_holder_repeat: int = 4,
        with_vae_latent_norm: bool = True,
        max_length: int = 128,
        
        # Cycle consistency config
        use_cycle_consistency: bool = True,
        cycle_weight: float = 0.1,
        m2t_decoder_layers: int = 4,
        
        multi_hidden: bool = True,
        
        # Compatibility params
        model_type: str = 'mt5_sign_v2',
        stage: str = 'lm_pretrain',
        motion_codebook_size: int = 512,
        mot_factor: float = 1.0,
        attention_mode: str = 'all',
        ablation: dict = None,
        diffusion_batch_mul: int = 4,
        predict_epsilon: bool = True,
        fake_latent_mode: str = 'learnable_zero',
        holder_num_in_input: int = 4,
        motion_holder_seq_mode: str = 'withse',
        with_hid_norm: bool = False,
        shared_layer_start: int = 1,  # LGL에서는 사용 안함
        model_path: str = None,
        **kwargs,
    ):
        super().__init__()
        
        # Save config
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.motion_holder_repeat = motion_holder_repeat
        self.guidance_uncondp = guidance_uncondp
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_uncondp > 0
        self.with_vae_latent_norm = with_vae_latent_norm
        self.num_shared_layers = num_shared_layers
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        self.stage = stage
        self.multi_hidden = multi_hidden
        self.use_cycle_consistency = use_cycle_consistency
        self.cycle_weight = cycle_weight
        self.vae_latent_channels = vae_latent_channels
        self.use_limm = use_limm
        self.use_atii = use_atii
        self._step_count = 0
        
        # =========================================
        # 1. mT5 Encoder
        # =========================================
        self._load_mt5(model_name)
        
        if freeze_encoder:
            for p in self.mt5_encoder.parameters():
                p.requires_grad = False
            self.mt5_encoder.eval()
            print(f"[MT5SignLMv2] mT5 Encoder: FROZEN ({self.mt5_hidden_dim}d)")
        else:
            print(f"[MT5SignLMv2] mT5 Encoder: TRAINABLE ({self.mt5_hidden_dim}d)")
        
        # Text projection
        if self.mt5_hidden_dim != hidden_dim:
            self.text_proj = nn.Sequential(
                nn.Linear(self.mt5_hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        
        # =========================================
        # 2. Motion Placeholder
        # =========================================
        self.num_motion_tokens = motion_holder_repeat
        self.motion_placeholder = nn.Parameter(
            torch.randn(1, self.num_motion_tokens, hidden_dim) * 0.1
        )
        
        # =========================================
        # 3. ★ ATII (Adaptive Textual Information Injector)
        # =========================================
        if use_atii:
            self.atii = ATII(hidden_dim)
            print(f"[MT5SignLMv2] ★ ATII: ENABLED")
        else:
            self.atii = None
        
        # =========================================
        # 4. ★ LGL Pattern: Local → Global → Local
        # =========================================
        if use_limm:
            # Local (LIMM)
            self.local_pre = LIMM(hidden_dim, kernel_size=limm_kernel_size)
            
            # Global (Shared Attention)
            self.shared_attn_layers = nn.ModuleList([
                SharedAttentionBlock(hidden_dim, motion_branch_heads, bidirectional=bidirectional_attention)
                for _ in range(num_shared_layers)
            ])
            
            # Local (LIMM)
            self.local_post = LIMM(hidden_dim, kernel_size=limm_kernel_size)
            
            print(f"[MT5SignLMv2] ★ LGL Pattern: LIMM → {num_shared_layers}x SharedAttn → LIMM")
        else:
            # 기존 구조 (fallback)
            self.local_pre = None
            self.local_post = None
            self.shared_attn_layers = nn.ModuleList([
                SharedAttentionBlock(hidden_dim, motion_branch_heads, bidirectional=bidirectional_attention)
                for _ in range(num_shared_layers)
            ])
        
        self.motion_final_ln = nn.LayerNorm(hidden_dim)
        
        # =========================================
        # 5. Diffusion Head
        # =========================================
        if diffhead is not None:
            diffhead_cfg = diffhead.copy()
            diffhead_cfg['params'] = diffhead_cfg.get('params', {}).copy()
            diffhead_cfg['params']['target_channels'] = vae_latent_channels
            diffhead_cfg['params']['z_channels'] = hidden_dim if multi_hidden else hidden_dim * self.num_motion_tokens
            diffhead_cfg['params']['target_size'] = 1
            self.diffloss = instantiate_from_config(diffhead_cfg)
        else:
            self.diffloss = None
            self.motion_head = nn.Sequential(
                nn.Linear(hidden_dim * self.num_motion_tokens, 1024),
                nn.SiLU(),
                nn.Linear(1024, 1024),
                nn.SiLU(),
                nn.Linear(1024, vae_latent_channels),
            )
        
        # =========================================
        # 6. CFG
        # =========================================
        self.fake_latent = nn.Parameter(torch.zeros(1, self.num_motion_tokens, hidden_dim))
        self.mean_std_inv = None
        
        # =========================================
        # 7. M2T Decoder (Cycle Consistency)
        # =========================================
        if use_cycle_consistency:
            self.motion_to_hidden = nn.Sequential(
                nn.Linear(vae_latent_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            self.m2t_decoder = MotionToTextDecoder(
                hidden_dim=hidden_dim,
                vocab_size=len(self.tokenizer),
                num_layers=m2t_decoder_layers,
                num_heads=motion_branch_heads,
                max_length=max_length,
            )
            print(f"[MT5SignLMv2] ★ Cycle Consistency: ENABLED (weight={cycle_weight})")
        else:
            self.motion_to_hidden = None
            self.m2t_decoder = None
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MT5SignLMv2] Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")
    
    def _load_mt5(self, model_name: str):
        """mT5 모델 로드"""
        print(f"[MT5SignLMv2] Loading mT5 from: {model_name}")
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.mt5_encoder = MT5EncoderModel.from_pretrained(model_name)
        self.mt5_hidden_dim = self.mt5_encoder.config.d_model
        print(f"[MT5SignLMv2] mT5: {self.mt5_hidden_dim}d, vocab={len(self.tokenizer)}")
    
    @property
    def device(self):
        return self.motion_placeholder.device
    
    def _compute_text_pooling(self, text_hidden, mask):
        """Max pooling"""
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        masked_hidden = text_hidden.masked_fill(~mask_bool.unsqueeze(-1), float('-inf'))
        pooled, _ = masked_hidden.max(dim=1)
        has_valid = mask_bool.any(dim=1, keepdim=True).expand_as(pooled)
        pooled = torch.where(has_valid, pooled, text_hidden.mean(dim=1))
        return pooled
    
    def encode_text(self, texts: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        """mT5 텍스트 인코딩"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.mt5_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_hidden = out.last_hidden_state
        else:
            out = self.mt5_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_hidden = out.last_hidden_state
        
        text_hidden = self.text_proj(text_hidden)
        
        return text_hidden, attention_mask, input_ids
    
    def forward(
        self,
        texts: List[str] = None,
        motion_feats: Tensor = None,
        motion_encode_net = None,
        lengths: List[int] = None,
        tasks: Optional[List[Dict]] = None,
        motion_z: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if texts is None and text is not None:
            texts = text
        if texts is None:
            raise ValueError("texts required")
        
        B = len(texts)
        device = self.device
        
        self._step_count += 1
        do_log = (self._step_count % 100 == 1)
        
        # 1. Target z
        if motion_z is not None:
            target_z = motion_z
            if target_z.dim() == 3:
                target_z = target_z.squeeze(0) if target_z.shape[0] == 1 else target_z.squeeze(1)
        elif motion_feats is not None and motion_encode_net is not None:
            with torch.no_grad():
                z, _ = motion_encode_net.encode(motion_feats, lengths)
                target_z = z.squeeze(0) if z.shape[0] == 1 else z.squeeze(1) if z.dim() == 3 else z
        else:
            raise ValueError("motion_z or motion_feats required")
        
        if do_log:
            print(f"\n[Step {self._step_count}] target_z: {target_z.mean():.4f}±{target_z.std():.4f}")
        
        # 2. Encode text
        text_hidden, mask, input_ids = self.encode_text(texts)
        text_pool = self._compute_text_pooling(text_hidden, mask)
        
        if do_log:
            print(f"  text_hidden: {text_hidden.mean():.4f}±{text_hidden.std():.4f}")
        
        # 3. Motion initialization
        motion_hidden = self.motion_placeholder.expand(B, -1, -1).clone()
        
        # ★ ATII: Adaptive text injection
        if self.atii is not None:
            motion_hidden = self.atii(motion_hidden, text_pool)
        else:
            # Fallback: simple addition
            motion_hidden = motion_hidden + text_pool.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
        
        # 4. CFG dropout
        if self.training and self.do_classifier_free_guidance:
            drop_mask = torch.rand(B, device=device) < self.guidance_uncondp
            if drop_mask.any():
                text_hidden = text_hidden.clone()
                motion_hidden = motion_hidden.clone()
                text_hidden[drop_mask] = 0.0
                motion_hidden[drop_mask] = self.fake_latent.expand(drop_mask.sum(), -1, -1)
        
        # =========================================
        # 5. ★ LGL Pattern: Local → Global → Local
        # =========================================
        
        # Local pre (LIMM)
        if self.local_pre is not None:
            motion_hidden = self.local_pre(motion_hidden)
        
        # Global (Shared Attention)
        for shared in self.shared_attn_layers:
            text_hidden, motion_hidden = shared(text_hidden, motion_hidden, mask)
        
        # Local post (LIMM)
        if self.local_post is not None:
            motion_hidden = self.local_post(motion_hidden)
        
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        if do_log:
            print(f"  motion_hidden: {motion_hidden.mean():.4f}±{motion_hidden.std():.4f}")
        
        # 6. Diffusion Loss
        if self.diffloss is not None:
            cond = motion_hidden if self.multi_hidden else motion_hidden.reshape(B, -1)
            if target_z.dim() == 2:
                target_z_for_diff = target_z.unsqueeze(1)
            else:
                target_z_for_diff = target_z
            diff_loss = self.diffloss(target_z_for_diff, cond)
        else:
            pred = self.motion_head(motion_hidden.reshape(B, -1))
            diff_loss = F.mse_loss(pred, target_z.squeeze() if target_z.dim() == 3 else target_z)
        
        # 7. Cycle Consistency Loss
        cycle_loss = torch.tensor(0.0, device=device)
        
        if self.use_cycle_consistency and self.training and self.m2t_decoder is not None:
            motion_z_flat = target_z.squeeze(1) if target_z.dim() == 3 else target_z
            m2t_memory = self.motion_to_hidden(motion_z_flat)
            m2t_memory = m2t_memory.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
            
            text_logits = self.m2t_decoder(m2t_memory, target_ids=input_ids, target_mask=mask)
            
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            cycle_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            
            if do_log:
                print(f"  cycle_loss: {cycle_loss.item():.4f}")
        
        # 8. Total Loss
        total_loss = diff_loss + self.cycle_weight * cycle_loss
        
        if do_log:
            print(f"  diff_loss: {diff_loss.item():.4f}, total: {total_loss.item():.4f}")
        
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        from types import SimpleNamespace
        return {'outputs': SimpleNamespace(
            loss=total_loss,
            diff_loss=diff_loss,
            cycle_loss=cycle_loss,
            gpt_loss=torch.tensor(0.0, device=device),
        )}
    
    @torch.no_grad()
    def generate(self, text: Optional[List[str]] = None, lengths: Optional[List[int]] = None, **kwargs) -> Tensor:
        if text is None:
            raise ValueError("text required")
        
        B = len(text)
        text_hidden, mask, _ = self.encode_text(text)
        text_pool = self._compute_text_pooling(text_hidden, mask)
        
        motion_hidden = self.motion_placeholder.expand(B, -1, -1).clone()
        
        if self.atii is not None:
            motion_hidden = self.atii(motion_hidden, text_pool)
        else:
            motion_hidden = motion_hidden + text_pool.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
        
        # LGL
        if self.local_pre is not None:
            motion_hidden = self.local_pre(motion_hidden)
        for shared in self.shared_attn_layers:
            text_hidden, motion_hidden = shared(text_hidden, motion_hidden, mask)
        if self.local_post is not None:
            motion_hidden = self.local_post(motion_hidden)
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        if self.diffloss is not None:
            cond = motion_hidden if self.multi_hidden else motion_hidden.reshape(B, -1)
            z = self.diffloss.sample(cond, cfg=1.0)
        else:
            z = self.motion_head(motion_hidden.reshape(B, -1))
        
        if z.dim() == 2:
            z = z.unsqueeze(0)
        elif z.dim() == 3:
            if z.shape[0] == B and z.shape[1] == 1:
                z = z.permute(1, 0, 2)
        
        return z
    
    def sample_tokens(self, outputs, device, temperature=1.0, cfg=1.0, vae_mean_std_inv=None):
        if isinstance(outputs, Tensor):
            z = outputs
        else:
            z = outputs.get('motion_z', outputs) if isinstance(outputs, dict) else outputs
        
        z = z.to(device)
        
        if z.dim() == 2:
            z = z.unsqueeze(0)
        elif z.dim() == 3 and z.shape[1] == 1:
            z = z.permute(1, 0, 2)
        
        B = z.shape[1] if z.dim() == 3 else z.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        if vae_mean_std_inv is not None:
            z = vae_mean_std_inv(z)
        
        return z, mask
    
    def generate_conditional(self, texts=None, **kwargs):
        return self.generate(text=texts, **kwargs)


# Aliases
MT5SignLM = MT5SignLMv2
mT5SignLMv2 = MT5SignLMv2