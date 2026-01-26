"""
Multilingual Bimodal LM for Sign Language Generation
=====================================================

MotionGPT3 스타일 Bimodal Architecture + 다국어 지원

Architecture:
    Text Branch (Multilingual LLM, frozen/trainable)
         │
         └── Shared Attention Layer ──┐
                                      │
    Motion Branch (From scratch) ─────┘
         │
    Diffusion Head → Motion Latent

특징:
- MotionGPT3처럼 Shared Attention 사용 (양방향 정보 교환)
- 다국어 LLM (BLOOM, mGPT, Qwen 등) 사용
- 기존 SignGPT3 pipeline과 호환

복사 위치: motGPT/archs/multilingual_bimodal_lm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict
from transformers import AutoModel, AutoTokenizer, AutoConfig
from ..config import instantiate_from_config


class SharedAttentionBlock(nn.Module):
    """
    MotionGPT3 스타일 Shared Attention
    
    Text와 Motion hidden states가 서로를 참조하여 업데이트
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Text → Motion attention
        self.text_to_motion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Motion → Text attention
        self.motion_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.text_ln = nn.LayerNorm(hidden_dim)
        self.motion_ln = nn.LayerNorm(hidden_dim)
        
        # FFN for both
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.motion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.text_ffn_ln = nn.LayerNorm(hidden_dim)
        self.motion_ffn_ln = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        text_hidden: Tensor,      # [B, T_text, D]
        motion_hidden: Tensor,    # [B, T_motion, D]
        text_mask: Optional[Tensor] = None,  # [B, T_text]
    ):
        """
        Bidirectional information exchange between text and motion
        """
        # 1. Motion attends to Text (motion이 text 정보 흡수)
        motion_query = self.motion_ln(motion_hidden)
        motion_attn_out, _ = self.text_to_motion_attn(
            query=motion_query,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=~text_mask.bool() if text_mask is not None else None,
        )
        motion_hidden = motion_hidden + motion_attn_out
        
        # 2. Text attends to Motion (text가 motion 정보 흡수)
        text_query = self.text_ln(text_hidden)
        text_attn_out, _ = self.motion_to_text_attn(
            query=text_query,
            key=motion_hidden,
            value=motion_hidden,
        )
        text_hidden = text_hidden + text_attn_out
        
        # 3. FFN
        motion_hidden = motion_hidden + self.motion_ffn(self.motion_ffn_ln(motion_hidden))
        text_hidden = text_hidden + self.text_ffn(self.text_ffn_ln(text_hidden))
        
        return text_hidden, motion_hidden


class MotionBranchLayer(nn.Module):
    """
    Motion Branch의 단일 레이어 (GPT-2 style Decoder)
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_ln = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        # Self-attention
        x_norm = self.self_attn_ln(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.ffn_ln(x))
        return x


class MultilingualBimodalLM(nn.Module):
    """
    MotionGPT3 스타일 Bimodal LM + 다국어 지원
    
    구조:
    1. Text Branch: 다국어 LLM (BLOOM, mGPT 등)
    2. Motion Branch: GPT-2 style Decoder
    3. Shared Attention: 양방향 정보 교환
    4. Diffusion Head: Motion latent 생성
    """
    
    def __init__(
        self,
        # Text Branch (다국어 LLM)
        text_model_name: str = 'bigscience/bloom-560m',  # 또는 'ai-forever/mGPT'
        freeze_text_branch: bool = True,
        
        # Motion Branch
        motion_branch_layers: int = 6,
        motion_branch_heads: int = 8,
        hidden_dim: int = 768,
        
        # Shared Attention
        num_shared_layers: int = 4,
        shared_layer_start: int = 2,  # Motion Branch의 몇 번째 layer부터 shared
        
        # Diffusion Head
        diffhead: Optional[Dict] = None,
        vae_latent_channels: int = 256,
        
        # CFG
        guidance_uncondp: float = 0.0,
        guidance_scale: float = 1.0,
        
        # Other
        motion_holder_repeat: int = 4,
        with_vae_latent_norm: bool = True,
        max_length: int = 256,
        **kwargs,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.motion_holder_repeat = motion_holder_repeat
        self.guidance_uncondp = guidance_uncondp
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_uncondp > 0
        self.with_vae_latent_norm = with_vae_latent_norm
        self.num_shared_layers = num_shared_layers
        self.shared_layer_start = shared_layer_start
        self.max_length = max_length
        
        # =========================================
        # 1. Text Branch (다국어 LLM)
        # =========================================
        print(f"[MultilingualBimodalLM] Loading text model: {text_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Handle tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_dim = self.text_model.config.hidden_size
        
        if freeze_text_branch:
            for p in self.text_model.parameters():
                p.requires_grad = False
            self.text_model.eval()
            print(f"[MultilingualBimodalLM] Text Branch: FROZEN")
        else:
            print(f"[MultilingualBimodalLM] Text Branch: TRAINABLE")
        
        # Text projection (if dimensions don't match)
        if self.text_hidden_dim != hidden_dim:
            self.text_proj = nn.Linear(self.text_hidden_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()
        
        # =========================================
        # 2. Motion Branch (GPT-2 style)
        # =========================================
        self.num_motion_tokens = motion_holder_repeat
        self.motion_placeholder = nn.Parameter(
            torch.randn(1, self.num_motion_tokens, hidden_dim) * 0.02
        )
        
        # Motion branch layers (before shared attention)
        self.motion_layers_pre = nn.ModuleList([
            MotionBranchLayer(hidden_dim, motion_branch_heads)
            for _ in range(shared_layer_start)
        ])
        
        # Motion branch layers (with shared attention)
        self.motion_layers_shared = nn.ModuleList([
            MotionBranchLayer(hidden_dim, motion_branch_heads)
            for _ in range(num_shared_layers)
        ])
        
        # Motion branch layers (after shared attention)
        remaining_layers = motion_branch_layers - shared_layer_start - num_shared_layers
        if remaining_layers > 0:
            self.motion_layers_post = nn.ModuleList([
                MotionBranchLayer(hidden_dim, motion_branch_heads)
                for _ in range(remaining_layers)
            ])
        else:
            self.motion_layers_post = nn.ModuleList()
        
        self.motion_final_ln = nn.LayerNorm(hidden_dim)
        
        # =========================================
        # 3. Shared Attention Layers
        # =========================================
        self.shared_attn_layers = nn.ModuleList([
            SharedAttentionBlock(hidden_dim, motion_branch_heads)
            for _ in range(num_shared_layers)
        ])
        
        # =========================================
        # 4. Diffusion Head
        # =========================================
        self.multi_hidden = diffhead.get('params', {}).get('multi_hidden', False) if diffhead else False
        
        if diffhead is not None:
            diffhead_cfg = diffhead.copy()
            diffhead_cfg['params'] = diffhead_cfg.get('params', {}).copy()
            diffhead_cfg['params']['target_channels'] = vae_latent_channels
            diffhead_cfg['params']['z_channels'] = hidden_dim if self.multi_hidden else hidden_dim * self.num_motion_tokens
            diffhead_cfg['params']['target_size'] = 1
            self.diffloss = instantiate_from_config(diffhead_cfg)
        else:
            self.diffloss = None
            self.motion_head = nn.Sequential(
                nn.Linear(hidden_dim * self.num_motion_tokens, 1024), nn.SiLU(),
                nn.Linear(1024, 1024), nn.SiLU(),
                nn.Linear(1024, vae_latent_channels),
            )
        
        # =========================================
        # 5. CFG components
        # =========================================
        self.fake_latent = nn.Parameter(torch.zeros(1, self.num_motion_tokens, hidden_dim))
        self.fake_text = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # For compatibility
        self.mean_std_inv = None
        
        print(f"[MultilingualBimodalLM] Ready!")
        print(f"  Text: {text_model_name} ({self.text_hidden_dim}d)")
        print(f"  Motion: {motion_branch_layers} layers, {self.num_motion_tokens} tokens")
        print(f"  Shared: {num_shared_layers} layers (starting at layer {shared_layer_start})")
    
    @property
    def device(self):
        return self.motion_placeholder.device
    
    def encode_text(self, texts: List[str]):
        """
        Encode texts using multilingual LLM
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        
        with torch.set_grad_enabled(self.text_model.training):
            outputs = self.text_model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
            )
        
        # Get last hidden state
        text_hidden = outputs.last_hidden_state  # [B, seq, text_hidden_dim]
        text_hidden = self.text_proj(text_hidden)  # [B, seq, hidden_dim]
        
        return text_hidden, encodings['attention_mask']
    
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
        """
        Training forward pass
        """
        if texts is None and text is not None:
            texts = text
        
        if texts is None:
            raise ValueError("texts must be provided")
        
        batch_size = len(texts)
        device = self.device
        
        # =========================================
        # 1. Get target motion latent
        # =========================================
        if motion_z is not None:
            target_z = motion_z
            if target_z.dim() == 3:
                if target_z.shape[0] == 1:
                    target_z = target_z.squeeze(0)
                elif target_z.shape[1] == 1:
                    target_z = target_z.squeeze(1)
        elif motion_feats is not None and motion_encode_net is not None:
            with torch.no_grad():
                motion_z_enc, _ = motion_encode_net.encode(motion_feats, lengths)
                target_z = motion_z_enc
                if target_z.dim() == 3:
                    if target_z.shape[0] == 1:
                        target_z = target_z.squeeze(0)
                    elif target_z.shape[1] == 1:
                        target_z = target_z.squeeze(1)
        else:
            raise ValueError("Either motion_z or (motion_feats + motion_encode_net) required")
        
        # =========================================
        # 2. Encode text
        # =========================================
        text_hidden, text_mask = self.encode_text(texts)  # [B, seq, D]
        
        # =========================================
        # 3. Initialize motion hidden
        # =========================================
        motion_hidden = self.motion_placeholder.expand(batch_size, -1, -1)  # [B, K, D]
        
        # CFG dropout
        if self.training and self.do_classifier_free_guidance:
            mask = torch.rand(batch_size, device=device) < self.guidance_uncondp
            if mask.any():
                text_hidden = text_hidden.clone()
                motion_hidden = motion_hidden.clone()
                text_hidden[mask] = self.fake_text.expand(mask.sum(), text_hidden.shape[1], -1)
                motion_hidden[mask] = self.fake_latent.expand(mask.sum(), -1, -1)
        
        # =========================================
        # 4. Motion Branch (Pre-shared layers)
        # =========================================
        for layer in self.motion_layers_pre:
            motion_hidden = layer(motion_hidden)
        
        # =========================================
        # 5. Shared Attention Layers (핵심!)
        # =========================================
        for i, (motion_layer, shared_attn) in enumerate(
            zip(self.motion_layers_shared, self.shared_attn_layers)
        ):
            # Bidirectional attention exchange
            text_hidden, motion_hidden = shared_attn(
                text_hidden, motion_hidden, text_mask
            )
            
            # Motion self-attention
            motion_hidden = motion_layer(motion_hidden)
        
        # =========================================
        # 6. Motion Branch (Post-shared layers)
        # =========================================
        for layer in self.motion_layers_post:
            motion_hidden = layer(motion_hidden)
        
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        # =========================================
        # 7. Diffusion Loss
        # =========================================
        if self.diffloss is not None:
            if self.multi_hidden:
                condition = motion_hidden
            else:
                condition = motion_hidden.reshape(batch_size, -1)
            
            if target_z.dim() == 2:
                target_z = target_z.unsqueeze(1)
            
            diff_loss = self.diffloss(target_z, condition)
        else:
            pred_z = self.motion_head(motion_hidden.reshape(batch_size, -1))
            diff_loss = F.mse_loss(pred_z, target_z)
        
        from types import SimpleNamespace
        outputs = SimpleNamespace(
            loss=diff_loss,
            diff_loss=diff_loss,
            gpt_loss=torch.tensor(0.0, device=device),
        )
        return {'outputs': outputs}
    
    @torch.no_grad()
    def generate(
        self,
        text: Optional[List[str]] = None,
        lengths: Optional[List[int]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Generate motion latent from text
        """
        if text is None:
            raise ValueError("text must be provided")
        
        batch_size = len(text)
        device = self.device
        
        # Encode text
        text_hidden, text_mask = self.encode_text(text)
        
        # Initialize motion
        motion_hidden = self.motion_placeholder.expand(batch_size, -1, -1)
        
        # Pre-shared layers
        for layer in self.motion_layers_pre:
            motion_hidden = layer(motion_hidden)
        
        # Shared attention layers
        for motion_layer, shared_attn in zip(self.motion_layers_shared, self.shared_attn_layers):
            text_hidden, motion_hidden = shared_attn(text_hidden, motion_hidden, text_mask)
            motion_hidden = motion_layer(motion_hidden)
        
        # Post-shared layers
        for layer in self.motion_layers_post:
            motion_hidden = layer(motion_hidden)
        
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        # Generate motion latent
        if self.diffloss is not None:
            if self.multi_hidden:
                condition = motion_hidden
            else:
                condition = motion_hidden.reshape(batch_size, -1)
            
            if self.do_classifier_free_guidance and self.guidance_scale > 1.0:
                # Unconditional
                uncond_motion = self.fake_latent.expand(batch_size, -1, -1)
                uncond_text = self.fake_text.expand(batch_size, text_hidden.shape[1], -1)
                
                for layer in self.motion_layers_pre:
                    uncond_motion = layer(uncond_motion)
                for motion_layer, shared_attn in zip(self.motion_layers_shared, self.shared_attn_layers):
                    uncond_text, uncond_motion = shared_attn(uncond_text, uncond_motion, text_mask)
                    uncond_motion = motion_layer(uncond_motion)
                for layer in self.motion_layers_post:
                    uncond_motion = layer(uncond_motion)
                uncond_motion = self.motion_final_ln(uncond_motion)
                
                if self.multi_hidden:
                    uncond_condition = uncond_motion
                else:
                    uncond_condition = uncond_motion.reshape(batch_size, -1)
                
                condition = torch.cat([condition, uncond_condition], dim=0)
                motion_z = self.diffloss.sample(condition, cfg=self.guidance_scale)
                motion_z = motion_z[:batch_size]
            else:
                motion_z = self.diffloss.sample(condition, cfg=1.0)
        else:
            motion_z = self.motion_head(motion_hidden.reshape(batch_size, -1))
        
        # Reshape for VAE
        if motion_z.dim() == 2:
            motion_z = motion_z.unsqueeze(0)
        elif motion_z.dim() == 3 and motion_z.shape[1] == 1:
            motion_z = motion_z.permute(1, 0, 2)
        
        return motion_z
    
    def sample_tokens(self, outputs, device, temperature=1.0, cfg=1.0, vae_mean_std_inv=None):
        if isinstance(outputs, Tensor):
            motion_z = outputs
        else:
            motion_z = outputs.get('motion_z', outputs)
        
        motion_z = motion_z.to(device)
        batch_size = motion_z.shape[0] if motion_z.dim() == 2 else motion_z.shape[1]
        motion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if vae_mean_std_inv is not None:
            motion_z = vae_mean_std_inv(motion_z)
        
        return motion_z, motion_mask
    
    def generate_conditional(self, texts=None, **kwargs):
        return self.generate(text=texts, **kwargs)
