"""
MBart Hybrid LM for Sign Language Generation - FIXED VERSION v2
================================================================

핵심 수정사항:
1. Text pooling을 motion queries에 직접 추가 (text conditioning 강화)
2. CFG dropout 시 text_features도 zero masking (올바른 unconditional training)
3. fake_text_pool 추가
4. generate()도 동일하게 수정

복사 위치: motGPT/archs/mbart_hybrid_lm.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict, Union
from transformers import MBartModel, MBartTokenizer
from ..config import instantiate_from_config


class MBartHybridLM(nn.Module):
    """
    mBART Encoder (다국어, frozen/trainable) + Motion Branch + Diffusion Head
    
    ★ v2 수정사항:
    - Text pooling을 motion queries에 직접 추가
    - CFG dropout 시 text_features도 masking
    - generate()에서 올바른 CFG 적용
    """
    
    def __init__(
        self,
        model_path: str = './deps/mbart-h2s-csl-phoenix',
        model_type: str = 'mbart_hybrid',
        stage: str = 'lm_pretrain',
        freeze_encoder: bool = True,
        motion_branch_layers: int = 6,
        motion_branch_heads: int = 8,
        diffhead: Optional[Dict] = None,
        vae_latent_channels: int = 256,
        vae_latent_size: Optional[int] = None,
        motion_holder_repeat: int = 4,
        holder_num_in_input: int = 4,
        motion_holder_seq_mode: str = 'withse',
        with_hid_norm: bool = False,
        with_vae_latent_norm: bool = True,
        diffusion_batch_mul: int = 4,
        guidance_uncondp: float = 0.1,
        predict_epsilon: bool = True,
        guidance_scale: float = 3.0,
        fake_latent_mode: str = 'learnable_zero',
        max_length: int = 256,
        motion_codebook_size: int = 512,
        ablation: dict = None,
        mot_factor: float = 1.0,
        attention_mode: str = 'all',
        **kwargs,
    ):
        super().__init__()
        
        self.stage = stage
        self.max_length = max_length
        self.motion_holder_repeat = motion_holder_repeat
        self.with_vae_latent_norm = with_vae_latent_norm
        self.diffusion_batch_mul = diffusion_batch_mul
        self.guidance_uncondp = guidance_uncondp
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_uncondp > 0
        self.vae_latent_channels = vae_latent_channels
        self.vae_latent_size = vae_latent_size
        self.freeze_encoder = freeze_encoder
        
        # Motion branch hidden dim
        self.llm_decoder_embed_dim = 768
        self.hidden_dim = self.llm_decoder_embed_dim * motion_holder_repeat
        self.multi_hidden = diffhead.get('params', {}).get('multi_hidden', False) if diffhead else False
        
        # =========================================
        # 1. mBART Encoder
        # =========================================
        print(f"[MBartHybridLM] Loading mBART from {model_path}")
        self.tokenizer = MBartTokenizer.from_pretrained(model_path)
        self.tokenizer.add_tokens(['en_ASL', 'zh_CSL', 'de_DGS'], special_tokens=True)
        
        mbart = MBartModel.from_pretrained(model_path)
        mbart.resize_token_embeddings(len(self.tokenizer))
        self.mbart_encoder = mbart.encoder
        self.mbart_hidden_dim = mbart.config.d_model  # 1024
        
        if freeze_encoder:
            for p in self.mbart_encoder.parameters():
                p.requires_grad = False
            self.mbart_encoder.eval()
            print(f"[MBartHybridLM] mBART Encoder: FROZEN")
        else:
            print(f"[MBartHybridLM] mBART Encoder: TRAINABLE")
        
        # =========================================
        # 2. Text Projection (1024 → 768)
        # =========================================
        self.text_proj = nn.Sequential(
            nn.Linear(self.mbart_hidden_dim, self.llm_decoder_embed_dim),
            nn.LayerNorm(self.llm_decoder_embed_dim),
        )
        
        # ★ NEW: Text pooling projection (for direct injection into motion queries)
        self.text_pool_proj = nn.Sequential(
            nn.Linear(self.llm_decoder_embed_dim, self.llm_decoder_embed_dim),
            nn.GELU(),
            nn.Linear(self.llm_decoder_embed_dim, self.llm_decoder_embed_dim),
        )
        
        # =========================================
        # 3. Motion Branch (Transformer Decoder)
        # =========================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.llm_decoder_embed_dim,
            nhead=motion_branch_heads,
            dim_feedforward=self.llm_decoder_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.motion_branch = nn.TransformerDecoder(decoder_layer, num_layers=motion_branch_layers)
        
        # Learnable placeholder tokens for motion output
        self.num_motion_tokens = motion_holder_repeat
        self.motion_placeholder = nn.Parameter(torch.randn(1, self.num_motion_tokens, self.llm_decoder_embed_dim) * 0.02)
        
        # =========================================
        # 4. Diffusion Head
        # =========================================
        if diffhead is not None:
            diffhead_cfg = diffhead.copy()
            diffhead_cfg['params'] = diffhead_cfg.get('params', {}).copy()
            diffhead_cfg['params']['target_channels'] = vae_latent_channels
            diffhead_cfg['params']['z_channels'] = self.hidden_dim if not self.multi_hidden else self.llm_decoder_embed_dim
            diffhead_cfg['params']['target_size'] = 1
            self.diffloss = instantiate_from_config(diffhead_cfg)
        else:
            self.diffloss = None
            self.motion_head = nn.Sequential(
                nn.Linear(self.hidden_dim, 1024), nn.SiLU(),
                nn.Linear(1024, 1024), nn.SiLU(),
                nn.Linear(1024, vae_latent_channels),
            )
        
        # =========================================
        # 5. Classifier-free guidance components
        # =========================================
        # Learnable fake latent for unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, self.num_motion_tokens, self.llm_decoder_embed_dim))
        # ★ NEW: Fake text pooling for unconditional generation
        self.fake_text_pool = nn.Parameter(torch.zeros(1, 1, self.llm_decoder_embed_dim))
        
        # For compatibility
        self.mean_std_inv = None
        self.with_hid_norm = with_hid_norm and self.multi_hidden
        if self.with_hid_norm:
            self.norm_layer = nn.LayerNorm(self.llm_decoder_embed_dim)
            self.diffusion_pos_embed_learned = nn.Parameter(
                torch.zeros(1, self.num_motion_tokens, self.llm_decoder_embed_dim)
            )
            nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        
        print(f"[MBartHybridLM] Ready: mBART {self.mbart_hidden_dim}d → Motion Branch {self.llm_decoder_embed_dim}d")
        print(f"[MBartHybridLM] Motion tokens: {self.num_motion_tokens}, Multi-hidden: {self.multi_hidden}")
        print(f"[MBartHybridLM] CFG: uncondp={guidance_uncondp}, scale={guidance_scale}")
        print(f"[MBartHybridLM] ★ Text pooling injection: ENABLED")
    
    @property
    def device(self):
        return self.motion_placeholder.device
    
    def _get_src_lang(self, src):
        """Get mBART language code from source dataset"""
        lang_map = {'how2sign': 'en_XX', 'csl': 'zh_CN', 'phoenix': 'de_DE'}
        return lang_map.get(src, 'en_XX')
    
    def _compute_text_pooling(self, text_features, attention_mask):
        """
        Compute masked mean pooling of text features
        
        Args:
            text_features: [B, seq_len, 768]
            attention_mask: [B, seq_len] (1 = valid, 0 = padding)
        
        Returns:
            text_pooled: [B, 768]
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq, 1]
        text_sum = (text_features * mask_expanded).sum(dim=1)  # [B, 768]
        text_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        text_pooled = text_sum / text_count  # [B, 768]
        return text_pooled
    
    def encode_text(self, texts: List[str], data_src: Optional[List[str]] = None):
        """
        Encode texts using mBART encoder
        
        Args:
            texts: List of text strings
            data_src: List of source datasets ('how2sign', 'csl', 'phoenix')
        
        Returns:
            text_features: [B, seq_len, 768]
            attention_mask: [B, seq_len]
        """
        if data_src is None:
            data_src = ['how2sign'] * len(texts)
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        
        # mBART encoder forward
        # Use no_grad only if encoder is frozen
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_outputs = self.mbart_encoder(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                )
        else:
            encoder_outputs = self.mbart_encoder(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
            )
        
        # Project to motion branch dimension
        text_features = self.text_proj(encoder_outputs.last_hidden_state)  # [B, seq, 768]
        
        return text_features, encodings['attention_mask']
    
    def forward(
        self,
        texts: List[str] = None,
        motion_feats: Tensor = None,
        motion_encode_net = None,
        lengths: List[int] = None,
        tasks: Optional[List[Dict]] = None,
        text_emb: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        data_src: Optional[List[str]] = None,
        motion_z: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """
        Training forward pass - FIXED VERSION
        
        ★ 핵심 수정:
        1. Text pooling을 motion queries에 직접 추가
        2. CFG dropout 시 text_features와 text_pooling 모두 masking
        
        Supports two calling conventions:
        1. Original: forward(texts, motion_feats, motion_encode_net, lengths, ...)
        2. motgpt.py style: forward(motion_z=..., text=..., lengths=...)
        """
        # Handle aliased arguments
        if texts is None and text is not None:
            texts = text
        
        if texts is None:
            raise ValueError("Either 'texts' or 'text' must be provided")
        
        batch_size = len(texts)
        
        # Determine device from available tensor
        if motion_z is not None:
            device = motion_z.device
        elif motion_feats is not None:
            device = motion_feats.device
        elif text_emb is not None:
            device = text_emb.device
        else:
            device = self.device
        
        # =========================================
        # 1. Encode text (or use cached)
        # =========================================
        if text_emb is not None and text_mask is not None:
            text_features = text_emb.to(device)
            attention_mask = text_mask.to(device)
        else:
            text_features, attention_mask = self.encode_text(texts, data_src)
        
        # =========================================
        # 2. Get motion latent (encode if needed)
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
            if self.with_vae_latent_norm and hasattr(motion_encode_net, 'mean_std_inv') and motion_encode_net.mean_std_inv is not None:
                target_z = motion_encode_net.mean_std_inv(target_z.clone())
        else:
            raise ValueError("Either 'motion_z' or ('motion_feats' + 'motion_encode_net') must be provided")
        
        # =========================================
        # 3. Motion Branch forward - ★ FIXED ★
        # =========================================
        # Expand placeholder tokens for batch
        motion_queries = self.motion_placeholder.expand(batch_size, -1, -1)  # [B, K, 768]
        
        # ★ Text pooling: 핵심 수정! Text 정보를 motion queries에 직접 주입
        text_pooled = self._compute_text_pooling(text_features, attention_mask)  # [B, 768]
        text_pooled = self.text_pool_proj(text_pooled)  # [B, 768]
        text_pooled_expanded = text_pooled.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)  # [B, K, 768]
        
        # ★ CFG Dropout: text_features와 text_pooling 모두 masking
        if self.training and self.do_classifier_free_guidance:
            mask = torch.rand(batch_size, device=device) < self.guidance_uncondp
            if mask.any():
                # Clone to avoid in-place modification
                text_features = text_features.clone()
                motion_queries = motion_queries.clone()
                text_pooled_expanded = text_pooled_expanded.clone()
                
                # ★ 핵심: Text features를 zero로 (unconditional)
                text_features[mask] = 0.0
                
                # Motion queries와 text pooling을 fake로 교체
                motion_queries[mask] = self.fake_latent.expand(mask.sum(), -1, -1)
                text_pooled_expanded[mask] = self.fake_text_pool.expand(mask.sum(), self.num_motion_tokens, -1)
        
        # ★ Add text pooling to motion queries (핵심!)
        motion_queries = motion_queries + text_pooled_expanded
        
        # Create memory mask from attention mask
        memory_key_padding_mask = ~attention_mask.bool()  # True = padding
        
        # Transformer decoder: query motion from text
        motion_hidden = self.motion_branch(
            tgt=motion_queries,
            memory=text_features,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, K, 768]
        
        # =========================================
        # 4. Compute diffusion loss
        # =========================================
        if self.diffloss is not None:
            if self.multi_hidden:
                if self.with_hid_norm:
                    motion_hidden = self.norm_layer(motion_hidden)
                    motion_hidden = motion_hidden + self.diffusion_pos_embed_learned
                condition = motion_hidden  # [B, K, 768]
            else:
                condition = motion_hidden.reshape(batch_size, -1)  # [B, K*768]
            
            # DiffLoss expects [B, 1, 256]
            if target_z.dim() == 2:
                target_z = target_z.unsqueeze(1)
            elif target_z.dim() == 3 and target_z.shape[0] == 1:
                target_z = target_z.permute(1, 0, 2)
            
            diff_loss = self.diffloss(target_z, condition)
        else:
            pred_z = self.motion_head(motion_hidden.reshape(batch_size, -1))
            target_flat = target_z.squeeze() if target_z.dim() == 3 else target_z
            diff_loss = nn.functional.mse_loss(pred_z, target_flat)
        
        # Return format compatible with motgpt.py
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
        text_emb: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        data_src: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Generate motion latent from text (inference) - FIXED VERSION
        
        Args:
            text: List of text strings
            lengths: Motion lengths (unused, for interface compatibility)
            text_emb: Pre-computed embeddings
            text_mask: Pre-computed mask
            data_src: Source datasets
        
        Returns:
            motion_z: [1, B, 256] motion latent (before VAE decode)
        """
        if text is None and text_emb is None:
            raise ValueError("Either 'text' or 'text_emb' must be provided")
        
        batch_size = len(text) if text is not None else text_emb.shape[0]
        device = self.device
        
        # =========================================
        # 1. Encode text
        # =========================================
        if text_emb is not None and text_mask is not None:
            text_features = text_emb.to(device)
            attention_mask = text_mask.to(device)
        else:
            text_features, attention_mask = self.encode_text(text, data_src)
        
        # =========================================
        # 2. Conditional Motion Branch forward
        # =========================================
        motion_queries = self.motion_placeholder.expand(batch_size, -1, -1)
        
        # ★ Text pooling injection
        text_pooled = self._compute_text_pooling(text_features, attention_mask)
        text_pooled = self.text_pool_proj(text_pooled)
        text_pooled_expanded = text_pooled.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
        
        # Add text info to queries
        cond_queries = motion_queries + text_pooled_expanded
        
        memory_key_padding_mask = ~attention_mask.bool()
        
        cond_hidden = self.motion_branch(
            tgt=cond_queries,
            memory=text_features,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        # =========================================
        # 3. Diffusion sampling
        # =========================================
        if self.diffloss is not None:
            if self.multi_hidden:
                if self.with_hid_norm:
                    cond_hidden = self.norm_layer(cond_hidden)
                    cond_hidden = cond_hidden + self.diffusion_pos_embed_learned
                cond_condition = cond_hidden
            else:
                cond_condition = cond_hidden.reshape(batch_size, -1)
            
            # Apply classifier-free guidance at inference
            if self.do_classifier_free_guidance and self.guidance_scale > 1.0:
                # ★ Unconditional: use fake_latent + fake_text_pool + zero text_features
                uncond_queries = self.fake_latent.expand(batch_size, -1, -1)
                uncond_text_pool = self.fake_text_pool.expand(batch_size, self.num_motion_tokens, -1)
                uncond_queries = uncond_queries + uncond_text_pool
                
                # Zero text features for unconditional
                zero_text_features = torch.zeros_like(text_features)
                
                uncond_hidden = self.motion_branch(
                    tgt=uncond_queries,
                    memory=zero_text_features,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                if self.multi_hidden:
                    if self.with_hid_norm:
                        uncond_hidden = self.norm_layer(uncond_hidden)
                        uncond_hidden = uncond_hidden + self.diffusion_pos_embed_learned
                    uncond_condition = uncond_hidden
                else:
                    uncond_condition = uncond_hidden.reshape(batch_size, -1)
                
                # Concatenate [cond, uncond] for DiffLoss CFG
                condition = torch.cat([cond_condition, uncond_condition], dim=0)
                motion_z = self.diffloss.sample(condition, cfg=self.guidance_scale)
                
                # DiffLoss returns [2*B, ...], take first half (cond results)
                motion_z = motion_z[:batch_size]
            else:
                motion_z = self.diffloss.sample(cond_condition, cfg=1.0)
        else:
            motion_z = self.motion_head(cond_hidden.reshape(batch_size, -1))
        
        # Reshape to [1, B, 256] for VAE decoder compatibility
        if motion_z.dim() == 2:
            motion_z = motion_z.unsqueeze(0)  # [B, 256] → [1, B, 256]
        elif motion_z.dim() == 3:
            if motion_z.shape[1] == 1:
                motion_z = motion_z.permute(1, 0, 2)  # [B, 1, 256] → [1, B, 256]
        
        return motion_z
    
    def sample_tokens(
        self,
        outputs: Dict,
        device: torch.device,
        temperature: float = 1.0,
        cfg: float = 1.0,
        vae_mean_std_inv=None,
    ):
        """
        Sample tokens from generation outputs (for compatibility with motgpt.py)
        """
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
    
    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_feats: Optional[Tensor] = None,
        motion_encode_net=None,
        lengths: Optional[List[int]] = None,
        task: str = "t2m",
        stage: str = 'train',
        tasks: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """
        Conditional generation (for compatibility with some motgpt versions)
        """
        if task == "t2m":
            return self.generate(text=texts, lengths=lengths, **kwargs)
        else:
            raise NotImplementedError(f"Task '{task}' not implemented for MBartHybridLM")