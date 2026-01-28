"""
MBart Bimodal LM for Sign Language Generation
==============================================

mBART Encoder (frozen) + MotionGPT3-style Bimodal Architecture

★ 수정사항 (v2):
- Vocab size 불일치 문제 해결
- embedding resize 또는 input_ids clamp

복사 위치: motGPT/archs/mbart_bimodal_lm.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

try:
    from transformers import MBartModel, MBartTokenizer, AutoModel, AutoTokenizer
except ImportError:
    raise ImportError("transformers 라이브러리가 필요합니다: pip install transformers")

from ..config import instantiate_from_config


class SharedAttentionBlock(nn.Module):
    """MotionGPT3 스타일 Shared Attention Block"""
    
    def __init__(
        self, 
        hidden_dim: int = 768, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.motion_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.motion_cross_ln = nn.LayerNorm(hidden_dim)
        
        if bidirectional:
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.text_cross_ln = nn.LayerNorm(hidden_dim)
        
        self.motion_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.motion_self_ln = nn.LayerNorm(hidden_dim)
        
        self.motion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout))
        self.motion_ffn_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, text_hidden, motion_hidden, text_mask=None):
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        
        motion_norm = self.motion_cross_ln(motion_hidden)
        motion_cross_out, _ = self.motion_cross_attn(
            query=motion_norm, key=text_hidden, value=text_hidden, key_padding_mask=key_padding_mask)
        motion_hidden = motion_hidden + motion_cross_out
        
        if self.bidirectional:
            text_norm = self.text_cross_ln(text_hidden)
            text_cross_out, _ = self.text_cross_attn(query=text_norm, key=motion_hidden, value=motion_hidden)
            text_hidden = text_hidden + text_cross_out
        
        motion_norm = self.motion_self_ln(motion_hidden)
        motion_self_out, _ = self.motion_self_attn(query=motion_norm, key=motion_norm, value=motion_norm)
        motion_hidden = motion_hidden + motion_self_out
        
        motion_hidden = motion_hidden + self.motion_ffn(self.motion_ffn_ln(motion_hidden))
        
        return text_hidden, motion_hidden


class MotionBranchLayer(nn.Module):
    """Motion Branch 단일 레이어"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout))
        self.ffn_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x_norm = self.self_attn_ln(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.ffn_ln(x))
        return x


class MBartBimodalLM(nn.Module):
    """mBART Encoder + MotionGPT3-style Bimodal Architecture"""
    
    def __init__(
        self,
        model_path: str = './deps/mbart-h2s-csl-phoenix',
        freeze_encoder: bool = True,
        motion_branch_layers: int = 6,
        motion_branch_heads: int = 8,
        hidden_dim: int = 768,
        num_shared_layers: int = 4,
        shared_layer_start: int = 2,
        bidirectional_attention: bool = False,
        diffhead: Optional[Dict] = None,
        vae_latent_channels: int = 256,
        guidance_uncondp: float = 0.0,
        guidance_scale: float = 1.0,
        motion_holder_repeat: int = 4,
        with_vae_latent_norm: bool = True,
        max_length: int = 256,
        # ★ Contrastive Learning ★
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        contrastive_temp: float = 0.07,
        # Compatibility
        model_type: str = 'mbart_bimodal',
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
        self.freeze_encoder = freeze_encoder
        self.stage = stage
        self._step_count = 0
        
        # =========================================
        # 1. mBART Encoder
        # =========================================
        self._load_mbart(model_path)
        
        if freeze_encoder:
            for p in self.mbart.parameters():
                p.requires_grad = False
            self.mbart.eval()
            print(f"[MBartBimodalLM] mBART: FROZEN ({self.mbart_hidden_dim}d)")
        else:
            print(f"[MBartBimodalLM] mBART: TRAINABLE ({self.mbart_hidden_dim}d)")
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(self.mbart_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # =========================================
        # 2. Motion Branch
        # =========================================
        self.num_motion_tokens = motion_holder_repeat
        self.motion_placeholder = nn.Parameter(
            torch.randn(1, self.num_motion_tokens, hidden_dim) * 0.1)
        
        self.motion_layers_pre = nn.ModuleList([
            MotionBranchLayer(hidden_dim, motion_branch_heads)
            for _ in range(shared_layer_start)])
        
        # =========================================
        # 3. Shared Attention
        # =========================================
        self.shared_attn_layers = nn.ModuleList([
            SharedAttentionBlock(hidden_dim, motion_branch_heads, bidirectional=bidirectional_attention)
            for _ in range(num_shared_layers)])
        
        remaining = motion_branch_layers - shared_layer_start - num_shared_layers
        self.motion_layers_post = nn.ModuleList([
            MotionBranchLayer(hidden_dim, motion_branch_heads)
            for _ in range(max(0, remaining))])
        
        self.motion_final_ln = nn.LayerNorm(hidden_dim)
        
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
                nn.Linear(1024, vae_latent_channels))
        
        # =========================================
        # 5. CFG
        # =========================================
        self.fake_latent = nn.Parameter(torch.zeros(1, self.num_motion_tokens, hidden_dim))
        self.mean_std_inv = None
        
        # =========================================
        # 6. ★ Text-Motion Contrastive Learning ★
        # =========================================
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        
        # Aligned space dimension
        self.align_dim = 256
        
        # Text projector: text_pooled → aligned space
        # ★ Deeper projector로 text-motion gap 메우기
        self.text_align_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 768 → 384
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.align_dim),  # 384 → 256
        )
        self.text_align_ln = nn.LayerNorm(self.align_dim)
        
        # Motion projector: motion_z → aligned space
        # ★ Motion도 deeper하게
        self.motion_align_proj = nn.Sequential(
            nn.Linear(vae_latent_channels, vae_latent_channels),  # 256 → 256
            nn.GELU(),
            nn.Linear(vae_latent_channels, self.align_dim),  # 256 → 256
        )
        self.motion_align_ln = nn.LayerNorm(self.align_dim)
        
        # ★ Aligned → Hidden (generation path에 연결!)
        self.align_to_hidden = nn.Sequential(
            nn.Linear(self.align_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        print(f"[MBartBimodalLM] Ready! Motion: {motion_branch_layers}L, Shared: {num_shared_layers}L")
        print(f"[MBartBimodalLM] ★ Text-Motion Contrastive Learning: ENABLED (weight={self.contrastive_weight})")
        print(f"[MBartBimodalLM] ★ Aligned embedding directly used in generation path!")
    
    def _load_mbart(self, model_path: str):
        """mBART 로딩 + vocab size 검증"""
        print(f"[MBartBimodalLM] Loading mBART from: {model_path}")
        
        model_path_abs = os.path.abspath(model_path) if not os.path.isabs(model_path) else model_path
        
        loaded = False
        
        if os.path.exists(model_path_abs):
            try:
                print(f"[MBartBimodalLM] Loading from local: {model_path_abs}")
                self.mbart = MBartModel.from_pretrained(model_path_abs)
                self.tokenizer = MBartTokenizer.from_pretrained(model_path_abs)
                self.mbart_hidden_dim = self.mbart.config.d_model
                loaded = True
            except Exception as e:
                print(f"[MBartBimodalLM] MBartModel failed: {e}")
                try:
                    self.mbart = AutoModel.from_pretrained(model_path_abs)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path_abs)
                    self.mbart_hidden_dim = getattr(self.mbart.config, 'd_model', self.mbart.config.hidden_size)
                    loaded = True
                except Exception as e2:
                    print(f"[MBartBimodalLM] AutoModel also failed: {e2}")
        
        if not loaded:
            print(f"[MBartBimodalLM] Falling back to HuggingFace...")
            self.mbart = MBartModel.from_pretrained('facebook/mbart-large-50')
            self.tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')
            self.mbart_hidden_dim = self.mbart.config.d_model
        
        # ★★★ Vocab size 검증 및 수정 ★★★
        self._fix_vocab_size_mismatch()
    
    def _fix_vocab_size_mismatch(self):
        """Vocab size 불일치 문제 해결 - Embedding Resize"""
        # 모델의 embedding vocab size
        if hasattr(self.mbart, 'shared'):
            model_vocab_size = self.mbart.shared.num_embeddings
        elif hasattr(self.mbart, 'encoder') and hasattr(self.mbart.encoder, 'embed_tokens'):
            model_vocab_size = self.mbart.encoder.embed_tokens.num_embeddings
        else:
            model_vocab_size = self.mbart.config.vocab_size
        
        # Tokenizer vocab size
        tokenizer_vocab_size = len(self.tokenizer)
        
        print(f"[MBartBimodalLM] Vocab size check:")
        print(f"  Model embedding: {model_vocab_size}")
        print(f"  Tokenizer: {tokenizer_vocab_size}")
        
        if tokenizer_vocab_size != model_vocab_size:
            print(f"[MBartBimodalLM] ⚠️ Vocab size mismatch! Resizing embeddings...")
            
            # 기존 embedding 저장 (평균 초기화용)
            if hasattr(self.mbart, 'shared'):
                old_embeddings = self.mbart.shared.weight.data.clone()
            elif hasattr(self.mbart, 'encoder') and hasattr(self.mbart.encoder, 'embed_tokens'):
                old_embeddings = self.mbart.encoder.embed_tokens.weight.data.clone()
            else:
                old_embeddings = None
            
            old_vocab_size = model_vocab_size
            
            # ★★★ 핵심: Embedding resize ★★★
            self.mbart.resize_token_embeddings(tokenizer_vocab_size)
            
            # 새로 추가된 embedding을 기존 embedding의 평균으로 초기화
            if old_embeddings is not None and tokenizer_vocab_size > old_vocab_size:
                mean_embedding = old_embeddings.mean(dim=0)
                
                if hasattr(self.mbart, 'shared'):
                    self.mbart.shared.weight.data[old_vocab_size:] = mean_embedding
                elif hasattr(self.mbart, 'encoder') and hasattr(self.mbart.encoder, 'embed_tokens'):
                    self.mbart.encoder.embed_tokens.weight.data[old_vocab_size:] = mean_embedding
                    if hasattr(self.mbart, 'decoder') and hasattr(self.mbart.decoder, 'embed_tokens'):
                        self.mbart.decoder.embed_tokens.weight.data[old_vocab_size:] = mean_embedding
                
                print(f"[MBartBimodalLM] ✓ New embeddings initialized with mean of existing embeddings")
            
            print(f"[MBartBimodalLM] ✓ Embeddings resized: {old_vocab_size} → {tokenizer_vocab_size}")
            
            # Config도 업데이트
            self.mbart.config.vocab_size = tokenizer_vocab_size
        else:
            print(f"[MBartBimodalLM] ✓ Vocab sizes match!")
        
        self._need_clamp = False
        self._max_token_id = tokenizer_vocab_size - 1
    
    @property
    def device(self):
        return self.motion_placeholder.device
    
    def _compute_text_pooling(self, text_hidden, mask):
        """Max pooling - 더 구별되는 feature 추출"""
        # Ensure mask is boolean
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        
        # Mask out padding tokens with very negative values
        masked_hidden = text_hidden.masked_fill(~mask_bool.unsqueeze(-1), float('-inf'))
        
        # Max pooling over sequence dimension
        pooled, _ = masked_hidden.max(dim=1)  # [B, hidden_dim]
        
        # Handle all-masked case (replace -inf with mean)
        has_valid = mask_bool.any(dim=1, keepdim=True).expand_as(pooled)
        pooled = torch.where(
            has_valid,
            pooled,
            text_hidden.mean(dim=1)  # fallback to mean
        )
        return pooled
    
    def encode_text(self, texts: List[str]):
        """mBART로 텍스트 인코딩"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        
        # mBART encoder forward
        if self.freeze_encoder:
            with torch.no_grad():
                if hasattr(self.mbart, 'encoder'):
                    out = self.mbart.encoder(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = self.mbart(input_ids=input_ids, attention_mask=attention_mask)
                text_hidden = out.last_hidden_state
        else:
            if hasattr(self.mbart, 'encoder'):
                out = self.mbart.encoder(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = self.mbart(input_ids=input_ids, attention_mask=attention_mask)
            text_hidden = out.last_hidden_state
        
        return self.text_proj(text_hidden), attention_mask
    
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
        text_hidden, mask = self.encode_text(texts)
        
        if do_log:
            print(f"  text_hidden: {text_hidden.mean():.4f}±{text_hidden.std():.4f}")
        
        # 3. Motion init + text pooling → ★ Aligned embedding 사용!
        motion_hidden = self.motion_placeholder.expand(B, -1, -1).clone()
        text_pool = self._compute_text_pooling(text_hidden, mask)  # [B, 768]
        
        # ★★★ 핵심: Aligned space를 거쳐서 generation에 사용 ★★★
        # 이렇게 하면 contrastive loss gradient가 직접 generation path에 영향!
        text_aligned = self.text_align_proj(text_pool)  # [B, 256] - aligned space
        text_for_motion = self.align_to_hidden(text_aligned)  # [B, 768] - back to hidden
        motion_hidden = motion_hidden + text_for_motion.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
        
        # 4. CFG dropout
        if self.training and self.do_classifier_free_guidance:
            drop_mask = torch.rand(B, device=device) < self.guidance_uncondp
            if drop_mask.any():
                text_hidden = text_hidden.clone()
                motion_hidden = motion_hidden.clone()
                text_hidden[drop_mask] = 0.0
                motion_hidden[drop_mask] = self.fake_latent.expand(drop_mask.sum(), -1, -1)
        
        # 5. Pre-shared
        for layer in self.motion_layers_pre:
            motion_hidden = layer(motion_hidden)
        
        # 6. Shared attention
        for shared in self.shared_attn_layers:
            text_hidden, motion_hidden = shared(text_hidden, motion_hidden, mask)
        
        # 7. Post-shared
        for layer in self.motion_layers_post:
            motion_hidden = layer(motion_hidden)
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        if do_log:
            print(f"  motion_hidden: {motion_hidden.mean():.4f}±{motion_hidden.std():.4f}")
        
        # 8. Loss
        if self.diffloss is not None:
            cond = motion_hidden if self.multi_hidden else motion_hidden.reshape(B, -1)
            if target_z.dim() == 2:
                target_z = target_z.unsqueeze(1)
            diff_loss = self.diffloss(target_z, cond)
        else:
            pred = self.motion_head(motion_hidden.reshape(B, -1))
            diff_loss = F.mse_loss(pred, target_z.squeeze() if target_z.dim() == 3 else target_z)
        
        # =========================================
        # 9. ★ Text-Motion Contrastive Loss ★
        # =========================================
        contrastive_loss = torch.tensor(0.0, device=device)
        
        if self.use_contrastive and self.training:
            # text_aligned는 위에서 이미 계산됨! (generation path에 직접 사용)
            # 여기서는 normalize만 해서 contrastive 계산
            text_aligned_norm = F.normalize(text_aligned, dim=-1)
            
            # target_z: [B, 1, 256] or [B, 256] → [B, 256]
            motion_z_flat = target_z.squeeze(1) if target_z.dim() == 3 else target_z
            
            # ★ 단순화된 motion projection + LayerNorm
            motion_aligned = self.motion_align_proj(motion_z_flat)
            motion_aligned_ln = self.motion_align_ln(motion_aligned)
            motion_aligned_norm = F.normalize(motion_aligned_ln, dim=-1)
            
            # text도 LayerNorm 적용
            text_aligned_ln = self.text_align_ln(text_aligned)
            text_aligned_norm = F.normalize(text_aligned_ln, dim=-1)
            
            # ★★★ 디버그: alignment 상태 확인 ★★★
            if do_log:
                # variance 확인 (batch dimension) - LayerNorm 전 값으로!
                text_pool_var = text_pool.var(dim=0).mean().item()
                text_var = text_aligned.var(dim=0).mean().item()
                motion_var = motion_aligned.var(dim=0).mean().item()
                motion_z_var = motion_z_flat.var(dim=0).mean().item()
                
                # Cosine similarity (같은 pair)
                diag_sim = (text_aligned_norm * motion_aligned_norm).sum(dim=-1).mean().item()
                
                print(f"  [Contrastive Debug]")
                print(f"    text_pool var (768d): {text_pool_var:.6f}")
                print(f"    text_aligned var (256d): {text_var:.6f}")
                print(f"    motion_z var (256d): {motion_z_var:.6f}")
                print(f"    motion_aligned var (256d): {motion_var:.6f}")
                print(f"    diagonal similarity: {diag_sim:.4f} (should increase)")
            
            # InfoNCE Loss (symmetric)
            # logits[i,j] = similarity(text_i, motion_j)
            logits = torch.matmul(text_aligned_norm, motion_aligned_norm.T) / self.contrastive_temp
            labels = torch.arange(B, device=device)
            
            # ★★★ 추가 디버그: logits 분석 ★★★
            if do_log:
                diag_logits = logits.diag().mean().item()
                off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                off_diag_logits = logits[off_diag_mask].mean().item()
                logits_gap = diag_logits - off_diag_logits
                print(f"    logits diag: {diag_logits:.4f}, off-diag: {off_diag_logits:.4f}, gap: {logits_gap:.4f}")
            
            # Text → Motion direction
            loss_t2m = F.cross_entropy(logits, labels)
            # Motion → Text direction
            loss_m2t = F.cross_entropy(logits.T, labels)
            
            contrastive_loss = (loss_t2m + loss_m2t) / 2
            
            if do_log:
                print(f"    log(batch_size={B}): {torch.log(torch.tensor(float(B))).item():.4f}")
                print(f"    contrastive_loss: {contrastive_loss.item():.4f}")
        
        # Total loss
        total_loss = diff_loss + self.contrastive_weight * contrastive_loss
        
        if do_log:
            print(f"  diff_loss: {diff_loss.item():.4f}")
            if self.use_contrastive:
                print(f"  contrastive_loss: {contrastive_loss.item():.4f}")
                print(f"  total_loss: {total_loss.item():.4f}")
        
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        from types import SimpleNamespace
        return {'outputs': SimpleNamespace(
            loss=total_loss, 
            diff_loss=diff_loss, 
            contrastive_loss=contrastive_loss,
            gpt_loss=torch.tensor(0.0, device=device)
        )}
    
    @torch.no_grad()
    def generate(self, text: Optional[List[str]] = None, lengths: Optional[List[int]] = None, **kwargs) -> Tensor:
        if text is None:
            raise ValueError("text required")
        
        B = len(text)
        text_hidden, mask = self.encode_text(text)
        
        # ★★★ Forward와 동일하게 aligned embedding 사용 ★★★
        motion_hidden = self.motion_placeholder.expand(B, -1, -1).clone()
        text_pool = self._compute_text_pooling(text_hidden, mask)
        text_aligned = self.text_align_proj(text_pool)
        text_for_motion = self.align_to_hidden(text_aligned)
        motion_hidden = motion_hidden + text_for_motion.unsqueeze(1).expand(-1, self.num_motion_tokens, -1)
        
        for layer in self.motion_layers_pre:
            motion_hidden = layer(motion_hidden)
        for shared in self.shared_attn_layers:
            text_hidden, motion_hidden = shared(text_hidden, motion_hidden, mask)
        for layer in self.motion_layers_post:
            motion_hidden = layer(motion_hidden)
        motion_hidden = self.motion_final_ln(motion_hidden)
        
        if self.diffloss is not None:
            cond = motion_hidden if self.multi_hidden else motion_hidden.reshape(B, -1)
            z = self.diffloss.sample(cond, cfg=1.0)
        else:
            z = self.motion_head(motion_hidden.reshape(B, -1))
        
        # ★★★ VAE decoder expects shape [1, B, latent_dim] ★★★
        if z.dim() == 2:
            # [B, 256] -> [1, B, 256]
            z = z.unsqueeze(0)
        elif z.dim() == 3:
            if z.shape[0] == B and z.shape[1] == 1:
                # [B, 1, 256] -> [1, B, 256]
                z = z.permute(1, 0, 2)
            elif z.shape[0] == 1 and z.shape[1] == B:
                # Already [1, B, 256] - OK
                pass
            else:
                # Unexpected shape - take mean over time and reshape
                # [B, T, 256] -> [B, 256] -> [1, B, 256]
                z = z.mean(dim=1).unsqueeze(0)
        
        return z
    
    def sample_tokens(self, outputs, device, temperature=1.0, cfg=1.0, vae_mean_std_inv=None):
        """motgpt.py 호환 메서드"""
        if isinstance(outputs, Tensor):
            z = outputs
        else:
            z = outputs.get('motion_z', outputs) if isinstance(outputs, dict) else outputs
        
        z = z.to(device)
        
        # Shape 정규화: [1, B, 256] 형태로
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [B, 256] -> [1, B, 256]
        elif z.dim() == 3 and z.shape[0] != 1:
            if z.shape[1] == 1:
                z = z.permute(1, 0, 2)  # [B, 1, 256] -> [1, B, 256]
        
        B = z.shape[1] if z.dim() == 3 else z.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        if vae_mean_std_inv is not None:
            z = vae_mean_std_inv(z)
        
        return z, mask
    
    def generate_conditional(self, texts=None, **kwargs):
        return self.generate(text=texts, **kwargs)