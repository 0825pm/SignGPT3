"""
motGPT/archs/clip_encoder.py
=============================
SignGPT3 Text Encoder - 두 가지 구현

1. CLIPTextEncoder
   - 모델: openai/clip-vit-base-patch32  (ViT-B/32)
   - 출력: [B, 512]
   - 언어: 영어 전용 (ASL How2Sign에 적합)
   - Light-T2M 기본 방식

2. MBartTextEncoder  (SOKE 방식)
   - 모델: facebook/mbart-large-cc25
   - 출력: [B, 512]  (768 → Linear projection)
   - 언어: EN/ZH/DE 멀티링구얼 지원
     en_XX → How2Sign (ASL)
     zh_CN → CSL-Daily (Chinese)
     de_DE → Phoenix-2014T (German)
   - SOKE 핵심: 시퀀스 끝에 [eos][lang_token] 삽입
                src별로 다른 언어 토큰 자동 선택

공통 인터페이스:
    out = encoder(texts, device, src_list=None)
    out["text_emb"]  → [B, 512]       (LDM conditioning에 사용)
    out["hidden"]    → [B, L, D]      (cross-attention에 활용 가능)
    out["mask"]      → [B, L] bool    (padding mask)

Config 예시:
    # CLIP
    text_encoder:
      clip:
        target: motGPT.archs.clip_encoder.CLIPTextEncoder
        params:
          freeze: true

    # mBART (SOKE 방식)
    text_encoder:
      mbart:
        target: motGPT.archs.clip_encoder.MBartTextEncoder
        params:
          model_name: facebook/mbart-large-cc25
          proj_dim: 512
          freeze: true
          pooling: mean          # mean | cls | eos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# =============================================================================
# 1. CLIP Text Encoder  (Light-T2M 기본 방식)
# =============================================================================

class CLIPTextEncoder(nn.Module):
    """
    CLIP ViT-B/32 텍스트 인코더.

    Light-T2M의 text_encoder.py CLIP 클래스를 SignGPT3 인터페이스에 맞게 포팅.
    영어 전용이므로 ASL(How2Sign) 단독 학습 또는 빠른 프로토타이핑에 적합.

    출력:
        text_emb: [B, 512]  – CLIP text embedding (L2 정규화됨)
        hidden:   [B, L, 512]  – token-level hidden states
        mask:     [B, L] bool  – attention mask (True=유효, False=pad)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError("transformers 패키지가 필요합니다: pip install transformers")

        self.tokenizer   = CLIPTokenizer.from_pretrained(model_name)
        self.text_model  = CLIPTextModel.from_pretrained(model_name)
        self.output_dim  = self.text_model.config.hidden_size  # 512

        if freeze:
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad = False

        print(f"[CLIPTextEncoder] {model_name}, dim={self.output_dim}, freeze={freeze}")

    @property
    def device(self):
        return next(self.text_model.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        src_list: Optional[List[str]] = None,   # CLIP은 src 무시 (인터페이스 통일용)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts:    List[str]  입력 텍스트 (빈 문자열 "" → null embedding for CFG)
            device:   출력 텐서를 올릴 device
            src_list: 사용 안 함 (MBartTextEncoder와 인터페이스 통일)

        Returns:
            dict with:
                "text_emb": [B, 512]       L2 정규화된 pooled embedding
                "hidden":   [B, L, 512]    token-level hidden states
                "mask":     [B, L] bool
        """
        if device is None:
            device = self.device

        # 빈 문자열 처리 (CFG null text)
        texts = [t if t else " " for t in texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids      = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # pooler_output: [EOS] 토큰 위치 (CLIP 공식 방식)
        text_emb = out.pooler_output                      # [B, 512]
        text_emb = F.normalize(text_emb, dim=-1)          # L2 정규화

        return {
            "text_emb": text_emb,                         # [B, 512]
            "hidden":   out.last_hidden_state,            # [B, L, 512]
            "mask":     attention_mask.bool(),            # [B, L]
        }


# =============================================================================
# 2. mBART Text Encoder  (SOKE 방식 - 멀티링구얼)
# =============================================================================

# SOKE 언어 토큰 매핑
#   src 키는 batch['src']와 동일한 문자열 사용
MBART_LANG_MAP = {
    "how2sign": "en_XX",   # American Sign Language
    "csl":      "zh_CN",   # Chinese Sign Language
    "phoenix":  "de_DE",   # German Sign Language (Phoenix-2014T)
    # fallback
    "en":       "en_XX",
    "zh":       "zh_CN",
    "de":       "de_DE",
}

class MBartTextEncoder(nn.Module):
    """
    mBART-large-cc25 인코더 기반 멀티링구얼 텍스트 인코더.

    SOKE 방식 핵심:
    ┌──────────────────────────────────────────────────────────────┐
    │  입력 형식: [토큰들...] [eos] [lang_token]                    │
    │                                                              │
    │  src='how2sign' → 끝에 en_XX 토큰 삽입                       │
    │  src='csl'      → 끝에 zh_CN 토큰 삽입                       │
    │  src='phoenix'  → 끝에 de_DE 토큰 삽입                       │
    │                                                              │
    │  mBART encoder는 이 lang_token 정보로 해당 언어의             │
    │  문장 표현을 생성함 (multilingual pretraining 활용)            │
    └──────────────────────────────────────────────────────────────┘

    출력: last_hidden_state → pooling → Linear(768→512)

    pooling 전략:
      "mean": padding 제외한 토큰들의 평균 (기본값, 가장 안정적)
      "cls":  첫 번째 토큰 (mBART는 BOS=lang_token이 앞에 옴)
      "eos":  eos 토큰 위치
    """

    def __init__(
        self,
        model_name: str = "facebook/mbart-large-cc25",
        proj_dim: int = 512,           # 최종 출력 차원 (CLIP과 통일)
        freeze: bool = True,           # encoder frozen 여부
        pooling: str = "mean",         # "mean" | "cls" | "eos"
        max_length: int = 64,
    ):
        super().__init__()
        try:
            from transformers import MBartForConditionalGeneration, MBartTokenizer
        except ImportError:
            raise ImportError("transformers 패키지가 필요합니다: pip install transformers")

        self.pooling    = pooling
        self.max_length = max_length
        self.hidden_dim = 1024   # mBART-large hidden dim

        # Tokenizer
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)

        # Encoder만 추출 (decoder 불필요 → 메모리/속도 절약)
        full_model       = MBartForConditionalGeneration.from_pretrained(model_name)
        self.mbart_enc   = full_model.get_encoder()
        del full_model   # decoder GC

        # Projection: 1024 → proj_dim (CLIP 512와 통일)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.output_dim = proj_dim

        # 언어 토큰 ID 캐시
        self._lang_token_ids: Dict[str, int] = {}

        if freeze:
            self.mbart_enc.eval()
            for p in self.mbart_enc.parameters():
                p.requires_grad = False

        total_enc = sum(p.numel() for p in self.mbart_enc.parameters())
        total_proj = sum(p.numel() for p in self.proj.parameters())
        print(f"[MBartTextEncoder] {model_name}")
        print(f"  encoder: {total_enc/1e6:.1f}M params (freeze={freeze})")
        print(f"  proj:    {total_proj/1e3:.1f}K params (trainable)")
        print(f"  pooling: {pooling}, output_dim: {proj_dim}")

    @property
    def device(self):
        return next(self.mbart_enc.parameters()).device

    def _get_lang_token_id(self, lang_code: str) -> int:
        """언어 토큰 ID 캐시 조회 (최초 1회만 변환)"""
        if lang_code not in self._lang_token_ids:
            self._lang_token_ids[lang_code] = self.tokenizer.convert_tokens_to_ids(lang_code)
        return self._lang_token_ids[lang_code]

    def _resolve_src(self, src: Optional[str]) -> str:
        """src 문자열 → mBART 언어 코드 변환"""
        if src is None:
            return "en_XX"
        return MBART_LANG_MAP.get(src, "en_XX")

    def _tokenize_with_lang(
        self,
        texts: List[str],
        src_list: List[str],
        device: torch.device,
    ):
        """
        SOKE 방식 토크나이징:
          각 텍스트에 해당 언어 토큰을 끝에 삽입
          형식: [토큰들...] [eos] [lang_token]

        mBART 표준 포맷:
          encoder input: X [eos, src_lang_code]
          → tokenizer의 src_lang 설정으로 자동 처리하거나
            수동으로 마지막 유효 위치에 lang_token 삽입
        """
        # 언어 코드 목록 (배치 내 다를 수 있음)
        lang_codes = [self._resolve_src(s) for s in src_list]

        # 빈 문자열 처리 (CFG null text → 기본 공백)
        texts_clean = [t if t else " " for t in texts]

        # Batch tokenize (padding, truncation)
        enc = self.tokenizer(
            texts_clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids      = enc.input_ids.to(device)       # [B, L]
        attention_mask = enc.attention_mask.to(device)  # [B, L]

        # SOKE 방식: 각 시퀀스의 마지막 유효 토큰 위치에 lang_token 삽입
        # mBART encoder input: X [eos] [lang_token]
        # tokenizer가 이미 eos를 붙이므로 → eos 뒤에 lang_token 삽입
        for i, lang_code in enumerate(lang_codes):
            lang_id  = self._get_lang_token_id(lang_code)
            seq_len  = int(attention_mask[i].sum().item())

            if seq_len < input_ids.shape[1]:
                # padding 영역에 lang_token 삽입 후 mask 확장
                input_ids[i, seq_len] = lang_id
                attention_mask[i, seq_len] = 1
            else:
                # 시퀀스가 꽉 찬 경우: 마지막 토큰을 lang_token으로 교체
                # (truncation으로 eos가 없을 수 있음 → 끝에 강제 삽입)
                input_ids[i, -1] = lang_id

        return input_ids, attention_mask

    def _pool(
        self,
        hidden: torch.Tensor,   # [B, L, 1024]
        mask: torch.Tensor,     # [B, L] bool
    ) -> torch.Tensor:          # [B, 1024]
        """pooling 전략에 따라 시퀀스를 단일 벡터로 압축"""
        if self.pooling == "mean":
            # padding 제외 평균
            mask_f = mask.float().unsqueeze(-1)             # [B, L, 1]
            summed = (hidden * mask_f).sum(dim=1)           # [B, 1024]
            count  = mask_f.sum(dim=1).clamp(min=1e-9)     # [B, 1]
            return summed / count

        elif self.pooling == "cls":
            # 첫 번째 토큰 (mBART BOS = lang_token 또는 eos)
            return hidden[:, 0, :]

        elif self.pooling == "eos":
            # 마지막 유효 토큰 위치 (lang_token 삽입 후)
            lengths = mask.sum(dim=1).long() - 1            # [B]
            idx     = lengths.clamp(min=0)
            return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

        else:
            raise ValueError(f"pooling='{self.pooling}' not supported. Choose: mean | cls | eos")

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        src_list: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts:    List[str]  입력 텍스트
            device:   출력 device
            src_list: List[str]  데이터셋 소스 ('how2sign' | 'csl' | 'phoenix')
                      None이면 전부 'how2sign'(en_XX)으로 처리

        Returns:
            dict with:
                "text_emb": [B, 512]        projected & normalized embedding
                "hidden":   [B, L, 1024]    encoder last_hidden_state
                "mask":     [B, L] bool     유효 토큰 마스크
        """
        if device is None:
            device = self.device

        B = len(texts)
        if src_list is None:
            src_list = ["how2sign"] * B

        # Tokenize with language token injection (SOKE 방식)
        input_ids, attention_mask = self._tokenize_with_lang(texts, src_list, device)

        # mBART Encoder forward
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.mbart_enc.parameters())
        ):
            enc_out = self.mbart_enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = enc_out.last_hidden_state   # [B, L, 1024]
        mask   = attention_mask.bool()       # [B, L]

        # Pooling: [B, 1024]
        pooled = self._pool(hidden, mask)

        # Projection: [B, 1024] → [B, 512]
        text_emb = self.proj(pooled)
        text_emb = F.normalize(text_emb, dim=-1)

        return {
            "text_emb": text_emb,   # [B, 512]
            "hidden":   hidden,     # [B, L, 1024]
            "mask":     mask,       # [B, L]
        }


# =============================================================================
# 유틸: batch에서 src_list 자동 추출하여 MBartTextEncoder 호출
# =============================================================================

def encode_texts_from_batch(
    encoder: nn.Module,
    batch: dict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    batch dict에서 text와 src를 추출하여 인코더 호출.
    CLIPTextEncoder / MBartTextEncoder 모두 지원.

    사용 예:
        text_out = encode_texts_from_batch(self.text_encoder, batch, self.device)
        text_emb = text_out["text_emb"]  # [B, 512]
    """
    texts    = batch["text"]
    src_list = batch.get("src", None)
    return encoder(texts, device=device, src_list=src_list)
