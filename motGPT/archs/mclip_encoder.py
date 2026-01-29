"""
Multilingual CLIP Text Encoder for SignGPT3
============================================

sentence-transformers의 multilingual CLIP 모델 wrapper
영어/중국어/독일어 텍스트를 512-dim 벡터로 인코딩

Usage:
    encoder = MCLIPEncoder()
    text_emb = encoder(["Hello", "你好", "Hallo"])  # [3, 512]

복사 위치: motGPT/archs/mclip_encoder.py
"""

import torch
import torch.nn as nn
from typing import List, Union


class MCLIPEncoder(nn.Module):
    """
    Multilingual CLIP Text Encoder
    
    지원 언어: 50+ (영어, 중국어, 독일어 포함)
    Output: [B, 512]
    """
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/clip-ViT-B-32-multilingual-v1',
        freeze: bool = True,
        device: str = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = 512
        self._device = device
        
        # Lazy loading (첫 forward에서 로드)
        self._model = None
        self._freeze = freeze
        
    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name)
            
            if self._freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
            
            print(f"[MCLIPEncoder] Loaded: {self.model_name}")
            print(f"[MCLIPEncoder] Output dim: {self.output_dim}")
    
    @property
    def device(self):
        if self._device is not None:
            return torch.device(self._device)
        # 모델이 로드되었으면 모델의 device 반환
        if self._model is not None:
            return next(self._model.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to(self, device):
        """Override to() to handle lazy loading"""
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            text_emb: [B, 512] tensor
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # SentenceTransformer encode
        with torch.no_grad() if self._freeze else torch.enable_grad():
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        
        # Ensure correct device
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        return embeddings  # [B, 512]
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Alias for forward()"""
        return self.forward(texts)


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    encoder = MCLIPEncoder()
    
    # Test multilingual
    texts = [
        "A person is signing hello",           # English
        "一个人在打手语",                        # Chinese
        "Eine Person gebärdet Hallo",          # German
    ]
    
    emb = encoder(texts)
    print(f"Input: {len(texts)} texts")
    print(f"Output: {emb.shape}")  # [3, 512]
    print(f"Device: {emb.device}")
