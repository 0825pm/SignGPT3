"""
SOKE-style Collate Functions for Sign Language Datasets

[FIXED] T2M task의 output 템플릿에 <Motion_Placeholder> 추가
"""

import torch
import numpy as np
from typing import List, Tuple, Any


def sign_collate(batch: List[Tuple]) -> dict:
    """
    Collate function for sign language datasets.
    
    [FIXED] T2M task에서 output에 <Motion_Placeholder> 추가
    """
    # Filter out None items
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    # Unpack batch
    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    srcs = [b[9] if len(b) > 9 else 'how2sign' for b in batch]
    
    # Get batch size and max length
    batch_size = len(batch)
    max_len = max(lengths)
    feat_dim = motions[0].shape[-1]
    
    # Pad motions
    motion_padded = torch.zeros(batch_size, max_len, feat_dim)
    for i, (motion, length) in enumerate(zip(motions, lengths)):
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion).float()
        motion_padded[i, :length] = motion[:length]
    
    # =========================================================================
    # [FIXED] Default tasks for LM training (t2m task)
    # output에 <Motion_Placeholder>를 추가해야 motion 토큰이 생성됨!
    # =========================================================================
    tasks = [{
        'input': ['Generate motion: <Caption_Placeholder>'],
        'output': ['<Motion_Placeholder>'],  # [FIXED] 빈 문자열 → Motion_Placeholder
        'class': 't2m'
    }] * batch_size
    # =========================================================================
    
    # Build output dict
    output = {
        'motion': motion_padded,
        'length': lengths,
        'text': texts,
        'name': names,
        'fname': names,
        'src': srcs,
        'tasks': tasks,
        'all_captions': [[t] for t in texts],
    }
    
    # Handle optional fields (m_tokens, word_emb, etc.)
    if len(batch[0]) > 4:
        m_tokens = [b[4] for b in batch]
        if m_tokens[0] is not None:
            m_tokens_lens = [b[5] for b in batch]
            max_token_len = max(m_tokens_lens) if m_tokens_lens[0] is not None else 0
            
            if max_token_len > 0:
                tokens_padded = torch.zeros(batch_size, max_token_len, dtype=torch.long)
                for i, (tokens, t_len) in enumerate(zip(m_tokens, m_tokens_lens)):
                    if tokens is not None:
                        tokens_padded[i, :t_len] = tokens[:t_len]
                output['m_tokens'] = tokens_padded
                output['m_tokens_len'] = m_tokens_lens
    
    # Word embeddings
    if len(batch[0]) > 6:
        word_embs = [b[6] for b in batch]
        if word_embs[0] is not None:
            word_emb_padded = torch.stack(word_embs)
            output['word_embs'] = word_emb_padded
    
    # POS one-hot
    if len(batch[0]) > 7:
        pos_ohots = [b[7] for b in batch]
        if pos_ohots[0] is not None:
            pos_ohot_padded = torch.stack(pos_ohots)
            output['pos_ohot'] = pos_ohot_padded
    
    # Text lengths
    if len(batch[0]) > 8:
        text_lens = [b[8] for b in batch]
        if text_lens[0] is not None:
            output['text_len'] = torch.tensor(text_lens)
    
    return output


def sign_collate_simple(batch: List[Tuple]) -> dict:
    """
    Simplified collate function for VAE training.
    """
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    srcs = [b[-1] if len(b) > 4 else 'how2sign' for b in batch]
    
    batch_size = len(batch)
    max_len = max(lengths)
    feat_dim = motions[0].shape[-1] if motions[0] is not None else 133
    
    motion_padded = torch.zeros(batch_size, max_len, feat_dim)
    for i, (motion, length) in enumerate(zip(motions, lengths)):
        if motion is not None:
            if isinstance(motion, np.ndarray):
                motion = torch.from_numpy(motion).float()
            motion_padded[i, :length] = motion[:length]
    
    return {
        'motion': motion_padded,
        'length': lengths,
        'text': texts,
        'name': names,
        'fname': names,
        'src': srcs,
    }