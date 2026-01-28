"""
SOKE-style Collate Functions for Sign Language Datasets

[FIXED] T2M task의 output 템플릿에 <Motion_Placeholder> 추가
[FIXED] pos_ohot, word_emb 타입 체크 추가 (list vs tensor)
"""

import torch
import numpy as np
from typing import List, Tuple, Any


def sign_collate(batch: List[Tuple]) -> dict:
    """
    Collate function for sign language datasets.
    
    Expected batch item format (from SignText2MotionDataset):
        b[0] = text (str)
        b[1] = motion (tensor)
        b[2] = length (int)
        b[3] = name (str)
        b[4] = m_tokens (tensor or None)
        b[5] = m_tokens_len (int or None)
        b[6] = word_emb (tensor or None)
        b[7] = all_captions (list) or pos_ohot (tensor)
        b[8] = text_len (int or None)
        b[9] = src (str)
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
    # Default tasks for LM training (t2m task)
    # =========================================================================
    tasks = [{
        'input': ['Generate motion: <Caption_Placeholder>'],
        'output': ['<Motion_Placeholder>'],
        'class': 't2m'
    }] * batch_size
    
    # =========================================================================
    # Build output dict
    # =========================================================================
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
    
    # =========================================================================
    # Handle optional fields (m_tokens, word_emb, etc.)
    # [FIXED] 타입 체크 추가 - tensor만 stack
    # =========================================================================
    
    # m_tokens
    if len(batch[0]) > 4:
        m_tokens = [b[4] for b in batch]
        if m_tokens[0] is not None and isinstance(m_tokens[0], (torch.Tensor, np.ndarray)):
            m_tokens_lens = [b[5] for b in batch]
            max_token_len = max(m_tokens_lens) if m_tokens_lens[0] is not None else 0
            
            if max_token_len > 0:
                tokens_padded = torch.zeros(batch_size, max_token_len, dtype=torch.long)
                for i, (tokens, t_len) in enumerate(zip(m_tokens, m_tokens_lens)):
                    if tokens is not None:
                        if isinstance(tokens, np.ndarray):
                            tokens = torch.from_numpy(tokens)
                        tokens_padded[i, :t_len] = tokens[:t_len]
                output['m_tokens'] = tokens_padded
                output['m_tokens_len'] = m_tokens_lens
    
    # Word embeddings
    if len(batch[0]) > 6:
        word_embs = [b[6] for b in batch]
        # [FIXED] 타입 체크 - tensor만 stack
        if word_embs[0] is not None and isinstance(word_embs[0], (torch.Tensor, np.ndarray)):
            if isinstance(word_embs[0], np.ndarray):
                word_embs = [torch.from_numpy(w) for w in word_embs]
            word_emb_padded = torch.stack(word_embs)
            output['word_embs'] = word_emb_padded
    
    # POS one-hot (or all_captions - b[7] can be either!)
    if len(batch[0]) > 7:
        item_7 = [b[7] for b in batch]
        # [FIXED] 타입 체크 - tensor만 stack, list는 all_captions로 처리
        if item_7[0] is not None:
            if isinstance(item_7[0], (torch.Tensor, np.ndarray)):
                # It's pos_ohot tensor
                if isinstance(item_7[0], np.ndarray):
                    item_7 = [torch.from_numpy(p) for p in item_7]
                pos_ohot_padded = torch.stack(item_7)
                output['pos_ohot'] = pos_ohot_padded
            elif isinstance(item_7[0], list):
                # It's all_captions - update the existing entry
                output['all_captions'] = item_7
    
    # Text lengths
    if len(batch[0]) > 8:
        text_lens = [b[8] for b in batch]
        if text_lens[0] is not None and isinstance(text_lens[0], (int, np.integer)):
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
    feat_dim = motions[0].shape[-1] if motions[0] is not None else 120
    
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