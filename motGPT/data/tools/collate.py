"""
SOKE-style collate functions for sign language data
"""
import torch
import numpy as np


def sign_collate(batch):
    """
    Collate function for sign language datasets
    Input batch: list of (text, motion, length, name, ...)
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None
    
    # Unpack
    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    srcs = [b[-1] for b in batch]  # Last element is src
    
    # Get max length
    max_len = max(lengths)
    
    # Pad motions
    nfeats = motions[0].shape[-1]
    padded_motions = torch.zeros(len(batch), max_len, nfeats)
    
    for i, (motion, length) in enumerate(zip(motions, lengths)):
        padded_motions[i, :length] = motion[:length]
    
    # Create batch dict
    batch_dict = {
        'text': texts,
        'motion': padded_motions,
        'length': lengths,
        'name': names,
        'src': srcs,
    }
    
    return batch_dict


def sign_collate_simple(batch):
    """
    Simple collate for VAE training (motion only)
    """
    batch = [b for b in batch if b is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None
    
    motions = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    srcs = [b[-1] for b in batch]
    
    max_len = max(lengths)
    nfeats = motions[0].shape[-1]
    
    padded_motions = torch.zeros(len(batch), max_len, nfeats)
    for i, (motion, length) in enumerate(zip(motions, lengths)):
        padded_motions[i, :length] = motion[:length]
    
    return {
        'motion': padded_motions,
        'length': lengths,
        'name': names,
        'src': srcs,
    }