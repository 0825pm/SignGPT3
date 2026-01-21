"""
Metric Utilities for Motion Evaluation

Provides functions for computing:
- MPJPE (Mean Per Joint Position Error)
- PA-MPJPE (Procrustes Aligned MPJPE)
- Acceleration Error
"""

import torch
import numpy as np


def compute_mpjpe(preds, target, valid_mask=None, sample_wise=True):
    """
    Compute Mean Per-Joint Position Error.
    
    Args:
        preds: (B, J, 3) or (T, J, 3) predicted joint positions
        target: (B, J, 3) or (T, J, 3) ground truth joint positions
        valid_mask: Optional mask for valid joints
        sample_wise: If True, return per-sample mean; else return all values
    
    Returns:
        mpjpe: Per-sample or per-joint MPJPE values
    """
    assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"
    
    # Euclidean distance per joint
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # (B, J)
    
    if valid_mask is not None:
        if sample_wise:
            mpjpe_seq = (mpjpe * valid_mask.float()).sum(-1) / valid_mask.float().sum(-1).clamp(min=1e-6)
        else:
            mpjpe_seq = mpjpe[valid_mask]
    else:
        if sample_wise:
            mpjpe_seq = mpjpe.mean(-1)  # (B,)
        else:
            mpjpe_seq = mpjpe
    
    return mpjpe_seq


def align_by_parts(joints, align_inds=None):
    """
    Align joints by subtracting the mean of specified joints.
    
    Args:
        joints: (B, J, 3) joint positions
        align_inds: List of joint indices to use for alignment (e.g., [0] for pelvis)
    
    Returns:
        aligned_joints: (B, J, 3) aligned joint positions
    """
    if align_inds is None:
        return joints
    
    pelvis = joints[:, align_inds].mean(1, keepdim=True)  # (B, 1, 3)
    return joints - pelvis


def calc_mpjpe(preds, target, align_inds=None, sample_wise=True):
    """
    Calculate MPJPE with optional root alignment.
    
    Args:
        preds: (B, J, 3) predicted joints
        target: (B, J, 3) target joints
        align_inds: Joint indices for alignment (default: [0] for pelvis)
        sample_wise: Return per-sample mean
    
    Returns:
        mpjpe: MPJPE values
    """
    # Create valid mask (exclude invalid joints marked with -2.0)
    valid_mask = target[:, :, 0] != -2.0
    
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    
    mpjpe_each = compute_mpjpe(
        preds_aligned, target_aligned, 
        valid_mask=valid_mask, 
        sample_wise=sample_wise
    )
    
    return mpjpe_each


def calc_accel(preds, target):
    """
    Calculate acceleration error.
    
    Args:
        preds: (T, J, 3) predicted joints sequence
        target: (T, J, 3) target joints sequence
    
    Returns:
        accel_error: Acceleration error per frame
    """
    assert preds.shape == target.shape
    assert preds.dim() == 3
    
    # Compute acceleration: joints[t-1] - 2*joints[t] + joints[t+1]
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    
    # Compute error
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)  # (T-2, J)
    accel_seq = normed.mean(1)  # (T-2,)
    
    return accel_seq


def calc_pampjpe(preds, target, sample_wise=True):
    """
    Calculate Procrustes-Aligned MPJPE (PA-MPJPE).
    
    Aligns predictions to target using Procrustes analysis
    before computing MPJPE.
    
    Args:
        preds: (B, J, 3) predicted joints
        target: (B, J, 3) target joints
        sample_wise: Return per-sample mean
    
    Returns:
        pampjpe: PA-MPJPE values
    """
    preds = preds.float()
    target = target.float()
    
    # Handle each sample in batch
    B = preds.shape[0]
    pampjpe_list = []
    
    for i in range(B):
        pred_i = preds[i]  # (J, 3)
        target_i = target[i]  # (J, 3)
        
        # Procrustes alignment
        pred_aligned = procrustes_align(pred_i, target_i)
        
        # Compute MPJPE
        error = torch.norm(pred_aligned - target_i, dim=-1)  # (J,)
        
        if sample_wise:
            pampjpe_list.append(error.mean())
        else:
            pampjpe_list.append(error)
    
    if sample_wise:
        return torch.stack(pampjpe_list)
    else:
        return torch.cat(pampjpe_list)


def procrustes_align(source, target):
    """
    Align source to target using Procrustes analysis (single sample).
    
    Args:
        source: (J, 3) source points
        target: (J, 3) target points
    
    Returns:
        aligned: (J, 3) aligned source points
    """
    # Center the points
    mu_source = source.mean(0, keepdim=True)
    mu_target = target.mean(0, keepdim=True)
    
    source_centered = source - mu_source
    target_centered = target - mu_target
    
    # Compute optimal rotation using SVD
    H = source_centered.T @ target_centered
    U, S, Vt = torch.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    # Handle reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    var_source = (source_centered ** 2).sum()
    scale = S.sum() / var_source.clamp(min=1e-8)
    
    # Apply transformation
    aligned = scale * (source_centered @ R.T) + mu_target
    
    return aligned


def batch_procrustes_align(source, target):
    """
    Align source to target using Procrustes analysis (batched).
    
    Args:
        source: (B, J, 3) source points
        target: (B, J, 3) target points
    
    Returns:
        aligned: (B, J, 3) aligned source points
    """
    B, J, _ = source.shape
    
    # Center the points
    mu_source = source.mean(1, keepdim=True)  # (B, 1, 3)
    mu_target = target.mean(1, keepdim=True)  # (B, 1, 3)
    
    source_centered = source - mu_source  # (B, J, 3)
    target_centered = target - mu_target  # (B, J, 3)
    
    # Compute covariance matrix: (B, 3, 3)
    H = source_centered.transpose(1, 2) @ target_centered
    
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation
    R = Vt.transpose(1, 2) @ U.transpose(1, 2)  # (B, 3, 3)
    
    # Handle reflection
    det = torch.det(R)
    sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
    Vt_corrected = Vt.clone()
    Vt_corrected[:, -1:, :] *= sign
    R = Vt_corrected.transpose(1, 2) @ U.transpose(1, 2)
    
    # Compute scale
    var_source = (source_centered ** 2).sum(dim=(1, 2))  # (B,)
    scale = S.sum(dim=1) / var_source.clamp(min=1e-8)  # (B,)
    
    # Apply transformation
    aligned = scale.unsqueeze(-1).unsqueeze(-1) * (source_centered @ R.transpose(1, 2)) + mu_target
    
    return aligned