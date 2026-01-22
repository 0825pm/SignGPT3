"""
SOKE 133-dim Feature Indices - Common Definitions

사용법:
  이 파일을 SignGPT3/motGPT/utils/soke_indices.py 로 복사하고
  다른 파일에서 import해서 사용하세요.

  예시:
    from motGPT.utils.soke_indices import SOKE_PART_INDICES, split_soke_features

SOKE 133-dim Feature Structure:
===============================
179-dim → 133-dim 변환 과정:
  1. clip_poses[:, (3+3*11):]  # 첫 36 제거 (root + lower body 11 joints)
  2. concat([..., :-20], [..., -10:])  # shape 제거, expr 유지

결과 133-dim:
  [0:30]    upper_body (10 joints × 3)
  [30:75]   lhand (15 joints × 3)
  [75:120]  rhand (15 joints × 3)
  [120:123] jaw (1 joint × 3)
  [123:133] expr (10 dims)
  
Total: 30 + 45 + 45 + 3 + 10 = 133 ✓
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union

# =============================================================================
# SOKE 133-dim Feature Part Indices
# =============================================================================

SOKE_PART_INDICES: Dict[str, Tuple[int, int]] = {
    'upper_body': (0, 30),     # 30 dims = 10 upper body joints × 3
    'lhand':      (30, 75),    # 45 dims = 15 left hand joints × 3
    'rhand':      (75, 120),   # 45 dims = 15 right hand joints × 3
    # 'jaw':        (120, 123),  # 3 dims = 1 jaw joint × 3
    # 'expr':       (123, 133),  # 10 dims = facial expression
}

SOKE_PART_DIMS: Dict[str, int] = {
    'upper_body': 30,
    'lhand': 45,
    'rhand': 45,
    # 'jaw': 3,
    # 'expr': 10,
}

SOKE_TOTAL_DIM = 120

# Aliases for compatibility
SOKE_BODY_IDX = SOKE_PART_INDICES['upper_body']
SOKE_LHAND_IDX = SOKE_PART_INDICES['lhand']
SOKE_RHAND_IDX = SOKE_PART_INDICES['rhand']
SOKE_JAW_IDX = SOKE_PART_INDICES['jaw']
SOKE_EXPR_IDX = SOKE_PART_INDICES['expr']

# Combined indices
SOKE_HAND_IDX = (30, 120)  # Both hands: 90 dims


# =============================================================================
# SMPL-X Joint Indices (55 joints)
# =============================================================================

SMPLX_JOINT_INDICES: Dict[str, list] = {
    'body': list(range(0, 22)),        # 22 body joints (pelvis to wrists)
    'lhand': list(range(25, 40)),      # 15 left hand joints
    'rhand': list(range(40, 55)),      # 15 right hand joints
    # Note: joints 22-24 are jaw, left_eye, right_eye
}

SMPLX_TOTAL_JOINTS = 55


# =============================================================================
# Helper Functions
# =============================================================================

def get_part_indices(part_name: str) -> Tuple[int, int]:
    """
    Get start and end indices for a body part.
    
    Args:
        part_name: 'upper_body', 'lhand', 'rhand', 'jaw', 'expr', 'hand', 'body'
        
    Returns:
        (start, end) tuple
    """
    if part_name == 'hand':
        return SOKE_HAND_IDX
    elif part_name == 'body':
        return SOKE_BODY_IDX
    else:
        return SOKE_PART_INDICES[part_name]


def get_part_features(features: Union[torch.Tensor, np.ndarray], 
                      part_name: str) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract specific body part features from 133-dim SOKE features.
    
    Args:
        features: [..., 133] tensor or array
        part_name: 'upper_body', 'lhand', 'rhand', 'jaw', 'expr', 'hand', 'body'
    
    Returns:
        Part features [..., part_dim]
    """
    start, end = get_part_indices(part_name)
    return features[..., start:end]


def split_soke_features(features: Union[torch.Tensor, np.ndarray]
                        ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Split 133-dim features into all parts.
    
    Args:
        features: [..., 133] tensor or array
    
    Returns:
        dict with keys: 'upper_body', 'lhand', 'rhand', 'jaw', 'expr'
    """
    return {
        part: features[..., start:end]
        for part, (start, end) in SOKE_PART_INDICES.items()
    }


def merge_soke_features(parts_dict: Dict[str, Union[torch.Tensor, np.ndarray]]
                        ) -> Union[torch.Tensor, np.ndarray]:
    """
    Merge body parts back into 133-dim features.
    
    Args:
        parts_dict: dict with 'upper_body', 'lhand', 'rhand', 'jaw', 'expr'
    
    Returns:
        merged features [..., 133]
    """
    first_val = list(parts_dict.values())[0]
    if isinstance(first_val, torch.Tensor):
        cat_fn = torch.cat
    else:
        cat_fn = np.concatenate
    
    return cat_fn([
        parts_dict['upper_body'],
        parts_dict['lhand'],
        parts_dict['rhand'],
        parts_dict['jaw'],
        parts_dict['expr'],
    ], axis=-1)


def verify_soke_structure() -> bool:
    """Verify that SOKE indices are correctly defined."""
    total = sum(SOKE_PART_DIMS.values())
    
    # Check continuity
    expected_start = 0
    for part in ['upper_body', 'lhand', 'rhand', 'jaw', 'expr']:
        start, end = SOKE_PART_INDICES[part]
        if start != expected_start:
            print(f"ERROR: {part} starts at {start}, expected {expected_start}")
            return False
        expected_start = end
    
    if total != SOKE_TOTAL_DIM:
        print(f"ERROR: Total {total} != {SOKE_TOTAL_DIM}")
        return False
    
    return True


# =============================================================================
# Validation on import
# =============================================================================

assert verify_soke_structure(), "SOKE indices validation failed!"


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("SOKE 133-dim Feature Indices")
    print("=" * 50)
    
    total = 0
    for part, (start, end) in SOKE_PART_INDICES.items():
        dim = end - start
        print(f"  {part:12s}: [{start:3d}:{end:3d}] = {dim:2d} dims")
        total += dim
    
    print("=" * 50)
    print(f"  Total: {total} dims")
    print(f"  Valid: {verify_soke_structure()}")
    
    # Test split/merge
    print("\n" + "=" * 50)
    print("Testing split/merge with torch tensor...")
    
    test_features = torch.randn(4, 50, 133)
    parts = split_soke_features(test_features)
    
    print("Split results:")
    for part, feat in parts.items():
        print(f"  {part:12s}: {list(feat.shape)}")
    
    merged = merge_soke_features(parts)
    print(f"\nMerged: {list(merged.shape)}")
    print(f"Match: {torch.allclose(test_features, merged)}")
    
    # Test with numpy
    print("\n" + "=" * 50)
    print("Testing split/merge with numpy array...")
    
    test_np = np.random.randn(4, 50, 133)
    parts_np = split_soke_features(test_np)
    merged_np = merge_soke_features(parts_np)
    print(f"Merged: {merged_np.shape}")
    print(f"Match: {np.allclose(test_np, merged_np)}")
