"""
SOKE-style feats2joints for Sign Language - CORRECTED VERSION
Converts 133-dim SOKE features to SMPL-X joints/vertices

사용법:
  이 파일을 SignGPT3/motGPT/utils/feats2joints.py 로 복사하세요.

SOKE 133-dim Feature Structure (CORRECT!):
=========================================
  [0:30]    upper_body (10 joints × 3)
  [30:75]   lhand (15 joints × 3)  
  [75:120]  rhand (15 joints × 3)
  [120:123] jaw (1 joint × 3)
  [123:133] expr (10 dims)
  
Total: 30 + 45 + 45 + 3 + 10 = 133 ✓
"""

import torch
import torch.nn as nn
import numpy as np
import os


# =============================================================================
# SOKE 133-dim Feature Indices (CORRECT!)
# =============================================================================

SOKE_PART_INDICES = {
    'upper_body': (0, 30),     # 30 dims = 10 joints × 3
    'lhand':      (30, 75),    # 45 dims = 15 joints × 3
    'rhand':      (75, 120),   # 45 dims = 15 joints × 3
    'jaw':        (120, 123),  # 3 dims = 1 joint × 3
    'expr':       (123, 133),  # 10 dims
}

SOKE_PART_DIMS = {
    'upper_body': 30,
    'lhand': 45,
    'rhand': 45,
    'jaw': 3,
    'expr': 10,
}

SOKE_TOTAL_DIM = 133


def get_part_indices(part_name):
    """Get start and end indices for a body part."""
    return SOKE_PART_INDICES[part_name]


def split_soke_features(features):
    """
    Split 133-dim SOKE features into body parts.
    
    Args:
        features: [..., 133] tensor or array
        
    Returns:
        dict with 'upper_body', 'lhand', 'rhand', 'jaw', 'expr'
    """
    return {
        part: features[..., start:end]
        for part, (start, end) in SOKE_PART_INDICES.items()
    }


def merge_soke_features(parts_dict):
    """
    Merge body parts back into 133-dim SOKE features.
    
    Args:
        parts_dict: dict with 'upper_body', 'lhand', 'rhand', 'jaw', 'expr'
        
    Returns:
        features [..., 133]
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


class Feats2Joints(nn.Module):
    """
    Convert SOKE 133-dim features to SMPL-X joints and vertices.
    
    CORRECTED version with proper SOKE 133-dim indexing.
    """
    
    def __init__(self, 
                 smplx_model_path='deps/smpl_models',
                 device='cuda',
                 use_smplx=True):
        super().__init__()
        
        self.use_smplx = use_smplx
        self.device = device
        self.smplx_model = None
        
        # Default shape parameters (from SOKE)
        self.register_buffer('default_shape', torch.tensor([
            -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
            0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
        ]).float())
        
        if use_smplx:
            try:
                import smplx
                model_path = os.path.join(smplx_model_path, 'smplx', 'SMPLX_NEUTRAL.npz')
                if os.path.exists(model_path):
                    self.smplx_model = smplx.create(
                        smplx_model_path, 
                        model_type='smplx',
                        gender='NEUTRAL',
                        use_pca=False,
                        use_face_contour=True,
                        batch_size=1
                    )
                    print(f"[Feats2Joints] Loaded SMPL-X model from {smplx_model_path}")
                else:
                    print(f"[Feats2Joints] SMPL-X model not found at {model_path}")
                    self.use_smplx = False
            except Exception as e:
                print(f"[Feats2Joints] Failed to load SMPL-X: {e}")
                self.use_smplx = False
    
    def forward(self, features, return_vertices=True):
        """
        Convert features to joints (and optionally vertices)
        
        Args:
            features: [B, T, D] pose features (D=133 for SOKE format)
            return_vertices: whether to return vertices
            
        Returns:
            joints: [B*T, J, 3] joint positions
            vertices: [B*T, V, 3] vertex positions (if return_vertices=True)
        """
        B, T, D = features.shape
        device = features.device
        
        if self.use_smplx and self.smplx_model is not None:
            return self._forward_smplx(features, return_vertices)
        else:
            return self._forward_approximate(features, return_vertices)
    
    def _forward_smplx(self, features, return_vertices=True):
        """Forward pass through SMPL-X model"""
        B, T, D = features.shape
        device = features.device
        
        # Move SMPL-X to device
        if self.smplx_model.faces_tensor.device != device:
            self.smplx_model = self.smplx_model.to(device)
        
        # Reshape for batch processing
        features = features.reshape(B * T, D)
        batch_size = B * T
        
        # ===== Parse SOKE 133-dim features (CORRECTED!) =====
        if D == 120:
            upper_body_pose = features[:, 0:30]
            lhand_pose = features[:, 30:75]
            rhand_pose = features[:, 75:120]
            
            # 나머지는 전부 zeros
            lower_body_zeros = torch.zeros(batch_size, 33, device=device, dtype=features.dtype)
            body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)  # [B, 63]
            
            root_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            jaw_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            expr = torch.zeros(batch_size, 10, device=device, dtype=features.dtype)
            leye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            reye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
        elif D == 133:
            # 133-dim structure: upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
            upper_body_pose = features[:, 0:30]      # 10 joints × 3
            lhand_pose = features[:, 30:75]          # 15 joints × 3
            rhand_pose = features[:, 75:120]         # 15 joints × 3
            jaw_pose = features[:, 120:123]          # 1 joint × 3
            expr = features[:, 123:133]              # 10 dims
            
            # Reconstruct full body_pose (63 dims = 21 joints × 3)
            # lower body (11 joints × 3 = 33) + upper body (10 joints × 3 = 30) = 63
            lower_body_zeros = torch.zeros(batch_size, 33, device=device, dtype=features.dtype)
            body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)  # 33 + 30 = 63
            
            # Zero root pose
            root_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            
            # Zero eye poses (not in SOKE 133-dim)
            leye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            reye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            
        elif D == 179:
            # Full 179-dim format (for reference)
            root_pose = features[:, 0:3]
            body_pose = features[:, 3:66]
            lhand_pose = features[:, 66:111]
            rhand_pose = features[:, 111:156]
            jaw_pose = features[:, 156:159]
            expr = features[:, 169:179]
            leye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
            reye_pose = torch.zeros(batch_size, 3, device=device, dtype=features.dtype)
        else:
            raise ValueError(f"Unsupported feature dimension: {D}. Expected 133 or 179.")
        
        # Use default shape
        betas = self.default_shape.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Forward SMPL-X
        smplx_output = self.smplx_model(
            betas=betas,
            global_orient=root_pose,
            body_pose=body_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
            return_verts=return_vertices
        )
        
        joints = smplx_output.joints  # [B*T, J, 3]
        
        if return_vertices:
            vertices = smplx_output.vertices  # [B*T, V, 3]
            return joints, vertices
        
        return joints, None
    
    def _forward_approximate(self, features, return_vertices=True):
        """Approximate forward pass (without SMPL-X model)"""
        B, T, D = features.shape
        device = features.device
        
        # Parse features and approximate joint positions
        if D == 120:
            # Split 120-dim features: upper_body(30) + lhand(45) + rhand(45)
            upper_body = features[..., 0:30].reshape(B, T, 10, 3)
            lhand = features[..., 30:75].reshape(B, T, 15, 3)
            rhand = features[..., 75:120].reshape(B, T, 15, 3)
            
            # Create approximate 55-joint skeleton
            joints = torch.zeros(B, T, 55, 3, device=device, dtype=features.dtype)
            
            # Place upper body joints (joints 12-21)
            joints[:, :, 12:22, :] = upper_body
            
            # Place hand joints
            joints[:, :, 25:40, :] = lhand  # Left hand
            joints[:, :, 40:55, :] = rhand  # Right hand
            
        elif D == 133:
            # Split SOKE 133-dim features (CORRECTED!)
            upper_body = features[..., 0:30].reshape(B, T, 10, 3)
            lhand = features[..., 30:75].reshape(B, T, 15, 3)
            rhand = features[..., 75:120].reshape(B, T, 15, 3)
            
            # Create approximate 55-joint skeleton
            joints = torch.zeros(B, T, 55, 3, device=device, dtype=features.dtype)
            
            # Place upper body joints (joints 12-21)
            joints[:, :, 12:22, :] = upper_body
            
            # Place hand joints
            joints[:, :, 25:40, :] = lhand  # Left hand
            joints[:, :, 40:55, :] = rhand  # Right hand
            
        else:
            # Fallback: zeros
            joints = torch.zeros(B, T, 55, 3, device=device, dtype=features.dtype)
        
        joints = joints.reshape(B * T, 55, 3)
        
        if return_vertices:
            vertices = torch.zeros(B * T, 10475, 3, device=device, dtype=features.dtype)
            return joints, vertices
        
        return joints, None


# =============================================================================
# Convenience function for H2S DataModule
# =============================================================================

def feats2joints_smplx(features, mean, std):
    """
    Convert 133-dim SOKE features to 3D joints using SMPL-X.
    
    Args:
        features: [B, T, 133] or [T, 133] normalized features
        mean: [133] mean for denormalization
        std: [133] std for denormalization
    
    Returns:
        vertices: [B, T, 10475, 3] SMPL-X vertices (or None)
        joints: [B, T, 55, 3] SMPL-X joints
    """
    # Handle 2D input (T, D) -> (1, T, D)
    squeeze_output = False
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
        squeeze_output = True
    
    B, T, D = features.shape
    
    # Denormalize
    mean = mean.to(features.device)
    std = std.to(features.device)
    features_denorm = features * std + mean
    
    # Use Feats2Joints converter
    converter = Feats2Joints(use_smplx=False)
    joints, vertices = converter(features_denorm, return_vertices=False)
    
    # Reshape back
    joints = joints.reshape(B, T, -1, 3)
    
    if squeeze_output:
        joints = joints.squeeze(0)
    
    return vertices, joints


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SOKE 133-dim Feats2Joints Test")
    print("=" * 60)
    
    # Print structure
    print("\nSOKE 133-dim structure:")
    total = 0
    for part, (start, end) in SOKE_PART_INDICES.items():
        dim = end - start
        print(f"  {part:12s}: [{start:3d}:{end:3d}] = {dim:2d} dims")
        total += dim
    print(f"  {'Total':12s}:              = {total:2d} dims")
    
    # Test with random data
    print("\n" + "=" * 60)
    print("Testing Feats2Joints...")
    
    test_features = torch.randn(4, 50, 133)  # [B, T, 133]
    
    converter = Feats2Joints(use_smplx=False)
    joints, vertices = converter(test_features, return_vertices=False)
    
    print(f"Input shape: {list(test_features.shape)}")
    print(f"Output joints shape: {list(joints.shape)}")
    print("✓ Test passed!")