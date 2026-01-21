"""
SOKE-style feats2joints for Sign Language
Converts 133-dim or 179-dim features to SMPL-X joints/vertices
"""
import torch
import torch.nn as nn
import numpy as np
import os


class Feats2Joints(nn.Module):
    """
    Convert SMPL-X pose features to 3D joints and vertices
    
    Input: features [B, T, D] where D is 133 (SOKE) or 179 (full)
    Output: joints [B, T, J, 3], vertices [B, T, V, 3]
    
    Feature format (133-dim SOKE):
    - jaw_pose: 0:3 (3)
    - leye_pose: 3:6 (3)  
    - reye_pose: 6:9 (3)
    - body_pose: 9:72 (63 = 21 joints * 3)
    - left_hand_pose: 72:117 (45 = 15 joints * 3)
    - right_hand_pose: 117:162 (45 = 15 joints * 3) -> only 117:133 in 133-dim
    - expression: 162:172 (10) -> not present in 133-dim
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
                    print(f"[Feats2Joints] SMPL-X model not found at {model_path}, using approximation")
                    self.use_smplx = False
            except Exception as e:
                print(f"[Feats2Joints] Failed to load SMPL-X: {e}, using approximation")
                self.use_smplx = False
    
    def forward(self, features, return_vertices=True):
        """
        Convert features to joints (and optionally vertices)
        
        Args:
            features: [B, T, D] pose features
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
        
        # Parse features based on dimension
        if D == 133:
            # SOKE format: jaw(3) + leye(3) + reye(3) + body(63) + lhand(45) + rhand(16)
            jaw_pose = features[:, 0:3]
            leye_pose = features[:, 3:6]
            reye_pose = features[:, 6:9]
            body_pose = features[:, 9:72]
            lhand_pose = features[:, 72:117]
            rhand_pose_partial = features[:, 117:133]
            # Pad right hand to 45 dims
            rhand_pose = torch.cat([
                rhand_pose_partial,
                torch.zeros(B * T, 45 - 16, device=device)
            ], dim=-1)
            expression = torch.zeros(B * T, 10, device=device)
            global_orient = torch.zeros(B * T, 3, device=device)
            transl = torch.zeros(B * T, 3, device=device)
        elif D == 179:
            # Full format with global orient and betas
            global_orient = features[:, 0:3]
            body_pose = features[:, 3:66]
            lhand_pose = features[:, 66:111]
            rhand_pose = features[:, 111:156]
            jaw_pose = features[:, 156:159]
            expression = features[:, 159:169]
            # betas = features[:, 169:179]  # Not used, use default
            leye_pose = torch.zeros(B * T, 3, device=device)
            reye_pose = torch.zeros(B * T, 3, device=device)
            transl = torch.zeros(B * T, 3, device=device)
        else:
            raise ValueError(f"Unsupported feature dimension: {D}")
        
        # Get shape parameters
        betas = self.default_shape.unsqueeze(0).expand(B * T, -1).to(device)
        
        # Forward SMPL-X
        outputs = []
        vertices_list = []
        
        batch_size = 64  # Process in chunks to avoid OOM
        for i in range(0, B * T, batch_size):
            end_idx = min(i + batch_size, B * T)
            
            output = self.smplx_model(
                global_orient=global_orient[i:end_idx],
                body_pose=body_pose[i:end_idx],
                left_hand_pose=lhand_pose[i:end_idx],
                right_hand_pose=rhand_pose[i:end_idx],
                jaw_pose=jaw_pose[i:end_idx],
                leye_pose=leye_pose[i:end_idx],
                reye_pose=reye_pose[i:end_idx],
                expression=expression[i:end_idx],
                betas=betas[i:end_idx],
                transl=transl[i:end_idx]
            )
            
            outputs.append(output.joints)
            if return_vertices:
                vertices_list.append(output.vertices)
        
        joints = torch.cat(outputs, dim=0)  # [B*T, J, 3]
        
        if return_vertices:
            vertices = torch.cat(vertices_list, dim=0)  # [B*T, V, 3]
            return joints, vertices
        
        return joints, None
    
    def _forward_approximate(self, features, return_vertices=True):
        """
        Approximate joints from pose features without SMPL-X
        Simply reshape pose features as pseudo-joints
        """
        B, T, D = features.shape
        device = features.device
        
        # Create approximate joints from pose features
        # This is a placeholder - real joints need SMPL-X
        if D == 133:
            # jaw(1) + eyes(2) + body(21) + hands(30) = 54 joints approx
            njoints = 55
        else:
            njoints = 55
        
        # Reshape features into pseudo-joints
        joints = torch.zeros(B * T, njoints, 3, device=device)
        
        # Fill with reshaped pose data
        # Body: indices 9:72 -> 21 joints
        if D >= 72:
            body_poses = features.reshape(B * T, D)[:, 9:72].reshape(B * T, 21, 3)
            joints[:, :21] = body_poses
        
        # Hands: 72:133 or 72:162
        if D >= 133:
            hand_poses = features.reshape(B * T, D)[:, 72:min(D, 132)].reshape(B * T, -1, 3)
            n_hand_joints = hand_poses.shape[1]
            joints[:, 21:21+n_hand_joints] = hand_poses
        
        # Vertices placeholder
        if return_vertices:
            nvertices = 10475
            vertices = torch.zeros(B * T, nvertices, 3, device=device)
            return joints, vertices
        
        return joints, None


def create_feats2joints(cfg=None, device='cuda'):
    """Factory function to create Feats2Joints"""
    smplx_path = 'deps/smpl_models'
    if cfg is not None and hasattr(cfg, 'SMPLX_PATH'):
        smplx_path = cfg.SMPLX_PATH
    
    return Feats2Joints(
        smplx_model_path=smplx_path,
        device=device,
        use_smplx=True
    )
