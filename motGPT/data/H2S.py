"""
H2S DataModule for Sign Language (SOKE-style)
Compatible with MotionGPT3 architecture
"""
import os
import torch
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from motGPT.data.signlang import (
    SignMotionDataset,
    SignText2MotionDataset,
    SignText2MotionDatasetEval,
    sign_collate,
    sign_collate_simple
)

# Try to import SMPL-X utilities
try:
    from motGPT.utils.human_models import get_coord, smpl_x
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False
    print("Warning: human_models not found. feats2joints will return zeros.")


def feats2joints_sign(features, njoints=55):
    """Placeholder for feature to joints conversion (returns zeros)"""
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
    batch, seq, feat_dim = features.shape
    joints = torch.zeros(batch, seq, njoints, 3, device=features.device)
    return joints


def feats2joints_smplx(features, mean, std):
    """
    Convert 120/133-dim SOKE features to 3D joints using SMPL-X.
    """
    # Handle 2D input (T, D) -> (1, T, D)
    squeeze_output = False
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
        squeeze_output = True
    
    B, T, D = features.shape
    
    if not HAS_SMPLX:
        # Fallback to zeros
        joints = torch.zeros(B, T, 55, 3, device=features.device)
        if squeeze_output:
            joints = joints.squeeze(0)
        return None, joints
    
    # Denormalize
    mean = mean.to(features.device)
    std = std.to(features.device)
    features = features * std + mean
    
    # Default shape parameters
    shape_param = torch.tensor([[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                                  0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
                               device=features.device, dtype=features.dtype)
    shape_param = shape_param.unsqueeze(0).repeat(B, T, 1).view(B*T, -1)
    
    # Parse features based on dimension
    # features_flat = features.view(B*T, D)
    features_flat = features.reshape(B*T, D)
    
    if D == 120:
        # 120-dim: upper_body(30) + lhand(45) + rhand(45)
        upper_body_pose = features_flat[:, 0:30]
        lhand_pose = features_flat[:, 30:75]
        rhand_pose = features_flat[:, 75:120]
        
        # Zeros for missing parts
        lower_body_zeros = torch.zeros(B*T, 33, device=features.device, dtype=features.dtype)
        body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)  # [B*T, 63]
        root_pose = torch.zeros(B*T, 3, device=features.device, dtype=features.dtype)
        jaw_pose = torch.zeros(B*T, 3, device=features.device, dtype=features.dtype)
        expr = torch.zeros(B*T, 10, device=features.device, dtype=features.dtype)
        
    elif D == 133:
        # 133-dim: upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
        upper_body_pose = features_flat[:, 0:30]
        lhand_pose = features_flat[:, 30:75]
        rhand_pose = features_flat[:, 75:120]
        jaw_pose = features_flat[:, 120:123]
        expr = features_flat[:, 123:133]
        
        lower_body_zeros = torch.zeros(B*T, 33, device=features.device, dtype=features.dtype)
        body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)
        root_pose = torch.zeros(B*T, 3, device=features.device, dtype=features.dtype)
        
    else:
        print(f"Warning: Unsupported feature dim {D}, returning zeros")
        joints = torch.zeros(B, T, 55, 3, device=features.device, dtype=features.dtype)
        if squeeze_output:
            joints = joints.squeeze(0)
        return None, joints
    
    try:
        vertices, joints = get_coord(
            root_pose=root_pose,
            body_pose=body_pose,
            lhand_pose=lhand_pose,
            rhand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            shape=shape_param,
            expr=expr
        )
        
        # Reshape back to [B, T, ...]
        if vertices is not None:
            vertices = vertices.view(B, T, -1, 3)
        joints = joints.view(B, T, -1, 3)
        
    except Exception as e:
        print(f"Warning: SMPL-X forward failed: {e}")
        vertices = None
        joints = torch.zeros(B, T, 55, 3, device=features.device, dtype=features.dtype)
    
    # Squeeze if input was 2D
    if squeeze_output:
        if vertices is not None:
            vertices = vertices.squeeze(0)
        joints = joints.squeeze(0)
    
    return vertices, joints


class H2SDataModule(LightningDataModule):
    """DataModule for How2Sign, CSL-Daily, Phoenix (SOKE-style)"""
    
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Required attributes for MotGPT model
        self.name = cfg.DATASET.H2S.get('DATASET_NAME', 'how2sign_csl_phoenix')
        self.njoints = cfg.DATASET.get('NJOINTS', 55)
        self.fps = cfg.DATASET.H2S.get('FPS', 25)
        self.nfeats = cfg.DATASET.get('NFEATS', 133)
        
        # Data paths
        self.data_root = cfg.DATASET.H2S.ROOT
        self.csl_root = cfg.DATASET.H2S.get('CSL_ROOT', None)
        self.phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)
        self.hparams.data_root = self.data_root
        self.hparams.fps = self.fps
        self.hparams.max_motion_length = cfg.DATASET.H2S.get('MAX_MOTION_LEN', 300)
        self.hparams.min_motion_length = cfg.DATASET.H2S.get('MIN_MOTION_LEN', 15)
        self.hparams.unit_length = cfg.DATASET.H2S.get('UNIT_LEN', 4)
        
        # Stage
        self.stage = cfg.TRAIN.get('STAGE', 'vae')
        
        # Load mean and std
        mean_path = cfg.DATASET.H2S.get('MEAN_PATH', None)
        std_path = cfg.DATASET.H2S.get('STD_PATH', None)
        
        if mean_path and os.path.exists(mean_path):
            print(f"Loading mean from: {mean_path}")
            self.mean = torch.load(mean_path)[:120]
        else:
            self.mean = torch.zeros(self.nfeats)
            
        if std_path and os.path.exists(std_path):
            print(f"Loading std from: {std_path}")
            self.std = torch.load(std_path)[:120]
        else:
            self.std = torch.ones(self.nfeats)
        
        # Convert 179-dim to 133-dim if needed
        if self.mean.shape[0] == 179:
            print(f"Converting mean/std from 179-dim to 133-dim")
            self.mean = self._convert_179_to_133(self.mean)
            self.std = self._convert_179_to_133(self.std)
        
        # Ensure correct dimension
        self.mean = self._ensure_dim(self.mean, self.nfeats)
        self.std = self._ensure_dim(self.std, self.nfeats)
        
        self.hparams.mean = self.mean
        self.hparams.std = self.std
        
        # Set feats2joints with SMPL-X support
        # Wrapper to handle tuple return (vertices, joints) -> joints only for motgpt_2optimizer
        _mean = self.mean
        _std = self.std
        _njoints = self.njoints
        
        def _feats2joints_wrapper(x):
            result = feats2joints_smplx(x, _mean, _std)
            if isinstance(result, tuple):
                return result[1]  # joints만 반환
            return result
        
        if HAS_SMPLX:
            self.feats2joints = _feats2joints_wrapper
            print(f"  feats2joints: SMPL-X enabled")
        else:
            self.feats2joints = lambda x: feats2joints_sign(x, _njoints)
            print(f"  feats2joints: Placeholder (zeros)")
        
        # Dataset tracking
        self.is_mm = False
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Set joint type
        cfg.DATASET.JOINT_TYPE = 'smplx'
        
        print(f"H2SDataModule initialized:")
        print(f"  - Dataset: {self.name}, nfeats: {self.nfeats}, fps: {self.fps}")
        print(f"  - Mean shape: {self.mean.shape}")
    
    def _convert_179_to_133(self, tensor):
        """Convert SOKE 179-dim to 133-dim format."""
        if tensor.shape[0] != 179:
            return tensor
        # Remove root position (3) and lower body (11*3=33) = 36 dims
        tensor = tensor[(3 + 3 * 11):]  # Skip first 36
        # Remove unused dims, keep expression (last 10)
        tensor = torch.cat([tensor[:-20], tensor[-10:]])
        return tensor
    
    def _ensure_dim(self, tensor, target_dim):
        """Ensure tensor has target dimension."""
        if tensor.shape[0] > target_dim:
            return tensor[:target_dim]
        elif tensor.shape[0] < target_dim:
            return torch.cat([tensor, torch.zeros(target_dim - tensor.shape[0])])
        return tensor
    
    def normalize(self, motion):
        """Normalize motion features."""
        mean = self.mean.to(motion.device)
        std = torch.clamp(self.std.to(motion.device), min=1e-8)
        return (motion - mean) / std
    
    def denormalize(self, motion):
        """Denormalize motion features."""
        mean = self.mean.to(motion.device)
        std = self.std.to(motion.device)
        return motion * std + mean
    
    def renorm4t2m(self, motion):
        """Renormalize for T2M evaluation (identity for sign language)."""
        return motion
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        from torch.utils.data import Subset  # 추가
        
        common_kwargs = {
            'data_root': self.data_root,
            'csl_root': self.csl_root,
            'phoenix_root': self.phoenix_root,
            'dataset_name': self.name,
            'max_motion_length': self.hparams.max_motion_length,
            'min_motion_length': self.hparams.min_motion_length,
            'unit_length': self.hparams.unit_length,
            'mean': self.mean,
            'std': self.std,
        }
        
        if self.stage == 'vae':
            DatasetClass = SignMotionDataset
        else:
            DatasetClass = SignText2MotionDataset
        
        # =========== [TINY MODE] 추가 ===========
        tiny = self.cfg.get('TINY', False)
        tiny_size = self.cfg.get('TINY_SIZE', 32)
        # ========================================
        
        if stage == 'fit' or stage is None:
            train_full = DatasetClass(split='train', **common_kwargs)
            val_full = DatasetClass(split='val', **common_kwargs)
            
            # =========== [TINY MODE] 추가 ===========
            if tiny:
                print(f"🔬 [TINY MODE] Train: {min(tiny_size, len(train_full))}, Val: {min(tiny_size//4, len(val_full))}")
                self.train_dataset = Subset(train_full, list(range(min(tiny_size, len(train_full)))))
                self.val_dataset = Subset(val_full, list(range(min(tiny_size//4, len(val_full)))))
            else:
                self.train_dataset = train_full
                self.val_dataset = val_full
            # ========================================
        
        if stage == 'test' or stage is None:
            if self.stage == 'vae':
                self.test_dataset = SignMotionDataset(split='test', **common_kwargs)
            else:
                self.test_dataset = SignText2MotionDatasetEval(split='test', **common_kwargs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.get('NUM_WORKERS', 4),
            collate_fn=sign_collate if self.stage != 'vae' else sign_collate_simple,
            drop_last=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.EVAL.get('BATCH_SIZE', 32),
            shuffle=False,
            num_workers=self.cfg.EVAL.get('NUM_WORKERS', 4),
            collate_fn=sign_collate if self.stage != 'vae' else sign_collate_simple,
            drop_last=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.TEST.get('BATCH_SIZE', 32),
            shuffle=False,
            num_workers=self.cfg.TEST.get('NUM_WORKERS', 4),
            collate_fn=sign_collate if self.stage != 'vae' else sign_collate_simple,
            drop_last=False,
            pin_memory=True,
        )
    
    def mm_mode(self, mm_on=True):
        """Toggle multimodal evaluation mode."""
        if mm_on:
            self.is_mm = True
            if hasattr(self.test_dataset, 'name_list'):
                self.name_list = self.test_dataset.name_list
                self.mm_list = np.random.choice(
                    self.name_list,
                    min(self.cfg.METRIC.MM_NUM_SAMPLES, len(self.name_list)),
                    replace=False
                )
                self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            if hasattr(self, 'name_list'):
                self.test_dataset.name_list = self.name_list