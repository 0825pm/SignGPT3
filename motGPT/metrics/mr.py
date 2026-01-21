"""
SOKE-style Motion Reconstruction Metrics (Fast Version)
Computes MPJPE and MPJPE_PA for sign language VAE evaluation
No DTW - for VAE where input/output lengths are equal
"""
from typing import List
from collections import defaultdict
import gc

import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric


def rigid_align_batch(source, target):
    """
    Procrustes alignment: align source to target using SVD (batched).
    
    Args:
        source: (T, N, 3) source points
        target: (T, N, 3) target points
    
    Returns:
        aligned: (T, N, 3) aligned source points
    """
    T, N, _ = source.shape
    aligned = np.zeros_like(source)
    
    for t in range(T):
        src = source[t]  # (N, 3)
        tgt = target[t]  # (N, 3)
        
        # Center the points
        mu_src = src.mean(axis=0, keepdims=True)
        mu_tgt = tgt.mean(axis=0, keepdims=True)
        
        src_centered = src - mu_src
        tgt_centered = tgt - mu_tgt
        
        # Compute optimal rotation using SVD
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute scale
        var_src = (src_centered ** 2).sum()
        scale = S.sum() / max(var_src, 1e-8)
        
        # Apply transformation
        aligned[t] = scale * (src_centered @ R.T) + mu_tgt
    
    return aligned


class MRMetrics(Metric):
    """
    Motion Reconstruction Metrics for Sign Language (SOKE-style)
    
    Computes per-dataset (how2sign, csl, phoenix) metrics:
    - MPJPE: Root/Wrist-aligned (translation only)
    - MPJPE_PA: Procrustes aligned (translation + rotation + scale)
    
    For body parts: body, lhand, rhand, hand
    """

    def __init__(self,
                 njoints=55,
                 num_joints=55,
                 jointstype: str = "smplx",
                 force_in_meter: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.njoints = njoints if njoints else num_joints
        self.force_in_meter = force_in_meter  # multiply by 1000 for mm
        
        # Dataset sources
        self.sources = ['how2sign', 'csl', 'phoenix']
        
        # SMPL-X joint indices (55 joints)
        self.joint_part2idx = {
            'body': list(range(0, 22)),        # Body joints
            'lhand': list(range(25, 40)),      # Left hand (15 joints)
            'rhand': list(range(40, 55)),      # Right hand (15 joints)
        }
        
        # Alignment indices
        self.pelvis_idx = 0      # Pelvis for body
        self.lwrist_idx = 20     # Left wrist
        self.rwrist_idx = 21     # Right wrist
        
        # Feature part indices (133-dim SOKE format)
        self.smplx_part2idx = {
            'body': list(range(9, 72)),          # 21 body joints * 3
            'lhand': list(range(72, 117)),       # 15 left hand joints * 3  
            'rhand': list(range(117, 133)),      # partial right hand
            'hand': list(range(72, 133)),        # both hands
        }
        
        # Initialize states for each source and metric
        for src in self.sources:
            self.add_state(f"{src}_count", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"{src}_count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
            
            # MPJPE metrics (root/wrist aligned - translation only)
            for part in ['body', 'lhand', 'rhand', 'hand']:
                self.add_state(f"{src}_MPJPE_{part}", 
                              default=torch.tensor(0.0), dist_reduce_fx="sum")
            
            # MPJPE_PA metrics (Procrustes aligned)
            for part in ['body', 'lhand', 'rhand', 'hand']:
                self.add_state(f"{src}_MPJPE_PA_{part}", 
                              default=torch.tensor(0.0), dist_reduce_fx="sum")
            
            # Feature error
            self.add_state(f"{src}_feat_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{src}_feat_error_hand", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _compute_mpjpe(self, joints_rst, joints_ref, part='body', use_pa=False):
        """
        Compute MPJPE for a specific body part.
        
        Args:
            joints_rst: (T, N, 3) reconstructed joints
            joints_ref: (T, N, 3) reference joints
            part: body part name ('body', 'lhand', 'rhand')
            use_pa: if True, use Procrustes alignment (MPJPE_PA)
                    if False, use wrist/root alignment (MPJPE)
        
        Returns:
            mpjpe: MPJPE in mm
        """
        # Get joint indices for this part
        if part not in self.joint_part2idx:
            return 0.0
        
        joint_idx = self.joint_part2idx[part]
        
        # Check if indices are valid
        if max(joint_idx) >= joints_rst.shape[1]:
            return 0.0
        
        # Extract part joints
        j_rst = joints_rst[:, joint_idx, :]  # (T, num_joints, 3)
        j_ref = joints_ref[:, joint_idx, :]
        
        if use_pa:
            # Procrustes alignment (per frame)
            j_rst = rigid_align_batch(j_rst, j_ref)
        else:
            # Translation alignment (wrist for hands, pelvis for body)
            if part == 'lhand':
                # Align by first joint of hand (wrist-relative)
                align_idx = 0
            elif part == 'rhand':
                align_idx = 0
            else:  # body
                align_idx = 0  # pelvis
            
            # Subtract alignment joint
            j_rst = j_rst - j_rst[:, align_idx:align_idx+1, :] + j_ref[:, align_idx:align_idx+1, :]
        
        # Compute MPJPE
        diff = j_rst - j_ref
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # (T, num_joints)
        mpjpe = dist.mean()
        
        # Convert to mm
        if self.force_in_meter:
            mpjpe = mpjpe * 1000.0
        
        return mpjpe

    @torch.no_grad()
    def update(self,
               feats_rst: Tensor = None,
               feats_ref: Tensor = None,
               joints_rst: Tensor = None,
               joints_ref: Tensor = None,
               vertices_rst: Tensor = None,
               vertices_ref: Tensor = None,
               lengths: List[int] = None,
               src: List[str] = None,
               name: List[str] = None,
               **kwargs):
        """
        Update metrics with batch data.
        """
        if lengths is None:
            return
        
        B = len(lengths)
        
        # Default source
        if src is None:
            src = ['how2sign'] * B
        
        # ===== Feature-based metrics =====
        if feats_rst is not None and feats_ref is not None:
            feats_rst = feats_rst.detach().cpu()
            feats_ref = feats_ref.detach().cpu()
            
            for i in range(B):
                cur_len = lengths[i]
                data_src = src[i]
                
                # Feature error
                feat_rst = feats_rst[i, :cur_len]
                feat_ref = feats_ref[i, :cur_len]
                
                feat_mse = torch.sqrt(((feat_rst - feat_ref) ** 2).mean()).item()
                setattr(self, f'{data_src}_feat_error',
                       getattr(self, f'{data_src}_feat_error') + feat_mse * cur_len)
                
                # Hand feature error
                hand_idx = self.smplx_part2idx.get('hand', list(range(72, 133)))
                max_idx = max(hand_idx) + 1 if hand_idx else 133
                if feat_rst.shape[-1] >= max_idx:
                    feat_rst_hand = feat_rst[..., hand_idx]
                    feat_ref_hand = feat_ref[..., hand_idx]
                    hand_mse = torch.sqrt(((feat_rst_hand - feat_ref_hand) ** 2).mean()).item()
                    setattr(self, f'{data_src}_feat_error_hand',
                           getattr(self, f'{data_src}_feat_error_hand') + hand_mse * cur_len)
                
                # Update counts
                setattr(self, f'{data_src}_count',
                       getattr(self, f'{data_src}_count') + cur_len)
                setattr(self, f'{data_src}_count_seq',
                       getattr(self, f'{data_src}_count_seq') + 1)
        
        # ===== Joint-based MPJPE metrics =====
        if joints_rst is not None and joints_ref is not None:
            # Convert to numpy
            if isinstance(joints_rst, torch.Tensor):
                joints_rst = joints_rst.detach().cpu().numpy()
            if isinstance(joints_ref, torch.Tensor):
                joints_ref = joints_ref.detach().cpu().numpy()
            
            # Handle different input shapes
            if joints_rst.ndim == 4:
                # Already (B, T, N, 3)
                pass
            elif joints_rst.ndim == 3:
                # (BT, N, 3) -> (B, T, N, 3)
                BT, N, _ = joints_rst.shape
                T = BT // B
                joints_rst = joints_rst.reshape(B, T, N, 3)
                joints_ref = joints_ref.reshape(B, T, N, 3)
            else:
                print(f"[MRMetrics] Unexpected joints shape: {joints_rst.shape}")
                return
            
            for i in range(B):
                cur_len = lengths[i]
                data_src = src[i]
                
                j_rst = joints_rst[i, :cur_len]  # (T, N, 3)
                j_ref = joints_ref[i, :cur_len]  # (T, N, 3)
                
                # Update count if not done in feature section
                if feats_rst is None:
                    setattr(self, f'{data_src}_count',
                           getattr(self, f'{data_src}_count') + cur_len)
                    setattr(self, f'{data_src}_count_seq',
                           getattr(self, f'{data_src}_count_seq') + 1)
                
                # Compute MPJPE and MPJPE_PA for each part
                for part in ['body', 'lhand', 'rhand']:
                    # MPJPE (wrist/root aligned)
                    mpjpe = self._compute_mpjpe(j_rst, j_ref, part=part, use_pa=False)
                    attr = f'{data_src}_MPJPE_{part}'
                    setattr(self, attr, getattr(self, attr) + mpjpe * cur_len)
                    
                    # MPJPE_PA (Procrustes aligned)
                    mpjpe_pa = self._compute_mpjpe(j_rst, j_ref, part=part, use_pa=True)
                    attr_pa = f'{data_src}_MPJPE_PA_{part}'
                    setattr(self, attr_pa, getattr(self, attr_pa) + mpjpe_pa * cur_len)
                
                # Combined hand metrics (average of lhand and rhand)
                lhand_mpjpe = self._compute_mpjpe(j_rst, j_ref, part='lhand', use_pa=False)
                rhand_mpjpe = self._compute_mpjpe(j_rst, j_ref, part='rhand', use_pa=False)
                hand_mpjpe = (lhand_mpjpe + rhand_mpjpe) / 2
                setattr(self, f'{data_src}_MPJPE_hand',
                       getattr(self, f'{data_src}_MPJPE_hand') + hand_mpjpe * cur_len)
                
                lhand_pa = self._compute_mpjpe(j_rst, j_ref, part='lhand', use_pa=True)
                rhand_pa = self._compute_mpjpe(j_rst, j_ref, part='rhand', use_pa=True)
                hand_pa = (lhand_pa + rhand_pa) / 2
                setattr(self, f'{data_src}_MPJPE_PA_hand',
                       getattr(self, f'{data_src}_MPJPE_PA_hand') + hand_pa * cur_len)
        
        # Cleanup
        gc.collect()

    def compute(self, sanity_flag=False):
        """Compute final metrics"""
        metrics = {}
        
        for src in self.sources:
            count = getattr(self, f'{src}_count')
            count_seq = getattr(self, f'{src}_count_seq')
            
            if count_seq == 0:
                continue
            
            count_val = count.item() if isinstance(count, torch.Tensor) else count
            
            # Feature errors
            feat_error = getattr(self, f'{src}_feat_error') / max(count_val, 1)
            feat_error_hand = getattr(self, f'{src}_feat_error_hand') / max(count_val, 1)
            metrics[f'{src}_feat_error'] = feat_error
            metrics[f'{src}_feat_error_hand'] = feat_error_hand
            
            # MPJPE metrics
            for part in ['body', 'lhand', 'rhand', 'hand']:
                # MPJPE
                mpjpe = getattr(self, f'{src}_MPJPE_{part}') / max(count_val, 1)
                metrics[f'{src}_MPJPE_{part}'] = mpjpe
                
                # MPJPE_PA
                mpjpe_pa = getattr(self, f'{src}_MPJPE_PA_{part}') / max(count_val, 1)
                metrics[f'{src}_MPJPE_PA_{part}'] = mpjpe_pa
        
        # Print summary
        if not sanity_flag:
            print("\n" + "=" * 50)
            print("=== MRMetrics Results ===")
            print("=" * 50)
            for name, value in sorted(metrics.items()):
                if isinstance(value, torch.Tensor):
                    value = value.item()
                print(f"  {name}: {value:.4f}")
            print("=" * 50 + "\n")
        
        self.reset()
        return metrics