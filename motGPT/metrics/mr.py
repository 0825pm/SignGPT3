"""
SOKE-style Motion Reconstruction Metrics
Feature-based computation for sign language VAE
"""
from typing import List
from collections import defaultdict

import torch
from torch import Tensor
from torchmetrics import Metric


class MRMetrics(Metric):
    """
    Motion Reconstruction Metrics for Sign Language (SOKE-style)
    Computes per-dataset (how2sign, csl, phoenix) metrics
    """

    def __init__(self,
                 njoints=55,
                 jointstype: str = "smplx",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.njoints = njoints
        
        # Dataset sources
        self.sources = ['how2sign', 'csl', 'phoenix']
        
        # Feature part indices (133-dim SOKE format)
        self.smplx_part2idx = {
            'face': list(range(0, 9)),           # jaw + eyes
            'body': list(range(9, 72)),          # 21 body joints * 3
            'lhand': list(range(72, 117)),       # 15 left hand joints * 3
            'rhand': list(range(117, 133)),      # partial right hand
            'hand': list(range(72, 133)),        # both hands
            'all': list(range(133)),
        }
        
        self.name2scores = defaultdict(dict)
        
        # Initialize states for each source
        for src in self.sources:
            self.add_state(f"{src}_count", default=torch.tensor(0), dist_reduce_fx="sum")
            
            # MPVPE metrics
            for metric in ['MPVPE_PA_all', 'MPVPE_all', 'MPVPE_PA_hand', 'MPVPE_hand',
                          'MPVPE_PA_face', 'MPVPE_face']:
                self.add_state(f"{src}_{metric}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            
            # MPJPE metrics
            for metric in ['MPJPE_PA_body', 'MPJPE_body', 'MPJPE_PA_hand', 'MPJPE_hand']:
                self.add_state(f"{src}_{metric}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            
            # Feature error metrics
            self.add_state(f"{src}_feat_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{src}_feat_error_hand", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        # Build metrics list
        self.MR_metrics = []
        for src in self.sources:
            self.MR_metrics.extend([
                f"{src}_MPVPE_PA_all", f"{src}_MPVPE_all",
                f"{src}_MPVPE_PA_hand", f"{src}_MPVPE_hand",
                f"{src}_MPVPE_PA_face", f"{src}_MPVPE_face",
                f"{src}_MPJPE_PA_body", f"{src}_MPJPE_body",
                f"{src}_MPJPE_PA_hand", f"{src}_MPJPE_hand",
                f"{src}_feat_error", f"{src}_feat_error_hand",
            ])
        
        self._update_count = 0

    def compute(self, sanity_flag=False):
        """Compute final metrics"""
        mr_metrics = {}
        
        print(f"\n=== MRMetrics Results (update_count={self._update_count}) ===")
        for name in self.MR_metrics:
            src = name.split('_')[0]
            count = getattr(self, f'{src}_count')
            value = getattr(self, name)
            
            if count > 0:
                mr_metrics[name] = (value / count).item()
            else:
                mr_metrics[name] = 0.0
            
            print(f"{name}: {mr_metrics[name]:.4f} (count={count.item()})")
        print("========================\n")
        
        self._update_count = 0
        self.reset()
        return mr_metrics

    def update(self, 
               feats_rst: Tensor = None,
               feats_ref: Tensor = None,
               lengths: List[int] = None,
               src: List[str] = None,
               name: List[str] = None,
               **kwargs):
        """
        Update metrics with batch data (SOKE-style)
        """
        self._update_count += 1
        
        # Handle different call signatures
        # Check if called with positional args (old style)
        if feats_rst is None and len(kwargs) > 0:
            # Try to get from kwargs
            feats_rst = kwargs.get('feats_rst', kwargs.get('m_rst'))
            feats_ref = kwargs.get('feats_ref', kwargs.get('m_ref'))
            lengths = kwargs.get('lengths', kwargs.get('length'))
        
        if feats_rst is None or feats_ref is None:
            print(f"[MRMetrics] WARNING: feats_rst or feats_ref is None")
            return
        
        if lengths is None:
            print(f"[MRMetrics] WARNING: lengths is None")
            return
        
        B = len(lengths)
        
        # Debug output (first call only)
        if self._update_count == 1:
            print(f"[MRMetrics] update called: B={B}, feats_rst.shape={feats_rst.shape}, "
                  f"feats_ref.shape={feats_ref.shape}, lengths={lengths[:3]}...")
            if src:
                print(f"[MRMetrics] src={src[:3]}...")
        
        if src is None:
            src = ['how2sign'] * B
        if name is None:
            name = [f'sample_{i}' for i in range(B)]
        
        feats_rst = feats_rst.detach().cpu()
        feats_ref = feats_ref.detach().cpu()
        
        for i in range(B):
            cur_len = lengths[i]
            data_src = src[i] if src[i] in self.sources else 'how2sign'
            cur_name = name[i] if name else f'sample_{i}'
            
            # Update count
            setattr(self, f'{data_src}_count', 
                   getattr(self, f'{data_src}_count') + cur_len)
            
            feat_rst = feats_rst[i, :cur_len]  # [T, D]
            feat_ref = feats_ref[i, :cur_len]  # [T, D]
            D = feat_rst.shape[-1]
            
            # Feature reconstruction error (all)
            feat_diff = feat_rst - feat_ref
            # MSE per frame, then sqrt for RMSE, then sum
            feat_error_per_frame = torch.sqrt(torch.mean(feat_diff ** 2, dim=-1))  # [T]
            feat_error_sum = feat_error_per_frame.sum()
            
            setattr(self, f'{data_src}_feat_error',
                   getattr(self, f'{data_src}_feat_error') + feat_error_sum)
            
            # NOTE: Features are in normalized space (mean=0, std=1-ish)
            # No scaling needed - values are comparable across parts
            # For actual MPJPE in mm, would need: denormalize → joints → mm
            
            # All body (same as feat_error, no scaling)
            mpvpe_all = feat_error_sum
            setattr(self, f'{data_src}_MPVPE_PA_all',
                   getattr(self, f'{data_src}_MPVPE_PA_all') + mpvpe_all)
            setattr(self, f'{data_src}_MPVPE_all',
                   getattr(self, f'{data_src}_MPVPE_all') + mpvpe_all)
            
            if D >= 72:
                # Hand features
                hand_end = min(D, 133)
                hand_idx = list(range(72, hand_end))
                if len(hand_idx) > 0:
                    feat_diff_hand = feat_rst[:, hand_idx] - feat_ref[:, hand_idx]
                    feat_error_hand = torch.sqrt(torch.mean(feat_diff_hand ** 2, dim=-1)).sum()
                    
                    setattr(self, f'{data_src}_feat_error_hand',
                           getattr(self, f'{data_src}_feat_error_hand') + feat_error_hand)
                    
                    # No scaling - keep in feature space
                    mpjpe_hand = feat_error_hand
                    setattr(self, f'{data_src}_MPVPE_PA_hand',
                           getattr(self, f'{data_src}_MPVPE_PA_hand') + mpjpe_hand)
                    setattr(self, f'{data_src}_MPVPE_hand',
                           getattr(self, f'{data_src}_MPVPE_hand') + mpjpe_hand)
                    setattr(self, f'{data_src}_MPJPE_PA_hand',
                           getattr(self, f'{data_src}_MPJPE_PA_hand') + mpjpe_hand)
                    setattr(self, f'{data_src}_MPJPE_hand',
                           getattr(self, f'{data_src}_MPJPE_hand') + mpjpe_hand)
            
            if D >= 9:
                # Face features
                face_idx = list(range(0, 9))
                feat_diff_face = feat_rst[:, face_idx] - feat_ref[:, face_idx]
                feat_error_face = torch.sqrt(torch.mean(feat_diff_face ** 2, dim=-1)).sum()
                
                # No scaling - keep in feature space
                mpvpe_face = feat_error_face
                setattr(self, f'{data_src}_MPVPE_PA_face',
                       getattr(self, f'{data_src}_MPVPE_PA_face') + mpvpe_face)
                setattr(self, f'{data_src}_MPVPE_face',
                       getattr(self, f'{data_src}_MPVPE_face') + mpvpe_face)
            
            if D >= 72:
                # Body features
                body_idx = list(range(9, 72))
                feat_diff_body = feat_rst[:, body_idx] - feat_ref[:, body_idx]
                feat_error_body = torch.sqrt(torch.mean(feat_diff_body ** 2, dim=-1)).sum()
                
                # No scaling - keep in feature space
                mpjpe_body = feat_error_body
                setattr(self, f'{data_src}_MPJPE_PA_body',
                       getattr(self, f'{data_src}_MPJPE_PA_body') + mpjpe_body)
                setattr(self, f'{data_src}_MPJPE_body',
                       getattr(self, f'{data_src}_MPJPE_body') + mpjpe_body)