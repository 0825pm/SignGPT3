"""
Motion Reconstruction Metrics for Sign Language (SOKE-compatible)
SignGPT3/motGPT/metrics/mr.py 로 복사하세요.
"""
from typing import List
from collections import defaultdict
import gc

import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric


def rigid_align_batch(source, target):
    """Procrustes alignment"""
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    T, N, _ = source.shape
    aligned = np.zeros_like(source)
    
    for t in range(T):
        src = source[t]
        tgt = target[t]
        
        mu_src = src.mean(axis=0, keepdims=True)
        mu_tgt = tgt.mean(axis=0, keepdims=True)
        
        src_centered = src - mu_src
        tgt_centered = tgt - mu_tgt
        
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        var_src = (src_centered ** 2).sum()
        scale = S.sum() / max(var_src, 1e-8)
        
        aligned[t] = scale * (src_centered @ R.T) + mu_tgt
    
    return aligned


def rigid_align_torch_batch(source, target):
    """Torch version of rigid alignment"""
    source_np = source.detach().cpu().numpy() if isinstance(source, torch.Tensor) else source
    target_np = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target
    aligned = rigid_align_batch(source_np, target_np)
    return torch.from_numpy(aligned).to(source.device) if isinstance(source, torch.Tensor) else aligned


class MRMetrics(Metric):
    """
    Motion Reconstruction Metrics for Sign Language
    Compatible with motgpt.py calling conventions
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
        self.align_root = align_root
        self.force_in_meter = force_in_meter
        self.njoints = njoints

        # SOKE 133-dim Feature Part Indices
        self.smplx_part2idx = {
            'upper_body': list(range(30)), 
            'lhand': list(range(30, 75)), 
            'rhand': list(range(75, 120)), 
            'hand': list(range(30, 120)), 
            'face': list(range(120, 133))
        }
        
        # Joint part indices (55 joints)
        self.joint_part2idx = {
            'body': list(range(0, 22)),
            'lhand': list(range(25, 40)),
            'rhand': list(range(40, 55)),
        }
        
        self.name2scores = defaultdict(dict)
        self.sources = ['how2sign', 'csl', 'phoenix']

        # Initialize all states for each source
        for src in self.sources:
            self.add_state(f"{src}_count", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"{src}_count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

            # MPVPE metrics (vertex-based)
            self.add_state(f"{src}_MPVPE_PA_all", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPVPE_PA_hand", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPVPE_PA_face", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPVPE_all", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPVPE_hand", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPVPE_face", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            
            # MPJPE metrics (joint-based)
            self.add_state(f"{src}_MPJPE_PA_body", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPJPE_PA_hand", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPJPE_body", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_MPJPE_hand", default=torch.tensor([0.0]), dist_reduce_fx="sum")

            # Feature error metrics
            self.add_state(f"{src}_feat_error", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state(f"{src}_feat_error_hand", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        # Metric names - feat_error 포함
        m = ["MPVPE_PA_all", "MPVPE_PA_hand", "MPVPE_PA_face",
             "MPJPE_PA_body", "MPJPE_PA_hand", "MPJPE_body", "MPJPE_hand",
             "MPVPE_all", "MPVPE_hand", "MPVPE_face",
             "feat_error", "feat_error_hand"]
        self.MR_metrics = []
        for d in self.sources:
            for m_ in m:
                self.MR_metrics.append(f'{d}_{m_}')

        self.metrics = self.MR_metrics

    def _compute_mpjpe(self, joints_rst, joints_ref, part='body', use_pa=False):
        """Compute MPJPE for a body part"""
        if part == 'hand':
            part_idx = self.joint_part2idx['lhand'] + self.joint_part2idx['rhand']
        else:
            part_idx = self.joint_part2idx.get(part, self.joint_part2idx['body'])
        
        max_joint = joints_rst.shape[1]
        part_idx = [i for i in part_idx if i < max_joint]
        
        if len(part_idx) == 0:
            return 0.0
        
        rst = joints_rst[:, part_idx, :]
        ref = joints_ref[:, part_idx, :]
        
        rst = rst - rst[:, [0], :]
        ref = ref - ref[:, [0], :]
        
        if use_pa:
            rst = rigid_align_batch(rst, ref)
            if isinstance(rst, torch.Tensor):
                rst = rst.numpy()
            if isinstance(ref, torch.Tensor):
                ref = ref.numpy()
        else:
            if isinstance(rst, torch.Tensor):
                rst = rst.numpy()
            if isinstance(ref, torch.Tensor):
                ref = ref.numpy()
        
        error = np.sqrt(((rst - ref) ** 2).sum(axis=-1))
        return error.mean()

    def compute(self, sanity_flag=False):
        """Compute final metrics - 모든 메트릭 항상 반환"""
        if self.force_in_meter:
            factor = 1000.0
        else:
            factor = 1.0

        mr_metrics = {}

        # 모든 메트릭을 항상 반환 (없으면 0)
        for name in self.MR_metrics:
            d = name.split('_')[0]
            count = getattr(self, f'{d}_count')
            value = getattr(self, name)
            
            # count가 0이면 0 반환
            if count == 0:
                mr_metrics[name] = torch.tensor([0.0])
            else:
                mr_metrics[name] = value / count
                if 'MPVPE' in name or 'MPJPE' in name:
                    mr_metrics[name] = mr_metrics[name] * factor

        if not sanity_flag:
            print("\n" + "=" * 50)
            print("=== MRMetrics Results ===")
            for name, v in mr_metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if val > 0:
                    print(f"  {name}: {val:.4f}")
            print("=" * 50)
        
        self.reset()
        return mr_metrics

    def update(self, 
               feats_rst: Tensor, 
               feats_ref: Tensor,
               joints_rst: Tensor = None, 
               joints_ref: Tensor = None,
               vertices_rst: Tensor = None, 
               vertices_ref: Tensor = None,
               lengths: List[int] = None, 
               src: List[str] = None, 
               name: List[str] = None,
               **kwargs):
        """Update metrics"""
        if lengths is None:
            lengths = [feats_rst.shape[1]] * feats_rst.shape[0]
        if src is None:
            src = ['how2sign'] * len(lengths)
        if name is None:
            name = [f'sample_{i}' for i in range(len(lengths))]
            
        B = len(lengths)
        
        # Reshape joints if needed
        if joints_rst is not None and joints_ref is not None:
            if joints_rst.dim() == 3:
                BT, N, _ = joints_rst.shape
                T = BT // B
                joints_rst = joints_rst.reshape(B, T, N, 3)
                joints_ref = joints_ref.reshape(B, T, N, 3)
            joints_rst = joints_rst.detach().cpu()
            joints_ref = joints_ref.detach().cpu()
        
        # Reshape vertices if needed
        if vertices_rst is not None and vertices_ref is not None:
            if vertices_rst.dim() == 3:
                BT, N, _ = vertices_rst.shape
                T = BT // B
                vertices_rst = vertices_rst.reshape(B, T, N, 3)
                vertices_ref = vertices_ref.reshape(B, T, N, 3)
            vertices_rst = vertices_rst.detach().cpu()
            vertices_ref = vertices_ref.detach().cpu()
        
        for i in range(B):
            cur_len = lengths[i]
            data_src = src[i] if isinstance(src, list) else src
            cur_name = name[i] if isinstance(name, list) else name
            
            # Update count
            setattr(self, f'{data_src}_count', cur_len + getattr(self, f'{data_src}_count'))
            setattr(self, f'{data_src}_count_seq', 1 + getattr(self, f'{data_src}_count_seq'))
            
            # ===== Feature error metrics =====
            if feats_rst is not None and feats_ref is not None:
                f_rst = feats_rst[i, :cur_len]
                f_ref = feats_ref[i, :cur_len]
                
                feat_error = torch.abs(f_rst - f_ref).mean() * cur_len
                setattr(self, f'{data_src}_feat_error', getattr(self, f'{data_src}_feat_error') + feat_error)
                
                hand_idx = self.smplx_part2idx['hand']
                feat_error_hand = torch.abs(f_rst[:, hand_idx] - f_ref[:, hand_idx]).mean() * cur_len
                setattr(self, f'{data_src}_feat_error_hand', getattr(self, f'{data_src}_feat_error_hand') + feat_error_hand)
            
            # ===== Vertex-based metrics (MPVPE) =====
            if vertices_rst is not None and vertices_ref is not None:
                mesh_gt = vertices_ref[i, :cur_len]
                mesh_out = vertices_rst[i, :cur_len]
                
                mesh_out_align = rigid_align_torch_batch(mesh_out, mesh_gt)
                if isinstance(mesh_out_align, np.ndarray):
                    mesh_out_align = torch.from_numpy(mesh_out_align)
                if isinstance(mesh_gt, np.ndarray):
                    mesh_gt = torch.from_numpy(mesh_gt)
                    
                value = torch.mean(torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1)), dim=-1).sum()
                setattr(self, f"{data_src}_MPVPE_PA_all", getattr(self, f"{data_src}_MPVPE_PA_all") + value)
                
                if isinstance(mesh_out, np.ndarray):
                    mesh_out = torch.from_numpy(mesh_out)
                value = torch.mean(torch.sqrt(torch.sum((mesh_out - mesh_gt) ** 2, dim=-1)), dim=-1).sum()
                setattr(self, f"{data_src}_MPVPE_all", getattr(self, f"{data_src}_MPVPE_all") + value)
            
            # ===== Joint-based metrics (MPJPE) =====
            if joints_rst is not None and joints_ref is not None:
                j_rst = joints_rst[i, :cur_len].numpy() if isinstance(joints_rst, torch.Tensor) else joints_rst[i, :cur_len]
                j_ref = joints_ref[i, :cur_len].numpy() if isinstance(joints_ref, torch.Tensor) else joints_ref[i, :cur_len]
                
                mpjpe_body = self._compute_mpjpe(j_rst, j_ref, part='body', use_pa=False)
                setattr(self, f'{data_src}_MPJPE_body', getattr(self, f'{data_src}_MPJPE_body') + mpjpe_body * cur_len)
                
                mpjpe_pa_body = self._compute_mpjpe(j_rst, j_ref, part='body', use_pa=True)
                setattr(self, f'{data_src}_MPJPE_PA_body', getattr(self, f'{data_src}_MPJPE_PA_body') + mpjpe_pa_body * cur_len)
                
                mpjpe_lhand = self._compute_mpjpe(j_rst, j_ref, part='lhand', use_pa=False)
                mpjpe_rhand = self._compute_mpjpe(j_rst, j_ref, part='rhand', use_pa=False)
                mpjpe_hand = (mpjpe_lhand + mpjpe_rhand) / 2
                setattr(self, f'{data_src}_MPJPE_hand', getattr(self, f'{data_src}_MPJPE_hand') + mpjpe_hand * cur_len)
                
                mpjpe_pa_lhand = self._compute_mpjpe(j_rst, j_ref, part='lhand', use_pa=True)
                mpjpe_pa_rhand = self._compute_mpjpe(j_rst, j_ref, part='rhand', use_pa=True)
                mpjpe_pa_hand = (mpjpe_pa_lhand + mpjpe_pa_rhand) / 2
                setattr(self, f'{data_src}_MPJPE_PA_hand', getattr(self, f'{data_src}_MPJPE_PA_hand') + mpjpe_pa_hand * cur_len)
        
        gc.collect()