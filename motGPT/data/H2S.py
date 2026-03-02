"""
motGPT/data/H2S.py
==================
DataModule for How2Sign, CSL-Daily, Phoenix-2014T

지원 feature type:
  - soke      : 120-dim axis-angle (.pt mean/std, 179→120 변환)
  - hml_style : 249-dim joint position (.npy mean/std, 변환 없음)

[수정 이력]
  - feature_type / hml_root 파라미터 추가
  - mean/std 로딩: .pt (SOKE) / .npy (HML) 분기
  - feats2joints: SOKE=SMPL-X FK, HML=position reshape
  - setup(): feature_type, hml_root를 dataset에 전달
"""

import os
import numpy as np
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# SMPL-X forward kinematics
try:
    from motGPT.utils.feats2joints import feats2joints_smplx
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False

from motGPT.data.signlang.dataset_sign import (
    SignMotionDataset,
    SignText2MotionDataset,
    SignText2MotionDatasetEval,
)
from motGPT.data.signlang.collate import sign_collate, sign_collate_simple


def feats2joints_sign(features, njoints=55):
    """Placeholder: zeros (SMPL-X 미설치 시)."""
    B, T = features.shape[:2]
    return torch.zeros(B, T, njoints, 3, device=features.device)


class H2SDataModule(pl.LightningDataModule):
    """DataModule for How2Sign, CSL-Daily, Phoenix (SOKE / HML style)."""

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        # ── 기본 속성 ──────────────────────────────────────────────────────
        self.name      = cfg.DATASET.H2S.get('DATASET_NAME', 'how2sign_csl_phoenix')
        self.njoints   = cfg.DATASET.get('NJOINTS', 55)
        self.fps       = cfg.DATASET.H2S.get('FPS', 25)
        self.nfeats    = cfg.DATASET.get('NFEATS', 120)
        self.stage     = cfg.TRAIN.get('STAGE', 'vae')

        # ── 데이터 경로 ────────────────────────────────────────────────────
        self.data_root    = cfg.DATASET.H2S.ROOT
        self.csl_root     = cfg.DATASET.H2S.get('CSL_ROOT', None)
        self.phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)

        self.hparams.data_root         = self.data_root
        self.hparams.fps               = self.fps
        self.hparams.max_motion_length = cfg.DATASET.H2S.get('MAX_MOTION_LEN', 300)
        self.hparams.min_motion_length = cfg.DATASET.H2S.get('MIN_MOTION_LEN', 15)
        self.hparams.unit_length       = cfg.DATASET.H2S.get('UNIT_LEN', 4)

        # ── feature type ───────────────────────────────────────────────────
        self.feature_type = cfg.DATASET.get('FEATURE_TYPE', 'soke')
        self.hml_root     = cfg.DATASET.H2S.get('HML_ROOT', None)

        # ── mean / std 로딩 ────────────────────────────────────────────────
        mean_path = cfg.DATASET.H2S.get('MEAN_PATH', None)
        std_path  = cfg.DATASET.H2S.get('STD_PATH', None)

        if self.feature_type == 'hml_style':
            # HML: .npy 파일, 통합 통계, 변환 없음
            if mean_path and os.path.exists(mean_path):
                print(f"[HML] Loading mean from: {mean_path}")
                self.mean = torch.from_numpy(np.load(mean_path)).float()
            else:
                print(f"[HML] mean.npy not found at {mean_path}, using zeros")
                self.mean = torch.zeros(self.nfeats)

            if std_path and os.path.exists(std_path):
                print(f"[HML] Loading std from: {std_path}")
                self.std = torch.from_numpy(np.load(std_path)).float()
            else:
                print(f"[HML] std.npy not found at {std_path}, using ones")
                self.std = torch.ones(self.nfeats)

        else:
            # SOKE: .pt 파일, 179→133→120 변환
            if mean_path and os.path.exists(mean_path):
                print(f"[SOKE] Loading mean from: {mean_path}")
                self.mean = torch.load(mean_path, weights_only=False)
            else:
                self.mean = torch.zeros(self.nfeats)

            if std_path and os.path.exists(std_path):
                print(f"[SOKE] Loading std from: {std_path}")
                self.std = torch.load(std_path, weights_only=False)
            else:
                self.std = torch.ones(self.nfeats)

            # 179-dim → 133-dim 변환
            if self.mean.shape[0] == 179:
                print("Converting mean/std from 179-dim to 133-dim")
                self.mean = self._convert_179_to_133(self.mean)
                self.std  = self._convert_179_to_133(self.std)

        # 최종 dim 맞추기
        self.mean = self._ensure_dim(self.mean, self.nfeats)
        self.std  = self._ensure_dim(self.std,  self.nfeats)

        self.hparams.mean = self.mean
        self.hparams.std  = self.std

        # ── feats2joints 설정 ──────────────────────────────────────────────
        _mean    = self.mean
        _std     = self.std
        _njoints = self.njoints

        if self.feature_type == 'hml_style':
            # HML: position 직접 reshape → SMPL-X FK 불필요
            # 249-dim 구조: 앞 123-dim = 41 joints × 3 (xyz position)
            # 41 joints = UPPER(11) + LHAND(15) + RHAND(15)
            UPPER  = [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            LHAND  = list(range(25, 40))
            RHAND  = list(range(40, 55))
            TARGET = UPPER + LHAND + RHAND  # 41 joints → SMPLX 55 index

            def _hml_feats2joints(x):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, T, _ = x.shape
                m = _mean.to(x.device)
                s = _std.to(x.device)
                x_denorm = x * s + m
                # 앞 123-dim → [B, T, 41, 3]
                joints_41 = x_denorm[..., :123].reshape(B, T, 41, 3)
                # SMPL-X 55-joint 텐서에 매핑
                out = torch.zeros(B, T, 55, 3, device=x.device)
                for local_i, smplx_j in enumerate(TARGET):
                    out[:, :, smplx_j, :] = joints_41[:, :, local_i, :]
                return out

            self.feats2joints = _hml_feats2joints
            print("  feats2joints: HML position reshape (SMPL-X 불필요)")

        else:
            # SOKE: SMPL-X FK
            def _feats2joints_wrapper(x):
                result = feats2joints_smplx(x, _mean, _std)
                if isinstance(result, tuple):
                    return result[1]  # (vertices, joints) → joints
                return result

            if HAS_SMPLX:
                self.feats2joints = _feats2joints_wrapper
                print("  feats2joints: SMPL-X FK enabled")
            else:
                self.feats2joints = lambda x: feats2joints_sign(x, _njoints)
                print("  feats2joints: Placeholder (SMPL-X not installed)")

        # ── 상태 초기화 ────────────────────────────────────────────────────
        self.is_mm        = False
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset   = None

        cfg.DATASET.JOINT_TYPE = 'smplx'

        print(f"H2SDataModule initialized:")
        print(f"  dataset={self.name}, nfeats={self.nfeats}, feature_type={self.feature_type}")
        print(f"  mean shape={self.mean.shape}, std shape={self.std.shape}")

    # ── 유틸리티 ──────────────────────────────────────────────────────────────

    def _convert_179_to_133(self, tensor):
        """SOKE 179-dim → 133-dim."""
        if tensor.shape[0] != 179:
            return tensor
        tensor = tensor[(3 + 3 * 11):]          # root + lower body 제거 → 143
        tensor = torch.cat([tensor[:-20], tensor[-10:]])  # shape 제거 → 133
        return tensor

    def _ensure_dim(self, tensor, target_dim):
        """dim 맞추기 (잘라내기 or 패딩)."""
        if tensor.shape[0] > target_dim:
            return tensor[:target_dim]
        elif tensor.shape[0] < target_dim:
            return torch.cat([tensor, torch.zeros(target_dim - tensor.shape[0])])
        return tensor

    # ── 정규화 ────────────────────────────────────────────────────────────────

    def normalize(self, motion):
        mean = self.mean.to(motion.device)
        std  = torch.clamp(self.std.to(motion.device), min=1e-8)
        return (motion - mean) / std

    def denormalize(self, motion):
        mean = self.mean.to(motion.device)
        std  = self.std.to(motion.device)
        return motion * std + mean

    def renorm4t2m(self, motion):
        """T2M 평가용 renormalize (sign language에서는 identity)."""
        return motion

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        from torch.utils.data import Subset

        common_kwargs = {
            'data_root':         self.data_root,
            'csl_root':          self.csl_root,
            'phoenix_root':      self.phoenix_root,
            'dataset_name':      self.name,
            'max_motion_length': self.hparams.max_motion_length,
            'min_motion_length': self.hparams.min_motion_length,
            'unit_length':       self.hparams.unit_length,
            'mean':              self.mean,
            'std':               self.std,
            # HML 지원
            'feature_type':      self.feature_type,
            'hml_root':          self.hml_root,
        }

        DatasetClass = SignMotionDataset if self.stage == 'vae' else SignText2MotionDataset

        # TINY 모드 (디버깅용)
        tiny      = self.cfg.get('TINY', False)
        tiny_size = self.cfg.get('TINY_SIZE', 32)

        if stage == 'fit' or stage is None:
            train_full = DatasetClass(split='train', **common_kwargs)
            val_full   = DatasetClass(split='val',   **common_kwargs)

            if tiny:
                print(f"[TINY] train={min(tiny_size, len(train_full))}, "
                      f"val={min(tiny_size//4, len(val_full))}")
                self.train_dataset = Subset(train_full, list(range(min(tiny_size, len(train_full)))))
                self.val_dataset   = Subset(val_full,   list(range(min(tiny_size//4, len(val_full)))))
            else:
                self.train_dataset = train_full
                self.val_dataset   = val_full

        if stage == 'test' or stage is None:
            if self.stage == 'vae':
                self.test_dataset = SignMotionDataset(split='test', **common_kwargs)
            else:
                self.test_dataset = SignText2MotionDatasetEval(split='test', **common_kwargs)

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.get('NUM_WORKERS', 4),
            collate_fn=sign_collate_simple if self.stage == 'vae' else sign_collate,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.EVAL.get('BATCH_SIZE', 32),
            shuffle=False,
            num_workers=self.cfg.EVAL.get('NUM_WORKERS', 4),
            collate_fn=sign_collate_simple if self.stage == 'vae' else sign_collate,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.TEST.get('BATCH_SIZE', 32),
            shuffle=False,
            num_workers=self.cfg.TEST.get('NUM_WORKERS', 4),
            collate_fn=sign_collate_simple if self.stage == 'vae' else sign_collate,
            drop_last=False,
            pin_memory=True,
        )

    def mm_mode(self, mm_on=True):
        """Multimodal evaluation mode toggle."""
        if mm_on:
            self.is_mm = True
            if hasattr(self.test_dataset, 'name_list'):
                self.name_list = self.test_dataset.name_list
                self.mm_list   = np.random.choice(
                    self.name_list,
                    min(self.cfg.METRIC.get('MM_NUM_SAMPLES', 100), len(self.name_list)),
                    replace=False,
                )
                self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            if hasattr(self, 'name_list'):
                self.test_dataset.name_list = self.name_list