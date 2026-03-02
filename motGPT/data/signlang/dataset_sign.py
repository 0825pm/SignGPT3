"""
SOKE-style Sign Language Dataset

변경점:
  - feature_type: 'soke' | 'hml_style' 파라미터 추가
  - hml_root: HML 전처리 결과 경로 파라미터 추가
  - _load_hml_feature(): HML 249-dim npy 직접 로드
  - _load_soke_feature(): SOKE 120-dim 로드 분리
  - _adjust_length(): 공통 길이 조정 메서드
  - nfeats: feature_type에 따라 자동 결정
"""
import os
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm

from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_iso_sample


bad_how2sign_ids = [
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front',
    '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front',
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front',
    '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front',
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front',
    'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front',
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front',
    'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front',
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front',
    '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front',
    'g3Cc_1-V31U_12-3-rgb_front'
]


class SignMotionDataset(Dataset):
    """
    Sign Motion Dataset for VAE training

    feature_type='soke'      : SOKE 120-dim axis-angle feature (기존)
    feature_type='hml_style' : HML 249-dim position feature (npy 직접 로드)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name='how2sign_csl_phoenix',
        max_motion_length=300,
        min_motion_length=40,
        unit_length=4,
        fps=25,
        csl_root=None,
        phoenix_root=None,
        feature_type='soke',   # 'soke' | 'hml_style'
        hml_root=None,         # HML 전처리 결과 루트 (feature_type='hml_style' 시 필요)
        **kwargs
    ):
        self.dataset_name  = dataset_name
        self.root_dir      = data_root
        self.csl_root      = csl_root
        self.phoenix_root  = phoenix_root
        self.split         = split
        self.feature_type  = feature_type
        self.hml_root      = hml_root

        self.unit_length       = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length

        # nfeats: feature_type에 따라 결정
        if feature_type == 'hml_style':
            self.nfeats = len(mean)   # cfg NFEATS=249
        else:
            self.nfeats = 120

        # mean/std 길이 맞춤
        self.mean = mean[:self.nfeats] if len(mean) > self.nfeats else mean
        self.std  = std[:self.nfeats]  if len(std)  > self.nfeats else std

        self.all_data    = []
        self.h2s_len     = 0
        self.csl_len     = 0
        self.phoenix_len = 0

        self._load_annotations()

        print(f'Data loading done. '
              f'All: {len(self.all_data)}, '
              f'How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')

    # -------------------------------------------------------------------------
    # Annotation 로딩
    # -------------------------------------------------------------------------

    def _load_annotations(self):
        split = self.split

        # How2Sign
        if 'how2sign' in self.dataset_name and self.root_dir:
            csv_path = os.path.join(self.root_dir, split, 'preprocessed_fps.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.root_dir, split, 're_aligned',
                                        f'how2sign_realigned_{split}_preprocessed_fps.csv')
            if os.path.exists(csv_path):
                csv = pd.read_csv(csv_path)
                csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
                csv = csv[csv['DURATION'] < 30].reset_index(drop=True)
                ids = csv['SENTENCE_NAME']
                print(f'{split}--loading how2sign annotations... {len(ids)}')
                for idx in tqdm(range(len(ids)), desc='How2Sign', leave=False):
                    name = ids[idx]
                    if name in bad_how2sign_ids:
                        continue
                    self.all_data.append({
                        'name':  name,
                        'fps':   csv[csv['SENTENCE_NAME'] == name]['fps'].item(),
                        'text':  csv[csv['SENTENCE_NAME'] == name]['SENTENCE'].item(),
                        'src':   'how2sign',
                        'split': split,
                    })
                self.h2s_len = len(self.all_data)

        # CSL-Daily
        if 'csl' in self.dataset_name and self.csl_root:
            ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            if os.path.exists(ann_path):
                try:
                    with gzip.open(ann_path, 'rb') as f:
                        ann = pickle.load(f)
                    print(f'{split}--loading csl annotations... {len(ann)}')
                    for idx in tqdm(range(len(ann)), desc='CSL-Daily', leave=False):
                        item = deepcopy(ann[idx])
                        item['src'] = 'csl'
                        self.all_data.append(item)
                    self.csl_len = len(ann)
                except Exception as e:
                    print(f'Failed to load CSL: {e}')

        # Phoenix-2014T
        if 'phoenix' in self.dataset_name and self.phoenix_root:
            ann_path = os.path.join(
                self.phoenix_root,
                'phoenix14t.dev' if split == 'val' else f'phoenix14t.{split}'
            )
            if os.path.exists(ann_path):
                try:
                    with gzip.open(ann_path, 'rb') as f:
                        ann = pickle.load(f)
                    print(f'{split}--loading phoenix annotations... {len(ann)}')
                    for idx in tqdm(range(len(ann)), desc='Phoenix', leave=False):
                        item = deepcopy(ann[idx])
                        item['src'] = 'phoenix'
                        self.all_data.append(item)
                    self.phoenix_len = len(ann)
                except Exception as e:
                    print(f'Failed to load Phoenix: {e}')

    def __len__(self):
        return len(self.all_data)

    # -------------------------------------------------------------------------
    # Feature 로딩 헬퍼
    # -------------------------------------------------------------------------

    def _load_hml_feature(self, sample):
        """HML 249-dim npy 직접 로드"""
        name = sample['name']
        src  = sample['src']
        path = os.path.join(self.hml_root, src, f"{name}.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32), sample.get('text', '')
        return None, sample.get('text', '')

    def _load_soke_feature(self, sample):
        """SOKE 120-dim 로드 (기존 방식)"""
        src = sample['src']
        if src == 'how2sign':
            feat, text, _, _ = load_h2s_sample(sample, self.root_dir)
        elif src == 'csl':
            feat, text, _, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            feat, text, _, _ = load_phoenix_sample(sample, self.phoenix_root)
        else:
            feat, text = None, ''
        return feat, (text or '')

    def _adjust_length(self, clip_poses):
        """시퀀스 길이를 [min, max] 범위로 조정"""
        m_length = clip_poses.shape[0]
        if m_length < self.min_motion_length:
            idx_arr    = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        elif m_length > self.max_motion_length:
            idx_arr    = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        else:
            m_length   = (m_length // self.unit_length) * self.unit_length
            start_idx  = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[start_idx:start_idx + m_length]
        return clip_poses

    # -------------------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------------------

    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src    = sample['src']
        name   = sample['name']

        # Feature 로딩
        if self.feature_type == 'hml_style':
            clip_poses, text = self._load_hml_feature(sample)
            if clip_poses is None:
                clip_poses = np.zeros((self.min_motion_length, self.nfeats), dtype=np.float32)
        else:
            clip_poses, text = self._load_soke_feature(sample)
            if clip_poses is None:
                clip_poses = np.zeros((self.min_motion_length, self.nfeats), dtype=np.float32)
                text = ''

        # Normalize
        clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)

        # Length 조정
        clip_poses = self._adjust_length(clip_poses)
        m_length   = clip_poses.shape[0]

        return text, torch.from_numpy(clip_poses).float(), m_length, name, \
               None, None, None, None, None, src


class SignText2MotionDataset(SignMotionDataset):
    """Sign Text-to-Motion Dataset for LM training"""

    def __init__(self, max_text_len=40, **kwargs):
        super().__init__(**kwargs)
        self.max_text_len = max_text_len


class SignText2MotionDatasetEval(SignText2MotionDataset):
    """Evaluation dataset"""

    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src    = sample['src']
        name   = sample['name']

        # Feature 로딩
        if self.feature_type == 'hml_style':
            clip_poses, text = self._load_hml_feature(sample)
        else:
            if src == 'how2sign':
                clip_poses, text, name, _ = load_h2s_sample(sample, self.root_dir)
            elif src == 'csl':
                clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
            elif src == 'phoenix':
                clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
            else:
                clip_poses, text = None, ''

        if clip_poses is None:
            clip_poses = np.zeros((self.min_motion_length, self.nfeats), dtype=np.float32)
            text = ''

        all_captions = [text] * 3

        clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)
        clip_poses = self._adjust_length(clip_poses)
        m_length   = clip_poses.shape[0]

        return text, torch.from_numpy(clip_poses).float(), m_length, name, \
               None, None, None, all_captions, None, src