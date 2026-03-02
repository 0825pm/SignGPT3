"""
SOKE 120-dim → HumanML3D-style 243-dim 전처리 스크립트
======================================================
기존 load_*_sample 을 그대로 활용하여 학습 시와 동일한 방식으로
데이터를 로드한 뒤 FK → HML feature 변환.

변환 결과 feature 구조 (243-dim):
    [0:3]       root_vel        어깨 중점 이동 속도 (3)
    [3:123]     joint_pos_rel   어깨 기준 상대 관절 위치 (40 joints × 3)
    [123:243]   joint_vel       관절 속도 (40 joints × 3)

사용법:
    cd ~/Projects/research/SignGPT3
    python preprocess_soke_to_hml.py \
        --cfg configs/sign_vae.yaml \
        --output_dir /home/user/Projects/research/SOKE/data_hml \
        --smplx_model_path deps/smpl_models \
        --device cuda:1
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 상수: HML-style feature 구조
# ─────────────────────────────────────────────────────────────────────────────
# spine3(9): neck(12)/left_collar(13)/right_collar(14)의 부모 관절
# SOKE 120-dim에서 lower body와 함께 제거됐지만, FK output에는 존재함
# TARGET에 없으면 복원 시 joint[9]=(0,0,0) → skeleton 연결선에 phantom 발생
SPINE3            = [9]
UPPER_BODY_JOINTS = list(range(12, 22))   # 10 joints (neck~wrist)
LHAND_JOINTS      = list(range(25, 40))   # 15 joints
RHAND_JOINTS      = list(range(40, 55))   # 15 joints
TARGET_JOINTS     = SPINE3 + UPPER_BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS  # 41 joints
SHOULDER_L, SHOULDER_R = 16, 17


# ─────────────────────────────────────────────────────────────────────────────
# SMPL-X FK
# ─────────────────────────────────────────────────────────────────────────────

class SMPLXFK:
    def __init__(self, smplx_model_path, device='cpu'):
        self.device = device
        import smplx
        self.model = smplx.create(
            smplx_model_path,
            model_type='smplx',
            gender='NEUTRAL',
            use_pca=False,
            use_face_contour=True,
            batch_size=1
        ).to(device)
        self.default_betas = torch.tensor([[
            -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
             0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
        ]], dtype=torch.float32, device=device)
        self._model_path = smplx_model_path
        print(f"[FK] SMPL-X loaded")

    def _init_batch_model(self, batch_size):
        """batch_size에 맞는 SMPL-X 모델 캐싱"""
        if not hasattr(self, '_batch_models'):
            self._batch_models = {}
        if batch_size not in self._batch_models:
            import smplx
            self._batch_models[batch_size] = smplx.create(
                self._model_path,
                model_type='smplx',
                gender='NEUTRAL',
                use_pca=False,
                use_face_contour=True,
                batch_size=batch_size
            ).to(self.device)
        return self._batch_models[batch_size]

    @torch.no_grad()
    def forward_120(self, feat_120, chunk_size=64):
        """
        Args:
            feat_120: np.ndarray [T, 120]
            chunk_size: 한번에 처리할 프레임 수 (GPU 메모리에 맞게 조정)
        Returns:
            joints: np.ndarray [T, 55, 3]
        """
        T    = feat_120.shape[0]
        feat = torch.from_numpy(feat_120).float().to(self.device)

        upper_body  = feat[:, 0:30]
        lhand       = feat[:, 30:75]
        rhand       = feat[:, 75:120]
        lower_zeros = torch.zeros(T, 33, device=self.device)
        body_pose   = torch.cat([lower_zeros, upper_body], dim=-1)
        root_pose   = torch.zeros(T, 3,  device=self.device)
        jaw_pose    = torch.zeros(T, 3,  device=self.device)
        expr        = torch.zeros(T, 10, device=self.device)
        betas       = self.default_betas.expand(T, -1)

        joints_all = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            B   = end - start
            m   = self._init_batch_model(B)
            out = m(
                betas=betas[start:end],
                global_orient=root_pose[start:end],
                body_pose=body_pose[start:end],
                left_hand_pose=lhand[start:end],
                right_hand_pose=rhand[start:end],
                jaw_pose=jaw_pose[start:end],
                leye_pose=torch.zeros(B, 3, device=self.device),
                reye_pose=torch.zeros(B, 3, device=self.device),
                expression=expr[start:end],
                return_verts=False
            )
            joints_all.append(out.joints.cpu())  # [B, 55, 3]

        return torch.cat(joints_all, dim=0).numpy()  # [T, 55, 3]


# ─────────────────────────────────────────────────────────────────────────────
# HML-style feature 조립
# ─────────────────────────────────────────────────────────────────────────────

def joints_to_hml_style(joints):
    """[T, 55, 3] → [T-1, 249]

    [0:3]     root_vel      어깨 중점 이동 속도 (3)
    [3:126]   joint_pos_rel 어깨 기준 상대 위치 (41j x 3 = 123)
    [126:249] joint_vel     관절 속도 (41j x 3 = 123)

    spine3(9) 포함 41 joints → phantom dot/line 제거
    """
    N = len(TARGET_JOINTS)  # 41
    root      = (joints[:, SHOULDER_L] + joints[:, SHOULDER_R]) / 2.0
    target    = joints[:, TARGET_JOINTS]
    pos_rel   = target - root[:, None, :]

    root_vel  = root[1:] - root[:-1]
    joint_vel = pos_rel[1:] - pos_rel[:-1]
    pos_rel   = pos_rel[1:]

    feat = np.concatenate([
        root_vel,
        pos_rel.reshape(-1, N * 3),
        joint_vel.reshape(-1, N * 3),
    ], axis=-1)  # [T-1, 249]
    return feat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Config 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(cfg_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)
    default_path = os.path.join(os.path.dirname(cfg_path), 'default.yaml')
    if os.path.exists(default_path):
        default = OmegaConf.load(default_path)
        cfg = OmegaConf.merge(default, cfg)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 메인 전처리: 기존 로더 활용
# ─────────────────────────────────────────────────────────────────────────────

def process_split(target_datasets, split, cfg, fk, output_dir):
    """
    전체 데이터셋을 한번에 로드 후 src별로 필터링.
    target_datasets: list of 'how2sign' | 'csl' | 'phoenix'
    """
    from motGPT.data.signlang.load_data import (
        load_h2s_sample, load_csl_sample, load_phoenix_sample
    )
    from motGPT.data.signlang.dataset_sign import SignMotionDataset

    h2s_root     = cfg.DATASET.H2S.ROOT
    csl_root     = cfg.DATASET.H2S.get('CSL_ROOT', None)
    phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)

    # 전체 데이터셋 한번에 로드 (학습 시와 동일)
    full_name = '_'.join(['how2sign', 'csl', 'phoenix'])
    dummy_mean = torch.zeros(120)
    dummy_std  = torch.ones(120)
    ds = SignMotionDataset(
        data_root=h2s_root,
        split=split,
        mean=dummy_mean,
        std=dummy_std,
        dataset_name=full_name,
        csl_root=csl_root,
        phoenix_root=phoenix_root,
        max_motion_length=99999,
        min_motion_length=1,
    )

    # src별 카운트 출력
    src_counts = {}
    for sample in ds.all_data:
        src_counts[sample['src']] = src_counts.get(sample['src'], 0) + 1
    print(f"  Loaded annotations: {src_counts}")

    # target_datasets만 처리
    samples = [s for s in ds.all_data if s['src'] in target_datasets]
    print(f"  Processing {len(samples)} samples for {target_datasets}")

    # 출력 디렉토리 (src별 분리)
    for ds_name in target_datasets:
        os.makedirs(os.path.join(output_dir, ds_name, split), exist_ok=True)

    success, fail, skip = 0, 0, 0
    missing = []

    for sample in tqdm(samples, desc=f"{target_datasets}/{split}"):
        src  = sample['src']
        name = sample['name']
        # phoenix처럼 name에 'train/filename' 형식의 경로가 포함된 경우
        # → output_dir/phoenix/train/filename.npy 로 저장 (name 구조 그대로 유지)
        out_path = os.path.join(output_dir, src, f"{name}.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            skip += 1
            continue

        # 기존 로더 사용 (학습 시와 동일한 경로 탐색)
        if src == 'how2sign':
            feat_120, _, _, _ = load_h2s_sample(sample, h2s_root)
        elif src == 'csl':
            feat_120, _, _, _ = load_csl_sample(sample, csl_root)
        elif src == 'phoenix':
            feat_120, _, _, _ = load_phoenix_sample(sample, phoenix_root)
        else:
            fail += 1
            continue

        if feat_120 is None:
            missing.append((src, name))
            fail += 1
            continue

        if len(feat_120) < 5:
            fail += 1
            continue

        try:
            joints  = fk.forward_120(feat_120)
        except Exception as e:
            print(f"\n  FK failed [{name}]: {e}")
            fail += 1
            continue

        hml_feat = joints_to_hml_style(joints)
        if len(hml_feat) < 4:
            fail += 1
            continue

        np.save(out_path, hml_feat)
        success += 1

    print(f"\n  [{target_datasets}/{split}] success={success}, fail={fail}, skip={skip}")

    # ── 왜 없는지 진단 ────────────────────────────────────────────────────
    if missing:
        print(f"\n  ⚠ load_*_sample이 None을 반환한 샘플: {len(missing)}개")
        roots = {
            'how2sign': h2s_root,
            'csl':      csl_root,
            'phoenix':  phoenix_root,
        }
        for src, name in missing[:10]:
            base = roots.get(src, '')
            candidates = [
                os.path.join(base, split, 'poses', name),
                os.path.join(base, 'poses', name),
                os.path.join(base, name),
            ]
            found = [p for p in candidates if os.path.exists(p)]
            status = f"found at: {found[0]}" if found else "NOT FOUND in any candidate path"
            print(f"    [{src}] {name} → {status}")
        if len(missing) > 10:
            print(f"    ... 및 {len(missing)-10}개 더")


def compute_mean_std(output_dir, splits=('train',)):
    """
    output_dir 하위 split 디렉토리를 재귀 탐색하여 모든 .npy 수집.
    파일이 없으면 skip (에러 없음).
    """
    all_feats = []
    for split in splits:
        feat_dir = os.path.join(output_dir, split)
        if not os.path.exists(feat_dir):
            continue
        # 재귀 탐색 (phoenix처럼 name에 경로가 포함된 경우 대비)
        npy_files = []
        for root, dirs, files in os.walk(feat_dir):
            for f in files:
                if f.endswith('.npy'):
                    npy_files.append(os.path.join(root, f))
        if not npy_files:
            print(f"  [mean/std] no files in {feat_dir}, skipping")
            continue
        for fpath in tqdm(npy_files, desc=f"mean/std [{split}]"):
            all_feats.append(np.load(fpath))

    if not all_feats:
        print(f"  [mean/std] no data found in {output_dir} for splits={splits}, skipping")
        return

    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0).astype(np.float32)
    std  = all_feats.std(axis=0).astype(np.float32)

    np.save(os.path.join(output_dir, 'mean.npy'), mean)
    np.save(os.path.join(output_dir, 'std.npy'),  std)
    print(f"\n[mean/std] saved → {output_dir}")
    print(f"  dim={mean.shape[0]}, frames={len(all_feats)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',              default='configs/sign_vae.yaml')
    parser.add_argument('--output_dir',       default='/home/user/Projects/research/SOKE/data_hml')
    parser.add_argument('--smplx_model_path', default='deps/smpl_models')
    parser.add_argument('--device',           default='cuda:0')
    parser.add_argument('--datasets',         nargs='+', default=['how2sign', 'csl', 'phoenix'])
    parser.add_argument('--splits',           nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    fk  = SMPLXFK(args.smplx_model_path, device=args.device)

    # 전체 데이터 한번에 로드 후 src 필터링
    for split in args.splits:
        process_split(args.datasets, split, cfg, fk, args.output_dir)

    # 데이터셋별 mean/std 계산 (처리된 split에서만)
    train_included = 'train' in args.splits
    mean_splits = ['train'] if train_included else list(args.splits)
    for ds in args.datasets:
        ds_output = os.path.join(args.output_dir, ds)
        compute_mean_std(ds_output, splits=mean_splits)

    print("\n=== 전처리 완료 ===")
    print(f"출력: {args.output_dir}")