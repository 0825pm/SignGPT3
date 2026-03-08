"""
preprocess_soke_to_shml.py — SOKE 120-dim → Sign-HML 213-dim
=============================================================
preprocess_soke_to_hml.py 구조 그대로, feature 변환만 교체.

S-HML Feature 구조 (213-dim):
  [0  :33 ] body_pos   — 11 joints x 3  (neck 기준 상대 position)
  [33 :78 ] lhand_pos  — 15 joints x 3
  [78 :123] rhand_pos  — 15 joints x 3
  [123:168] lhand_vel  — 15 joints x 3  (frame-to-frame velocity)
  [168:213] rhand_vel  — 15 joints x 3

HML 249-dim 대비 제거:
  - root_vel  : 수어는 제자리 -> std~0 -> 정규화 폭발
  - body_vel  : body 거의 정지 -> noise만 추가
  - foot_contact: 수어에 무의미
  - T-1 프레임 손실 없음 (velocity 마지막 프레임 0 패딩)

사용법:
    cd ~/Projects/research/SignGPT3
    python preprocess_soke_to_shml.py \
        --cfg configs/sign_vae.yaml \
        --output_dir /home/user/Projects/research/SOKE/data_shml \
        --smplx_model_path deps/smpl_models \
        --device cuda:0 \
        --datasets how2sign csl phoenix \
        --splits train val test
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Constants
SPINE3            = [9]
UPPER_BODY_JOINTS = list(range(12, 22))
LHAND_JOINTS      = list(range(25, 40))
RHAND_JOINTS      = list(range(40, 55))
TARGET_JOINTS     = SPINE3 + UPPER_BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS  # 41
SHOULDER_L, SHOULDER_R = 16, 17

BODY_IDX  = list(range(0, 11))
LHAND_IDX = list(range(11, 26))
RHAND_IDX = list(range(26, 41))
NECK_IDX  = 1   # TARGET[1] = joint12 (neck)


class SMPLXFK:
    def __init__(self, smplx_model_path, device='cpu'):
        self.device = device
        import smplx
        self.model = smplx.create(
            smplx_model_path, model_type='smplx', gender='NEUTRAL',
            use_pca=False, use_face_contour=True, batch_size=1
        ).to(device)
        self.default_betas = torch.tensor([[
            -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
             0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
        ]], dtype=torch.float32, device=device)
        self._model_path = smplx_model_path
        print("[FK] SMPL-X loaded")

    def _init_batch_model(self, batch_size):
        if not hasattr(self, '_batch_models'):
            self._batch_models = {}
        if batch_size not in self._batch_models:
            import smplx
            self._batch_models[batch_size] = smplx.create(
                self._model_path, model_type='smplx', gender='NEUTRAL',
                use_pca=False, use_face_contour=True, batch_size=batch_size
            ).to(self.device)
        return self._batch_models[batch_size]

    @torch.no_grad()
    def forward_120(self, feat_120, chunk_size=64):
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
        joints_all  = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            B   = end - start
            m   = self._init_batch_model(B)
            out = m(
                betas=betas[start:end], global_orient=root_pose[start:end],
                body_pose=body_pose[start:end], left_hand_pose=lhand[start:end],
                right_hand_pose=rhand[start:end], jaw_pose=jaw_pose[start:end],
                leye_pose=torch.zeros(B, 3, device=self.device),
                reye_pose=torch.zeros(B, 3, device=self.device),
                expression=expr[start:end], return_verts=False
            )
            joints_all.append(out.joints.cpu())
        return torch.cat(joints_all, dim=0).numpy()  # [T, 55, 3]


def interpolate_bad_frames(joints, jump_thresh=0.15):
    """
    FK 결과 [T, 55, 3]에서 tracking 실패 프레임을 선형 보간으로 교체.
    jump_thresh: 한 프레임에 hand joint가 이동 가능한 최대 거리 (m)
                 수어 25fps 기준 최대 속도 ~3.75m/s -> 0.15m/frame
    """
    T = joints.shape[0]
    if T < 2:
        return joints
    joints = joints.copy()

    hand_idx = LHAND_JOINTS + RHAND_JOINTS          # 30 joints
    hand_pos = joints[:, hand_idx, :]               # [T, 30, 3]
    diffs     = np.linalg.norm(hand_pos[1:] - hand_pos[:-1], axis=-1)  # [T-1, 30]
    max_diffs = diffs.max(axis=-1)                  # [T-1]

    bad = np.zeros(T, dtype=bool)
    bad[1:] = max_diffs > jump_thresh

    if not bad.any():
        return joints

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return joints

    for t in np.where(bad)[0]:
        prev_good = good_idx[good_idx < t]
        next_good = good_idx[good_idx > t]
        if len(prev_good) == 0:
            joints[t] = joints[next_good[0]]
        elif len(next_good) == 0:
            joints[t] = joints[prev_good[-1]]
        else:
            t0, t1 = int(prev_good[-1]), int(next_good[0])
            alpha   = (t - t0) / (t1 - t0)
            joints[t] = (1 - alpha) * joints[t0] + alpha * joints[t1]

    return joints


def joints_to_shml(joints):
    """[T, 55, 3] -> [T, 213]"""
    T      = joints.shape[0]
    target = joints[:, TARGET_JOINTS, :]
    neck   = target[:, NECK_IDX:NECK_IDX + 1, :]
    pos    = target - neck
    body_pos  = pos[:, BODY_IDX,  :]
    lhand_pos = pos[:, LHAND_IDX, :]
    rhand_pos = pos[:, RHAND_IDX, :]
    lhand_vel = np.concatenate([
        lhand_pos[1:] - lhand_pos[:-1],
        np.zeros((1, 15, 3), dtype=np.float32)], axis=0)
    rhand_vel = np.concatenate([
        rhand_pos[1:] - rhand_pos[:-1],
        np.zeros((1, 15, 3), dtype=np.float32)], axis=0)
    feat = np.concatenate([
        body_pos.reshape(T, -1),
        lhand_pos.reshape(T, -1),
        rhand_pos.reshape(T, -1),
        lhand_vel.reshape(T, -1),
        rhand_vel.reshape(T, -1),
    ], axis=-1)
    return feat.astype(np.float32)


def load_cfg(cfg_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)
    default_path = os.path.join(os.path.dirname(cfg_path), 'default.yaml')
    if os.path.exists(default_path):
        default = OmegaConf.load(default_path)
        cfg = OmegaConf.merge(default, cfg)
    return cfg


def process_split(target_datasets, split, cfg, fk, output_dir):
    from motGPT.data.signlang.load_data import (
        load_h2s_sample, load_csl_sample, load_phoenix_sample)
    from motGPT.data.signlang.dataset_sign import SignMotionDataset

    h2s_root     = cfg.DATASET.H2S.ROOT
    csl_root     = cfg.DATASET.H2S.get('CSL_ROOT', None)
    phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)

    dummy_mean = torch.zeros(120)
    dummy_std  = torch.ones(120)
    ds = SignMotionDataset(
        data_root=h2s_root, split=split,
        mean=dummy_mean, std=dummy_std,
        dataset_name='how2sign_csl_phoenix',
        csl_root=csl_root, phoenix_root=phoenix_root,
        max_motion_length=99999, min_motion_length=1,
    )

    src_counts = {}
    for sample in ds.all_data:
        src_counts[sample['src']] = src_counts.get(sample['src'], 0) + 1
    print(f"  Loaded annotations: {src_counts}")

    samples = [s for s in ds.all_data if s['src'] in target_datasets]
    print(f"  Processing {len(samples)} samples for {target_datasets}")

    for ds_name in target_datasets:
        os.makedirs(os.path.join(output_dir, ds_name, split), exist_ok=True)

    success, fail, skip = 0, 0, 0
    missing = []

    for sample in tqdm(samples, desc=f"{target_datasets}/{split}"):
        src  = sample['src']
        name = sample['name']
        out_path = os.path.join(output_dir, src, f"{name}.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            skip += 1
            continue

        if src == 'how2sign':
            feat_120, _, _, _ = load_h2s_sample(sample, h2s_root)
        elif src == 'csl':
            feat_120, _, _, _ = load_csl_sample(sample, csl_root)
        elif src == 'phoenix':
            feat_120, _, _, _ = load_phoenix_sample(sample, phoenix_root)
        else:
            fail += 1; continue

        if feat_120 is None:
            missing.append((src, name)); fail += 1; continue
        if len(feat_120) < 5:
            fail += 1; continue

        try:
            joints = fk.forward_120(feat_120)
        except Exception as e:
            print(f"\n  FK failed [{name}]: {e}")
            fail += 1; continue

        joints    = interpolate_bad_frames(joints)
        shml_feat = joints_to_shml(joints)
        if len(shml_feat) < 4:
            fail += 1; continue

        np.save(out_path, shml_feat)
        success += 1

    print(f"\n  [{target_datasets}/{split}] success={success}, fail={fail}, skip={skip}")

    if missing:
        print(f"\n  WARNING load_*_sample returned None: {len(missing)} samples")
        roots = {'how2sign': h2s_root, 'csl': csl_root, 'phoenix': phoenix_root}
        for src, name in missing[:10]:
            base = roots.get(src, '')
            candidates = [
                os.path.join(base, split, 'poses', name),
                os.path.join(base, 'poses', name),
                os.path.join(base, name),
            ]
            found = [p for p in candidates if os.path.exists(p)]
            status = f"found at: {found[0]}" if found else "NOT FOUND"
            print(f"    [{src}] {name} -> {status}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")


def compute_mean_std(output_dir, splits=('train',)):
    all_feats = []

    # 1) split 서브폴더 탐색 (phoenix처럼 train/ 안에 있는 경우)
    found_in_split = False
    for split in splits:
        feat_dir = os.path.join(output_dir, split)
        if not os.path.exists(feat_dir):
            continue
        npy_files = []
        for root, dirs, files in os.walk(feat_dir):
            for f in files:
                if f.endswith('.npy'):
                    npy_files.append(os.path.join(root, f))
        if not npy_files:
            continue
        found_in_split = True
        for fpath in tqdm(npy_files, desc=f"mean/std [{split}]"):
            all_feats.append(np.load(fpath))

    # 2) flat 구조 탐색 (how2sign/csl처럼 서브폴더 없이 바로 .npy가 있는 경우)
    if not found_in_split:
        npy_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith('.npy') and f not in ('Mean.npy', 'Std.npy')
        ]
        if npy_files:
            print(f"  [mean/std] flat 구조 감지 ({len(npy_files)}개)")
            for fpath in tqdm(npy_files, desc="mean/std [flat]"):
                all_feats.append(np.load(fpath))

    if not all_feats:
        print(f"  [mean/std] no data in {output_dir}, skipping")
        return

    all_feats  = np.concatenate(all_feats, axis=0)  # [N, 213]
    mean       = all_feats.mean(axis=0).astype(np.float32)
    std        = all_feats.std(axis=0).astype(np.float32)

    # near-zero std clip: p10 (하위 10%) 값으로 clip
    # → median * 0.01 보다 훨씬 보수적, 실제 분포를 반영
    std_p10 = float(np.percentile(std[std > 1e-8], 10))
    std     = np.maximum(std, std_p10)

    np.save(os.path.join(output_dir, 'Mean.npy'), mean)
    np.save(os.path.join(output_dir, 'Std.npy'),  std)

    sample_n = (all_feats[:100] - mean) / (std + 1e-8)
    std_median = float(np.median(std))
    print(f"\n  [mean/std] saved -> {output_dir}")
    print(f"  dim={mean.shape[0]}, total_frames={len(all_feats)}")
    print(f"  std: min={std.min():.5f}  median={std_median:.5f}  max={std.max():.5f}")
    print(f"  norm check: range=[{sample_n.min():.2f}, {sample_n.max():.2f}]  std={sample_n.std():.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',              default='configs/sign_vae.yaml')
    parser.add_argument('--output_dir',       default='/home/user/Projects/research/SOKE/data_shml')
    parser.add_argument('--smplx_model_path', default='deps/smpl_models')
    parser.add_argument('--device',           default='cuda:0')
    parser.add_argument('--datasets',  nargs='+', default=['how2sign', 'csl', 'phoenix'])
    parser.add_argument('--splits',    nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    fk  = SMPLXFK(args.smplx_model_path, device=args.device)

    print("=" * 60)
    print("S-HML: SOKE 120-dim -> FK -> Sign-HML 213-dim")
    print(f"  datasets : {args.datasets}")
    print(f"  splits   : {args.splits}")
    print(f"  output   : {args.output_dir}")
    print("=" * 60)

    for split in args.splits:
        process_split(args.datasets, split, cfg, fk, args.output_dir)

    train_included = 'train' in args.splits
    mean_splits = ['train'] if train_included else list(args.splits)
    for ds in args.datasets:
        ds_output = os.path.join(args.output_dir, ds)
        compute_mean_std(ds_output, splits=mean_splits)

    print("\n=== Done ===")
    print(f"Output: {args.output_dir}")
    print("\nNext: update configs/data/h2s_light.yaml")
    for ds in args.datasets:
        print(f"  [{ds}] hml_dir -> {os.path.join(args.output_dir, ds)}")
    print("  motion_dim: 213")