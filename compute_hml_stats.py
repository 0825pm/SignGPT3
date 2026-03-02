"""
compute_hml_stats.py
====================
HML .npy 파일들에서 3개 데이터셋 통합 mean/std 계산

실행:
    python compute_hml_stats.py \
        --hml_root /home/user/Projects/research/SOKE/data_hml \
        --datasets how2sign csl phoenix

출력:
    data_hml/mean.npy  [249]
    data_hml/std.npy   [249]
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


def compute_stats(hml_root, datasets, splits=('train',)):
    dataset_dirs = {
        'how2sign': os.path.join(hml_root, 'how2sign'),
        'csl':      os.path.join(hml_root, 'csl'),
        'phoenix':  os.path.join(hml_root, 'phoenix'),
    }

    all_feats = []
    total_files = 0

    for ds in datasets:
        ds_dir = dataset_dirs.get(ds)
        if not ds_dir or not os.path.exists(ds_dir):
            print(f"  [SKIP] {ds}: {ds_dir} not found")
            continue

        # train split만 사용 (val/test data leakage 방지)
        # 파일이 split 서브폴더 안에 있는 경우와 직접 있는 경우 모두 처리
        npy_files = []
        for split in splits:
            split_dir = os.path.join(ds_dir, split)
            if os.path.exists(split_dir):
                found = glob(os.path.join(split_dir, '*.npy'))
                npy_files.extend(found)

        # split 서브폴더가 없으면 루트에서 직접 찾기
        if not npy_files:
            npy_files = [f for f in glob(os.path.join(ds_dir, '*.npy'))
                         if 'mean' not in os.path.basename(f)
                         and 'std' not in os.path.basename(f)]

        # mean/std 파일 제외
        npy_files = [f for f in npy_files
                     if os.path.basename(f) not in ('mean.npy', 'std.npy')]

        print(f"  {ds}: {len(npy_files)} train files")
        total_files += len(npy_files)

        for fp in tqdm(npy_files, desc=f"  {ds}", leave=False):
            try:
                feat = np.load(fp)   # [T, D]
                if feat.ndim == 2 and feat.shape[0] > 0 and feat.shape[1] > 0:
                    all_feats.append(feat)
            except Exception as e:
                print(f"    [WARN] {fp}: {e}")

    if not all_feats:
        raise RuntimeError("No HML features found.")

    print(f"\nTotal files processed: {total_files:,}")
    all_feats = np.concatenate(all_feats, axis=0)   # [N_frames, D]
    print(f"Total frames: {all_feats.shape[0]:,}")
    print(f"Feature dim:  {all_feats.shape[1]}")

    mean = all_feats.mean(axis=0).astype(np.float32)
    std  = all_feats.std(axis=0).astype(np.float32)
    std  = np.clip(std, a_min=1e-8, a_max=None)

    print(f"\nMean: min={mean.min():.4f}, max={mean.max():.4f}")
    print(f"Std:  min={std.min():.4f},  max={std.max():.4f}")

    mean_path = os.path.join(hml_root, 'mean.npy')
    std_path  = os.path.join(hml_root, 'std.npy')
    np.save(mean_path, mean)
    np.save(std_path,  std)

    print(f"\nSaved:")
    print(f"  {mean_path}")
    print(f"  {std_path}")
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hml_root',  required=True)
    parser.add_argument('--datasets',  nargs='+', default=['how2sign', 'csl', 'phoenix'])
    parser.add_argument('--splits',    nargs='+', default=['train'])
    args = parser.parse_args()

    print("=" * 60)
    print("HML Unified Mean/Std Computation")
    print("=" * 60)
    print(f"HML root: {args.hml_root}")
    print(f"Datasets: {args.datasets}")
    print(f"Splits:   {args.splits}")
    print()

    compute_stats(args.hml_root, args.datasets, splits=args.splits)
    print("\nDone. Now run:")
    print("  python train.py --cfg configs/sign_vae_hml.yaml --nodebug")


if __name__ == '__main__':
    main()