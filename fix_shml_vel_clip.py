"""
fix_shml_vel_clip.py — S-HML velocity spike 제거 + Mean/Std 재계산
"""
import os
import argparse
import numpy as np
from tqdm import tqdm


def fix_dataset(ds_dir, vel_clip, dry_run=False):
    # 모든 npy 수집 (flat + 서브폴더)
    npy_files = []
    for root, dirs, files in os.walk(ds_dir):
        for f in files:
            if f.endswith('.npy') and f not in ('Mean.npy', 'Std.npy'):
                npy_files.append(os.path.join(root, f))

    print(f"  총 {len(npy_files)}개 파일  vel_clip={vel_clip}")

    clip_count = 0
    for fpath in tqdm(npy_files, desc="clip"):
        d = np.load(fpath)
        if d.ndim != 2 or d.shape[1] != 213:
            continue
        before = d[:, 123:].copy()
        d[:, 123:] = np.clip(d[:, 123:], -vel_clip, vel_clip)
        if np.any(before != d[:, 123:]):
            clip_count += 1
        if not dry_run:
            np.save(fpath, d)

    pct = 100.0 * clip_count / max(len(npy_files), 1)
    print(f"  clip 적용 파일: {clip_count} / {len(npy_files)}  ({pct:.1f}%)")

    if dry_run:
        return

    # Mean/Std 재계산
    # train 서브폴더 우선, 없으면 flat
    train_feats = []
    train_dir = os.path.join(ds_dir, 'train')
    if os.path.isdir(train_dir):
        for root, dirs, files in os.walk(train_dir):
            for f in files:
                if f.endswith('.npy'):
                    train_feats.append(np.load(os.path.join(root, f)))
    else:
        for f in os.listdir(ds_dir):
            if f.endswith('.npy') and f not in ('Mean.npy', 'Std.npy'):
                train_feats.append(np.load(os.path.join(ds_dir, f)))

    if not train_feats:
        print("  Mean/Std 계산할 데이터 없음")
        return

    cat  = np.concatenate(train_feats, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std  = cat.std(axis=0).astype(np.float32)
    std_p10 = float(np.percentile(std[std > 1e-8], 10))
    std  = np.maximum(std, std_p10)

    np.save(os.path.join(ds_dir, 'Mean.npy'), mean)
    np.save(os.path.join(ds_dir, 'Std.npy'),  std)

    sample_n = (cat[:100] - mean) / (std + 1e-8)
    print(f"  Mean/Std 저장 완료")
    print(f"  std: min={std.min():.5f}  median={np.median(std):.5f}  max={std.max():.5f}")
    print(f"  norm check: range=[{sample_n.min():.2f}, {sample_n.max():.2f}]  std={sample_n.std():.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',     default='/home/user/Projects/research/SOKE/data_shml')
    parser.add_argument('--datasets', nargs='+', default=['how2sign', 'csl', 'phoenix'])
    parser.add_argument('--vel_clip', type=float, default=None,
                        help='공통 clip 값. 미지정시 데이터셋별 기본값')
    parser.add_argument('--dry_run',  action='store_true')
    args = parser.parse_args()

    DEFAULT_CLIP = {'how2sign': 0.5, 'csl': 0.6, 'phoenix': 0.6}

    for ds in args.datasets:
        ds_dir   = os.path.join(args.root, ds)
        if not os.path.isdir(ds_dir):
            print(f"[{ds}] 경로 없음: {ds_dir}")
            continue
        clip_val = args.vel_clip if args.vel_clip is not None else DEFAULT_CLIP.get(ds, 0.5)
        print(f"\n=== {ds}  (vel_clip={clip_val}) ===")
        fix_dataset(ds_dir, clip_val, dry_run=args.dry_run)

    print("\n=== Done ===")