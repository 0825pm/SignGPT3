"""
vis_hml_preprocess.py — HML 243-dim 전처리 검증 시각화
=======================================================
VAE 없이 전처리 결과만 검증:
  좌: SOKE 120-dim → SMPL-X FK → GT joint positions
  우: HML 243-dim [3:123] joint_pos_rel (전처리 결과)

두 side가 일치하면 전처리 정상.

지원: how2sign / csl / phoenix  ×  train / val / test

Usage:
    cd ~/Projects/research/SignGPT3

    # 기본 (전체 데이터셋/split × 5개)
    python vis_hml_preprocess.py --cfg configs/sign_vae.yaml

    # phoenix val만 10개
    python vis_hml_preprocess.py \
        --cfg configs/sign_vae.yaml \
        --datasets phoenix \
        --splits val \
        --num_samples 10 \
        --viewport 0.4
"""

import os, sys, argparse
import numpy as np
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Constants — vis_vae.py 와 동일
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND      = list(range(25, 40))
SMPLX_RHAND      = list(range(40, 55))
SMPLX_VALID      = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND

# HML target joints (preprocess_soke_to_hml.py 와 동일하게 유지)
SPINE3           = [9]
TARGET_JOINTS    = SPINE3 + list(range(12, 22)) + list(range(25, 40)) + list(range(40, 55))  # 41개
SHOULDER_L, SHOULDER_R = 16, 17


# =============================================================================
# Skeleton — vis_vae.py 와 동일
# =============================================================================

def get_connections(hml=False):
    """
    hml=False (GT):  spine 포함 전체 skeleton
    hml=True  (HML): TARGET_JOINTS(12~)만 사용, spine(0,3,6,9) 제외
    """
    if hml:
        # TARGET_JOINTS(12~21, 25~54)만 사용
        upper = [
            (12,15),                          # neck → head
            (12,13),(13,16),(16,18),(18,20),  # l_collar → l_shoulder → l_elbow → l_wrist
            (12,14),(14,17),(17,19),(19,21),  # r_collar → r_shoulder → r_elbow → r_wrist
        ]
    else:
        # GT: 전체 spine 포함
        upper = [
            (0,3),(3,6),(6,9),(9,12),(12,15),
            (9,13),(13,16),(16,18),(18,20),
            (9,14),(14,17),(17,19),(19,21),
        ]
    hands = []
    for base in [25, 40]:
        root_j = 20 if base == 25 else 21
        for f in range(5):
            b = base + f * 3
            hands += [(root_j, b), (b, b+1), (b+1, b+2)]
    return upper + hands


def normalize_to_root(joints, root_idx=9, flip_y=True):
    root = joints[:, root_idx:root_idx+1, :]
    out  = joints - root
    if flip_y:
        out[..., 1] *= -1
    return out


def normalize_to_neck(joints, flip_y=True):
    """GT/HML 양쪽에 동일하게 적용: neck(12)을 origin으로
    neck은 TARGET_JOINTS에 포함되므로 GT/HML 모두 올바른 값을 가짐"""
    root = joints[:, 12:13, :]   # neck
    out  = joints - root
    if flip_y:
        out[..., 1] *= -1
    return out


def _setup_ax(ax, label, color, xlim, ylim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.axis('off')


# 핵심 관절만 dot: neck, head, collar, shoulder, elbow, wrist
# GT/HML 모두 이 관절은 올바른 값을 가짐
KEY_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

def _build_elements(ax, connections, colors, hml=False):
    lines = []
    for (i, j) in connections:
        if i >= 40 or j >= 40: c, lw = colors['rhand'], 1.0
        elif i >= 25 or j >= 25: c, lw = colors['lhand'], 1.0
        else: c, lw = colors['body'], 1.5
        ln, = ax.plot([], [], color=c, lw=lw, alpha=0.8)
        lines.append((ln, i, j))
    # body: KEY_JOINTS만 (spine 0,3,6,9 제외 → 불필요한 dot 없음)
    bs = ax.scatter([], [], c=colors['body'],  s=18, zorder=5)
    # hand: finger tip만 (각 손가락 끝 5개씩)
    ls = ax.scatter([], [], c=colors['lhand'], s=8,  zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=8,  zorder=5)
    return lines, bs, ls, rs


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='SOKE-FK (GT)', right_label='HML-Restored',
                          flip_y=False):
    T   = min(left_joints.shape[0], right_joints.shape[0])
    # GT/HML 모두 neck(12) 기준 정규화
    lft = normalize_to_neck(left_joints[:T].copy(),  flip_y=flip_y)
    rgt = normalize_to_neck(right_joints[:T].copy(), flip_y=flip_y)

    if viewport > 0:
        xlim = (-viewport, viewport)
        # neck(12) 기준 정규화: neck=0, 머리=+y, 손=크게 -y
        # flip_y=True이므로 화면상 위=+y
        # 손끝까지 보이려면 아래쪽을 크게 잡아야 함
        ylim = (-viewport * 2.0, viewport * 0.5)
    else:
        valid = [i for i in SMPLX_VALID if i < 55]
        all_xy = np.concatenate([lft[:, valid], rgt[:, valid]], axis=0)
        ax_, ay_ = all_xy[..., 0].flatten(), all_xy[..., 1].flatten()
        rng = max(ax_.max()-ax_.min(), ay_.max()-ay_.min(), 0.1) * 1.2
        xlim = ((ax_.max()+ax_.min())/2 - rng/2, (ax_.max()+ax_.min())/2 + rng/2)
        ylim = ((ay_.max()+ay_.min())/2 - rng/2, (ay_.max()+ay_.min())/2 + rng/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=9)
    colors = {'body': 'black', 'lhand': 'royalblue', 'rhand': 'crimson'}
    _setup_ax(ax_l, left_label,  'steelblue', xlim, ylim)
    _setup_ax(ax_r, right_label, 'firebrick', xlim, ylim)
    conns_gt  = get_connections(hml=False)  # GT: spine 포함
    conns_hml = get_connections(hml=True)   # HML: TARGET joints만
    el_l = _build_elements(ax_l, conns_gt,  colors)
    el_r = _build_elements(ax_r, conns_hml, colors)
    ftxt    = fig.text(0.5, 0.02, '', ha='center', fontsize=8, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    def update(f):
        ftxt.set_text(f'frame {f}/{T-1}')
        # finger tips: 각 손가락 마지막 관절 (5개씩)
        LHAND_TIPS = [27, 30, 33, 36, 39]   # lhand finger tips
        RHAND_TIPS = [42, 45, 48, 51, 54]   # rhand finger tips
        for (lines, bs, ls, rs), data in [(el_l, lft), (el_r, rgt)]:
            fd  = data[min(f, T-1)]
            x, y = fd[:, 0], fd[:, 1]
            for (ln, i, j) in lines:
                ln.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[KEY_JOINTS], y[KEY_JOINTS]])
            ls.set_offsets(np.c_[x[LHAND_TIPS], y[LHAND_TIPS]])
            rs.set_offsets(np.c_[x[RHAND_TIPS], y[RHAND_TIPS]])

    ani = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    try:
        ani.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=2000))
    except Exception:
        ani.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# SMPL-X FK — 120-dim → [T, 55, 3]  (vis_vae.py 와 동일)
# =============================================================================

class SMPLXFK:
    def __init__(self, model_path, device='cpu'):
        import smplx
        self._path   = model_path
        self.device  = device
        self._models = {}
        # base model (batch=1) 미리 로드
        self._get(1)
        print(f"[FK] SMPL-X ready on {device}")

    def _get(self, B):
        if B not in self._models:
            import smplx
            self._models[B] = smplx.create(
                self._path, model_type='smplx', gender='NEUTRAL',
                use_pca=False, use_face_contour=True, batch_size=B
            ).to(self.device)
        return self._models[B]

    @torch.no_grad()
    def run(self, feat_120, chunk=64):
        T    = feat_120.shape[0]
        feat = torch.from_numpy(feat_120).float().to(self.device)
        upper = feat[:, 0:30]
        lhand = feat[:, 30:75]
        rhand = feat[:, 75:120]
        zeros33 = torch.zeros(T, 33, device=self.device)
        body_pose = torch.cat([zeros33, upper], dim=-1)
        root = torch.zeros(T, 3, device=self.device)
        jaw  = torch.zeros(T, 3, device=self.device)
        expr = torch.zeros(T, 10, device=self.device)
        betas = torch.tensor([[
            -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
             0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
        ]], device=self.device).expand(T, -1)

        out_all = []
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            B = e - s
            m = self._get(B)
            o = m(betas=betas[s:e], global_orient=root[s:e],
                  body_pose=body_pose[s:e], left_hand_pose=lhand[s:e],
                  right_hand_pose=rhand[s:e], jaw_pose=jaw[s:e],
                  leye_pose=torch.zeros(B,3,device=self.device),
                  reye_pose=torch.zeros(B,3,device=self.device),
                  expression=expr[s:e], return_verts=False)
            out_all.append(o.joints.cpu())
        return torch.cat(out_all, 0).numpy()  # [T, 55, 3]


# =============================================================================
# HML 243-dim → [T, 55, 3]
# joint_pos_rel [3:123] 은 어깨 중점 기준 상대 위치
# → 어깨(idx=SHOULDER_L/R)를 origin에 놓으면 그대로 사용 가능
# =============================================================================

def hml_to_joints55(hml):
    """
    hml: [T, 249]
    pos_rel [3:126]: 어깨 중점 기준 41 target joints 상대 위치
      TARGET_JOINTS = [9, 12..21, 25..39, 40..54]
      spine3(9) 포함으로 neck/collar 연결선 phantom 제거
    반환: [T, 55, 3]
    """
    T       = hml.shape[0]
    N       = len(TARGET_JOINTS)          # 41
    pos_rel = hml[:, 3:3+N*3].reshape(T, N, 3)  # [T, 41, 3]
    j55     = np.zeros((T, 55, 3), dtype=np.float32)
    for li, si in enumerate(TARGET_JOINTS):
        j55[:, si] = pos_rel[:, li]
    return j55


# =============================================================================
# 데이터 로딩
# =============================================================================

def get_annotations(cfg, split):
    """SignMotionDataset으로 어노테이션만 수집"""
    from motGPT.data.signlang.dataset_sign import SignMotionDataset
    dummy = torch.zeros(120)
    ds = SignMotionDataset(
        data_root     = cfg.DATASET.H2S.ROOT,
        split         = split,
        mean          = dummy,
        std           = torch.ones(120),
        dataset_name  = 'how2sign_csl_phoenix',
        csl_root      = cfg.DATASET.H2S.get('CSL_ROOT'),
        phoenix_root  = cfg.DATASET.H2S.get('PHOENIX_ROOT'),
        max_motion_length = 99999,
        min_motion_length = 1,
    )
    counts = {}
    for s in ds.all_data:
        counts[s['src']] = counts.get(s['src'], 0) + 1
    print(f"  [{split}] annotations: {counts}")
    return ds.all_data


def load_soke(sample, cfg):
    from motGPT.data.signlang.load_data import (
        load_h2s_sample, load_csl_sample, load_phoenix_sample)
    src = sample['src']
    if   src == 'how2sign': f, t, *_ = load_h2s_sample(sample, cfg.DATASET.H2S.ROOT)
    elif src == 'csl':      f, t, *_ = load_csl_sample(sample, cfg.DATASET.H2S.CSL_ROOT)
    elif src == 'phoenix':  f, t, *_ = load_phoenix_sample(sample, cfg.DATASET.H2S.PHOENIX_ROOT)
    else: return None, ''
    return f, (t or '')


def load_hml(sample, hml_root):
    path = os.path.join(hml_root, sample['src'], f"{sample['name']}.npy")
    return np.load(path) if os.path.exists(path) else None


# =============================================================================
# Main
# =============================================================================

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--cfg',          required=True)
    pa.add_argument('--hml_root',     default='/home/user/Projects/research/SOKE/data_hml')
    pa.add_argument('--datasets',     nargs='+', default=['how2sign', 'csl', 'phoenix'])
    pa.add_argument('--splits',       nargs='+', default=['train', 'val', 'test'])
    pa.add_argument('--num_samples',  type=int,   default=5)
    pa.add_argument('--fps',          type=int,   default=25)
    pa.add_argument('--viewport',     type=float, default=0.5, help='0=auto')
    pa.add_argument('--smplx_path',   default='deps/smpl_models')
    pa.add_argument('--device',       default='cuda:0')
    pa.add_argument('--output',       default='vis_hml_output')
    args = pa.parse_args()

    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, ts)
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("HML Preprocess Verification")
    print(f"hml_root : {args.hml_root}")
    print(f"datasets : {args.datasets}")
    print(f"splits   : {args.splits}")
    print("=" * 60)

    # Config
    sys.argv = [sys.argv[0], '--cfg', args.cfg, '--nodebug']
    from motGPT.config import parse_args
    cfg = parse_args(phase="test")

    # FK
    fk = SMPLXFK(args.smplx_path, device=args.device)

    all_rows = []  # 최종 요약 테이블용

    for split in args.splits:
        print(f"\n{'─'*60}")
        print(f"Split: {split}")
        annotations = get_annotations(cfg, split)

        for ds_name in args.datasets:
            samples = [s for s in annotations if s['src'] == ds_name]
            if not samples:
                print(f"  [{ds_name}] no samples"); continue

            # 균등 샘플링
            step     = max(1, len(samples) // args.num_samples)
            selected = samples[::step][:args.num_samples]

            out_dir  = os.path.join(output_root, ds_name, split)
            os.makedirs(out_dir, exist_ok=True)

            stats = []
            print(f"\n  [{ds_name}/{split}]  total={len(samples)}, showing={len(selected)}")

            for idx, sample in enumerate(selected):
                name = sample['name']

                # ── 1. SOKE 120-dim 로드 ──────────────────────────────
                feat_120, text = load_soke(sample, cfg)
                if feat_120 is None:
                    print(f"    [{idx+1}] SKIP — soke load failed: {name}"); continue

                # ── 2. HML 243-dim 로드 ──────────────────────────────
                hml_feat = load_hml(sample, args.hml_root)
                if hml_feat is None:
                    print(f"    [{idx+1}] SKIP — hml not found: {name}"); continue

                # ── 3. FK ────────────────────────────────────────────
                gt_j55  = fk.run(feat_120)          # [T,   55, 3]
                hml_j55 = hml_to_joints55(hml_feat) # [T-1, 55, 3]

                # HML은 velocity로 파생된 T-1 프레임 → GT도 맞춤
                T = min(gt_j55.shape[0] - 1, hml_j55.shape[0])
                gt_crop  = gt_j55[1:T+1]   # [T, 55, 3]
                hml_crop = hml_j55[:T]     # [T, 55, 3]

                # ── 4. RMSE (TARGET 40 joints) ────────────────────────
                diff      = gt_crop[:, TARGET_JOINTS] - hml_crop[:, TARGET_JOINTS]
                rmse      = float(np.sqrt(np.mean(diff**2)))
                body_rmse = float(np.sqrt(np.mean(diff[:, :10]**2)))
                hand_rmse = float(np.sqrt(np.mean(diff[:, 10:]**2)))
                stats.append((rmse, body_rmse, hand_rmse))

                safe = name[:40].replace('/', '_').replace(' ', '_')
                print(f"    [{idx+1:02d}/{len(selected)}] {name[:50]}")
                print(f"           T={T}  RMSE={rmse:.5f}  body={body_rmse:.5f}  hand={hand_rmse:.5f}")
                if text: print(f"           \"{text[:70]}\"")

                # ── 5. 비교 영상 ──────────────────────────────────────
                vid = os.path.join(out_dir, f'{idx:03d}_{safe}.mp4')
                title = (f'[{ds_name}/{split}]  {name}\n'
                         f'T={T}   RMSE={rmse:.5f}  (body={body_rmse:.5f}  hand={hand_rmse:.5f})')
                save_comparison_video(
                    gt_crop, hml_crop, vid, title=title,
                    fps=args.fps, viewport=args.viewport,
                    left_label='SOKE-FK  (GT)', right_label='HML pos_rel',
                )
                print(f"           → {vid}")

            # ── split×dataset 통계 ────────────────────────────────────
            if stats:
                r = np.mean([s[0] for s in stats])
                b = np.mean([s[1] for s in stats])
                h = np.mean([s[2] for s in stats])
                print(f"\n  [{ds_name}/{split}] mean  RMSE={r:.5f}  body={b:.5f}  hand={h:.5f}")
                all_rows.append(dict(ds=ds_name, split=split, n=len(stats),
                                     rmse=r, body=b, hand=h))

                # summary.txt
                with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
                    f.write(f"Dataset : {ds_name}\nSplit   : {split}\n")
                    f.write(f"Samples : {len(stats)}\n\n")
                    f.write(f"mean RMSE : {r:.6f}\n")
                    f.write(f"body RMSE : {b:.6f}\n")
                    f.write(f"hand RMSE : {h:.6f}\n\n")
                    for i,(ri,bi,hi) in enumerate(stats):
                        f.write(f"  [{i:03d}] {ri:.6f}  body={bi:.6f}  hand={hi:.6f}\n")

    # ==========================================================================
    # 전체 요약 테이블
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Split':<6} {'N':>4}  {'RMSE':>10}  {'Body':>10}  {'Hand':>10}")
    print("─" * 58)
    for row in all_rows:
        print(f"{row['ds']:<12} {row['split']:<6} {row['n']:>4}  "
              f"{row['rmse']:>10.5f}  {row['body']:>10.5f}  {row['hand']:>10.5f}")
    print(f"\nOutput: {output_root}")
    print("=" * 60)


if __name__ == '__main__':
    main()