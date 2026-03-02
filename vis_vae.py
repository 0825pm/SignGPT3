"""
vis_vae.py — SignGPT3 VAE Reconstruction 시각화

Usage:
    python vis_vae.py \
        --cfg configs/sign_vae.yaml \
        --ckpt experiments/.../checkpoints/last.ckpt \
        --num_samples 10 --dataset phoenix --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND      = list(range(25, 40))
SMPLX_RHAND      = list(range(40, 55))

JOINT44_TO_SMPLX55 = (
    [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    + list(range(25, 40))
    + list(range(40, 55))
)


# =============================================================================
# Connections
# =============================================================================

def get_connections_soke():
    """
    120-dim SOKE용: 상체 + 손만 그림.

    [핵심 수정] 원본 코드의 get_connections()는 (0,3),(3,6),(6,9) spine 하체 연결선 포함.
    SMPL-X FK를 lower_body=zeros로 돌리면 joint 0~11이 T-pose 위치로 계산되어
    전신 skeleton이 생성되고, 수어에 불필요한 다리가 화면을 가득 채움.
    → spine3(9) 이상 상체만 연결.
    """
    upper = [
        (9, 12), (12, 15),                            # spine3 → neck → head
        (9, 13),  (13, 16), (16, 18), (18, 20),       # spine3 → l_collar → l_shoulder → l_elbow → l_wrist
        (9, 14),  (14, 17), (17, 19), (19, 21),       # spine3 → r_collar → r_shoulder → r_elbow → r_wrist
    ]
    lhand = []
    for fi in range(5):
        b = 25 + fi * 3
        lhand += [(20, b), (b, b+1), (b+1, b+2)]
    rhand = []
    for fi in range(5):
        b = 40 + fi * 3
        rhand += [(21, b), (b, b+1), (b+1, b+2)]
    return upper + lhand + rhand


def get_connections_full():
    """528D용: 전신 연결선 (원본 그대로)"""
    upper = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    lhand = []
    for fi in range(5):
        b = 25 + fi * 3
        lhand += [(20, b), (b, b+1), (b+1, b+2)]
    rhand = []
    for fi in range(5):
        b = 40 + fi * 3
        rhand += [(21, b), (b, b+1), (b+1, b+2)]
    return upper + lhand + rhand


# =============================================================================
# Normalization
# =============================================================================

def normalize_to_spine3(joints, flip_y=False):
    """
    spine3(joint 9) 기준 centering.
    SMPL-X FK 결과에는 joint[9]가 실제 좌표로 채워져 있으므로 유효.
    """
    if len(joints.shape) == 3:
        root = joints[:, 9:10, :]
    else:
        root = joints[9:10, :]
    out = joints - root
    if flip_y:
        out[..., 1] *= -1
    return out


# =============================================================================
# Drawing
# =============================================================================

def _setup_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')


def _build_elements(ax, J, connections, colors):
    # upper-body scatter joints (spine3 이상만)
    upper_scatter = [i for i in [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] if i < J]
    lines = []
    for (i, j) in connections:
        if i >= J or j >= J:
            continue
        if i >= 40 or j >= 40:
            c, lw = colors['rhand'], 1.0
        elif i >= 25 or j >= 25:
            c, lw = colors['lhand'], 1.0
        else:
            c, lw = colors['body'], 1.5
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.85)
        lines.append((line, i, j))
    bs = ax.scatter([], [], c=colors['body'],  s=18, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=6,  zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=6,  zorder=5)
    return lines, bs, ls, rs, upper_scatter


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='GT', right_label='Reconstructed',
                          soke_mode=True):
    """
    soke_mode=True  : 상체 전용 연결선 + spine3(9) 기준 normalize
    soke_mode=False : 전신 연결선 + spine3(9) 기준 normalize
    """
    T = min(left_joints.shape[0], right_joints.shape[0])
    J = min(left_joints.shape[1], right_joints.shape[1])

    # [FIX] spine3(9) 기준 normalize (flip 없음 - SMPL-X Y-up과 matplotlib Y-up 동일)
    left  = normalize_to_spine3(left_joints[:T, :J].copy(),  flip_y=False)
    right = normalize_to_spine3(right_joints[:T, :J].copy(), flip_y=False)

    connections = get_connections_soke() if soke_mode else get_connections_full()

    if viewport > 0:
        x_lim = (-viewport, viewport)
        # [FIX] spine3(9) 기준 SMPL-X 상체 좌표 범위:
        #   head(15):     y ≈ +0.55 ~ +0.65m  (위)
        #   shoulder:     y ≈ +0.30 ~ +0.40m
        #   wrist:        y ≈ -0.20 ~ +0.30m  (수어 동작에 따라)
        #   fingertip:    y ≈ -0.35 ~ +0.25m
        # viewport=0.5 → y_lim = (-0.7v, +1.4v) = (-0.35, +0.70) 이면 충분
        y_lim = (-viewport * 0.7, viewport * 1.4)
    else:
        valid_idx = [i for i in [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                     + list(range(25, 55)) if i < J]
        all_data = np.concatenate([left[:, valid_idx], right[:, valid_idx]], axis=0)
        all_x = all_data[:, :, 0].flatten()
        all_y = all_data[:, :, 1].flatten()
        rng   = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max()+all_x.min()) / 2
        y_mid = (all_y.max()+all_y.min()) / 2
        x_lim = (x_mid - rng/2, x_mid + rng/2)
        y_lim = (y_mid - rng/2, y_mid + rng/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)
    _setup_ax(ax_l, left_label,  'blue', x_lim, y_lim)
    _setup_ax(ax_r, right_label, 'red',  x_lim, y_lim)

    colors = {'body': 'black', 'lhand': 'red', 'rhand': 'green'}
    el_l = _build_elements(ax_l, J, connections, colors)
    el_r = _build_elements(ax_r, J, connections, colors)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (lines, bs, ls, rs, ub_scatter), data in [(el_l, left), (el_r, right)]:
            fd = data[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            ub_xy = [(x[i], y[i]) for i in ub_scatter if i < J]
            if ub_xy:
                bs.set_offsets(np.array(ub_xy))
            if J > 25:
                ls.set_offsets(np.c_[x[25:min(40,J)], y[25:min(40,J)]])
            if J > 40:
                rs.set_offsets(np.c_[x[40:min(55,J)], y[40:min(55,J)]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# 528D / 120D → Joint positions
# =============================================================================

def feats528_to_joints55(features_np, mean_np=None, std_np=None):
    T = features_np.shape[0]
    if mean_np is not None and std_np is not None:
        features_np = features_np * (std_np + 1e-10) + mean_np
    pos_44 = features_np[:, :132].reshape(T, 44, 3)
    joints = np.zeros((T, 55, 3), dtype=np.float32)
    for li, si in enumerate(JOINT44_TO_SMPLX55):
        joints[:, si, :] = pos_44[:, li, :]
    return joints


# =============================================================================
# Model Loading
# =============================================================================

def load_vae_model(ckpt_path, cfg, device):
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model

    print(f"  Loading: {ckpt_path}")
    datamodule = build_data(cfg, phase="test")
    datamodule.setup(stage='fit')
    model = build_model(cfg, datamodule)

    ckpt       = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    nfeats   = datamodule.nfeats
    n_params = sum(p.numel() for p in model.vae.parameters())
    epoch    = ckpt.get('epoch', '?')
    step     = ckpt.get('global_step', '?')
    print(f"  {type(model.vae).__name__}: {n_params/1e6:.2f}M params, "
          f"nfeats={nfeats}, epoch={epoch}, step={step}")
    return model, nfeats, datamodule


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--cfg',         required=True)
    parser.add_argument('--dataset',     default=None)
    parser.add_argument('--split',       default='val')
    parser.add_argument('--num_samples', type=int,   default=10)
    parser.add_argument('--fps',         type=int,   default=25)
    parser.add_argument('--viewport',    type=float, default=0.5,
                        help='viewport half-width in meters (0=auto-fit)')
    parser.add_argument('--output',      default='vis_vae_output')
    parser.add_argument('--device',      default='cuda:0')
    args = parser.parse_args()

    dev_str = f'cuda:{args.device}' if args.device.isdigit() else args.device
    device  = torch.device(dev_str if torch.cuda.is_available() else 'cpu')
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'vae_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("SignGPT3 VAE Reconstruction Visualization")
    print("=" * 60)

    # 1. Model
    print("\n[1/3] Loading VAE...")
    sys.argv = [sys.argv[0], '--cfg', args.cfg, '--nodebug']
    from motGPT.config import parse_args
    cfg = parse_args(phase="test")
    if args.dataset:
        cfg.DATASET.H2S.DATASET_NAME = args.dataset

    model, nfeats, datamodule = load_vae_model(args.ckpt, cfg, device)

    # 2. Stats
    print("\n[2/3] Loading data...")
    D    = nfeats
    mean = datamodule.mean
    std  = datamodule.std
    m_np = mean.numpy()
    s_np = std.numpy()
    print(f"  nfeats={D}, mean shape: {mean.shape}")

    soke_mode = (D <= 133)   # 120/133-dim = SOKE, 528D = 위치기반

    # 3. Dataset
    from motGPT.data.signlang.dataset_sign import SignMotionDataset

    dataset_name = args.dataset or cfg.DATASET.H2S.DATASET_NAME
    dataset = SignMotionDataset(
        data_root=cfg.DATASET.H2S.ROOT,
        split=args.split,
        mean=mean,
        std=std,
        dataset_name=dataset_name,
        max_motion_length=cfg.DATASET.H2S.get('MAX_MOTION_LEN', 300),
        min_motion_length=cfg.DATASET.H2S.get('MIN_MOTION_LEN', 40),
        unit_length=cfg.DATASET.H2S.get('UNIT_LEN', 4),
        csl_root=cfg.DATASET.H2S.get('CSL_ROOT', None),
        phoenix_root=cfg.DATASET.H2S.get('PHOENIX_ROOT', None),
        feature_type=cfg.DATASET.get('FEATURE_TYPE', 'soke'),
        hml_root=cfg.DATASET.H2S.get('HML_ROOT', None),
    )

    n       = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset)-1, n, dtype=int)
    print(f"  Dataset: {dataset_name}, {len(dataset)} samples, mode={'SOKE' if soke_mode else '528D'}")

    # to_joints 함수 선택
    if D >= 528:
        def to_joints(feats_norm):
            return feats528_to_joints55(feats_norm, m_np, s_np)
    else:
        from motGPT.data.H2S import feats2joints_smplx

        def to_joints(feats_norm):
            t   = torch.from_numpy(feats_norm).float().unsqueeze(0).to(device)
            m_t = torch.from_numpy(m_np).float().to(device)
            s_t = torch.from_numpy(s_np).float().to(device)
            try:
                _, joints = feats2joints_smplx(t, m_t, s_t)
                return joints.squeeze(0).cpu().numpy()   # [T, 55, 3]
            except Exception as e:
                print(f"    [WARN] SMPL-X failed: {e}. Returning zeros.")
                return np.zeros((feats_norm.shape[0], 55, 3), dtype=np.float32)

    # 4. Reconstruct & Visualize
    print(f"\n[3/3] Generating {n} reconstructions...\n")
    rmse_all, body_rmse_all, hand_rmse_all = [], [], []

    for ii, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        text    = item[0]
        gt_norm = item[1]        # [T, D] normalized tensor
        T_len   = int(item[2])
        name    = item[3]
        src     = item[9] if len(item) > 9 else 'unknown'

        with torch.no_grad():
            motion_in  = gt_norm.unsqueeze(0).float().to(device)
            z, dist    = model.vae.encode(motion_in, [T_len])
            feats_rst  = model.vae.decode(z, [T_len])
            recon_norm = feats_rst[0].cpu().numpy()

        gt_np = gt_norm.numpy()

        T          = min(T_len, recon_norm.shape[0], gt_np.shape[0])
        gt_crop    = gt_np[:T]
        recon_crop = recon_norm[:T]
        diff       = gt_crop - recon_crop

        rmse      = np.sqrt(np.mean(diff**2))
        body_rmse = np.sqrt(np.mean(diff[:, :30]   **2))
        hand_rmse = np.sqrt(np.mean(diff[:, 30:120] **2))

        rmse_all.append(rmse)
        body_rmse_all.append(body_rmse)
        hand_rmse_all.append(hand_rmse)

        gt_joints    = to_joints(gt_crop)
        recon_joints = to_joints(recon_crop)

        safe_name = str(name)[:30].replace('/', '_').replace(' ', '_')
        print(f"  [{ii+1}/{n}] {name} (T={T_len}, src={src})")
        print(f"    RMSE: total={rmse:.4f}  body={body_rmse:.4f}  hand={hand_rmse:.4f}")
        print(f"    z: mean={z.cpu().numpy().mean():.3f}, std={z.cpu().numpy().std():.3f}")
        if text:
            print(f"    text: \"{text[:60]}\"")

        path  = os.path.join(out_dir, f'{ii:03d}_{safe_name}.mp4')
        title = (f'{name} [{src}]  T={T_len}\n'
                 f'RMSE={rmse:.4f}  (body={body_rmse:.4f}, hand={hand_rmse:.4f})')
        save_comparison_video(
            gt_joints, recon_joints, path, title,
            fps=args.fps, viewport=args.viewport,
            soke_mode=soke_mode,
        )
        print(f"    → {path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary ({n} samples)")
    print(f"  RMSE:      {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f}")
    print(f"  Body RMSE: {np.mean(body_rmse_all):.4f} ± {np.std(body_rmse_all):.4f}")
    print(f"  Hand RMSE: {np.mean(hand_rmse_all):.4f} ± {np.std(hand_rmse_all):.4f}")
    print(f"\nVideos: {out_dir}")
    print("=" * 60)

    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(f"Checkpoint: {args.ckpt}\nConfig: {args.cfg}\n"
                f"Dataset: {dataset_name}/{args.split}\nSamples: {n}\n\n")
        f.write(f"RMSE:      {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f}\n")
        f.write(f"Body RMSE: {np.mean(body_rmse_all):.4f} ± {np.std(body_rmse_all):.4f}\n")
        f.write(f"Hand RMSE: {np.mean(hand_rmse_all):.4f} ± {np.std(hand_rmse_all):.4f}\n\n")
        for i, (r, b, h) in enumerate(zip(rmse_all, body_rmse_all, hand_rmse_all)):
            f.write(f"  [{i}] total={r:.4f}  body={b:.4f}  hand={h:.4f}\n")


if __name__ == '__main__':
    main()