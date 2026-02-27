"""
vis_vae.py — SignGPT3 VAE Reconstruction 시각화

원본: sign-t2m vis_vae.py
변경: 모델 로딩(MotGPT), VAE forward(encode+decode), import 경로(motGPT.*), dataset 반환형식(tuple)

Usage:
    cd ~/Projects/research/SignGPT3

    python vis_vae.py \
        --cfg configs/sign_vae.yaml \
        --ckpt experiments/.../checkpoints/last.ckpt \
        --num_samples 10

    # Phoenix만
    python vis_vae.py \
        --cfg configs/sign_vae.yaml \
        --ckpt experiments/.../last.ckpt \
        --dataset phoenix --viewport 0
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
# Constants (원본 그대로)
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND

# 44 joints → SMPLX 55 index mapping
JOINT44_TO_SMPLX55 = (
    [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    + list(range(25, 40))
    + list(range(40, 55))
)


# =============================================================================
# Skeleton Visualization (원본 그대로)
# =============================================================================

def get_connections():
    upper_body = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    hand_connections = []
    for finger in range(5):
        base = 25 + finger * 3
        hand_connections.extend([(20, base), (base, base + 1), (base + 1, base + 2)])
    for finger in range(5):
        base = 40 + finger * 3
        hand_connections.extend([(21, base), (base, base + 1), (base + 1, base + 2)])
    return upper_body + hand_connections


def normalize_to_root(joints, root_idx=9, flip_y=True):
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx + 1, :]
    else:
        root = joints[root_idx:root_idx + 1, :]
    out = joints - root
    if flip_y:
        out[..., 1] *= -1
    return out


def _setup_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')


def _build_elements(ax, J, connections, colors):
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]
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
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))
    bs = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
    return lines, bs, ls, rs, ub_idx


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='GT', right_label='Reconstructed',
                          flip_y=True):
    T = min(left_joints.shape[0], right_joints.shape[0])
    J = min(left_joints.shape[1], right_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    left = normalize_to_root(left_joints[:T, :J].copy(), root_idx, flip_y=flip_y)
    right = normalize_to_root(right_joints[:T, :J].copy(), root_idx, flip_y=flip_y)

    valid_idx = [i for i in SMPLX_VALID if i < J]

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_data = np.concatenate([left[:, valid_idx], right[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range / 2, x_mid + max_range / 2)
        y_lim = (y_mid - max_range / 2, y_mid + max_range / 2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)

    _setup_ax(ax_l, left_label, 'blue', x_lim, y_lim)
    _setup_ax(ax_r, right_label, 'red', x_lim, y_lim)

    connections = get_connections()
    colors_l = {'body': 'black', 'lhand': 'red', 'rhand': 'green'}
    colors_r = {'body': 'black', 'lhand': 'red', 'rhand': 'green'}

    el_l = _build_elements(ax_l, J, connections, colors_l)
    el_r = _build_elements(ax_r, J, connections, colors_r)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T - 1}')
        for (lines, bs, ls, rs, ub_idx), data in [(el_l, left), (el_r, right)]:
            fd = data[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
            if J > 25:
                ls.set_offsets(np.c_[x[25:min(40, J)], y[25:min(40, J)]])
            if J > 40:
                rs.set_offsets(np.c_[x[40:min(55, J)], y[40:min(55, J)]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# 528D / 120D → Joint positions (원본 그대로)
# =============================================================================

def feats528_to_joints55(features_np, mean_np=None, std_np=None):
    T = features_np.shape[0]
    if mean_np is not None and std_np is not None:
        features_np = features_np * (std_np + 1e-10) + mean_np
    pos_44 = features_np[:, :132].reshape(T, 44, 3)
    joints = np.zeros((T, 55, 3), dtype=np.float32)
    for local_idx, smplx_idx in enumerate(JOINT44_TO_SMPLX55):
        joints[:, smplx_idx, :] = pos_44[:, local_idx, :]
    return joints


def feats120_to_joints55(features_np):
    T, D = features_np.shape
    joints = np.zeros((T, 55, 3), dtype=np.float32)
    if D >= 120:
        joints[:, 12:22, :] = features_np[:, 0:30].reshape(T, 10, 3)
        joints[:, 25:40, :] = features_np[:, 30:75].reshape(T, 15, 3)
        joints[:, 40:55, :] = features_np[:, 75:120].reshape(T, 15, 3)
    return joints


# =============================================================================
# Model Loading — [변경] SignGPT3 (MotGPT)
# =============================================================================

def load_vae_model(ckpt_path, cfg, device):
    """Load MotGPT VAE from SignGPT3 checkpoint."""
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model

    print(f"  Loading: {ckpt_path}")

    datamodule = build_data(cfg, phase="test")
    datamodule.setup(stage='fit')
    model = build_model(cfg, datamodule)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    nfeats = datamodule.nfeats
    n_params = sum(p.numel() for p in model.vae.parameters())
    epoch = ckpt.get('epoch', '?')
    step = ckpt.get('global_step', '?')
    vae_type = type(model.vae).__name__
    print(f"  {vae_type}: {n_params / 1e6:.2f}M, nfeats={nfeats}, epoch={epoch}, step={step}")

    return model, nfeats, datamodule


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SignGPT3 VAE Reconstruction Visualization')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cfg', required=True, help='SignGPT3 config yaml')
    # Dataset
    parser.add_argument('--dataset', default=None,
                        help='Override dataset (phoenix/csl/how2sign/how2sign_csl_phoenix)')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=10)
    # Vis
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5,
                        help='fixed viewport (0=auto)')
    parser.add_argument('--output', default='vis_vae_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    dev_str = f'cuda:{args.device}' if args.device.isdigit() else args.device
    device = torch.device(dev_str if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'vae_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("SignGPT3 VAE Reconstruction Visualization")
    print("=" * 60)

    # =========================================================================
    # 1. Load Model — [변경] parse_args + build_model
    # =========================================================================
    print("\n[1/3] Loading VAE...")

    sys.argv = [sys.argv[0], '--cfg', args.cfg, '--nodebug']
    from motGPT.config import parse_args
    cfg = parse_args(phase="test")

    if args.dataset:
        cfg.DATASET.H2S.DATASET_NAME = args.dataset

    model, nfeats, datamodule = load_vae_model(args.ckpt, cfg, device)

    # =========================================================================
    # 2. Load normalization stats — [변경] datamodule에서 가져옴
    # =========================================================================
    print("\n[2/3] Loading data...")
    D = nfeats

    mean = datamodule.mean
    std = datamodule.std

    print(f"  nfeats={D}, mean shape: {mean.shape}")

    # =========================================================================
    # 3. Load dataset — [변경] motGPT import + tuple 반환
    # =========================================================================
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
    )

    n = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    print(f"  Dataset: {dataset_name}, {len(dataset)} samples, visualizing {n}")

    # =========================================================================
    # 4. Reconstruct & Visualize
    # =========================================================================
    print(f"\n[3/3] Generating reconstructions...\n")

    rmse_all_list, body_list, hand_list = [], [], []
    m_np = mean.numpy()
    s_np = std.numpy()

    # feats→joints (원본 로직 그대로, import 경로만 변경 + CPU 강제로 device mismatch 방지)
    if D >= 528:
        def to_joints(feats_norm, m_np, s_np):
            return feats528_to_joints55(feats_norm, m_np, s_np)
    else:
        # [변경] motGPT import + 모든 텐서를 CPU로 (SMPL-X 모델이 CPU에 있으므로)
        from motGPT.data.H2S import feats2joints_smplx

        def to_joints(feats_norm, m_np, s_np):
            smplx_dev = torch.device('cuda:0')
            t = torch.from_numpy(feats_norm).float().unsqueeze(0).to(smplx_dev)
            m_t = torch.from_numpy(m_np).float().to(smplx_dev)
            s_t = torch.from_numpy(s_np).float().to(smplx_dev)
            try:
                _, joints = feats2joints_smplx(t, m_t, s_t)
                return joints.squeeze(0).cpu().numpy()[:, :55, :]
            except Exception as e:
                print(f"    Warning: SMPL-X forward failed: {e}")
                raw = feats_norm * (s_np + 1e-10) + m_np
                return feats120_to_joints55(raw)

    for idx_i, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        # [변경] SignGPT3 dataset returns tuple, not dict
        text = item[0]
        gt_norm = item[1]               # [T, D] tensor
        T_len = int(item[2])
        name = item[3]
        src = item[9] if len(item) > 9 else 'unknown'

        # [변경] VAE forward: encode + decode
        with torch.no_grad():
            motion_in = gt_norm.unsqueeze(0).float().to(device)  # [1, T, D]
            z, dist = model.vae.encode(motion_in, [T_len])
            feats_rst = model.vae.decode(z, [T_len])
            recon_norm = feats_rst[0].cpu().numpy()              # [T, D]

        gt_np = gt_norm.numpy()

        # ---- Metrics (원본 그대로) ----
        T = min(T_len, recon_norm.shape[0], gt_np.shape[0])
        gt_crop = gt_np[:T]
        recon_crop = recon_norm[:T]
        diff = gt_crop - recon_crop

        rmse = np.sqrt(np.mean(diff ** 2))

        if D >= 528:
            body_rmse = np.sqrt(np.mean(diff[:, :42] ** 2))
            hand_rmse = np.sqrt(np.mean(diff[:, 42:132] ** 2))
        else:
            body_rmse = np.sqrt(np.mean(diff[:, :30] ** 2))
            hand_rmse = np.sqrt(np.mean(diff[:, 30:120] ** 2))

        rmse_all_list.append(rmse)
        body_list.append(body_rmse)
        hand_list.append(hand_rmse)

        # ---- Joints (원본 그대로) ----
        gt_joints = to_joints(gt_crop, m_np, s_np)
        recon_joints = to_joints(recon_crop, m_np, s_np)

        z_np = z.cpu().numpy()

        safe_name = str(name)[:30].replace('/', '_').replace(' ', '_')
        print(f"  [{idx_i + 1}/{n}] {name} (T={T_len}, src={src})")
        print(f"    RMSE: total={rmse:.4f}  body={body_rmse:.4f}  hand={hand_rmse:.4f}")
        print(f"    z: mean={z_np.mean():.3f}, std={z_np.std():.3f}")
        if text:
            print(f"    text: \"{text[:60]}\"")

        path = os.path.join(output_root, f'{idx_i:03d}_{safe_name}.mp4')
        title = (f'{name} [{src}] T={T_len}\n'
                 f'RMSE={rmse:.4f} (body={body_rmse:.4f}, hand={hand_rmse:.4f})')
        save_comparison_video(
            gt_joints, recon_joints, path, title,
            args.fps, args.viewport,
            flip_y=(D >= 528),  # 528D=Z-up→flip, 120D=SMPLX Y-up→no flip
        )
        print(f"    → {path}")

    # =========================================================================
    # Summary (원본 그대로)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Summary ({n} samples)")
    print(f"  RMSE:      {np.mean(rmse_all_list):.4f} ± {np.std(rmse_all_list):.4f}")
    print(f"  Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}")
    print(f"  Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}")
    print(f"\nVideos: {output_root}")
    print("=" * 60)

    with open(os.path.join(output_root, 'summary.txt'), 'w') as f:
        f.write(f"VAE Reconstruction Summary\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Config: {args.cfg}\n")
        f.write(f"Dataset: {dataset_name} / {args.split}\n")
        f.write(f"Samples: {n}\n\n")
        f.write(f"RMSE:      {np.mean(rmse_all_list):.4f} ± {np.std(rmse_all_list):.4f}\n")
        f.write(f"Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}\n")
        f.write(f"Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}\n\n")
        for i, (r, b, h) in enumerate(zip(rmse_all_list, body_list, hand_list)):
            f.write(f"  [{i}] total={r:.4f}  body={b:.4f}  hand={h:.4f}\n")


if __name__ == '__main__':
    main()