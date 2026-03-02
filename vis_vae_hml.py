"""
vis_vae_hml.py — SignGPT3 HML VAE Reconstruction 시각화

vis_hml_preprocess.py의 visualization 코드를 그대로 사용.
좌: GT  /  우: VAE Reconstructed

Usage:
    python vis_vae_hml.py \
        --cfg  configs/sign_vae_hml.yaml \
        --ckpt experiments/motgpt/SignGPT3_vae_hml249/checkpoints/last.ckpt \
        --dataset phoenix --split val --num_samples 5 --device cuda:1
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
# Constants — vis_hml_preprocess.py 와 동일
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND      = list(range(25, 40))
SMPLX_RHAND      = list(range(40, 55))
SMPLX_VALID      = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND

SPINE3        = [9]
TARGET_JOINTS = SPINE3 + list(range(12, 22)) + list(range(25, 40)) + list(range(40, 55))  # 41개

KEY_JOINTS    = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LHAND_TIPS    = [27, 30, 33, 36, 39]
RHAND_TIPS    = [42, 45, 48, 51, 54]


# =============================================================================
# HML 249-dim → 55-joint positions  (vis_hml_preprocess.py: hml_to_joints55)
# =============================================================================

def hml_to_joints55(hml):
    """
    hml: [T, 249] denormalized
    pos_rel [3:126]: 41 target joints 상대 위치
      TARGET_JOINTS = [9, 12..21, 25..39, 40..54]
    반환: [T, 55, 3]
    """
    T       = hml.shape[0]
    N       = len(TARGET_JOINTS)              # 41
    pos_rel = hml[:, 3:3+N*3].reshape(T, N, 3)  # [T, 41, 3]
    j55     = np.zeros((T, 55, 3), dtype=np.float32)
    for li, si in enumerate(TARGET_JOINTS):
        j55[:, si] = pos_rel[:, li]
    return j55


# =============================================================================
# Skeleton — vis_hml_preprocess.py 와 동일 (hml=True 고정)
# =============================================================================

def get_connections():
    """HML TARGET joints만 사용 (spine 0,3,6 제외)"""
    upper = [
        (12, 15),
        (12, 13), (13, 16), (16, 18), (18, 20),
        (12, 14), (14, 17), (17, 19), (19, 21),
    ]
    hands = []
    for base in [25, 40]:
        root_j = 20 if base == 25 else 21
        for f in range(5):
            b = base + f * 3
            hands += [(root_j, b), (b, b+1), (b+1, b+2)]
    return upper + hands


CONNECTIONS = get_connections()


def normalize_to_neck(joints, flip_y=True):
    """neck(joint 12) 기준 상대좌표, flip_y=True"""
    root = joints[:, 12:13, :]
    out  = joints - root
    if flip_y:
        out[..., 1] *= -1
    return out


# =============================================================================
# Matplotlib elements — vis_hml_preprocess.py 와 동일
# =============================================================================

def _setup_ax(ax, label, color, xlim, ylim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.axis('off')


def _build_elements(ax, colors):
    lines = []
    for (i, j) in CONNECTIONS:
        if i >= 40 or j >= 40:   c, lw = colors['rhand'], 1.0
        elif i >= 25 or j >= 25: c, lw = colors['lhand'], 1.0
        else:                     c, lw = colors['body'],  1.5
        ln, = ax.plot([], [], color=c, lw=lw, alpha=0.8)
        lines.append((ln, i, j))
    bs = ax.scatter([], [], c=colors['body'],  s=18, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=8,  zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=8,  zorder=5)
    return lines, bs, ls, rs


def save_comparison_video(gt_j55, recon_j55, save_path,
                          title='', fps=25, viewport=0.5):
    T = min(gt_j55.shape[0], recon_j55.shape[0])

    # normalize — flip_y=False (SMPL-X y=UP, 손이 머리 위)
    lft = normalize_to_neck(gt_j55[:T].copy(),    flip_y=False)
    rgt = normalize_to_neck(recon_j55[:T].copy(), flip_y=False)

    if viewport > 0:
        xlim = (-viewport, viewport)
        ylim = (-viewport * 0.5, viewport * 2.0)
    else:
        valid = [i for i in SMPLX_VALID if i < 55]
        all_xy = np.concatenate([lft[:, valid], rgt[:, valid]], axis=0)
        ax_ = all_xy[..., 0].flatten()
        ay_ = all_xy[..., 1].flatten()
        rng  = max(ax_.max()-ax_.min(), ay_.max()-ay_.min(), 0.1) * 1.2
        xlim = ((ax_.max()+ax_.min())/2 - rng/2, (ax_.max()+ax_.min())/2 + rng/2)
        ylim = ((ay_.max()+ay_.min())/2 - rng/2, (ay_.max()+ay_.min())/2 + rng/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=9)
    colors = {'body': 'black', 'lhand': 'royalblue', 'rhand': 'crimson'}
    _setup_ax(ax_l, 'GT',            'steelblue', xlim, ylim)
    _setup_ax(ax_r, 'Reconstructed', 'firebrick', xlim, ylim)
    el_l = _build_elements(ax_l, colors)
    el_r = _build_elements(ax_r, colors)
    ftxt = fig.text(0.5, 0.02, '', ha='center', fontsize=8, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    def update(f):
        ftxt.set_text(f'frame {f}/{T-1}')
        for (lines, bs, ls, rs), data in [(el_l, lft), (el_r, rgt)]:
            fd  = data[min(f, T-1)]
            x, y = fd[:, 0], fd[:, 1]
            for (ln, i, j) in lines:
                ln.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[KEY_JOINTS],  y[KEY_JOINTS]])
            ls.set_offsets(np.c_[x[LHAND_TIPS],  y[LHAND_TIPS]])
            rs.set_offsets(np.c_[x[RHAND_TIPS],  y[RHAND_TIPS]])

    ani = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    try:
        ani.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=2000))
    except Exception:
        ani.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Model loading
# =============================================================================

def load_model(ckpt_path, cfg, device):
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model

    print(f"  Loading: {ckpt_path}")
    datamodule = build_data(cfg, phase="test")
    datamodule.setup(stage='fit')
    model = build_model(cfg, datamodule)

    ckpt   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd     = ckpt.get('state_dict', ckpt)
    result = model.load_state_dict(sd, strict=False)
    if result is not None and hasattr(result, 'missing_keys') and result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
    model.eval().to(device)

    nfeats   = datamodule.nfeats
    n_params = sum(p.numel() for p in model.vae.parameters())
    epoch, step = ckpt.get('epoch', '?'), ckpt.get('global_step', '?')
    print(f"  VAE {n_params/1e6:.2f}M  nfeats={nfeats}  epoch={epoch}  step={step}")
    return model, nfeats, datamodule


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',         required=True)
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--dataset',     default=None)
    parser.add_argument('--split',       default='val')
    parser.add_argument('--num_samples', type=int,   default=5)
    parser.add_argument('--fps',         type=int,   default=25)
    parser.add_argument('--viewport',    type=float, default=0.5, help='0=auto')
    parser.add_argument('--output',      default='vis_vae_hml_output')
    parser.add_argument('--device',      default='cuda:0')
    args = parser.parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'hml_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("SignGPT3 HML VAE Visualization")
    print("=" * 60)

    # Config
    sys.argv = [sys.argv[0], '--cfg', args.cfg, '--nodebug']
    from motGPT.config import parse_args
    cfg = parse_args(phase='test')
    if args.dataset:
        cfg.DATASET.H2S.DATASET_NAME = args.dataset

    # Model
    print("\n[1/3] Loading model...")
    model, nfeats, datamodule = load_model(args.ckpt, cfg, device)
    assert nfeats == 249, f"HML 249-dim 전용. nfeats={nfeats}"

    mean     = datamodule.mean.numpy().astype(np.float32)
    std      = datamodule.std.numpy().astype(np.float32)
    std_safe = np.clip(std, 1e-8, None)

    # Dataset
    print("\n[2/3] Loading dataset...")
    from motGPT.data.signlang.dataset_sign import SignMotionDataset

    dataset_name = args.dataset or cfg.DATASET.H2S.DATASET_NAME
    dataset = SignMotionDataset(
        data_root         = cfg.DATASET.H2S.ROOT,
        csl_root          = cfg.DATASET.H2S.get('CSL_ROOT'),
        phoenix_root      = cfg.DATASET.H2S.get('PHOENIX_ROOT'),
        dataset_name      = dataset_name,
        split             = args.split,
        mean              = torch.from_numpy(mean),
        std               = torch.from_numpy(std),
        max_motion_length = cfg.DATASET.H2S.get('MAX_MOTION_LEN', 300),
        min_motion_length = cfg.DATASET.H2S.get('MIN_MOTION_LEN', 40),
        unit_length       = cfg.DATASET.H2S.get('UNIT_LEN', 4),
        feature_type      = 'hml_style',
        hml_root          = cfg.DATASET.H2S.get('HML_ROOT'),
    )

    n       = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset)-1, n, dtype=int)
    print(f"  {dataset_name}/{args.split}: {len(dataset)} samples → {n}")

    # Inference & render
    print(f"\n[3/3] Generating...\n")
    rmse_list = []

    for ii, ds_idx in enumerate(indices):
        item = dataset[int(ds_idx)]
        if item is None:
            continue

        text  = item[0]
        gt_n  = item[1]           # [T, 249] normalized
        T_len = int(item[2])
        name  = item[3]
        src   = item[9] if len(item) > 9 else dataset_name

        # VAE forward
        with torch.no_grad():
            x    = gt_n[:T_len].unsqueeze(0).to(device)
            z, _ = model.vae.encode(x, [T_len])
            rec  = model.vae.decode(z, [T_len])   # [1, T, 249]

        # Denormalize
        gt_raw  = gt_n[:T_len].numpy()        * std_safe + mean  # [T, 249]
        rec_raw = rec[0,:T_len].cpu().numpy() * std_safe + mean  # [T, 249]

        # HML → 55-joint positions
        gt_j55   = hml_to_joints55(gt_raw)   # [T, 55, 3]
        recon_j55 = hml_to_joints55(rec_raw)  # [T, 55, 3]

        # RMSE (TARGET joints)
        diff      = gt_j55[:, TARGET_JOINTS] - recon_j55[:, TARGET_JOINTS]
        rmse_all  = float(np.sqrt(np.mean(diff**2)))
        rmse_body = float(np.sqrt(np.mean(diff[:, :10]**2)))
        rmse_hand = float(np.sqrt(np.mean(diff[:, 10:]**2)))
        rmse_list.append((rmse_all, rmse_body, rmse_hand))

        safe_name = name[:40].replace('/', '_').replace(' ', '_')
        save_path = os.path.join(out_dir, f'{ii:03d}_{src}_{safe_name}.mp4')

        title = (f'[{src}]  {name}\n'
                 f'T={T_len}   RMSE={rmse_all:.5f}  '
                 f'(body={rmse_body:.5f}  hand={rmse_hand:.5f})')
        save_comparison_video(
            gt_j55, recon_j55, save_path,
            title=title, fps=args.fps, viewport=args.viewport,
        )

        print(f"  [{ii+1}/{n}] {safe_name:<40s} "
              f"RMSE={rmse_all:.5f}  body={rmse_body:.5f}  hand={rmse_hand:.5f}")

    # Summary
    if rmse_list:
        arr = np.array(rmse_list)
        print(f"\n{'='*60}")
        print(f"Results ({n} samples)")
        print(f"  RMSE avg={arr[:,0].mean():.5f}  min={arr[:,0].min():.5f}  max={arr[:,0].max():.5f}")
        print(f"  body avg={arr[:,1].mean():.5f}   hand avg={arr[:,2].mean():.5f}")
        print(f"  Best  [{arr[:,0].argmin()}]: {arr[:,0].min():.5f}")
        print(f"  Worst [{arr[:,0].argmax()}]: {arr[:,0].max():.5f}")
        print(f"\nOutput: {out_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()