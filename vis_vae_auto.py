"""
SignGPT3 VAE Reconstruction Visualization
For MotionGPT3's Continuous MldVae and MultiPartVae

Features:
- Supports all splits: train, val, test
- Supports all datasets: how2sign, csl, phoenix
- Continuous VAE (MldVae) with KL divergence
- MultiPartVae with part-specific latents
- Multiple visualization modes: 2-panel, 3-view
- Detailed metrics with JSON export

Usage:
    # Basic usage (default: 2 samples per dataset per split)
    python vis_vae.py --cfg configs/sign_vae.yaml --output vis_output --nodebug

    # Custom options
    python vis_vae.py --cfg configs/sign_vae.yaml --output vis_output \\
        --num_samples 3 --splits val,test --datasets how2sign,csl

    # Custom checkpoint
    python vis_vae.py --cfg configs/sign_vae.yaml --checkpoint path/to/ckpt.ckpt

    # 3-view videos (front, side, top)
    python vis_vae.py --cfg configs/sign_vae.yaml --three_view
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =============================================================================
# SOKE 133 dims format constants
# =============================================================================
SOKE_BODY_DIM = 30      # upper body (10 joints × 3)
SOKE_LHAND_DIM = 45     # left hand (15 joints × 3)
SOKE_RHAND_DIM = 45     # right hand (15 joints × 3)
# SOKE_JAW_DIM = 3        # jaw (1 joint × 3)
# SOKE_EXPR_DIM = 10      # expression
SOKE_TOTAL_DIM = 120
POSE_SCALE = 2.0


# =============================================================================
# Skeleton Visualization Utilities
# =============================================================================

def get_connections(num_joints):
    """Get skeleton connections for upper body + hands."""
    upper_body = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    
    hand_connections = []
    if num_joints >= 55:
        for finger in range(5):
            base = 25 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([
                    (20, base), (base, base + 1), (base + 1, base + 2)
                ])
        for finger in range(5):
            base = 40 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([
                    (21, base), (base, base + 1), (base + 1, base + 2)
                ])
    
    all_connections = upper_body + hand_connections
    return [(i, j) for i, j in all_connections if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    """Normalize joints relative to root joint."""
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx+1, :]
    else:
        root = joints[root_idx:root_idx+1, :]
    return joints - root


def get_joint_colors():
    return {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}


# =============================================================================
# Video Saving Functions
# =============================================================================

def save_comparison_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save GT vs Reconstruction comparison video (2-panel)."""
    seqs = [gt_joints, recon_joints]
    panel_titles = ['Ground Truth', 'Reconstruction']
    
    T = min(seq.shape[0] for seq in seqs)
    J = gt_joints.shape[1]
    
    root_idx = 9 if J > 21 else 0
    normalized_seqs = [normalize_to_root(seq.copy(), root_idx) for seq in seqs]
    
    all_joints = np.concatenate(normalized_seqs, axis=0)
    
    if J >= 55:
        upper_body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        hand_idx = list(range(25, min(55, J)))
        valid_idx = upper_body_idx + hand_idx
    else:
        valid_idx = list(range(min(22, J)))
    
    all_x = all_joints[:, valid_idx, 0].flatten()
    all_y = all_joints[:, valid_idx, 1].flatten()
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    max_range = max(x_max - x_min, y_max - y_min) * 1.2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    
    x_lim = (x_mid - max_range/2, x_mid + max_range/2)
    y_lim = (y_mid - max_range/2, y_mid + max_range/2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if title:
        fig.suptitle(title, fontsize=10)
    
    for ax, panel_title in zip(axes, panel_titles):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
    
    connections = get_connections(J)
    colors = get_joint_colors()
    
    all_lines = []
    all_scatters = []
    
    for ax in axes:
        lines = []
        for (i, j) in connections:
            if i >= 40 or j >= 40:
                color = colors['rhand']
                lw = 1.0
            elif i >= 25 or j >= 25:
                color = colors['lhand']
                lw = 1.0
            else:
                color = colors['body']
                lw = 1.5
            line, = ax.plot([], [], color=color, linewidth=lw, alpha=0.8)
            lines.append((line, i, j))
        all_lines.append(lines)
        
        body_scatter = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
        lhand_scatter = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
        rhand_scatter = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
        all_scatters.append((body_scatter, lhand_scatter, rhand_scatter))
    
    plt.tight_layout()
    
    upper_body_idx = [i for i in [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] if i < J]
    
    def update(frame):
        for panel_idx, (seq, lines, scatters) in enumerate(zip(normalized_seqs, all_lines, all_scatters)):
            frame_data = seq[min(frame, len(seq)-1)]
            x, y = frame_data[:, 0], frame_data[:, 1]
            
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            
            body_scatter, lhand_scatter, rhand_scatter = scatters
            body_scatter.set_offsets(np.c_[x[upper_body_idx], y[upper_body_idx]])
            
            if J > 25:
                lhand_scatter.set_offsets(np.c_[x[25:40], y[25:40]])
            if J > 40:
                rhand_scatter.set_offsets(np.c_[x[40:55], y[40:55]])
        return []
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
    except Exception as e:
        print(f"    FFMpeg error: {e}, trying GIF...")
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=min(fps, 10))
    
    plt.close(fig)


def save_three_view_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save 3-view (front, side, top) comparison video."""
    T = min(gt_joints.shape[0], recon_joints.shape[0])
    J = gt_joints.shape[1]
    
    root_idx = 9 if J > 21 else 0
    gt_norm = normalize_to_root(gt_joints.copy(), root_idx)
    recon_norm = normalize_to_root(recon_joints.copy(), root_idx)
    
    views = [('Front', 0, 1), ('Side', 2, 1), ('Top', 0, 2)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if title:
        fig.suptitle(title, fontsize=11)
    
    bounds = {}
    for view_name, xi, yi in views:
        all_data = np.concatenate([gt_norm, recon_norm], axis=0)
        all_x = all_data[:, :, xi].flatten()
        all_y = all_data[:, :, yi].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        bounds[view_name] = {
            'xlim': (x_mid - max_range/2, x_mid + max_range/2),
            'ylim': (y_mid - max_range/2, y_mid + max_range/2),
        }
    
    connections = get_connections(J)
    row_titles = ['Ground Truth', 'Reconstruction']
    all_elements = []
    
    for row, (seq, row_title) in enumerate(zip([gt_norm, recon_norm], row_titles)):
        for col, (view_name, xi, yi) in enumerate(views):
            ax = axes[row, col]
            b = bounds[view_name]
            ax.set_xlim(b['xlim'])
            ax.set_ylim(b['ylim'])
            ax.set_aspect('equal')
            ax.axis('off')
            
            if row == 0:
                ax.set_title(view_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, row_title, transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            
            lines = []
            for (i, j) in connections:
                if i >= 40 or j >= 40:
                    color = 'green'
                elif i >= 25 or j >= 25:
                    color = 'red'
                else:
                    color = 'blue'
                line, = ax.plot([], [], color=color, linewidth=1.2, alpha=0.8)
                lines.append((line, i, j))
            
            scatter = ax.scatter([], [], c='black', s=5, zorder=5)
            all_elements.append((seq, lines, scatter, xi, yi))
    
    plt.tight_layout()
    
    def update(frame):
        for (seq, lines, scatter, xi, yi) in all_elements:
            frame_data = seq[min(frame, len(seq)-1)]
            x, y = frame_data[:, xi], frame_data[:, yi]
            
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            
            scatter.set_offsets(np.c_[x, y])
        return []
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer)
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    
    plt.close(fig)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(gt_feats, recon_feats, gt_joints=None, recon_joints=None):
    """Compute reconstruction metrics."""
    metrics = {}
    
    # Feature space metrics
    metrics['feat_mse'] = float(((gt_feats - recon_feats) ** 2).mean())
    metrics['feat_rmse'] = float(np.sqrt(metrics['feat_mse']))
    metrics['feat_l1'] = float(np.abs(gt_feats - recon_feats).mean())
    
    # Part-wise feature metrics (SOKE 133-dim)
    if gt_feats.shape[-1] == 133:
        # body: [0:30], lhand: [30:75], rhand: [75:120], face: [120:133]
        metrics['feat_rmse_body'] = float(np.sqrt(((gt_feats[:, 0:30] - recon_feats[:, 0:30]) ** 2).mean()))
        metrics['feat_rmse_lhand'] = float(np.sqrt(((gt_feats[:, 30:75] - recon_feats[:, 30:75]) ** 2).mean()))
        metrics['feat_rmse_rhand'] = float(np.sqrt(((gt_feats[:, 75:120] - recon_feats[:, 75:120]) ** 2).mean()))
        metrics['feat_rmse_face'] = float(np.sqrt(((gt_feats[:, 120:133] - recon_feats[:, 120:133]) ** 2).mean()))
    elif gt_feats.shape[-1] == 120:
        # body: [0:30], lhand: [30:75], rhand: [75:120]
        metrics['feat_rmse_body'] = float(np.sqrt(((gt_feats[:, 0:30] - recon_feats[:, 0:30]) ** 2).mean()))
        metrics['feat_rmse_lhand'] = float(np.sqrt(((gt_feats[:, 30:75] - recon_feats[:, 30:75]) ** 2).mean()))
        metrics['feat_rmse_rhand'] = float(np.sqrt(((gt_feats[:, 75:120] - recon_feats[:, 75:120]) ** 2).mean()))
    # Joint space metrics
    if gt_joints is not None and recon_joints is not None:
        min_len = min(len(gt_joints), len(recon_joints))
        gt_j = gt_joints[:min_len]
        recon_j = recon_joints[:min_len]
        
        diff = gt_j - recon_j
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        metrics['mpjpe'] = float(dist.mean())
        
        J = gt_j.shape[1]
        if J >= 55:
            body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            metrics['mpjpe_body'] = float(dist[:, body_idx].mean())
            metrics['mpjpe_lhand'] = float(dist[:, 25:40].mean())
            metrics['mpjpe_rhand'] = float(dist[:, 40:55].mean())
        
        gt_vel = np.diff(gt_j, axis=0)
        recon_vel = np.diff(recon_j, axis=0)
        metrics['vel_error'] = float(np.abs(gt_vel - recon_vel).mean())
        
        gt_vel_mag = np.abs(gt_vel).mean()
        recon_vel_mag = np.abs(recon_vel).mean()
        metrics['vel_ratio'] = float(recon_vel_mag / (gt_vel_mag + 1e-8))
    
    return metrics


# =============================================================================
# VAE Forward Functions (MldVae & MultiPartVae)
# =============================================================================

def mldvae_forward(model, feats_ref, length):
    """
    Forward pass through MldVae or MultiPartVae.
    
    MldVae uses:
    - encode(features, lengths) → (z, dist)
    - decode(z, lengths) → features
    
    MultiPartVae uses:
    - encode(features, lengths) → (z_dict, dist_dict)
    - decode(z_dict, lengths) → features
    
    Returns:
        feats_pred: reconstructed features
        vae_info: dict with mu, var statistics
    """
    # Check if MultiPartVae (has 'parts' attribute)
    is_multipart = hasattr(model.vae, 'parts')
    
    if is_multipart:
        # MultiPartVae: returns dict
        z_dict, dist_dict = model.vae.encode(feats_ref[:, :length], [length])
        feats_pred = model.vae.decode(z_dict, [length])
        
        # Get distribution info for each part
        vae_info = {
            'latent': {
                'type': 'multipart',
                'parts': {}
            }
        }
        
        total_mu_mean = 0
        total_var_mean = 0
        
        for part in model.vae.parts:
            dist = dist_dict[part]
            z = z_dict[part]
            
            part_info = {
                'mu_mean': float(dist.loc.mean().item()),
                'mu_std': float(dist.loc.std().item()),
                'var_mean': float(dist.scale.mean().item()),
                'var_std': float(dist.scale.std().item()),
                'z_norm': float(z.norm().item()),
            }
            vae_info['latent']['parts'][part] = part_info
            
            total_mu_mean += part_info['mu_mean']
            total_var_mean += part_info['var_mean']
        
        # Average across parts for compatibility
        num_parts = len(model.vae.parts)
        vae_info['latent']['mu_mean'] = total_mu_mean / num_parts
        vae_info['latent']['mu_std'] = np.mean([vae_info['latent']['parts'][p]['mu_std'] for p in model.vae.parts])
        vae_info['latent']['var_mean'] = total_var_mean / num_parts
        vae_info['latent']['var_std'] = np.mean([vae_info['latent']['parts'][p]['var_std'] for p in model.vae.parts])
        vae_info['latent']['z_norm'] = sum(vae_info['latent']['parts'][p]['z_norm'] for p in model.vae.parts)
        
    else:
        # Standard MldVae
        z, dist = model.vae.encode(feats_ref[:, :length], [length])
        feats_pred = model.vae.decode(z, [length])
        
        vae_info = {
            'latent': {
                'type': 'mldvae',
                'mu_mean': float(dist.loc.mean().item()),
                'mu_std': float(dist.loc.std().item()),
                'var_mean': float(dist.scale.mean().item()),
                'var_std': float(dist.scale.std().item()),
                'z_norm': float(z.norm().item()),
            }
        }
    
    return feats_pred, vae_info


# =============================================================================
# Dataset Source Detection
# =============================================================================

def normalize_source_name(src):
    """Normalize dataset source name."""
    src_lower = src.lower() if src else 'unknown'
    
    if 'how2sign' in src_lower or 'h2s' in src_lower:
        return 'how2sign'
    elif 'csl' in src_lower:
        return 'csl'
    elif 'phoenix' in src_lower:
        return 'phoenix'
    else:
        return src_lower


# =============================================================================
# Main Visualization Function
# =============================================================================

def main():
    # =========================
    # Parse Arguments
    # =========================
    parser = argparse.ArgumentParser(
        description='SignGPT3 VAE Reconstruction Visualization',
        add_help=False
    )
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of samples per dataset per split (default: 2)')
    parser.add_argument('--output', type=str, default='vis_output',
                       help='Output directory (default: vis_output)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS (default: 25)')
    parser.add_argument('--three_view', action='store_true',
                       help='Generate 3-view videos instead of 2-panel')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Override checkpoint path')
    parser.add_argument('--splits', type=str, default='val,test',
                       help='Comma-separated splits to process (default: val,test)')
    parser.add_argument('--datasets', type=str, default='how2sign,csl,phoenix',
                       help='Comma-separated datasets (default: how2sign,csl,phoenix)')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video generation (metrics only)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    custom_args, remaining = parser.parse_known_args()
    
    # Set CUDA device
    if custom_args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(custom_args.gpu)
    
    # Replace sys.argv for motGPT's parse_args
    sys.argv = [sys.argv[0]] + remaining
    
    # Import motGPT modules (SignGPT3)
    from motGPT.config import parse_args
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model
    from motGPT.utils.load_checkpoint import load_pretrained_vae
    
    # Parse config
    cfg = parse_args(phase="test")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = custom_args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed
    seed = cfg.SEED_VALUE if hasattr(cfg, 'SEED_VALUE') else 42
    pl.seed_everything(seed)
    
    # Parse splits and datasets
    splits_to_process = [s.strip() for s in custom_args.splits.split(',')]
    datasets_to_process = [d.strip().lower() for d in custom_args.datasets.split(',')]
    
    # =========================
    # Print Configuration
    # =========================
    print(f"\n{'='*70}")
    print("SignGPT3 VAE Reconstruction Visualization")
    print(f"{'='*70}")
    print(f"Config: {cfg.NAME if hasattr(cfg, 'NAME') else 'Unknown'}")
    print(f"Output: {output_dir}")
    print(f"Samples per dataset per split: {custom_args.num_samples}")
    print(f"Splits: {splits_to_process}")
    print(f"Datasets: {datasets_to_process}")
    print(f"Video mode: {'3-view' if custom_args.three_view else '2-panel'}")
    print(f"FPS: {custom_args.fps}")
    
    # =========================
    # Build DataModule
    # =========================
    print("\n[1/3] Loading datamodule...")
    datamodule = build_data(cfg, phase="test")
    datamodule.setup(stage='test')
    
    # =========================
    # Build Model
    # =========================
    print("[2/3] Loading model...")
    model = build_model(cfg, datamodule)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    
    print(f"  Device: {device}")
    
    # Load weights
    def load_checkpoint(path):
        state_dict = torch.load(path, map_location=device, weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        return state_dict
    
    if custom_args.checkpoint:
        print(f"  Loading checkpoint: {custom_args.checkpoint}")
        state_dict = load_checkpoint(custom_args.checkpoint)
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(cfg.TRAIN, 'PRETRAINED_VAE') and cfg.TRAIN.PRETRAINED_VAE:
        print(f"  Loading VAE: {cfg.TRAIN.PRETRAINED_VAE}")
        load_pretrained_vae(cfg, model, logger=None)
    elif hasattr(cfg.TEST, 'CHECKPOINTS') and cfg.TEST.CHECKPOINTS:
        print(f"  Loading checkpoint: {cfg.TEST.CHECKPOINTS}")
        state_dict = load_checkpoint(cfg.TEST.CHECKPOINTS)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("  Warning: No checkpoint loaded, using random weights")
    
    model.eval()
    model.to(device)
    
    # =========================
    # Print Model Info
    # =========================
    is_multipart = hasattr(model.vae, 'parts')
    vae_type = 'MultiPartVae' if is_multipart else 'MldVae'
    
    print(f"\n[Model Info]")
    print(f"  VAE type: {vae_type}")
    if hasattr(model.vae, 'latent_dim'):
        print(f"  Latent dim: {model.vae.latent_dim}")
    if hasattr(model.vae, 'nfeats'):
        print(f"  Input features: {model.vae.nfeats}")
    if is_multipart:
        print(f"  Parts: {model.vae.parts}")
        print(f"  Part latent dims: {model.vae.part_latent_dims}")
    
    # Get feats2joints converter
    feats2joints = None
    if hasattr(datamodule, 'feats2joints'):
        feats2joints = datamodule.feats2joints
        print(f"  feats2joints: Available")
    else:
        print(f"  feats2joints: Not available (will use feature visualization)")
    
    # =========================
    # Process Each Split
    # =========================
    print(f"\n[3/3] Processing samples...")
    
    all_metrics = []
    global_sample_idx = 0
    
    for split in splits_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        # Get dataloader
        try:
            if split == 'train':
                datamodule.setup(stage='fit')
                dataloader = datamodule.train_dataloader()
            elif split == 'val':
                datamodule.setup(stage='fit')
                dataloader = datamodule.val_dataloader()
            else:
                datamodule.setup(stage='test')
                dataloader = datamodule.test_dataloader()
        except Exception as e:
            print(f"  Warning: Could not load {split} dataloader: {e}")
            continue
        
        # Create output subdirectory
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Track collected samples
        collected = {ds: 0 for ds in datasets_to_process}
        target_per_dataset = custom_args.num_samples
        
        def all_collected():
            return all(collected[ds] >= target_per_dataset for ds in datasets_to_process)
        
        for batch in tqdm(dataloader, desc=f"{split}"):
            if all_collected():
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            feats_ref = batch['motion']
            lengths = batch['length']
            names = batch.get('name', [f'sample_{i}' for i in range(len(lengths))])
            srcs = batch.get('src', ['how2sign'] * len(lengths))
            B = feats_ref.shape[0]
            
            with torch.no_grad():
                for i in range(B):
                    if all_collected():
                        break
                    
                    length = lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i]
                    if length == 0:
                        continue
                    
                    name = names[i] if names else f'sample_{global_sample_idx}'
                    src = srcs[i] if srcs else 'unknown'
                    
                    src_key = normalize_source_name(src)
                    
                    if src_key not in datasets_to_process:
                        continue
                    if collected[src_key] >= target_per_dataset:
                        continue
                    
                    # Forward through VAE (MldVae or MultiPartVae)
                    feats_pred, vae_info = mldvae_forward(model, feats_ref[i:i+1], length)
                    
                    # Convert to joints
                    has_joints = False
                    gt_joints_np = None
                    recon_joints_np = None
                    
                    if feats2joints is not None:
                        try:
                            gt_input = feats_ref[i:i+1, :length].to(device)
                            recon_input = feats_pred.to(device)
                            
                            # Handle both single and tuple return values
                            gt_result = feats2joints(gt_input)
                            recon_result = feats2joints(recon_input)
                            
                            # feats2joints may return (vertices, joints) or just joints
                            if isinstance(gt_result, tuple):
                                gt_joints = gt_result[-1]  # joints is usually last
                                recon_joints = recon_result[-1]
                            else:
                                gt_joints = gt_result
                                recon_joints = recon_result
                            
                            gt_joints = gt_joints * POSE_SCALE
                            recon_joints = recon_joints * POSE_SCALE
                            gt_joints_np = gt_joints.cpu().numpy()
                            recon_joints_np = recon_joints.cpu().numpy()
                            
                            if gt_joints_np.ndim == 4:
                                gt_joints_np = gt_joints_np[0]
                                recon_joints_np = recon_joints_np[0]
                            
                            has_joints = True
                        except Exception as e:
                            if global_sample_idx == 0:
                                print(f"    Note: feats2joints failed ({e})")
                            has_joints = False
                    
                    # Get feature arrays
                    gt_feats_np = feats_ref[i, :length].cpu().numpy()
                    recon_feats_np = feats_pred[0].cpu().numpy()
                    
                    # Compute metrics
                    metrics = compute_metrics(gt_feats_np, recon_feats_np, gt_joints_np, recon_joints_np)
                    metrics['name'] = name
                    metrics['src'] = src_key
                    metrics['split'] = split
                    metrics['length'] = int(length)
                    metrics['vae_info'] = vae_info
                    all_metrics.append(metrics)
                    
                    # Print sample info
                    print(f"\n[{split}/{src_key}] {name}")
                    print(f"  Length: {length} frames")
                    print(f"  Feature RMSE: {metrics['feat_rmse']:.6f}")
                    
                    # Print part-wise feature RMSE if available
                    if 'feat_rmse_body' in metrics:
                        print(f"  Feature RMSE - Body: {metrics['feat_rmse_body']:.6f}, "
                              f"LHand: {metrics['feat_rmse_lhand']:.6f}, "
                              f"RHand: {metrics['feat_rmse_rhand']:.6f}")
                    
                    if has_joints and 'mpjpe' in metrics:
                        print(f"  MPJPE: {metrics['mpjpe']:.4f}")
                        if 'mpjpe_body' in metrics:
                            print(f"  MPJPE - Body: {metrics['mpjpe_body']:.4f}, "
                                  f"LHand: {metrics['mpjpe_lhand']:.4f}, "
                                  f"RHand: {metrics['mpjpe_rhand']:.4f}")
                    
                    # Print VAE info
                    latent_info = vae_info['latent']
                    if latent_info.get('type') == 'multipart':
                        print(f"  Latent (MultiPart):")
                        for part, pinfo in latent_info['parts'].items():
                            print(f"    {part}: mu={pinfo['mu_mean']:.4f}±{pinfo['mu_std']:.4f}, "
                                  f"var={pinfo['var_mean']:.4f}, z_norm={pinfo['z_norm']:.4f}")
                    else:
                        print(f"  Latent - mu: {latent_info['mu_mean']:.4f}±{latent_info['mu_std']:.4f}, "
                              f"var: {latent_info['var_mean']:.4f}")
                    
                    # Save video
                    if has_joints and not custom_args.no_video:
                        ds_output_dir = os.path.join(split_output_dir, src_key)
                        os.makedirs(ds_output_dir, exist_ok=True)
                        
                        safe_name = name.replace('/', '_').replace('\\', '_')
                        
                        if custom_args.three_view:
                            video_path = os.path.join(ds_output_dir, 
                                                      f'{collected[src_key]:03d}_{safe_name}_3view.mp4')
                            save_three_view_video(gt_joints_np, recon_joints_np, video_path,
                                                 f'{split}/{src_key}: {name}', custom_args.fps)
                        else:
                            video_path = os.path.join(ds_output_dir, 
                                                      f'{collected[src_key]:03d}_{safe_name}.mp4')
                            save_comparison_video(gt_joints_np, recon_joints_np, video_path,
                                                 f'{split}/{src_key}: {name}', custom_args.fps)
                        print(f"  Saved: {video_path}")
                    
                    collected[src_key] += 1
                    global_sample_idx += 1
        
        # Print collection summary
        print(f"\n[{split}] Collection summary:")
        for ds, count in collected.items():
            print(f"  {ds}: {count}/{target_per_dataset}")
    
    # =========================
    # Print Summary
    # =========================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_metrics)}")
    print(f"VAE type: {vae_type}")
    
    if all_metrics:
        print(f"\n[Overall]")
        print(f"  Avg Feature RMSE: {np.mean([m['feat_rmse'] for m in all_metrics]):.6f}")
        
        # Part-wise feature RMSE
        if 'feat_rmse_body' in all_metrics[0]:
            print(f"  Avg Feature RMSE - Body: {np.mean([m['feat_rmse_body'] for m in all_metrics]):.6f}")
            print(f"  Avg Feature RMSE - LHand: {np.mean([m['feat_rmse_lhand'] for m in all_metrics]):.6f}")
            print(f"  Avg Feature RMSE - RHand: {np.mean([m['feat_rmse_rhand'] for m in all_metrics]):.6f}")
        
        if 'mpjpe' in all_metrics[0]:
            print(f"  Avg MPJPE: {np.mean([m['mpjpe'] for m in all_metrics]):.4f}")
            if 'mpjpe_body' in all_metrics[0]:
                print(f"  Avg MPJPE - Body: {np.mean([m['mpjpe_body'] for m in all_metrics]):.4f}")
                print(f"  Avg MPJPE - LHand: {np.mean([m['mpjpe_lhand'] for m in all_metrics]):.4f}")
                print(f"  Avg MPJPE - RHand: {np.mean([m['mpjpe_rhand'] for m in all_metrics]):.4f}")
        
        # Per-split metrics
        for split in splits_to_process:
            split_metrics = [m for m in all_metrics if m.get('split') == split]
            if not split_metrics:
                continue
            
            print(f"\n[{split.upper()}] ({len(split_metrics)} samples)")
            print(f"  Avg Feature RMSE: {np.mean([m['feat_rmse'] for m in split_metrics]):.6f}")
            if 'mpjpe' in split_metrics[0]:
                print(f"  Avg MPJPE: {np.mean([m['mpjpe'] for m in split_metrics]):.4f}")
            
            for ds in datasets_to_process:
                ds_metrics = [m for m in split_metrics if m.get('src') == ds]
                if not ds_metrics:
                    continue
                
                print(f"\n    [{ds}] ({len(ds_metrics)} samples)")
                print(f"      Feature RMSE: {np.mean([m['feat_rmse'] for m in ds_metrics]):.6f}")
                if 'mpjpe' in ds_metrics[0]:
                    print(f"      MPJPE: {np.mean([m['mpjpe'] for m in ds_metrics]):.4f}")
        
        # VAE latent stats
        print(f"\n[VAE Latent Statistics]")
        mu_means = [m['vae_info']['latent']['mu_mean'] for m in all_metrics]
        var_means = [m['vae_info']['latent']['var_mean'] for m in all_metrics]
        print(f"  Avg mu: {np.mean(mu_means):.4f} ± {np.std(mu_means):.4f}")
        print(f"  Avg var: {np.mean(var_means):.4f} ± {np.std(var_means):.4f}")
        
        # MultiPart-specific stats
        if is_multipart and 'parts' in all_metrics[0]['vae_info']['latent']:
            print(f"\n[Per-Part Latent Statistics]")
            for part in model.vae.parts:
                part_mu = [m['vae_info']['latent']['parts'][part]['mu_mean'] for m in all_metrics]
                part_var = [m['vae_info']['latent']['parts'][part]['var_mean'] for m in all_metrics]
                print(f"  {part}: mu={np.mean(part_mu):.4f}±{np.std(part_mu):.4f}, var={np.mean(part_var):.4f}")
    
    # =========================
    # Save Metrics to JSON
    # =========================
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save config summary
    config_summary = {
        'config_name': cfg.NAME if hasattr(cfg, 'NAME') else 'Unknown',
        'timestamp': timestamp,
        'num_samples_per_dataset_per_split': custom_args.num_samples,
        'splits': splits_to_process,
        'datasets': datasets_to_process,
        'vae_type': vae_type,
        'total_samples': len(all_metrics),
    }
    if is_multipart:
        config_summary['parts'] = list(model.vae.parts)
    
    config_path = os.path.join(output_dir, 'config_summary.json')
    with open(config_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"Config summary saved to: {config_path}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
