"""
SignGPT3 LM (Text-to-Motion) Visualization Script
For MotionGPT3's mBART Hybrid LM with Diffusion Head

Features:
- Supports all splits: train, val, test
- Supports all datasets: how2sign, csl, phoenix
- Text-to-Motion generation with diffusion
- GT vs Generated comparison (2-panel)
- Detailed metrics with JSON export

Usage:
    # Basic usage (val split, 2 samples per dataset)
    python vis_lm.py --cfg configs/sign_lm_mbart_hybrid.yaml --output vis_lm_output --nodebug

    # Custom options
    python vis_lm.py --cfg configs/sign_lm_mbart_hybrid.yaml --output vis_lm_output \\
        --num_samples 5 --splits val,test --datasets how2sign,csl,phoenix

    # Custom checkpoint
    python vis_lm.py --cfg configs/sign_lm_mbart_hybrid.yaml --checkpoint path/to/ckpt.ckpt

    # Custom text generation (without GT comparison)
    python vis_lm.py --cfg configs/sign_lm_mbart_hybrid.yaml \\
        --text "A person waves hello with their right hand"

    # Multiple texts from file
    python vis_lm.py --cfg configs/sign_lm_mbart_hybrid.yaml --text_file prompts.txt
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
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants
# =============================================================================
SOKE_BODY_DIM = 30      # upper body (10 joints × 3)
SOKE_LHAND_DIM = 45     # left hand (15 joints × 3)
SOKE_RHAND_DIM = 45     # right hand (15 joints × 3)
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


def get_src_label(src):
    """Normalize source dataset name."""
    src_lower = src.lower() if src else 'unknown'
    if 'how2sign' in src_lower or 'h2s' in src_lower:
        return 'how2sign'
    elif 'csl' in src_lower:
        return 'csl'
    elif 'phoenix' in src_lower:
        return 'phoenix'
    return src_lower


# =============================================================================
# Video Saving Functions
# =============================================================================

def save_comparison_video(gt_joints, gen_joints, save_path, title='', fps=25):
    """Save GT vs Generated comparison video (2-panel)."""
    T = max(len(gt_joints), len(gen_joints))
    J = gt_joints.shape[1] if len(gt_joints) > 0 else gen_joints.shape[1]
    
    # Normalize to root
    gt_joints = normalize_to_root(gt_joints)
    gen_joints = normalize_to_root(gen_joints)
    
    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    if title:
        # Truncate long titles
        display_title = title[:100] + '...' if len(title) > 100 else title
        fig.suptitle(display_title, fontsize=10, wrap=True)
    
    connections = get_connections(J)
    colors = get_joint_colors()
    
    all_elements = []
    
    for ax_idx, (ax, seq, label) in enumerate([
        (axes[0], gt_joints, 'GT'),
        (axes[1], gen_joints, 'Generated')
    ]):
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        xi, yi = 0, 1  # x-y projection
        
        lines = []
        for (i, j) in connections:
            if i < 22 and j < 22:
                color = colors['body']
            elif i < 40 and j < 40:
                color = colors['lhand']
            else:
                color = colors['rhand']
            line, = ax.plot([], [], color=color, linewidth=2, alpha=0.8)
            lines.append((line, i, j))
        
        scatter = ax.scatter([], [], c='black', s=30, zorder=5)
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


def save_single_video(joints, save_path, title='', fps=25):
    """Save single motion video (for custom text generation)."""
    T, J, _ = joints.shape
    
    # Normalize to root
    joints = normalize_to_root(joints)
    
    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    if title:
        display_title = title[:80] + '...' if len(title) > 80 else title
        ax.set_title(display_title, fontsize=12, wrap=True)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    connections = get_connections(J)
    colors = get_joint_colors()
    xi, yi = 0, 1
    
    lines = []
    for (i, j) in connections:
        if i < 22 and j < 22:
            color = colors['body']
        elif i < 40 and j < 40:
            color = colors['lhand']
        else:
            color = colors['rhand']
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.8)
        lines.append((line, i, j))
    
    scatter = ax.scatter([], [], c='black', s=30, zorder=5)
    
    def update(frame):
        frame_data = joints[min(frame, T-1)]
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


def save_three_view_video(gt_joints, gen_joints, save_path, title='', fps=25):
    """Save 3-view comparison video (front, side, top)."""
    T = max(len(gt_joints), len(gen_joints))
    J = gt_joints.shape[1]
    
    gt_joints = normalize_to_root(gt_joints)
    gen_joints = normalize_to_root(gen_joints)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    if title:
        display_title = title[:100] + '...' if len(title) > 100 else title
        fig.suptitle(display_title, fontsize=10, wrap=True)
    
    connections = get_connections(J)
    colors = get_joint_colors()
    
    views = [
        ('Front (X-Y)', 0, 1),
        ('Side (Z-Y)', 2, 1),
        ('Top (X-Z)', 0, 2),
    ]
    
    all_elements = []
    
    for row_idx, (seq, row_label) in enumerate([(gt_joints, 'GT'), (gen_joints, 'Generated')]):
        for col_idx, (view_name, xi, yi) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.set_title(f'{row_label} - {view_name}', fontsize=11)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            lines = []
            for (i, j) in connections:
                if i < 22 and j < 22:
                    color = colors['body']
                elif i < 40 and j < 40:
                    color = colors['lhand']
                else:
                    color = colors['rhand']
                line, = ax.plot([], [], color=color, linewidth=2, alpha=0.8)
                lines.append((line, i, j))
            
            scatter = ax.scatter([], [], c='black', s=20, zorder=5)
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

def compute_metrics(gt_feats, gen_feats, gt_joints=None, gen_joints=None):
    """Compute generation metrics."""
    metrics = {}
    
    # Ensure same length
    min_len = min(len(gt_feats), len(gen_feats))
    gt_feats = gt_feats[:min_len]
    gen_feats = gen_feats[:min_len]
    
    # Feature space metrics
    metrics['feat_mse'] = float(((gt_feats - gen_feats) ** 2).mean())
    metrics['feat_rmse'] = float(np.sqrt(metrics['feat_mse']))
    metrics['feat_l1'] = float(np.abs(gt_feats - gen_feats).mean())
    
    # Part-wise feature error (SOKE 120-dim format)
    D = gt_feats.shape[-1]
    if D >= 120:
        body_idx = list(range(0, 30))
        lhand_idx = list(range(30, 75))
        rhand_idx = list(range(75, 120))
        
        metrics['feat_l1_body'] = float(np.abs(gt_feats[..., body_idx] - gen_feats[..., body_idx]).mean())
        metrics['feat_l1_lhand'] = float(np.abs(gt_feats[..., lhand_idx] - gen_feats[..., lhand_idx]).mean())
        metrics['feat_l1_rhand'] = float(np.abs(gt_feats[..., rhand_idx] - gen_feats[..., rhand_idx]).mean())
        metrics['feat_l1_hand'] = float(np.abs(
            gt_feats[..., lhand_idx + rhand_idx] - gen_feats[..., lhand_idx + rhand_idx]
        ).mean())
    
    # Joint space metrics (MPJPE)
    if gt_joints is not None and gen_joints is not None:
        min_len = min(len(gt_joints), len(gen_joints))
        gt_j = gt_joints[:min_len]
        gen_j = gen_joints[:min_len]
        
        diff = gt_j - gen_j
        dist = np.sqrt((diff ** 2).sum(axis=-1))  # [T, J]
        metrics['mpjpe'] = float(dist.mean())
        
        J = gt_j.shape[1]
        if J >= 55:
            body_idx = list(range(0, 22))
            lhand_idx = list(range(25, 40))
            rhand_idx = list(range(40, 55))
            
            metrics['mpjpe_body'] = float(dist[:, body_idx].mean())
            metrics['mpjpe_lhand'] = float(dist[:, lhand_idx].mean())
            metrics['mpjpe_rhand'] = float(dist[:, rhand_idx].mean())
            metrics['mpjpe_hand'] = float(dist[:, lhand_idx + rhand_idx].mean())
        
        # Velocity error
        gt_vel = np.diff(gt_j, axis=0)
        gen_vel = np.diff(gen_j, axis=0)
        if len(gt_vel) > 0 and len(gen_vel) > 0:
            min_vel_len = min(len(gt_vel), len(gen_vel))
            metrics['vel_error'] = float(np.abs(gt_vel[:min_vel_len] - gen_vel[:min_vel_len]).mean())
    
    return metrics


# =============================================================================
# LM Forward Functions
# =============================================================================

def lm_generate(model, text, length, device='cuda'):
    """
    Generate motion from text using LM.
    
    Args:
        model: MotGPT model with vae and lm
        text: list of text strings or single string
        length: list of target lengths or single int
        
    Returns:
        feats_gen: [B, T, D] generated features
    """
    if isinstance(text, str):
        text = [text]
    if isinstance(length, int):
        length = [length] * len(text)
    
    model.eval()
    
    with torch.no_grad():
        # LM generates motion latent z
        motion_z = model.lm.generate(
            text=text,
            lengths=length,
        )
        
        # VAE decodes to motion features
        feats_gen = model.vae.decode(motion_z, length)
    
    return feats_gen


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_checkpoint(ckpt_path, model, strict=False):
    """Load model checkpoint."""
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # load_state_dict returns different things depending on PyTorch/Lightning version
    result = model.load_state_dict(new_state_dict, strict=strict)
    
    if result is not None:
        missing, unexpected = result.missing_keys, result.unexpected_keys
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"  Checkpoint loaded (strict={strict})")
    
    return model


# =============================================================================
# Main Function
# =============================================================================

def main():
    # =========================
    # Parse Arguments
    # =========================
    parser = argparse.ArgumentParser(
        description='SignGPT3 LM Text-to-Motion Visualization',
        add_help=False
    )
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of samples per dataset per split (default: 2)')
    parser.add_argument('--output', type=str, default='vis_lm_output',
                       help='Output directory (default: vis_lm_output)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS (default: 25)')
    parser.add_argument('--three_view', action='store_true',
                       help='Generate 3-view videos instead of 2-panel')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Override checkpoint path')
    parser.add_argument('--splits', type=str, default='val',
                       help='Comma-separated splits to process (default: val)')
    parser.add_argument('--datasets', type=str, default='how2sign,csl,phoenix',
                       help='Comma-separated datasets (default: how2sign,csl,phoenix)')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video generation (metrics only)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--text', type=str, default=None,
                       help='Custom text for generation (no GT comparison)')
    parser.add_argument('--text_file', type=str, default=None,
                       help='File with text prompts (one per line)')
    parser.add_argument('--gen_length', type=int, default=100,
                       help='Generated motion length for custom text (default: 100)')
    
    custom_args, remaining = parser.parse_known_args()
    
    # Set CUDA device
    if custom_args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(custom_args.gpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Replace sys.argv for motGPT's parse_args
    sys.argv = [sys.argv[0]] + remaining
    
    # Import motGPT modules
    from motGPT.config import parse_args
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model
    
    # Parse config
    cfg = parse_args()
    
    # Setup output directory
    output_dir = Path(custom_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.FOLDER_EXP = str(output_dir)
    cfg.DEBUG = False
    
    # Parse options
    splits_to_process = [s.strip() for s in custom_args.splits.split(',')]
    datasets_to_process = [d.strip() for d in custom_args.datasets.split(',')]
    
    # =========================
    # Print Configuration
    # =========================
    print(f"\n{'='*70}")
    print("SignGPT3 LM Text-to-Motion Visualization")
    print(f"{'='*70}")
    print(f"Config: {cfg.NAME}")
    print(f"Output: {output_dir}")
    print(f"Splits: {splits_to_process}")
    print(f"Datasets: {datasets_to_process}")
    print(f"Samples per dataset: {custom_args.num_samples}")
    print(f"Video mode: {'3-view' if custom_args.three_view else '2-panel'}")
    print(f"FPS: {custom_args.fps}")
    print(f"Device: {device}")
    
    # =========================
    # Build Model & Data
    # =========================
    print(f"\n[1/3] Building datamodule...")
    datamodule = build_data(cfg)
    datamodule.setup('fit')
    
    print(f"[2/3] Building model...")
    model = build_model(cfg, datamodule)
    
    # Load checkpoint
    if custom_args.checkpoint:
        ckpt_path = custom_args.checkpoint
    elif hasattr(cfg.TEST, 'CHECKPOINTS') and cfg.TEST.CHECKPOINTS:
        ckpt_path = cfg.TEST.CHECKPOINTS
    elif hasattr(cfg.TRAIN, 'PRETRAINED') and cfg.TRAIN.PRETRAINED:
        ckpt_path = cfg.TRAIN.PRETRAINED
    else:
        ckpt_path = None
        print("  Warning: No checkpoint specified!")
    
    if ckpt_path and os.path.exists(ckpt_path):
        model = load_checkpoint(ckpt_path, model, strict=False)
    
    model.eval()
    model.to(device)
    
    # =========================
    # Print Model Info
    # =========================
    print(f"\n[Model Info]")
    print(f"  Stage: {cfg.TRAIN.STAGE}")
    if hasattr(model, 'vae'):
        print(f"  VAE: {type(model.vae).__name__}")
    if hasattr(model, 'lm'):
        print(f"  LM: {type(model.lm).__name__}")
    
    # Get feats2joints converter
    feats2joints = None
    if hasattr(datamodule, 'feats2joints'):
        feats2joints = datamodule.feats2joints
        print(f"  feats2joints: Available")
    else:
        print(f"  feats2joints: Not available")
    
    # =========================
    # Custom Text Generation Mode
    # =========================
    if custom_args.text or custom_args.text_file:
        print(f"\n{'='*70}")
        print("Custom Text Generation Mode")
        print(f"{'='*70}")
        
        # Get texts
        if custom_args.text:
            texts = [custom_args.text]
        else:
            with open(custom_args.text_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        custom_output_dir = output_dir / 'custom'
        custom_output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(texts):
            print(f"\n[{i+1}/{len(texts)}] Text: {text[:80]}...")
            
            with torch.no_grad():
                # Generate motion
                feats_gen = lm_generate(
                    model, text, custom_args.gen_length, device
                )
                feats_gen = feats_gen[0].cpu().numpy()  # [T, D]
            
            # Convert to joints
            joints_gen = None
            if feats2joints is not None:
                try:
                    result = feats2joints(torch.from_numpy(feats_gen).unsqueeze(0).to(device))
                    if isinstance(result, tuple):
                        _, joints = result
                    else:
                        joints = result
                    joints_gen = joints[0].cpu().numpy()  # [T, J, 3]
                except Exception as e:
                    print(f"    Warning: feats2joints failed: {e}")
            
            # Save video
            if not custom_args.no_video and joints_gen is not None:
                video_path = custom_output_dir / f'gen_{i:03d}.mp4'
                save_single_video(joints_gen, str(video_path), title=text, fps=custom_args.fps)
                print(f"    Saved: {video_path}")
            
            # Save features
            np.save(custom_output_dir / f'gen_{i:03d}_feats.npy', feats_gen)
            
            # Save text
            with open(custom_output_dir / f'gen_{i:03d}_text.txt', 'w') as f:
                f.write(text)
        
        print(f"\nCustom generation complete! Output: {custom_output_dir}")
        return
    
    # =========================
    # Dataset-based Generation Mode
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
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        # Track collected samples
        collected = {ds: 0 for ds in datasets_to_process}
        target_per_dataset = custom_args.num_samples
        
        def all_collected():
            return all(collected[ds] >= target_per_dataset for ds in datasets_to_process)
        
        # Process batches (vis_vae.py style - individual sample processing)
        for batch in tqdm(dataloader, desc=f'{split}'):
            if all_collected():
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            motion = batch['motion']
            lengths = batch['length']
            texts = batch['text']
            names = batch.get('name', [f'sample_{i}' for i in range(len(lengths))])
            srcs = batch.get('src', ['unknown'] * len(lengths))
            B = motion.shape[0]
            
            with torch.no_grad():
                for i in range(B):
                    if all_collected():
                        break
                    
                    length = lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i]
                    if length == 0:
                        continue
                    
                    name = names[i] if names else f'sample_{global_sample_idx}'
                    src = srcs[i] if srcs else 'unknown'
                    text = texts[i] if texts else ''
                    
                    src_key = get_src_label(src)
                    
                    if src_key not in datasets_to_process:
                        continue
                    if collected[src_key] >= target_per_dataset:
                        continue
                    
                    # ========================================
                    # Generate motion for this single sample
                    # ========================================
                    try:
                        feats_gen = lm_generate(model, [text], [length], device)
                        feats_gen = feats_gen[0]  # [T, D]
                    except Exception as e:
                        print(f"    Warning: Generation failed for {name}: {e}")
                        continue
                    
                    # Get GT features
                    gt_feats = motion[i, :length]
                    
                    # ========================================
                    # Convert to joints
                    # ========================================
                    has_joints = False
                    gt_joints_np = None
                    gen_joints_np = None
                    
                    if feats2joints is not None:
                        try:
                            gt_input = motion[i:i+1, :length]
                            gen_input = feats_gen.unsqueeze(0)
                            
                            # feats2joints may return (vertices, joints) or just joints
                            gt_result = feats2joints(gt_input)
                            gen_result = feats2joints(gen_input)
                            
                            if isinstance(gt_result, tuple):
                                gt_joints = gt_result[-1]
                                gen_joints = gen_result[-1]
                            else:
                                gt_joints = gt_result
                                gen_joints = gen_result
                            
                            gt_joints = gt_joints * POSE_SCALE
                            gen_joints = gen_joints * POSE_SCALE
                            gt_joints_np = gt_joints.cpu().numpy()
                            gen_joints_np = gen_joints.cpu().numpy()
                            
                            if gt_joints_np.ndim == 4:
                                gt_joints_np = gt_joints_np[0]
                                gen_joints_np = gen_joints_np[0]
                            
                            has_joints = True
                        except Exception as e:
                            if global_sample_idx == 0:
                                print(f"    Note: feats2joints failed ({e})")
                            has_joints = False
                    
                    # Get numpy arrays for metrics
                    gt_feats_np = gt_feats.cpu().numpy()
                    gen_feats_np = feats_gen.cpu().numpy()
                    
                    # ========================================
                    # Compute metrics
                    # ========================================
                    metrics = compute_metrics(gt_feats_np, gen_feats_np, gt_joints_np, gen_joints_np)
                    metrics['name'] = name
                    metrics['src'] = src_key
                    metrics['split'] = split
                    metrics['text'] = text[:200]
                    metrics['length'] = int(length)
                    all_metrics.append(metrics)
                    
                    # ========================================
                    # Print sample info
                    # ========================================
                    print(f"\n[{split}/{src_key}] {name}")
                    print(f"  Text: {text[:60]}...")
                    print(f"  Length: {length} frames")
                    print(f"  Feature L1: {metrics['feat_l1']:.4f}")
                    
                    if 'feat_l1_body' in metrics:
                        print(f"  Feature L1 - Body: {metrics['feat_l1_body']:.4f}, "
                              f"Hand: {metrics.get('feat_l1_hand', 0):.4f}")
                    
                    if has_joints and 'mpjpe' in metrics:
                        print(f"  MPJPE: {metrics['mpjpe']:.4f}")
                        if 'mpjpe_body' in metrics:
                            print(f"  MPJPE - Body: {metrics['mpjpe_body']:.4f}, "
                                  f"LHand: {metrics['mpjpe_lhand']:.4f}, "
                                  f"RHand: {metrics['mpjpe_rhand']:.4f}")
                    
                    # ========================================
                    # Save video
                    # ========================================
                    if has_joints and not custom_args.no_video:
                        ds_output_dir = split_output_dir / src_key
                        ds_output_dir.mkdir(exist_ok=True)
                        
                        safe_name = name.replace('/', '_').replace('\\', '_')[:50]
                        
                        if custom_args.three_view:
                            video_path = ds_output_dir / f'{collected[src_key]:03d}_{safe_name}_3view.mp4'
                            save_three_view_video(gt_joints_np, gen_joints_np, str(video_path),
                                                 f'{split}/{src_key}: {text[:60]}', custom_args.fps)
                        else:
                            video_path = ds_output_dir / f'{collected[src_key]:03d}_{safe_name}.mp4'
                            save_comparison_video(gt_joints_np, gen_joints_np, str(video_path),
                                                 f'{split}/{src_key}: {text[:60]}', custom_args.fps)
                        print(f"  Saved: {video_path}")
                    
                    # Save features
                    ds_output_dir = split_output_dir / src_key
                    ds_output_dir.mkdir(exist_ok=True)
                    safe_name = name.replace('/', '_').replace('\\', '_')[:50]
                    np.savez(
                        ds_output_dir / f'{collected[src_key]:03d}_{safe_name}_feats.npz',
                        gt_feats=gt_feats_np,
                        gen_feats=gen_feats_np,
                        text=text,
                        name=name,
                        src=src_key
                    )
                    
                    collected[src_key] += 1
                    global_sample_idx += 1
        
        # Print collection summary
        print(f"\n[{split}] Collection summary:")
        for ds, count in collected.items():
            print(f"  {ds}: {count}/{target_per_dataset}")
    
    # =========================
    # Save Summary
    # =========================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    # Save metrics JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Compute aggregate metrics
    if all_metrics:
        print(f"\nAggregate Metrics:")
        
        for src in datasets_to_process:
            src_metrics = [m for m in all_metrics if m['src'] == src]
            if not src_metrics:
                continue
            
            print(f"\n  [{src.upper()}] ({len(src_metrics)} samples)")
            
            for key in ['feat_l1', 'feat_l1_body', 'feat_l1_hand', 'mpjpe', 'mpjpe_body', 'mpjpe_hand']:
                values = [m[key] for m in src_metrics if key in m]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"    {key}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\nTotal samples processed: {len(all_metrics)}")
    print(f"Output directory: {output_dir}")
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()