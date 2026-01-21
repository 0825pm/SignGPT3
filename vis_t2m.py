"""
SignGPT3 Text-to-Motion (T2M) Visualization Script
Usage: python vis_t2m.py --checkpoint <path> --num_samples 10

Features:
- Trainer 없이 독립적으로 동작
- vis_vae.py와 동일한 visualization 스타일
- 2-panel (GT vs Generated) 또는 3-view 모드 지원
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants (same as vis_vae.py)
# =============================================================================
POSE_SCALE = 2.0


# =============================================================================
# Skeleton Visualization Utilities (from vis_vae.py)
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
# Video Saving Functions (from vis_vae.py)
# =============================================================================

def save_comparison_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save GT vs Generated comparison video (2-panel)."""
    seqs = [gt_joints, recon_joints]
    panel_titles = ['Ground Truth', 'Generated']
    
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
    row_titles = ['Ground Truth', 'Generated']
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


def save_single_video(joints, save_path, title='Generated', fps=20):
    """Save single motion video (for custom text generation without GT)."""
    T = joints.shape[0]
    J = joints.shape[1]
    
    root_idx = 9 if J > 21 else 0
    joints_norm = normalize_to_root(joints.copy(), root_idx)
    
    if J >= 55:
        upper_body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        hand_idx = list(range(25, min(55, J)))
        valid_idx = upper_body_idx + hand_idx
    else:
        valid_idx = list(range(min(22, J)))
    
    all_x = joints_norm[:, valid_idx, 0].flatten()
    all_y = joints_norm[:, valid_idx, 1].flatten()
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    max_range = max(x_max - x_min, y_max - y_min) * 1.2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    
    x_lim = (x_mid - max_range/2, x_mid + max_range/2)
    y_lim = (y_mid - max_range/2, y_mid + max_range/2)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    connections = get_connections(J)
    colors = get_joint_colors()
    
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
    
    body_scatter = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
    lhand_scatter = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rhand_scatter = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
    
    upper_body_idx = [i for i in [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] if i < J]
    
    def update(frame):
        frame_data = joints_norm[min(frame, T-1)]
        x, y = frame_data[:, 0], frame_data[:, 1]
        
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        
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


# =============================================================================
# Feature to Joints Conversion (vis_vae.py 방식 그대로)
# =============================================================================

def convert_feats_to_joints(feats, feats2joints, length):
    """
    vis_vae.py와 동일한 방식으로 features를 joints로 변환
    
    Args:
        feats: [1, T, D] or [T, D] features
        feats2joints: conversion function
        length: actual sequence length
        
    Returns:
        joints_np: [T, J, 3] numpy array
    """
    # Ensure 3D input [1, T, D]
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    
    # Slice to actual length
    feats_input = feats[:, :length, :]
    
    # Convert to joints
    result = feats2joints(feats_input)
    
    # Handle tuple return (vertices, joints)
    if isinstance(result, tuple):
        joints = result[-1]  # joints is usually last
    else:
        joints = result
    
    # Apply scale
    joints = joints * POSE_SCALE
    
    # Convert to numpy
    joints_np = joints.cpu().numpy()
    
    # Handle 4D output [B, T, J, 3] -> [T, J, 3]
    if joints_np.ndim == 4:
        joints_np = joints_np[0]
    
    return joints_np


# =============================================================================
# T2M Generation Function
# =============================================================================

def t2m_generate(model, texts, lengths, device):
    """
    Trainer 없이 Text-to-Motion generation 수행
    """
    # Task 설정 (caption condition) - 'class' 키 필수!
    tasks = [{
        'class': 't2m',
        'input': ['Generate motion: <Caption_Placeholder>'],
        'output': ['']
    }] * len(texts)
    
    # 1. LM으로 text → motion latent 생성
    outputs = model.lm.generate_conditional(
        texts,
        lengths=lengths,
        stage='test',
        tasks=tasks,
    )
    
    # 2. Diffusion sampling으로 motion latent 추출
    sampled_token_latents, motion_mask = model.lm.sample_tokens(
        outputs, 
        device,
        temperature=1.0,
        cfg=model.guidance_scale,
        vae_mean_std_inv=model.vae.mean_std_inv
    )
    
    # 3. Reshape latents: [bs, latent_size * latent_dim] → [latent_size, bs, latent_dim]
    sampled_token_latents = sampled_token_latents.reshape(
        len(lengths), model.vae.latent_size, -1
    ).permute(1, 0, 2)
    
    # 4. VAE decode로 motion features 복원
    feats_rst = model.vae.decode(sampled_token_latents, lengths)
    
    # 5. 실패한 샘플 처리 (motion_mask == 1인 경우)
    for i in range(len(lengths)):
        if motion_mask[i] == 1:
            feats_rst[i] = torch.zeros_like(feats_rst[i])
    
    return feats_rst


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SignGPT3 T2M Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to LM checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--text_file', type=str, default=None,
                       help='File with text prompts (one per line)')
    parser.add_argument('--output', type=str, default='vis_t2m_output',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples from validation set')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--cfg', type=str, default='configs/sign_lm_stage1.yaml',
                       help='Config file path')
    parser.add_argument('--three_view', action='store_true',
                       help='Generate 3-view videos instead of 2-panel')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video generation')
    
    args = parser.parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import after setting device
    from motGPT.config import parse_args
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model
    
    # Parse config
    sys.argv = [sys.argv[0], '--cfg', args.cfg]
    cfg = parse_args()
    
    # 핵심: FOLDER_EXP 설정
    cfg.FOLDER_EXP = str(output_dir)
    cfg.DEBUG = False
    
    # =========================
    # Print Configuration
    # =========================
    print(f"\n{'='*70}")
    print("SignGPT3 Text-to-Motion Visualization")
    print(f"{'='*70}")
    print(f"Config: {args.cfg}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Video mode: {'3-view' if args.three_view else '2-panel'}")
    print(f"FPS: {args.fps}")
    
    # Build datamodule
    print("\n[1/3] Building datamodule...")
    datamodule = build_data(cfg)
    datamodule.setup('fit')  # val_dataset 초기화를 위해 'fit' 사용
    
    # Build model
    print("[2/3] Building model...")
    model = build_model(cfg, datamodule)
    
    # Load checkpoint
    print(f"[3/3] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Get feats2joints from datamodule (vis_vae.py 방식)
    feats2joints = None
    if hasattr(datamodule, 'feats2joints'):
        feats2joints = datamodule.feats2joints
        print(f"  feats2joints: Available")
    else:
        print(f"  feats2joints: Not available")
    
    # Get text prompts
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = None
    
    all_metrics = []
    
    # ========== Custom Text Generation ==========
    if texts:
        print(f"\n{'='*70}")
        print(f"Generating motions for {len(texts)} custom texts")
        print(f"{'='*70}")
        
        for i, text in enumerate(texts):
            print(f"\n[{i+1}/{len(texts)}] Text: {text[:80]}...")
            
            with torch.no_grad():
                # Generate motion
                gen_length = 100
                feats_rst = t2m_generate(model, [text], [gen_length], device)
                
                # Convert to joints (vis_vae.py 방식)
                if feats2joints is not None:
                    try:
                        joints_np = convert_feats_to_joints(
                            feats_rst[0], feats2joints, gen_length
                        )
                        has_joints = True
                    except Exception as e:
                        print(f"  Warning: feats2joints failed: {e}")
                        has_joints = False
                else:
                    has_joints = False
            
            # Save video
            if has_joints and not args.no_video:
                fname = f"custom_{i:03d}"
                video_path = str(output_dir / f"{fname}.mp4")
                save_single_video(joints_np, video_path, 
                                 title=f"Generated: {text[:50]}...", fps=args.fps)
                print(f"  Saved: {video_path}")
            
            # Save text
            with open(output_dir / f"custom_{i:03d}.txt", 'w') as f:
                f.write(text)
            
            all_metrics.append({
                'idx': i,
                'text': text,
                'length': gen_length,
            })
    
    # ========== Validation Set Generation ==========
    else:
        print(f"\n{'='*70}")
        print(f"Generating motions from validation set ({args.num_samples} samples)")
        print(f"{'='*70}")
        
        val_loader = datamodule.val_dataloader()
        
        count = 0
        for batch_idx, batch in enumerate(val_loader):
            if count >= args.num_samples:
                break
            
            # Move to device
            batch['motion'] = batch['motion'].to(device)
            texts_batch = batch['text']
            lengths = batch['length']
            names = batch.get('fname', batch.get('name', [f'sample_{i}' for i in range(len(texts_batch))]))
            srcs = batch.get('src', ['unknown'] * len(texts_batch))
            
            B = len(texts_batch)
            feats_ref = batch['motion']
            
            with torch.no_grad():
                # Generate via T2M
                feats_rst = t2m_generate(model, texts_batch, lengths, device)
            
            # Process each sample individually (vis_vae.py 방식)
            for i in range(min(B, args.num_samples - count)):
                length = lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i]
                text = texts_batch[i]
                name = str(names[i]).split('/')[-1] if '/' in str(names[i]) else str(names[i])
                src = srcs[i] if i < len(srcs) else 'unknown'
                
                print(f"\n[{count+1}/{args.num_samples}] {name}")
                print(f"  Source: {src}")
                print(f"  Text: {text[:80]}...")
                print(f"  Length: {length} frames")
                
                # Convert to joints (샘플별로 처리 - vis_vae.py 방식)
                has_joints = False
                gt_joints_np = None
                rst_joints_np = None
                
                if feats2joints is not None:
                    try:
                        # GT joints
                        gt_joints_np = convert_feats_to_joints(
                            feats_ref[i:i+1], feats2joints, length
                        )
                        
                        # Generated joints  
                        rst_joints_np = convert_feats_to_joints(
                            feats_rst[i:i+1], feats2joints, length
                        )
                        
                        has_joints = True
                        
                    except Exception as e:
                        print(f"  Warning: feats2joints failed: {e}")
                        has_joints = False
                
                # Save video
                if has_joints and not args.no_video:
                    safe_name = name.replace('/', '_').replace('\\', '_')
                    
                    if args.three_view:
                        video_path = str(output_dir / f"{count:03d}_{safe_name}_3view.mp4")
                        save_three_view_video(gt_joints_np, rst_joints_np, video_path,
                                            title=f"{name}: {text[:50]}...", fps=args.fps)
                    else:
                        video_path = str(output_dir / f"{count:03d}_{safe_name}.mp4")
                        save_comparison_video(gt_joints_np, rst_joints_np, video_path,
                                            title=f"{name}: {text[:50]}...", fps=args.fps)
                    print(f"  Saved: {video_path}")
                
                # Save text
                with open(output_dir / f"{count:03d}_{name}.txt", 'w') as f:
                    f.write(text)
                
                # Compute metrics
                if has_joints:
                    diff = gt_joints_np - rst_joints_np
                    dist = np.sqrt((diff ** 2).sum(axis=-1))
                    mpjpe = float(dist.mean())
                    print(f"  MPJPE: {mpjpe:.4f}")
                else:
                    mpjpe = None
                
                all_metrics.append({
                    'idx': count,
                    'name': name,
                    'src': src,
                    'text': text,
                    'length': int(length),
                    'mpjpe': mpjpe,
                })
                
                count += 1
                if count >= args.num_samples:
                    break
    
    # =========================
    # Print Summary
    # =========================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_metrics)}")
    
    mpjpe_list = [m['mpjpe'] for m in all_metrics if m.get('mpjpe') is not None]
    if mpjpe_list:
        avg_mpjpe = np.mean(mpjpe_list)
        print(f"Average MPJPE: {avg_mpjpe:.4f}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print(f"\n✅ Done! Results saved to: {output_dir}")
    print(f"   Videos: {len(list(output_dir.glob('*.mp4')))} files")


if __name__ == "__main__":
    main()