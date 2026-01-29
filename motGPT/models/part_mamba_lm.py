"""
Part Mamba LM for SignGPT3 Stage 2
==================================

Text → Sign Language Motion Generation
VAE latent space에서 Diffusion 수행

Architecture:
    Text (EN/ZH/DE) → M-CLIP → text_emb [B, 512]
                                   ↓
    z [B, 3, 256] + noise → PartMambaDenoiser → z_clean
                                   ↓
                            VAE Decoder → Motion [B, T, 120]

Features:
- Multilingual support (M-CLIP)
- Part-aware latent diffusion (body, lhand, rhand)
- Mamba-based denoiser (O(n) complexity)
- DDPM with sample prediction (Light-T2M style)
- Classifier-free guidance (CFG)

복사 위치: motGPT/models/part_mamba_lm.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from motGPT.config import instantiate_from_config


# =============================================================================
# Utility Functions
# =============================================================================

def lengths_to_mask(lengths: List[int], device: torch.device, max_len: int = None) -> torch.Tensor:
    """Convert lengths to boolean mask"""
    if max_len is None:
        max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len)
    mask = mask < torch.tensor(lengths, device=device).unsqueeze(1)
    return mask


# =============================================================================
# Part Mamba LM (NOT inheriting BaseModel - clean implementation)
# =============================================================================

class PartMambaLM(pl.LightningModule):
    """
    Stage 2: Text → Motion Latent Diffusion
    
    Inherits from pl.LightningModule directly (not BaseModel)
    to avoid conflicts with MotLosses system.
    
    Components:
    - VAE: LightPartMambaVae (frozen, from Stage 1)
    - Text Encoder: MCLIPEncoder (frozen)
    - Denoiser: PartMambaDenoiser (trainable)
    - Scheduler: DDPM (training) / DPM-Solver (inference)
    """
    
    def __init__(
        self,
        cfg,
        datamodule,
        # Model configs (instantiated externally)
        vae_config: Dict,
        text_encoder_config: Dict,
        denoiser_config: Dict,
        # Diffusion configs
        num_train_timesteps: int = 1000,
        prediction_type: str = 'sample',  # 'sample' or 'epsilon'
        beta_schedule: str = 'squaredcos_cap_v2',
        # Training configs
        guidance_scale: float = 4.0,
        guidance_uncond_prob: float = 0.1,  # CFG uncond dropout
        # Pretrained
        pretrained_vae: str = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['datamodule'], logger=False)
        self.datamodule = datamodule
        self.cfg = cfg
        
        # =====================================================================
        # 1. VAE (frozen)
        # =====================================================================
        self.vae = instantiate_from_config(vae_config)
        
        # Load pretrained VAE
        if pretrained_vae and os.path.exists(pretrained_vae):
            print(f"[PartMambaLM] Loading pretrained VAE: {pretrained_vae}")
            ckpt = torch.load(pretrained_vae, map_location='cpu')
            vae_state = {k.replace('vae.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('vae.')}
            self.vae.load_state_dict(vae_state, strict=False)
        
        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Get latent info
        self.latent_dim = self.vae.latent_dim
        self.latent_size = self.vae.latent_size  # 3 for parts
        
        # =====================================================================
        # 2. Text Encoder (frozen)
        # =====================================================================
        self.text_encoder = instantiate_from_config(text_encoder_config)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # =====================================================================
        # 3. Denoiser (trainable)
        # =====================================================================
        self.denoiser = instantiate_from_config(denoiser_config)
        
        # =====================================================================
        # 4. Noise Scheduler
        # =====================================================================
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,
        )
        
        # Inference scheduler (faster sampling)
        self.inference_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            algorithm_type='dpmsolver++',
            solver_order=2,
        )
        
        # =====================================================================
        # 5. Configs
        # =====================================================================
        self.guidance_scale = guidance_scale
        self.guidance_uncond_prob = guidance_uncond_prob
        self.prediction_type = prediction_type
        
        # Null text embedding for CFG (learned)
        self.null_text_emb = nn.Parameter(torch.zeros(1, 512))
        
        # Data transform (SMPL-X based, same as VAE training)
        self.feats2joints = datamodule.feats2joints if hasattr(datamodule, 'feats2joints') else None
        
        # =====================================================================
        # 6. Metrics (MRMetrics for validation) - use nn.ModuleDict
        # =====================================================================
        from motGPT.metrics.mr import MRMetrics
        self.mr_metrics = nn.ModuleDict({
            'val': MRMetrics(
                njoints=datamodule.njoints,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP if hasattr(cfg.METRIC, 'DIST_SYNC_ON_STEP') else True,
                data_name=datamodule.name if hasattr(datamodule, 'name') else 'unknown',
            ),
            'test': MRMetrics(
                njoints=datamodule.njoints,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP if hasattr(cfg.METRIC, 'DIST_SYNC_ON_STEP') else True,
                data_name=datamodule.name if hasattr(datamodule, 'name') else 'unknown',
            ),
        })
        
        # =====================================================================
        # 8. Dataset/Language Embedding (ASL, CSL, DGS 구분)
        # =====================================================================
        self.dataset_names = ['how2sign', 'csl', 'phoenix']  # EN, ZH, DE
        self.dataset_to_idx = {name: idx for idx, name in enumerate(self.dataset_names)}
        self.dataset_emb = nn.Embedding(len(self.dataset_names), 512)  # Same dim as text_emb
        nn.init.normal_(self.dataset_emb.weight, std=0.02)
        
        print(f"[PartMambaLM] Initialized")
        print(f"  VAE latent: [{self.latent_size}, {self.latent_dim}]")
        print(f"  Prediction type: {prediction_type}")
        print(f"  Guidance scale: {guidance_scale}")
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Training step with diffusion loss"""
        motion = batch['motion']  # [B, T, 120]
        texts = batch['text']     # List[str]
        lengths = batch['length'] # List[int]
        src = batch.get('src', ['how2sign'] * len(texts))  # Dataset source
        
        B = motion.shape[0]
        device = motion.device
        
        # 1. Encode motion to latent (VAE frozen)
        with torch.no_grad():
            z, _ = self.vae.encode(motion, lengths)  # LightPartMambaVae: [B, 3, 256]
            # z is [B, 3, 256] - no permute needed for LightPartMambaVae
        
        # 2. Encode text (frozen)
        with torch.no_grad():
            text_emb = self.text_encoder(texts)  # [B, 512]
        
        # 3. Add dataset embedding
        dataset_idx = torch.tensor([
            self.dataset_to_idx.get(s.lower() if isinstance(s, str) else 'how2sign', 0) 
            for s in src
        ], device=device)
        dataset_emb = self.dataset_emb(dataset_idx)  # [B, 512]
        text_emb = text_emb + dataset_emb  # Combine text + dataset
        
        # 4. CFG: randomly drop text (replace with null embedding)
        if self.training and self.guidance_uncond_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.guidance_uncond_prob
            null_emb = self.null_text_emb.expand(B, -1)
            text_emb = torch.where(drop_mask.unsqueeze(1), null_emb, text_emb)
        
        # 5. Sample timestep and add noise
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        noise = torch.randn_like(z)
        z_noisy = self.noise_scheduler.add_noise(z, noise, timestep)
        
        # 6. Predict
        z_pred = self.denoiser(z_noisy, timestep, text_emb)
        
        # 7. Compute loss
        if self.prediction_type == 'sample':
            target = z  # Predict x0 directly
        else:
            target = noise  # Predict noise
        
        loss = F.mse_loss(z_pred, target)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step with generation and metrics"""
        motion = batch['motion']
        texts = batch['text']
        lengths = batch['length']
        src = batch.get('src', ['how2sign'] * len(texts))
        
        B = motion.shape[0]
        device = motion.device
        
        # Encode GT motion
        with torch.no_grad():
            z_gt, _ = self.vae.encode(motion, lengths)  # LightPartMambaVae: [B, 3, 256]
            # z_gt is [B, 3, 256] - no permute needed
        
        # Encode text
        with torch.no_grad():
            text_emb = self.text_encoder(texts)
        
        # Add dataset embedding
        dataset_idx = torch.tensor([
            self.dataset_to_idx.get(s.lower() if isinstance(s, str) else 'how2sign', 0) 
            for s in src
        ], device=device)
        dataset_emb = self.dataset_emb(dataset_idx)
        text_emb = text_emb + dataset_emb
        
        # Sample timestep and compute loss
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        noise = torch.randn_like(z_gt)
        z_noisy = self.noise_scheduler.add_noise(z_gt, noise, timestep)
        
        z_pred = self.denoiser(z_noisy, timestep, text_emb)
        
        if self.prediction_type == 'sample':
            target = z_gt
        else:
            target = noise
        
        loss = F.mse_loss(z_pred, target)
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        
        # =====================================================================
        # Generate motion for metrics
        # =====================================================================
        with torch.no_grad():
            # Generate motion from text (with dataset info)
            motion_pred = self.generate(
                texts=texts,
                lengths=lengths,
                datasets=src,  # Pass dataset info
                num_inference_steps=10,
                guidance_scale=self.guidance_scale,
            )
        
        # Prepare for metrics (rs_set format)
        rs_set = {
            'motion': motion,           # GT [B, T, 120]
            'm_rst': motion_pred,       # Pred [B, T, 120]
            'length': lengths,
            'src': src,
        }
        
        # Update metrics
        self.allsplit_step('val', rs_set)
        
        return {'val_loss': loss}
    
    def allsplit_step(self, split: str, rs_set: Dict):
        """Update metrics (simplified - metrics only)"""
        motion_gt = rs_set['motion']
        motion_pred = rs_set['m_rst']
        lengths = rs_set['length']
        src = rs_set.get('src', ['unknown'] * len(lengths))
        
        # Ensure same length
        max_len = max(motion_gt.shape[1], motion_pred.shape[1])
        if motion_gt.shape[1] < max_len:
            pad = torch.zeros(motion_gt.shape[0], max_len - motion_gt.shape[1], motion_gt.shape[2], device=motion_gt.device)
            motion_gt = torch.cat([motion_gt, pad], dim=1)
        if motion_pred.shape[1] < max_len:
            pad = torch.zeros(motion_pred.shape[0], max_len - motion_pred.shape[1], motion_pred.shape[2], device=motion_pred.device)
            motion_pred = torch.cat([motion_pred, pad], dim=1)
        
        # DEBUG: Track src distribution across all validation batches
        if split == 'val':
            if not hasattr(self, '_val_src_counts'):
                self._val_src_counts = {'how2sign': 0, 'csl': 0, 'phoenix': 0}
            for s in src:
                s_lower = s.lower()
                if 'how2sign' in s_lower:
                    self._val_src_counts['how2sign'] += 1
                elif 'csl' in s_lower:
                    self._val_src_counts['csl'] += 1
                elif 'phoenix' in s_lower:
                    self._val_src_counts['phoenix'] += 1
        
        # Compute joints using datamodule.feats2joints (SMPL-X based, same as VAE training)
        joints_rst = None
        joints_ref = None
        if self.feats2joints is not None:
            try:
                with torch.no_grad():
                    joints_ref = self.feats2joints(motion_gt)
                    joints_rst = self.feats2joints(motion_pred)
                    
                    # Handle tuple return (vertices, joints) -> extract joints
                    if isinstance(joints_ref, tuple):
                        joints_ref = joints_ref[-1]
                        joints_rst = joints_rst[-1]
                    
                    # Ensure [B, T, J, 3] shape
                    if joints_ref is not None and joints_ref.dim() == 3:
                        B = motion_gt.shape[0]
                        T = motion_gt.shape[1]
                        joints_ref = joints_ref.view(B, T, -1, 3)
                        joints_rst = joints_rst.view(B, T, -1, 3)
            except Exception as e:
                pass  # Continue without joints if feats2joints fails
        
        # Update metrics (MRMetrics)
        if split in self.mr_metrics:
            self.mr_metrics[split].update(
                feats_rst=motion_pred,
                feats_ref=motion_gt,
                joints_rst=joints_rst,
                joints_ref=joints_ref,
                lengths=lengths,
                src=src,
            )
    
    def on_validation_epoch_end(self):
        """Compute and log metrics at end of validation epoch"""
        # DEBUG: Print src distribution for this epoch
        if hasattr(self, '_val_src_counts') and self.trainer.is_global_zero:
            print(f"\n[DEBUG] Val samples per dataset: {self._val_src_counts}")
            self._val_src_counts = {'how2sign': 0, 'csl': 0, 'phoenix': 0}
        
        # Compute metrics
        if 'val' in self.mr_metrics:
            metrics_dict = self.mr_metrics['val'].compute()
            
            # Log metrics with both prefixes (for ModelCheckpoint compatibility)
            for k, v in metrics_dict.items():
                self.log(f'val/{k}', v, sync_dist=True)
                self.log(f'Metrics/{k}', v, sync_dist=True)  # For ModelCheckpoint
            
            # Print metrics summary
            if self.trainer.is_global_zero:
                print("\n" + "=" * 50)
                print("=== MRMetrics Results ===")
                for k, v in sorted(metrics_dict.items()):
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.item():.4f}")
                    else:
                        print(f"  {k}: {v:.4f}")
                print("=" * 50)
            
            # Reset metrics
            self.mr_metrics['val'].reset()
    
    def on_train_epoch_end(self):
        """End of training epoch - just pass"""
        pass
    
    # =========================================================================
    # Generation
    # =========================================================================
    
    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        lengths: List[int],
        datasets: List[str] = None,  # 'how2sign', 'csl', 'phoenix'
        num_inference_steps: int = 10,
        guidance_scale: float = None,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """
        Generate motion from text
        
        Args:
            texts: List of text prompts
            lengths: List of motion lengths
            datasets: List of dataset names ('how2sign', 'csl', 'phoenix')
                     Used to add dataset-specific embedding for style control
            num_inference_steps: Number of denoising steps (default: 10)
            guidance_scale: CFG scale (default: self.guidance_scale)
            use_cfg: Whether to use classifier-free guidance
            
        Returns:
            motion: [B, max_len, 120] generated motion
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        B = len(texts)
        device = next(self.parameters()).device
        
        # Default to how2sign if not specified
        if datasets is None:
            datasets = ['how2sign'] * B
        
        # 1. Encode text
        text_emb = self.text_encoder(texts)  # [B, 512]
        
        # 2. Add dataset embedding
        dataset_idx = torch.tensor([
            self.dataset_to_idx.get(d.lower() if isinstance(d, str) else 'how2sign', 0) 
            for d in datasets
        ], device=device)
        dataset_emb = self.dataset_emb(dataset_idx)  # [B, 512]
        text_emb = text_emb + dataset_emb  # [B, 512]
        
        # 3. Prepare null embedding for CFG
        if use_cfg and guidance_scale > 1.0:
            null_emb = self.null_text_emb.expand(B, -1)
            text_emb_cfg = torch.cat([null_emb, text_emb], dim=0)  # [2B, 512]
        
        # 4. Initialize noise
        z = torch.randn(B, self.latent_size, self.latent_dim, device=device)
        
        # 5. Setup inference scheduler
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inference_scheduler.timesteps
        
        # 6. Denoising loop
        for t in timesteps:
            t_batch = t.expand(B)
            
            if use_cfg and guidance_scale > 1.0:
                # CFG: concatenate unconditional and conditional
                z_input = torch.cat([z, z], dim=0)  # [2B, 3, 256]
                t_input = torch.cat([t_batch, t_batch], dim=0)
                
                z_pred = self.denoiser(z_input, t_input, text_emb_cfg)
                
                # Split and apply guidance
                z_pred_uncond, z_pred_cond = z_pred.chunk(2)
                z_pred = z_pred_uncond + guidance_scale * (z_pred_cond - z_pred_uncond)
            else:
                z_pred = self.denoiser(z, t_batch, text_emb)
            
            # Scheduler step
            z = self.inference_scheduler.step(z_pred, t, z).prev_sample
        
        # 7. Decode latent to motion
        # LightPartMambaVae expects z: [B, 3, 256] (NOT [3, B, 256])
        motion = self.vae.decode(z, lengths)  # [B, max_len, 120]
        
        return motion
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Train: denoiser, null_text_emb, dataset_emb
        params = (
            list(self.denoiser.parameters()) + 
            [self.null_text_emb] + 
            list(self.dataset_emb.parameters())
        )
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.cfg.TRAIN.OPTIM.params.lr,
            betas=tuple(self.hparams.cfg.TRAIN.OPTIM.params.betas),
            weight_decay=self.hparams.cfg.TRAIN.OPTIM.params.weight_decay,
        )
        
        # Cosine annealing
        if hasattr(self.hparams.cfg.TRAIN, 'LR_SCHEDULER'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.cfg.TRAIN.LR_SCHEDULER.params.T_max,
                eta_min=self.hparams.cfg.TRAIN.LR_SCHEDULER.params.eta_min,
            )
            return [optimizer], [scheduler]
        
        return optimizer


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("PartMambaLM module loaded successfully")