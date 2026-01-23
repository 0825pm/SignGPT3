"""
SignGPT3 Model with SOKE-style MPJPE support

복사 위치: motGPT/models/motgpt.py
(또는 현재 사용 중인 모델 파일에 덮어쓰기)

수정 사항:
1. val_vae_forward: vertices도 반환
2. allsplit_step: MRMetrics에 vertices 전달
"""
import numpy as np
import os
import random
import torch
import time
from motGPT.config import instantiate_from_config
from os.path import join as pjoin
from motGPT.losses.motgpt import MotLosses
from motGPT.models.base import BaseModel
from .base import BaseModel
import json
from motGPT.utils.render_utils import render_motion


def sig(x):
    s = 1./(1+np.exp(-x))
    return s


class MotGPT(BaseModel):
    """
    SignGPT3 Model with SOKE-style metrics support
    
    Stage 1: Motion Tokenizer (VAE)
    Stage 2: Motion-language pretrain
    Stage 3: Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 fps=20,
                 guidance_scale=1.0,
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        self.njoints = self.datamodule.njoints
        self.fps = self.datamodule.fps
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae is not None:
            motion_vae['params']['datatype'] = self.datamodule.name
            self.vae = instantiate_from_config(motion_vae)

        self.vae_latent_channels = self.vae.latent_dim

        # Instantiate motion-language model
        if lm is not None:
            lm['params']['vae_latent_channels'] = self.vae_latent_channels
            lm['params']['vae_latent_size'] = self.vae.latent_size if hasattr(
                self.vae, 'latent_size') else None
            self.lm = instantiate_from_config(lm)
        else:
            self.lm = None

        # Freeze the motion tokenizer for lm training
        if 'adaptor' in self.hparams.stage:
            self.vae.training = False
            if self.lm is not None:
                self.lm.language_model.eval()
                self.lm.language_model.training = False
                self.lm.tokenizer.training = False
                for p in self.lm.language_model.parameters():
                    p.requires_grad = False

        # Loss and metrics
        self._losses = torch.nn.ModuleDict()
        self._losses['losses_train'] = MotLosses(
            cfg=cfg,
            stage=self.hparams.stage,
            num_joints=self.njoints,
        )
        self._losses['losses_val'] = MotLosses(
            cfg=cfg,
            stage=self.hparams.stage,
            num_joints=self.njoints,
        )

        self.feats2joints = datamodule.feats2joints
        self.model_dir = cfg.FOLDER_EXP
        self.vis_num = min(3, cfg.TRAIN.BATCH_SIZE)

    def feats2joints(self, features):
        """Convert features to joints (and vertices if available)"""
        return self.datamodule.feats2joints(features)

    def train_vae_forward(self, batch):
        """VAE training forward pass - NO SMPL-X vertices (memory optimization)"""
        feats_ref = batch["motion"]
        lengths = batch["length"]
        
        # Motion encode & decode
        motion_z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)
        recons_z, _ = self.vae.encode(feats_rst, lengths)
        
        # ============================================
        # IMPORTANT: Do NOT use feats2joints during training!
        # It triggers SMPL-X forward pass which creates huge vertices tensor.
        # For training loss, we only need features, not joints.
        # ============================================
        # joints_ref = self.feats2joints(feats_ref)  # DON'T DO THIS
        # joints_rst = self.feats2joints(feats_rst)  # DON'T DO THIS
        
        # Use None for joints - loss is computed on features only
        joints_ref = None
        joints_rst = None
        
        if dist_m is not None:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            dist_ref = None

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="val"):
        """
        VAE Validation Forward - SOKE-style MPJPE를 위해 vertices도 반환
        Memory optimized: vertices are detached and moved to CPU immediately
        """
        import gc
        
        # Detach batch
        feats_ref = batch["motion"].detach().clone()
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(z, lengths)
        del z, dist_m  # Free latent immediately

        # ============================================
        # SOKE-style: Extract both vertices and joints
        # Memory optimization: detach and move to CPU immediately
        # ============================================
        result_ref = self.feats2joints(feats_ref)
        result_rst = self.feats2joints(feats_rst)
        
        if isinstance(result_ref, tuple):
            vertices_ref, joints_ref = result_ref
            vertices_rst, joints_rst = result_rst
            # Move vertices to CPU immediately to free GPU memory
            if vertices_ref is not None:
                vertices_ref = vertices_ref.detach().cpu()
                vertices_rst = vertices_rst.detach().cpu()
            joints_ref = joints_ref.detach()
            joints_rst = joints_rst.detach()
        else:
            joints_ref = result_ref.detach()
            joints_rst = result_rst.detach()
            vertices_ref = None
            vertices_rst = None

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set with vertices for SOKE-style metrics
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,  # Already on CPU
            "vertices_rst": vertices_rst,  # Already on CPU
            "length": lengths,
        }
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        return rs_set

    def train_lm_forward(self, batch):
        """LM training forward pass"""
        motion = batch["motion"]
        lengths = batch["length"]
        text = batch["text"]
        
        # Encode motion to latent
        with torch.no_grad():
            motion_z, _ = self.vae.encode(motion, lengths)
        
        # LM forward
        outputs = self.lm(
            motion_z=motion_z,
            lengths=lengths,
            text=text,
        )
        
        return outputs

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        """Text-to-motion validation forward"""
        lengths = batch["length"]
        text = batch["text"]
        feats_ref = batch["motion"]
        
        # Generate motion from text
        motion_z = self.lm.generate(
            text=text,
            lengths=lengths,
        )
        
        # Decode to motion
        feats_rst = self.vae.decode(motion_z, lengths)
        
        # Get joints and vertices
        result_ref = self.feats2joints(feats_ref)
        result_rst = self.feats2joints(feats_rst)
        
        if isinstance(result_ref, tuple):
            vertices_ref, joints_ref = result_ref
            vertices_rst, joints_rst = result_rst
        else:
            joints_ref = result_ref
            joints_rst = result_rst
            vertices_ref = None
            vertices_rst = None

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "lengths_rst": lengths,
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        """Motion-to-text validation forward"""
        motion = batch["motion"]
        lengths = batch["length"]
        text_ref = batch["text"]
        
        # Encode motion
        motion_z, _ = self.vae.encode(motion, lengths)
        
        # Generate text from motion
        text_pred = self.lm.generate_text(
            motion_z=motion_z,
            lengths=lengths,
        )

        rs_set = {
            "t_ref": text_ref,
            "t_pred": text_pred,
            "length": lengths,
        }

        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        """
        Main step function for all splits (train/val/test)
        With SOKE-style MPJPE support
        """
        loss = None
        lengths = batch['length']
        src = batch.get('src', ['how2sign'] * len(lengths))
        name = batch.get('name', [f'sample_{i}' for i in range(len(lengths))])
        
        # ---- Training ----
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
            
        elif self.hparams.stage in ["lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_adaptor_pretrain"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
            
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # ---- Validation / Test ----
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                
                # ============================================
                # SOKE-style MRMetrics update with vertices
                # ============================================
                getattr(self.metrics, 'MRMetrics').update(
                    feats_rst=rs_set["m_rst"],
                    feats_ref=rs_set["m_ref"],
                    joints_rst=rs_set["joints_rst"],
                    joints_ref=rs_set["joints_ref"],
                    vertices_rst=rs_set.get("vertices_rst"),
                    vertices_ref=rs_set.get("vertices_ref"),
                    lengths=lengths,
                    src=src,
                    name=name
                )
                
            elif self.hparams.stage in ["lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                    
                    # TM2TMetrics or MRMetrics
                    if hasattr(self.metrics, 'TM2TMetrics'):
                        getattr(self.metrics, 'TM2TMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set.get("vertices_rst"),
                            vertices_ref=rs_set.get("vertices_ref"),
                            lengths=lengths,
                            lengths_rst=rs_set.get('lengths_rst', lengths),
                            split=split,
                            src=src,
                            name=name
                        )
                    if hasattr(self.metrics, 'MRMetrics'):
                        getattr(self.metrics, 'MRMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set.get("vertices_rst"),
                            vertices_ref=rs_set.get("vertices_ref"),
                            lengths=lengths,
                            src=src,
                            name=name
                        )
                        
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                    if hasattr(self.metrics, 'M2TMetrics'):
                        getattr(self.metrics, 'M2TMetrics').update(
                            pred_texts=rs_set["t_pred"],
                            gt_texts=rs_set["t_ref"],
                            lengths=rs_set['length'],
                            src=src,
                        )

            # ---- Visualization ----
            if self.hparams.stage == "vae" and self.hparams.task not in ["m2t"]:
                if (self.current_epoch + 1) % 20 == 0 and batch_idx == 0 and self.global_rank == 0:
                    self._visualize_validation(batch, rs_set)

        return loss

    def _visualize_validation(self, batch, rs_set):
        """Visualize validation samples"""
        try:
            joints_ref = rs_set['joints_ref']
            joints_rst = rs_set['joints_rst']
            lengths = batch['length']
            
            rand_save_idx = random.sample(range(len(lengths)), min(self.vis_num, len(lengths)))
            
            for idx in rand_save_idx:
                output_dir = os.path.join(self.model_dir, 'validate_motion', f'epoch_{self.current_epoch}')
                os.makedirs(output_dir, exist_ok=True)
                
                if 'fname' in batch:
                    keyid = batch['fname'][idx].split('/')[-1]
                else:
                    keyid = f'sample_{idx}'
                
                joint_ref = joints_ref[idx][:lengths[idx]]
                joint_rst = joints_rst[idx][:lengths[idx]]
                
                render_motion(joint_ref, None, output_dir=output_dir, fname=f'{keyid}_gt')
                render_motion(joint_rst, None, output_dir=output_dir, fname=f'{keyid}')
                
                if 'text' in batch:
                    np.savetxt(os.path.join(output_dir, f'{keyid}.txt'), [batch['text'][idx]], fmt='%s')
        except Exception as e:
            print(f"[Visualization Error] {e}")

    def configure_optimizers(self):
        """Configure optimizers"""
        if self.hparams.stage == "vae":
            optimizer = torch.optim.AdamW(
                self.vae.parameters(),
                lr=self.hparams.cfg.TRAIN.OPTIM.params.lr,
                betas=self.hparams.cfg.TRAIN.OPTIM.params.betas,
                weight_decay=self.hparams.cfg.TRAIN.OPTIM.params.weight_decay
            )
        else:
            # LM stage
            params = list(self.lm.parameters())
            optimizer = torch.optim.AdamW(
                params,
                lr=self.hparams.cfg.TRAIN.OPTIM.params.lr,
                betas=self.hparams.cfg.TRAIN.OPTIM.params.betas,
                weight_decay=self.hparams.cfg.TRAIN.OPTIM.params.weight_decay
            )
        
        # Scheduler
        if hasattr(self.hparams.cfg.TRAIN, 'LR_SCHEDULER'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.cfg.TRAIN.LR_SCHEDULER.params.T_max,
                eta_min=self.hparams.cfg.TRAIN.LR_SCHEDULER.params.eta_min
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    def train_lm_forward_modified(self, batch):
        """
        LM Training forward with cached text embedding support
        
        기존 코드에서 수정된 부분:
        - batch['text_emb'], batch['text_mask'] 추출
        - self.lm() 호출 시 text_emb, text_mask 전달
        """
        # 기존 코드
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch.get("tasks", None)
        
        # ★★★ 추가: Cached text embedding ★★★
        text_emb = batch.get('text_emb', None)   # [B, seq, 768] or None
        text_mask = batch.get('text_mask', None)  # [B, seq] or None
        
        # LLM Forward with cached embedding
        outputs = self.lm(
            texts, 
            feats_ref, 
            self.vae.encode_dist,  # motion_encode_net
            lengths, 
            tasks,
            text_emb=text_emb,      # ★ 추가
            text_mask=text_mask,    # ★ 추가
        )
        
        return {'outputs': outputs}


    # ============================================================================
    # 또는 더 간단하게: monkey patch 방식
    # ============================================================================
    # train.py 상단에 추가하면 됨:

    def patch_motgpt_for_cached_emb():
        """Monkey patch MotGPT to support cached text embeddings"""
        from motGPT.models.motgpt import MotGPT
        
        original_train_lm_forward = MotGPT.train_lm_forward
        
        def new_train_lm_forward(self, batch):
            feats_ref = batch["motion"]
            texts = batch["text"]
            lengths = batch["length"]
            tasks = batch.get("tasks", None)
            
            # ★ Cached embedding
            text_emb = batch.get('text_emb', None)
            text_mask = batch.get('text_mask', None)
            
            outputs = self.lm(
                texts, feats_ref, self.vae.encode_dist, lengths, tasks,
                text_emb=text_emb, text_mask=text_mask,
            )
            return {'outputs': outputs}
        
        MotGPT.train_lm_forward = new_train_lm_forward
        print("[Patch] MotGPT.train_lm_forward patched for cached text embeddings")
