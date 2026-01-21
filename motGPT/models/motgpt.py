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
# import motGPT.render.matplot.plot_3d_global as plot_3d
from motGPT.utils.render_utils import render_motion



def sig(x):
    s = 1./(1+np.exp(-x))
    return s

class MotGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
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
        if motion_vae != None:
            motion_vae['params']['datatype'] = self.datamodule.name
            self.vae = instantiate_from_config(motion_vae)  # mld.models.architectures.mld_vae.MldVae

        self.vae_latent_channels = self.vae.latent_dim  # 256

        # Instantiate motion-language model (only if lm config is provided)
        if lm is not None:
            lm['params']['vae_latent_channels'] = self.vae_latent_channels
            lm['params']['vae_latent_size'] = self.vae.latent_size if hasattr(
                self.vae,'latent_size') else None
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
            
            for p in self.vae.parameters():
                p.requires_grad = False
            if self.lm is not None:
                for p in self.lm.language_model.parameters():
                    p.requires_grad = False
        elif 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
        self.model_dir = cfg.FOLDER_EXP
        self.vis_num = 2

        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: MotLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        if self.lm is None:
            raise ValueError("Language model not initialized. Cannot perform forward pass for text-to-motion.")

        # Forward
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            else:
                motion = self.vae.decode(outputs[i])
            feats_rst_lst.append(motion)
            lengths.append(motion.shape[0])
            if motion.shape[0] > max_len:
                max_len = motion.shape[0]

        feats_rst = torch.zeros(
            len(texts), max_len, feats_rst_lst[0].shape[-1]).to(outputs[0].device)
        for i in range(len(texts)):
            feats_rst[i, :lengths[i]] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(batch["motion"])

        return feats_rst, joints_rst, lengths, output_texts

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion encode & decode
        motion_z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)
        recons_z = motion_z  # Use the encoded latent as recons_z

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        if dist_m is not None:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            dist_ref = None

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "length": lengths,
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
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

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        if hasattr(self.datamodule, 'renorm4t2m'):
            feats_ref = self.datamodule.renorm4t2m(feats_ref)
            feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }

        return rs_set

    def train_lm_forward(self, batch):
        if self.lm is None:
            raise ValueError("Language model not initialized for LM training.")
            
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch.get("tasks", None)

        # Encode motion into latent
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)

        # Language model forward
        outputs = self.lm(texts, z, lengths, tasks=tasks)

        rs_set = {
            "m_ref": feats_ref,
            "outputs": outputs,
            "length": lengths,
        }
        return rs_set

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        if self.lm is None:
            raise ValueError("Language model not initialized for text-to-motion.")
            
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                lengths=lengths,
                                                stage='test',
                                                task='t2m',
                                                tasks=tasks)

        # Decode motion from latent
        feats_rst_lst = []
        lengths_rst = []
        max_len = 0
        
        for i in range(len(texts)):
            motion = self.vae.decode(outputs[i])
            feats_rst_lst.append(motion)
            lengths_rst.append(motion.shape[0])
            if motion.shape[0] > max_len:
                max_len = motion.shape[0]

        feats_rst = torch.zeros(
            len(texts), max_len, feats_rst_lst[0].shape[-1]).to(feats_ref.device)
        for i in range(len(texts)):
            feats_rst[i, :lengths_rst[i]] = feats_rst_lst[i]

        # Recover joints
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm
        if hasattr(self.datamodule, 'renorm4t2m'):
            feats_ref = self.datamodule.renorm4t2m(feats_ref)
            feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": lengths,
            "length_rst": lengths_rst,
        }
        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        if self.lm is None:
            raise ValueError("Language model not initialized for motion-to-text.")
            
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]

        # Encode motion
        z, dist = self.vae.encode(feats_ref, lengths)

        # Generate text
        outputs = self.lm.generate_conditional(z,
                                              lengths=lengths,
                                              stage='test',
                                              task='m2t')

        rs_set = {
            "m_ref": feats_ref,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_finetune", "lm_adaptor_pretrain", "lm_t2m"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain", "lm_t2m"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t"]:
                # MultiModality evaluation separately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

            # Visualization
            if self.hparams.task not in ["m2t"]:
                if (self.current_epoch+1) % 20 == 0 and batch_idx == 0 and self.global_rank == 0:
                    try:
                        joints_ref = rs_set['joints_ref']
                        joints_rst = rs_set['joints_rst']
                        lengths = batch['length']
                        rand_save_idx = random.sample(range(len(lengths)), min(self.vis_num, len(lengths)))
                        for i in rand_save_idx:
                            render_motion(
                                joints_ref[i, :lengths[i]].cpu().numpy(),
                                outdir=os.path.join(self.model_dir, 'vis'),
                                step=self.current_epoch,
                                name=f'{split}_{batch_idx}_{i}_ref',
                                fps=self.fps
                            )
                            rst_len = rs_set.get('length_rst', lengths)[i] if isinstance(rs_set.get('length_rst', lengths), list) else lengths[i]
                            render_motion(
                                joints_rst[i, :rst_len].cpu().numpy(),
                                outdir=os.path.join(self.model_dir, 'vis'),
                                step=self.current_epoch,
                                name=f'{split}_{batch_idx}_{i}_rst',
                                fps=self.fps
                            )
                    except Exception as e:
                        print(f"Visualization error: {e}")

            # Update metrics
            if self.hparams.task not in ["m2t"]:
                # Get src and name from batch
                src = batch.get('src', ['how2sign'] * len(rs_set["length"]))
                names = batch.get('name', [f'sample_{i}' for i in range(len(rs_set["length"]))])
                
                for metric in metrics_dicts:
                    if hasattr(self.metrics, metric):
                        if metric == "MRMetrics":
                            # MRMetrics: pass feats, lengths, src, name
                            getattr(self.metrics, metric).update(
                                feats_rst=rs_set["m_rst"],
                                feats_ref=rs_set["m_ref"],
                                lengths=rs_set["length"],
                                src=src,
                                name=names,
                            )
                        elif metric == "TM2TMetrics":
                            getattr(self.metrics, metric).update(
                                feats_ref=rs_set["m_ref"],
                                feats_rst=rs_set["m_rst"],
                                lengths_ref=rs_set["length"],
                                lengths_rst=rs_set["length"]
                            )
                        elif metric == "MMMetrics":
                            getattr(self.metrics, metric).update(
                                rs_set["m_rst"],
                                rs_set["length"]
                            )
                        else:
                            # Generic fallback
                            try:
                                getattr(self.metrics, metric).update(
                                    rs_set["joints_rst"],
                                    rs_set["joints_ref"],
                                    rs_set["length"]
                                )
                            except Exception as e:
                                print(f"Metric {metric} update error: {e}")

        if split in ["test"]:
            return rs_set
        return loss