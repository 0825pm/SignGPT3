"""
motGPT/models/light_ldm.py
===========================
SignGPT3 Light Latent Diffusion Model

학습: Frozen VAE로 motion → z 인코딩 후, latent space에서 diffusion 학습
추론: text → CFG denoising (UniPC 10 step) → z → VAE decode → motion

핵심 설계 (Light-T2M 차용):
  - 학습: prediction_type=sample  → target = z_clean (x_0 직접 예측)
  - 추론: obtain_eps_from_x0 변환 후 UniPC epsilon step
  - CFG:  cond / uncond 배치 합산 → single forward pass → weighted sum

연결 관계:
  VAE      : motGPT/archs/mld_vae.py          (MldVae, frozen)
  Text     : motGPT/archs/clip_encoder.py     (CLIP, frozen)
  Denoiser : motGPT/archs/light_denoiser.py   (SignLatentDenoiser, trainable)
  Metrics  : motGPT/metrics/mr.py             (MRMetrics)
  Base     : motGPT/models/base.py            (BaseModel - PL 공통 훅)

Config: configs/sign_ldm.yaml
  TRAIN.STAGE: ldm
  TRAIN.PRETRAINED_VAE: experiments/.../checkpoints/last.ckpt
  DIFFUSION.PREDICTION_TYPE: sample  (x_0 예측 학습 권장)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from diffusers import DDPMScheduler, UniPCMultistepScheduler
from motGPT.config import instantiate_from_config
from motGPT.models.base import BaseModel


# =============================================================================
# 간단한 Loss 트래커
# (MotLosses는 ldm stage 미지원 + num_joints 필수 → 직접 구현)
# =============================================================================

class LDMLossTracker(nn.Module):
    """
    epoch 단위 loss 평균 추적.
    BaseModel.loss_log_dict()가 self._losses[split].loss_dict를 참조하는
    인터페이스를 맞춰줌.
    """
    def __init__(self, split: str):
        super().__init__()
        self.split = split
        self.reset()

    def reset(self):
        self._sum   = 0.0
        self._count = 0

    def update(self, rs_set: dict) -> torch.Tensor:
        loss = rs_set["loss"]
        self._sum   += loss.item()
        self._count += 1
        return loss

    @property
    def loss_dict(self) -> dict:
        avg = self._sum / max(self._count, 1)
        return {f"{self.split}/ldm_loss": avg}


# =============================================================================
# LightLDM
# =============================================================================

class LightLDM(BaseModel):
    """
    SignGPT3 Latent Diffusion Model (LDM)

    Stage: ldm
      - VAE 는 frozen  (PRETRAINED_VAE 로부터 로드)
      - CLIP text encoder frozen
      - SignLatentDenoiser 만 학습

    학습 루프:
      motion [B,T,D]  →  vae.encode()  →  z [B,256]
      z + noise       →  denoiser(z_t, t, text_emb)  →  z_pred
      loss = MSE(z_pred, z_clean)          # prediction_type=sample

    추론 루프 (CFG + UniPC 10-step):
      z_T ~ N(0,I)  →  10× denoising  →  z_0
      z_0  →  vae.decode()  →  motion [B,T,D]
    """

    def __init__(
        self,
        cfg,
        datamodule,
        denoiser,                     # config dict → SignLatentDenoiser
        motion_vae,                   # config dict → MldVae
        text_encoder,                 # config dict → CLIPTextEncoder
        guidance_scale: float = 4.0,
        text_replace_prob: float = 0.1,
        step_num: int = 10,
        metrics_dict: List[str] = None,
        **kwargs,
    ):
        self.save_hyperparameters(
            ignore=['datamodule'],
            logger=False,
        )
        self.datamodule = datamodule
        self.njoints    = datamodule.njoints
        self.fps        = datamodule.fps

        super().__init__()   # BaseModel.__init__ → configure_metrics()

        # ── VAE (frozen) ──────────────────────────────────────────────────
        motion_vae['params']['datatype'] = self.datamodule.name
        self.vae = instantiate_from_config(motion_vae)
        self._freeze(self.vae)

        # ── Text Encoder (frozen) ─────────────────────────────────────────
        self.text_encoder = instantiate_from_config(text_encoder)
        self._freeze(self.text_encoder)

        # ── Denoiser (trainable) ──────────────────────────────────────────
        self.denoiser = instantiate_from_config(denoiser)

        # ── Diffusion schedulers (cfg에서 내부 생성) ──────────────────────
        # build_model이 yaml params를 그대로 전달하므로
        # 객체 인자 대신 cfg.DIFFUSION에서 직접 생성
        diff_cfg = cfg.DIFFUSION
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps = diff_cfg.NUM_TRAIN_TIMESTEPS,
            beta_start          = diff_cfg.BETA_START,
            beta_end            = diff_cfg.BETA_END,
            beta_schedule       = diff_cfg.BETA_SCHEDULE,
            prediction_type     = diff_cfg.PREDICTION_TYPE,  # "sample" 권장
        )
        # 추론 scheduler: 항상 epsilon space로 step
        self.sample_scheduler = UniPCMultistepScheduler(
            num_train_timesteps = diff_cfg.NUM_TRAIN_TIMESTEPS,
            beta_start          = diff_cfg.BETA_START,
            beta_end            = diff_cfg.BETA_END,
            beta_schedule       = diff_cfg.BETA_SCHEDULE,
            prediction_type     = "epsilon",
        )
        self.sample_scheduler.set_timesteps(step_num)

        # ── Loss trackers ─────────────────────────────────────────────────
        self._losses = nn.ModuleDict({
            'losses_train': LDMLossTracker('train'),
            'losses_val':   LDMLossTracker('val'),
        })

        # ── feats2joints (DataModule 제공) ────────────────────────────────
        self.feats2joints = datamodule.feats2joints

        n = sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad)
        print(f"[LightLDM] Denoiser trainable params: {n/1e6:.2f}M")

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _freeze(module: nn.Module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _encode_motion(self, motion: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        """
        motion [B, T, D] → z [B, 256]
          학습: rsample  (stochastic, VAE regularization 유지)
          추론: mu only  (deterministic, 안정적 생성)
        """
        if self.training:
            z, _ = self.vae.encode(motion, lengths)   # [1, B, 256]
        else:
            dist_tokens = self.vae.encode_dist(motion, lengths)  # [2, B, 256]
            z = dist_tokens[0:1, ...]                            # mu [1, B, 256]
        return z.squeeze(0)   # [B, 256]

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """texts → CLIP embedding [B, 512]"""
        with torch.no_grad():
            out = self.text_encoder(texts, self.device)
        return out["text_emb"]

    def _drop_text_for_cfg(self, texts: List[str]) -> List[str]:
        """CFG 학습: text_replace_prob 확률로 빈 문자열 대체"""
        return [
            "" if random.random() < self.hparams.text_replace_prob else t
            for t in texts
        ]

    def _obtain_eps_from_x0(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        x_0 예측 → epsilon 역산  (Light-T2M 방식)
        ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)
        """
        alpha_prod_t = self.sample_scheduler.alphas_cumprod[timestep].to(x_t.device)
        beta_prod_t  = 1.0 - alpha_prod_t
        return (x_t - alpha_prod_t ** 0.5 * x0) / (beta_prod_t ** 0.5)

    # =========================================================================
    # Training
    # =========================================================================

    def train_ldm_forward(self, batch: dict) -> dict:
        motion  = batch["motion"]
        lengths = batch["length"]
        texts   = batch["text"]
        B = motion.shape[0]

        # 1. Text (CFG dropout)
        text_emb = self._encode_text(self._drop_text_for_cfg(texts))  # [B, 512]

        # 2. Motion → latent (no_grad: VAE frozen)
        with torch.no_grad():
            z_clean = self._encode_motion(motion, lengths)  # [B, 256]

        # 3. Forward diffusion
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(z_clean)
        z_t   = self.noise_scheduler.add_noise(z_clean, noise, t)  # [B, 256]

        # 4. Denoiser
        z_pred = self.denoiser(z_t, t, text_emb)   # [B, 256]

        # 5. Target
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "sample":
            target = z_clean
        elif prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"prediction_type '{prediction_type}' not supported")

        return {"loss": F.mse_loss(z_pred, target)}

    # =========================================================================
    # Validation / Test
    # =========================================================================

    @torch.no_grad()
    def val_ldm_forward(self, batch: dict) -> dict:
        motion  = batch["motion"]
        lengths = batch["length"]
        texts   = batch["text"]
        B, T, D = motion.shape

        motion_gen = self._sample_motion(texts, lengths, B)

        return {
            "m_ref":      motion,
            "m_rst":      motion_gen,
            "joints_ref": self.feats2joints(motion),
            "joints_rst": self.feats2joints(motion_gen),
            "length":     lengths,
        }

    @torch.no_grad()
    def _sample_motion(
        self,
        texts: List[str],
        lengths: List[int],
        B: int,
    ) -> torch.Tensor:
        """
        CFG + UniPC denoising loop → motion [B, T, D]
        """
        # cond + uncond 텍스트 임베딩
        text_emb = self._encode_text(texts + [""] * B)   # [2B, 512]

        # 초기 노이즈
        latent_dim = self.denoiser.latent_dim
        z_t = torch.randn(B, latent_dim, device=self.device)
        z_t = z_t * self.sample_scheduler.init_noise_sigma

        # Denoising loop
        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        prediction_type = self.noise_scheduler.config.prediction_type

        for t in self.sample_scheduler.timesteps:
            # single forward (cond + uncond 동시)
            z_pred = self.denoiser(
                z_t.repeat(2, 1),
                t.repeat(2 * B).to(self.device),
                text_emb,
            )  # [2B, 256]
            z_pred_cond, z_pred_uncond = z_pred.chunk(2, dim=0)

            # x_0 예측이면 epsilon으로 변환
            if prediction_type == "sample":
                eps_cond   = self._obtain_eps_from_x0(z_pred_cond,   z_t, t)
                eps_uncond = self._obtain_eps_from_x0(z_pred_uncond, z_t, t)
            else:
                eps_cond, eps_uncond = z_pred_cond, z_pred_uncond

            # CFG
            guided_eps = eps_uncond + self.hparams.guidance_scale * (eps_cond - eps_uncond)

            # Scheduler step
            z_t = self.sample_scheduler.step(guided_eps, t, z_t).prev_sample.float()

        # z → motion
        return self.vae.decode(z_t.unsqueeze(0), lengths)   # [B, T, D]

    # =========================================================================
    # allsplit_step
    # =========================================================================

    def allsplit_step(self, split: str, batch, batch_idx):
        loss    = None
        lengths = batch["length"]
        src     = batch.get("src",  ["how2sign"] * len(lengths))
        name    = batch.get("name", batch.get("fname", [f"sample_{i}" for i in range(len(lengths))]))

        if split == "train":
            rs_set = self.train_ldm_forward(batch)
            loss   = self._losses["losses_train"].update(rs_set)

        if split in ("val", "test"):
            rs_set = self.val_ldm_forward(batch)
            if hasattr(self.metrics, "MRMetrics"):
                self.metrics.MRMetrics.update(
                    feats_rst  = rs_set["m_rst"],
                    feats_ref  = rs_set["m_ref"],
                    joints_rst = rs_set["joints_rst"],
                    joints_ref = rs_set["joints_ref"],
                    lengths    = lengths,
                    src        = src,
                    name       = name,
                )

        return loss

    # =========================================================================
    # BaseModel 인터페이스 override
    # =========================================================================

    def loss_log_dict(self, split: str) -> dict:
        key = f"losses_{split}"
        if key in self._losses:
            dico = self._losses[key].loss_dict
            self._losses[key].reset()
            return dico
        return {}

    def step_log_dict(self) -> dict:
        return {"epoch": float(self.current_epoch), "step": float(self.global_step)}

    # =========================================================================
    # Optimizer
    # =========================================================================

    def configure_optimizers(self):
        cfg_optim = self.hparams.cfg.TRAIN.OPTIM
        cfg_lrs   = self.hparams.cfg.TRAIN.get("LR_SCHEDULER", None)

        OptClass = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}.get(cfg_optim.target)
        if OptClass is None:
            raise ValueError(f"Optimizer '{cfg_optim.target}' not supported")
        optimizer = OptClass(self.denoiser.parameters(), **cfg_optim.params)

        if cfg_lrs is None:
            return optimizer
        if cfg_lrs.target == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg_lrs.params)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer

    # =========================================================================
    # Checkpoint hooks
    # =========================================================================

    def on_save_checkpoint(self, checkpoint):
        """text_encoder / vae 제외 (frozen, 용량 절약)"""
        rm = [k for k in checkpoint["state_dict"] if k.startswith(("text_encoder.", "vae."))]
        for k in rm:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        keys = list(checkpoint["state_dict"].keys())
        for k in keys:
            if "_orig_mod." in k:
                checkpoint["state_dict"][k.replace("_orig_mod.", "")] = \
                    checkpoint["state_dict"].pop(k)

    # =========================================================================
    # VAE 사전학습 가중치 로드 (train.py에서 호출)
    # =========================================================================

    def load_pretrained_vae(self, ckpt_path: str):
        """
        PRETRAINED_VAE 체크포인트에서 VAE 가중치만 추출하여 로드.

        train.py에서:
            if cfg.TRAIN.PRETRAINED_VAE:
                model.load_pretrained_vae(cfg.TRAIN.PRETRAINED_VAE)
        """
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"[LightLDM] PRETRAINED_VAE not found: '{ckpt_path}'")
            return

        ckpt      = torch.load(ckpt_path, map_location="cpu")
        state     = ckpt.get("state_dict", ckpt)
        vae_state = {k.replace("vae.", "", 1): v
                     for k, v in state.items() if k.startswith("vae.")}

        if not vae_state:
            print(f"[LightLDM] Warning: no 'vae.*' keys in {ckpt_path}")
            return

        missing, unexpected = self.vae.load_state_dict(vae_state, strict=False)
        print(f"[LightLDM] VAE loaded from: {ckpt_path}")
        if missing:
            print(f"  missing    ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"  unexpected ({len(unexpected)}): {unexpected[:3]}")