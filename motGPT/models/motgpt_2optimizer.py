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
        # self.ep = 0
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            motion_vae['params']['datatype'] = self.datamodule.name
            self.vae = instantiate_from_config(motion_vae)  # mld.models.architectures.mld_vae.MldVae

        self.vae_latent_channels = self.vae.latent_dim  # 256

        # Instantiate motion-language model
        lm['params']['vae_latent_channels'] = self.vae_latent_channels
        lm['params']['vae_latent_size'] = self.vae.latent_size if hasattr(
            self.vae,'latent_size') else None
        self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm training
        if 'adaptor' in self.hparams.stage:
            self.vae.training = False
            self.lm.language_model.eval()
            self.lm.language_model.training = False
            self.lm.tokenizer.training = False
            
            for p in self.vae.parameters():
                p.requires_grad = False
            for p in self.lm.language_model.parameters():
                p.requires_grad = False
            # for p in self.lm.tokenizer.parameters():
            #     p.requires_grad = False
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

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        loss = self.allsplit_step("train", batch, batch_idx)
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        return loss

    def configure_optimizers(self):
        from motGPT.config import get_obj_from_str
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target

        optimizers_0 = [*self.vae.parameters(), *self.lm.language_model.parameters()]
        optimizer = get_obj_from_str(optim_target)(
            params=optimizers_0, **self.hparams.cfg.TRAIN.OPTIM.params)

        optimizers_diff = [
            *self.lm.motion_und_head.parameters(), 
            *self.lm.diffloss.parameters()
        ]
        if hasattr(self.lm, 'fake_latent'):
            optimizers_diff.append(self.lm.fake_latent)
        if hasattr(self.lm, 'norm_layer'):
            optimizers_diff.append(*self.lm.norm_layer.parameters())
        if hasattr(self.lm, 'diffusion_pos_embed_learned'):
            optimizers_diff.append(self.lm.diffusion_pos_embed_learned)
        if hasattr(self.lm, 'motion_gen_head'):
            optimizers_diff.append(*self.lm.motion_gen_head.parameters())

        optimizer_diff = get_obj_from_str(optim_target)(
            params=optimizers_diff, **self.hparams.cfg.TRAIN.OPTIM.params_diff)

        # Scheduler
        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params)

        return ({'optimizer': optimizer, 'lr_scheduler': lr_scheduler}, 
                {"optimizer": optimizer_diff})

        
    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]
        if task in ['inbetween']:
            lengths = lengths_ref
        else:
            lengths = [random.randint(20,50)*4 for l in lengths_ref]
        motion_tokens_input = batch['motion_tokens_input']

        if task in ['t2m', 'pred', 'prediction', 'inbetween']:
            outputs = self.lm.generate_direct_motion(
                    texts,
                    motion_tokens=motion_tokens_input,
                    num_beams=1,
                    do_sample=False,
                    )
            sampled_token_latents, motion_mask = self.lm.sample_tokens(
                outputs, self.lm.device, 
                temperature=1.0, cfg=self.guidance_scale, 
            vae_mean_std_inv=self.vae.mean_std_inv)
            sampled_token_latents = sampled_token_latents.reshape(len(lengths), self.vae.latent_size, -1).permute(1,0,2)
            feats_rst = self.vae.decode(sampled_token_latents, lengths=lengths)

            joints_rst = self.feats2joints(feats_rst)
            feats_rst = self.datamodule.denormalize(feats_rst)
            gen_texts = ['<Motion_Placeholder>' for t in texts]
            outputs = {
                "texts": gen_texts,
                "feats": feats_rst,
                "joints": joints_rst,
                "length": lengths
            }
        elif task in ['m2t', 't2t']:
            outputs_tokens, cleaned_text = self.lm.generate_direct(
                texts,
                motion_tokens=motion_tokens_input,
                max_length=40,
                num_beams=1,
                do_sample=True,
                gen_mode='text',
                bad_words_ids=[[self.lm.som_id], [self.lm.eom_id]]
            )
            gen_texts = cleaned_text
            outputs = {
                "texts": gen_texts,
                "feats": None,
                "joints": None,
                "length": None,
            }
        else:
            assert False, f'{task} Not implemented yet'
            
        return outputs

    # def train_lm_forward(self, batch):
    #     feats_ref = batch["motion"]
    #     texts = batch["text"]
    #     lengths = batch["length"]
    #     tasks = batch["tasks"]

    #     outputs = self.lm(texts, feats_ref, self.vae, lengths, tasks)
    #     return {'outputs': outputs}
    def train_lm_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]

        outputs = self.lm(texts, feats_ref, self.vae, lengths, tasks)
        
        # ===== 디버깅 추가 =====
        print(f"[DEBUG] outputs type: {type(outputs)}")
        if hasattr(outputs, 'loss'):
            print(f"[DEBUG] outputs.loss: {outputs.loss}")
        else:
            print("[DEBUG] outputs has NO 'loss' attribute!")
        if hasattr(outputs, 'diff_loss'):
            print(f"[DEBUG] outputs.diff_loss: {outputs.diff_loss}")
        else:
            print("[DEBUG] outputs has NO 'diff_loss' attribute!")
        # ======================
        
        return {'outputs': outputs}
    
    @torch.no_grad()
    def val_t2t_forward(self, batch):
        texts = batch["text"]
        tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)
        
        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                stage='test',
                                                task='t2t',
                                                tasks=tasks,
                                                )

        rs_set = {
            "t_pred": outputs,
        }

        return rs_set

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_t2m_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                lengths=lengths,
                                                stage='test',
                                                tasks=tasks,
                                                )
        sampled_token_latents, motion_mask = self.lm.sample_tokens(
            outputs, feats_ref.device, 
            temperature=1.0, cfg=self.guidance_scale, 
            vae_mean_std_inv=self.vae.mean_std_inv)
        sampled_token_latents = sampled_token_latents.reshape(len(lengths), self.vae.latent_size, -1).permute(1,0,2)
        
        feats_rst = self.vae.decode(sampled_token_latents, lengths)
        feats_rst[motion_mask==1] = torch.zeros_like(feats_ref[0, ...])

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        with torch.no_grad():
            outputs = self.lm.generate_conditional(motion_feats=feats_ref,
                                                motion_encode_net=self.vae,
                                                lengths=lengths,
                                                task="m2t",
                                                stage='test')

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        with torch.no_grad():
            outputs = self.lm.generate_conditional(motion_feats=feats_ref,
                                                motion_encode_net=self.vae,
                                                lengths=lengths,
                                                task=task,
                                                stage='test')

        sampled_token_latents, motion_mask = self.lm.sample_tokens(
            outputs, feats_ref.device, 
            temperature=1.0, cfg=self.guidance_scale, 
            vae_mean_std_inv=self.vae.mean_std_inv)
        sampled_token_latents = sampled_token_latents.reshape(len(lengths), self.vae.latent_size, -1).permute(1,0,2)
        feats_rst = self.vae.decode(sampled_token_latents, lengths)
        feats_rst[motion_mask==1] = torch.zeros_like(feats_ref[0, ...])
        
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        motion_z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)
        recons_z, _ = self.vae.encode(feats_rst, lengths)
        
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        if dist_m is not None:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)

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
    def val_vae_forward(self, batch, split="train"):
        feats_ref = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(z, lengths)

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }

        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_adaptor_pretrain"
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
            elif self.hparams.stage in ["lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t","t2t"]:
                # Visualization
                if (self.current_epoch+1) % 99999 == 0 and batch_idx == 0 and self.global_rank == 0:
                    lengths = batch['length']
                    feats_ref, joints_ref = rs_set['m_ref'], rs_set['joints_ref']
                    feats_rst, joints_rst = rs_set['m_rst'], rs_set['joints_rst']
                    rand_save_idx = random.sample(range(feats_ref.shape[0]),self.vis_num)
                    for idd in rand_save_idx:
                        idx = idd % len(lengths)
                        output_dir = os.path.join(self.model_dir, 'validate_motion', f'epoch_{self.current_epoch}')
                        os.makedirs(output_dir, exist_ok=True)
                        keyid = (batch['fname'][idx]).split('/')[-1]
                        motion = batch['motion'][idx]
                        joint_ref = self.feats2joints(motion)
                        feat_ref, joint_ref = feats_ref[idx][:lengths[idx]], joints_ref[idx][:lengths[idx]]
                        feat_rst, joint_rst = feats_rst[idx][:lengths[idx]], joints_rst[idx][:lengths[idx]]
                        render_motion(joint_ref, joint_ref.cpu().numpy(), output_dir=output_dir, fname=f'{keyid}_gt',method='fast', fps=self.fps)
                        render_motion(joint_rst, joint_rst.cpu().numpy(), output_dir=output_dir, fname=f'{keyid}',method='fast')
                        np.savetxt(os.path.join(output_dir, f'{keyid}.txt'), [batch['text'][idx]], fmt='%s')
                
                # MultiModality evaluation separately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain"
                        ]:
                            word_embs = batch.get('word_embs')
                            pos_ohot = batch.get('pos_ohot')
                            text_lengths = batch.get('text_len')
                            if self.trainer.datamodule.is_mm and word_embs is not None:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "TMRMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            texts=batch["text"]
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    # ============ MRMetrics 수정됨! ============
                    elif metric == "MRMetrics":
                        # Get src and name from batch
                        src = batch.get('src', ['how2sign'] * len(lengths))
                        name = batch.get('name', batch.get('fname', [f'sample_{i}' for i in range(len(lengths))]))
                        
                        getattr(self.metrics, metric).update(
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
                    # ============ MRMetrics 수정 끝 ============
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_t2m", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain"
            ]:
                if batch_idx == 0 and self.global_rank == 0:
                    feats_ref = rs_set['m_ref']
                    gen_texts = rs_set["t_pred"]

                    rand_save_idx = random.sample(range(feats_ref.shape[0]),self.vis_num)
                    lengths = batch['length']
                    for idx in rand_save_idx:
                        output_dir = os.path.join(self.model_dir, 'validate_motion', f'epoch_{self.current_epoch}')
                        os.makedirs(output_dir, exist_ok=True)
                        keyid = (batch['fname'][idx]).split('/')[-1]

                        feat_ref = feats_ref[idx][:lengths[idx]]
                        joint_ref = self.feats2joints(self.datamodule.renorm4m(feat_ref))
                        render_motion(joint_ref, None, output_dir=output_dir, fname=f'{keyid}_gt',
                                        method='fast', fps=self.datamodule.fps)
                        np.savetxt(os.path.join(output_dir, f'{keyid}_gt.txt'), [batch['text'][idx]], fmt='%s')
                        np.savetxt(os.path.join(output_dir, f'{keyid}.txt'), [gen_texts[idx]], fmt='%s')
                        
                self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
                for metric in metrics_dicts:
                    if metric == "M2TMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["all_captions"],
                            lengths=rs_set['length'],
                            word_embs=batch.get("word_embs"),
                            pos_ohot=batch.get("pos_ohot"),
                            text_lengths=batch.get("text_len"),
                        )

        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.task == "t2m":
                return rs_set["m_rst"], rs_set["length"], rs_set["m_ref"], batch['text'], batch['fname']
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"], rs_set["m_ref"], rs_set['t_ref'], batch['fname']

        return loss