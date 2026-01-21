"""
SOKE-style callbacks for SignGPT3 training
"""
import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks


def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    
    # Logging metric monitor (for progress logger)
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        # MRMetrics - per dataset
        "how2sign_MPJPE_PA_hand": "Metrics/how2sign_MPJPE_PA_hand",
        "how2sign_MPVPE_PA_all": "Metrics/how2sign_MPVPE_PA_all",
        "how2sign_feat_error": "Metrics/how2sign_feat_error",
        "csl_MPJPE_PA_hand": "Metrics/csl_MPJPE_PA_hand",
        "csl_MPVPE_PA_all": "Metrics/csl_MPVPE_PA_all",
        "csl_feat_error": "Metrics/csl_feat_error",
        "phoenix_MPJPE_PA_hand": "Metrics/phoenix_MPJPE_PA_hand",
        "phoenix_MPVPE_PA_all": "Metrics/phoenix_MPVPE_PA_all",
        "phoenix_feat_error": "Metrics/phoenix_feat_error",
    }
    callbacks.append(
        progressLogger(logger, metric_monitor=metric_monitor, log_every_n_steps=1))

    # Base checkpoint params
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True,
        'save_on_train_epoch_end': False,  # Save on validation end
    }
    
    # Save last checkpoint
    callbacks.append(ModelCheckpoint(**checkpointParams))

    # Metric-based checkpoints (SOKE style)
    metrics = cfg.METRIC.TYPE
    
    # Define metric monitor map for each metric type
    metric_monitor_map = {
        'MRMetrics': {
            'Metrics/how2sign_MPJPE_PA_hand': {
                'abbr': 'how2sign_MPJPE_PA_hand',
                'mode': 'min'
            },
            'Metrics/csl_MPJPE_PA_hand': {
                'abbr': 'csl_MPJPE_PA_hand',
                'mode': 'min'
            },
            'Metrics/phoenix_MPJPE_PA_hand': {
                'abbr': 'phoenix_MPJPE_PA_hand',
                'mode': 'min'
            },
            'Metrics/how2sign_feat_error': {
                'abbr': 'how2sign_feat_error',
                'mode': 'min'
            },
            'Metrics/csl_feat_error': {
                'abbr': 'csl_feat_error',
                'mode': 'min'
            },
            'Metrics/phoenix_feat_error': {
                'abbr': 'phoenix_feat_error',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'how2sign_DTW_lhand',
                'mode': 'min'
            },
            'Metrics/csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'csl_DTW_lhand',
                'mode': 'min'
            },
            'Metrics/phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'phoenix_DTW_lhand',
                'mode': 'min'
            },
        },
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
    }

    # Create checkpoint callbacks for each monitored metric
    for metric in metrics:
        if metric in metric_monitor_map:
            metric_monitors = metric_monitor_map[metric]

            for metric_monitor, metric_info in metric_monitors.items():
                ckpt_params = {
                    'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
                    'filename': f"{metric_info['mode']}-{metric_info['abbr']}" + "{epoch}",
                    'monitor': metric_monitor,
                    'mode': metric_info['mode'],
                    'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
                    'save_top_k': 1,
                    'save_last': False,
                    'save_on_train_epoch_end': False,
                }
                callbacks.append(ModelCheckpoint(**ckpt_params))
                
    return callbacks


class progressBar(RichProgressBar):
    def __init__(self):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        if self.logger:
            self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        if self.logger:
            self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            if self.logger:
                self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        if self.logger:
            self.logger.info(line)