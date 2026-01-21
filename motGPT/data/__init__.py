"""
motGPT Data Module
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BASEDataModule(pl.LightningDataModule):
    """Base DataModule class for all datasets."""
    
    def __init__(self, collate_fn=None):
        super().__init__()
        
        if collate_fn is not None:
            self.dataloader_options = {"collate_fn": collate_fn}
        else:
            self.dataloader_options = {}
        
        self.persistent_workers = True
        self.is_mm = False
        
        # Dataset placeholders
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def get_sample_set(self, overrides={}):
        """Get a sample dataset for initialization."""
        sample_params = {**self.hparams, **overrides}
        return self.Dataset(**sample_params)
    
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(
                split=self.cfg.TRAIN.SPLIT,
                **self.hparams
            )
        return self._train_dataset
    
    @property
    def val_dataset(self):
        if self._val_dataset is None:
            params = dict(self.hparams)
            params['split'] = self.cfg.EVAL.SPLIT
            self._val_dataset = self.DatasetEval(**params)
        return self._val_dataset
    
    @property
    def test_dataset(self):
        if self._test_dataset is None:
            params = dict(self.hparams)
            params['split'] = self.cfg.TEST.SPLIT
            self._test_dataset = self.DatasetEval(**params)
        return self._test_dataset
    
    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            **self.dataloader_options,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.EVAL.BATCH_SIZE,
            num_workers=self.cfg.EVAL.NUM_WORKERS,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            **self.dataloader_options,
        )
    
    def test_dataloader(self):
        batch_size = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.TEST.NUM_WORKERS,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            **self.dataloader_options,
        )


# Import dataset modules
from .HumanML3D import HumanML3DDataModule
from .H2S import H2SDataModule  # Sign Language DataModule 추가

# Import humanml datasets
from .humanml import (
    Text2MotionDataset,
    Text2MotionDatasetCB,
    Text2MotionDatasetToken,
    MotionDataset,
    MotionDatasetVQ,
    Text2MotionDatasetCBV3,
    Text2MotionDatasetEvalV3,
)

# Import sign language datasets
from .signlang import (
    SignMotionDataset,
    SignText2MotionDataset,
    SignText2MotionDatasetEval,
)

# Import utilities
from .utils import humanml3d_collate


__all__ = [
    # Base
    'BASEDataModule',
    
    # DataModules
    'HumanML3DDataModule',
    'H2SDataModule',  # 추가
    
    # HumanML3D Datasets
    'Text2MotionDataset',
    'Text2MotionDatasetCB',
    'Text2MotionDatasetToken',
    'MotionDataset',
    'MotionDatasetVQ',
    'Text2MotionDatasetCBV3',
    'Text2MotionDatasetEvalV3',
    
    # Sign Language Datasets (추가)
    'SignMotionDataset',
    'SignText2MotionDataset',
    'SignText2MotionDatasetEval',
    
    # Collate functions
    'humanml3d_collate',
]