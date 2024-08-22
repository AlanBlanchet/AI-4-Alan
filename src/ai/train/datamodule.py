import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..dataset.base_dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..utils.env import AIEnv

DL_WORKERS = max(2, os.cpu_count() - 4)
BATCH_SIZE = 16

DL_WORKERS = 0


class AIDataModule(LightningDataModule):
    def __init__(self, dataset: BaseDataset, params: dict):
        super().__init__()
        self.dataset = dataset
        self.params = params

    def train_dataloader(self):
        params = self.params.copy()
        num_workers = params.pop("num_workers", AIEnv.DEFAULT_NUM_PROC)
        return DataLoader(
            self.dataset.train(),
            shuffle=True,
            pin_memory=True,
            **params,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            collate_fn=masked_collator,
        )

    def val_dataloader(self):
        params = self.params.copy()
        num_workers = params.pop("num_workers", AIEnv.DEFAULT_NUM_PROC)
        return DataLoader(
            self.dataset.val(),
            shuffle=False,
            pin_memory=True,
            **params,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            collate_fn=masked_collator,
        )
