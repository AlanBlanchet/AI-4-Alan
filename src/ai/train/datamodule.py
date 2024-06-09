import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..dataset.base_dataset import BaseDataset

DL_WORKERS = min(2, os.cpu_count() - 4)

# DL_WORKERS = 0


class AIDataModule(LightningDataModule):
    def __init__(self, dataset: BaseDataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset.train(), batch_size=8, shuffle=True, num_workers=DL_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.val(),
            batch_size=8,
            shuffle=False,
            num_workers=DL_WORKERS,
        )
