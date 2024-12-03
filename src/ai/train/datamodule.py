from __future__ import annotations

from typing import Optional

import lightning as pl
from lightning import LightningDataModule
from pydantic import computed_field, field_validator
from torch.utils.data import DataLoader

from ..configs.base import Base
from ..configs.log import Color
from ..dataset.base_dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..utils.env import AIEnv

EXTRA_DM_KEYS = [
    "trainer",
    "_log_hyperparams",
    "prepare_data_per_node",
    "allow_zero_length_dataloader_with_multiple_devices",
    "name",
    "dataset",
]


class AIDataModule(Base, LightningDataModule):
    log_name = "datamodule"
    color = Color.gold

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    # Map these LightningDataModule keys to the current class
    trainer: Optional[pl.Trainer] = None
    _log_hyperparams: bool = False
    prepare_data_per_node: bool = False
    allow_zero_length_dataloader_with_multiple_devices: bool = False

    # Real fields
    dataset: BaseDataset

    num_workers: int = AIEnv.DEFAULT_NUM_PROC
    pin_memory: bool = True
    batch_size: int = 1
    val_batch_size: int = None
    worker_init_fn: Optional[callable] = None

    train_shuffle: bool = True
    _train_dataloader: Optional[DataLoader] = None
    _val_dataloader: Optional[DataLoader] = None

    def model_post_init(self, _):
        self.info("Using config\n", self.model_dump(exclude=set(EXTRA_DM_KEYS)))

        self._train_dataloader = DataLoader(
            **self.real_params,
            **self.necessary_params,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            dataset=self.dataset.get_train(),
        )
        self._val_dataloader = DataLoader(
            **self.real_params,
            **self.necessary_params,
            batch_size=self.val_batch_size,
            dataset=self.dataset.get_val(),
        )

    @computed_field
    @property
    def persistent_workers(self) -> bool:
        return self.num_workers > 0

    @property
    def necessary_params(self):
        return dict(collate_fn=masked_collator)

    @property
    def real_params(self):
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self.worker_init_fn,
        )

    @field_validator("val_batch_size", mode="before")
    @classmethod
    def validate_val_batch_size(cls, v, others):
        if v is None:
            return others.data["batch_size"]
        return v

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
