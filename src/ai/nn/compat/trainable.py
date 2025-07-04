from __future__ import annotations

from functools import wraps
from typing import ClassVar, Self

import lightning as L
from torch import optim

from myconf.core import F

from ...data.batch import Batch
from ...data.dataloader import SmartDataLoader
from ...data.dataset import Data
from ..compat.module import Module


def requires_prepared(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._data.prepare()
        return func(self, *args, **kwargs)

    return wrapper


class Trainable(Module):
    default_data: ClassVar[Data]

    _data: Data = F(lambda self: self.default_data)

    class LightningModule(L.LightningModule):
        def __init__(self, model: Self):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

        def step(self, batch, batch_idx):
            step_data = self.model.step(batch)
            return step_data

        def training_step(self, batch, batch_idx):
            return self.step(batch, batch_idx)

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=1e-3)

    class LightningDataModule(L.LightningDataModule):
        def __init__(self, data: Data):
            super().__init__()

            data.prepare()
            self.dataloader = SmartDataLoader(data=data)

        def setup(self, stage: str): ...

        def train_dataloader(self):
            return self.dataloader.train_dataloader()

    def prepare(self): ...

    def data(self, data: Data):
        self._data = data
        return self

    @classmethod
    def step(self, model: Self, batch: Batch):
        out = model(batch)
        return out

    def fit(self):
        trainer = L.Trainer()

        pl_module = self.LightningModule(self)

        trainer.fit(pl_module, train_dataloaders=self.LightningDataModule(self._data))
