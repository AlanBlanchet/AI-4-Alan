from functools import partial
from typing import Callable

from myconf import F, MyConf
from torch.utils.data import DataLoader

from .dataset import Data


class SmartDataLoader(MyConf):
    data: Data
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0

    _dataloader_cls: Callable = F(lambda self: self._make_partial_dataloader())

    def _make_partial_dataloader(self):
        return partial(
            DataLoader,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def setup(self, stage: str):
        super().setup(stage)

    def train_dataloader(self):
        return self._dataloader_cls(dataset=self.data.get_train())
