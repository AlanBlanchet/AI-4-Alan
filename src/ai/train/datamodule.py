from __future__ import annotations

from lightning import LightningDataModule
from pydantic import BaseModel, computed_field
from torch.utils.data import DataLoader

from ..dataset.base_dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..task.task import Task
from ..utils.env import AIEnv


class AIDataModuleConfig(BaseModel):
    num_workers: int = AIEnv.DEFAULT_NUM_PROC
    pin_memory: bool = True
    batch_size: int = 1

    @property
    def persistent_workers(self):
        return self.num_workers > 0


class AIDataModule(BaseModel, LightningDataModule):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        use_enum_values = True
        validate_assignment = False

    task: Task
    config: AIDataModuleConfig

    def __init__(self, **kwargs):
        """
        We are overriding since we need to call both the parent and the LightningDataModule
        """
        super().__init__(**kwargs)
        LightningDataModule.__init__(self)

    @computed_field
    @property
    def dataset(self) -> BaseDataset:
        return self.task.dataset

    @property
    def necessary_params(self):
        return dict(collate_fn=masked_collator)

    def train_dataloader(self):
        return DataLoader(
            **self.config.model_dump(),
            **self.necessary_params,
            shuffle=self.task.train_shuffle,
            dataset=self.dataset.create_train(),
        )

    def val_dataloader(self):
        return DataLoader(
            **self.config.model_dump(),
            **self.necessary_params,
            dataset=self.dataset.create_val(),
        )
