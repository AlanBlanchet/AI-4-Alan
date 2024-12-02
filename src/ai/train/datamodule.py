from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..dataset.collator.mask import masked_collator
from ..task.task import Task


class AIDataModule(LightningDataModule):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.dataset = task.dataset
        self.config = task.config.run.datamodule

    @property
    def necessary_params(self):
        return dict(collate_fn=masked_collator)

    def train_dataloader(self):
        return DataLoader(
            **self.config.model_dump(),
            **self.necessary_params,
            shuffle=self.task.train_shuffle,
            dataset=self.dataset.train(),
        )

    def val_dataloader(self):
        return DataLoader(
            **self.config.model_dump(),
            **self.necessary_params,
            dataset=self.dataset.val(),
        )
