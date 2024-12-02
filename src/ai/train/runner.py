from __future__ import annotations

from functools import cached_property
from pprint import pformat
from typing import ClassVar

import torch
from lightning import Trainer
from pydantic import computed_field

from ..configs.base import Base
from ..configs.main import MainConfig
from ..configs.log import Color
from ..task.task import Task

# from ..dataset.base_dataset import BaseDataset
from .datamodule import AIDataModule


class Runner(Base):
    log_name: ClassVar[str] = "runner"
    color: ClassVar[str] = Color.green

    task: Task
    config: MainConfig

    @property
    def dataset(self):
        # Helper
        return self.task.dataset

    @computed_field
    @cached_property
    def trainer(self) -> Trainer:
        lightning = self.config.run.lightning

        if hasattr(self.task, "logger"):
            logger = self.task.logger
            if logger:
                lightning["logger"] = [logger]

        # Training configuration
        # TODO add as config
        torch.set_float32_matmul_precision("medium")

        return Trainer(**lightning, default_root_dir=self.task.run_p)

    @computed_field
    @cached_property
    def datamodule(self) -> AIDataModule:
        return AIDataModule(task=self.task)

    def __call__(self):
        self.log(f"Dataset example sample :\n{pformat(self.dataset.example())}")
        self.log(f"Dataset :\n{pformat(self.dataset)}")

        if self.config.run.val_only:
            self.log("Validating with pytorch-lightning")
            self.trainer.validate(self.task.lightning_model, datamodule=self.datamodule)
        else:
            self.log("Training with pytorch-lightning")
            self.trainer.fit(self.task.lightning_model, datamodule=self.datamodule)
