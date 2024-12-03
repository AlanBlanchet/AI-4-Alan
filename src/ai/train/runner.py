from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import ClassVar

import torch
from lightning import Trainer
from pydantic import field_validator

from ..configs import ActionEnum, Base, Color
from ..task.task import Task
from .callbacks.tqdm import CustomTQDMProgressBar


class Runner(Base):
    model_config = {"arbitrary_types_allowed": True}

    log_name: ClassVar[str] = "runner"
    color: ClassVar[str] = Color.green

    # Shared
    task: Task

    action: ActionEnum
    lightning: dict = {}
    checkpoint: Path = None

    # @field_validator("checkpoint", mode="before")
    # @classmethod
    # def validate_checkpoint(cls, v, others):
    #     # TODO change this
    #     if isinstance(v, bool):
    #         return others.data["task"].run_p / "checkpoints"
    #     return v

    @property
    def dataset(self):
        return self.task.dataset

    @cached_property
    def trainer(self) -> Trainer:
        lightning = self.lightning

        if hasattr(self.task, "logger"):
            logger = self.task.logger
            if logger:
                lightning["logger"] = [logger]

        # Training configuration
        # TODO add as config
        torch.set_float32_matmul_precision("medium")

        return Trainer(
            **lightning,
            default_root_dir=self.task.run_p,
            callbacks=[CustomTQDMProgressBar()],
            num_sanity_val_steps=0,
        )

    def __call__(self):
        self.task.save_config()

        self.info("Dataset example sample :\n", self.dataset.example)
        self.info("Dataset :\n", self.dataset)

        if self.checkpoint:
            self.task.load(self.checkpoint)

        if self.val_only:
            self.info("Validating with pytorch-lightning")
            self.trainer.validate(model=self.task, datamodule=self.task.datamodule)
        else:
            self.info("Training with pytorch-lightning")
            self.trainer.fit(model=self.task, datamodule=self.task.datamodule)

    @property
    def val_only(self):
        return self.action == ActionEnum.val

    @field_validator("action", mode="before")
    def _flexible_action(cls, v):
        match v:
            case "train":
                return ActionEnum.fit
            case "eval":
                return ActionEnum.val
            case _:
                return ActionEnum[v]
