from __future__ import annotations

from functools import cached_property
from pprint import pformat
from typing import ClassVar

import torch
from lightning import Trainer
from pydantic import Field, field_validator

from ..configs import ActionEnum, Base, Color
from ..task.task import Task
from .callbacks.tqdm import CustomTQDMProgressBar
from .datamodule import AIDataModule, AIDataModuleConfig


class Runner(Base):
    class Config:
        """
        Required for the trainer
        """

        arbitrary_types_allowed = True

    log_name: ClassVar[str] = "runner"
    color: ClassVar[str] = Color.green

    # Shared
    task: Task

    datamodule: AIDataModule = Field(None, validate_default=True)
    action: ActionEnum
    lightning: dict = {}

    @field_validator("datamodule", mode="before")
    @classmethod
    def validate_datamodule(cls, v, others):
        v = v if v else {}
        config = AIDataModuleConfig(**v)
        return AIDataModule(task=others.data["task"], config=config)

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
        )

    def __call__(self):
        self.log(f"Dataset example sample :\n{pformat(self.dataset.example())}")
        self.log(f"Dataset :\n{pformat(self.dataset)}")

        if self.val_only:
            self.log("Validating with pytorch-lightning")
            self.trainer.validate(
                model=self.task.lightning_model, datamodule=self.datamodule
            )
        else:
            self.log("Training with pytorch-lightning")
            self.trainer.fit(
                model=self.task.lightning_model, datamodule=self.datamodule
            )

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
