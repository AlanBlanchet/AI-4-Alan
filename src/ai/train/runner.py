from __future__ import annotations

from functools import cached_property
from pprint import pformat
from typing import ClassVar

from lightning import Trainer
from pydantic import computed_field

from ..configs.base import Base
from ..configs.main import MainConfig
from ..task.task import Task

# from ..dataset.base_dataset import BaseDataset
from .datamodule import AIDataModule


class Runner(Base):
    _: ClassVar[str] = "trainer"

    log_name: ClassVar[str] = "trainer"

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

        return Trainer(**lightning, default_root_dir=self.task.run_p)

    # @computed_field
    # @cached_property
    # def module(self) -> AIModule:
    # path = Path(__file__).parents[3] / "lightning_logs"
    # recent_ckpts = sorted(path.rglob("*.ckpt"), key=lambda x: x.lstat().st_mtime)
    # if len(recent_ckpts) > 0 and LOAD_CKPT:
    #     self.log(f"Loaded model from {recent_ckpts[-1]}")
    #     return AIModule.load_from_checkpoint(recent_ckpts[-1])
    # else:
    # module = AIModule(self.task)
    # self.log("Compiling model")
    # torch.compile(module)
    # return module

    @computed_field
    @cached_property
    def datamodule(self) -> AIDataModule:
        return AIDataModule(dataset=self.dataset, params=self.config.run.datamodule)

    def __call__(self):
        self.log(f"Dataset example sample :\n{pformat(self.dataset.example())}")
        self.log(f"Dataset :\n{pformat(self.dataset)}")

        if self.config.task.val_only:
            self.log("Validation only")
            self.trainer.validate(self.task.lightning_model, datamodule=self.datamodule)
        else:
            self.trainer.fit(self.task.lightning_model, datamodule=self.datamodule)
