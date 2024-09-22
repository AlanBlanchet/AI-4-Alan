from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import ClassVar, Literal, Self

import torch
import torch.nn as nn

from ..configs.base import Base
from ..configs.main import MainConfig
from ..dataset.base_dataset import BaseDataset
from ..dataset.hf import HuggingFaceDataset
from ..train.model import AIModule
from ..utils.env import AIEnv

TASK_TYPE = Literal["binary", "multiclass", "multilabel"]


class Task(Base):
    log_name: ClassVar[str] = "task"

    config: MainConfig
    dataset: BaseDataset

    # name: ClassVar[str]
    # # alias: ClassVar[str]

    # logger: AILogger = None
    # dataset: BaseDataset
    # params: dict
    # # model_info: dict
    # model_config_: dict = None

    def model_post_init(self, _):
        self.log(f"Loading {self.name} task")
        self.setup_dataset()

    # @classmethod
    # def build(cls, **kwargs):
    #     # Model can't be created at the start since it requires the task to be initialized
    #     # model_info = kwargs.pop("model")
    #     return super().build(**kwargs)

    def augmentations(): ...

    @property
    def model(self) -> nn.Module: ...

    @property
    def lightning_model(self):
        module = AIModule(self)
        torch.compile(module)
        return module

    @property
    def metric_params(self):
        return self.params.get("metrics", {})

    @property
    def metric_val_only(self):
        return self.metric_params.get("val_only", False)

    def modules(self, **kwargs) -> dict[str, nn.Module]:
        return dict(metrics=self.metrics, **kwargs)

    def setup(self) -> nn.Module:
        self.setup_dataset()
        if self.logger:
            self.logger.setup(self.run_name)
        return self.setup_model()

    # @abstractmethod
    # def setup_model(self, model_cls: Type): ...

    @abstractmethod
    def setup_dataset(self, **kwargs): ...

    @classmethod
    def all_names(self):
        return [task.name for task in self.all()]

    @classmethod
    def get_all_valid_tasks(cls, field: str, dtype: str):
        for task in cls.all():
            if task.is_valid(field, dtype):
                yield task

    @cached_property
    def run_name(self):
        name = f"{self.alias}-{self.dataset.name.replace('/', '_')}"
        self.log(f"Creating run '{name}'")
        return name

    @cached_property
    def run_p(self):
        run_p = AIEnv.runs_p / self.alias / self.dataset.name.replace("/", "_")
        run_p.mkdir(exist_ok=True, parents=True)
        self.log(f"Setting run path to {run_p}")
        return run_p

    @classmethod
    def from_config(cls, config: dict) -> Self:
        # Do not create the model as it depends on many variables relative to the specific task
        model_config = config.pop("model")
        task = super().from_config(config)
        task.model_config_ = model_config
        return task

    @classmethod
    def run(cls, config: MainConfig):
        from ..train.runner import Runner
        from .classification.task import Classification

        # Init dataset
        dataset: BaseDataset = None
        if config.dataset.source == "hf":
            dataset = HuggingFaceDataset(config=config.dataset, task_config=config.task)
        else:
            raise NotImplementedError(
                f"Dataset source {config.dataset.source} not implemented"
            )

        task: Task = None
        if config.task.type == "classification":
            task = Classification(config=config, dataset=dataset)

        runner = Runner(task=task, config=config)
        runner()
