import os
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Generator, Self

import numpy as np
import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from neptune import Run
from pydantic import computed_field

from ..configs.base import Base


class AILogger(Logger, Base):
    log_name: ClassVar[str] = "ai-logger"

    all_loggers: dict = {}
    _run_name: Path = None

    def setup(self, run_name: str):
        self._run_name = run_name

    @computed_field
    @cached_property
    def _loggers(self) -> dict[str, Run | Any]:
        loggers = {}
        for name, logger in self.all_loggers.items():
            if name.lower() == "neptune":
                loggers[name.lower()] = Run(
                    name=self._run_name,
                    api_token=os.environ["NEPTUNE_API_TOKEN"],
                    **logger,
                )
                super().log(f"Added {name} logger with name {self._run_name}")
        return loggers

    @classmethod
    def build(cls, **kwargs) -> Self:
        return cls(all_loggers=kwargs)

    @property
    def name(self):
        return "GlobalLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for k, v in self._loggers.items():
            if k == "neptune":
                for name, value in metrics.items():
                    self.log(name, value, step=step, only=k)

    def log(self, name: str, value: Any, **kwargs: Any):
        only = kwargs.pop("only", None)
        loggers = self._loggers if only is None else {only: self._loggers[only]}
        for k, v in loggers.items():
            if k == "neptune":
                if isinstance(value, (torch.Tensor, np.ndarray, float, int)):
                    v[name].append(value)
                else:
                    v[name].upload(value)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.log(*args, **kwds)

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        for k, v in self._loggers.items():
            if k == "neptune":
                v.stop()
        pass

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        return self._loggers.values()

    def __getitem__(self, idx: int) -> Self:
        return list(self._loggers.values())[idx]

    class Config:
        arbitrary_types_allowed = True
