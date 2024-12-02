from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn as nn

from ..configs.base import Base
from ..configs.log import Color
from ..configs.main import ActionEnum, MainConfig
from ..dataset.base_dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..dataset.env.environment import EnvironmentDataset
from ..dataset.huggingface import HuggingFaceDataset
from ..nn.compat.pretrained import Pretrained
from ..train.model import AIModule
from ..utils.env import AIEnv

TASK_TYPE = Literal["binary", "multiclass", "multilabel"]

if TYPE_CHECKING:
    pass


class Task(Base):
    log_name: ClassVar[str] = "task"
    color: ClassVar[str] = Color.magenta

    config: MainConfig
    dataset: BaseDataset

    train_shuffle: bool = True

    def model_post_init(self, _):
        self.log(f"Loading {self.name} task")
        self.setup_dataset()

    @property
    def model_kwargs(self): ...

    @cached_property
    def model(self):
        self.log("Setting up model")
        config = {
            **self.config.task.model.model_dump(),
            **self.model_kwargs,
            "train": self.train,
        }
        ex = self.dataset.example()
        items = masked_collator([ex])
        input = self.dataset.extract_inputs(items)

        model: nn.Module = Base.from_config(config)

        if isinstance(model, Pretrained):
            model.init_weights(*input)

        return model

    @property
    def lightning_model(self):
        module = AIModule(self)
        torch.compile(module)
        return module

    @property
    def metric_params(self):
        return self.params.get("metrics", {})

    @property
    def val_only_metrics(self):
        return self.config.task.val_only_metrics

    @property
    def train(self):
        return self.config.run.action == ActionEnum.fit

    def modules(self, **kwargs) -> dict[str, nn.Module]:
        return dict(metrics=self.metrics, **kwargs)

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

    def _create_path(self, path: Path):
        i = 1
        p = path / f"{i}"
        while p.exists():
            i += 1
            p = path / f"{i}"
        return p

    @cached_property
    def run_p(self):
        run_p = AIEnv.runs_p / self.alias / self.dataset.name.replace("/", "_")
        run_p = self._create_path(run_p)
        run_p.mkdir(exist_ok=True, parents=True)
        self.log(f"Setting run path to {run_p}")
        return run_p

    def map_params(self, item: dict) -> dict:
        dataset_conf = self.config.dataset
        map = dataset_conf.map_params
        return self.dataset.parse_items(item, map)

    @abstractmethod
    def default_loss(self, out: dict, batch: dict) -> dict: ...

    def wrap_output(self, out: dict):
        return out

    def process(
        self,
        model: AIModule,
        batch: dict[str, torch.Tensor],
        split: str,
        item_idx: int | None,
    ) -> dict:
        # Extract model required inputs
        inputs = self.dataset.extract_inputs(batch)

        # Forward pass
        out = model(inputs)

        # Format as dict
        out = self.wrap_output(out)

        # Compute the loss
        if hasattr(self.model, "compute_loss"):
            losses = self.model.compute_loss(out, batch)
        else:
            losses = self.default_loss(out, batch)

        # Format losses
        if isinstance(losses, torch.Tensor):
            losses = dict(loss=losses)
        elif isinstance(losses, (list, tuple)):
            losses = dict(loss=losses[0], other=losses[1:])

        # Postprocess the output
        if hasattr(self.model, "postprocess"):
            out = self.postprocess(out, batch)

        # Update metrics
        self.metrics.update(out, batch, split=split)

        # Show chosen sample
        if item_idx is not None:
            try:
                self.example(
                    self._extract_batch_item(out, item_idx),
                    self._extract_batch_item(batch, item_idx),
                    split,
                )
            except Exception as e:
                self.log(f"Error showing example: {e}")

        # Return the losses
        return losses

    @abstractmethod
    def example(self, pred: dict, item: dict, split: str): ...

    def _extract_batch_item(self, batch: dict, idx: int) -> dict:
        return {k: v[idx] for k, v in batch.items()}

    @classmethod
    def run(cls, config: MainConfig):
        # Prevent circular imports
        from ..train.runner import Runner
        from .classification.task import Classification
        from .detection.task import Detection
        from .rl.task import ReinforcementLearning

        # Init dataset
        dataset: BaseDataset = None
        source = config.dataset.source
        if source == "hf":
            dataset = HuggingFaceDataset(config=config)
        elif source == "gym":
            dataset = EnvironmentDataset(config=config)
        else:
            raise NotImplementedError(
                f"Dataset source {config.dataset.source} not implemented"
            )

        task: Task = None
        type = config.task.type
        if type == "classification":
            task = Classification(config=config, dataset=dataset)
        elif type == "detection":
            task = Detection(config=config, dataset=dataset)
        elif type == "reinforcementlearning":
            task = ReinforcementLearning(config=config, dataset=dataset)
        else:
            raise NotImplementedError(f"Task type {config.task.type} not implemented")

        runner = Runner(task=task, config=config)
        runner()
