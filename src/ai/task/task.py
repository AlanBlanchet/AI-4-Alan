from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal, override

import numpy as np
import torch
import torch.nn as nn
import yaml
from pydantic import computed_field, field_validator

from ..configs import ActionEnum, Base, Color
from ..dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..nn import *  # noqa
from ..nn.compat.pretrained import Pretrained
from ..nn.modules import Module
from ..train.model import AIModule
from ..utils.env import AIEnv

TASK_TYPE = Literal["binary", "multiclass", "multilabel"]


class Task(Base, buildable=True):
    class Config:
        arbitrary_types_allowed = True

    log_name: ClassVar[str] = "task"
    color: ClassVar[str] = Color.magenta
    alias: ClassVar[str]

    dataset: BaseDataset
    model: Module
    checkpoint: Path = None

    train_shuffle: bool = True
    val_only_metrics: bool = False

    @computed_field()
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @override
    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {cls.alias}

    def model_post_init(self, _):
        self.log(f"Loading {self.spaced_name()} task")
        self.setup_dataset()

    def save_config(self):
        config = self.run_p / "config.yml"
        dump_config = self.run_p / "dump.yml"
        config.write_text(yaml.dump(self.root_config.config, sort_keys=False))
        dump_config.write_text(
            yaml.dump(self.root_config.model_dump(), sort_keys=False)
        )
        self.log(f"Saved configs to \nconfig {config}\ndump {dump_config}")

    @classmethod
    def model_kwargs(cls, params: dict): ...

    def model_dump(self, **kwargs):
        return super().model_dump(**kwargs, exclude=["dataset"])

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, config, others):
        cls.log("Setting up model")

        config = {
            **config,
            # **self.root_config.task.model.model_dump(),
            **cls.model_kwargs(others.data),
            # "train": self.train,
        }
        dataset = others.data["dataset"]
        ex = dataset.example()
        items = masked_collator([ex])
        input = dataset.extract_inputs(items)

        model: Module = Module.from_config(config)

        if isinstance(model, Pretrained):
            model.init_weights(*input)

        return model

    # TODO: fix this to be on each subclass
    @property
    def required_fields(self):
        if self.type == "classification":
            return ["labels"]
        elif self.type == "detection":
            return ["labels", "bbox"]
        else:
            raise ValueError(f"Unknown task type {self.type}")

    @cached_property
    def lightning_model(self):
        module = AIModule(self)
        torch.compile(module)
        return module

    @property
    def metric_params(self):
        return self.params.get("metrics", {})

    @property
    def train(self):
        return self.root_config.run.action == ActionEnum.fit

    def modules(self, **kwargs) -> dict[str, nn.Module]:
        return dict(metrics=self.metrics, **kwargs)

    def __call__(self):
        self.log("Running task")
        self.save_config()
        self.root_config.run()

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
        run_p = (
            AIEnv.runs_p
            / self.alias
            / self.dataset.name.replace("/", "_")
            / self.model.__class__.__name__
        )
        run_p = self._create_path(run_p)
        run_p.mkdir(exist_ok=True, parents=True)
        self.log(f"Setting run path to {run_p}")
        return run_p

    def map_params(self, item: dict) -> dict:
        dataset_conf = self.root_config.dataset
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
            self.example(
                self._extract_batch_item(out, item_idx),
                self._extract_batch_item(batch, item_idx),
                split,
            )
        # try:
        # except Exception as e:
        #     self.log(f"Error showing example: {e}")

        # Return the losses
        return losses

    @abstractmethod
    def example(self, pred: dict, item: dict, split: str): ...

    def _extract_batch_item(self, batch: dict, idx: int) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, (list, torch.Tensor, np.ndarray)):
                out[k] = v[idx]
            else:
                out[k] = v
        return out
