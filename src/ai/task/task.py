from __future__ import annotations

import json
import shutil
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal, override

import numpy as np
import torch
import yaml
from lightning import LightningModule
from pydantic import computed_field, field_validator

from ..configs import ActionEnum, Color
from ..nn import *  # noqa
from ..nn.compat.module import ModuleConfig
from ..nn.compat.pretrained import PretrainedConfig
from ..utils.env import AIEnv
from .metrics import GroupedMetric, Metric
from .trainer import TaskModule

TASK_TYPE = Literal["binary", "multiclass", "multilabel"]


class EmptyMetric(Metric):
    def update(self, *args, **kwargs): ...

    def compute(self, **kwargs):
        return {}


class Task(TaskModule, buildable=False):
    log_name: ClassVar[str] = "task"
    color: ClassVar[str] = Color.magenta
    alias: ClassVar[str]

    metrics: GroupedMetric = GroupedMetric(
        lambda: EmptyMetric(),
        ["train", "val"],
    )
    checkpoint: Path = None

    val_only_metrics: bool = False

    @override
    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {cls.alias}

    @classmethod
    def all_names(self):
        return [task.name for task in self.all()]

    @classmethod
    def get_all_valid_tasks(cls, field: str, dtype: str):
        for task in cls.all():
            if task.is_valid(field, dtype):
                yield task

    @classmethod
    def model_kwargs(cls, params: dict): ...

    @classmethod
    def datamodule_kwargs(cls, params: dict): ...

    @computed_field()
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def metric_params(self):
        return self.params.get("metrics", {})

    @property
    def task_action(self):
        return self.root_config.run.action == ActionEnum.fit

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
    def run_p(self):
        run_p = (
            AIEnv.runs_p
            / self.alias
            / self.dataset.name.replace("/", "_")
            / self.model.__class__.__name__
        )
        run_p = self._create_path(run_p)
        run_p.mkdir(exist_ok=True, parents=True)
        self.info(f"Setting run path to {run_p}")
        return run_p

    def init(self):
        super().init()
        # Move the logs to the run path
        shutil.move(AIEnv.tmp_log_p, self.run_p / "log")
        # Load the dataset
        self.info(f"Loading {self.spaced_name()} task")
        self.setup_dataset()

    @abstractmethod
    def setup_dataset(self, **kwargs): ...

    @abstractmethod
    def default_loss(self, out: dict, batch: dict) -> dict: ...

    @abstractmethod
    def example(self, pred: dict, item: dict, split: str): ...

    def save_config(self):
        config = self.run_p / "config.yml"
        dump_config = self.run_p / "dump.yml"
        config.write_text(yaml.dump(self.root_config.config, sort_keys=False))
        obj = self.root_config.model_dump(exclude={"config"})
        try:
            dump_config.write_text(yaml.dump(obj, sort_keys=False))
        except Exception as e:
            self.error(f"Error saving dump config:\n{e}\nTrying to save as json")
            dump_config.with_suffix(".json").write_text(json.dumps(obj, indent=2))

        self.info(f"Saved configs to \nconfig {config}\ndump {dump_config}")

    def _create_path(self, path: Path):
        i = 1
        p = path / f"{i}"
        while p.exists():
            i += 1
            p = path / f"{i}"
        return p

    def map_params(self, item: dict) -> dict:
        dataset_conf = self.root_config.dataset
        map = dataset_conf.map_params
        return self.dataset.parse_items(item, map)

    def wrap_output(self, out: dict):
        return out

    def process(
        self,
        model: LightningModule,
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
        #     self.info(f"Error showing example: {e}")

        # Return the losses
        return losses

    def _extract_batch_item(self, batch: dict, idx: int) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, (list, torch.Tensor, np.ndarray)):
                out[k] = v[idx]
            else:
                out[k] = v
        return out


class DimensionConfig(ModuleConfig):
    num_channels: int
    fixed_size: list[int] = None

    @field_validator("fixed_size", mode="before")
    @classmethod
    def validate_fixed_size(cls, value):
        if isinstance(value, int):
            return [value, value]
        if value is None and issubclass(cls, PretrainedConfig):
            return cls.pretrained[0].weights.fixed_size
        return value


class ClassificationConfig(ModuleConfig):
    num_classes: int = None

    @field_validator("num_classes", mode="before")
    @classmethod
    def validate_num_classes(cls, value):
        if value is None and issubclass(cls, PretrainedConfig):
            return cls.pretrained_recommendations[0].weights.num_classes
        return value
