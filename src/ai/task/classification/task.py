from functools import cached_property
from typing import ClassVar

import torch
import torch.nn.functional as F
from pydantic import computed_field

from ...dataset.label_map import LabelMap
from ..metrics import GroupedMetric
from ..task import TASK_TYPE, Task
from .metrics import ClassificationMetrics


class Classification(Task):
    alias: ClassVar[str] = "clf"

    @cached_property
    def model_kwargs(self):
        return dict(num_classes=len(self.label_map))

    @cached_property
    def label_map(self):
        self.info("Creating label map")
        return LabelMap(labels=self.dataset._labels)

    @cached_property
    def metrics(self):
        return GroupedMetric(
            lambda: ClassificationMetrics(num_classes=len(self.label_map)),
            ["train", "val"],
        )

    def setup_dataset(self, **kwargs):
        self.dataset.prepare(label_map=self.label_map)

    def default_loss(self, out: dict, batch: dict) -> dict:
        return F.cross_entropy(out["logits"], batch["labels"])

    def wrap_output(self, out: dict | torch.Tensor):
        if isinstance(out, torch.Tensor):
            return dict(logits=out)
        return out

    @computed_field
    @cached_property
    def task_type(self) -> TASK_TYPE:
        if self.type:
            self.info(f"Using task type {self.type}")
            return self.type

        if len(self.labels) > 1:
            self.info("Using multiclass task by default")
            return "multiclass"
        else:
            self.info("Using binary task by default")
            return "binary"

    def example(self, out: dict, item: dict, split: str):
        # TODO implement
        ...
