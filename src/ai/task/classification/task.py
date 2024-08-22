from functools import cached_property
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (
    Compose,
    GaussNoise,
    MotionBlur,
    Normalize,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
    ShiftScaleRotate,
)
from albumentations.pytorch import ToTensorV2
from pydantic import computed_field

from ...registry.registers import SOURCE
from ..metrics import GroupedMetric
from ..task import TASK_TYPE, Task
from .label_map import LabelMap
from .metrics import ClassificationMetrics
from .model import ClassificationModel


@SOURCE.register
class Classification(Task):
    name: ClassVar[str] = "classification"
    alias: ClassVar[str] = "clf"
    input: str
    labels: str = "labels"
    label_map_params: dict = {}
    type: TASK_TYPE = None

    @property
    def model(self):
        self.log("Setting up model")
        model_cls = self.model_info["model_cls"]
        model_ = model_cls(num_classes=len(self.label_map))
        meta = self.model_info.get("meta", {})
        requires = meta.get("requires", [])

        if "num_classes" in requires or not self.model_info.get("adapt", True):
            # The model already has a classification head
            return model_

        ex = self.dataset.example()

        out = model_(ex["input"].unsqueeze(dim=0))

        self.log(f"Calculated in_channels for classification head: {out.shape[1:]}")

        in_channels = torch.tensor(out.shape[1:]).sum().item()
        return ClassificationModel(model_, in_channels, len(self.label_map))

    @cached_property
    def label_map(self):
        self.log("Creating label map")
        return LabelMap(labels=self.dataset._labels, **self.label_map_params)

    @cached_property
    def preprocessing(self):
        img_size = self.params["img_size"]
        return Compose([Resize(img_size, img_size)])

    @cached_property
    def augmentations(self):
        return Compose(
            [
                ShiftScaleRotate(p=0.5),
                RandomBrightnessContrast(p=0.5),
                MotionBlur(p=0.5),
                GaussNoise(p=0.5),
                RandomGamma(p=0.5),
            ]
        )

    @cached_property
    def metrics(self):
        return GroupedMetric(
            lambda: ClassificationMetrics(num_classes=len(self.label_map)),
            ["train", "val"],
        )

    def get_transforms(self, val=False, **compose_params):
        if val:
            return Compose(
                [self.preprocessing, Normalize(), ToTensorV2()], **compose_params
            )
        return Compose(
            [self.preprocessing, self.augmentations, Normalize(), ToTensorV2()],
            **compose_params,
        )

    def setup_dataset(self, **kwargs):
        self.dataset.prepare(
            input=self.input,
            outputs=dict(labels=self.labels, **kwargs),
            label_map=self.label_map,
            params=self.params,
            get_transforms=self.get_transforms,
        )

    @computed_field
    @cached_property
    def task_type(self) -> TASK_TYPE:
        if self.type:
            self.log(f"Using task type {self.type}")
            return self.type

        if len(self.labels) > 1:
            self.log("Using multiclass task by default")
            return "multiclass"
        else:
            self.log("Using binary task by default")
            return "binary"

    def process(
        self, model: nn.Module, batch: dict[str, torch.Tensor], split: str
    ) -> dict:
        input = batch["input"]
        labels = batch["labels"]

        out: torch.Tensor = model(input)

        loss = F.cross_entropy(out, labels)

        self.metrics.update(out, labels, split=split)

        return loss
