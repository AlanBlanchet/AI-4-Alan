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

from ...configs.base import Base
from ..metrics import GroupedMetric
from ..task import TASK_TYPE, Task
from .label_map import LabelMap
from .metrics import ClassificationMetrics
from .model import ClassificationModel


class Classification(Task):
    name: ClassVar[str] = "classification"
    alias: ClassVar[str] = "clf"
    # input: str
    # labels: str = "labels"
    # label_map_params: dict = {}
    # type: TASK_TYPE = None

    # def setup_model(self, config):

    # ex = self.dataset.example()

    # out = model(ex["input"].unsqueeze(dim=0))[-1]

    # self.log(
    #     f"Calculated in_features from input{tuple(ex['input'].shape)} for classification head: {tuple(out.shape[1:])}"
    # )

    # in_features = torch.tensor(out.shape[1:]).sum().item()
    # model = ClassificationModel(model, in_features, len(self.label_map))

    # print(model)

    # exit(0)

    @cached_property
    def model(self):
        self.log("Setting up model")
        num_classes = len(self.label_map)

        # Treat as if model doesn't have a classification head and isn't initialized
        model: nn.Module = Base.from_config(
            dict(**self.config.task.model.model_dump(), num_classes=num_classes)
        )

        return model

        model_cls = self.model_info["model_cls"]
        model_config = model_cls.config
        print(model_config)
        model_ = model_cls(num_classes=len(self.label_map))

        ex = self.dataset.example()

        out = model_(ex["input"].unsqueeze(dim=0))

        self.log(f"Calculated in_features for classification head: {out.shape[1:]}")

        in_features = torch.tensor(out.shape[1:]).sum().item()
        return ClassificationModel(model_, in_features, len(self.label_map))

    @cached_property
    def label_map(self):
        self.log("Creating label map")
        return LabelMap(labels=self.dataset._labels)  # **self.label_map_params)

    @cached_property
    def preprocessing(self):
        img_size = self.config.task.params["img_size"]
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
        # bbox_params = BboxParams(
        #     format="pascal_voc",
        #     label_fields=["labels"],
        # )
        if val:
            return Compose(
                [self.preprocessing, Normalize(), ToTensorV2()],
                **compose_params,
                # bbox_params=bbox_params,
            )
        return Compose(
            [self.preprocessing, self.augmentations, Normalize(), ToTensorV2()],
            **compose_params,
            # bbox_params=bbox_params,
        )

    def setup_dataset(self, **kwargs):
        self.dataset.prepare(
            label_map=self.label_map,
            # params=self.params,
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

        return dict(loss=loss)
