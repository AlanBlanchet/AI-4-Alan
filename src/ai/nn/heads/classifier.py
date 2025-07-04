from typing import Generic, TypeVar

import torch
import torch.nn as nn

from myconf import Cast, F

from ...data.batch import Batch
from ...data.task.classification import ClassificationData
from ...modality.modality import Modality
from ..compat.backbone import IsBackboneMixin
from ..compat.trainable import Trainable, requires_prepared
from .head import Head

T = TypeVar("T", bound=Modality)


class Classifier(IsBackboneMixin, Head, Trainable, Generic[T]):
    _data: ClassificationData = F(init=False)

    def prepare(self):
        self._data.prepare()

    def classifier(self):
        return self.change_mode(Classifier, self.classify_logits)

    @requires_prepared
    def create_head(self):
        return nn.Linear(self.out_dim, self._data.num_classes)

    def classify_logits(self, modality: Cast[T, Batch[T]]) -> torch.Tensor:
        feats = self.features(modality)
        neck = self.bottleneck(feats[-1])
        return self.head(neck)

    @requires_prepared
    def classify(self, modality: Cast[T, Batch[T]]) -> torch.Tensor:
        out = self.classify_logits(modality)
        pred = self.postprocess(out)
        return self._data.labels[pred]

    @staticmethod
    def loss(x: torch.Tensor, labels: torch.Tensor):
        return nn.functional.cross_entropy(x, labels)

    @staticmethod
    def postprocess(x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1)
