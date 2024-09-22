from abc import ABC, abstractmethod
from typing import Literal

from .base import AdvancedBase


class PretrainedSourceConfig(AdvancedBase):
    weights: str | None = None
    source: Literal["timm", "torch", "torchvision"] | None = None


class PretrainedConfig(AdvancedBase):
    pretrained: bool | str | PretrainedSourceConfig = False


class ClassificationConfig(AdvancedBase):
    num_classes: int


class Backbone(ABC):
    @abstractmethod
    def features(self): ...
