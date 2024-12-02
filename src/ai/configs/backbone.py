from __future__ import annotations

from abc import abstractmethod

from pydantic import Field, field_validator

from ..nn.compat.backbone import Backbone
from .base import ModuleConfig
from .pretrained import PretrainedConfig
from .variants import VariantConfig


class BackboneConfig(PretrainedConfig):
    class Config:
        extra = "allow"

    def build(self, **kwargs) -> Backbone:
        return ModuleConfig.from_config({**self.model_dump(), **kwargs})


class HasBackbone(VariantConfig):
    backbone: BackboneConfig = Field(default=None, validate_default=True)

    @field_validator("backbone", mode="before")
    def validate_backbone(cls, value: BackboneConfig, values):
        if value is None or "type" not in value:
            if issubclass(cls, VariantConfig):
                backbone = cls.backbone_from_variant(values.data["variant"])
                value = backbone.merge(value)
            else:
                raise ValueError(
                    "Backbone must be provided or config must inherit from VariantConfig"
                )
        value.train = values.data["train"]
        return value

    def merge(self, config: HasBackbone | dict, **kwargs):
        res = config
        config = super().merge(config, **kwargs)

        is_same_module = self.backbone.__class__ == config.backbone.__class__
        if is_same_module:
            # The config is for the same module
            is_same_variant = self.backbone.variant == config.backbone.variant
            if is_same_variant:
                # The config is for the same variant
                return config

        config.backbone = res.backbone
        return config

    @classmethod
    @abstractmethod
    def backbone_from_variant(self, name: str) -> BackboneConfig: ...

    @abstractmethod
    def backbone_forward_args(self, *args): ...
