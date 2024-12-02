from pydantic import field_validator

from .base import ModuleConfig
from .pretrained import PretrainedConfig


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
