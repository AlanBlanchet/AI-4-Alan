from __future__ import annotations

from typing import Self

from deepmerge import always_merger
from pydantic import BaseModel, Field, field_validator

from ..registry import REGISTER
from .log import Loggable


class Base(BaseModel, Loggable):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, config: dict) -> Self:
        config = config.copy()
        source = config.pop("type")

        src_cls = REGISTER[source]

        if hasattr(src_cls, "config"):
            src_config = src_cls.config(**config)
        else:
            cls.log(
                f"{src_cls.__name__} has no config. Falling back to default constructor."
            )
            return src_cls(**config)

        return src_cls(src_config)


class ModuleConfig(Base):
    type: str = Field(default=None, validate_default=True)
    train: bool = False

    @field_validator("type", mode="before")
    def validate_type(cls, value):
        if value is None:
            name = cls.__name__.split("Config")[0]
            return name
        return value

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def merge(self, config: ModuleConfig | dict, **kwargs):
        if isinstance(config, ModuleConfig):
            # We are receiving a config
            config = config.model_dump()

        merged = {}
        always_merger.merge(merged, self.model_dump(exclude_none=True))
        always_merger.merge(merged, config)
        always_merger.merge(merged, kwargs)

        return self.__class__(**merged)
