from __future__ import annotations

import re
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, Self

from deepmerge import always_merger
from pydantic import Field, PrivateAttr, field_validator

from ..registry import REGISTER
from .log import Loggable

if TYPE_CHECKING:
    from ..launch import Main


class ActionEnum(IntEnum):
    fit = 0
    val = 1


class Base(Loggable):
    _root_config: Main | None = PrivateAttr(None)

    identification_name: ClassVar[str] = "identifier"

    def model_post_init(self, __context):
        self.info(f"Initialized {self.__class__.__name__}")

    @property
    def root_config(self) -> Main:
        return self._root_config

    def __init_subclass__(cls, buildable=True, **kwargs):
        super().__init_subclass__(**kwargs)
        # Manually check if the Model is buildable
        cls.is_buildable = buildable

    @classmethod
    def all_subclasses(cls):
        return cls.__subclasses__() + [
            g for s in cls.__subclasses__() for g in s.all_subclasses()
        ]

    @classmethod
    def from_config(cls, config: dict | str | Any) -> Self:
        """
        Function to be called whenever an object wants to use the identifier in the config to build its own object
        """
        if isinstance(config, cls):
            return config

        try:
            if isinstance(config, str):
                id = config
                config = {}
            else:
                id = config.pop(cls.identification_name)
        except KeyError:
            raise ValueError(
                f"Could not find key '{cls.identification_name}' in the config. "
                f"Please provide a '{cls.identification_name}' to build the object.\n"
                f"Current config:\n{config}"
            )

        sub = cls.get_cls(id)
        return sub(**config)

    @classmethod
    def get_cls(cls, id: str) -> type[Self]:
        for sub in cls.all_subclasses():
            if id in sub.get_identifiers():
                if sub.is_buildable:
                    return sub
                else:
                    raise ValueError(
                        f"You cannot directly build an object of type {sub.__name__}. "
                        f"Please use one of its subclasses."
                    )

        available_options = ", ".join(cls.get_subclass_identifiers())
        raise ValueError(
            f"Could not find a subclass with {cls.identification_name} '{id}'. "
            f"Available options are: [{available_options}]"
        )

    @classmethod
    def spaced_name(cls):
        return re.sub(r"(?<![A-Z])([A-Z])", r" \1", cls.__name__).strip()

    @classmethod
    def get_identifiers(cls):
        name = cls.__name__
        spaced_name = cls.spaced_name()
        return {name, name.lower(), spaced_name, spaced_name.lower()}

    @classmethod
    def get_subclass_identifiers(cls):
        return set().union(
            *[sub.get_identifiers() for sub in cls.__subclasses__() if sub.is_buildable]
        )

    @classmethod
    def from_config_model(cls, config: dict) -> Self:
        config = config.copy()
        source = config.pop("type")

        src_cls = REGISTER[source]

        if hasattr(src_cls, "config"):
            src_config = src_cls.config(**config)
        else:
            cls.info(
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
