import os
from typing import Any, ClassVar, Self

from pydantic import BaseModel

from ..registry import REGISTER


class Loggable:
    log_name: ClassVar[str] = "main"

    @classmethod
    def _log_prefix(self) -> str:
        return f"[{self.log_name.capitalize()}] " if self.log_name else ""

    @classmethod
    def log(cls, *msg: list[Any]):
        if os.getenv("MAIN_PROCESS", "1") == "1":
            print(cls._log_prefix() + " ".join([str(m) for m in msg]))


class AdvancedBase(BaseModel):
    def merge(self, **kwargs):
        print({**self.model_dump(), **kwargs})
        return self.__class__(**{**self.model_dump(), **kwargs})


class Base(BaseModel, Loggable):
    @classmethod
    def from_config(cls, config: dict, deep=False) -> Self:
        config = config.copy()
        source = config.pop("type")

        if deep:
            # Build subconfigs
            for key, value in config.items():
                if isinstance(value, dict):
                    if "type" in value:
                        # Value is another config
                        config[key] = cls.from_config(value)

        src_cls = REGISTER[source]

        try:
            base_config = src_cls.config(**config)
        except AttributeError:
            cls.log(
                f"{src_cls.__name__} has no config. Falling back to default constructor."
            )
            return src_cls(**config)

        return src_cls(base_config)

    @classmethod
    def build(cls, **kwargs) -> Self:
        return cls(**kwargs)

    class Config:
        arbitrary_types_allowed = True
