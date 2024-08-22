from typing import Any, ClassVar, Self

from pydantic import BaseModel

from ..registry.registers import SOURCE


class BaseConfig(BaseModel):
    log_name: ClassVar[str] = "main"

    @classmethod
    def from_config(cls, config: dict) -> Self:
        config = config.copy()
        source = config.pop("_")

        # Build subconfigs
        for key, value in config.items():
            if isinstance(value, dict):
                if "_" in value:
                    # Value is another config
                    config[key] = cls.from_config(value)

        return SOURCE[source].build(**config)

    @classmethod
    def build(cls, **kwargs) -> Self:
        return cls(**kwargs)

    @classmethod
    def _log_prefix(self) -> str:
        return f"[{self.log_name.capitalize()}] " if self.log_name else ""

    def log(self, *msg: list[Any]):
        print(self._log_prefix() + " ".join([str(m) for m in msg]))

    class Config:
        arbitrary_types_allowed = True
