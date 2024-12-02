from typing import ClassVar

from pydantic import Field, field_validator

from .base import ModuleConfig


class VariantConfig(ModuleConfig):
    # Default variant for the model
    variants: ClassVar[list[str]]
    variant: str = Field(default=None, validate_default=True)

    @field_validator("variant", mode="before")
    def validate_variant(cls, value):
        if value is None:
            return cls.variants[0]
        value = str(value)
        if value not in cls.variants:
            raise ValueError(f"Unknown variant {value} for {cls.__name__}")
        return value

    @classmethod
    def from_variant(cls, name: str):
        return cls(variant=name)
