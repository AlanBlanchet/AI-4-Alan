from typing import ClassVar

from .module import Module


class VariantMixin(Module):
    # Default variant for the model
    variants: ClassVar[list[str]]
    variant: ClassVar[str]
