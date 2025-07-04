"""Core types and classes for MyConf"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, Union

I = TypeVar("I")  # Input type
O = TypeVar("O")  # Output type


class Cast(Generic[I, O]):
    """Type annotation for input/output type conversion"""

    def __init__(self, input_type: type, output_type: type, converter: Callable = None):
        self.input_type = input_type
        self.output_type = output_type
        self.converter = converter


class Consumed(Generic[I]):
    """Type annotation for parameters consumed during construction (not stored)"""

    def __init__(self, input_type: type, default: Any = None):
        self.input_type = input_type
        self.default = default


@dataclass
class PropertyInfo:
    fn: Optional[Callable] = None
    value: Any = None
    annotation: Optional[type] = None
    init: bool = True
    # New fields for Cast/Consumed
    is_cast: bool = False
    is_consumed: bool = False
    input_type: Optional[type] = None
    output_type: Optional[type] = None
    converter: Optional[Callable] = None


def has_fn(info):
    """Check if a PropertyInfo has a function"""
    return hasattr(info, "fn") and info.fn is not None


def has_value(info: PropertyInfo) -> bool:
    return info.value is not None


def F(
    fn_or_value: Union[Callable, Any] = None,
    *,
    annotation: Optional[type] = None,
    init: bool = True,
) -> PropertyInfo:
    if callable(fn_or_value):
        return PropertyInfo(
            fn=fn_or_value, value=None, annotation=annotation, init=init
        )
    else:
        return PropertyInfo(
            fn=None, value=fn_or_value, annotation=annotation, init=init
        )
