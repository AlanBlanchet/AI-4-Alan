from typing import Any, Callable, Generic, Literal, Self, TypeVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

FORMATS_TYPE = Literal["project", "pytorch"]
FORMATS: list[FORMATS_TYPE] = list(get_args(FORMATS_TYPE))


MODELS_TYPE = Literal["transformer"]
MODELS: list[MODELS_TYPE] = list(get_args(MODELS_TYPE))

STACK_TYPE = Literal["frame", "mask"]

T = TypeVar("T")


class CallableList(list[Callable[..., T]], Generic[T]):
    def __call__(self, *args, **kwargs):
        return [item(*args, **kwargs) for item in self]

    def __get_pydantic_core_schema__(
        self, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        schema = handler.generate_schema(list[Callable[..., T]])
        return core_schema.no_info_after_validator_function(
            self._validate_callable_list, schema
        )

    @classmethod
    def _validate_callable_list(cls, value: Any) -> Self:
        if not isinstance(value, list):
            raise TypeError("Value must be a list")
        for item in value:
            if not callable(item):
                raise TypeError("All items in the list must be callable")
        return cls(value)
