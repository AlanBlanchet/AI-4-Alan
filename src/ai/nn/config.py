from typing import ClassVar

from timm import create_model

from ..configs.base import BaseConfig
from ..registry.registers import MODEL, SOURCE


@SOURCE.register
class CustomModel(BaseConfig):
    _: ClassVar[str] = "custom"

    @classmethod
    def build(cls, name: str, **kwargs):
        return dict(model_cls=MODEL[name], **kwargs, **MODEL.get_info(name))


@SOURCE.register
class TimmModel(BaseConfig):
    _: ClassVar[str] = "timm"

    @classmethod
    def build(cls, name: str, **kwargs):
        def build_(**other_params: dict):
            return create_model(name, **kwargs, **other_params)

        return dict(model_cls=build_, **kwargs, adapt=False)
