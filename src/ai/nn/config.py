from typing import ClassVar

from timm import create_model

from ..configs.base import Base
from ..registry import REGISTER


@REGISTER
class CustomModel(Base):
    _: ClassVar[str] = "custom"

    @classmethod
    def build(cls, name: str, **kwargs):
        return dict(model_cls=REGISTER[name], **kwargs, **REGISTER.get_info(name))


@REGISTER
class TimmModel(Base):
    _: ClassVar[str] = "timm"

    @classmethod
    def build(cls, name: str, **kwargs):
        def build_(**other_params: dict):
            return create_model(name, **kwargs, **other_params)

        return dict(model_cls=build_, **kwargs, adapt=False)
