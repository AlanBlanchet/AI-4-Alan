from ..configs.base import Base
from ..utils.types import CallableList


class Preprocess(Base):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Preprocesses(CallableList):
    def __call__(self, *args):
        for pre in self:
            input = pre(*args)
        return input
