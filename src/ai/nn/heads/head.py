from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Self

from myconf.core import F

from ..compat.module import Module


class Head(Module):
    default_mode: ClassVar[Self]

    _current_mode: Self = F(lambda self: self.default_mode(), init=False)

    @cached_property
    def head(self) -> Module:
        return self.create_head()

    @abstractmethod
    def create_head(self) -> Module: ...

    def change_mode(self, mode: Head, forward: Callable):
        self._current_mode = mode
        self.forward = forward
        # Clear the cached property so it gets recreated with new mode
        if "head" in self.__dict__:
            del self.__dict__["head"]
