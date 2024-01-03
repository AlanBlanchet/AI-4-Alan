from typing import Callable

from attr import Factory, field


class SetException(Exception):
    pass


def _raise_set_exception(self, *_):
    raise SetException("You can't manually set this attribute")


def private_field(func: Callable, **kwargs):
    return field(
        default=Factory(func, takes_self=True),
        init=False,
        on_setattr=_raise_set_exception,
        **kwargs,
    )
