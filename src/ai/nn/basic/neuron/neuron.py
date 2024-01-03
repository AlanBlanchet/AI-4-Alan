import random

import torch
from attr import Factory, define, field

from ....grad import Value
from ....utils.types import FORMATS


class WeightException(Exception):
    pass


def _raise_set_exception(self, *_):
    raise WeightException("You can't set the weight/bias attributes of a neuron")


def _field(use_in_features: bool = False, format=FORMATS[0]):
    def factory_in_features(self):
        in_features = self.in_features if use_in_features else 1
        if format == "pytorch":
            out = torch.randn(in_features)
        else:
            out = [Value(random.gauss()) for _ in range(in_features)]
        return out if use_in_features else out[0]

    return field(
        default=Factory(factory_in_features, takes_self=True),
        init=False,
        on_setattr=_raise_set_exception,
        repr=False,
        type=list[Value] if use_in_features else Value,
    )


@define(slots=False)
class LNeuron:
    in_features: int = field(converter=int)
    w = _field(True)
    b = _field()

    def __call__(self, x):
        assert len(x) == len(
            self.w
        ), "The input and the weights must have the same length"

        return sum([xx * ww for xx, ww in zip(x, self.w)], self.b)


@define(slots=False)
class Neuron:
    in_features: int = field(converter=int)
    w = _field(True, format="pytorch")
    b = _field(format="pytorch")

    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x @ self.w).sum() + self.b
