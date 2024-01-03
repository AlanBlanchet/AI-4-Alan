import torch
from attr import define, field

from ..neuron.neuron import LNeuron, Neuron


@define(slots=False)
class LLayer:
    in_features: int = field()
    out_features: int = field()

    def __attrs_post_init__(self):
        self.neurons = [LNeuron(self.in_features) for _ in range(self.out_features)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


@define(slots=False)
class Layer:
    in_features: int = field()
    out_features: int = field()

    def __attrs_post_init__(self):
        self.neurons = [Neuron(self.in_features) for x in range(self.out_features)]

    def __call__(self, x):
        return torch.stack([n(x) for n in self.neurons])
