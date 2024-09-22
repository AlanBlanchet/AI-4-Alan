# TODO remove
import torch
from attr import define, field

from ..layer.layer import Layer, LLayer


@define(slots=False)
class LMLP:
    in_features: int = field()
    layers: list[int] = field()

    def __attrs_post_init__(self):
        layers = [self.in_features] + self.layers
        self.model = [LLayer(layers[i], layers[i + 1]) for i in range(len(self.layers))]

    def __call__(self, x):
        for layer in self.model:
            x = layer(x)
        return x


@define(slots=False)
class MLP:
    in_features: int = field()
    layers: list[int] = field()

    def __attrs_post_init__(self):
        layers = [self.in_features] + self.layers
        self.model = [Layer(layers[i], layers[i + 1]) for i in range(len(self.layers))]

    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        for layer in self.model:
            x = layer(x)
        return x
