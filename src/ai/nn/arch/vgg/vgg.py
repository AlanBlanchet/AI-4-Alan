from collections import OrderedDict

import torch.nn as nn

from ....registry import REGISTER
from ...modules import ConvBlock
from .configs import configs


@REGISTER
class VGG(nn.Module):
    def __init__(self, config=configs["C"], in_channels=3):
        super().__init__()

        self.features = self._generate(config, in_channels)

    def _generate(self, arch: list, in_channels: int):
        layers = []
        arch[0][1] = in_channels

        conv = 0
        sub_conv = 0
        mp = 0

        for item in arch:
            if isinstance(item, str):
                if item == "M":
                    layers.append((f"max_pool_{mp+1}", nn.MaxPool2d(2, ceil_mode=True)))
                    conv += 1
                    sub_conv = 0
                    mp += 1
                elif item == "L":
                    layers.append(("local_response", nn.LocalResponseNorm(2)))
            elif isinstance(item, list):
                t, *ps = item
                if t == "C":
                    # Padding = 0 for K=1
                    padding = 0 if len(ps) == 3 and ps[-1] == 1 else 1
                    layers.append(
                        (f"conv{conv+1}_{sub_conv+1}", ConvBlock(*ps, padding=padding))
                    )
                    sub_conv += 1

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.features(x)
