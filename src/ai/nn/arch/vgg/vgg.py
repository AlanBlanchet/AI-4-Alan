from collections import OrderedDict

import torch.nn as nn

from ....registry import REGISTER
from ...compat.backbone import Backbone
from ...modules import ConvBlock
from .configs import VGGConfig


@REGISTER
class VGG(Backbone):
    config: VGGConfig = VGGConfig

    def __init__(self, config: VGGConfig):
        super().__init__(config)

        in_channels = config.in_channels
        self.layers = self._generate(config.config, in_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(4096, config.num_classes),
        )

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
                        (
                            f"conv{conv+1}_{sub_conv+1}",
                            ConvBlock(*ps, padding=padding, norm=None),
                        )
                    )
                    sub_conv += 1

        return nn.Sequential(OrderedDict(layers))

    def features(self, x):
        return self.layers(x)

    def forward(self, x):
        return self.head(self.features(x))
