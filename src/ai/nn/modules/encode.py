from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..compat.module import Module
from .conv import ConvBlock

IMAGE_TYPE = Literal["rgb", "grayscale"]


class Encoder(Module):
    in_shape: int | np.ndarray | tuple[int, ...]
    history: int = 1
    dims: list[int, int] = [16, 32]

    def init(self):
        super().init()

        layers = []
        in_shape = list(self.in_shape)

        if len(in_shape) == 3:
            layers.extend(
                [
                    nn.AdaptiveAvgPool2d((84, 84)),  # (B, 4, 84, 84)
                    ConvBlock(
                        in_channels=in_shape[0] * self.history,
                        out_channels=self.dims[0],
                        kernel_size=8,
                        stride=4,
                    ),  # (B, 16, 20, 20)
                    ConvBlock(
                        in_channels=self.dims[0],
                        out_channels=self.dims[1],
                        kernel_size=4,
                        stride=2,
                    ),  # (B, 16, 9, 9)
                    ConvBlock(
                        in_channels=self.dims[1],
                        out_channels=256,
                        kernel_size=9,
                        norm=None,
                    ),
                    nn.Flatten(),
                ]
            )

        else:
            layers.extend(
                [
                    nn.Linear(in_shape[0], 128),
                    nn.ReLU(True),
                    nn.Linear(128, 128),
                    nn.ReLU(True),
                ]
            )

        self.out = 256
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode x
        if x.ndim == 5:
            # Using some sort of sequencing
            S = x.shape[1]
            x = rearrange(x, "b s c h w -> (b s) c h w")
            x = self.encoder(x)  # (B*S, 3136)
            x = rearrange(x, "(b s) n -> b s n", s=S)
        else:
            x = self.encoder(x)

        # Return logits in shape of the output dimension
        return x
