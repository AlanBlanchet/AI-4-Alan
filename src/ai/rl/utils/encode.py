from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

IMAGE_TYPE = Literal["rgb", "grayscale"]


class Encoder(nn.Module):
    def __init__(self, in_shape: int | np.ndarray, history=1):
        super().__init__()
        self.in_shape = in_shape
        self.history = history

        layers = []
        in_shape = list(in_shape)

        if len(in_shape) == 3:
            layers.extend(
                [
                    nn.AdaptiveAvgPool2d((84, 84)),
                    nn.Conv2d(in_shape[0] * history, 32, 8, 4),
                    nn.ReLU(True),
                    nn.Conv2d(32, 64, 4, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Flatten(),
                ]
            )
            self.out = 3136
        else:
            layers.extend(
                [
                    nn.Linear(in_shape[0], 128),
                    nn.ReLU(True),
                    nn.Linear(128, 128),
                    nn.ReLU(True),
                ]
            )
            self.out = 128

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
