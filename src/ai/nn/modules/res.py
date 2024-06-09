import torch.nn as nn

from .conv import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(
        self, params: list[tuple[int, int]], in_channels, identity=None, stride=1
    ):
        super().__init__()

        self.layers = nn.Sequential()

        self.identity = identity

        if in_channels is None:
            in_channels = params[0][0]

        for i, param in enumerate(params):
            out_channels, kernel = param
            self.layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel,
                    padding=1 if kernel == 3 else 0,
                    stride=1 if i < len(params) - 1 else stride,
                )
            )
            in_channels = out_channels

    def forward(self, x):
        residual = x
        if self.identity is not None:
            residual = self.identity(residual)
        return self.layers(x) + residual
