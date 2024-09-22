import torch.nn as nn

from .conv import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(
        self, params: list[tuple[int, int]], in_channels, shortcut=None, stride=1
    ):
        super().__init__()

        layers = []

        if in_channels is None:
            in_channels = params[0][0]

        for i, param in enumerate(params):
            out_channels, kernel = param
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel,
                    padding=1 if kernel == 3 else 0,
                    stride=1 if i > 0 else stride,
                    activation=nn.ReLU if i < len(params) - 1 else None,
                )
            )
            in_channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        for layer in self.layers:
            x = layer(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        return self.act(x + residual)
