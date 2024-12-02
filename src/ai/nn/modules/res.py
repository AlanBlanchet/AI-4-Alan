import torch.nn as nn

from .conv import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        blocks_out_channels: list[int],
        blocks_kernels: list[int],
        shortcut=None,
        stride=1,
    ):
        super().__init__()

        layers = []

        if in_channels is None:
            in_channels = blocks_out_channels[0]

        for i, (out_c, kernel_size) in enumerate(
            zip(blocks_out_channels, blocks_kernels)
        ):
            last_block = i == len(blocks_out_channels) - 1
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding=1 if kernel_size == 3 else 0,
                    stride=stride if i == 1 else 1,
                    activation=None if last_block else nn.ReLU,
                )
            )
            in_channels = out_c

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
