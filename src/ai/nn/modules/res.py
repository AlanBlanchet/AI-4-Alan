from typing import Optional

from myconf import F

from ..compat.activations import Activation, ReLU
from ..compat.module import Module, ModuleList
from .conv import ConvBlock


class ResidualBlock(Module):
    in_channels: int
    out_channels: list[int]
    kernels: list[tuple[int, int]]
    shortcut: Optional[Module] = None
    stride: int = 1
    act: type[Activation] = ReLU

    _conv_blocks: ModuleList = F(lambda self: self._create_conv_blocks(), init=False)

    def _create_conv_blocks(self):
        layers = ModuleList()

        in_channels = self.in_channels
        if in_channels is None:
            in_channels = self.out_channels[0]  # First element is int

        for i, (out_c, kernel_size) in enumerate(zip(self.out_channels, self.kernels)):
            last_block = i == len(self.out_channels) - 1
            layers.append(
                ConvBlock(
                    in_channels,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding=1 if kernel_size == (3, 3) else 0,
                    stride=self.stride if i == 1 else 1,
                    activation=None if last_block else self.act,
                )
            )
            in_channels = out_c
        return layers

    def forward(self, x):
        residual = x

        for conv_block in self._conv_blocks:
            x = conv_block(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        return self.act()(x + residual)
