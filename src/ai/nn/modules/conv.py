from typing import Optional

import torch.nn as nn

from myconf import F

from ..compat.activations import Activation, ReLU
from ..compat.merge import SumSequential
from ..compat.module import Module
from .norm import BatchNorm2d, Norm


class ConvBlock(Module):
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int] = (3, 3)
    padding: int = 0
    stride: int = 1
    bias: bool = False
    dilation: int = 1
    activation: type[Activation] = ReLU
    norm: Optional[type[Norm]] = BatchNorm2d
    norm_first: bool = True

    _conv: nn.Module = F(
        lambda self: nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            bias=self.bias or self.norm is None,
            dilation=self.dilation,
        ),
        init=False,
    )

    _post: nn.Sequential = F(lambda self: self._create_post(), init=False)

    def _create_post(self):
        post = [
            self.activation() if self.activation is not None else nn.Identity(),
            self.norm(self.out_channels) if self.norm is not None else nn.Identity(),
        ]
        if self.norm_first:
            post.reverse()
        return SumSequential(*post)

    def __str__(self):
        return self._flat_repr()

    def forward(self, x):
        return self._post(self._conv(x))
