from typing import Callable, Optional

import torch.nn as nn
from pydantic import Field, field_validator

from ..compat.activations import Activation
from ..compat.merge import SumSequential
from ..compat.module import Module


class ConvBlock(Module):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int] = Field(3, validate_default=True)
    padding: int = 0
    stride: int = 1
    bias: bool = False
    dilation: int = 1
    activation: Optional[Callable[[], Activation]] = Activation.get_cls("relu")
    norm: Optional[type[nn.Module]] = nn.BatchNorm2d
    norm_first: bool = True

    @field_validator("in_channels", mode="before")
    def validate_in_channels(cls, value):
        return value

    @field_validator("activation", mode="before")
    def validate_activation(cls, value):
        if isinstance(value, str):
            return Activation.get_cls(value)

    @field_validator("kernel_size", mode="before")
    def validate_kernel_size(cls, value):
        if isinstance(value, int):
            return value, value
        return value

    def init(self):
        super().init()

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            bias=self.bias or self.norm is None,
            dilation=self.dilation,
        )
        post = [
            self.activation() if self.activation is not None else nn.Identity(),
            self.norm(self.out_channels) if self.norm is not None else nn.Identity(),
        ]
        if self.norm_first:
            post.reverse()
        self.post = SumSequential(*post)

    def forward(self, x):
        return self.post(self.conv(x))


class ConvNet(Module):
    blocks: nn.ModuleList = Field(None, validate_default=True)
    head: nn.Module

    @field_validator("blocks", mode="before")
    def validate_blocks(cls, value):
        return nn.ModuleList([Module.from_config(block) for block in value])

    @field_validator("head", mode="before")
    def validate_head(cls, value):
        if isinstance(value, dict):
            return Module.from_config(value)
        return value

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
