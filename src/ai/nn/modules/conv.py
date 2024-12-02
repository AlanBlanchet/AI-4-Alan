import torch.nn as nn

from ..compat.merge import SumSequential


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        stride=1,
        bias=False,
        dilation=1,
        activation=nn.ReLU,
        norm=nn.BatchNorm2d,
        norm_first=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias or norm is None,
            dilation=dilation,
        )
        post = [
            activation() if activation is not None else nn.Identity(),
            norm(out_channels) if norm is not None else nn.Identity(),
        ]
        if norm_first:
            post.reverse()
        self.post = SumSequential(*post)

    def forward(self, x):
        return self.post(self.conv(x))
