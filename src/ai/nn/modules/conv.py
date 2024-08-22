import torch.nn as nn


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
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )
        self.act = activation() if activation is not None else nn.Identity()
        self.norm = norm(out_channels) if norm is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))
