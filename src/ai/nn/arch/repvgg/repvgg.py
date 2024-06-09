import torch.nn as nn

from ...modules.conv import ConvBlock


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            bias=False,
            activation=None,
        )

        self.res = ConvBlock(in_channels, out_channels, 1, 0, stride, bias=False)

        self.bn = nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.block(x)
        x += self.res(res)
        if self.bn is not None:
            x += self.bn(res)
        return self.relu(x)


class RepVGG(nn.Module):
    def __init__(self, task="clf"):
        super().__init__()
        self.task = task

    def forward(self): ...


class RepVGG_A(RepVGG):
    def __init__(self, task="clf"):
        super().__init__(task=task)
