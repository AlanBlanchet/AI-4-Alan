from typing import Literal, get_args

import torch.nn as nn

from ....registry import REGISTER

MOBILE_NET_V1_VALID_RES_TYPE = Literal["224", "192", "160", "128", "96", "64"]
MOBILE_NET_V1_VALID_RES: list[MOBILE_NET_V1_VALID_RES_TYPE] = list(
    get_args(MOBILE_NET_V1_VALID_RES_TYPE)
)

MOBILE_NET_V1_POOL_KERNELS = list(range(7, 1, -1))


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.depth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


@REGISTER
class MobileNet(nn.Module):
    """
    https://arxiv.org/pdf/1704.04861
    """

    def __init__(
        self,
        in_channels: int = 3,
        alpha=1,
        in_res: MOBILE_NET_V1_VALID_RES_TYPE = "224",
    ):
        super().__init__()

        d_32 = int(32 * alpha)
        d_64 = int(64 * alpha)
        d_128 = int(128 * alpha)
        d_256 = int(256 * alpha)
        d_512 = int(512 * alpha)
        d_1024 = int(1024 * alpha)

        self.conv1 = nn.Conv2d(in_channels, d_32, kernel_size=3, stride=2)
        self.dws1 = DepthWiseSeparableConv2d(d_32, d_64, stride=1)
        self.dws2 = DepthWiseSeparableConv2d(d_64, d_128, stride=2)
        self.dws3 = DepthWiseSeparableConv2d(d_128, d_128, stride=1)
        self.dws4 = DepthWiseSeparableConv2d(d_128, d_256, stride=2)
        self.dws5 = DepthWiseSeparableConv2d(d_256, d_256, stride=1)
        self.dws6 = DepthWiseSeparableConv2d(d_256, d_512, stride=2)
        self.bottleneck = nn.ModuleList(
            [DepthWiseSeparableConv2d(d_512, d_512, stride=1) for _ in range(5)]
        )
        self.dws7 = DepthWiseSeparableConv2d(d_512, d_1024, stride=2)
        self.dws8 = DepthWiseSeparableConv2d(
            d_1024, d_1024, stride=1
        )  # Error here in paper 'stride'
        res_idx = MOBILE_NET_V1_VALID_RES.index(in_res)
        kernel_value = MOBILE_NET_V1_POOL_KERNELS[res_idx]
        self.pool = nn.AvgPool2d(kernel_size=kernel_value)

    def forward(self, x):
        assert x.shape[-2:] == (224, 224), "Input shape should be 224x224"
        x = self.conv1(x)
        x = self.dws1(x)
        x = self.dws2(x)
        x = self.dws3(x)
        x = self.dws4(x)
        x = self.dws5(x)
        x = self.dws6(x)
        x = self.bottleneck(x)
        x = self.dws7(x)
        x = self.dws8(x)
        x = self.pool(x)
        return x


@REGISTER
class MobileNetV1(MobileNet): ...


@REGISTER
class MobileNetV2(nn.Module):
    # TODO
    def __init__(self):
        super().__init__()

        raise NotImplementedError()
