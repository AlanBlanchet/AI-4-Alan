import torch.nn as nn

from ....registry.registers import MODEL
from ...modules import ResidualBlock
from .configs import configs


@MODEL.register
class ResNet(nn.Module):
    def __init__(self, in_channels=3, config: list = configs["18"]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(config[0])
        self.conv3_x = self._make_layer(config[1], config[0][0][-1][0])
        self.conv4_x = self._make_layer(config[2], config[1][0][-1][0])
        self.conv5_x = self._make_layer(config[3], config[2][0][-1][0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self, config: tuple[list[tuple[int, int]], int], in_channels: int = 64
    ):
        layers = []

        params, n = config

        out_channels = params[-1][0]
        for i in range(n):
            identity = None
            stride = 2 if i == n - 1 else 1

            if stride != 1 or in_channels != out_channels:
                identity = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )

            layers.append(
                ResidualBlock(
                    params,
                    in_channels,
                    identity,
                    stride=stride,
                )
            )
            in_channels = out_channels

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for conv_x in [self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]:
            for layer in conv_x:
                x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


@MODEL.register
class ResNet18(ResNet):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, config=configs["18"])


@MODEL.register
class ResNet34(ResNet):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, config=configs["34"])


@MODEL.register
class ResNet50(ResNet):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, config=configs["50"])


@MODEL.register
class ResNet101(ResNet):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, config=configs["101"])


@MODEL.register
class ResNet152(ResNet):
    def __init__(self, in_channels=3):
        super().__init__(in_channels, config=configs["152"])
