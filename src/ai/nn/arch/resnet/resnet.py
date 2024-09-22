import torch.nn as nn

from ....configs.models import Backbone
from ....registry import REGISTER
from ...modules import ResidualBlock
from ...modules.conv import ConvBlock
from ..compat import Pretrained
from .configs import ResNetConfig, configs


@REGISTER
class ResNet(Pretrained, Backbone):
    config = ResNetConfig
    default_task = "classification"

    def __init__(self, config: ResNetConfig):
        super().__init__()

        self.config = config
        res_config = config.config

        self.conv1 = ConvBlock(
            config.in_channels, 64, kernel_size=7, stride=2, padding=3
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(res_config[0])
        self.conv3_x = self._make_layer(res_config[1], res_config[0][0][-1][0])
        self.conv4_x = self._make_layer(res_config[2], res_config[1][0][-1][0])
        self.conv5_x = self._make_layer(res_config[3], res_config[2][0][-1][0])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(res_config[-1][0][-1][0], config.num_classes),
        )

        self.conv1.post[0].running_mean.fill_(0)
        self.init_weights()

    @classmethod
    def build(cls, **kwargs):
        config = ResNetConfig(**kwargs)
        return cls(config)

    def _make_layer(self, config: tuple[list[tuple[int, int]], int], in_channels=None):
        layers = []

        is_first = in_channels is None

        if is_first:
            in_channels = 64

        params, n = config

        out_channels = params[-1][0]
        for i in range(n):
            shortcut = None
            # Stride 2 on first block after first layer
            stride = 2 if i == 0 and not is_first else 1

            if i == 0 and in_channels != out_channels:
                # Downsample
                shortcut = nn.Sequential(
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
                    shortcut,
                    stride=stride,
                )
            )
            in_channels = out_channels

        return nn.ModuleList(layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        feats = []
        for conv_x in [self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]:
            for layer in conv_x:
                x = layer(x)
            feats.append(x)

        return feats

    def forward(self, x):
        feats = self.features(x)
        return self.head(feats[-1])

    def init_weights(self):
        if self.config.pretrained:
            self.load_pretrained()
        else:
            ...


@REGISTER
class ResNet18(ResNet):
    def __init__(self, config: ResNetConfig = ResNetConfig(config=configs["18"])):
        super().__init__(config.merge(config=configs["18"]))


@REGISTER
class ResNet34(ResNet):
    def __init__(self, config: ResNetConfig = ResNetConfig(config=configs["34"])):
        super().__init__(config.merge(config=configs["34"]))


@REGISTER
class ResNet50(ResNet):
    def __init__(self, config: ResNetConfig = ResNetConfig(config=configs["50"])):
        super().__init__(config.merge(config=configs["50"]))


@REGISTER
class ResNet101(ResNet):
    def __init__(self, config: ResNetConfig = ResNetConfig(config=configs["101"])):
        super().__init__(config.merge(config=configs["101"]))


@REGISTER
class ResNet152(ResNet):
    def __init__(self, config: ResNetConfig = ResNetConfig(config=configs["152"])):
        super().__init__(config.merge(config=configs["152"]))
