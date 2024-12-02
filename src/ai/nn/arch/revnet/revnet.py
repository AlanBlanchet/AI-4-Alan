import torch.nn as nn

from ...compat.merge import SumSequential
from ...compat.pretrained import Pretrained
from ...modules import ResidualBlock
from ...modules.conv import ConvBlock
from ..resnet.configs import ResNetConfig


class RevNetConfig(ResNetConfig): ...


# @REGISTER
class RevNet(Pretrained):
    config: RevNetConfig = RevNetConfig
    default_task = "classification"

    def __init__(self, config: RevNetConfig):
        super().__init__()

        self.config = config
        res_config = config.config

        self.conv1 = ConvBlock(
            config.in_channels, 64, kernel_size=7, stride=2, padding=3
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(res_config[0])
        self.layer2 = self._make_layer(res_config[1], res_config[0][0][-1][0])
        self.layer3 = self._make_layer(res_config[2], res_config[1][0][-1][0])
        self.layer4 = self._make_layer(res_config[3], res_config[2][0][-1][0])

        self.head = SumSequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(res_config[-1][0][-1][0], config.num_classes),
        )

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
                    in_channels,
                    blocks_out_channels=[p[0] for p in params],
                    blocks_kernels=[p[1] for p in params],
                    shortcut=shortcut,
                    stride=stride,
                )
            )
            in_channels = out_channels

        return nn.ModuleList(layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        feats = []
        for conv_x in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in conv_x:
                x = layer(x)

            feats.append(x)

        return feats

    def forward(self, x):
        feats = self.features(x)
        return self.head(feats[-1])

    def init_weights(self):
        if self.config.pretrained:
            self._load_pretrained()
        else:
            ...
