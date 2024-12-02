import torch.nn as nn

from ....registry import REGISTER
from ...compat.backbone import Backbone
from ...compat.merge import SumSequential
from ...modules import ResidualBlock
from ...modules.conv import ConvBlock
from .configs import ResNetConfig, ResNetLayerConfig


@REGISTER
class ResNet(Backbone):
    config: ResNetConfig = ResNetConfig
    default_task = "classification"

    def __init__(self, config: ResNetConfig):
        super().__init__(config)

        self.conv1 = ConvBlock(
            config.in_channels, 64, kernel_size=7, stride=2, padding=3
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        in_channels = config.first_layer_channels
        # Build all the ResNetLayers
        for i, layer_conf in enumerate(config.layers):
            layer = self.make_layer(in_channels, layer_conf, first_layer=i == 0)
            in_channels = layer_conf.out_channels
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # Create a head for the model
        self.head = SumSequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.out_channels, config.num_classes),
        )

    @staticmethod
    def make_layer(in_channels: int, layer_conf: ResNetLayerConfig, first_layer=False):
        """
        We are building a ResNetLayer
        """
        layers = []

        # Initialisation
        n, out_channels = layer_conf.num, layer_conf.out_channels

        # Create the blocks inside the ResNetLayer
        for i in range(n):
            shortcut = None
            # Stride 2 on first block after first layer
            stride = 2 if i == 0 and not first_layer else 1

            # If first block in layer and in_channels != out_channels, downsample
            if i == 0 and in_channels != out_channels:
                shortcut = ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    activation=None,  # No activation on shortcut
                )

            res_block = ResidualBlock(
                in_channels,
                blocks_out_channels=layer_conf.blocks_out_channels,
                blocks_kernels=layer_conf.blocks_kernel_size,
                shortcut=shortcut,
                stride=stride,
            )

            layers.append(res_block)
            # Next block in layer will have the same in_channels as the previous block
            in_channels = out_channels

        return nn.ModuleList(layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        feats = []
        for layer in self.layers:
            for block in layer:
                x = block(x)

            # Store the feature map of the layer
            feats.append(x)

        return feats

    def forward(self, x):
        feats = self.features(x)
        return self.head(feats[-1])


# @REGISTER
# class ResNet18(ResNet):
#     variant = "18"


# @REGISTER
# class ResNet34(ResNet):
#     variant = "34"


# @REGISTER
# class ResNet50(ResNet):
#     variant = "50"


# @REGISTER
# class ResNet101(ResNet):
#     variant = "101"


# @REGISTER
# class ResNet152(ResNet):
#     variant = "152"
