from typing import ClassVar

import torch.nn as nn

from myconf import F
from myconf.core import Cast

from ....data.batch import Batch
from ....data.task.classification import ImageNet1k
from ....modality.image.modality import ChannelData
from ...compat.module import Module, ModuleList
from ...compat.pretrained import HFHubPretrainedWeights, Pretrained
from ...compat.variants import VariantMixin
from ...heads.classifier import Classifier
from ...modules.conv import ConvBlock
from ...modules.res import ResidualBlock
from .configs import Layer


class ResNet(Classifier[ChannelData], Pretrained, VariantMixin):
    default_data = ImageNet1k()
    default_mode: ClassVar[Classifier] = Classifier

    variants: ClassVar[list[str]] = ["18", "34", "50", "101", "152"]

    sources = []

    layer_config: ClassVar[list[Layer]]

    input_proj: ConvBlock = F(
        lambda _: ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
    )
    first_layer_channels: int = 64
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    layers: ModuleList[Module] = F(lambda self: self.make_layers())
    out_dim: int = F(lambda self: self.layer_config[-1].out_channels)

    def make_layers(self):
        layers = ModuleList()
        in_channels = self.first_layer_channels
        for i, layer_conf in enumerate(self.layer_config):
            layer = self.make_layer(in_channels, layer_conf, first_layer=i == 0)
            in_channels = layer_conf.out_channels
            layers.append(layer)
        return layers

    @staticmethod
    def make_layer(in_channels: int, layer_conf: Layer, first_layer=False):
        layers = ModuleList()
        n, out_channels = layer_conf.num, layer_conf.out_channels

        for i in range(n):
            shortcut = None
            stride = 2 if i == 0 and not first_layer else 1

            if i == 0 and in_channels != out_channels:
                shortcut = ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    activation=None,
                )

            res_block = ResidualBlock(
                in_channels,
                out_channels=layer_conf.blocks_out_channels,
                kernels=layer_conf.blocks_kernel_size,
                shortcut=shortcut,
                stride=stride,
            )

            layers.append(res_block)
            # Next block in layer will have the same in_channels as the previous block
            in_channels = out_channels

        return layers

    def features(self, x: Cast[ChannelData, Batch[ChannelData]]):
        x = self.input_proj(x.data)
        x = self.maxpool(x)

        feats = []
        for layer in self.layers:
            for block in layer:
                x = block(x)

            feats.append(x)

        return feats


class ResNet18(ResNet):
    variant: ClassVar[str] = "18"
    layer_config: ClassVar[list[Layer]] = [
        [[[64, 3], [64, 3]], 2],
        [[[128, 3], [128, 3]], 2],
        [[[256, 3], [256, 3]], 2],
        [[[512, 3], [512, 3]], 2],
    ]

    available_weights = [HFHubPretrainedWeights(ImageNet1k, "microsoft/resnet-18")]


class ResNet34(ResNet):
    variant: ClassVar[str] = "34"
    layer_config: ClassVar[list[Layer]] = [
        [[[64, 3], [64, 3]], 3],
        [[[128, 3], [128, 3]], 4],
        [[[256, 3], [256, 3]], 6],
        [[[512, 3], [512, 3]], 3],
    ]

    available_weights = [HFHubPretrainedWeights(ImageNet1k, "timm/resnet34.a1_in1k")]


class ResNet50(ResNet):
    variant: ClassVar[str] = "50"
    layer_config: ClassVar[list[Layer]] = [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 4],
        [[[256, 1], [256, 3], [1024, 1]], 6],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ]

    available_weights = [HFHubPretrainedWeights(ImageNet1k, "microsoft/resnet-50")]


class ResNet101(ResNet):
    variant: ClassVar[str] = "101"
    layer_config: ClassVar[list[Layer]] = [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 4],
        [[[256, 1], [256, 3], [1024, 1]], 23],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ]

    available_weights = [HFHubPretrainedWeights(ImageNet1k, "microsoft/resnet-101")]


class ResNet152(ResNet):
    variant: ClassVar[str] = "152"
    layer_config: ClassVar[list[Layer]] = [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 8],
        [[[256, 1], [256, 3], [1024, 1]], 36],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ]

    available_weights = [HFHubPretrainedWeights(ImageNet1k, "microsoft/resnet-152")]
