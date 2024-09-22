import torch
import torch.nn as nn

from ....registry import REGISTER
from ....task.detection.detection import decode
from ....utils.arch import build_arch_module
from ...modules.conv import ConvBlock
from ..compat import ISSD
from ..vgg.models import VGG16


@REGISTER(requires=["backbone", "num_classes"])
class SSD(nn.Module):
    def __init__(self, backbone: ISSD | str = None, num_classes=20):
        super().__init__()
        backbone = backbone if backbone else VGG16()
        if isinstance(backbone, str):
            backbone = build_arch_module(backbone)

        self.num_classes = num_classes

        compat = getattr(backbone, "ssd_compat", None)
        if not compat or not callable(compat):
            raise AttributeError(
                f"Backbone {backbone.__class__.__name__} is not SSD compatible"
            )

        # Backbone features
        ssd_compat = backbone.ssd_compat()
        self.features = ssd_compat["features"]
        extras = ssd_compat.get("extras", [])
        # Convert ModuleList to Sequential
        for i, extra in enumerate(extras):
            if isinstance(extra, nn.ModuleList):
                extras[i] = nn.Sequential(*extra)

        default_channels = [512, 1024, 512, 256, 256, 256]
        channels = ssd_compat.get("channels", default_channels)

        # Add default channels if not provided
        if len(channels) != len(default_channels):
            channels.extend(default_channels[len(channels) :])

        self.num_defaults = [4, 6, 6, 6, 4, 4]

        classifiers = []
        for i, (default, oc) in enumerate(zip(self.num_defaults, channels)):
            classifiers.append(nn.Conv2d(oc, default * (num_classes + 4), 1))
        self.classifiers = nn.ModuleList(classifiers)

        inter_channels = [1024, 256, 128, 128, 128]  # Conv 1x1 channels
        for i, (in_channel, inter_channel, out_channel) in list(
            enumerate(zip(channels[:-1], inter_channels, channels[1:]))
        )[len(extras) :]:
            p = 0 if i > 2 else 1 if i != 0 else 6
            s = 2 if i < 3 else 1
            # Dilation is 6 for the first layer with p=6
            d = 6 if i == 0 else 1
            norm = None if i == len(channels) - 2 else nn.BatchNorm2d
            extras.append(
                nn.Sequential(
                    ConvBlock(in_channel, inter_channel, 1, padding=0),
                    ConvBlock(
                        inter_channel,
                        out_channel,
                        3,
                        padding=p,
                        stride=s,
                        norm=norm,
                        dilation=d,
                    ),
                )
            )
        self.extras = nn.ModuleList(extras)

        self.feature_scaling = nn.Parameter(torch.ones(512) * 20)

    def forward(self, x):
        x = self.features(x)

        # Rescaling the features
        # x = self.feature_scaling.view(1, -1, 1, 1) * F.normalize(x)

        boxes = [self.classifiers[0](x)]
        for clf, extra in zip(self.classifiers[1:], self.extras):
            x = extra(x)
            boxes.append(clf(x))

        conf, loc = decode(boxes, self.num_defaults, self.num_classes)

        return dict(scores=conf, boxes=loc)


@REGISTER
class SSD300(SSD):
    def __init__(self, backbone: ISSD | str = None, num_classes=20):
        super().__init__(backbone, num_classes)


@REGISTER
class SSD512(SSD):
    def __init__(self, backbone: ISSD | str = None, num_classes=20):
        super().__init__(backbone, num_classes)
