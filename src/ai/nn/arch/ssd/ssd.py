import torch.nn as nn

from ....registry.registers import MODEL
from ....task.detection import decode
from ...modules.conv import ConvBlock
from ..compat import ISSD
from ..vgg.models import VGG16


@MODEL.register
class SSD(nn.Module):
    def __init__(self, backbone: ISSD = None, num_classes=20):
        super().__init__()
        backbone = backbone if backbone else VGG16()
        self.num_classes = num_classes

        compat = getattr(backbone, "ssd_compat", None)
        if not compat or not callable(compat):
            raise AttributeError(
                f"Backbone {backbone.__class__.__name__} is not SSD compatible"
            )

        # Backbone features
        ssd_compat = backbone.ssd_compat()
        self.features = ssd_compat["features"]
        channels = ssd_compat.get("channels", [512, 1024, 512, 256, 256, 256])

        expansion_channels = [512, 1024, 512, 256, 256, 256]

        self.num_defaults = [4, 6, 6, 6, 4, 4]

        classifiers = []
        for i, (default, oc) in enumerate(zip(self.num_defaults, channels)):
            classifiers.append(
                ConvBlock(
                    oc,
                    default * (num_classes + 4),
                    1,
                    norm=None if i == len(self.num_defaults) - 1 else nn.BatchNorm2d,
                )
            )
        self.classifiers = nn.ModuleList(classifiers)

        extras = []
        for i, (in_channel, inter_channel, out_channel) in enumerate(
            zip(channels[:-1], expansion_channels[1:], channels[1:])
        ):
            extras.append(
                nn.Sequential(
                    ConvBlock(in_channel, inter_channel, 1, padding=0),
                    ConvBlock(
                        inter_channel,
                        out_channel,
                        3,
                        padding=1 if i < 3 else 0,
                        stride=2 if i < 3 else 1,
                        norm=None if i == len(channels) - 2 else nn.BatchNorm2d,
                    ),
                )
            )
        self.extras = nn.ModuleList(extras)

    def forward(self, x):
        x = self.features(x)

        boxes = [self.classifiers[0](x)]
        for clf, extra in zip(self.classifiers[1:], self.extras):
            x = extra(x)
            boxes.append(clf(x))

        conf, loc = decode(boxes, self.num_defaults, self.num_classes)

        return dict(confidence=conf, location=loc)
