from pydantic import Field, field_validator

from ....configs.backbone import BackboneConfig
from ....configs.pretrained import PretrainedConfig
from ....configs.task import ClassificationConfig

configs = {
    "11": [
        ["C", 3, 64],
        "M",
        ["C", 64, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "11-LRN": [
        ["C", 3, 64],
        "L",
        "M",
        ["C", 64, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "13": [
        ["C", 3, 64],
        ["C", 64, 64],
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "16": [
        ["C", 3, 64],
        ["C", 64, 64],
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        # Ususally its ["C", 512, 512, 1] but timm doesn't use conv 1x1
        "M",
    ],
    "D": [
        ["C", 3, 64],
        ["C", 64, 64],
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "E": [
        ["C", 3, 64],
        ["C", 64, 64],
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
}


class VGGConfig(BackboneConfig, ClassificationConfig):
    variants = ["16"]

    pretrained_recommendations = PretrainedConfig.create_recommendations(
        "IMAGENET", variants=variants
    )

    in_channels: int = 3
    config: list = Field(default=None, validate_default=True)

    @property
    def out_channels(self):
        return self.layers[-1].out_channels

    @field_validator("config", mode="before")
    def validate_layers(cls, value, values):
        if value is None:
            variant = values.data["variant"]
            return configs[variant]
        return value
