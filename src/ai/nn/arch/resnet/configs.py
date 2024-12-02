from pydantic import BaseModel, Field, field_validator

from ....configs import (
    BackboneConfig,
    ClassificationConfig,
    PretrainedConfig,
)

_configs = {
    "18": [
        [[[64, 3], [64, 3]], 2],
        [[[128, 3], [128, 3]], 2],
        [[[256, 3], [256, 3]], 2],
        [[[512, 3], [512, 3]], 2],
    ],
    "34": [
        [[[64, 3], [64, 3]], 3],
        [[[128, 3], [128, 3]], 4],
        [[[256, 3], [256, 3]], 6],
        [[[512, 3], [512, 3]], 3],
    ],
    "50": [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 4],
        [[[256, 1], [256, 3], [1024, 1]], 6],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ],
    "101": [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 4],
        [[[256, 1], [256, 3], [1024, 1]], 23],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ],
    "152": [
        [[[64, 1], [64, 3], [256, 1]], 3],
        [[[128, 1], [128, 3], [512, 1]], 8],
        [[[256, 1], [256, 3], [1024, 1]], 36],
        [[[512, 1], [512, 3], [2048, 1]], 3],
    ],
}


class ResNetBlockConfig(BaseModel):
    out_channels: int
    kernel_size: int

    def from_list(block: list[int]):
        return ResNetBlockConfig(out_channels=block[0], kernel_size=block[1])


class ResNetLayerConfig(BaseModel):
    layer: list[ResNetBlockConfig]
    num: int

    @property
    def out_channels(self):
        return self.layer[-1].out_channels

    @property
    def blocks_out_channels(self):
        return [block.out_channels for block in self.layer]

    @property
    def blocks_kernel_size(self):
        return [block.kernel_size for block in self.layer]

    @staticmethod
    def from_lists(layer: list[list[list[int]]], num: int):
        return ResNetLayerConfig(
            layer=[ResNetBlockConfig.from_list(block=b) for b in layer], num=num
        )


class ResNetConfig(BackboneConfig, ClassificationConfig):
    variants = ["18", "34", "50", "101", "152"]

    pretrained_recommendations = PretrainedConfig.create_recommendations(
        "IMAGENET", variants=variants
    )

    in_channels: int = 3
    layers: list[ResNetLayerConfig] = Field(default=None, validate_default=True)
    first_layer_channels: int = 64

    @property
    def out_channels(self):
        return self.layers[-1].out_channels

    @field_validator("layers", mode="before")
    def validate_layers(cls, value, values):
        if value is None:
            variant = values.data["variant"]
            return cls.get_layers(name=variant)
        return value

    @staticmethod
    def get_layers(name: str):
        if name is None:
            return []
        return [
            ResNetLayerConfig.from_lists(layer=config, num=num)
            for config, num in _configs[name]
        ]

    @classmethod
    def from_variant(cls, name: str):
        return cls(layers=cls.get_layers(name=name), variant=name)
