from ....configs.backbone import BackboneConfig
from ....configs.pretrained import PretrainedSourceConfig
from ....configs.task import ClassificationConfig, DimensionConfig

_SIZE_MAP = dict(B="base", L="large", H="huge")


def pretrained_timm(variants):
    configs = []
    # Only timm for now
    for variant in variants:
        size, patch_size = variant.split("_")
        size = _SIZE_MAP[size]
        configs.append(
            PretrainedSourceConfig(
                weights="imagenet",
                variant=variant,
                source="timm",
                source_params=dict(model_name=f"vit_{size}_patch{patch_size}_224"),
            )
        )
    return configs


class ViTConfig(BackboneConfig, ClassificationConfig, DimensionConfig):
    # TODO add the rest
    variants = ["B_16", "B_8", "B_32", "L_16", "L_32", "H_14", "H_28"]

    pretrained_recommendations = pretrained_timm(variants)

    def from_variant(cls, variant: str):
        size, patch_size = variant.split("_")
        size = _SIZE_MAP[size]

        n_head = 6 if patch_size == 8 else 12
        hidden_size = 128 if n_head < 12 else 256

        return cls(
            variant=variant,
            hidden_size=hidden_size,
            num_layers=12,
            n_head=n_head,
            patch_size=int(patch_size),
        )

    num_layers: int = 12

    n_head: int = 12

    patch_size: int = 16
    num_channels: int = 3

    hidden_size: int = 256

    fixed_size: list[int] = [224, 224]
