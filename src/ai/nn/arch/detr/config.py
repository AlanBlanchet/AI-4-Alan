import torch
from pydantic import field_validator

from ....configs import (
    ClassificationConfig,
    PretrainedConfig,
)
from ....configs.backbone import HasBackbone
from ....configs.pretrained import PretrainedSourceConfig
from ..resnet.configs import ResNetConfig


class DETRConfig(PretrainedConfig, ClassificationConfig, HasBackbone):
    variants = ["r50", "r101", "r18", "r34", "r150"]

    hidden_dim: int = 256
    num_queries: int = 100

    class_coef: float = 1
    bbox_coef: float = 5
    giou_coef: float = 2

    freeze_norm: bool = None

    pretrained_recommendations = [
        PretrainedSourceConfig(
            variant=f"r{id}",
            source="torch",
            weights="COCO",
            source_params=dict(
                repo="facebookresearch/detr",
                model_name=f"detr_resnet{id}",
                pretrained=True,
            ),
            weights_params=dict(layer="class_embed", extra_features=1),
            state_params=dict(
                ignores=["num_batches_tracked"],
                additional_mapping={"query_embed.weight": "query_emb.weight"},
            ),
            forward_args=[torch.ones(1, 3, 640, 640)],
        )
        for id in ["50", "101"]
    ]

    @field_validator("freeze_norm", mode="before")
    def validate_freeze_norm(cls, value, values):
        train = values.data["train"]
        if value is None:
            return not train
        return value

    @classmethod
    def backbone_from_variant(cls, name: str):
        variant = name.split("r")[-1]
        return ResNetConfig.from_variant(variant)

    def backbone_forward_args(self, *args):
        return args[:1]  # Image
