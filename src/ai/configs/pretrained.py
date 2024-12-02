from typing import Any, Callable, ClassVar, Literal

import torch.nn as nn
from pydantic import BaseModel, field_validator

from ..dataset.patches.patch import patch_linear
from ..nn.modules import Module
from .variants import VariantConfig


class PretrainedDatasetConfig(BaseModel):
    dataset: str

    # Whenever the dataset can has a different number of classes for example
    resolve_fn: Callable | None = None

    # TODO handle source specific configurations

    class Config:
        extra = "allow"


def patch_pretrained_linear(model: nn.Module, layer: str, mapping: str, **kwargs):
    module = Module._module_by_name(model, layer)
    patched_layer = patch_linear(module, mapping, **kwargs)
    Module._replace_module(model, module, patched_layer)


class PretrainedDatasetConfigs:
    """
    Pretrained dataset configurations.
    """

    # Popular imagenet dataset
    IMAGENET = PretrainedDatasetConfig(dataset="imagenet", num_classes=1000)
    # Popular COCO dataset for object detection
    COCO = PretrainedDatasetConfig(
        dataset="coco",
        num_classes=80,
        resolve_fn=lambda model, layer, **kwargs: patch_pretrained_linear(
            model, layer, "ms_coco_91_2_80", **kwargs
        ),
    )

    def from_name(name: str):
        return getattr(PretrainedDatasetConfigs, name.upper())


class PretrainedSourceConfig(BaseModel):
    class Config:
        exclude = ["forward_args"]

    variant: str | None = None
    source: Literal["timm", "torch", "torchvision"] | None = None
    weights: PretrainedDatasetConfig = None
    source_params: dict = {}
    weights_params: dict = {}
    state_params: dict = {}
    forward_args: list[Any] | Any = []

    @field_validator("forward_args", mode="before")
    def validate_forward_args(cls, value):
        if isinstance(value, list):
            return value
        return [value]

    @field_validator("weights", mode="before")
    def validate_weights(cls, value):
        if isinstance(value, str):
            return PretrainedDatasetConfigs.from_name(value)
        return value


class PretrainedConfig(VariantConfig):
    pretrained: list[PretrainedSourceConfig] = []
    pretrained_recommendations: ClassVar[list[PretrainedSourceConfig]] = []

    @field_validator("pretrained", mode="before")
    @classmethod
    def validate_pretrained(cls, value, values):
        data = values.data
        variant = data["variant"]
        if isinstance(value, PretrainedSourceConfig):
            return [value]
        elif isinstance(value, str):
            # TODO change to weights instead of source
            return [PretrainedSourceConfig(source=value, variant=variant)]
        elif isinstance(value, dict):
            return [PretrainedSourceConfig(**value)]
        elif isinstance(value, bool):
            if value:
                return cls.pretrained_for_variant(variant)
            else:
                return []
        # To list
        if isinstance(value, list):
            return value
        return value

    @classmethod
    def create_recommendations(
        cls,
        datasets: list[PretrainedDatasetConfig | str] | str,
        sources: list[str] | str = None,
        variants: list[str] | str = None,
        params: dict = {},
    ):
        if sources is None:
            sources = ["timm", "torch", "torchvision"]
        elif isinstance(sources, str):
            sources = [sources]
        if isinstance(datasets, str):
            datasets = [datasets]
        if variants is None:
            variants = cls.variants
        elif isinstance(variants, str):
            variants = [variants]
        return [
            PretrainedSourceConfig(
                variant=variant, source=source, weights=dataset, params=params
            )
            for dataset in datasets
            for source in sources
            for variant in variants
        ]

    @classmethod
    def pretrained_for_variant(cls, variant: str | None):
        if variant is None:
            return cls.pretrained_recommendations
        return [
            recommendation
            for recommendation in cls.pretrained_recommendations
            if recommendation.variant is None or recommendation.variant == variant
        ]

    @classmethod
    def pretrained_for_dataset(cls):
        recommended = cls.pretrained_for_variant()
        if cls.pretrained.weights is None:
            return recommended
        return [
            recommendation
            for recommendation in recommended
            if recommendation.weights.dataset == cls.pretrained.weights.dataset
        ]
