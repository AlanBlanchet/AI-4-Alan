# import json
# from abc import ABC, abstractmethod
# from typing import AnyStr, Literal, Union

# import torch
# import torch.nn as nn
# import torchvision
# from pyparsing import Any
# from timm import create_model

# from ...configs.pretrained import PretrainedConfig, PretrainedSourceConfig
# from ..modules import Module


from typing import Any, Callable, ClassVar, Literal

import torch.nn as nn
from pydantic import BaseModel, field_validator

from ...dataset.patches.patch import patch_linear
from .module import Module
from .variants import VariantConfig


class Pretrained: ...


# class Pretrained(Module, buildable=False):
#     """
#     Definition for the already trained models
#     """

#     config: PretrainedConfig

#     def init_weights(self, *forward_args):
#         """
#         Make the model compatible with external source of weights
#         """
#         if self.config.pretrained:
#             self._load_pretrained(*forward_args)
#         else:
#             "Using random weights")

#     def _load_pretrained(self, *forward_args):
#         """
#         Load the weights from the external source
#         """
#         pretrained = self.config.pretrained
#         model_name = self.__class__.__name__.lower()

#         for rec in pretrained:
#             try:
#                 variant = rec.variant
#                 load_name = f"{rec.source}/{rec.weights.dataset}"
#                 f"Trying to load {load_name}")
#                 name = model_name + ("" if variant is None else variant)
#                 params = {"model_name": name, **rec.source_params}
#                 name = params.pop("model_name")
#                 model = SOURCES[rec.source](model_name=name, **params)
#                 f"Successfully loaded {load_name}")
#                 break
#             except Exception as _:
#                 continue
#         else:
#             raise ValueError(
#                 f"No source for pretrained weights could be found for {model_name}"
#             )

#         # Resolve the weights if needed
#         resolve_fn = rec.weights.resolve_fn
#         if resolve_fn:
#             resolve_fn(model, **rec.weights_params)

#         # Load the pretrained model into the current model
#         self.load_state(
#             model, pretrained=rec, forward_args=forward_args, **rec.state_params
#         )

#     def load_state(
#         self,
#         source: nn.Module,
#         pretrained: PretrainedSourceConfig,
#         forward_args: list[Any],
#         ignores: list[str] = [],
#         additional_mapping: dict = {},
#     ):
#         """
#         Load the state of the nn.Module from another nn.Module

#         This function throws a tensor to the model and computes the order of insertion for the weights
#         """
#         "Making the model compatible with the source model")
#         # curr_input = self.config.forward_args
#         source_input = (
#             pretrained.forward_args if pretrained.forward_args else forward_args
#         )
#         # Compute order
#         source_order = self.forward_info(source, source_input, ignores=ignores)["order"]
#         current_order = self.forward_info(self, forward_args, ignores=ignores)["order"]

#         # TODO remove
#         json.dump(source_order, open("source_order.json", "w"))
#         json.dump(current_order, open("current_order.json", "w"))

#         # Get unique values
#         source_order = [
#             *list(dict.fromkeys(source_order)),
#             *list(additional_mapping.keys()),
#         ]
#         current_order = [
#             *list(dict.fromkeys(current_order)),
#             *list(additional_mapping.values()),
#         ]

#         # Check if the models have the same number of active modules
#         if len(source_order) != len(current_order):
#             raise ValueError(
#                 f"Source model has {len(source_order)} parameters while current model has {len(current_order)} parameters"
#             )

#         # Create the mapping
#         mapping = dict(zip(source_order, current_order))

#         # TODO remove
#         json.dump(mapping, open("mapping.json", "w"))

#         state_dict = {}
#         for k, v in source.state_dict().items():
#             # Map to new key
#             if k in mapping:
#                 new_key = mapping[k]
#             else:
#
#                     f"Key {k} not found in mapping. Using the same key for mapping"
#                 )
#                 new_key = k
#             state_dict[new_key] = v

#         # Prune useless weights from our model
#         for k, m in self.named_modules():
#             # Check if in keys and has parameters / buffers
#             if (
#                 not any([k in curr_k for curr_k in current_order])
#                 and len(m._parameters) > 0
#             ):
#                 f"Pruning {k} since it is not used")
#                 self.replace_module(k)

#         self.load_state_dict(state_dict, strict=False)


# class ISSD(ABC):
#     """
#     Layer definition for the SSD detection boxes
#     """

#     def __init__(self, **kwargs): ...

#     @abstractmethod
#     def ssd_compat(
#         self,
#     ) -> dict[Union[Literal["features"], AnyStr], list[int]]:
#         """
#         Make your model compatible with the SSD detection boxes
#         """
#         ...


# class ITimm(ABC):
#     """
#     Layer definition for the Timm models
#     """

#     @abstractmethod
#     def timm_compat(self) -> dict[str, AnyStr]:
#         """
#         Make your model compatible with the Timm models
#         """
#         ...


# SOURCES = dict(
#     timm=lambda model_name, **kwargs: create_model(
#         model_name, pretrained=True, **kwargs
#     ),
#     torch=lambda model_name, repo="pytorch/vision", **kwargs: torch.hub.load(
#         repo, model_name, **kwargs
#     ),
#     torchvision=lambda model_name, **kwargs: getattr(torchvision.models, model_name)(
#         weights="DEFAULT", **kwargs
#     ),
# )


class PretrainedDatasetConfig(BaseModel):
    model_config = {"extra": "allow"}

    dataset: str

    # Whenever the dataset can has a different number of classes for example
    resolve_fn: Callable | None = None

    # TODO handle source specific configurations


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
    model_config = {"exclude": {"forward_args"}}

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
