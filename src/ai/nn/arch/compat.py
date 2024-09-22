import json
from abc import ABC, abstractmethod
from typing import Any, AnyStr, Literal, Union

import torch
import torch.nn as nn
import torchvision
from timm import create_model

from ...configs.models import PretrainedConfig, PretrainedSourceConfig
from ..modules import Module


class Pretrained(Module):
    """
    Definition for the already trained models
    """

    config: PretrainedConfig

    @abstractmethod
    def init_weights(self):
        """
        Make the model compatible with external source of weights
        """
        raise NotImplementedError

    def load_pretrained(self):
        """
        Load the weights from the external source
        """
        pretrained = self.config.pretrained
        model_name = self.__class__.__name__.lower()
        if isinstance(pretrained, str):
            # TODO implement loading from path / name
            raise NotImplementedError(
                "Pretrained weights from path / excplicit name is not yet supported"
            )
        elif isinstance(pretrained, PretrainedSourceConfig):
            # Resolve the source weights / source
            if pretrained.weights:
                # TODO change the weights to load
                raise NotImplementedError(
                    f"Pretrained weights from source {pretrained.weights} is not yet supported"
                )

            source = pretrained.source
            if source in SOURCES:
                self.log(f"Loading from {source}")
                model = SOURCES[source](model_name)
            else:
                raise NotImplementedError(
                    f"Pretrained weights from source {source} is not yet supported"
                )
        else:
            # Load default weights if they exist
            for source, fn in SOURCES.items():
                try:
                    self.log(f"Trying to load from {source}")
                    model = fn(model_name)
                    break
                except Exception as _:
                    continue
            else:
                raise ValueError("No source for pretrained weights could be found")
        # Load the pretrained model into the current model
        self.load_state(model)

    def _compute_weight_order(self, module: nn.Module, input: Any):
        order = []
        # Defined the order of weights from the passed module
        for k, v in module.named_modules():
            # Capture k in the closure
            def _hook(sub: nn.Module, *args, k=k):
                # Get current module's parameters
                for p in sub._parameters:
                    order.append(f"{k}.{p}")

                # Get the module's buffers
                for b in sub._buffers:
                    order.append(f"{k}.{b}")

            v.register_forward_hook(_hook)
        # Call the hooks
        module.eval()(input)
        # Remove all the hooks
        for k, v in module.named_modules():
            v._forward_hooks = {}
        return order  # Order should be filled

    def load_state(self, source: nn.Module):
        """
        Load the state of the nn.Module from another nn.Module

        This function throws a parameter to the model and computes the order of insertion for the weights
        """
        x = torch.rand(1, 3, 224, 224)

        # Compute order
        source_order = self._compute_weight_order(source, x)
        current_order = self._compute_weight_order(self, x)

        # Check if the models have the same number of active modules
        if len(source_order) != len(current_order):
            raise ValueError(
                f"Source model has {len(source_order)} parameters while current model has {len(current_order)} parameters"
            )

        # Create the mapping
        mapping = dict(zip(source_order, current_order))

        json.dump(mapping, open("mapping.json", "w"))

        state_dict = {}
        for k, v in source.state_dict().items():
            new_key = mapping[k]
            state_dict[new_key] = v

        self.load_state_dict(state_dict, strict=True)


class ISSD(ABC):
    """
    Layer definition for the SSD detection boxes
    """

    def __init__(self, **kwargs): ...

    @abstractmethod
    def ssd_compat(
        self,
    ) -> dict[Union[Literal["features"], AnyStr], list[int]]:
        """
        Make your model compatible with the SSD detection boxes
        """
        ...


class ITimm(ABC):
    """
    Layer definition for the Timm models
    """

    @abstractmethod
    def timm_compat(self) -> dict[str, AnyStr]:
        """
        Make your model compatible with the Timm models
        """
        ...


SOURCES = dict(
    timm=lambda model_name: create_model(model_name, pretrained=True),
    torch=lambda model_name: torch.hub.load(
        "pytorch/vision", model_name, pretrained=True
    ),
    torchvision=lambda model_name: getattr(torchvision.models, model_name)(
        pretrained=True
    ),
)
