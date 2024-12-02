from __future__ import annotations

import inspect
from functools import cached_property
from typing import Any, Callable, ClassVar, Mapping

import numpy as np
import torch
import torch.nn as nn

from ...configs.base import ModuleConfig
from ...configs.log import Color, Loggable
from ..fusion.fuse import FusedModule


class Module(nn.Module, Loggable):
    """
    Base class for all models
    """

    log_name = "module"
    color: ClassVar[str] = Color.blue
    config: ClassVar[ModuleConfig] = None

    def __init__(self, config: ModuleConfig):
        super().__init__()
        self.config = config

    @classmethod
    def log(cls, *msg: list[Any], table=False):
        super().log(f"[{cls.__name__}]", *msg, table=table)

    @classmethod
    def _module_by_name(cls, m: nn.Module, name: str) -> nn.Module:
        current_name, *rest = name.split(".")
        if current_name == "":
            return m

        module = getattr(m, current_name)
        if len(rest) > 0:
            return cls._module_by_name(module, ".".join(rest))
        return module

    def module_by_name(self, name: str):
        """
        Get a module by name
        """
        return self._module_by_name(self, name)

    @classmethod
    def _replace_module(
        cls,
        module: nn.Module,
        criteria: str | nn.Module | type | Callable,
        other: nn.Module | type = nn.Identity(),
    ):
        """
        Replace a module by name, module, module class or predicate
        """
        criteria_name = criteria
        _cmp_fn = None
        if isinstance(criteria, str):
            # criteria is a string
            def _cmp_fn(name, _):
                return criteria in name
        elif inspect.isclass(criteria):
            # criteria is a class
            criteria_name = criteria.__name__

            def _cmp_fn(_, mod):
                return mod.__class__ == criteria
        elif isinstance(criteria, nn.Module):
            # criteria is a module
            criteria_name = criteria

            def _cmp_fn(_, mod):
                return mod == criteria
        elif callable(criteria):
            # criteria is a function
            criteria_name = criteria.__name__
            _cmp_fn = criteria
        else:
            raise ValueError(f"Unknown criteria type {criteria}")

        other_name = other if isinstance(other, nn.Module) else other.__name__
        cls.log(f"Replacing '{criteria_name}' with '{other_name}'")
        for name, mod in list(module.named_modules())[1:]:  # Exclude self
            if _cmp_fn(name, mod):
                replace_with = other
                # Handle weights if we want to replace with a FusedModule
                if inspect.isclass(replace_with) and issubclass(
                    replace_with, FusedModule
                ):
                    replace_with: FusedModule  # For typing
                    replace_with = replace_with.load_from(mod)

                if not isinstance(replace_with, nn.Module):
                    raise ValueError(
                        f"Cannot create module type {replace_with} automatically"
                    )

                # Get the parent module
                parent = Module._module_by_name(module, ".".join(name.split(".")[:-1]))
                mod_name = name.split(".")[-1]
                # Attach the new module
                setattr(parent, mod_name, replace_with)

    def replace_module(
        self,
        criteria: str | nn.Module | type | Callable,
        other: nn.Module | type = nn.Identity(),
    ):
        """
        Replace a module by name, module, module class or predicate
        """
        self._replace_module(self, criteria, other)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Load the state dict into the model and try to resolve shape missmatches
        """
        curr_state_dict = self.state_dict()
        no_missmatch_state_dict = state_dict.copy()
        # Iterate on the new values
        for k, tensor in state_dict.items():
            self_tensor = curr_state_dict[k]
            # Check shape of tensors
            t_shape = tuple(tensor.shape)
            s_shape = tuple(self_tensor.shape)
            if t_shape != s_shape:
                if np.prod(t_shape) == np.prod(s_shape):
                    # If the number of elements is the same, we can reshape
                    self.log(
                        f"Resolving shape missmatch - model {s_shape} -> weights {t_shape}"
                    )
                    no_missmatch_state_dict[k] = tensor.view_as(self_tensor)
                else:
                    self.log(
                        f"Removing '{k}' because of shape missmatch - model {s_shape} -> weights {t_shape}"
                    )
                    del no_missmatch_state_dict[k]

        self.log("Loading state dict")
        super().load_state_dict(no_missmatch_state_dict, strict=strict, assign=assign)

    @classmethod
    def forward_info(
        cls, module: nn.Module, forward_args: list[Any], ignores: list[str] = []
    ):
        """
        We utilize the forward hooks to compute relevant information about the model structure and weights
        """
        info = dict(order=[], shape=[], dtype=[], device=[])
        handles = []
        # Defined the order of weights from the passed module
        for k, v in module.named_modules():
            # Capture k in the closure
            def _hook(sub: nn.Module, *args, k=k):
                if isinstance(sub, nn.MultiheadAttention):
                    # MHA hides parameters
                    parameters = sub.named_parameters()
                else:
                    parameters = sub._parameters.items()

                # Get current module's parameters / buffers
                for pb, t in list(parameters) + list(sub._buffers.items()):
                    if t is None:
                        # Tensor can be None, in that case it will not be in the module state_dict
                        continue

                    if not any([i in pb for i in ignores]):
                        # We gather the info
                        order_name = f"{k}.{pb}" if k != "" else pb
                        info["order"].append(order_name)
                        info["shape"].append(t.shape)
                        info["dtype"].append(t.dtype)
                        info["device"].append(t.device)

            handles.append(v.register_forward_hook(_hook))

        # Call the hooks
        module.eval()(*forward_args)
        # Remove all the hooks
        for h in handles:
            h.remove()

        return info  # Order should be filled

    @cached_property
    def memoized_forward_info(self):
        """
        Get the forward information
        """
        return self.forward_info(self, self.config.forward_args)
