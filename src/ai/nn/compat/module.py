from __future__ import annotations

import inspect
from functools import cached_property
from inspect import signature
from itertools import chain
from typing import Any, Callable, ClassVar, Mapping, dataclass_transform

import numpy as np
import torch
import torch.nn as nn
from deepmerge import always_merger
from pydantic import Field, field_validator
from pydantic._internal._model_construction import ModelMetaclass, NoInitField
from pydantic.fields import Field as PydanticModelField
from pydantic.fields import PrivateAttr as PydanticModelPrivateAttr

from ...configs import Base, Color
from ...configs.external_base import ExternalBase
from ..fusion.fuse import FusedModule

BASE_MODULE_KEYS = dir(Base)
BASE_MODULE_KEYS.remove("__repr__")
NN_MODULE_KEYS = dir(nn.Module)
NN_MODULE_KEYS.remove("forward")

NN_MODULE_ANNOTATIONS = set(nn.Module.__annotations__.keys())
if "forward" in NN_MODULE_ANNOTATIONS:
    NN_MODULE_ANNOTATIONS.remove("forward")


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(PydanticModelField, PydanticModelPrivateAttr, NoInitField),
)
class _PydanticRemoveNNModuleForwardAnnotation(ModelMetaclass):
    """
    This class only serves to trick pydantic into not parsing the annotations in nn.Modules

    Since the forward method isn't an attribute of nn.Module, we need to remove it from the annotations
    because pydantic will parse the variable and interpret it as a field.

    Same goes for every other nn.Modules in pytorch.
    """

    KEYS = NN_MODULE_ANNOTATIONS

    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        # Remove forward from Pydantic's field parsing
        for base in bases:
            split = base.__module__.split(".")
            package_name = split[0]
            if package_name == "torch":
                module_annotations = base.__dict__.get("__annotations__", {})
                keys = set(module_annotations.keys())
                # Remove everything concerning nn.Module and keep subclass specifics
                keys -= mcs.KEYS

                sign = signature(base.__init__)
                sign = dict(sign.parameters)
                init_fn_params_list = list(sign.values())[1:]
                init_fn_params = {p.name: p for p in init_fn_params_list}

                if "forward" in keys:
                    del module_annotations["forward"]
                    keys.remove("forward")

                for k in set(keys) & set(init_fn_params):
                    param = init_fn_params[k]
                    if param.default != param.empty:
                        setattr(base, k, param.default)

        return super().__new__(mcs, cls_name, bases, namespace, **kwargs)


class Module(
    ExternalBase,
    nn.Module,
    buildable=False,
    metaclass=_PydanticRemoveNNModuleForwardAnnotation,
):
    """A pydantic compatible nn.Module"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    BASE_REMOVE_KEYS: ClassVar[list[str]] = ["forward"]

    log_name = "module"
    color: ClassVar[str] = Color.blue

    @classmethod
    def log_extras(cls):
        return f"[{cls.__name__}]"

    @classmethod
    def _module_by_name(cls, m: nn.Module, name: str) -> nn.Module:
        current_name, *rest = name.split(".")
        if current_name == "":
            return m

        module = getattr(m, current_name)
        if len(rest) > 0:
            return cls._module_by_name(module, ".".join(rest))
        return module

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
        cls.info(f"Replacing '{criteria_name}' with '{other_name}'")
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

    # Prevents training from being in the constructor
    @property
    def training(self):
        return self.INIT_CLS.training

    @property
    def device(self):
        params = self.parameters()
        buffers = self.buffers()
        gen = chain(params, buffers)
        try:
            next(gen).device
        except StopIteration:
            return torch.device("cpu")

    def init(self):
        """
        Initialize the model
        """
        super().init()
        for k, field in self.model_computed_fields.items():
            unwrapped = field.wrapped_property.fget(self)
            if k in self._XT_SPECIAL_KEYS or isinstance(unwrapped, nn.Module):
                raise ValueError(
                    f"Do not use any form of caching for Modules with pydantic. '{k}' is a Module",
                    "Caching prevents nn.Module from accessing it's 'real' modules from _modules, _buffers and _parameters when changing device",
                )

    def pydantic_post_init(self, pydantic_state, xt_state):
        """For modules we need to set the state after the pydantic init for it to
        go through the nn.Module's __setattr__ method and register as module, parameter or buffer
        """
        for k, v in pydantic_state.items():
            if isinstance(v, nn.Module):
                setattr(self, k, v)

    def should_go_to_xt_state(self, name, value):
        if isinstance(value, nn.Module):
            return True
        return super().should_go_to_xt_state(name, value)

    def __repr__(self):
        """Use nn.Module's __repr__ explicitly"""
        return self.INIT_CLS.__repr__(self)

    def __str__(self):
        return self.INIT_CLS.__str__(self)

    def extra_repr(self):
        def _get_src(obj, cached=False):
            elems = []
            for k, v in obj.items():
                if isinstance(v, cached_property):
                    v = f"{v.__get__(self)}"
                    k = f"{k} (cached)"

                elems.append(f"{k}: {v}")
            return elems

        mods = _get_src(self.model_dump(exclude_none=True, exclude_defaults=True))
        mods.extend(_get_src(self.__cached_properties__))
        return "\n".join(mods)

    def module_by_name(self, name: str):
        """
        Get a module by name
        """
        return self._module_by_name(self, name)

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
            if not isinstance(tensor, torch.Tensor):
                continue

            self_tensor = curr_state_dict[k]
            # Check shape of tensors
            t_shape = tuple(tensor.shape)
            s_shape = tuple(self_tensor.shape)
            if t_shape != s_shape:
                if np.prod(t_shape) == np.prod(s_shape):
                    # If the number of elements is the same, we can reshape
                    self.warn(
                        f"Resolving shape missmatch - model {s_shape} -> weights {t_shape}"
                    )
                    no_missmatch_state_dict[k] = tensor.view_as(self_tensor)
                else:
                    self.warn(
                        f"Removing '{k}' because of shape missmatch - model {s_shape} -> weights {t_shape}"
                    )
                    del no_missmatch_state_dict[k]

        self.info("Loading state dict")
        super().load_state_dict(no_missmatch_state_dict, strict=strict, assign=assign)

    # @cached_property
    # def memoized_forward_info(self):
    #     """
    #     Get the forward information
    #     """
    #     return self.forward_info(self, self.forward_args)

    @classmethod
    def create_classes(
        cls,
        *,
        namespace: dict[str, Any],
        module: type,
        selected_names: list[str] = None,
        required_base: type = nn.Module,
    ):
        super().create_classes(
            namespace=namespace,
            module=module,
            selected_names=selected_names,
            required_base=required_base,
        )

    def __hash__(self):
        return id(self)


Module.create_classes(namespace=globals(), module=nn.modules)


class ModuleConfig(Base):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    type: str = Field(default=None, validate_default=True)
    train: bool = False

    @field_validator("type", mode="before")
    def validate_type(cls, value):
        if value is None:
            name = cls.__name__.split("Config")[0]
            return name
        return value

    def merge(self, config: ModuleConfig | dict, **kwargs):
        if isinstance(config, ModuleConfig):
            # We are receiving a config
            config = config.model_dump()

        merged = {}
        always_merger.merge(merged, self.model_dump(exclude_none=True))
        always_merger.merge(merged, config)
        always_merger.merge(merged, kwargs)

        return self.__class__(**merged)


if __name__ == "__main__":

    class TestModule(Module):
        def init(self):
            self.linear = nn.Linear(1, 10)

        @cached_property
        def test_cache(self):
            return 1

        def forward(self, x):
            return self.linear(x)

    module = TestModule()

    print(module.test_cache)

    print(module)
