from __future__ import annotations

import inspect
from collections import defaultdict
from itertools import chain
from typing import Callable, ClassVar, Generic, Mapping, TypeVar

import torch
import torch.nn as nn

from ...configs.base import Base
from ..fusion.fuse import FusedModule


class Module(Base, nn.Module):
    """A pydantic compatible nn.Module"""

    log_name: ClassVar[str] = "module"
    color: ClassVar[str] = "blue"

    def __getattribute__(self, name):
        """Override to auto-register nn.Module instances on first access"""
        value = super().__getattribute__(name)

        # If this is an nn.Module and we have _modules, register it
        if (
            isinstance(value, nn.Module)
            and hasattr(self, "_modules")
            and not name.startswith("__")
            and name not in self._modules
        ):
            # Clean up the name for PyTorch registration
            module_name = name.lstrip("_") or name
            self._modules[module_name] = value

            # If the module is also a Module class, ensure its modules are registered
            if hasattr(value, "_ensure_all_modules_registered"):
                value._ensure_all_modules_registered()

        return value

    def parameters(self, recurse: bool = True):
        """Override to ensure all F fields are registered before returning parameters"""
        self._ensure_all_modules_registered()
        return super().parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        """Override to ensure all F fields are registered before returning named parameters"""
        self._ensure_all_modules_registered()
        return super().named_parameters(prefix, recurse, remove_duplicate)

    def _ensure_all_modules_registered(self):
        """Access all F field attributes to trigger registration recursively"""
        if not hasattr(self, "_myconf_properties"):
            return

        for prop_name, prop_info in self._myconf_properties.items():
            # Only check F fields (function-based properties)
            if hasattr(prop_info, "fn") and prop_info.fn is not None:
                try:
                    # Access the attribute to trigger __getattribute__ registration
                    value = getattr(self, prop_name)

                    # If the value is also a Module, ensure its F fields are registered
                    if hasattr(value, "_ensure_all_modules_registered"):
                        value._ensure_all_modules_registered()

                    # If it's a ModuleList, ensure all contained modules are registered
                    if isinstance(value, nn.ModuleList):
                        for item in value:
                            if hasattr(item, "_ensure_all_modules_registered"):
                                item._ensure_all_modules_registered()
                            # Also force access to trigger registration in parent
                            if isinstance(item, nn.Module):
                                # Force registration by re-accessing nested modules
                                for name, module in item.named_modules():
                                    pass  # Accessing triggers registration
                except (AttributeError, RuntimeError):
                    # Skip if accessing the attribute fails
                    pass

    def __setattr__(self, name, value):
        # Let Base handle its own attribute setting
        super().__setattr__(name, value)

        # If we're initialized and this is a PyTorch module, register it
        if (
            isinstance(value, nn.Module)
            and not name.startswith("_")
            and hasattr(self, "_modules")
        ):
            self._modules[name] = value

    def __repr__(self):
        """Smart representation that balances detail and collapsing"""
        # Get the MyConf properties
        if not hasattr(self, "_myconf_properties"):
            return f"{self.__class__.__name__}()"

        # Use annotation order for parameter ordering, like MyConf does
        annots = getattr(self.__class__, "__annotations__", {})
        prop_names = [
            k
            for k in annots.keys()
            if k in self._myconf_properties
            and k not in ("args", "kwargs")
            and not k.startswith("_")
        ]

        # Extract only the most important parameters for compact display
        important_params = []
        for k in prop_names:
            info = self._myconf_properties[k]

            # Get the value
            if hasattr(self, k):
                v = getattr(self, k)
            elif hasattr(info, "fn") and info.fn is not None:
                try:
                    v = info.fn(self)
                except:
                    continue  # Skip if function fails
            elif hasattr(info, "value") and info.value is not None:
                v = info.value
            else:
                continue

            # Skip empty or default-like values
            if (
                hasattr(v, "__class__") and v.__class__.__name__ == "_empty"
            ) or v is None:
                continue

            # Skip if this is the default value from the annotation
            if hasattr(info, "value") and info.value is not None and v == info.value:
                continue

            # Format value compactly
            if isinstance(v, list) and len(v) <= 3:
                # Show short lists inline
                important_params.append(f"{k}={v}")
            elif isinstance(v, (int, float, str)) and len(str(v)) <= 10:
                # Show simple values
                important_params.append(f"{k}={v}")
            elif hasattr(v, "__name__"):
                # Show class names for types/functions
                important_params.append(f"{k}={v.__name__}")
            # Skip complex objects to keep compact

        if important_params:
            return f"{self.__class__.__name__}({', '.join(important_params)})"
        else:
            return f"{self.__class__.__name__}()"

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

    @property
    def device(self):
        params = self.parameters()
        buffers = self.buffers()
        gen = chain(params, buffers)
        try:
            next(gen).device
        except StopIteration:
            return torch.device("cpu")

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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Override to ensure all F fields are registered before returning state dict"""
        self._ensure_all_modules_registered()
        return super().state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]):
        """
        Load the state dict into the model and try to resolve shape missmatches
        """
        curr_groups = defaultdict(list)
        other_groups = defaultdict(list)
        curr_state_dict: dict[str, torch.Tensor] = self.state_dict()

        final_state_dict = {}

        for kc, vc in curr_state_dict.items():
            priority1, priority2, priority3 = {}, {}, {}
            for ko, vo in state_dict.items():
                if kc == ko:
                    # TODO check if this can cause problems
                    priority1[kc] = vo
                elif vc.shape == vo.shape:
                    priority2[kc] = vo
                elif vc.numel() == vo.numel():
                    priority3[kc] = vo

            p1, p2, p3 = len(priority1), len(priority2), len(priority3)
            if p1:
                assert p1 == 1
                final_state_dict[kc] = priority1[kc]
            elif p2:
                assert p2 == 1
                final_state_dict[kc] = priority2[kc]
            elif p3:
                assert p3 == 1
                final_state_dict[kc] = priority3[kc]
            else:
                self.log_warn(
                    f"Can't find any pretrained weights for '{kc}' ({tuple(vc.shape)})"
                )

        super().load_state_dict(final_state_dict, strict=False, assign=True)

        # no_missmatch_state_dict = state_dict.copy()
        # # Iterate on the new values
        # for k, tensor in state_dict.items():
        #     if not isinstance(tensor, torch.Tensor):
        #         continue

        #     self_tensor = curr_state_dict[k]
        #     # Check shape of tensors
        #     t_shape = tuple(tensor.shape)
        #     s_shape = tuple(self_tensor.shape)
        #     if t_shape != s_shape:
        #         if np.prod(t_shape) == np.prod(s_shape):
        #             # If the number of elements is the same, we can reshape
        #             self.warn(
        #                 f"Resolving shape missmatch - model {s_shape} -> weights {t_shape}"
        #             )
        #             no_missmatch_state_dict[k] = tensor.view_as(self_tensor)
        #         else:
        #             self.warn(
        #                 f"Removing '{k}' because of shape missmatch - model {s_shape} -> weights {t_shape}"
        #             )
        #             del no_missmatch_state_dict[k]

        # self.info("Loading state dict")
        # super().load_state_dict(no_missmatch_state_dict, strict=strict, assign=assign)

    def weight_accept(self, tensor: torch.Tensor) -> bool:
        """
        Check if the model can accept the weight tensor
        """
        # Get the first parameter's shape as a reference
        for param in self.parameters(False):
            if param.shape == tensor.shape:
                return True
            elif param.numel() == tensor.numel():
                # If the number of elements matches, we can reshape
                self.log_warn(
                    f"Reshaping {tensor.shape=} to match model {param.shape=}"
                )
                return True
        return False


T = TypeVar("T", bound=Module)


class ModuleList(nn.ModuleList, Module, Generic[T]):
    def __getitem__(self, index: int) -> T:
        return super().__getitem__(index)

    def parameters(self, recurse: bool = True):
        """Override to completely bypass the custom Module.parameters method"""
        # For ModuleList, we want standard PyTorch behavior, not custom MyConf behavior
        # This ensures parameters are found in nested modules
        for param in self._parameters.values():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        """Override to completely bypass the custom Module.named_parameters method"""
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def _ensure_all_modules_registered(self):
        """For ModuleList, just ensure the modules are properly added"""
        # Don't call parent _ensure_all_modules_registered as it interferes
        pass

    def __repr__(self):
        """PyTorch-like hierarchical representation for ModuleList"""
        if len(self) == 0:
            return f"{self.__class__.__name__}()"

        # Create hierarchical representation like PyTorch
        lines = [f"{self.__class__.__name__}("]
        for i, module in enumerate(self):
            module_str = repr(module)
            # Indent the module representation
            if "\n" in module_str:
                # Multi-line module: indent each line
                module_lines = module_str.split("\n")
                first_line = f"  ({i}): {module_lines[0]}"
                lines.append(first_line)
                for line in module_lines[1:]:
                    lines.append(f"  {line}")
            else:
                # Single-line module
                lines.append(f"  ({i}): {module_str}")
        lines.append(")")
        return "\n".join(lines)


if __name__ == "__main__":

    class TestModule(Module):
        linear: nn.Linear = nn.Linear(1, 10)

        def forward(self, x):
            return self.linear(x)

    module = TestModule()
    print(module)
