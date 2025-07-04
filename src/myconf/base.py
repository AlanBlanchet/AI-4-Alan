"""Base MyConf classes"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .core import F, PropertyInfo

# Add the myconf directory to Python path for relative imports
myconf_dir = Path(__file__).parent
if str(myconf_dir) not in sys.path:
    sys.path.insert(0, str(myconf_dir))

# Import dataclass_transform for proper IDE support
try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform


@dataclass_transform(
    init_only_init=False,
    kw_only_default=True,
    field_specifiers=(F,),
)
class MetaMyConf(type):
    """Metaclass for MyConf"""

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Process private attributes before creating the class
        private_attrs = {}
        annotations = namespace.get("__annotations__", {}).copy()

        # Process namespace values that are private attributes
        for attr_name, attr_value in list(namespace.items()):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                # Skip methods (callable objects) - they should remain as methods
                if callable(attr_value):
                    continue

                # Skip cached_property and other descriptors - let them work normally
                from functools import cached_property

                if isinstance(attr_value, cached_property):
                    continue

                # Handle private attributes

                if isinstance(attr_value, PropertyInfo):
                    # If it's already an F() Info object, just set init=False
                    attr_value.init = False
                    private_attrs[attr_name] = attr_value
                else:
                    # This is a regular private attribute, convert it to F with init=False
                    private_attrs[attr_name] = F(attr_value, init=False)

                # Remove from namespace so static analyzers never see it
                del namespace[attr_name]
                # Also remove from annotations if present
                annotations.pop(attr_name, None)

        # Process annotations that are private attributes (without values in namespace)
        for attr_name, attr_type in list(annotations.items()):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                # Private attribute with only annotation, no default value
                if (
                    attr_name not in private_attrs
                ):  # Don't override if already processed above
                    private_attrs[attr_name] = F(init=False)
                # Remove from annotations so static analyzers never see it
                del annotations[attr_name]

        # Update the cleaned annotations
        namespace["__annotations__"] = annotations

        # Store private attributes to be processed later
        namespace["_private_attrs"] = private_attrs

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Call __set_name__ on all descriptors (including cached_property)
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__set_name__"):
                attr_value.__set_name__(cls, attr_name)

        # Auto-wrap methods with type conversion
        cls._wrap_methods_with_auto_convert()

        # Auto-generate stub files for better IDE support with Cast types
        try:
            from .stub_generation import auto_generate_stubs

            properties = getattr(cls, "_myconf_properties", {})
            has_cast_types = any(
                getattr(info, "is_cast", False) for info in properties.values()
            )
            if has_cast_types:
                auto_generate_stubs(cls)
        except Exception:
            # Silently ignore stub generation errors
            pass

        return cls

    def _wrap_methods_with_auto_convert(cls):
        """Automatically wrap methods that have type hints with auto_convert"""
        import inspect
        from typing import get_type_hints

        from .decorators import auto_convert

        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(cls, attr_name)

            # Only wrap methods defined in this class (not inherited)
            if callable(attr_value) and attr_name in cls.__dict__:
                try:
                    # Check if method has Cast type annotations
                    type_hints = get_type_hints(attr_value)
                    sig = inspect.signature(attr_value)

                    # Skip if no parameters or only 'self'
                    params = list(sig.parameters.keys())
                    if len(params) <= 1:  # Only self or no params
                        continue

                    # Check if any parameter (except self) has Cast type hints
                    has_cast_params = False
                    for param_name in params[1:]:  # Skip 'self'
                        if param_name in type_hints:
                            type_hint = type_hints[param_name]
                            origin = getattr(type_hint, "__origin__", None)
                            from .core import Cast

                            if origin is Cast:
                                has_cast_params = True
                                break

                    if has_cast_params:
                        # Wrap the method with auto_convert
                        wrapped_method = auto_convert(attr_value)
                        setattr(cls, attr_name, wrapped_method)

                except Exception:
                    # If anything fails, skip this method
                    continue


class MyConf(metaclass=MetaMyConf):
    """Base configuration class"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if __init__ is manually defined in this class (not inherited)
        if "__init__" in cls.__dict__:
            # Allow __init__ for classes that inherit from torch.Tensor or have special cases
            init_method = cls.__dict__["__init__"]

            # Check if this class inherits from torch.Tensor (special case)
            try:
                import torch

                is_tensor_subclass = any(
                    issubclass(base, torch.Tensor) for base in cls.__mro__
                )
            except ImportError:
                is_tensor_subclass = False

            # Check if the __init__ is just a pass-through (simple case)
            import inspect

            try:
                source = inspect.getsource(init_method).strip()
                is_simple_pass = "pass" in source and len(source.split("\n")) <= 3
            except (OSError, TypeError):
                is_simple_pass = False

            if not (is_tensor_subclass or is_simple_pass):
                raise TypeError(
                    f"Class {cls.__name__} defines __init__ method. MyConf classes should not define __init__ manually - initialization is automatically generated from type hints."
                )

        # Process any private attributes that were detected by the metaclass
        private_attrs = getattr(cls, "_private_attrs", {})
        if private_attrs:
            # Add them to the class so they can be used normally
            for attr_name, f_instance in private_attrs.items():
                setattr(cls, attr_name, f_instance)

        from myconf.init_generation import generate_init
        from myconf.property_processing import update_class_signatures

        generate_init(cls)
        update_class_signatures(cls)

    def __str__(self):
        """String representation with hierarchical display for modules"""
        cls_name = self.__class__.__name__
        if not hasattr(self.__class__, "_myconf_properties"):
            return object.__str__(self)

        # Check if this is a PyTorch module for hierarchical display
        try:
            import torch.nn as nn

            is_module = isinstance(self, nn.Module)
        except ImportError:
            is_module = False

        # Check if we have any module properties for hierarchical display
        has_modules = False
        if hasattr(self, "_modules") and self._modules:
            has_modules = True
        elif hasattr(self.__class__, "_myconf_properties"):
            # Check if we have any module properties in MyConf
            properties = self.__class__._myconf_properties
            for name, info in properties.items():
                if hasattr(self, name):
                    value = getattr(self, name)
                    try:
                        import torch.nn as nn

                        if isinstance(value, (nn.Module, nn.ModuleList)):
                            has_modules = True
                            break
                    except ImportError:
                        pass

        if is_module and has_modules:
            # Use hierarchical representation for modules with submodules
            return self._hierarchical_repr()
        else:
            # Use flat representation for simple objects
            return self._flat_repr()

    def _flat_repr(self):
        """Flat representation like current MyConf"""
        cls_name = self.__class__.__name__
        parts = []
        properties = self.__class__._myconf_properties
        for name, info in properties.items():
            if not name.startswith("_") and hasattr(self, name):
                value = getattr(self, name)

                # Skip if this value matches the default value
                if info.value is not None and value == info.value:
                    continue

                if hasattr(value.__class__, "_myconf_properties"):
                    value_str = str(value)
                elif isinstance(value, str):
                    value_str = value  # No quotes for strings
                elif hasattr(value, "__name__"):
                    value_str = value.__name__
                else:
                    value_str = repr(value)
                parts.append(f"{name}={value_str}")

        if parts:
            return f"{cls_name}({', '.join(parts)})"
        else:
            return f"{cls_name}()"

    def _hierarchical_repr(self):
        """PyTorch-like hierarchical representation for modules"""
        cls_name = self.__class__.__name__

        # Collect simple properties and module properties separately
        simple_props = []
        module_props = []
        seen_modules = set()  # Track seen modules to avoid duplication

        if hasattr(self.__class__, "_myconf_properties"):
            properties = self.__class__._myconf_properties
            for name, info in properties.items():
                if not name.startswith("_") and hasattr(self, name):
                    value = getattr(self, name)

                    # Check if this is a module (torch.nn.Module or ModuleList)
                    is_module = False
                    try:
                        import torch.nn as nn

                        if isinstance(value, (nn.Module, nn.ModuleList)):
                            is_module = True
                    except ImportError:
                        pass

                    if is_module:
                        # Add module properties (private properties already filtered out above)
                        module_props.append((name, value))
                        seen_modules.add(name)
                    else:
                        # Handle simple properties (only include non-private properties for hierarchical view)
                        # Skip if this value matches the default value
                        from .core import has_value

                        if has_value(info) and value == info.value:
                            continue

                        if hasattr(value.__class__, "_myconf_properties"):
                            value_str = str(value)
                        elif isinstance(value, str):
                            value_str = value
                        elif isinstance(value, (int, float, bool, type(None))):
                            value_str = str(value)
                        elif hasattr(value, "__name__"):
                            value_str = value.__name__
                        else:
                            value_str = repr(value)
                        simple_props.append((name, value_str))

        # Also check pytorch _modules if they exist (but avoid duplicates)
        try:
            import torch.nn as nn

            if hasattr(self, "_modules"):
                for name, module in self._modules.items():
                    if module is not None and name not in seen_modules:
                        module_props.append((name, module))
        except ImportError:
            pass

        # Start building the representation - always start with just class name
        lines = [f"{cls_name}("]

        # Add simple properties first
        for name, value_str in simple_props:
            lines.append(f"  {name}={value_str}")

        # Add module properties
        for name, module in module_props:
            self._add_module_to_repr(lines, name, module, indent=1)

        lines.append(")")
        return "\n".join(lines)

    def _add_module_to_repr(self, lines, name, module, indent=1):
        """Add a module to the representation with proper indentation"""
        indent_str = "  " * indent

        try:
            import torch.nn as nn

            # Handle ModuleList specially
            if isinstance(module, nn.ModuleList):
                lines.append(f"{indent_str}{name}=ModuleList(")
                for i, submodule in enumerate(module):
                    self._add_module_to_repr(lines, str(i), submodule, indent + 1)
                lines.append(f"{indent_str})")
                return
        except ImportError:
            pass

        # Get module representation
        module_str = str(module)

        if "\n" in module_str:
            # Multi-line module: indent each line
            module_lines = module_str.split("\n")
            lines.append(f"{indent_str}{name}={module_lines[0]}")
            for line in module_lines[1:]:
                if line.strip():  # Skip empty lines
                    lines.append(f"{indent_str}{line}")
        else:
            # Single-line module
            lines.append(f"{indent_str}{name}={module_str}")

    def config(self):
        """Get configuration dictionary"""
        result = {}
        if hasattr(self.__class__, "_myconf_properties"):
            properties = self.__class__._myconf_properties
            for name, info in properties.items():
                if hasattr(self, name):
                    value = getattr(self, name)
                    if hasattr(value.__class__, "_myconf_properties"):
                        result[name] = value.config()
                    elif isinstance(value, list):
                        result[name] = [
                            item.config()
                            if hasattr(item.__class__, "_myconf_properties")
                            else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        result[name] = {
                            k: v.config()
                            if hasattr(v.__class__, "_myconf_properties")
                            else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, Path):
                        result[name] = str(value)
                    else:
                        result[name] = value
        return result

    def save(self, path: Path, fmt: list[str] = None):
        """Save MyConf instance to file"""
        indent = 2
        if fmt and fmt[0]:
            indent = int(fmt[0])

        with open(path, "w") as f:
            json.dump(self.config(), f, indent=indent)

    @classmethod
    def load(cls, path: Path):
        """Load configuration from file"""
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)

    @classmethod
    def from_config(cls, config):
        """Create instance from config dictionary"""
        if len(config) == 1 and isinstance(list(config.values())[0], dict):
            class_name = list(config.keys())[0]
            for module in sys.modules.values():
                if module and hasattr(module, class_name):
                    target_cls = getattr(module, class_name)
                    if hasattr(target_cls, "_myconf_properties"):
                        inner_config = config[class_name]
                        return target_cls(**inner_config)
        return cls(**config)

    def __setattr__(self, name, value):
        """Handle property assignment with type conversion"""
        # Avoid recursion during import or class setup
        if name.startswith("_") or not hasattr(self, "__dict__"):
            object.__setattr__(self, name, value)
            return

        try:
            cls = object.__getattribute__(self, "__class__")
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        if hasattr(cls, "_myconf_properties") and name in cls._myconf_properties:
            info = cls._myconf_properties[name]
            # Only do type conversion if this is not an F() property
            from .core import has_fn

            if (
                hasattr(info, "annotation")
                and info.annotation is not None
                and not has_fn(info)
            ):
                from .type_conversion import _format

                value = _format(value, info.annotation)

        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if name in ("__class__", "__dict__"):
            return object.__getattribute__(self, name)

        cls = object.__getattribute__(self, "__class__")

        # Handle cached_property descriptors
        for klass in cls.__mro__:
            if name in klass.__dict__:
                attr = klass.__dict__[name]
                from functools import cached_property

                if isinstance(attr, cached_property):
                    return attr.__get__(self, klass)
                break

        # Try to get the attribute from instance first
        try:
            value = object.__getattribute__(self, name)
        except AttributeError:
            # Fallback to class attributes
            value = getattr(cls, name)

        # Handle ClassVar type conversion
        annotations = getattr(cls, "__annotations__", {})
        if name in annotations:
            from .utils import is_class_var

            annotation = annotations[name]
            if (
                is_class_var(annotation)
                and hasattr(annotation, "__args__")
                and annotation.__args__
            ):
                inner_type = annotation.__args__[0]
                from .type_conversion import _format

                return _format(value, inner_type)

        return value

    def __getattr__(self, name):
        # Check if any parent class has a __getattr__ method
        cls = object.__getattribute__(self, "__class__")
        for base in cls.__mro__[1:]:
            if (
                hasattr(base, "__getattr__")
                and not hasattr(base, "_myconf_properties")
                and base.__name__ != "MyConf"
            ):
                return base.__getattr__(self, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
