"""Init generation for MyConf"""

from textwrap import dedent

from .core import has_fn
from .type_conversion import _format


def _apply_automatic_conversion(value, target_type):
    """Apply automatic type conversion to match target type"""
    if target_type is None:
        return value

    # If value is already the target type, no conversion needed
    if isinstance(value, target_type):
        return value

    # Special handling for tensor loading
    if hasattr(value, "load_tensor"):
        try:
            return value.load_tensor()
        except Exception:
            pass

    # Try to convert using target type constructor
    if hasattr(target_type, "__call__"):
        try:
            return target_type(value)
        except Exception:
            # If conversion fails, return original value
            return value

    return value


def generate_init(cls):
    """Generate __init__ method for a class"""
    # CRITICAL: Set up the properties first before using them
    from .property_processing import setup_myconf_properties

    setup_myconf_properties(cls)

    properties = getattr(cls, "_myconf_properties", {})

    def __init__(self, *args, **kwargs):
        # Check if class has a custom __new__ method and extract its parameter order
        positional_params = []

        # Look for custom __new__ method in the MRO
        custom_new_method = None
        for base_cls in cls.__mro__:
            if (
                hasattr(base_cls, "__new__")
                and base_cls.__new__ != object.__new__
                and getattr(base_cls.__new__, "__self__", None) is not object
            ):
                custom_new_method = base_cls.__new__
                break

        if custom_new_method:
            import inspect

            try:
                new_sig = inspect.signature(custom_new_method)
                # Get parameter names from __new__, skipping 'cls'
                new_param_names = list(new_sig.parameters.keys())[1:]

                # Map __new__ parameters to properties that exist and have init=True
                for param_name in new_param_names:
                    if (
                        param_name in properties
                        and properties[param_name].init
                        and not param_name.startswith("_")
                    ):
                        positional_params.append(param_name)
            except Exception:
                pass

        # If no custom __new__ or couldn't extract params, use declaration order
        if not positional_params:
            # Respect declaration order by using class annotations order
            annotations = getattr(cls, "__annotations__", {})
            ordered_prop_names = list(annotations.keys())

            # Add any properties not in annotations (from parent classes) at the end
            for name in properties.keys():
                if name not in ordered_prop_names and not name.startswith("_"):
                    ordered_prop_names.append(name)

            for name in ordered_prop_names:
                if (
                    name in properties
                    and not name.startswith("_")
                    and properties[name].init
                ):
                    positional_params.append(name)

        # Map positional arguments to positional parameters
        for i, arg in enumerate(args):
            if i < len(positional_params):
                prop_name = positional_params[i]
                if prop_name not in kwargs:  # Don't override explicit kwargs
                    kwargs[prop_name] = arg

        # Store original kwargs for parent initialization
        original_kwargs = kwargs.copy()

        # Build complete init arguments by combining kwargs with MyConf property defaults
        # Only include properties that have init=True
        all_init_args = {}
        for name, info in properties.items():
            if info.init:
                if name in kwargs:
                    all_init_args[name] = kwargs[name]
                elif info.value is not None and not has_fn(info):
                    all_init_args[name] = info.value

        # Initialize parent classes FIRST, before setting any MyConf properties
        for base in cls.__mro__[1:]:
            if (
                hasattr(base, "__init__")
                and base.__init__ != object.__init__
                and not hasattr(base, "_myconf_properties")
                and base.__name__ != "MyConf"
            ):
                if "ModuleList" in base.__name__ and args:
                    base.__init__(self, *args)
                    break
                else:
                    # Try to pass arguments that the parent class might need
                    import inspect

                    try:
                        base_sig = inspect.signature(base.__init__)
                        base_params = list(base_sig.parameters.keys())[
                            1:
                        ]  # Skip 'self'
                        base_args = {}
                        for param in base_params:
                            if param in all_init_args:
                                base_args[param] = all_init_args[param]

                        if base_args:
                            base.__init__(self, **base_args)
                        else:
                            base.__init__(self)
                    except:
                        base.__init__(self)
                    break

        # Now handle MyConf property initialization
        for name, info in properties.items():
            if info.init and name in kwargs:
                # If value provided in kwargs and init=True, use it
                value = kwargs.pop(name)

                # Handle Cast conversion first (whether consumed or not)
                if getattr(info, "is_cast", False):
                    value = _apply_cast_conversion(value, info)
                # Then handle regular type conversion if this is NOT an F() property override
                elif info.annotation and not has_fn(info):
                    value = _format(value, info.annotation)

                # Only set attribute if not consumed
                if not getattr(info, "is_consumed", False):
                    setattr(self, name, value)

            elif has_fn(info):
                # Only set lazy flag if no value was provided
                computed_value = info.fn(self)
                if info.annotation:
                    computed_value = _format(computed_value, info.annotation)
                setattr(self, name, computed_value)
            elif info.value is not None:
                # Set default value regardless of init parameter (for private attrs)
                # BUT skip consumed properties - they should never be stored
                if not getattr(info, "is_consumed", False):
                    value = info.value
                    # Apply Cast conversion to default values too
                    if getattr(info, "is_cast", False):
                        value = _apply_cast_conversion(value, info)
                    elif info.annotation:
                        value = _format(value, info.annotation)
                    setattr(self, name, value)
            elif getattr(info, "is_cast", False) and not getattr(
                info, "is_consumed", False
            ):
                # For Cast fields with None default, apply Cast conversion to None
                # But skip if consumed
                value = _apply_cast_conversion(None, info)
                setattr(self, name, value)

        # Set any remaining kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    cls.__init__ = __init__


def generate_init_signature(properties):
    params = []
    for name, info in properties.items():
        if not info.init:
            continue

        param = name
        if info.value is not None:
            param += f" = {repr(info.value)}"
        elif not has_fn(info):
            if info.annotation:
                param += " = None"
            else:
                param += " = None"

        params.append(param)

    if params:
        return f"self, {', '.join(params)}"
    return "self"


def generate_init_body(properties):
    lines = []

    for name, info in properties.items():
        if not info.init:
            continue

        if has_fn(info):
            if info.value is not None:
                lines.append(f"if {name} is not None:")
                lines.append(f"    self.{name} = {name}")
                lines.append("else:")
                lines.append(f"    self.{name} = self._compute_{name}()")
            else:
                lines.append(f"self.{name} = self._compute_{name}()")
        else:
            lines.append(f"self.{name} = {name}")

    if not lines:
        lines = ["pass"]

    return "\n".join(f"    {line}" for line in lines)


def generate_compute_methods(properties):
    methods = []

    for name, info in properties.items():
        if has_fn(info):
            method_code = dedent(f"""
                def _compute_{name}(self):
                    return ({info.fn.__name__})(self)
            """).strip()
            methods.append(method_code)

    return "\n\n".join(methods)


def create_init_method(cls, properties):
    from .core import has_fn

    signature = generate_init_signature(properties)
    body = generate_init_body(properties)
    compute_methods = generate_compute_methods(properties)

    init_code = f"""
def __init__({signature}):
{body}
"""

    if compute_methods:
        full_code = f"{init_code}\n{compute_methods}"
    else:
        full_code = init_code

    namespace = {"has_fn": has_fn}
    for name, info in properties.items():
        if has_fn(info):
            namespace[info.fn.__name__] = info.fn

    exec(full_code, namespace)
    return namespace["__init__"]
