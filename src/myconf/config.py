"""
Configuration metaclass for MyConf that handles class creation and property setup.
"""

# Import base metaclass
from .base import MetaMyConf
from .init_generation import generate_init
from .property_processing import setup_myconf_properties, update_class_signatures


class MetaConfig(MetaMyConf):
    """Metaclass for MyConf that sets up properties and generates __init__"""

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class first using parent metaclass
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip setup for the base MyConf class itself
        if name == "MyConf" and not bases:
            return cls

        # Set up MyConf properties from annotations and defaults
        setup_myconf_properties(cls)

        # Generate __init__ method with proper signature
        generate_init(cls)

        # Update class signatures for better IDE support
        update_class_signatures(cls)

        # Post-process ClassVar annotations for type conversion
        _process_classvar_type_conversion(cls)

        # Filter private attributes from __annotations__ for IDE support
        _filter_private_annotations(cls)

        return cls


def _process_classvar_type_conversion(cls):
    """Process ClassVar annotations for automatic type conversion"""
    from .type_conversion import _format
    from .utils import is_class_var

    annots = getattr(cls, "__annotations__", {})

    for prop_name, annotation in annots.items():
        if is_class_var(annotation) and hasattr(cls, prop_name):
            if hasattr(annotation, "__args__") and annotation.__args__:
                inner_type = annotation.__args__[0]
                current_value = getattr(cls, prop_name)

                should_convert = False
                if hasattr(inner_type, "__origin__"):
                    origin_type = inner_type.__origin__
                    if not isinstance(current_value, origin_type):
                        should_convert = True
                    else:
                        # Check if the elements need conversion
                        type_args = getattr(inner_type, "__args__", ())
                        if (
                            origin_type is list
                            and type_args
                            and isinstance(current_value, list)
                        ):
                            element_type = type_args[0]
                            # Check if any element needs conversion
                            for item in current_value:
                                if not isinstance(item, element_type):
                                    should_convert = True
                                    break
                else:
                    if not isinstance(current_value, inner_type):
                        should_convert = True

                if should_convert:
                    converted_value = _format(current_value, inner_type)
                    setattr(cls, prop_name, converted_value)


def _filter_private_annotations(cls):
    """Filter private attributes from __annotations__ for IDE signature support"""
    if hasattr(cls, "__annotations__"):
        # Create a filtered version that excludes private attributes
        filtered_annotations = {
            name: annotation
            for name, annotation in cls.__annotations__.items()
            if not name.startswith("_")
        }
        cls.__annotations__ = filtered_annotations


def resolve_metaclass(*bases):
    """
    Resolve metaclass conflicts when multiple bases have different metaclasses.
    Create a new metaclass that properly inherits from all base metaclasses.
    """
    from .base import MetaMyConf

    metaclasses = []
    for base in bases:
        meta = type(base)
        if meta not in metaclasses and meta is not type:
            metaclasses.append(meta)

    if not metaclasses:
        return MetaMyConf
    elif len(metaclasses) == 1:
        # If only one metaclass, enhance it with Cast conversion support
        if metaclasses[0] is MetaMyConf:
            return MetaMyConf
        else:
            # Create a hybrid metaclass
            class ResolvedMeta(metaclasses[0], MetaMyConf):
                def __call__(cls, *args, **kwargs):
                    # Map positional arguments to property names first
                    consumed_positional_args = []
                    if args and hasattr(cls, "_myconf_properties"):
                        properties = cls._myconf_properties

                        # Check if class has a custom __new__ method and extract its parameter order
                        positional_params = []

                        # Look for custom __new__ method in the MRO
                        custom_new_method = None
                        for base_cls in cls.__mro__:
                            if (
                                hasattr(base_cls, "__new__")
                                and base_cls.__new__ != object.__new__
                                and getattr(base_cls.__new__, "__self__", None)
                                is not object
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
                                if (
                                    name not in ordered_prop_names
                                    and not name.startswith("_")
                                ):
                                    ordered_prop_names.append(name)

                            for name in ordered_prop_names:
                                if (
                                    name in properties
                                    and not name.startswith("_")
                                    and properties[name].init
                                ):
                                    positional_params.append(name)

                        # Map positional args to property names
                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if (
                                    prop_name not in kwargs
                                ):  # Don't override explicit kwargs
                                    kwargs[prop_name] = arg
                                    # Track consumed positional args for __new__
                                    if getattr(
                                        properties.get(prop_name), "is_consumed", False
                                    ):
                                        consumed_positional_args.append(arg)

                    # Apply Cast conversions before calling __new__
                    if hasattr(cls, "_myconf_properties"):
                        from .init_generation import _apply_cast_conversion

                        for name, info in cls._myconf_properties.items():
                            if name in kwargs and getattr(info, "is_cast", False):
                                kwargs[name] = _apply_cast_conversion(
                                    kwargs[name], info
                                )

                    # Pass consumed positional args to __new__ if the class has a custom __new__
                    if consumed_positional_args and custom_new_method:
                        # Remove consumed positional args from kwargs to avoid duplicate values
                        consumed_param_names = []
                        converted_positional_args = []

                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if getattr(
                                    properties.get(prop_name), "is_consumed", False
                                ):
                                    consumed_param_names.append(prop_name)

                                    # Apply Cast conversion to consumed positional args
                                    if hasattr(cls, "_myconf_properties"):
                                        info = cls._myconf_properties.get(prop_name)
                                        if info and getattr(info, "is_cast", False):
                                            converted_arg = _apply_cast_conversion(
                                                arg, info
                                            )
                                            converted_positional_args.append(
                                                converted_arg
                                            )
                                        else:
                                            converted_positional_args.append(arg)
                                    else:
                                        converted_positional_args.append(arg)

                        # Remove consumed parameters from kwargs
                        for param_name in consumed_param_names:
                            kwargs.pop(param_name, None)

                        return super().__call__(*converted_positional_args, **kwargs)
                    else:
                        return super().__call__(**kwargs)

            return ResolvedMeta
    else:
        # Multiple metaclasses - create a new one that inherits from all
        # Check if MetaMyConf is already in the list
        if MetaMyConf in metaclasses:

            class ResolvedMeta(*metaclasses):
                def __call__(cls, *args, **kwargs):
                    # Map positional arguments to property names first
                    consumed_positional_args = []
                    if args and hasattr(cls, "_myconf_properties"):
                        properties = cls._myconf_properties

                        # Check if class has a custom __new__ method and extract its parameter order
                        positional_params = []

                        # Look for custom __new__ method in the MRO
                        custom_new_method = None
                        for base_cls in cls.__mro__:
                            if (
                                hasattr(base_cls, "__new__")
                                and base_cls.__new__ != object.__new__
                                and getattr(base_cls.__new__, "__self__", None)
                                is not object
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
                                if (
                                    name not in ordered_prop_names
                                    and not name.startswith("_")
                                ):
                                    ordered_prop_names.append(name)

                            for name in ordered_prop_names:
                                if (
                                    name in properties
                                    and not name.startswith("_")
                                    and properties[name].init
                                ):
                                    positional_params.append(name)

                        # Map positional args to property names
                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if (
                                    prop_name not in kwargs
                                ):  # Don't override explicit kwargs
                                    kwargs[prop_name] = arg
                                    # Track consumed positional args for __new__
                                    if getattr(
                                        properties.get(prop_name), "is_consumed", False
                                    ):
                                        consumed_positional_args.append(arg)

                    # Apply Cast conversions before calling __new__
                    if hasattr(cls, "_myconf_properties"):
                        from .init_generation import _apply_cast_conversion

                        for name, info in cls._myconf_properties.items():
                            if name in kwargs and getattr(info, "is_cast", False):
                                kwargs[name] = _apply_cast_conversion(
                                    kwargs[name], info
                                )

                    # Pass consumed positional args to __new__ if the class has a custom __new__
                    if consumed_positional_args and custom_new_method:
                        # Remove consumed positional args from kwargs to avoid duplicate values
                        consumed_param_names = []
                        converted_positional_args = []

                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if getattr(
                                    properties.get(prop_name), "is_consumed", False
                                ):
                                    consumed_param_names.append(prop_name)

                                    # Apply Cast conversion to consumed positional args
                                    if hasattr(cls, "_myconf_properties"):
                                        info = cls._myconf_properties.get(prop_name)
                                        if info and getattr(info, "is_cast", False):
                                            converted_arg = _apply_cast_conversion(
                                                arg, info
                                            )
                                            converted_positional_args.append(
                                                converted_arg
                                            )
                                        else:
                                            converted_positional_args.append(arg)
                                    else:
                                        converted_positional_args.append(arg)

                        # Remove consumed parameters from kwargs
                        for param_name in consumed_param_names:
                            kwargs.pop(param_name, None)

                        return super().__call__(*converted_positional_args, **kwargs)
                    else:
                        return super().__call__(**kwargs)
        else:

            class ResolvedMeta(*metaclasses, MetaMyConf):
                def __call__(cls, *args, **kwargs):
                    # Map positional arguments to property names first
                    consumed_positional_args = []
                    if args and hasattr(cls, "_myconf_properties"):
                        properties = cls._myconf_properties

                        # Check if class has a custom __new__ method and extract its parameter order
                        positional_params = []

                        # Look for custom __new__ method in the MRO
                        custom_new_method = None
                        for base_cls in cls.__mro__:
                            if (
                                hasattr(base_cls, "__new__")
                                and base_cls.__new__ != object.__new__
                                and getattr(base_cls.__new__, "__self__", None)
                                is not object
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
                                if (
                                    name not in ordered_prop_names
                                    and not name.startswith("_")
                                ):
                                    ordered_prop_names.append(name)

                            for name in ordered_prop_names:
                                if (
                                    name in properties
                                    and not name.startswith("_")
                                    and properties[name].init
                                ):
                                    positional_params.append(name)

                        # Map positional args to property names
                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if (
                                    prop_name not in kwargs
                                ):  # Don't override explicit kwargs
                                    kwargs[prop_name] = arg
                                    # Track consumed positional args for __new__
                                    if getattr(
                                        properties.get(prop_name), "is_consumed", False
                                    ):
                                        consumed_positional_args.append(arg)

                    # Apply Cast conversions before calling __new__
                    if hasattr(cls, "_myconf_properties"):
                        from .init_generation import _apply_cast_conversion

                        for name, info in cls._myconf_properties.items():
                            if name in kwargs and getattr(info, "is_cast", False):
                                kwargs[name] = _apply_cast_conversion(
                                    kwargs[name], info
                                )

                    # Pass consumed positional args to __new__ if the class has a custom __new__
                    if consumed_positional_args and custom_new_method:
                        # Remove consumed positional args from kwargs to avoid duplicate values
                        consumed_param_names = []
                        converted_positional_args = []

                        for i, arg in enumerate(args):
                            if i < len(positional_params):
                                prop_name = positional_params[i]
                                if getattr(
                                    properties.get(prop_name), "is_consumed", False
                                ):
                                    consumed_param_names.append(prop_name)

                                    # Apply Cast conversion to consumed positional args
                                    if hasattr(cls, "_myconf_properties"):
                                        info = cls._myconf_properties.get(prop_name)
                                        if info and getattr(info, "is_cast", False):
                                            converted_arg = _apply_cast_conversion(
                                                arg, info
                                            )
                                            converted_positional_args.append(
                                                converted_arg
                                            )
                                        else:
                                            converted_positional_args.append(arg)
                                    else:
                                        converted_positional_args.append(arg)

                        # Remove consumed parameters from kwargs
                        for param_name in consumed_param_names:
                            kwargs.pop(param_name, None)

                        return super().__call__(*converted_positional_args, **kwargs)
                    else:
                        return super().__call__(**kwargs)

        return ResolvedMeta


# Import base MyConf and create the actual MyConf class
from .base import MyConf as BaseMyConf


class MyConf(BaseMyConf, metaclass=MetaConfig):
    """Main MyConf class with proper metaclass setup"""

    pass
