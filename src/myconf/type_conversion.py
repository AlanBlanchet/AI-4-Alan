"""Type conversion for MyConf"""

from .constants import MYCONF_PROPERTIES_ATTR


def _format(value, cls):
    """Convert value to the specified class type"""
    if value is None:
        return None

    # Handle generic types (like list[Layer], dict[str, int], Batch[T], etc.)
    if hasattr(cls, "__origin__"):
        origin = cls.__origin__
        type_args = getattr(cls, "__args__", ())

        # Special handling for custom generic classes like Batch[T]
        if hasattr(origin, "collate_fn") and hasattr(origin, "__class__"):
            # This is likely a Batch[T] type - check if value is already the right type
            # Use direct class comparison instead of isinstance for Generic types
            if value.__class__ is origin or value.__class__.__name__ == origin.__name__:
                return value
            # Convert single item to batch using collate_fn
            return origin.collate_fn([value])

        if origin is list:
            if not isinstance(value, (list, tuple)):
                return value
            if type_args:
                element_type = type_args[0]
                return [_format(item, element_type) for item in value]
            else:
                return list(value)
        elif origin is dict:
            if not isinstance(value, dict):
                return value
            if len(type_args) >= 2:
                key_type, value_type = type_args[0], type_args[1]
                return {
                    _format(k, key_type): _format(v, value_type)
                    for k, v in value.items()
                }
            else:
                return dict(value)
        elif origin is tuple:
            if not isinstance(value, (list, tuple)):
                # Handle single value conversion to tuple
                if type_args:
                    if len(type_args) == 2 and type_args[0] == type_args[1]:
                        # tuple[T, T] - convert single value to (value, value)
                        converted_value = _format(value, type_args[0])
                        return (converted_value, converted_value)
                    elif len(type_args) == 1:
                        # tuple[T] - convert single value to (value,)
                        converted_value = _format(value, type_args[0])
                        return (converted_value,)
                return value
            if type_args:
                if len(type_args) == len(value):
                    return tuple(
                        _format(item, arg_type)
                        for item, arg_type in zip(value, type_args)
                    )
                elif len(type_args) == 1:
                    # tuple[T, ...] means variable length tuple of T
                    element_type = type_args[0]
                    return tuple(_format(item, element_type) for item in value)
            return tuple(value)

    # Handle Batch[T] type conversion
    if hasattr(cls, "__origin__") and hasattr(cls, "__args__"):
        if getattr(cls, "__name__", None) == "Batch" or (
            hasattr(cls, "__origin__")
            and getattr(cls.__origin__, "__name__", None) == "Batch"
        ):
            # This is a Batch[T] type
            if hasattr(cls, "__args__") and cls.__args__:
                element_type = cls.__args__[0]
                # If value is already the target type, wrap in batch
                if isinstance(value, element_type):
                    # Import Batch dynamically to avoid circular imports
                    from ai.data.batch import Batch

                    return Batch.collate_fn([value])
                # If value is already a Batch, return as-is
                elif hasattr(value, "collate_fn"):
                    return value

    if not isinstance(cls, type):
        return value

    if isinstance(value, cls):
        return value

    # Handle MyConf classes specially
    if hasattr(cls, MYCONF_PROPERTIES_ATTR):
        if isinstance(value, dict):
            return cls(**value)
        elif isinstance(value, str):
            # For string values, try to create the object with no args
            return cls()
        elif isinstance(value, (list, tuple)):
            # For sequences, intelligently map to properties based on type hints
            properties = cls._myconf_properties

            # Check if class inherits from a collection type
            collection_element_type = None
            for base in getattr(cls, "__orig_bases__", []):
                if hasattr(base, "__origin__") and base.__origin__ in (list, tuple):
                    if hasattr(base, "__args__") and base.__args__:
                        collection_element_type = base.__args__[0]
                        break

            # Prioritize class-specific properties over inherited ones
            class_annotations = getattr(cls, "__annotations__", {})
            class_specific_names = [
                k for k in class_annotations.keys() if not k.startswith("_")
            ]
            inherited_names = [
                k
                for k in properties.keys()
                if not k.startswith("_") and k not in class_specific_names
            ]

            # For collection classes, assume format: [collection_data, scalar_properties...]
            if collection_element_type is not None and len(value) >= 2:
                # First element is collection data, rest are scalar properties
                collection_data = value[0]
                scalar_values = value[1:]
                param_names = class_specific_names + inherited_names

                # Map scalar values to parameters
                kwargs = {}
                for i, param_name in enumerate(param_names):
                    if i < len(scalar_values):
                        param_info = properties[param_name]
                        param_type = param_info.annotation
                        kwargs[param_name] = _format(scalar_values[i], param_type)

                # Create instance with scalar properties
                instance = cls(**kwargs)

                # Add collection data
                if isinstance(collection_data, (list, tuple)):
                    collection_items = [
                        _format(item, collection_element_type)
                        for item in collection_data
                    ]
                    instance.extend(collection_items)

                return instance
            else:
                # Standard property mapping for non-collection classes
                param_names = class_specific_names + inherited_names

                if len(value) <= len(param_names):
                    kwargs = {}
                    for i, param_name in enumerate(param_names):
                        if i < len(value):
                            param_info = properties[param_name]
                            param_type = param_info.annotation
                            # Recursively format the value to match the parameter type
                            kwargs[param_name] = _format(value[i], param_type)

                    return cls(**kwargs)
                else:
                    return cls(*value)
        else:
            return cls()

    # Handle regular classes
    if isinstance(value, dict):
        return cls(**value)
    elif isinstance(value, (list, tuple)):
        return cls(*value)
    else:
        return cls(value)
