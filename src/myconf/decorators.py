import inspect
from functools import wraps
from typing import get_args, get_origin, get_type_hints

from .core import Cast
from .type_conversion import _format


def auto_convert(func):
    """Decorator that automatically converts method arguments based on Cast type hints"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Bind arguments to parameters
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Convert arguments based on Cast type hints
        for param_name, value in bound_args.arguments.items():
            if param_name in type_hints and param_name != "self":
                type_hint = type_hints[param_name]

                # Check if this is a Cast type
                origin = get_origin(type_hint)
                if origin is Cast:
                    args_tuple = get_args(type_hint)
                    if len(args_tuple) >= 2:
                        input_type, output_type = args_tuple[0], args_tuple[1]

                        # Convert the value to the output type
                        if value is not None:
                            converted_value = _format(value, output_type)
                            bound_args.arguments[param_name] = converted_value

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
