from inspect import signature

from pydantic import field_validator


def validator(field: str):
    """Decorator to create a validator for a field in the 'before' mode

    Don't be afraid to return a config dict if the validator is on the Base class.
    If the validator returns a dict, it will automatically be converted to the corresponding class.
    """

    def wrapper(f):
        """Wrapper taking the function to call as argument"""
        sign = signature(f)
        n = len(sign.parameters)

        @field_validator(field, mode="before")
        def _field_validator(cls, v, values):
            others = values if isinstance(values, dict) else values.data
            args = (cls, v, others)
            return f(*args[:n])

        return _field_validator

    return wrapper
