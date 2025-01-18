from inspect import signature

from pydantic import field_validator


def validator(field: str):
    def wrapper(f):
        """Wrapper taking the function to call as argument"""
        sign = signature(f)
        n = len(sign.parameters)

        @field_validator(field, mode="before")
        def _field_validator(cls, v, values):
            args = (cls, v, values.data)
            return f(*args[:n])

        return _field_validator

    return wrapper
