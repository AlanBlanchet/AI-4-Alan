from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


def parse_hyperparam(v: HYPERPARAM, **kwargs):
    if isinstance(v, Hyperparam):
        for k, kv in kwargs.items():
            setattr(v, k, kv)
        return v
    elif isinstance(v, (float, int)):
        return Hyperparam(start=v, **kwargs)
    elif isinstance(v, dict):
        return Hyperparam(**v, **kwargs)
    elif isinstance(v, str):
        return Hyperparam(**json.loads(v), **kwargs)
    else:
        raise TypeError(f"Cannot convert {v} to Hyperparam")


class Hyperparam(BaseModel):
    start: float
    end: int | float = 0
    decay: float = 0.995
    steps: int = 5000

    current: float = Field(None, validate_default=True)
    real_value: float = Field(None, validate_default=True)
    _train: bool = True

    @field_validator("current", mode="before")
    def validate_current(cls, v, values):
        return float(v or values.data["start"])

    @field_validator("real_value", mode="before")
    def validate_real_value(cls, v, values):
        return float(v or values.data["start"])

    def step(self, step=None):
        if step is None:
            self.current **= 2 - self.decay
        else:
            step = max(1, min(step, self.steps))

            self.current = self.start * (self.end / self.start) ** (step / self.steps)
        if self._train:
            self.real_value = self.current
        return self.current

    def __call__(self, step=None):
        if not self._train:
            return 0

        if step is not None:
            self.step(step)

        return self.real_value

    def __lt__(self, other):
        return self.real_value < other

    def __rlt__(self, other):
        return other < self.real_value

    def __le__(self, other):
        return self.real_value <= other

    def __rle__(self, other):
        return other <= self.current

    def __ge__(self, other):
        return self.real_value >= other

    def __rge__(self, other):
        return other >= self.real_value

    def __gt__(self, other):
        return self.real_value > other

    def __rgt__(self, other):
        return other > self.real_value

    def __eq__(self, other):
        return self.real_value == other

    def __req__(self, other):
        return other == self.real_value

    def eval_(self):
        self._train = False
        self.real_value = 0

    def train_(self):
        self._train = True
        self.real_value = self.current

    def __float__(self):
        return float(self.real_value)


HYPERPARAM = int | float | Hyperparam | dict | str
