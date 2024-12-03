from __future__ import annotations

from typing import Callable, Generator

import numpy as np
import torch
from pydantic import BaseModel, field_validator
from random_word import RandomWords


def random_run_name(num_words=3):
    generator = RandomWords()
    return "-".join([generator.get_random_word().lower() for _ in range(num_words)])


def get_epsilon_exponential_decay_fn(
    eps_max: float,
    eps_min: float,
    decay: float,
) -> Callable:
    """
    Returns function epsilon_fn, which depends on
    a single input, step, which is the current episode
    """

    def epsilon_fn(episode: int) -> float:
        return max(eps_min, eps_max * (decay**episode))

    return epsilon_fn


def batch_it(
    *elements: list[torch.Tensor], size=32
) -> Generator[tuple[torch.Tensor, ...], None, None]:
    for i in range(0, len(elements[0]), size):
        yield tuple(e[i : i + size] for e in elements)


def parse_tensor(data, device: torch.device = "cpu", dtype=torch.float32):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        data = data.clone().detach()
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], np.ndarray):
            data = torch.stack([torch.from_numpy(d) for d in data])
        elif isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
        else:
            return torch.tensor(data, dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)

    return data.to(dtype).to(device)


def keep_kwargs_prefixed(kwargs: dict, prefix: str) -> dict:
    x = {k.removeprefix(prefix): v for k, v in kwargs.items() if k.startswith(prefix)}
    return x


class TensorInfo(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, v):
        if isinstance(v, (int, np.integer)):
            return (int(v),)
        return tuple(v)


class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)
