from abc import abstractmethod
from functools import cache
from typing import Any, ClassVar

import torch

from ..configs.base import Base
from ..configs.log import Color


class Modality(Base):
    log_name: ClassVar[str] = "modality"
    color: ClassVar[str] = Color.red

    @classmethod
    def log(cls, *msg: list[Any], table=False):
        super().log(f"[{cls.__name__}]", *msg, table=table)

    @staticmethod
    @abstractmethod
    def preprocess(data: torch.Tensor): ...

    @classmethod
    @cache
    def modalities(cls):
        return {k: sub for sub in cls.__subclasses__() for k in sub.model_fields.keys()}

    @abstractmethod
    def collate_fn(self, name: str, samples: list):
        raise NotImplementedError

    @classmethod
    def collate(cls, items: dict[str, list]):
        collated = {}
        modalities = cls.modalities()
        for name, samples in items.items():
            samples = [torch.as_tensor(s) for s in samples]
            if name in modalities:
                collated.update(modalities[name].collate_fn(name, samples))
            else:
                collated[name] = torch.stack(samples)
        return collated

    @staticmethod
    def mask_collate(name: str, samples: list):
        res = {}
        lengths = [len(s) if s.ndim != 0 else 0 for s in samples]
        max_len = max(lengths)

        elem = torch.zeros(
            (len(samples), max_len, *samples[0].shape[1:]),
            dtype=samples[0].dtype,
        )
        mask = torch.zeros((len(samples), max_len), dtype=torch.bool)
        for i, ln in enumerate(lengths):
            elem[i, :ln] = samples[i]
            mask[i, :ln] = True
        res[name] = elem
        res[f"{name}_mask"] = mask
        return res
