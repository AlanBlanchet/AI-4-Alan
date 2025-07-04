from functools import cache
from typing import Any, ClassVar, Optional, final

import torch
import torch.nested as nested

from ..configs.log import Color
from ..utils.tensor import TensorBase
from ..utils.types import CallableList, is_dict, is_number


class Modality(TensorBase):
    log_name: ClassVar[str] = "modality"
    color: ClassVar[str] = Color.red

    @classmethod
    def modality_format(
        cls, tensor: torch.Tensor, kwargs: dict[str, Any]
    ) -> torch.Tensor:
        return tensor

    @final
    @classmethod
    def tensor_format(
        cls, tensor: torch.Tensor, kwargs: dict[str, Any]
    ) -> torch.Tensor:
        return cls.modality_format(tensor, kwargs)

    @classmethod
    @cache
    def modalities(cls):
        return {k: sub for sub in cls.__subclasses__() for k in sub.model_fields.keys()}

    def item_accept(self, key: str, value: Any):
        """Method responsible for specifying if the modality accepts the item"""
        return len(self.input) == 0 or key in self.input

    def _gather_accepted(self, items: dict[str, list]):
        """Gathers only the accepted items"""
        return {k: v for k, v in items.items() if self.item_accept(k, v)}

    @classmethod
    def collate(cls, batch: list):
        collated = {}
        names = batch[0].keys()
        transposed = zip(*[b.values() for b in batch])
        items = {name: samples for name, samples in zip(names, transposed)}

        for name, samples in items.items():
            ex = samples[0]

            if is_number(ex):
                collated[name] = torch.as_tensor(samples)
            elif is_dict(ex):
                collated[name] = cls.collate(samples)
            else:
                collated[name] = nested.nested_tensor(samples)

        return collated


class NormalizedModality(Modality):
    def int(self) -> Modality:
        """Convert to int dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().int())

    def float(self) -> Modality:
        """Convert to float dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().float())

    def long(self) -> Modality:
        """Convert to long dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().long())

    def bool(self) -> Modality:
        """Convert to bool dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().bool())

    def short(self) -> Modality:
        """Convert to short dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().short())

    def byte(self) -> Modality:
        """Convert to byte dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().byte())

    def char(self) -> Modality:
        """Convert to char dtype - returns Modality since dtype change loses Image specialization"""
        return Modality._make_subclass_efficient(super().char())

    def to(
        self,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Modality:
        return Modality._make_subclass_efficient(
            super().to(dtype, non_blocking, copy, memory_format=memory_format)
        )

    def squeeze(self, dim: Optional[int] = None) -> Modality:
        """Squeeze tensor - returns Modality since shape change may lose Image semantic meaning"""
        return Modality._make_subclass_efficient(super().squeeze(dim=dim))

    def unsqueeze(self, dim: int) -> Modality:
        """Unsqueeze tensor - returns Modality since shape change may lose Image semantic meaning"""
        return Modality._make_subclass_efficient(super().unsqueeze(dim))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Modality:
        return Modality._make_subclass_efficient(
            torch.Tensor.flatten(self, start_dim, end_dim)
        )


class Modalities(CallableList[Modality]):
    def __call__(self, data):
        for modality in self:
            data = modality(data)
        return data

    def get(self, modality: type[Modality]):
        """Get a modality by type and create it if it doesn't exist"""
        for m in self:
            if isinstance(m, modality):
                return m
        m = modality()
        self.append(m)
        return m
