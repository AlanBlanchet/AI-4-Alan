from abc import abstractmethod
from functools import cached_property
from typing import ClassVar

from pydantic import BaseModel
from torch.utils.data import Dataset

from ..utils.label_map import LabelMap


class BaseTorchDataset(Dataset): ...


class BaseDataset(BaseModel):
    source: ClassVar[str]
    name: str

    @abstractmethod
    def train(self) -> BaseTorchDataset: ...

    def val(self) -> BaseTorchDataset:
        raise NotImplementedError

    def test(self) -> BaseTorchDataset:
        raise NotImplementedError


class DetBaseDataset(BaseDataset):
    @cached_property
    def label_map(self) -> LabelMap: ...
