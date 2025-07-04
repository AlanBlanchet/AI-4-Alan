from typing import Any, Generic, List, TypeVar, Union

from torch.utils.data import default_collate

T = TypeVar("T")


class Batch(Generic[T]):
    def __init__(self, data: Any):
        self._data = data

    @classmethod
    def collate_fn(cls, batch: List[T]) -> "Batch[T]":
        collated = default_collate(batch)
        return cls(collated)

    @property
    def data(self):
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, name):
        if hasattr(self._data, name):
            return getattr(self._data, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def to_batch(modality_or_batch: Union[T, "Batch[T]"]) -> "Batch[T]":
    if isinstance(modality_or_batch, Batch):
        return modality_or_batch
    return Batch.collate_fn([modality_or_batch])
