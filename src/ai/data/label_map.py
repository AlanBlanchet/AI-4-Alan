from functools import cached_property
from typing import overload

import torch

from myconf import MyConf


class LabelMap(MyConf):
    labels: list[str]
    specials: list[str] = []
    keep_order: bool = True

    @cached_property
    def _label2id(self) -> dict[str, int]:
        if self.keep_order:
            # Keep order
            return {label: i for i, label in enumerate(self.labels)}
        # Mix
        for special in self.specials:
            if special in self.labels:
                self.labels.remove(special)
        # Sort & Deduplicate
        self.labels = sorted(set(self.labels))
        # Mapping
        return {label: i for i, label in enumerate([*self.specials, *self.labels])}

    @cached_property
    def _id2label(self) -> dict[int, str]:
        return {i: label for label, i in self._label2id.items()}

    def __get(self, item):
        if isinstance(item, str):
            return self._label2id[item]
        return self._id2label[item]

    @overload
    def __getitem__(self, item: int) -> str: ...

    @overload
    def __getitem__(self, item: str) -> int: ...

    @overload
    def __getitem__(self, item: list[int]) -> list[str]: ...

    @overload
    def __getitem__(self, item: list[str]) -> list[int]: ...

    @overload
    def __getitem__(self, item: torch.Tensor) -> list[str]: ...

    @overload
    def _getitem__(self, item: slice) -> list[str]: ...

    @overload
    def _getitem__(self, item: range) -> list[str]: ...

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.__get(i) for i in item]
        elif isinstance(item, torch.Tensor):
            if item.ndim == 0:
                return self.__get(item.item())
            elif item.ndim > 1:
                raise ValueError("Cannot index with tensor of dim > 1")
            return [self.__get(i.item()) for i in item]
        elif isinstance(item, slice):
            if isinstance(item.start, str) or isinstance(item.stop, str):
                # TODO: Implement slicing by label
                raise ValueError("Cannot slice by label")
            return [
                self.__get(i)
                for i in range(item.start or 0, item.stop or len(self), item.step or 1)
            ]
        elif isinstance(item, range):
            return [self.__get(i) for i in item]
        return self.__get(item)

    def __len__(self):
        return len(self._id2label)

    def __iter__(self):
        return iter(self._id2label.items())
