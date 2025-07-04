from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, Callable, ClassVar, Iterable, Self

import torch

from ..configs.base import Base
from ..configs.log import Color
from ..modality.modality import Modality
from ..modality.preprocess import Preprocess


class DatasetSplitConfig(Base):
    name: str = "train"
    size: float = 1.0


class Data(Base):
    log_name: ClassVar[str] = "data"
    color: ClassVar[str] = Color.yellow

    identification_name: ClassVar[str] = "source"

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            dataset: Iterable,
            item_process: Callable,
        ):
            self.dataset = dataset
            self.item_process = item_process

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            for item in self.dataset:
                yield self.item_process(item)

        def __getitem__(self, idx):
            return self.item_process(self.dataset[idx])

    @property
    def process_fields(self):
        params = self.map_params_config.copy()
        input_val = params.pop("input")
        # Create missing input value
        if isinstance(input_val, str):
            if input_val not in params:
                params[input_val] = input_val
        elif isinstance(input_val, dict):
            name = input_val["name"]
            if name not in params:
                params[name] = name
        elif isinstance(input_val, list):
            raise NotImplementedError("Multiple inputs not supported yet")
        return params

    def item_process(self):
        def _process(item):
            # Get wanted fields
            fields = ...
            ...
            # Convert to wanted tensor format
            ...
            # Apply augmentations
            ...
            self.parse_items()
            return item

        return _process

    @property
    def name() -> str: ...

    @abstractmethod
    def item_from_id(self, id: Any, split: str) -> dict: ...

    def prepare(self, **kwargs: dict): ...

    def extract_inputs(self, item: dict) -> dict:
        """
        Extracts the input required for the model
        """
        inputs = []
        input_conf = self.config.dataset.input
        if isinstance(input_conf, str):
            input_conf = dict(name=input_conf)
        elif isinstance(input_conf, list):
            raise NotImplementedError("Multiple inputs not supported yet")
        inputs.append(item[input_conf["name"]])
        other = input_conf.get("other", [])
        if isinstance(other, str):
            other = [other]
        for name in other:
            inputs.append(item[name])
        return inputs


class Dataset(Data):
    train: DatasetSplitConfig = "train"
    val: DatasetSplitConfig = "val"

    @abstractmethod
    def get_train(self, **kwargs) -> Iterable: ...

    @abstractmethod
    def get_val(self, **kwargs) -> Iterable: ...

    @cached_property
    def example(self):
        return next(iter(self.get_val()))


class DataList(Base, list[Data]):
    datasets: list[Data]

    @property
    def names(self):
        return [dataset.name for dataset in self.datasets]

    @cached_property
    def example(self):
        return self.datasets[0].example

    @cached_property
    def examples(self):
        return [dataset.example for dataset in self.datasets]

    def __getitem__(self, name_or_idx: str | int) -> Data:
        if isinstance(name_or_idx, str):
            for dataset in self.datasets:
                if dataset.name == name_or_idx:
                    return dataset
            raise KeyError(f"Dataset with name {name_or_idx} not found")
        return self.datasets[name_or_idx]

    def prepare_for_model(self, batch: dict):
        """
        Prepare the batch for the model
        """
        # TODO add support for multiple datasets
        for dataset in self.datasets:
            prepared = {}
            for k, v in dataset.expose.items():
                prepared[v] = batch[k]
            return prepared

    def add_preprocess(
        self, preprocess: Preprocess | list[Preprocess], modality: type[Modality]
    ):
        if not isinstance(preprocess, list):
            preprocess = [preprocess]
        for dataset in self.datasets:
            mod = dataset.modalities.get(modality)
            mod.preprocesses.extend(preprocess)

    def __iter__(self):
        return iter(self.datasets)

    def prepare(self):
        for dataset in self.datasets:
            dataset.prepare()

    @classmethod
    def resolve(cls, data: str | Data | Self) -> Self:
        if isinstance(data, cls):
            return data
        elif isinstance(data, str):
            dataset = Data.resolve(data)
            return cls(datasets=[dataset])
        elif isinstance(data, Data):
            return cls(datasets=[data])
        raise ValueError(f"Cannot resolve {type(data)} to DataList")
