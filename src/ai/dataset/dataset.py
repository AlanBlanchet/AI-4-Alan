from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Iterable, Literal, Optional, Self

from pydantic import Field

from ..configs.base import Base
from ..configs.log import Color
from ..modality.modality import Modalities, Modality
from ..modality.preprocess import Preprocess
from ..utils.env import AIEnv
from ..utils.pydantic_ import validator
from ..utils.types import is_float, is_int, is_list


class DatasetSplitConfig(Base):
    auto_build = True

    name: str = "train"
    size: float = 1.0


class InputInfo(Base):
    name: str
    dtype: Optional[Literal["int", "float", "str", "list"]] = None

    def __call__(self, value: Any) -> Any:
        if self.dtype is None:
            return value
        elif self.dtype == "int":
            return int(value)
        elif self.dtype == "float":
            return float(value)
        elif self.dtype == "str":
            return str(value)
        elif self.dtype == "list":
            return list(value)
        else:
            raise ValueError(f"Invalid dtype {self.dtype}")


class Dataset(Base, buildable=False):
    log_name: ClassVar[str] = "dataset"
    color: ClassVar[str] = Color.yellow

    identification_name: ClassVar[str] = "source"

    inputs: dict[str, InputInfo] = Field(None, validate_default=True)
    """The inputs the dataset will receive. InputInfo will be calculated and populated after seeing examples if not manually set"""

    expose: dict[str, str] = Field(None, validate_default=True)
    modalities: Modalities = Field(Modalities([]), validate_default=True)
    train: DatasetSplitConfig
    val: DatasetSplitConfig

    @validator("inputs")
    def validate_inputs(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Inputs must be a dictionary")
        return {
            k: InputInfo(**vv) if isinstance(vv, dict) else InputInfo(name=vv)
            for k, vv in v.items()
        }

    @validator("expose")
    def validate_expose(cls, v, others):
        if v is None:
            v = {val: val for val in others["inputs"].values()}
        elif isinstance(v, (list, tuple)):
            v = {val: val for val in v}
        elif not isinstance(v, dict):
            v = {v: v}
        return v

    @cached_property
    def _labels(self) -> list[str]: ...

    # @cached_property
    # def image_modality(self):
    #     """
    #     Process the image modality
    #     """
    #     image_config = self.process_fields.get("image", {})
    #     bbox_config = self.process_fields.get("bbox", {})
    #     return Image(image=image_config, bbox=bbox_config)

    @property
    def map_params_config(self):
        _map = {"input": self.input, "id": "id"}
        for k, v in self.model_extra.items():
            if isinstance(v, str):
                _map.update({k: v})
            elif isinstance(v, dict):
                # Can also be a dict with name (support for transforms or other linked to the type)
                if "name" not in v:
                    v["name"] = k  # Default
                _map.update({k: v})
        return _map

    @property
    def process_fields(self):
        # self.config.dataset.model_extra.keys()
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
        # # Make all required fields have a default value
        # for required in self.required_fields:
        #     if required not in params:
        #         params[required] = required
        return params

    @property
    def name() -> str: ...

    @validator("modalities")
    def validate_modalities(cls, v: list[dict]):
        modalities = []
        for modality in v:
            modality = Modality.from_config(modality)
            modalities.append(modality)
        return Modalities(modalities)

    @abstractmethod
    def get_train(self, **kwargs) -> Iterable: ...

    @abstractmethod
    def item_from_id(self, id: Any, split: str) -> dict: ...

    @abstractmethod
    def parse_items(self, item: dict, map: dict[str, InputInfo]) -> dict: ...

    def prepare_model_inputs(self, batch: dict) -> dict:
        """
        Prepare the model inputs
        """
        return {k: v for k, v in batch.items() if k in self.expose}

    def get_val(self, **kwargs) -> Iterable:
        raise NotImplementedError

    def create_test(self) -> Iterable:
        raise NotImplementedError

    def prepare(self, **kwargs: dict): ...

    def determine_modality(self, example: dict):
        """
        Determine the modality of the dataset
        """
        modality = []
        if "image" in self.process_fields:
            modality.append("image")
        if "labels" in self.process_fields:
            modality.append("labels")
        return modality

    def item_process(self, split: str):
        """
        Process an item from the dataset and standardize it for the task
        """

        def process(item):
            # Map dataset specific parameters to a standard
            item = self.parse_items(item, self.inputs)

            # Apply correct types from field info
            for k, v in self.inputs.items():
                if v.dtype is None:
                    # Apply correct dtype
                    item_v = item[k]
                    if isinstance(item_v, str):
                        v.dtype = "str"
                    elif is_list(item_v):
                        v.dtype = "list"
                    elif is_int(item_v):
                        v.dtype = "int"
                    elif is_float(item_v):
                        v.dtype = "float"

                item[k] = v(item[k])

            item = self.modalities(item)

            # # Process images - transform, augment, normalize...
            # if "image" in self.process_fields:
            #     item = self.image_modality.preprocess(item, split)

            # # Process labels - try getting uniques, convert to tensor, etc.
            # if "labels" in self.process_fields:
            #     item = Label.single_process(
            #         item, multiple="bbox" in self.process_fields
            #     )

            return item

        return process

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

    # @property
    @cached_property
    def example(self):
        return next(iter(self.get_val()))

    @classmethod
    def from_config(
        cls, config: dict | str | Any | list, root: dict | str | Any = {}
    ) -> Self:
        if isinstance(config, str):
            # Can resolve paths
            config = AIEnv.load(AIEnv.configs_p / config)
        elif isinstance(config, dict) and "file" in config:
            file = config.pop("file")
            config = {**AIEnv.load(AIEnv.configs_p / file), **config}

        return super().from_config(config, root)


class Datasets(Base):
    datasets: list[Dataset]

    @property
    def names(self):
        return [dataset.name for dataset in self.datasets]

    @cached_property
    def example(self):
        return self.datasets[0].example

    def __getitem__(self, name_or_idx: str | int) -> Dataset:
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
