from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Iterable

from pydantic import Field, field_validator

from ..configs.base import Base
from ..configs.log import Color
from ..modality import Image, Label


class DatasetSplitConfig(Base):
    name: str = "train"
    size: float = 1.0


class BaseDataset(Base, buildable=False):
    log_name: ClassVar[str] = "dataset"
    color: ClassVar[str] = Color.yellow

    identification_name: ClassVar[str] = "source"

    params: dict = {}

    train: DatasetSplitConfig = Field(None, validate_default=True)
    val: DatasetSplitConfig = Field(None, validate_default=True)

    @field_validator("train", mode="before")
    @classmethod
    def validate_train(cls, v):
        if v is None:
            v = {}
        return DatasetSplitConfig(**v)

    @field_validator("val", mode="before")
    @classmethod
    def validate_val(cls, v):
        if v is None:
            v = {}
        return DatasetSplitConfig(**v)

    @property
    def name() -> str: ...

    @abstractmethod
    def get_train(self, **kwargs) -> Iterable: ...

    def get_val(self, **kwargs) -> Iterable:
        raise NotImplementedError

    def create_test(self) -> Iterable:
        raise NotImplementedError

    @cached_property
    def _labels(self) -> list[str]: ...

    def prepare(self, **kwargs: dict): ...

    @abstractmethod
    def item_from_id(self, id: Any, split: str) -> dict: ...

    @abstractmethod
    def parse_items(self, item: dict, map: dict) -> dict: ...

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
        # Make all required fields have a default value
        for required in self.root_config.task.required_fields:
            if required not in params:
                params[required] = required
        return params

    @cached_property
    def image_modality(self):
        """
        Process the image modality
        """
        image_config = self.process_fields.get("image", {})
        bbox_config = self.process_fields.get("bbox", {})
        return Image(image=image_config, bbox=bbox_config)

    def item_process(self, split: str):
        """
        Process an item from the dataset and standardize it for the task
        """

        def process(item):
            # Map dataset specific parameters to a standard
            item = self.map_params(item)

            # Process images - transform, augment, normalize...
            if "image" in self.process_fields:
                item = self.image_modality.preprocess(item, split)

            # Process labels - try getting uniques, convert to tensor, etc.
            if "labels" in self.process_fields:
                item = Label.single_process(
                    item, multiple="bbox" in self.process_fields
                )

            return item

        return process

    def map_params(self, item: dict) -> dict:
        name_mapping = {
            k: v if isinstance(v, str) else v["name"]
            for k, v in self.process_fields.items()
        }
        return self.parse_items(item, name_mapping)

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

    def example(self):
        return next(iter(self.get_val()))
