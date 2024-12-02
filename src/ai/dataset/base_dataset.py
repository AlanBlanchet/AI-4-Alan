from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

from ..configs import Color
from ..configs.base import Base
from ..configs.main import MainConfig
from ..modality import Image, Label

if TYPE_CHECKING:
    from ..task.classification.label_map import LabelMap


class BaseDataset(Base):
    log_name: ClassVar[str] = "dataset"
    color: ClassVar[str] = Color.yellow

    config: MainConfig

    @property
    def ds_config(self):
        return self.config.dataset

    @property
    def name() -> str: ...

    @abstractmethod
    def train(self) -> Iterable: ...

    def val(self) -> Iterable:
        raise NotImplementedError

    def test(self) -> Iterable:
        raise NotImplementedError

    @cached_property
    def _labels(self) -> list[str]: ...

    def prepare(self, label_map: LabelMap = None): ...

    @abstractmethod
    def item_from_id(self, id: Any, split: str) -> dict: ...

    @abstractmethod
    def parse_items(self, item: dict, map: dict) -> dict: ...

    @property
    def process_fields(self):
        # self.config.dataset.model_extra.keys()
        params = self.config.dataset.map_params.copy()
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
        for required in self.config.task.required_fields:
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
        return next(iter(self.val()))
