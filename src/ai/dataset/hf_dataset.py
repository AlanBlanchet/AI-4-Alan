from functools import cached_property
from typing import ClassVar

from datasets import Dataset, load_dataset
from pydantic import computed_field

from ..utils.label_map import HFLabelMap, LabelMap
from .base_dataset import DetBaseDataset
from .convert import HuggingFaceTorchDataset


class HuggingFaceDataset(DetBaseDataset):
    source: ClassVar[str] = "huggingface"

    @computed_field(repr=False)
    @cached_property
    def _dataset(self) -> Dataset:
        return load_dataset(self.name)

    @computed_field(repr=False)
    @cached_property
    def label_map(self) -> LabelMap:
        categories = self._dataset["train"].features["objects"].feature["category"]
        return HFLabelMap(
            labels=categories.names, hf_classes=categories, specials=["background"]
        )

    def train(self):
        train_ds = self._dataset["train"]

        return HuggingFaceTorchDataset(
            train_ds, background_id=self.label_map["background"]
        )

    def val(self):
        val_ds = self._dataset["val"]

        return HuggingFaceTorchDataset(
            val_ds, background_id=self.label_map["background"]
        )

    class Config:
        arbitrary_types_allowed = True
