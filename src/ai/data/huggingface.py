from functools import cache, cached_property
from typing import Any, Type

from datasets import (
    ClassLabel,
    Features,
    Sequence,
    load_dataset,
)
from datasets import (
    Dataset as HFDataset,
)

from ..utils.env import AIEnv
from .dataset import Dataset, DatasetSplitConfig


class HuggingFaceDataset(Dataset):
    path: str
    params: dict = {}

    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {"hf"}

    @cached_property
    def name(self) -> str:
        return self.path

    @cached_property
    def _hf_dataset(self) -> HFDataset:
        path = self.path
        self.log_msg(f"Loading dataset {path}")
        ds = load_dataset(**self.params, path=path, trust_remote_code=True)
        self.log_msg(f"Available splits: {list(ds.keys())}")
        return ds

    @cached_property
    def _hf_features(self) -> Features:
        return self._hf_dataset["train"].features

    @cached_property
    def _hf_class_label(self) -> tuple[str, ClassLabel]:
        return self._hf_find_field(self._hf_features, field_type=ClassLabel)

    def _hf_find_field(
        self,
        feature: Features | dict,
        field_name: str = None,
        field_type: Type = None,
    ):
        """
        Finds a hugging face field from a hf Features object or dict
        """
        for k, feat in feature.items():
            if field_name:
                if k == field_name:
                    return k, feature[feat]
            if field_type:
                if isinstance(feat, field_type):
                    return k, feat

            if isinstance(feat, (Features, Sequence)):
                if isinstance(feat, Sequence):
                    feat = feat.feature

                new_field_name = field_name
                if new_field_name:
                    new_field_name = ".".join(field_name.split(".")[1:])
                try:
                    p, field = self._hf_find_field(feat, new_field_name, field_type)
                    return f"{k}.{p}", field
                except Exception as _:
                    ...

        if field_name:
            raise ValueError(f"Field {field_name} not found in dataset")
        if field_type:
            raise ValueError(f"Field of type {field_type} not found in dataset")
        raise ValueError("No field or field_type provided")

    def split_info(self, name: str):
        first_k = list(self._hf_dataset.keys())[0]
        return self._hf_dataset[first_k].info.splits[name]

    def resolve_ids(self, dataset: HFDataset, split: str):
        """
        Tries to get a valid field as id or creates a custom id
        """
        for name in dataset.column_names:
            if "id" in name and "int" in dataset.features[name].dtype:
                # Use this column as id
                dataset = dataset.rename_column(name, "id")
                self.log_msg(f"Using '{name}' as id column for '{split}' split")
                break
        else:
            self.log_msg(
                f"No id column found in '{split}' split. Creating custom ids for dataset"
            )
            # Add a custom id column
            dataset = dataset.map(
                lambda ex, idx: {**ex, "id": idx},
                with_indices=True,
                num_proc=AIEnv.DEFAULT_NUM_PROC,
                batched=True,
            )
        return dataset

    def resolve_split(self, config: DatasetSplitConfig, split: str):
        """
        Formats the split according to the config
        """
        dataset = self._hf_dataset[config.name]
        num = self.split_info(config.name).num_examples
        skip_num = int(num * (1 - config.size))
        num -= skip_num
        if skip_num > 0:
            self.log_msg(f"Skipping {skip_num} examples in '{config.name}' split")
            dataset = dataset.skip(skip_num)
        dataset = self.resolve_ids(dataset, split)

        return self.TorchDataset(
            dataset=dataset.with_format("np"),
            item_process=self.item_process(),
        )

    @cache
    def get_train(self):
        return self.resolve_split(self.train, "train")

    @cache
    def get_val(self):
        return self.resolve_split(self.val, "val")

    def _parse_item(self, item, key):
        key = key.split(".")
        elem = item[key[0]]
        if len(key) > 1:
            return self._parse_item(elem, ".".join(key[1:]))
        return elem.copy()

    def parse_items(self, item, map: dict[str, Any]):
        items = {}
        for k, v in map.items():
            try:
                items[k] = self._parse_item(item, v.name)
            except KeyError as e:
                raise KeyError(
                    f"Key '{v}' not found. Possible keys are {item.keys()}"
                ) from e
        return items

    def item_from_id(self, id, split: str):
        return self._hf_dataset[split][id]

    def preprocess_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocesses an item from the dataset.
        This method can be overridden to apply custom preprocessing.
        """
        return item

    def __hash__(self):
        return id(self)
