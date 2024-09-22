from functools import cached_property
from typing import Callable, Type

import numpy as np
import torch
from datasets import (
    ClassLabel,
    Dataset,
    Features,
    IterableDataset,
    Sequence,
    load_dataset,
)
from pydantic import BaseModel

from ..configs.main import DatasetConfig, TaskConfig
from ..task.classification.label_map import LabelMap
from ..utils.env import AIEnv
from .base_dataset import BaseDataset

MAX_DATA = 100000
# MAX_DATA = 80


class HuggingFaceDataset(BaseDataset):
    config: DatasetConfig
    task_config: TaskConfig

    @property
    def name(self) -> str:
        return self.config.params["path"]

    @classmethod
    def build(cls, **kwargs):
        t, v = kwargs.pop("split_train", {}), kwargs.pop("split_val", {})
        return cls(hf_params=kwargs, split_train=t, split_val=v)

    @cached_property
    def _hf_dataset(self) -> Dataset:
        params = self.config.params
        path = params["path"]
        self.log(f"Loading dataset {path}")
        ds = load_dataset(**params, trust_remote_code=True)
        self.log(f"Available splits: {list(ds.keys())}")
        return ds

    @cached_property
    def _hf_features(self) -> Features:
        return self._hf_dataset["train"].features

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

    @cached_property
    def _hf_class_label(self) -> tuple[str, ClassLabel]:
        return self._hf_find_field(self._hf_features, field_type=ClassLabel)

    @cached_property
    def _labels(self):
        return self._hf_class_label[-1]._int2str

    def prepare(
        self,
        label_map: LabelMap = None,
        # params: dict = None,
        get_transforms: Callable = None,
    ):
        if label_map and not label_map.keep_order:
            p, class_label = self._hf_class_label

            s2i = class_label._str2int
            i2s = {v: k for k, v in s2i.items()}

            # Build dict schema
            obj_p = p.split(".")
            obj_parent_p = obj_p[: len(obj_p) - 1]
            obj_last_key = obj_p[-1]

            def get_obj(x):
                for p in obj_parent_p:
                    x = x[p]
                return x

            # Apply our labels instead of hugging face labels
            def to_local_label_apply(x):
                # Get parent
                parent = get_obj(x)
                # Map
                hf_ids = parent[obj_last_key]
                hf_labels = i2s[hf_ids]  # Get label from old id
                parent[obj_last_key] = label_map[hf_labels]  # Replace with new id
                return x

            # Apply the transformation
            self._dataset = self._hf_dataset.map(
                to_local_label_apply,
                batched=True,
                num_proc=AIEnv.DEFAULT_NUM_PROC,
            )

        self._info = dict(input=self.task_config.input, labels=self.task_config.labels)
        # self._params = params

        self._get_transforms = get_transforms

    def split_info(self, name: str):
        first_k = list(self._hf_dataset.keys())[0]
        return self._hf_dataset[first_k].info.splits[name]

    def train(self):
        name = self.config.train.name
        ratio = self.config.train.size
        train_ds = self._hf_dataset[name]
        num = self.split_info(name).num_examples
        skip_num = int(num * (1 - ratio))
        num -= skip_num
        if skip_num > 0:
            train_ds = train_ds.skip(skip_num)

        return HuggingFaceTorchDataset(
            dataset=train_ds.with_format("np"),
            num=num,
            info=self._info,
            # params=self._params,
            transforms=self._get_transforms(),
        )

    def val(self):
        name = self.config.val.name
        ratio = self.config.val.size
        val_ds = self._hf_dataset[name]
        num = self.split_info(name).num_examples
        skip_num = int(num * (1 - ratio))
        num -= skip_num
        if skip_num > 0:
            val_ds = val_ds.skip(skip_num)

        return HuggingFaceTorchDataset(
            dataset=val_ds.with_format("np"),
            num=num,
            info=self._info,
            # params=self._params,
            transforms=self._get_transforms(val=True),
        )

    class Config:
        arbitrary_types_allowed = True


class HuggingFaceTorchDataset(BaseModel):
    dataset: Dataset | IterableDataset
    info: dict
    num: int
    # params: dict
    transforms: Callable

    def __len__(self):
        return self.num

    def get_elem(self, item: dict, k: str):
        try:
            keys = k.split(".")
            elem = item[keys[0]]
            if len(keys) > 1:
                return self.get_elem(elem, ".".join(keys[1:]))
            return elem.copy()
        except KeyError as e:
            raise KeyError(
                f"Key '{k}' not found. Possible keys are {item.keys()}"
            ) from e

    # def __iter__(self):
    #     for item in iter(self.dataset):
    #         yield self.process(item)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.process(item)

    def process(self, item):
        res = {}

        all_infos = self.info.copy()
        input = all_infos.pop("input")
        input_elem = self.get_elem(item, input)
        boxes = all_infos.pop("boxes", None)
        boxes_elem = self.get_elem(item, boxes) if boxes else None
        labels = all_infos.pop("labels", None)
        labels_elem = self.get_elem(item, labels) if labels else None

        if input_elem.ndim == 2:
            input_elem = input_elem[:, :, None]

        if input_elem.shape[-1] == 1:
            input_elem = np.repeat(input_elem, 3, axis=-1)

        transformed = self.transforms(image=input_elem)

        if "labels" not in transformed and labels_elem is not None:
            labels = torch.as_tensor(labels_elem)
            if labels.numel() > 1:
                # Maybe there are multiple labels per image ?
                labels = labels[0]

            if labels.ndim > 0:
                labels = labels.item()

            transformed["labels"] = labels

        for k, v in zip(["input", "labels", "boxes"], transformed.values()):
            if isinstance(v, (np.ndarray, torch.Tensor)):
                res[k] = v.clone()
            else:
                v = torch.tensor(v)
                if k == "boxes":
                    if len(boxes_elem) == 0:
                        boxes_elem = np.zeros((0, 4))

                    h, w = transformed["image"].shape[-2:]
                    v[..., 0::2] /= w
                    v[..., 1::2] /= h
                res[k] = v

        for k, v in all_infos.items():
            elem = self.get_elem(item, v)

            if isinstance(elem, np.ndarray):
                elem = torch.from_numpy(elem)
            else:
                torch.tensor(elem)

            res[k] = elem
        return res

    class Config:
        arbitrary_types_allowed = True
