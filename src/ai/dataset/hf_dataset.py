from functools import cached_property
from typing import Callable, Type

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Sequence, load_dataset
from pydantic import BaseModel

from ..registry.registers import SOURCE
from ..task.classification.label_map import LabelMap
from ..utils.env import AIEnv
from .base_dataset import BaseDataset

MAX_DATA = 100000
# MAX_DATA = 80


@SOURCE.register
class HuggingFaceDataset(BaseDataset):
    hf_params: dict
    split_train: dict
    split_val: dict

    _dataset: Dataset = None
    _info: dict = None
    _params: dict = None
    _get_transforms: Callable = None

    @property
    def name(self) -> str:
        return self.hf_params["path"]

    @classmethod
    def build(cls, **kwargs):
        t, v = kwargs.pop("split_train", {}), kwargs.pop("split_val", {})
        return cls(hf_params=kwargs, split_train=t, split_val=v)

    @cached_property
    def _hf_dataset(self) -> Dataset:
        path = self.hf_params["path"]
        self.log(f"Loading dataset {path}")
        return load_dataset(**self.hf_params, trust_remote_code=True)

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
        input: str,
        outputs: dict[str, str],
        label_map: LabelMap = None,
        params: dict = None,
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

        self._info = dict(input=input, **outputs)
        self._params = params
        self._get_transforms = get_transforms

    def train(self):
        name = self.split_train.get("name", "train")
        ratio = self.split_train.get("size", 1)
        train_ds = self._hf_dataset[name]
        skip_num = int(len(train_ds) * (1 - ratio))
        if skip_num > 0:
            train_ds = train_ds.skip(skip_num)

        return HuggingFaceTorchDataset(
            dataset=train_ds.with_format("np"),
            info=self._info,
            params=self._params,
            transforms=self._get_transforms(),
        )

    def val(self):
        name = self.split_val.get("name", "val")
        ratio = self.split_val.get("size", 1)
        val_ds = self._hf_dataset[name]

        skip_num = int(len(val_ds) * (1 - ratio))
        if skip_num > 0:
            val_ds = val_ds.skip(skip_num)

        return HuggingFaceTorchDataset(
            dataset=val_ds.with_format("np"),
            info=self._info,
            params=self._params,
            transforms=self._get_transforms(val=True),
        )

    class Config:
        arbitrary_types_allowed = True


class HuggingFaceTorchDataset(BaseModel):
    dataset: Dataset
    info: dict
    params: dict
    transforms: Callable

    def __len__(self):
        return len(self.dataset)

    def get_elem(self, item: dict, k: str):
        keys = k.split(".")

        elem = item[keys[0]]

        if len(keys) > 1:
            return self.get_elem(elem, ".".join(keys[1:]))

        return elem.copy()

    def __getitem__(self, idx):
        item = self.dataset[idx]
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

        transformed = self.transforms(
            image=input_elem, bboxes=boxes_elem, labels=labels_elem
        )

        for k, v in zip(["input", "boxes", "labels"], transformed.values()):
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
