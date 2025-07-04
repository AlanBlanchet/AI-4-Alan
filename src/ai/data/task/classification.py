from functools import cached_property

from ...modality.image.modality import Image
from ...modality.modality import Modality
from ...utils.env import AIEnv
from ..dataset import Data
from ..huggingface import HuggingFaceDataset
from ..label_map import LabelMap


class ClassificationData(Data):
    input_map: dict[str, type[Modality]]
    target_map: dict[str, type[Modality]]

    @cached_property
    def labels(self):
        if isinstance(self, HuggingFaceDataset):
            return LabelMap(labels=self._hf_class_label[-1]._int2str)
        raise NotImplementedError(f"Labels not found for {self.__class__}")

    @property
    def num_classes(self):
        return len(self.labels)

    def prepare(self, **_):
        if isinstance(self, HuggingFaceDataset):
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
                hf_labels = [i2s[id] for id in hf_ids]  # Get label from old id
                parent[obj_last_key] = self.labels[hf_labels]  # Replace with new id
                return x

            # Apply the transformation
            self._dataset = self._hf_dataset.map(
                to_local_label_apply,
                batched=True,
                num_proc=AIEnv.DEFAULT_NUM_PROC,
            )


class Cifar10(HuggingFaceDataset, ClassificationData):
    path = "uoft-cs/cifar10"
    val = "test"

    input_map = {"img": Image}
    target_map = {"label": "labels"}


class ImageNet1k(HuggingFaceDataset, ClassificationData):
    path = "ILSVRC/imagenet-1k"
    val = "validation"

    input_map = {"img": Image}
    target_map = {"label": "labels"}
