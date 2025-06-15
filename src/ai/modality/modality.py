from functools import cache
from typing import Any, ClassVar

import torch
import torch.nested as nested
from pydantic import Field

from ..configs.base import Base
from ..configs.log import Color
from ..utils.pydantic_ import validator
from ..utils.types import CallableList, is_dict, is_number
from .preprocess import Preprocess, Preprocesses


class Modality(Base):
    log_name: ClassVar[str] = "modality"
    color: ClassVar[str] = Color.red

    input: list[str] = Field([], validate_default=True)

    preprocesses: Preprocesses = Preprocesses([])

    @validator("input")
    def validate_input(cls, value):
        if not isinstance(value, list):
            return [value]
        return value

    @validator("preprocesses")
    def validate_preprocesses(cls, value):
        real_list = []
        for v in value:
            if isinstance(v, str):
                real_list.append({"type": v})
            elif isinstance(v, dict):
                if "type" not in v:
                    v["type"] = list(v.keys())[0]
                    poped_val = v.pop(v["type"])
                    v["_args"] = (poped_val,)
                real_list.append(v)
            else:
                real_list.append(v)

        return Preprocess.from_config(real_list)

    def __call__(self, data: dict):
        """Calls the preprocess method for accepted inputs"""
        for k, v in self._gather_accepted(data).items():
            data[k] = self.preprocesses(v)
        return data

    @classmethod
    @cache
    def modalities(cls):
        return {k: sub for sub in cls.__subclasses__() for k in sub.model_fields.keys()}

    def item_accept(self, key: str, value: Any):
        """Method responsible for specifying if the modality accepts the item"""
        return len(self.input) == 0 or key in self.input

    def _gather_accepted(self, items: dict[str, list]):
        """Gathers only the accepted items"""
        return {k: v for k, v in items.items() if self.item_accept(k, v)}

    @classmethod
    def collate(cls, batch: list):
        collated = {}
        # modalities = cls.modalities()
        names = batch[0].keys()
        transposed = zip(*[b.values() for b in batch])
        items = {name: samples for name, samples in zip(names, transposed)}

        for name, samples in items.items():
            ex = samples[0]

            if is_number(ex):
                collated[name] = torch.as_tensor(samples)
            elif is_dict(ex):
                collated[name] = cls.collate(samples)
            else:
                # masked = cls.mask_collate(name, samples)
                collated[name] = nested.nested_tensor(samples)

        return collated

    @classmethod
    def mask_collate(cls, name: str, samples: list[Any]):
        """Generates a mask for the data if necessary"""
        return nested.nested_tensor(samples)
        # samples = [torch.as_tensor(s) for s in samples]

        # res = {}
        # lengths = [len(s) if s.dim != 0 else 0 for s in samples]
        # max_len = max(lengths)

        # elem = torch.zeros(
        #     (len(samples), max_len, *samples[0].shape[1:]),
        #     dtype=samples[0].dtype,
        # )
        # mask = torch.zeros((len(samples), max_len), dtype=torch.bool)
        # for i, ln in enumerate(lengths):
        #     elem[i, :ln] = samples[i]
        #     mask[i, :ln] = True
        # res[name] = elem
        # res[f"{name}_mask"] = mask
        # return res


class Modalities(CallableList[Modality]):
    def __call__(self, data):
        for modality in self:
            data = modality(data)
        return data

    def get(self, modality: type[Modality]):
        """Get a modality by type and create it if it doesn't exist"""
        for m in self:
            if isinstance(m, modality):
                return m
        m = modality()
        self.append(m)
        return m
