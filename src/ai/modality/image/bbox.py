from typing import Literal, Self

import torch

from ...utils.tensor import TensorBase
from ..modality import NormalizedModality


class ModalityAttachment(NormalizedModality): ...


class Label(ModalityAttachment): ...


class ChannelDataAttachement(ModalityAttachment): ...


FORMATS = Literal["xyxy", "xywh", "cxcywh"]


class bboxes(ChannelDataAttachement):
    # Internal format stores the boxes as xyxy
    orig_size: TensorBase
    format: FORMATS = "xyxy"
    normalized: bool = False

    @classmethod
    def modality_format(cls, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            ...
        else:
            raise ValueError(
                f"Expected 1 or 2 dimensions, got {tensor.dim()} dimensions"
            )
        return tensor

    @property
    def orig_sizes(self):
        return self.orig_size[::-1].tile(2)

    def normalize(self):
        if self.normalized:
            return self
        return self / self.orig_sizes

    def unnormalize(self):
        if not self.normalized:
            return self
        return self * self.orig_sizes

    @property
    def cxcywh(self) -> Self:
        if self.format == "cxcywh":
            return self

        new_tensor = self.clone()
        new_tensor.format = "cxcywh"
        if self.format == "xyxy":
            wh = self[..., -2:] - self[..., :2]
            center = self[..., :2] + wh / 2
            new_tensor[..., :2] = center
            new_tensor[..., -2:] = wh
        elif self.format == "xywh":
            new_tensor[..., :2] += new_tensor[..., 2:] / 2

        return new_tensor

    @property
    def xywh(self) -> Self:
        if self.format == "xywh":
            return self

        new_tensor = self.clone()
        new_tensor.format = "xywh"
        if self.format == "xyxy":
            wh = new_tensor[..., -2:] - new_tensor[..., :2]
            new_tensor[..., 2:] = wh
        elif self.format == "cxcywh":
            wh = new_tensor[..., 2:]
            new_tensor[..., :2] -= wh / 2

        return new_tensor

    @property
    def xyxy(self) -> Self:
        if self.format == "xyxy":
            return self

        new_tensor = self.clone()
        new_tensor.format = "xyxy"
        if self.format == "cxcywh":
            wh = new_tensor[..., 2:]
            new_tensor[..., :2] -= wh / 2
            new_tensor[..., -2:] = new_tensor[..., :2] + wh
        elif self.format == "xywh":
            new_tensor[..., :2] += new_tensor[..., 2:] / 2
            new_tensor[..., -2:] = new_tensor[..., :2] + new_tensor[..., 2:]

        return new_tensor
