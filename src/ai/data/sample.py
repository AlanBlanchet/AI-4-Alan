import torch

from ..configs.base import Base


class Sample(Base):
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor] = {}
