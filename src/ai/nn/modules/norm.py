from typing import Optional

import torch.nn as nn

from ..compat.module import Module


class Norm(Module): ...


class BatchNorm2d(Norm, nn.BatchNorm2d):
    num_features: int
    eps: float = 1e-5
    momentum: Optional[float] = 0.1
    affine: bool = True

    track_running_stats: bool = True
    device = None
    dtype = None
