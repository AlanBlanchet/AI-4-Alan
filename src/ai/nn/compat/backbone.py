from __future__ import annotations

import torch

from .input_proj import InputProjectionMixin


class IsBackboneMixin(InputProjectionMixin):
    out_dim: int

    def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out = avg.flatten(1)
        return out
