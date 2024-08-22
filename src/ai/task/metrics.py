from typing import Callable

import torch
import torch.nn as nn


class GroupedMetric(nn.Module):
    def __init__(
        self,
        metric: Callable[[], nn.Module],
        groups: list[str],
    ):
        super().__init__()
        self.groups = groups
        self.metrics = {k: metric() for k in groups}

        # To consider as valid module
        self._ = nn.Identity()

    def has_updated(self, split):
        return self.metrics[split]._has_updated

    def _apply(self, fn):
        for _, v in self.metrics.items():
            v._apply(fn)

    def reset(self, split: str):
        self.metrics[split].reset()

    def update(self, x: torch.Tensor, y: torch.Tensor, split: str):
        self.metrics[split].update(x, y)

    def compute(self, split: str):
        return self.metrics[split].compute()
