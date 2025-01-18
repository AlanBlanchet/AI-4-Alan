from typing import Callable, ClassVar

import torch
import torch.nn as nn
import torchmetrics

from ..nn.compat.module import Module

METRIC_MODULE_KEYS = dir(torchmetrics.Metric)
METRIC_MODULE_KEYS.remove("forward")

METRIC_MODULE_KEYS.extend(
    [
        "_TORCH_GREATER_EQUAL_2_1",
        "_device",
        "_dtype",
        "compute_on_cpu",
        "dist_sync_on_step",
        "process_group",
        "dist_sync_fn",
        "distributed_available_fn",
        "sync_on_compute",
        "compute_with_cache",
        "_update_signature",
        "update",
        "compute",
        "_computed",
        "_forward_cache",
        "_update_count",
        "_to_sync",
        "_should_unsync",
        "_enable_grad",
        "_dtype_convert",
        "_defaults",
        "_persistent",
        "_reductions",
        "_is_synced",
        "_cache",
    ]
)


class Metric(Module, torchmetrics.Metric, buildable=False):
    INIT_CLS: ClassVar[type] = torchmetrics.Metric
    BASE_SPECIAL_KEYS: ClassVar[list[str]] = METRIC_MODULE_KEYS


class GroupedMetric(nn.Module):
    def __init__(
        self,
        metric: Callable[[], nn.Module],
        groups: list[str],
    ):
        super().__init__()
        self.groups = groups
        self.metrics = {k: metric() for k in groups}

        for k in groups:
            setattr(self, k, self.metrics[k])

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


class EmptyMetric(Metric):
    def update(self, *args, **kwargs): ...

    def compute(self, **kwargs):
        return {}
