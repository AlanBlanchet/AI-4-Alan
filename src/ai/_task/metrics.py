# from functools import cached_property
# from typing import Callable, ClassVar

# import torch
# import torchmetrics

# from ..nn.compat.module import Module


import torchmetrics

from ..nn.compat.module import Module, ModuleList


class Metric(Module, torchmetrics.Metric): ...


class Metrics(ModuleList[Metric]): ...


# class Metrics(Metric):
#     metric: Callable[[], Module]
#     groups: list[str]

#     @cached_property
#     def metrics(self):
#         return {g: self.metric() for g in self.groups}

#     def has_updated(self, split):
#         return self.metrics[split]._has_updated

#     def _apply(self, fn):
#         for _, v in self.metrics.items():
#             v._apply(fn)

#     def reset(self, split: str):
#         self.metrics[split].reset()

#     def update(self, x: torch.Tensor, y: torch.Tensor, split: str):
#         self.metrics[split].update(x, y)

#     def compute(self, split: str):
#         return self.metrics[split].compute()


# class EmptyMetric(Metric):
#     def update(self, *args, **kwargs): ...

#     def compute(self, **kwargs):
#         return {}
