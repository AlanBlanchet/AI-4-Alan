import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision


class GroupedMetrics(nn.Module):
    def __init__(
        self,
        groups: list[str],
        metrics: list[nn.Module],
    ):
        super().__init__()

        self.prefix = "_"

        self.metrics = nn.ModuleDict(
            {
                f"{self.prefix}{g}": MetricCollection(
                    [MeanAveragePrecision() for metric in metrics], prefix=f"{g}/"
                )
                for g in groups
            }
        )

        # To consider as valid module
        self._ = nn.Identity()

    def has_updated(self, split):
        return self.metrics[f"{self.prefix}{split}"]._has_updated

    def _apply(self, fn):
        for _, v in self.metrics.items():
            v._apply(fn)

    def reset(self, split: str):
        self.metrics[f"{self.prefix}{split}"].reset()

    def update(self, x: torch.Tensor, y: torch.Tensor, split: str):
        self.metrics[f"{self.prefix}{split}"].update(x, y)

    def compute(self, split: str):
        return self.metrics[f"{self.prefix}{split}"].compute()
