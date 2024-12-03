import torch
import torchmetrics


class RLMetric(torchmetrics.Metric):
    def __init__(self, keys: list[str]):
        super().__init__()

        self.keys = keys

        self.means = torchmetrics.MetricCollection(
            {key: torchmetrics.MeanMetric() for key in keys}
        )

    def update(self, preds, batch: dict[str, torch.Tensor]):
        for key in self.keys:
            if key in preds:
                self.means[key].update(preds[key])

    def reset(self):
        self.means.reset()

    def compute(self):
        return self.means.compute()
