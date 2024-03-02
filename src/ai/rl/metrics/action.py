from torchmetrics import Metric, SumMetric


class ActionMetric(Metric):
    def __init__(self, num_classes: int, labels: list[str]):
        super().__init__()
        self.num_classes = num_classes
        self.labels = labels

        self.sum_metrics = [SumMetric() for _ in range(self.num_classes)]
        self._num = 0

    def update(self, x):
        self.sum_metrics[x].update(1)
        self._num += 1

    def reset(self):
        for metric in self.sum_metrics:
            metric.reset()
        self._num = 0

    def compute(self):
        return {
            label: metric.compute() / self._num
            for label, metric in zip(self.labels, self.sum_metrics)
        }
