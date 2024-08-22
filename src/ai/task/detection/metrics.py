import torch
import torchmetrics
import torchmetrics.detection

from ..classification.metrics import ClassificationMetrics


class DetectionMetrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, task_type: str = "multiclass"):
        super().__init__()

        self.clf = ClassificationMetrics(num_classes, task_type)

        metric_classes: list[torchmetrics.Metric] = [
            # torchmetrics.detection.GeneralizedIntersectionOverUnion,
            # torchmetrics.detection.IntersectionOverUnion,
            lambda: torchmetrics.detection.MeanAveragePrecision(
                backend="faster_coco_eval"
            )
        ]

        self.det = torchmetrics.MetricCollection(
            *[metric() for metric in metric_classes],
            compute_groups=False,
        )

    def update(
        self, x: list[dict[str, torch.Tensor]], y: list[dict[str, torch.Tensor]]
    ):
        for xx, yy in zip(x, y):
            self.clf.update(xx["scores"], yy["labels"])
            xx["scores"], xx["labels"] = xx["scores"].max(dim=-1)
        self.det.update(x, y)

    def reset(self):
        self.clf.reset()
        self.det.reset()

    def compute(self):
        detection = self.det.compute()
        detection.pop("classes")  # Not required
        res = dict(classification=self.clf.compute(), detection=detection)
        self.reset()
        return res
