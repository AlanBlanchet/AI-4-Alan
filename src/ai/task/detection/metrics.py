import torch
import torchmetrics
import torchmetrics.detection


def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class DetectionMetrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, task_type: str = "multiclass"):
        super().__init__()

        # self.clf = ClassificationMetrics(num_classes, task_type)

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

    def update(self, preds, batch: dict[str, torch.Tensor]):
        x, y = [], []

        for scores, labels, bbox, gt_labels, gt_bbox in zip(
            preds["scores"],
            preds["labels"],
            preds["bbox"],
            batch["labels"],
            batch["bbox"],
        ):
            x.append(dict(boxes=bbox, scores=scores, labels=labels))
            y.append(dict(boxes=gt_bbox, labels=gt_labels))

        self.det.update(x, y)

    def reset(self):
        # self.clf.reset()
        self.det.reset()

    def compute(self):
        detection = self.det.compute()
        detection.pop("classes")  # Not required
        res = dict(detection=detection)  # classification=self.clf.compute()
        self.reset()
        return res
