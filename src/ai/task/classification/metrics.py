import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.utilities import dim_zero_cat


class ClassificationMetrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, task_type: str = "multiclass"):
        super().__init__()

        self.num_classes = num_classes

        metric_classes: list[torchmetrics.Metric] = [
            torchmetrics.Accuracy,
            torchmetrics.Precision,
            torchmetrics.Recall,
            torchmetrics.F1Score,
            torchmetrics.Specificity,
            # torchmetrics.AUROC,
        ]

        self.common = torchmetrics.MetricCollection(
            *[
                metric(task=task_type, num_classes=num_classes, average=None)
                for metric in metric_classes
            ],
            compute_groups=False,
        )

        self.other_losses = torchmetrics.MetricCollection(
            torchmetrics.MeanAbsoluteError(), torchmetrics.MeanSquaredError()
        )

        self.roc = torchmetrics.ROC(
            task=task_type, num_classes=num_classes, thresholds=20
        )

        cm_normalizations = [None, "true"]
        self.cms = torchmetrics.MetricCollection(
            {
                f"confusion_matrix{'' if normalize is None else f' {normalize}'}": torchmetrics.ConfusionMatrix(
                    task=task_type,
                    num_classes=num_classes,
                    normalize=normalize,
                )
                for normalize in cm_normalizations
            },
            compute_groups=False,
        )

        # TODO fix probabilities
        # self.probs = ClassProbabilities(num_classes)

    def update(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape(-1, self.num_classes)
        y = y.flatten()

        self.common.update(x, y)
        self.roc.update(x, y)
        self.cms.update(x, y)
        # self.probs.update(x, y)
        self.other_losses.update(x.softmax(dim=-1), F.one_hot(y, self.num_classes))

    def reset(self):
        self.common.reset()
        self.roc.reset()
        self.cms.reset()
        # self.probs.reset()
        self.other_losses.reset()

    def compute(self):
        res = dict(
            common=self.common.compute(),
            roc=self.roc.compute(),
            cms=self.cms.compute(),
            # probs=self.probs.compute(),
            other_losses=self.other_losses.compute(),
        )
        self.reset()
        return res


class ClassProbabilities(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("preds", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([]), dist_reduce_fx="cat")
        self.preds: torch.Tensor
        self.target: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self._reduce_states({"preds": preds, "target": target})

    def compute(self):
        cls_probs = [[] for _ in range(self.num_classes)]

        preds = dim_zero_cat(self.preds)  # B, C
        target = dim_zero_cat(self.target).int()  # B

        has_nan = preds.isnan().any(dim=-1)  # B

        preds = preds[~has_nan]  # B'
        target = target[~has_nan]  # B'

        preds = preds.softmax(dim=-1)  # B', C

        for pred, t in zip(preds, target):
            t_prob = pred[t]  # B'
            cls_probs[t].append(t_prob)

        return [torch.tensor(p) for p in cls_probs]
