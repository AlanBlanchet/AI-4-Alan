import torch

from .box import dot_iou


def average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    thresholds: torch.Tensor,
):
    ious = dot_iou(pred_boxes, gt_boxes)
    precisions = []
    recalls = []
    for threshold in thresholds:
        ious = ious > threshold

        # True Positive
        tp = torch.zeros_like(scores)
        for i, pred_ious in enumerate(ious):
            for j, valid in enumerate(pred_ious):
                if ious[:i, j].any() and valid and labels[i] == gt_labels[j]:
                    tp[i] = 1

        # False Positive
        fp = 1 - tp

        # Precision and Recall
        tp = tp.sum(dim=-1)
        fp = fp.sum(dim=-1)
        precision = tp / (tp + fp)
        recall = tp / gt_labels.shape[0]

        precisions.append(precision)
        recalls.append(recall)

    precisions = torch.array(precisions)
    recalls = torch.array(recalls)

    # AP Calculation
    ap = 0
    for precision, recall in zip(precisions, recalls):
        ap += precision * (recall[1:] - recall[:-1]).sum()
    return ap


def mean_average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
):
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    classes = torch.unique(gt_labels).sort()[0]

    # [score, labels, x1, y1, x2, y2]
    detections = torch.cat(
        [scores.unsqueeze(-1), labels.unsqueeze(-1), pred_boxes], dim=-1
    )
    # [labels, x1, y1, x2, y2]
    targets = torch.cat([gt_labels.unsqueeze(-1), gt_boxes], dim=-1)

    # Sort predictions by score
    detections = detections.sort(key=lambda x: x[2], reverse=True)[0]

    auc = torch.zeros(len(classes))
    for cls_idx, cls in enumerate(classes):
        precisions = torch.zeros(len(iou_thresholds))
        recalls = torch.zeros(len(iou_thresholds))
        # Calculate for each class
        gt_idx = targets[:, 0] == cls
        cls_targets = targets[gt_idx]

        idx = detections[:, 1] == cls
        cls_detections = detections[idx]

        ious = dot_iou(cls_detections[2:], cls_targets[1:])

        for thresh_idx, threshold in enumerate(iou_thresholds):
            iou_mask = ious > threshold

            # True Positive
            tp = torch.zeros_like(scores)
            for i, pred_ious in enumerate(iou_mask):
                for j, above in enumerate(pred_ious):
                    if iou_mask[:i, j].any() and above and labels[i] == gt_labels[j]:
                        # Here the predictions matched by having the highest IoU
                        tp[i] = 1

            # In every other case where we missed the TP, we have a FP
            fp = 1 - tp
            # FN - GT that we missed
            fn = iou_mask.any(dim=-2).logical_not().sum()
            # Overall TP and FP
            tp = tp.sum()
            fp = fp.sum()
            # Precision and Recall
            precisions[thresh_idx] = tp / (tp + fp)
            recalls[thresh_idx] = tp / (tp + fn)

        # Area under the precision / recall curve with trapezoidal rule
        auc[cls_idx] = torch.trapz(precisions, recalls)

    recall_thresholds = torch.linspace(0, 1, 11)

    # Calculate area under the curve
    aps = []
    for cls_idx in range(len(classes)):
        aps.append(torch.trapz(precisions[cls_idx], recalls[cls_idx]))

    return torch.mean(torch.stack(aps))
