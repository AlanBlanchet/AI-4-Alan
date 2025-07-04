import torch


def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    assert box1.shape == box2.shape, "Both boxes must have the same shape"

    # Intersection
    tl = torch.max(box1[..., :2], box2[..., :2])
    br = torch.min(box1[..., 2:], box2[..., 2:])

    iou_dim = box1.ndim - 1

    intersection = torch.prod(br - tl, dim=iou_dim).clamp(0)

    # Union
    area1 = torch.prod(box1[..., 2:] - box1[..., :2], dim=iou_dim)
    area2 = torch.prod(box2[..., 2:] - box2[..., :2], dim=iou_dim)

    union = area1 + area2 - intersection

    return intersection / union


def dot_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    ndim = box1.ndim
    assert ndim == box2.ndim, "Both boxes must have the same number of dimensions"
    # In case of ndim > 2, we need to flatten the last two dimensions and bring it back at the end
    iou1_shapes = torch.tensor(box1.shape[:-1])
    iou2_shapes = torch.tensor(box2.shape[:-1])
    # For the last 2 dimensions
    extra_shapes = iou1_shapes[:-1]
    assert (
        extra_shapes == iou2_shapes[:-1]
    ).all(), "Both boxes must have the same shapes after the last 2 dimensions"

    ious: torch.Tensor = torch.zeros((iou1_shapes.prod(), iou2_shapes.prod()))

    box1 = box1.flatten(end_dim=-2)
    box2 = box2.flatten(end_dim=-2)

    for i, b1 in enumerate(box1):
        for j, b2 in enumerate(box2):
            ious[i, j] = iou(b1, b2)

    return ious.view(*extra_shapes, iou1_shapes[-1], iou2_shapes[-1])


def nms(
    scores: torch.Tensor, boxes: torch.Tensor, iou_threshold: float = 0.5
) -> torch.Tensor:
    assert (
        scores.shape == boxes.shape[:-1]
    ), "Scores and boxes must have the same shape (excluding box dimension)"

    # Sorted scores and boxes
    sorted_scores, idx = scores.sort(descending=True)
    sorted_boxes = boxes[idx]

    # IoU between boxes - Triangular matrix
    ious = dot_iou(sorted_boxes, sorted_boxes)

    selected_scores = []
    selected_boxes = []
    while len(sorted_scores) > 0:
        # Select the next box
        selected_scores.append(sorted_scores[0])
        selected_boxes.append(sorted_boxes[0])
        # We now need to remove all similar boxes i.e. boxes with high IoU
        # Get the IoU scores with the rest
        iou_scores = ious[0]
        # Get the boxes below the threshold
        mask = iou_scores < iou_threshold
        # These are still valid boxes since they don't overlap with the selected
        # Keep only the remaining valid boxes
        sorted_scores = sorted_scores[mask]
        sorted_boxes = sorted_boxes[mask]
        ious = ious[mask][:, mask]

    return torch.stack(selected_scores), torch.stack(selected_boxes)
