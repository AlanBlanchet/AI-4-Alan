from typing import Literal, get_args

import torch
import torch.nn.functional as F

REDUCTION_TYPES = Literal["mean", "none"] | None
REDUCTIONS: list[REDUCTION_TYPES] = list(get_args(REDUCTION_TYPES))


def _format(logits, targets):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    targets: torch.Tensor = torch.as_tensor(targets, dtype=int)
    targets = F.one_hot(targets, logits.shape[-1])
    return logits, targets


def _format_softmax(logits, targets):
    logits, targets = _format(logits, targets)
    return logits.softmax(dim=-1), targets


def mse(logits, targets, reduction: REDUCTION_TYPES = "mean"):
    preds, targets = _format_softmax(logits, targets)

    targets_batch = torch.mean((preds - targets) ** 2, dim=-1)

    if reduction in [None, "none"]:
        return targets_batch

    return targets_batch.mean()


def mae(logits, targets, reduction: REDUCTION_TYPES = "mean"):
    preds, targets = _format_softmax(logits, targets)

    print(logits)
    print(logits.softmax(dim=-1))

    targets_batch = torch.mean((preds - targets).abs(), dim=-1)

    if reduction in [None, "none"]:
        return targets_batch

    return targets_batch.mean()


def rmse(logits, targets):
    return torch.sqrt(mse(logits, targets))
