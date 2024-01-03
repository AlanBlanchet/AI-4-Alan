import torch

from .common import mae, mse

INF = 1e50


def test_mae():
    logits = torch.tensor([[-INF, -INF, 0.9, 0.9], [-INF, 50, -INF, -INF]])  # B, C
    targets = [0, 1]

    res = mae(logits, targets, None)

    assert torch.allclose(res, torch.tensor([0.5, 0]))


def test_mse():
    logits = torch.tensor([[-INF, -INF, 0.9, 0.9], [-INF, 50, -INF, -INF]])  # B, C
    targets = [0, 1]

    res = mse(logits, targets, None)

    assert torch.allclose(res, torch.tensor([0.375, 0]))
