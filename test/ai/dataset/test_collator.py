import torch
from ai.dataset.collator.mask import masked_collator


def test_masked_collate():
    B = 4
    batch = [dict(boxes=torch.tensor([[0, 0, 1, 1], [1, 0.5, 1, 1]])) for _ in range(B)]
    batch[1]["boxes"] = torch.tensor([]).reshape(0, 4)

    masked_collator(batch)

    assert True
