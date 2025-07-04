import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

DATASET_MAPPING = Literal["ms_coco_91_2_80"]


def patch_weight(mapping: DATASET_MAPPING, normalize: bool = False):
    """
    Set the weights of the patch.

    Args:
        weights (list): The weights of the patch.
    """
    file = Path(__file__).parent / f"dataset/{mapping}.json"
    mapping: dict = json.loads(file.read_text())
    keys = torch.as_tensor([int(k) for k in mapping.keys()])
    values = torch.as_tensor([int(v) for v in mapping.values()])
    if normalize:
        # If index starts at 1
        keys -= torch.min(keys.min(), values.min())
    return keys


def patch_linear(
    l1: nn.Linear,
    mapping: DATASET_MAPPING,
    extra_features: int = 0,
    normalize: bool = False,
):
    """
    Extra features are supposed to be at the end of the linear layer.
    """
    mapping_idx = patch_weight(mapping, normalize=normalize)
    # Concat the extra features
    extra_tensor = torch.arange(l1.out_features - extra_features, l1.out_features)
    mapping_idx = torch.cat([mapping_idx, extra_tensor])
    num_features = len(mapping_idx)
    bias = l1.bias is not None
    # Create new linear layer
    l2 = nn.Linear(l1.in_features, num_features, bias=bias)
    l2.weight.data[...] = l1.weight[mapping_idx]

    if bias:
        l2.bias.data[...] = l1.bias[mapping_idx]

    return l2
