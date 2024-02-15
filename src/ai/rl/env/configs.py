import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.transforms import CenterCrop, Compose, Grayscale


def to_tensor_fn():
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            x = x / 255.0
            return rearrange(x, "... h w c -> ... c h w")
        else:
            return TF.to_tensor(x)

    return to_tensor


ENV_CONFIGS = {
    "ALE/Breakout": {
        "preprocess": Compose([to_tensor_fn(), Grayscale(), CenterCrop((140, 180))]),
        "out_shape": (1, 140, 180),
    },
}
