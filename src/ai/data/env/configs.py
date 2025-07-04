import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.transforms import CenterCrop, Compose, Grayscale, Resize


def to_tensor_fn(x):
    # def to_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x / 255.0
        return rearrange(x, "... h w c -> ... c h w")
    else:
        return TF.to_tensor(x)

    # return to_tensor


def to_uint8_fn(x):
    # def to_uint8(x):
    x *= 255.0
    return x.byte()

    # return to_uint8


def crop(x_min, y_min, x_max, y_max):
    def crop_fn(x):
        return TF.crop(x, y_min, x_min, y_max - y_min, x_max - x_min)

    return crop_fn


ENV_CONFIGS = {
    "Breakout": {
        "preprocess": Compose(
            [
                to_tensor_fn,
                Grayscale(),
                Resize(84),
                CenterCrop((84, 84)),
                to_uint8_fn,
            ]
        ),
        "out_shape": (1, 84, 84),
        "max_same_eval": 200,
    },
    "Pong": {
        "preprocess": Compose(
            [to_tensor_fn, Grayscale(), crop(4, 34, 156, 194), to_uint8_fn]
        ),
        "out_shape": (1, 160, 152),
        "max_same_eval": 200,
    },
    # default for ALE (rgb)
    "ALE": {
        "preprocess": Compose([to_tensor_fn]),
    },
}
