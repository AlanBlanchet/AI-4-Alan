from typing import Literal

import numpy as np
import torchvision.transforms.functional as TF

from ...utils.pydantic_ import validator
from ..preprocess import Preprocess


class ImageAugmentation(Preprocess, buildable=False): ...


class Crop(ImageAugmentation):
    """Simple crop with different modes"""

    size: tuple[int, int]
    """Desired output size"""
    mode: Literal["center", "random"] = "center"
    """Mode of cropping"""

    @validator("size")
    def validate_size(cls, value):
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return value
        return value

    def __call__(self, image):
        img_h, img_w = image.shape[-2:]
        h, w = self.size  # Size of the wanted crop inside the image

        if self.mode == "center":
            return TF.center_crop(image, (h, w))
        elif self.coords == "random":
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            return TF.crop(image, top, left, h, w)


class PadCrop(ImageAugmentation):
    size: tuple[int, int, int, int]

    @validator("size")
    def validate_size(cls, value):
        if isinstance(value, int):
            return (value, value, value, value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return (value[0], value[1], value[0], value[1])
        return value

    def __call__(self, image):
        img_h, img_w = image.shape[-2:]
        t, r, b, l = self.size
        return TF.crop(image, t, l, img_h - (b + t), img_w - (r + l))


class GrayScale(ImageAugmentation):
    def __call__(self, image):
        return TF.rgb_to_grayscale(image)


class Resize(ImageAugmentation):
    size: int | tuple[int, int]

    def __call__(self, image):
        return TF.resize(image, self.size)
