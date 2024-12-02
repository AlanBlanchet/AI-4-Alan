import inspect
import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.pydantic import Field, InterpolationType, ProbabilityType
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    Targets,
)


class RandomResize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        sizes: int = Field(ge=1, description="Possible sizes")
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1

    def __init__(
        self,
        scales: list[int],
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p, always_apply)
        self.scales = scales
        self.interpolation = interpolation
        self._current_scale = None

    def apply(self, img: np.ndarray, interpolation: int, **params: Any) -> np.ndarray:
        self._current_scale = random.choice(self.sizes)
        return fgeometric.resize(
            img, (self._current_scale, self._current_scale), interpolation=interpolation
        )

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        height, width = params["shape"][:2]
        scale_x = self._current_scale / width
        scale_y = self._current_scale / height
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "sizes", "interpolation"


# Generate all possible augs
AUGS = {
    **{k: v for k, v in A.augmentations.__dict__.items() if inspect.isclass(v)},
    **{v.__name__: v for v in [RandomResize]},
}
