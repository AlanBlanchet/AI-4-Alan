from ai.modality.image.bbox import bboxes
from myconf.core import Consumed
from torch import Tensor
from torch import dtype
from typing import Optional

class Image:
    data: Tensor
    device: Consumed
    dtype: Consumed
    requires_grad: Consumed
    channel_first: Consumed
    bounding_boxes: bboxes
    def __init__(self, **kwargs) -> None: ...