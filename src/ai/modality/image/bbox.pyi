from ai.utils.tensor import TensorBase
from myconf.core import Consumed
from torch import Tensor
from torch import dtype
from typing import Literal

class bboxes:
    data: Tensor
    device: Consumed
    dtype: Consumed
    requires_grad: Consumed
    orig_size: TensorBase
    format: Literal
    normalized: bool
    def __init__(self, **kwargs) -> None: ...