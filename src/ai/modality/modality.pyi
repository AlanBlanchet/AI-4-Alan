from myconf.core import Consumed
from torch import Tensor
from torch import dtype

class NormalizedModality:
    data: Tensor
    device: Consumed
    dtype: Consumed
    requires_grad: Consumed
    def __init__(self, **kwargs) -> None: ...