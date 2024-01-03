import torch
import torch.nn.functional as F
from attr import define, field


@define(slots=False)
class ConvXd:
    in_channels: int = field()
    out_channels: int = field()
    kernel_size: int = field(default=3)
    padding: int = field(default=0)
    stride: int = field(default=1)
    dim = field(default=1)

    def __attrs_post_init__(self):
        self.filters = torch.randn(
            self.out_channels, self.kernel_size, requires_grad=True
        )

    def __call__(self, x):
        n = x.shape[-1]
        p = self.padding
        s = self.stride
        ks = self.kernel_size
        out_dim = n + p * 2 - (n // s if s != 1 else 0) - int(ks - 1)
        out = torch.empty((self.out_channels, out_dim))
        x = F.pad(x, pad=(1, 1))
        for i in range(out_dim):
            out[..., i] = (x[..., i : i + ks] * self.filters).mean(dim=-1)

        return out


@define(slots=False)
class Conv1d(ConvXd):
    dim = field(default=1)


