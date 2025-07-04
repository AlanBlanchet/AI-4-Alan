import torch
import torch.optim as optim

from ..nn.compat.module import Module


class Optimizer(Module, optim.Optimizer): ...


class AdamW(Optimizer, optim.AdamW):
    lr: float | torch.Tensor = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None


class LRDecay(Module):
    optimizer: Optimizer


class ExponentialLR(LRDecay, optim.lr_scheduler.ExponentialLR):
    gamma: float
