"""
This module contains the function to fuse nn.Modules together in order to gain speedups.
"""

from abc import abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn


class FusedModule(nn.Module):
    @classmethod
    def build(cls, state_dict: dict, *args, **kwargs):
        # Build the new module from parameters
        module = cls(*args, **kwargs)
        # Load the state into the new module
        module.load_state_dict(state_dict)
        return module

    @staticmethod
    @abstractmethod
    def load_from(x: nn.Module): ...


class FrozenBatchNorm2d(FusedModule):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Register weight / bias into a buffer instead of parameter so that they are not updated by autograd
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return f"num_features={self.weight.shape[0]}, eps={self.eps}"

    @classmethod
    def load_from(cls, x: nn.Module):
        if isinstance(x, nn.BatchNorm2d):
            state = x.state_dict()
            state.pop("num_batches_tracked", None)
            return super().build(state, x.num_features)
        else:
            raise ValueError(f"Unknown module {x}")


def fuse() -> nn.Module:
    """Fuses the possibly fusable modules a main module if possible"""
    raise NotImplementedError()


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse a Conv2d and a BatchNorm2d layer into a single Conv2d layer"""
    conv_fused = deepcopy(conv)

    # Batch norm parameters
    bn_mean, bn_var, bn_gamma, bn_beta = (
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
    )
    # BN std
    bn_std = (bn_var + bn.eps).sqrt()

    # New conv weights
    conv_fused.weight = nn.Parameter(
        (bn_gamma / bn_std).view(-1, 1, 1, 1) * conv.weight
    )
    conv_fused.bias = nn.Parameter(bn_beta - bn_mean * bn_gamma / bn_std)
    return conv_fused


def fuse_conv_conv(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """Fuse two Conv2d layers into a single Conv2d layer"""
    conv_fused = nn.Conv2d(
        in_channels=conv1.in_channels,
        out_channels=conv2.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        dilation=conv1.dilation,
        groups=conv1.groups,
        bias=True,
    )
    conv_fused.weight = nn.Parameter(conv1.weight + conv2.weight)
    conv_fused.bias = nn.Parameter(conv1.bias + conv2.bias)
    return conv_fused
