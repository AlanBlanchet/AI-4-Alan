"""
This module contains the function to fuse nn.Modules together in order to gain speedups.
"""

from copy import deepcopy

import torch.nn as nn


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
