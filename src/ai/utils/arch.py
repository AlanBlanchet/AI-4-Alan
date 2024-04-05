from importlib import import_module

import torch.nn as nn

from .paths import AIPaths


@AIPaths.fn_cache
def get_arch_module(arch_name: str) -> nn.Module:
    # Get the main module
    arch_ms = AIPaths.path2module(AIPaths.archs_p)
    arch_m = import_module(arch_ms)

    # Get the correct class
    return getattr(arch_m, arch_name)
