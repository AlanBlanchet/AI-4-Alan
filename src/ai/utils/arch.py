from importlib import import_module

import torch.nn as nn

from .env import AIEnv


@AIEnv.fn_cache
def get_arch_module(arch_name: str) -> nn.Module:
    # Get the main module
    arch_ms = AIEnv.path2module(AIEnv.archs_p)
    arch_m = import_module(arch_ms)

    # Get the correct class
    return getattr(arch_m, arch_name)


def build_arch_module(arch_name: str, *args, **kwargs) -> nn.Module:
    return get_arch_module(arch_name)(*args, **kwargs)
