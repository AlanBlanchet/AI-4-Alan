import io

import torch
import torch.nn as nn


def clone_module(module: nn.Module) -> nn.Module:
    # Write to a buffer
    buffer = io.BytesIO()

    torch.save(module, buffer)

    # Read from buffer
    buffer.seek(0)
    clone: nn.Module = torch.load(buffer)

    return clone


def copy_weights(copy: nn.Module, to: nn.Module):
    for c, t in zip(copy.parameters(), to.parameters()):
        t.data.copy_(c.data)
    return to
