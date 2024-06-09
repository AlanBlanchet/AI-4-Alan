import torch.nn as nn

from ....registry.registers import MODEL


@MODEL.register
class EfficientNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # TODO
        raise NotImplementedError()
