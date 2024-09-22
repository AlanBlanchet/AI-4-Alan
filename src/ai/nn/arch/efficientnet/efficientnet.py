import torch.nn as nn

from ....registry import REGISTER


@REGISTER
class EfficientNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # TODO
        raise NotImplementedError()
