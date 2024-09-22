import torch.nn as nn

from ...configs.base import Loggable


class Module(nn.Module, Loggable):
    """
    Base class for all models
    """

    log_name = "module"
