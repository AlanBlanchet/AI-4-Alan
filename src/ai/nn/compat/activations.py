import torch.nn as nn

from .module import Module


class Activation(Module): ...


class ReLU(Activation, nn.ReLU): ...


class Hardtanh(Activation, nn.Hardtanh): ...


class ReLU6(Activation, nn.ReLU6): ...


class Sigmoid(Activation, nn.Sigmoid): ...
