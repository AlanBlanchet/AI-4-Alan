import torch.nn as nn

from ..compat.module import Module

ACTIVATIONS_NAMES = [
    "ReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
    "ELU",
    "SELU",
    "CELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "PReLU",
    "Softsign",
    "Softmin",
    "Tanhshrink",
    "RReLU",
]


class Activation(Module, buildable=False): ...


Activation.create_classes(
    namespace=globals(), module=nn, selected_names=ACTIVATIONS_NAMES
)
