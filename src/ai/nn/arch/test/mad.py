from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

from ....configs.base import Base
from ...compat.module import Module

TENSOR_ACTIONS = dict(
    softmax=dict(
        op=torch.softmax,
        weight=2
    ),
    add=dict(
        op=torch.add,
        weight=2
    ),
    mul=dict(
        op=torch.mul,
        weight=1
    ),
    div=dict(
        op=torch.div,
        weight=1
    )
)

class TensorActionPredictor(Module):
    def init(self):
        self.selector = torch.nn.Linear(5, len(TENSOR_ACTIONS))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        ...


class Combination(BaseModel):
    args: list[Any]


class PossibleNetwork(Base):
    module: type[nn.Module]
    combinations: dict[str, Combination]
    args: dict[str, Any] = {}


NETWORK_SELECTION_3D = [
    PossibleNetwork(
        module=nn.MultiheadAttention,
        combinations=dict(
            embed_dim=Combination(args=[128, 256, 512, 1024]),
            num_heads=Combination(args=[1, 2, 4, 8]),
        ),
        args=dict(
            batch_first=True
        )
    ),
]

class NetworkSelector(Module):
    def init(self):
        self.network_combinations = []

        networks_3d = self.gather_networks(NETWORK_SELECTION_3D)


        self.dim2_selector = torch.nn.Linear(5, sum(self.network_combinations))
        self.dim3_selector = torch.nn.Linear(5, sum(self.network_combinations))

    def gather_networks(self, available_networks: list[PossibleNetwork]):
        network_combinations = []
        for net in NETWORK_SELECTION_3D:
            net_combinations = 1
            for k, v in net.combinations.items():
                net_combinations *= len(v.args)
            
            network_combinations.append(net_combinations)
        return network_combinations

    def select2d(self, x: torch.Tensor):
        ...

    def select3d(self, x: torch.Tensor):
        ...

    def forward(self, shape: tuple):
        shape = shape[1:] # Remove batch dim

        if len(shape) == 2:
            return self.select2d(shape)
        elif len(shape) == 3:
            return self.select3d(shape)
        else:
            raise ValueError("Only 2D and 3D shapes are supported")


class MAD(Module):
    def init(self):
        self.embedder = SelectiveEmbedding()
        self.mixer = SelectiveMixer()

    def forward(self, inputs: dict):
        # Transform inputs into embedding spaces
        embeddings = []
        for input in inputs:
            embeddings.append(self.embedder(input))

        # Mix embeddings to create relationships
        self.mixer(embeddings)


class SelectiveEmbedding(Module):
    def init(self):
        self.selector = torch.nn.Linear(5, len(NETWORK_SELECTION))

        starting_network = nn.MultiheadAttention(1024, 4)

        self.networks = nn.ModuleList([starting_network])

    def generate_from_input(self, input: torch.Tensor):
        for net in NETWORK_SELECTION:


    def forward(self, x: torch.Tensor):
        """x can be any input tensor"""

        shape = x.shape


class SelectiveMixer(Module):
    def init(self): ...

    def forward(self, x: list[torch.Tensor]):
        """x is a list of embedding tensors"""

        # Mix embeddings to create relationships
        pass


class MADLayer(Module):
    def init(self):
        self.lienar = torch.nn.Linear(128, 128)

    def forward(self, features: torch.Tensor):
        features.shape
