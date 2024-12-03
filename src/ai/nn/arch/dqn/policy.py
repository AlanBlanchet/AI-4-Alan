from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ...compat.module import Module
from ...modules.encode import Encoder


class DQNPolicy(Module):
    in_shape: int | np.ndarray | tuple[int, ...]
    out_dim: int
    last_layer: Literal["lstm", "linear"] = "linear"
    history: int = 0
    hidden_dim: int = 256
    dual: bool = False
    act: nn.Module = nn.ReLU()
    dims: list[int, int] = [16, 32]

    def init(self):
        super().init()

        conv_stacks = 1 if self.is_lstm else self.history
        self.encoder = Encoder(
            in_shape=self.in_shape, history=conv_stacks, dims=self.dims
        )

        encoder_dim = self.encoder.out
        if self.is_lstm:
            self.lstm = nn.LSTM(encoder_dim, self.hidden_dim, batch_first=True)
            encoder_dim = self.hidden_dim

        self.l1 = nn.Linear(encoder_dim, self.out_dim)

        if self.dual:
            self.v = nn.Linear(encoder_dim, 1)

    @property
    def is_lstm(self):
        return self.last_layer == "lstm"

    def forward(self, x, hx: tuple[torch.Tensor, torch.Tensor] = None):
        # if self.history > 0 and not self.is_lstm and x.ndim == 5:
        #     x = rearrange(x, "b s c ... -> b (s c) ...")
        if not self.is_lstm:
            # We are stacking the sequence & channel dimensions
            x = rearrange(x, "b s c ... -> b (s c) ...")

        x = self.encoder(x)
        x = self.act(x)

        # Optionally use an LSTM
        if self.is_lstm:
            x, hx = self.lstm(x, hx)
            x = self.act(x[:, -1])

        if self.dual:
            value: torch.Tensor = self.v(x)
            adv: torch.Tensor = self.l1(x)
            x = value + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            x = self.l1(x)

        return x, hx
