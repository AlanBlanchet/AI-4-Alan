from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from ..utils.encode import Encoder


class DQNPolicy(nn.Module):
    def __init__(
        self,
        in_shape: int | np.ndarray,
        out_dim: int,
        last_layer: Literal["lstm", "linear"] = "linear",
        history: int = 0,
        hidden_dim: int = 256,
        duel: bool = False,
    ):
        super().__init__()

        self.last_layer = last_layer
        self.duel = duel
        self.is_lstm = last_layer == "lstm"
        self.history = history

        # Do not use multiple channels if we are using lstm
        conv_stacks = 1 if self.is_lstm else history

        self.encoder = Encoder(in_shape, history=conv_stacks)
        out = self.encoder.out

        encoder_dim = out
        if self.is_lstm:
            self.lstm = nn.LSTM(out, hidden_dim, batch_first=True)
            encoder_dim = hidden_dim

        self.l1 = nn.Linear(encoder_dim, out_dim)

        if self.duel:
            self.v = nn.Linear(encoder_dim, 1)

        self.act = nn.ReLU()

    def forward(self, x, h: torch.Tensor = None, c: torch.Tensor = None):
        x = self.encoder(x)
        x = self.act(x)

        # Optionally use an LSTM
        if self.is_lstm:
            h = h[..., 0, :].unsqueeze(dim=0).detach()
            c = c[..., 0, :].unsqueeze(dim=0).detach()

            # h = rearrange(h, "b h n -> h b n")
            # c = rearrange(c, "b h n -> h b n")
            # print(x.shape, h.shape, c.shape)
            x, (h, c) = self.lstm(x, (h, c))
            x = x.squeeze(dim=1)
            # x = x.squeeze(dim=0)
            # h = h.squeeze(dim=0)
            # c = c.squeeze(dim=0)
            # print("h", h.shape)
            # h = rearrange(h, "h b n -> b h n")
            # c = rearrange(c, "h b n -> b h n")
            # print("out", x.shape, h.shape, c.shape)
            x = self.act(x)

        if self.duel:
            value: torch.Tensor = self.v(x)
            adv: torch.Tensor = self.l1(x)
            # value = value.expand_as(adv)
            x = value + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            x = self.l1(x)

        return x, h, c
