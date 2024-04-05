import torch
import torch.nn as nn

from .block import Block


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()

        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Block(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, S = x.shape
        position = torch.arange(S, device=x.device).expand(N, S)
        x = self.dropout(self.token_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x
