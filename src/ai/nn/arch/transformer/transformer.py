import torch.nn as nn

from ....registry import REGISTER
from .decoder import Decoder
from .encoder import Encoder


@REGISTER
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
        )

    def forward(self, src, target, src_mask, trg_mask):
        N, L = target.shape
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_mask = torch.tril(torch.ones((L, L))).expand(N, 1, L, L)
        enc_src = self.encoder(src, src_mask)
        return self.decoder(target, enc_src, src_mask, trg_mask)
