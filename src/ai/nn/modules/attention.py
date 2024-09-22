import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        temperature: float = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    @staticmethod
    def pos_emb(
        d_model: int, H, W, temperature=10000, device=None, dtype=None, flat=True
    ):
        pos_emb = d_model // 2
        res_pos_emb = pos_emb // 2

        # Generate positions
        x_embed = torch.arange(W, dtype=dtype, device=device)
        y_embed = torch.arange(H, dtype=dtype, device=device)

        # Compute frequencies
        dim_t = torch.arange(res_pos_emb, dtype=dtype, device=device)
        dim_t = temperature ** (dim_t / res_pos_emb)

        # Encode positions
        pos_x = x_embed[None, :, None] / dim_t  # [1, W, res_pos_emb]
        pos_y = y_embed[:, None, None] / dim_t  # [H, 1, res_pos_emb]
        pos_x = pos_x.repeat(H, 1, 1)
        pos_y = pos_y.repeat(1, W, 1)

        # Encode sin/cos
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        # [H, W, pos_emb]

        # Merge positions
        pos = torch.cat((pos_y, pos_x), dim=-1)  # [H, W, d_model]

        if flat:
            pos = pos.flatten(0, 1)

        return pos

    def forward(self, x):
        _, H, W, _ = x.shape
        pos = self.pos_emb(self.d_model, H=H, W=W, temperature=self.temperature)
        return x + pos
