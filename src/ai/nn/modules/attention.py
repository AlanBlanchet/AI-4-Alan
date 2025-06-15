import math

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

    def forward(self, x: torch.Tensor):
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
        normalize: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize

    @staticmethod
    def pos_emb_from_mask(
        d_model: int,
        mask: torch.Tensor,
        temperature=10000,
        device=None,
        dtype=None,
        flat=True,
        normalize: bool | int = False,
    ):
        pos_emb = d_model // 2
        res_pos_emb = pos_emb // 2

        H, W = mask.shape[1:]

        # Generate positions
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if normalize:
            eps = 1e-6
            # Normalize by a factor or by the resolution
            if isinstance(normalize, bool):
                h, w = y_embed[:, -1:, :], x_embed[:, :, -1:]
            else:
                h, w = normalize, normalize

            y_embed = y_embed / (h + eps) * 2 * math.pi
            x_embed = x_embed / (w + eps) * 2 * math.pi

        # Compute frequencies for the embedding dimension
        dim_t = torch.arange(res_pos_emb, dtype=dtype, device=device) * 2
        dim_t = temperature ** (dim_t / pos_emb)

        # Encode positions
        pos_x = x_embed[:, :, :, None] / dim_t  # [1, W, res_pos_emb]
        pos_y = y_embed[:, :, :, None] / dim_t  # [H, 1, res_pos_emb]
        # pos_x = pos_x.repeat(H, 1, 1)  # [H, W, res_pos_emb]
        # pos_y = pos_y.repeat(1, W, 1)

        # Encode sin/cos
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        # [H, W, pos_emb]

        # Merge positions
        pos = torch.cat((pos_y, pos_x), dim=-1)  # [H, W, d_model]

        if flat:
            pos = pos.flatten(0, 1)

        return pos

    @staticmethod
    def pos_emb(
        d_model: int,
        H,
        W,
        temperature=10000,
        device=None,
        dtype=None,
        flat=True,
        normalize: bool | int = False,
    ):
        pos_emb = d_model // 2
        res_pos_emb = pos_emb // 2

        # Generate positions
        x_embed = torch.arange(W, dtype=dtype, device=device) + 1
        y_embed = torch.arange(H, dtype=dtype, device=device) + 1

        if normalize:
            eps = 1e-6
            # Normalize by a factor or by the resolution
            if isinstance(normalize, int):
                h, w = normalize, normalize
            else:
                h, w = H, W
            y_embed = y_embed / (h + eps) * 2 * math.pi
            x_embed = x_embed / (w + eps) * 2 * math.pi

        # Compute frequencies for the embedding dimension
        dim_t = torch.arange(res_pos_emb, dtype=dtype, device=device) * 2
        dim_t = temperature ** (dim_t / pos_emb)

        # Encode positions
        pos_x = x_embed[None, :, None] / dim_t  # [1, W, res_pos_emb]
        pos_y = y_embed[:, None, None] / dim_t  # [H, 1, res_pos_emb]
        pos_x = pos_x.repeat(H, 1, 1)  # [H, W, res_pos_emb]
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
        pos = self.pos_emb(
            self.d_model,
            H=H,
            W=W,
            temperature=self.temperature,
            normalize=self.normalize,
        )
        return x + pos
