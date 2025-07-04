import torch
import torch.nn as nn


class ResidualAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        residual: torch.Tensor = None,
        norm_first: torch.Tensor | bool = None,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # Normalize first
        normed = None
        if norm_first is True:
            normed = self.norm(query)
        elif norm_first:
            normed = self.norm(norm_first)

        if normed is not None:
            # Full self attn
            tokens = self.attn(
                normed,
                normed,
                normed,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )[0]
            tokens = residual + self.dropout(tokens)
            return tokens
        else:
            if key is None:
                key = query
            if value is None:
                value = query

            tokens = self.attn(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )[0]
            tokens = residual + self.dropout(tokens)
            tokens = self.norm(tokens)
            return tokens


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        cross=False,
        dim_feedforward=None,
        dropout=0.1,
        act=nn.ReLU,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 8 * d_model

        self.residual_self_attn = ResidualAttention(d_model, n_head, dropout=dropout)

        if cross:
            # Cross attention
            self.residual_cross_attn = ResidualAttention(
                d_model, n_head, dropout=dropout
            )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model)

    def with_pos_emb(self, x, pos_emb=None):
        return x if pos_emb is None else x + pos_emb

    def forward(
        self,
        tokens: torch.Tensor,
        norm_first: torch.Tensor | bool = False,
        pos: torch.Tensor = None,
        cross_tokens: torch.Tensor = None,
        cross_pos: torch.Tensor = None,
        key_mask: torch.Tensor = None,
    ):
        # If cross_pos is not None, we are wanting cross attention
        # Thus we are in the decoder
        cross = cross_pos is not None
        resolve_pos = cross_pos if cross else pos if pos is not None else None

        # Add positional encoding - from encoder or from pos
        q = k = self.with_pos_emb(tokens, resolve_pos)

        # Self attention
        tokens = self.residual_self_attn(
            q,
            k,
            tokens,
            residual=tokens,
            norm_first=norm_first,
            key_padding_mask=None if cross else key_mask,
        )

        # Cross attention
        if cross:
            tokens = self.residual_cross_attn(
                query=self.with_pos_emb(tokens, cross_pos),
                key=self.with_pos_emb(cross_tokens, pos),
                value=cross_tokens,
                residual=tokens,
                key_padding_mask=key_mask,
            )

        # MLP
        if norm_first:
            tokens += self.dropout(self.mlp(self.norm(tokens)))
        else:
            tokens = self.norm(tokens + self.dropout(self.mlp(tokens)))

        return tokens


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_encoder_layers=6, nhead=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, nhead) for _ in range(num_encoder_layers)]
        )

    def forward(self, tokens, pos, key_mask=None):
        for layer in self.layers:
            tokens = layer(tokens=tokens, pos=pos, key_mask=key_mask)
        return tokens


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_decoder_layers=6, nhead=8):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, cross=True)
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, tokens, cross_tokens, pos, cross_pos, key_mask=None):
        features = []
        for layer in self.layers:
            tokens = layer(
                tokens=tokens,
                cross_tokens=cross_tokens,
                pos=pos,
                cross_pos=cross_pos,
                key_mask=key_mask,
            )
            features.append(self.norm(tokens))

        return features
