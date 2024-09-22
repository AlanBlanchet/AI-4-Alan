import torch
import torch.nn as nn
from einops import rearrange

from ....registry.registry import REGISTER
from ...modules.attention import PositionalEncoding2D
from ...modules.mlp import MLP
from ..compat import Pretrained
from .config import DETRConfig


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        cross=False,
        dim_feedforward=None,
        dropout=0.1,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        if cross:
            # Cross attention
            self.cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_emb(self, x, pos_emb):
        return x if pos_emb is None else x + pos_emb

    def forward(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        cross_tokens: torch.Tensor = None,
        cross_pos: torch.Tensor = None,
    ):
        # If query_emb is not None, we are wanting cross attention
        # Thus we are in the decoder
        decode = cross_pos is not None

        # Add positional encoding - from encoder or from pos
        q = k = self.with_pos_emb(tokens, cross_pos if decode else pos)

        # Self attention
        res = self.self_attn(q, k, tokens)[0]
        tokens = tokens + self.dropout(res)
        tokens = self.norm1(tokens)

        # Cross attention
        if decode:
            res = self.cross_attn(
                query=self.with_pos_emb(tokens, cross_pos),
                key=self.with_pos_emb(cross_tokens, pos),
                value=cross_tokens,
            )[0]
            tokens = tokens + self.dropout(res)
            tokens = self.norm2(tokens)

        # MLP
        res = self.mlp(tokens)
        tokens = tokens + self.dropout(tokens)
        tokens = self.norm3(tokens)

        return tokens


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_encoder_layers=6, nhead=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, nhead) for _ in range(num_encoder_layers)]
        )

    def forward(self, tokens, pos):
        for layer in self.layers:
            tokens = layer(tokens=tokens, pos=pos)
        return tokens


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_decoder_layers=6, nhead=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, cross=True)
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, tokens, cross_tokens, pos, cross_pos):
        for layer in self.layers:
            tokens = layer(
                tokens=tokens, cross_tokens=cross_tokens, pos=pos, cross_pos=cross_pos
            )
        return cross_tokens


@REGISTER
class DETR(Pretrained):
    config = DETRConfig

    def __init__(self, config: DETRConfig):
        super().__init__()

        self.config = config

        # Backbone
        self.backbone = config.backbone.build()
        x = self.backbone.features(torch.randn(1, 3, 640, 640))
        C = x[-1].shape[1]
        self.input_proj = nn.Conv2d(C, config.hidden_dim, 1)

        # Encoder
        self.encoder = TransformerEncoder(config.hidden_dim)

        # Decoder
        self.decoder = TransformerDecoder(config.hidden_dim)
        self.query_emb = nn.Embedding(config.num_queries, config.hidden_dim)

        # Output
        self.clf = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox = MLP(
            config.hidden_dim, config.hidden_dim, output_dim=4, num_layers=3
        )

        x = torch.randn(1, 3, 640, 640)

        x = self(x)
        print(self)

        self.init_weights()

    def forward(self, x):
        # Backbone stages
        features = self.backbone.features(x)

        print(features[-1].shape)

        x = self.input_proj(features[-1])
        B, _, H, W = x.shape
        # Channel as last dim - merge resolution
        x = rearrange(x, "b c h w -> b (h w) c")

        # Positional encoding
        pos = PositionalEncoding2D.pos_emb(
            self.config.hidden_dim, H=H, W=W, device=x.device, dtype=x.dtype
        )  # [H * W, hidden_dim]

        # Send to encoder - output is called the memory
        memory = self.encoder(x, pos=pos)

        # Send to decoder - uses queries to attend to memory
        query_pos = self.query_emb.weight.repeat(B, 1, 1)
        tgt = torch.zeros_like(query_pos)
        hidden = self.decoder(
            tokens=tgt, cross_tokens=memory, pos=pos, cross_pos=query_pos
        )

        # Classification and bounding box regression
        clf = self.clf(hidden)
        bbox = self.bbox(hidden).sigmoid()

        return dict(logits=clf, boxes=bbox)

    def load_pretrained(self):
        model = torch.hub.load(
            "facebookresearch/detr", "detr_resnet50", pretrained=True
        )
        self.load_state(model)

    def init_weights(self):
        if self.config.pretrained:
            self.load_pretrained()
        else:
            ...
