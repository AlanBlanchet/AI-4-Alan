import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ....registry.registry import REGISTER
from ...compat.backbone import IsBackboneMixin
from ..detr.detr import TransformerLayer
from .config import ViTConfig


@REGISTER
class ViT(IsBackboneMixin):
    config: ViTConfig = ViTConfig

    def __init__(self, config: ViTConfig):
        super().__init__(config)

        self.ps = self.config.patch_size
        emb_size = config.hidden_size * config.num_channels

        # Patch projection (we can use nn.Linear as well)
        # TODO config Linear proj
        self.patch_proj = nn.Conv2d(
            config.num_channels, emb_size, kernel_size=self.ps, stride=self.ps
        )

        # Cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        if config.fixed_size is None:
            # TODO flexible pos
            raise NotImplementedError("Flexible positional encoding not implemented")
        else:
            h, w = config.fixed_size
            h //= self.ps
            w //= self.ps
            self.pos = nn.Parameter(torch.randn(h * w + 1, emb_size) * 0.02)

        self.norm = nn.LayerNorm(emb_size, eps=1e-6)
        # self.fc_norm = nn.LayerNorm(emb_size, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    emb_size,
                    config.n_head,
                    dim_feedforward=emb_size * 4,
                    dropout=0.0,
                    act=nn.GELU,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.head = nn.Linear(emb_size, config.num_classes)

    def features(self, x):
        # Make sure the image is divisible by the patch size
        h, w = x.shape[-2:]
        rh, rw = int(round(h / self.ps) * self.ps), int(round(w / self.ps) * self.ps)
        x = F.interpolate(x, size=(rh, rw))

        # Create features
        x = self.patch_proj(x)
        # Rearrange into patches
        x = rearrange(x, "b e h w -> b (h w) e")
        # Adapt cls token
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        # Put inside the features
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos

        for block in self.blocks:
            x = block(x, norm_first=True)

        x = self.norm(x)

        return x

    def forward(self, x):
        # Extract features
        x = self.features(x)

        # Take the cls token
        x = x[:, 0]

        # Normalize
        # x = self.fc_norm(x)

        # Classify
        x = self.head(x)

        return x
