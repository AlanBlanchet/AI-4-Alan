import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area

from ....registry.registry import REGISTER
from ...compat.pretrained import Pretrained
from ...fusion.fuse import FrozenBatchNorm2d
from ...modules.attention import PositionalEncoding2D
from ...modules.mlp import MLP
from .config import DETRConfig


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


# TODO refactor for all boxes
def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# TODO refactor for all boxes
def box_xyxy_to_cxcywh(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# TODO refactor for all boxes
def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# TODO refactor for all boxes
# modified from torchvision to also return the union
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


@REGISTER
class DETR(Pretrained):
    config: DETRConfig = DETRConfig

    def __init__(self, config: DETRConfig):
        super().__init__(config)

        self.config = config

        # Backbone
        self.backbone = config.backbone.build()
        # TODO get out_channels from backbone directly
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

    def init_weights(self, *forward_args):
        self.backbone.init_weights(*forward_args[:1])
        super().init_weights(*forward_args)

        if self.config.freeze_norm:
            self.replace_module(nn.BatchNorm2d, FrozenBatchNorm2d)

    def rearrange(self, x):
        return rearrange(x, "... c h w -> ... (h w) c")

    def forward(self, image, mask):
        # In attention, a True value in a mask means that the value is ignored
        mask = ~mask

        # Backbone stages
        features = self.backbone.features(image)
        x = features[-1]

        # Make it fit the out_features
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        B, _, H, W = x.shape

        x = self.input_proj(x)
        # Channel as last dim
        x = self.rearrange(x)

        # Positional encoding
        pos = PositionalEncoding2D.pos_emb_from_mask(
            self.config.hidden_dim,
            mask=~mask,
            device=x.device,
            dtype=x.dtype,
            flat=False,
            normalize=True,
        ).permute(0, 3, 1, 2)  # [B, C, H, W]
        # merge resolution
        pos = self.rearrange(pos)
        mask = mask.flatten(1)
        # Send to encoder - output is called the memory
        memory = self.encoder(x, pos=pos, key_mask=mask)

        # Send to decoder - uses queries to attend to memory
        query_pos = self.query_emb.weight.repeat(B, 1, 1)
        tgt = torch.zeros_like(query_pos)
        hidden = self.decoder(
            tokens=tgt, cross_tokens=memory, key_mask=mask, pos=pos, cross_pos=query_pos
        )[-1]

        # Classification and bounding box regression
        logits = self.clf(hidden)
        bbox = self.bbox(hidden).sigmoid()

        return dict(logits=logits, bbox=bbox)

    @staticmethod
    def postprocess(out, batch):
        logits = out["logits"]
        bbox = out["bbox"]
        size = batch["image_size"]

        # Remove background prediction
        probas = logits.softmax(-1)[..., :-1]

        # Get the top class
        scores, labels = probas.max(-1)
        bbox = box_cxcywh_to_xyxy(bbox) * size.tile(2)[:, None, :]

        return dict(scores=scores, labels=labels, bbox=bbox)

    def compute_loss(
        self, out: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Any:
        # Compute the matching
        logits = out["logits"]  # (B, N, C)
        bbox = out["bbox"]  # (B, N, 4)

        # Extract shapes
        B, N = logits.shape[:2]

        # Flatten preds
        logits = logits.flatten(0, 1).softmax(-1)
        bbox = bbox.flatten(0, 1)

        # Flatten targets
        gt_bbox_mask = target["bbox_mask"]
        gt_num_boxes_per_input = gt_bbox_mask.sum(-1)  # (B,)
        labels = target["labels"][gt_bbox_mask].flatten()  # (num_boxes,)
        gt_bbox = target["bbox"][gt_bbox_mask].flatten(1)  # (num_boxes, 4)
        num_boxes = gt_num_boxes_per_input.sum()

        # Hungarian matching
        with torch.no_grad():
            # Compute cost
            cost_class = -logits[..., labels]
            cost_bbox = torch.cdist(bbox, gt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(bbox), box_cxcywh_to_xyxy(gt_bbox)
            )

            # Cost matrix
            C = (
                self.config.class_coef * cost_class
                + self.config.bbox_coef * cost_bbox
                + self.config.giou_coef * cost_giou
            )
            C = C.view(B, N, -1).cpu()

            # Heart of hungarian matching
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(gt_num_boxes_per_input.tolist(), dim=-1))
            ]

        gt_num_boxes_per_input_cumsum = gt_num_boxes_per_input.cumsum(0)

        # Get matches in range [0 - num_boxes]
        matches_idx = torch.as_tensor(
            [
                (
                    s + N * i,
                    t + (gt_num_boxes_per_input_cumsum[i - 1] if i > 0 else 0),
                )  # Index of prediction in B*N and target in B*T
                for i, (src, tgt) in enumerate(indices)
                for s, t in zip(src, tgt)
            ]
        )  # (B*N, 2)

        # We now have the matches and can compute all specific losses
        real_logits = logits[matches_idx[:, 0]]
        real_labels = labels[matches_idx[:, 1]]
        real_bbox = bbox[matches_idx[:, 0]]
        real_gt_bbox = gt_bbox[matches_idx[:, 1]]

        # Labels
        ce_loss = F.cross_entropy(real_logits, real_labels)

        # Boxes
        bbox_loss = F.l1_loss(real_bbox, real_gt_bbox, reduction="none")
        bbox_loss = bbox_loss.sum() / num_boxes

        # GIoU
        giou_loss = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(real_bbox),
                box_cxcywh_to_xyxy(real_gt_bbox),
            )
        )
        giou_loss = giou_loss.sum() / num_boxes

        # Final loss
        loss = (
            self.config.class_coef * ce_loss
            + self.config.bbox_coef * bbox_loss
            + self.config.giou_coef * giou_loss
        )

        return dict(
            loss=loss, ce_loss=ce_loss, bbox_loss=bbox_loss, giou_loss=giou_loss
        )
