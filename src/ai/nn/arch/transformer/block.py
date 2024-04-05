import torch.nn as nn

from .attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()

        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.ff(x)
        return self.dropout(self.norm2(forward + x))
