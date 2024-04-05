import torch
from datasets import load_dataset
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads

        self.head_splits = embed_size // heads

        self.keys = nn.Linear(self.head_splits, self.head_splits, bias=False)
        self.queries = nn.Linear(self.head_splits, self.head_splits, bias=False)
        self.values = nn.Linear(self.head_splits, self.head_splits, bias=False)
        self.out = nn.Linear(self.head_splits * heads, embed_size)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ):
        N = query.shape[0]
        Q, K, V = query.shape[1], key.shape[1], value.shape[1]

        queries = query.reshape(N, Q, self.heads, self.head_splits)
        keys = key.reshape(N, K, self.heads, self.head_splits)
        values = value.reshape(N, V, self.heads, self.head_splits)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = (energy / (self.embed_size ** (1 / 2))).softmax(dim=-1)

        x = torch.einsum("nhql,nlhd->nqhd", [attention, values]).flatten(2)

        return self.out(x)


if __name__ == "__main__":
    from tiktoken import get_encoding

    dataset = load_dataset("nampdn-ai/tiny-lessons")
    train = dataset["train"]

    tokenizer = get_encoding("cl100k_base")

    print("---")
    print(train["text"][0])
    print("---")
    print(tokenizer.encode(train["text"][0]))
    print("---")
    print(tokenizer.n_vocab)
    print(tokenizer.special_tokens_set)
