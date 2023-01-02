import torch

from torch import nn


class PokedexEmbedding(nn.Module):
    def __init__(self, gen: int):
        super().__init__()

        embeddings = torch.load(f"pretrained")
        self.emb = nn.Embedding.from_pretrained(embeddings)
