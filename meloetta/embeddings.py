import torch

from torch import nn

from typing import List


class PokedexEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"meloetta/pretrained/gen{gen}/pokedex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class AbilityEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"meloetta/pretrained/gen{gen}/abilitydex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class MoveEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"meloetta/pretrained/gen{gen}/movedex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class ItemEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"meloetta/pretrained/gen{gen}/itemdex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)
