import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.mewzero.model import config
from meloetta.data import VOLATILES


class ResidualLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.res(x))


class PublicSpatialEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PublicEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.project = nn.Linear(
            config.entity_embedding_dim * 2, config.entity_embedding_dim
        )

        self.resblock_stacks = nn.Sequential(
            *[ResidualLayer(2 * config.entity_embedding_dim) for _ in range(2)]
        )

        self.boosts_onehot = nn.Sequential(
            nn.Embedding.from_pretrained(torch.eye(13)),
            nn.Flatten(-2),
        )
        self.embed_boosts_volatiles = nn.Linear(
            13 * 8 + len(VOLATILES), config.entity_embedding_dim
        )
        self.spatial_out = nn.Linear(
            2 * config.entity_embedding_dim, 2 * config.entity_embedding_dim
        )

    def forward(
        self,
        entity_embeddings: torch.Tensor,
        mask: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
        scalar_emb: torch.Tensor,
    ) -> torch.Tensor:
        active_mask = mask[..., : self.n_active, :]
        active = entity_embeddings[..., : self.n_active, :].sum(2, keepdim=True)
        active = active / active_mask.sum(2, keepdim=True).clamp(min=1)

        reserve_mask = mask[..., self.n_active :, :]
        reserve = entity_embeddings[..., self.n_active :, :].sum(2, keepdim=True)
        reserve = reserve / reserve_mask.sum(2, keepdim=True).clamp(min=1)

        boosts_volatiles = torch.cat((self.boosts_onehot(boosts), volatiles), dim=-1)
        boosts_volatiles = torch.flatten(
            self.embed_boosts_volatiles(boosts_volatiles), 0, 1
        )

        scalar_emb = torch.flatten(scalar_emb, 0, 1)
        scalar_emb = scalar_emb.unsqueeze(2)

        spatial_emb = torch.cat((active + boosts_volatiles, reserve), dim=-1)
        spatial_emb = self.project(spatial_emb) + scalar_emb
        spatial_emb = torch.flatten(spatial_emb, 1)

        spatial_emb = self.resblock_stacks(spatial_emb)
        spatial_emb = self.spatial_out(spatial_emb)

        return spatial_emb
