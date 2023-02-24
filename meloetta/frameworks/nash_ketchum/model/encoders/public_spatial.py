import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.data import VOLATILES


class ResidualLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.res(x))


class PublicSpatialEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PublicEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.down = nn.Linear(
            config.entity_embedding_dim, config.entity_embedding_dim // 2
        )

        self.resblock_stacks = nn.Sequential(
            *[ResidualLayer(config.entity_embedding_dim // 2) for _ in range(2)]
        )

        self.scalar = nn.Linear(
            config.scalar_embedding_dim, config.entity_embedding_dim // 2
        )

        self.boosts_onehot = nn.Sequential(
            nn.Embedding.from_pretrained(torch.eye(13)),
            nn.Flatten(-2),
        )
        self.embed_boosts_volatiles = nn.Linear(
            13 * 8 + len(VOLATILES), config.entity_embedding_dim // 2
        )
        self.spatial_out = nn.Sequential(
            *[
                nn.ReLU(),
                nn.Linear(
                    (config.entity_embedding_dim // 2) * 4 * 2,
                    config.entity_embedding_dim,
                ),
            ]
        )

    def forward(
        self,
        entity_embeddings: torch.Tensor,
        mask: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
        scalar_emb: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings = self.down(entity_embeddings).permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        active_mask = mask[..., : self.n_active]
        active = entity_embeddings[..., : self.n_active].sum(-1, keepdim=True)
        active = active / active_mask.sum(-1, keepdim=True).clamp(min=1)

        reserve_mask = mask[..., self.n_active :]
        reserve = entity_embeddings[..., self.n_active :].sum(-1, keepdim=True)
        reserve = reserve / reserve_mask.sum(-1, keepdim=True).clamp(min=1)

        boosts_volatiles = torch.cat((self.boosts_onehot(boosts), volatiles), dim=-1)
        boosts_volatiles = self.embed_boosts_volatiles(boosts_volatiles)
        boosts_volatiles = torch.flatten(boosts_volatiles, 0, 1)
        boosts_volatiles = F.relu(boosts_volatiles.permute(0, 3, 1, 2))

        scalar_emb = torch.flatten(scalar_emb, 0, 1)
        scalar_emb = self.scalar(scalar_emb)
        scalar_emb = scalar_emb.unsqueeze(1)
        scalar_emb = F.relu(scalar_emb.permute(0, 3, 2, 1))

        spatial_emb = torch.cat((active, reserve, boosts_volatiles, scalar_emb), dim=-1)
        spatial_emb = F.relu(self.resblock_stacks(spatial_emb))
        spatial_emb = torch.flatten(spatial_emb, 1)
        spatial_emb = F.relu(self.spatial_out(spatial_emb))

        return spatial_emb
