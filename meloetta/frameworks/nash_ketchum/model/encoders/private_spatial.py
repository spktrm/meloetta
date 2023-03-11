import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model import config


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


class PrivateSpatialEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PrivateEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.project = nn.Linear(
            2 * config.entity_embedding_dim, config.entity_embedding_dim
        )

        self.resblock_stacks = nn.Sequential(
            *[ResidualLayer(config.entity_embedding_dim) for _ in range(4)]
        )

        self.spatial_out = nn.Linear(
            config.entity_embedding_dim, config.entity_embedding_dim
        )

    def forward(
        self, entity_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        active_mask = mask[..., : self.n_active, :]
        active = entity_embeddings[..., : self.n_active, :].sum(1, keepdim=True)
        active = active / active_mask.sum(1, keepdim=True).clamp(min=1)

        reserve_mask = mask[..., self.n_active :, :]
        reserve = entity_embeddings[..., self.n_active :, :].sum(1, keepdim=True)
        reserve = reserve / reserve_mask.sum(1, keepdim=True).clamp(min=1)

        spatial_emb = torch.cat((active, reserve), dim=-1)
        spatial_emb = self.project(spatial_emb)
        spatial_emb = self.resblock_stacks(spatial_emb)
        spatial_emb = self.spatial_out(spatial_emb)

        return spatial_emb
