import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model import config


class ResidualLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.res(x))


class PrivateSpatialEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PrivateEncoderConfig):
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

        self.spatial_out = nn.Linear(
            (config.entity_embedding_dim // 2) * 2,
            config.entity_embedding_dim,
        )

    def forward(
        self, entity_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask.transpose(1, 2)
        entity_embeddings = self.down(entity_embeddings).transpose(1, 2)

        active_mask = mask[..., : self.n_active]
        active = entity_embeddings[..., : self.n_active].sum(-1, keepdim=True)
        active = active / active_mask.sum(-1, keepdim=True).clamp(min=1)

        reserve_mask = mask[..., self.n_active :]
        reserve = entity_embeddings[..., self.n_active :].sum(-1, keepdim=True)
        reserve = reserve / reserve_mask.sum(-1, keepdim=True).clamp(min=1)

        spatial_emb = torch.cat((active, reserve), dim=-1)
        spatial_emb = F.relu(self.resblock_stacks(spatial_emb))
        spatial_emb = torch.flatten(spatial_emb, 1)
        spatial_emb = F.relu(self.spatial_out(spatial_emb))

        return spatial_emb
