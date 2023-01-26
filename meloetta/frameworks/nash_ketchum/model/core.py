import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model.utils import ResBlock

from meloetta.frameworks.nash_ketchum.model import config


class ResNetCore(nn.Module):
    def __init__(self, config: config.ResNetCoreConfig):
        super().__init__()
        self.lin_in = nn.Linear(
            config.raw_state_embedding_dim, config.projected_state_embedding_dim
        )
        self.resblock_stack = nn.ModuleList(
            [
                ResBlock(config.projected_state_embedding_dim)
                for _ in range(config.num_resblocks)
            ]
        )

    def forward(self, state_embedding: torch.Tensor):
        state_embedding = self.lin_in(state_embedding)
        for resblock in self.resblock_stack:
            state_embedding = resblock(state_embedding)
        return state_embedding
