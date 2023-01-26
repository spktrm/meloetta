import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model.utils import ResBlock

from meloetta.frameworks.nash_ketchum.model import config


class ValueHead(nn.Module):
    def __init__(self, config: config.ValueHeadConfig):
        super().__init__()
        self.resblock_stack = nn.ModuleList(
            [ResBlock(config.state_embedding_dim) for _ in range(config.num_resblocks)]
        )
        self.lin_out = nn.Linear(config.state_embedding_dim, 1)

    def forward(self, x: torch.Tensor):
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = self.lin_out(x)
        return x
