import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model.utils import Resblock

from meloetta.frameworks.nash_ketchum.model import config


class ValueHead(nn.Module):
    def __init__(self, config: config.ValueHeadConfig):
        super().__init__()
        self.lin_in = nn.Linear(config.state_embedding_dim, config.hidden_dim)

        self.resblock_stack = nn.ModuleList(
            [
                Resblock(config.hidden_dim, use_layer_norm=False)
                for _ in range(config.num_resblocks)
            ]
        )
        self.lin_out = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.lin_in(x)
        for resblock in self.resblock_stack:
            x = resblock(x)
        return self.lin_out(x)
