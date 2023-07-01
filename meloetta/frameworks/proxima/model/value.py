import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.proxima.model.utils import Resblock, MLP

from meloetta.frameworks.proxima.model import config


class ValueHead(nn.Module):
    def __init__(self, config: config.ValueHeadConfig):
        super().__init__()
        self.resblock_stack = nn.ModuleList(
            [Resblock(config.hidden_dim) for _ in range(config.num_resblocks)]
        )
        self.lin_out = MLP([config.hidden_dim, 1])

    def forward(self, x: torch.Tensor):
        for resblock in self.resblock_stack:
            x = resblock(x)
        return F.tanh(self.lin_out(x))
