import torch.nn as nn

from collections import OrderedDict

from meloetta.actors.types import TensorDict
from meloetta.frameworks.nash_ketchum.modelv2 import config
from meloetta.frameworks.nash_ketchum.modelv2.utils import VectorResblock, VectorMerge


class Core(nn.Module):
    def __init__(self, config: config.CoreConfig):
        super().__init__()
        self.config = config

        self.merge = VectorMerge(
            OrderedDict(
                private_embedding=config.side_encoder_dim,
                public_side=config.side_encoder_dim,
                scalar_embedding=config.side_encoder_dim,
            ),
            output_size=config.hidden_dim,
        )
        self.resblocks = nn.ModuleList(
            [VectorResblock(config.hidden_dim) for _ in range(config.num_layers)]
        )

    def forward(self, encoder_output: TensorDict):
        state_embedding = self.merge(
            OrderedDict(
                private_embedding=encoder_output["private_embedding"],
                public_side=encoder_output["public_side"],
                scalar_embedding=encoder_output["scalar_embedding"],
            )
        )
        for resblock in self.resblocks:
            state_embedding = resblock(state_embedding)
        return state_embedding
