import torch.nn as nn

from collections import OrderedDict

from meloetta.actors.types import TensorDict
from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import VectorResblock, VectorMerge


class Core(nn.Module):
    def __init__(self, config: config.CoreConfig):
        super().__init__()
        self.config = config

        self.merge = VectorMerge(
            OrderedDict(
                side_embs=config.side_encoder_dim,
                scalar_emb=config.side_encoder_dim,
                # hist_emb=config.side_encoder_dim,
            ),
            output_size=config.hidden_dim,
        )
        self.resblocks = nn.ModuleList(
            [VectorResblock(config.hidden_dim) for _ in range(config.num_layers)]
        )

    def forward(self, encoder_output: TensorDict):
        state_embedding = self.merge(
            OrderedDict(
                side_embs=encoder_output["pokemon_embedding"],
                scalar_emb=encoder_output["scalar_embedding"],
                # hist_emb=encoder_output["history_embedding"],
            )
        )
        for resblock in self.resblocks:
            state_embedding = resblock(state_embedding)
        return state_embedding
