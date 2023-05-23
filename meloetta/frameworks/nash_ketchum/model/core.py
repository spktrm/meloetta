import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model.interfaces import EncoderOutput

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import Resblock, VectorMerge


class Core(nn.Module):
    def __init__(self, config: config.CoreConfig):
        super().__init__()
        self.config = config

        self.merge = VectorMerge(
            {
                "pokemon_embedding": config.side_encoder_dim,
                "boosts": config.side_encoder_dim,
                "volatiles": config.side_encoder_dim,
                "side_conditions": config.side_encoder_dim,
                "weather_emb": config.side_encoder_dim,
                "scalar_emb": config.side_encoder_dim,
            },
            output_size=config.hidden_dim,
        )
        self.resblocks = nn.ModuleList(
            [Resblock(config.hidden_dim) for _ in range(config.num_layers)]
        )

    def forward(self, encoder_output: EncoderOutput):
        state_embedding = self.merge(
            {
                "pokemon_embedding": encoder_output.pokemon_embedding,
                "boosts": encoder_output.boosts,
                "volatiles": encoder_output.volatiles,
                "side_conditions": encoder_output.side_conditions,
                "weather_emb": encoder_output.weather_emb,
                "scalar_emb": encoder_output.scalar_emb,
            }
        )
        for resblock in self.resblocks:
            state_embedding = resblock(state_embedding)

        return state_embedding
