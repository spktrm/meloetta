import torch
import torch.nn as nn

from meloetta.frameworks.porygon.model import config


class ObservationDecoder(nn.Module):
    def __init__(self, gen: int, config: config.EncoderConfig):
        super().__init__()

        self.decode_private_entity = nn.Sequential(
            nn.Linear(config.private_encoder_config.entity_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1996),
        )
        self.decode_public_entity = nn.Sequential(
            nn.Linear(config.public_encoder_config.entity_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3611),
        )
        self.decode_public_scalars = nn.Sequential(
            nn.Linear(config.public_encoder_config.entity_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1268),
        )
        self.decode_weather = nn.Sequential(
            nn.Linear(config.weather_encoder_config.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 242),
        )
        self.decode_scalars = nn.Sequential(
            nn.Linear(config.scalar_encoder_config.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 34),
        )

    def forward(
        self,
        moves: torch.Tensor,
        private_entities_emb: torch.Tensor,
        public_entities_emb: torch.Tensor,
        public_scalar_emb: torch.Tensor,
        weather_emb: torch.Tensor,
        scalar_emb: torch.Tensor,
    ):
        return {
            "private_entity_pred": self.decode_private_entity(private_entities_emb),
            "public_entity_pred": self.decode_public_entity(public_entities_emb),
            "public_scalars_pred": self.decode_public_scalars(public_scalar_emb),
            "weather_pred": self.decode_weather(weather_emb),
            "scalar_pred": self.decode_scalars(scalar_emb),
        }
