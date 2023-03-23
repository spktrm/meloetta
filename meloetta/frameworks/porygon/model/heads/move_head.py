import torch
import torch.nn as nn

from meloetta.frameworks.porygon.model import config


class MoveHead(nn.Module):
    def __init__(self, config: config.MoveHeadConfig):
        super().__init__()

        self.key_fc = nn.Sequential(
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.key_dim),
        )
        self.query_fc = nn.Sequential(
            nn.Linear(config.autoregressive_embedding_dim, config.query_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.query_hidden_dim, config.key_dim),
        )

    def forward(
        self,
        state_emb: torch.Tensor,
        moves: torch.Tensor,
    ):
        T, B, *_ = moves.shape

        keys = self.key_fc(moves)

        query = self.query_fc(state_emb)
        query = query.view(T, B, 1, -1)

        move_logits = query @ keys.transpose(-2, -1)
        move_logits = move_logits.view(T, B, -1)

        return move_logits, keys
