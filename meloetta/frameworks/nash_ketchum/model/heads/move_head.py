import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    _legal_policy,
    gather_along_rows,
)


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
        self.proj_move = nn.Sequential(
            nn.Linear(config.key_dim, config.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.autoregressive_embedding_dim),
        )

    def forward(
        self,
        action_type_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        moves: torch.Tensor,
        move_mask: torch.Tensor,
    ):
        T, B, *_ = autoregressive_embedding.shape

        keys = self.key_fc(moves)

        query = self.query_fc(autoregressive_embedding)
        query = query.view(T, B, 1, 1, -1)

        move_logits = query @ keys.transpose(-2, -1)
        move_logits = move_logits.view(T, B, -1)

        move_mask = move_mask.view(T * B, -1)
        move_mask[move_mask.sum() == 0] = True
        move_mask = move_mask.view(T, B, -1)

        move_policy = _legal_policy(move_logits, move_mask)
        embedding_index = torch.multinomial(move_policy.view(T * B, -1), 1)
        move_index = embedding_index.view(T, B, -1)

        move_embedding = gather_along_rows(keys.flatten(0, -3), embedding_index, 1)
        move_embedding = move_embedding.view(T, B, -1)
        projected_move_embedding = self.proj_move(move_embedding)

        valid_indices = action_type_index == 0
        valid_indices = valid_indices.unsqueeze(-1)

        return (
            move_logits,
            move_policy,
            move_index,
            autoregressive_embedding + valid_indices * projected_move_embedding,
        )
