import torch
import torch.nn as nn

from meloetta.frameworks.mewzero.model import config
from meloetta.frameworks.mewzero.model.utils import (
    _legal_policy,
    gather_along_rows,
)


class MaxMoveHead(nn.Module):
    def __init__(self, config: config.MaxMoveHeadConfig):
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
        max_moves: torch.Tensor,
        max_move_mask: torch.Tensor,
    ):
        T, B, *_ = autoregressive_embedding.shape

        query = self.query_fc(autoregressive_embedding)
        query = query.view(T, B, 1, 1, -1)
        keys = self.key_fc(max_moves)

        max_move_logits = query @ keys.transpose(-2, -1)
        max_move_logits = max_move_logits.view(T, B, -1)

        max_move_mask = max_move_mask.view(T * B, -1)
        max_move_mask[max_move_mask.sum() == 0] = True
        max_move_mask = max_move_mask.view(T, B, -1)

        max_move_policy = _legal_policy(max_move_logits, max_move_mask)
        embedding_index = torch.multinomial(max_move_policy.view(T * B, -1), 1)
        max_move_index = embedding_index.view(T, B, -1)

        move_embedding = gather_along_rows(keys.flatten(0, -3), embedding_index, 1)
        move_embedding = move_embedding.view(T, B, -1)
        projected_move_embedding = self.proj_move(move_embedding)

        valid_indices = action_type_index == 0
        valid_indices = valid_indices.unsqueeze(-1)

        autoregressive_embedding = (
            autoregressive_embedding + valid_indices * projected_move_embedding
        )

        return (
            max_move_logits,
            max_move_policy,
            max_move_index,
        )
