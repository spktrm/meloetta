import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    _legal_policy,
    gather_along_rows,
    _multinomial,
    MLP,
    BIAS,
)


class MoveHead(nn.Module):
    def __init__(self, config: config.MoveHeadConfig):
        super().__init__()

        # self.size = config.entity_embedding_dim**0.5
        self.key_fc = MLP(
            [
                config.entity_embedding_dim,
                config.entity_embedding_dim,
                config.key_dim,
            ]
        )
        self.query_fc = MLP(
            [
                config.autoregressive_embedding_dim,
                config.autoregressive_embedding_dim,
                config.key_dim,
            ]
        )
        self.proj_move = MLP(
            [
                config.key_dim,
                config.autoregressive_embedding_dim,
                config.autoregressive_embedding_dim,
            ]
        )

    def forward(
        self,
        action_type_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        moves: torch.Tensor,
        move_mask: torch.Tensor,
        move_index: torch.Tensor = None,
    ):
        T, B, *_ = autoregressive_embedding.shape

        keys = self.key_fc(moves)
        query = self.query_fc(autoregressive_embedding)
        query = query.view(T, B, 1, -1)

        move_logits = query @ keys.transpose(-2, -1)
        # move_logits = move_logits / self.size
        move_logits = move_logits.view(T, B, -1)

        move_mask = move_mask.view(T * B, -1)
        move_mask[move_mask.sum() == 0] = True
        move_mask = move_mask.view(T, B, -1)

        move_policy = _legal_policy(move_logits, move_mask)
        move_logits = torch.where(move_mask, move_logits, BIAS)
        move_policy = move_policy.view(T * B, -1)

        if move_index is None:
            embedding_index = _multinomial(move_policy)
            move_index = embedding_index.view(T, B, -1)

        embedding_index = move_index.view(T * B, -1)
        move_policy = move_policy.view(T, B, -1)

        move_embedding = gather_along_rows(keys.flatten(0, -3), embedding_index, 1)
        move_embedding = move_embedding.view(T, B, -1)
        projected_move_embedding = self.proj_move(move_embedding)

        valid_indices = action_type_index == 0
        valid_indices = valid_indices.unsqueeze(-1)
        autoregressive_embedding = (
            autoregressive_embedding + valid_indices * projected_move_embedding
        )

        return (
            move_logits,
            move_policy,
            move_index,
            autoregressive_embedding,
        )
