import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    Resblock,
    _legal_policy,
    _multinomial,
    BIAS,
)


class ActionTypeHead(nn.Module):
    def __init__(self, config: config.ActionTypeHeadConfig):
        super().__init__()

        self.resblocks = nn.Sequential(
            *[Resblock(config.residual_dim) for _ in range(1)]
        )
        self.action_fc = nn.Linear(config.residual_dim, config.num_action_types)
        self.action_type_emb = nn.Embedding(
            config.num_action_types, config.state_embedding_dim
        )

    def forward(
        self,
        state: torch.Tensor,
        action_type_mask: torch.Tensor,
        action_type_index: torch.Tensor = None,
    ):
        T, B, *_ = state.shape

        embedded_state = self.resblocks(state)
        action_type_logits = self.action_fc(embedded_state)

        action_type_policy = _legal_policy(action_type_logits, action_type_mask)
        action_type_logits = torch.where(action_type_mask, action_type_logits, BIAS)
        action_type_policy = action_type_policy.view(T * B, -1)

        if action_type_index is None:
            action_type_index = _multinomial(action_type_policy)

        action_type_policy = action_type_policy.view(T, B, -1)
        action_type_index = action_type_index.view(T, B)

        action_type_embedding = self.action_type_emb(action_type_index.long())
        autoregressive_embedding = embedded_state + action_type_embedding

        return (
            action_type_logits,
            action_type_policy,
            action_type_index,
            autoregressive_embedding,
        )
