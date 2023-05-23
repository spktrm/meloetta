import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.utils import Resblock
from meloetta.frameworks.porygon.model.utils import _legal_policy


class ActionTypeHead(nn.Module):
    def __init__(self, config: config.ActionTypeHeadConfig):
        super().__init__()

        self.num_action_types = config.num_action_types
        self.project = nn.Linear(config.state_embedding_dim, config.residual_dim)
        self.resblocks = nn.Sequential(
            *[Resblock(config.residual_dim, use_layer_norm=True) for _ in range(1)]
        )

        self.action_fc = nn.Linear(config.residual_dim, config.num_action_types)

        self.action_type_emb = nn.Embedding(3, config.state_embedding_dim)

    def forward(self, state: torch.Tensor, action_type_mask: torch.Tensor):
        T, B, *_ = state.shape

        embedded_state = self.project(state)
        embedded_state = F.relu(self.resblocks(embedded_state))
        action_type_logits = self.action_fc(embedded_state)
        action_type_policy = _legal_policy(action_type_logits, action_type_mask)
        action_type_index = torch.multinomial(action_type_policy.view(T * B, -1), 1)
        action_type_index = action_type_index.view(T, B)

        action_type_embedding = self.action_type_emb(action_type_index.long())

        autoregressive_embedding = state + action_type_embedding

        return (
            action_type_logits,
            action_type_policy,
            action_type_index,
            autoregressive_embedding,
        )
