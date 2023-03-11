import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import Resblock, GLU
from meloetta.frameworks.nash_ketchum.model.utils import _legal_policy


class ActionTypeHead(nn.Module):
    def __init__(self, config: config.ActionTypeHeadConfig):
        super().__init__()

        self.num_action_types = config.num_action_types
        self.project = nn.Linear(config.state_embedding_dim, config.residual_dim)
        self.action_type_resblocks = nn.Sequential(
            *[Resblock(config.residual_dim, use_layer_norm=True) for _ in range(4)]
        )
        self.action_fc = nn.Linear(config.residual_dim, config.num_action_types)
        self.action_type_embedding = nn.Embedding(
            config.num_action_types, config.action_map_dim
        )

        self.state_resblocks = nn.Sequential(
            *[
                Resblock(config.state_embedding_dim, use_layer_norm=True)
                for _ in range(4)
            ]
        )
        self.state_lin = nn.Linear(config.state_embedding_dim, config.action_map_dim)

    def forward(
        self,
        state: torch.Tensor,
        action_type_mask: torch.Tensor,
    ):
        T, B, *_ = state.shape

        embedded_state = self.project(state)
        embedded_state = self.action_type_resblocks(embedded_state)
        action_type_logits = self.action_fc(embedded_state)
        action_type_policy = _legal_policy(action_type_logits, action_type_mask)
        action_type_index = torch.multinomial(action_type_policy.view(T * B, -1), 1)
        action_type_index = action_type_index.view(T, B)

        embedding1 = self.action_type_embedding(action_type_index.long())
        embedding2 = self.state_lin(F.relu(self.state_resblocks(state)))

        autoregressive_embedding = embedding1 + embedding2

        return (
            action_type_logits,
            action_type_policy,
            action_type_index,
            autoregressive_embedding,
        )
