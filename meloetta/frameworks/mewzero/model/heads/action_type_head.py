import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.mewzero.model import config
from meloetta.frameworks.mewzero.model.utils import Resblock, GLU
from meloetta.frameworks.mewzero.model.utils import _legal_policy


class ActionTypeHead(nn.Module):
    def __init__(self, config: config.ActionTypeHeadConfig):
        super().__init__()

        self.num_action_types = config.num_action_types
        self.project = nn.Linear(config.state_embedding_dim, config.residual_dim)
        self.resblocks = nn.Sequential(
            *[Resblock(config.residual_dim, use_layer_norm=True) for _ in range(2)]
        )

        self.action_fc = GLU(
            config.residual_dim, config.num_action_types, config.context_dim
        )
        self.fc1 = nn.Linear(config.num_action_types, config.action_map_dim)
        self.fc2 = nn.Linear(config.action_map_dim, config.action_map_dim)

        self.glu1 = GLU(
            config.action_map_dim,
            config.autoregressive_embedding_dim,
            config.context_dim,
        )
        self.glu2 = GLU(
            config.state_embedding_dim,
            config.autoregressive_embedding_dim,
            config.context_dim,
        )
        self.one_hot = nn.Embedding.from_pretrained(torch.eye(3))

    def forward(
        self,
        state: torch.Tensor,
        scalar_context: torch.Tensor,
        action_type_mask: torch.Tensor,
    ):
        T, B, *_ = state.shape

        embedded_state = self.project(state)
        embedded_state = F.relu(self.resblocks(embedded_state))
        action_type_logits = self.action_fc(embedded_state, scalar_context)
        action_type_policy = _legal_policy(action_type_logits, action_type_mask)
        action_type_index = torch.multinomial(action_type_policy.view(T * B, -1), 1)
        action_type_index = action_type_index.view(T, B)

        action_one_hot = self.one_hot(action_type_index.long())
        embedding1 = F.relu(self.fc1(action_one_hot))
        embedding1 = self.fc2(embedding1)
        embedding1 = self.glu1(embedding1, scalar_context)
        embedding2 = self.glu2(state, scalar_context)
        autoregressive_embedding = embedding1 + embedding2

        return (
            action_type_logits,
            action_type_policy,
            action_type_index,
            autoregressive_embedding,
        )
