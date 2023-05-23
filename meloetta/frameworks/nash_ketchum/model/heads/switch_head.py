import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    _legal_policy,
    _multinomial,
    gather_along_rows,
    MLP,
    BIAS,
)


class SwitchHead(nn.Module):
    def __init__(self, config: config.SwitchHeadConfig):
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
        self.proj_switch = MLP(
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
        switches: torch.Tensor,
        switch_mask: torch.Tensor,
        switch_index: torch.Tensor = None,
    ):
        T, B, *_ = autoregressive_embedding.shape

        keys = self.key_fc(switches)
        query = self.query_fc(autoregressive_embedding)
        query = query.view(T, B, 1, -1)

        switch_logits = query @ keys.transpose(-2, -1)
        # switch_logits = switch_logits / self.size
        switch_logits = switch_logits.view(T, B, -1)

        switch_mask = switch_mask.view(T * B, -1)
        switch_mask[switch_mask.sum() == 0] = True
        switch_mask = switch_mask.view(T, B, -1)

        switch_policy = _legal_policy(switch_logits, switch_mask)
        switch_logits = torch.where(switch_mask, switch_logits, BIAS)
        switch_policy = switch_policy.view(T * B, -1)

        if switch_index is None:
            embedding_index = _multinomial(switch_policy)
            switch_index = embedding_index.view(T, B, -1)

        embedding_index = switch_index.view(T * B, -1)
        switch_policy = switch_policy.view(T, B, -1)

        switch_embedding = gather_along_rows(keys.flatten(0, -3), embedding_index, 1)
        switch_embedding = switch_embedding.view(T, B, -1)
        projected_move_embedding = self.proj_switch(switch_embedding)

        valid_indices = action_type_index == 1
        valid_indices = valid_indices.unsqueeze(-1)

        autoregressive_embedding = (
            autoregressive_embedding + valid_indices * projected_move_embedding
        )

        return (
            switch_logits,
            switch_policy,
            switch_index,
            autoregressive_embedding,
        )
