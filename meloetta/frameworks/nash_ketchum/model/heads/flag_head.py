import torch
import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    _legal_policy,
    _multinomial,
    MLP,
    BIAS,
)

from meloetta.data import CHOICE_FLAGS


class FlagsHead(nn.Module):
    def __init__(self, config: config.FlagHeadConfig):
        super().__init__()

        self.mlp = MLP(
            [
                config.autoregressive_embedding_dim,
                config.autoregressive_embedding_dim,
                len(CHOICE_FLAGS),
            ]
        )
        self.proj_flag = MLP(
            [
                len(CHOICE_FLAGS),
                config.autoregressive_embedding_dim,
                config.autoregressive_embedding_dim,
            ]
        )
        self.one_hot = nn.Embedding.from_pretrained(torch.eye(len(CHOICE_FLAGS)))

    def forward(
        self,
        action_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        flag_mask: torch.Tensor,
        flag_index: torch.Tensor = None,
    ):
        T, B, *_ = autoregressive_embedding.shape

        flag_logits = self.mlp(autoregressive_embedding)
        flag_policy = _legal_policy(flag_logits, flag_mask)
        flag_logits = torch.where(flag_mask, flag_logits, BIAS)
        flag_policy = flag_policy.view(T * B, -1)

        if flag_index is None:
            flag_index = _multinomial(flag_policy)

        flag_policy = flag_policy.view(T, B, -1)
        flag_index = flag_index.view(T, B, -1)

        flag_one_hot = self.one_hot(flag_index.long())
        projected_flag_embedding = self.proj_flag(flag_one_hot)
        projected_flag_embedding = projected_flag_embedding.view(T, B, -1)

        valid_indices = action_index == 0
        valid_indices = valid_indices.view(T, B, 1)

        autoregressive_embedding = (
            autoregressive_embedding + valid_indices * projected_flag_embedding
        )

        return (
            flag_logits,
            flag_policy,
            flag_index,
            autoregressive_embedding,
        )
