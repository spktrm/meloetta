import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.mewzero.model import config
from meloetta.frameworks.mewzero.model.utils import _legal_policy

from meloetta.data import CHOICE_FLAGS


class FlagsHead(nn.Module):
    def __init__(self, config: config.FlagHeadConfig):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.autoregressive_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(CHOICE_FLAGS)),
        )
        self.proj_flag = nn.Sequential(
            nn.Linear(len(CHOICE_FLAGS), config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.autoregressive_embedding_dim),
        )
        self.one_hot = nn.Embedding.from_pretrained(torch.eye(len(CHOICE_FLAGS)))

    def forward(
        self,
        action_type_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        flag_mask: torch.Tensor,
    ):
        T, B, *_ = autoregressive_embedding.shape

        flag_logits = self.mlp(autoregressive_embedding)
        flag_policy = _legal_policy(flag_logits, flag_mask)
        flag_index = torch.multinomial(flag_policy.view(T * B, -1), 1)
        flag_index = flag_index.view(T, B, -1)

        flag_one_hot = self.one_hot(flag_index.long())
        projected_flag_embedding = self.proj_flag(flag_one_hot)
        projected_flag_embedding = projected_flag_embedding.view(T, B, -1)

        valid_indices = action_type_index == 0
        valid_indices = valid_indices.view(T, B, 1)

        return (
            flag_logits,
            flag_policy,
            flag_index,
            autoregressive_embedding + valid_indices * projected_flag_embedding,
        )
