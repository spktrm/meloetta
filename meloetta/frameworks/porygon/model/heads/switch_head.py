import torch
import torch.nn as nn

from meloetta.frameworks.porygon.model import config


class SwitchHead(nn.Module):
    def __init__(self, config: config.SwitchHeadConfig):
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

    def forward(
        self,
        state_emb: torch.Tensor,
        switches: torch.Tensor,
    ):
        T, B, *_ = state_emb.shape

        keys = self.key_fc(switches)

        query = self.query_fc(state_emb)
        query = query.view(T, B, 1, -1)

        switch_logits = query @ keys.transpose(-2, -1)
        switch_logits = switch_logits.view(T, B, -1)

        return switch_logits, keys
