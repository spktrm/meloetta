import torch
import torch.nn as nn

from typing import Tuple

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.interfaces import (
    EncoderOutput,
    Indices,
    Logits,
    Policy,
    State,
)
from meloetta.frameworks.porygon.model.heads import (
    MoveHead,
    FlagsHead,
    SwitchHead,
    MaxMoveHead,
)
from meloetta.frameworks.porygon.model.utils import (
    _legal_policy,
    gather_along_rows,
)


class PolicyHeads(nn.Module):
    def __init__(self, gen: int, gametype: str, config: config.PolicyHeadsConfig):
        super().__init__()

        self.config = config
        self.gametype = gametype
        self.gen = gen

        self.move_keys = nn.Sequential(
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
            nn.LayerNorm(config.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.key_dim),
        )
        self.key_fc = nn.Sequential(
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
            nn.LayerNorm(config.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.key_dim),
        )
        self.query_fc = nn.Sequential(
            nn.Linear(config.autoregressive_embedding_dim, config.query_hidden_dim),
            nn.LayerNorm(config.query_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.query_hidden_dim, config.key_dim),
        )
        self.proj_key = nn.Sequential(
            nn.Linear(config.key_dim, config.entity_embedding_dim),
            nn.LayerNorm(config.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.autoregressive_embedding_dim),
        )

        if gen == 8:
            self.max_move_head = MaxMoveHead(config.move_head_config)

        if gen >= 6:
            self.flag_head = FlagsHead(config.flag_head_config)

        # if gametype != "singles":
        #     self.target_head = PolicyHead(config.target_head_config)

    def forward(
        self,
        state_emb: torch.Tensor,
        encoder_output: EncoderOutput,
        state: State,
    ) -> Tuple[Indices, Logits, Policy]:

        T, B, *_ = state_emb.shape

        moves = encoder_output.moves
        switches = encoder_output.switches

        move_mask = state["move_mask"]
        switch_mask = state["switch_mask"]

        T, B, *_ = moves.shape

        pre_keys = torch.cat((moves.squeeze(2), switches), dim=-2)
        post_keys = self.key_fc(pre_keys)

        query = self.query_fc(state_emb)
        query = query.view(T, B, 1, -1)

        logits = query @ post_keys.transpose(-2, -1)
        logits_scaled = logits / (logits.shape[-1] ** 0.5)
        logits = logits_scaled.view(T, B, -1)

        mask = torch.cat((move_mask, switch_mask), dim=-1)

        policy = _legal_policy(logits, mask)
        embedding_index = torch.multinomial(policy.view(T * B, -1), 1)
        action_index = embedding_index.view(T, B, -1)

        action_embedding = gather_along_rows(
            post_keys.flatten(0, -3), embedding_index, 1
        )
        action_embedding = action_embedding.view(T, B, -1)
        autoregressive_embedding = state_emb + self.proj_key(action_embedding)

        if self.gen >= 6:
            (
                flag_logits,
                flag_policy,
                flag_index,
                autoregressive_embedding,
            ) = self.flag_head(
                action_index,
                autoregressive_embedding,
                state["flag_mask"],
            )
        else:
            flag_logits = None
            flag_policy = None
            flag_index = None

        if self.gen == 8:
            max_move_logits, max_move_policy, max_move_index = self.max_move_head(
                action_index,
                autoregressive_embedding,
                moves,
                state["max_move_mask"],
            )
        else:
            max_move_index = None
            max_move_logits = None
            max_move_policy = None

        if self.gametype != "singles":
            target_logits, target_policy, target_index = self.target_head(
                state["prev_choices"],
                moves,
                switches,
                action_index,
            )
        else:
            target_index = None
            target_logits = None
            target_policy = None

        indices = Indices(
            action_index=action_index,
            max_move_index=max_move_index,
            flag_index=flag_index,
            target_index=target_index,
        )
        logits = Logits(
            action_logits=logits,
            max_move_logits=max_move_logits,
            flag_logits=flag_logits,
            target_logits=target_logits,
        )
        policy = Policy(
            action_policy=policy,
            max_move_policy=max_move_policy,
            flag_policy=flag_policy,
            target_policy=target_policy,
        )

        return indices, logits, policy