import torch
import torch.nn as nn

from typing import Tuple

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.interfaces import (
    EncoderOutput,
    Indices,
    Logits,
    Policy,
    State,
)
from meloetta.frameworks.nash_ketchum.model.heads import (
    ActionTypeHead,
    MoveHead,
    FlagsHead,
    SwitchHead,
    MaxMoveHead,
)


class PolicyHeads(nn.Module):
    def __init__(self, gen: int, gametype: str, config: config.PolicyHeadsConfig):
        super().__init__()

        self.config = config
        self.gametype = gametype
        self.gen = gen

        self.action_type_head = ActionTypeHead(config.action_type_head_config)
        self.move_head = MoveHead(config.move_head_config)
        self.switch_head = SwitchHead(config.switch_head_config)

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
        moves = encoder_output.moves
        switches = encoder_output.switches

        (
            action_type_logits,
            action_type_policy,
            action_type_index,
            at_autoregressive_embedding,
        ) = self.action_type_head(
            state_emb,
            state["action_type_mask"],
        )

        (
            move_logits,
            move_policy,
            move_index,
            autoregressive_embedding,
        ) = self.move_head(
            action_type_index,
            at_autoregressive_embedding,
            moves,
            state["move_mask"],
        )

        (
            switch_logits,
            switch_policy,
            switch_index,
            autoregressive_embedding,
        ) = self.switch_head(
            action_type_index,
            autoregressive_embedding,
            switches,
            state["switch_mask"],
        )

        if self.gen >= 6:
            (
                flag_logits,
                flag_policy,
                flag_index,
                autoregressive_embedding,
            ) = self.flag_head(
                action_type_index,
                autoregressive_embedding,
                state["flag_mask"],
            )
        else:
            flag_logits = None
            flag_policy = None
            flag_index = None

        if self.gen == 8:
            max_move_logits, max_move_policy, max_move_index = self.max_move_head(
                action_type_index,
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
                at_autoregressive_embedding,
            )
        else:
            target_index = None
            target_logits = None
            target_policy = None

        indices = Indices(
            action_type_index=action_type_index,
            move_index=move_index,
            max_move_index=max_move_index,
            switch_index=switch_index,
            flag_index=flag_index,
            target_index=target_index,
        )
        logits = Logits(
            action_type_logits=action_type_logits,
            move_logits=move_logits,
            max_move_logits=max_move_logits,
            switch_logits=switch_logits,
            flag_logits=flag_logits,
            target_logits=target_logits,
        )
        policy = Policy(
            action_type_policy=action_type_policy,
            move_policy=move_policy,
            max_move_policy=max_move_policy,
            switch_policy=switch_policy,
            flag_policy=flag_policy,
            target_policy=target_policy,
        )

        return indices, logits, policy
