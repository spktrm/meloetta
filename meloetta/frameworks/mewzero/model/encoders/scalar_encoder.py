import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.mewzero.model import config
from meloetta.frameworks.mewzero.model.utils import binary_enc_matrix

from meloetta.data import CHOICE_FLAGS, CHOICE_TARGETS, CHOICE_TOKENS


class ScalarEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.ScalarEncoderConfig):
        super().__init__()

        self.gen = gen
        self.turn1_bin = nn.Embedding.from_pretrained(binary_enc_matrix(202))
        self.turn2_bin = nn.Embedding.from_pretrained(binary_enc_matrix(53))

        action_mask_size = (
            3  # action types
            + 4  # move indices
            + 6  # switch indices
            + 5  # move flags
        )
        if gen == 8:
            action_mask_size += 4  # max move indices

        if n_active > 1:
            action_mask_size += 2 * n_active  # n_targets

        lin_in = (
            self.turn1_bin.embedding_dim
            + self.turn2_bin.embedding_dim
            + action_mask_size
            + 2
        )
        self.n_active = n_active

        if n_active > 1:
            prev_choice_token_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(CHOICE_TOKENS) + 1)
            )
            self.prev_choice_token_onehot = nn.Sequential(
                prev_choice_token_onehot,
                nn.Flatten(2),
            )
            prev_choice_index_onehot = nn.Embedding.from_pretrained(torch.eye(7))
            self.prev_choice_index_onehot = nn.Sequential(
                prev_choice_index_onehot,
                nn.Flatten(2),
            )
            prev_choice_flag_token_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(CHOICE_FLAGS) + 1)
            )
            self.prev_choice_flag_token_onehot = nn.Sequential(
                prev_choice_flag_token_onehot,
                nn.Flatten(2),
            )
            prev_choice_targ_token_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(CHOICE_TARGETS) + 1)
            )
            self.prev_choice_targ_token_onehot = nn.Sequential(
                prev_choice_targ_token_onehot,
                nn.Flatten(2),
            )
            lin_in += n_active * (
                prev_choice_token_onehot.embedding_dim
                + prev_choice_index_onehot.embedding_dim
                + prev_choice_flag_token_onehot.embedding_dim
                + prev_choice_targ_token_onehot.embedding_dim
            )
            self.choices_done_onehot = nn.Embedding.from_pretrained(
                torch.eye(n_active * 2)
            )
            lin_in += self.choices_done_onehot.embedding_dim

        self.lin = nn.Sequential(
            nn.Linear(lin_in, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(
        self,
        turn: torch.Tensor,
        turns_since_last_move: torch.Tensor,
        prev_choices: torch.Tensor,
        choices_done: torch.Tensor,
        action_type_mask: torch.Tensor,
        move_mask: torch.Tensor,
        max_move_mask: torch.Tensor,
        switch_mask: torch.Tensor,
        flag_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        turn = turn.clamp(min=0, max=200)
        turns_since_last_move = turns_since_last_move.clamp(min=0, max=50)
        scalar_emb = [
            self.turn1_bin(turn),
            self.turn2_bin(turns_since_last_move),
            (turn / 200).unsqueeze(-1),
            (turns_since_last_move / 50).unsqueeze(-1),
            action_type_mask,
            move_mask,
            switch_mask,
            flag_mask,
        ]
        if self.gen == 8:
            scalar_emb.append(max_move_mask)

        if self.n_active > 1:
            prev_choices_x = prev_choices + 1
            prev_choice_token = prev_choices_x[..., 0]
            prev_choice_index = prev_choices[..., 1] % 4
            prev_choice_index = prev_choice_index + 1
            prev_choice_index[prev_choices[..., 1] == -1] = 0
            prev_choice_flag_token = prev_choices_x[..., 2]
            prev_choice_targ_token = prev_choices_x[..., 3]
            scalar_emb += [
                self.choices_done_onehot(choices_done),
                self.prev_choice_token_onehot(prev_choice_token),
                self.prev_choice_index_onehot(prev_choice_index),
                self.prev_choice_flag_token_onehot(prev_choice_flag_token),
                self.prev_choice_targ_token_onehot(prev_choice_targ_token),
                target_mask,
            ]

        scalar_emb = torch.cat(scalar_emb, dim=-1)
        scalar_emb = self.lin(scalar_emb)

        return scalar_emb
