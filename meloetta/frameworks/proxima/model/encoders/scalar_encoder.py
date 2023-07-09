import torch
import torch.nn as nn

from collections import OrderedDict

from meloetta.types import TensorDict

from meloetta.frameworks.proxima.model import config
from meloetta.frameworks.proxima.model.utils import (
    sqrt_one_hot_matrix,
    MLP,
    Resblock,
    linear_layer,
)


from meloetta.data import (
    CHOICE_FLAGS,
    CHOICE_TARGETS,
    CHOICE_TOKENS,
    BOOSTS,
    VOLATILES,
    WEATHERS,
    PSEUDOWEATHERS,
    SIDE_CONDITIONS,
)

DRAW_BY_TURNS = 300


class ScalarEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.ScalarEncoderConfig):
        super().__init__()

        self.gen = gen
        self.turn_sqrt_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(DRAW_BY_TURNS + 2)
        )

        action_mask_size = 0
        # action_mask_size = (
        #     3  # action types
        #     + 4  # move indices
        #     + 6  # switch indices
        #     + 5  # move flags
        # )
        if gen == 8:
            action_mask_size += 4  # max move indices

        if n_active > 1:
            action_mask_size += 2 * n_active  # n_targets

        self.n_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        lin_in = (
            self.turn_sqrt_onehot.embedding_dim
            + action_mask_size
            + 1
            + self.n_onehot.embedding_dim * 2 * 3
        )
        self.n_active = n_active

        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))

        self.toxicspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])

        self.min_dur = nn.Embedding.from_pretrained(torch.eye(6)[..., 1:])
        self.max_dur = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])

        self.boosts_mlp = nn.Sequential(
            linear_layer(2 * len(BOOSTS), config.embedding_dim),
            MLP([config.embedding_dim, config.embedding_dim]),
        )
        self.volatiles_mlp = nn.Sequential(
            linear_layer(2 * len(VOLATILES), config.embedding_dim),
            MLP([config.embedding_dim, config.embedding_dim]),
        )

        side_con_size = (
            2
            + self.toxicspikes_onehot.embedding_dim
            + self.spikes_onehot.embedding_dim
            + (len(SIDE_CONDITIONS) - 4)
            * (self.min_dur.embedding_dim + self.max_dur.embedding_dim)
        )
        self.side_conditions_mlp = nn.Sequential(
            linear_layer(2 * side_con_size, config.embedding_dim),
            MLP([config.embedding_dim, config.embedding_dim]),
        )

        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(len(WEATHERS) + 1))
        self.time_left_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.min_time_left_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        pw_min_onehot = nn.Embedding.from_pretrained(torch.eye(8))
        self.pw_min_onehot = nn.Sequential(pw_min_onehot, nn.Flatten(2))
        pw_max_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.pw_max_onehot = nn.Sequential(pw_max_onehot, nn.Flatten(2))

        weather_size = (
            self.weather_onehot.embedding_dim
            + self.time_left_onehot.embedding_dim
            + self.min_time_left_onehot.embedding_dim
            + pw_min_onehot.embedding_dim * len(PSEUDOWEATHERS)
            + pw_max_onehot.embedding_dim * len(PSEUDOWEATHERS)
        )
        self.weather_mlp = nn.Sequential(
            linear_layer(weather_size, config.embedding_dim),
            MLP([config.embedding_dim, config.embedding_dim]),
        )

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

        self.scalar_mlp = nn.Sequential(
            linear_layer(lin_in, config.embedding_dim),
            MLP([config.embedding_dim, config.embedding_dim]),
        )

        layers = [Resblock(config.embedding_dim) for _ in range(config.num_resblocks)]
        self.encoder = nn.Sequential(*layers)

    def encode_boosts(self, boosts: torch.Tensor) -> torch.Tensor:
        norm_boosts_mask = boosts[..., :6] >= 0
        norm_boosts_embed = ((boosts[..., :6] + 2) / 2) * norm_boosts_mask + (
            (2 / (boosts[..., :6] - 2).clamp(min=1)) * ~norm_boosts_mask
        )
        other_boosts_mask = boosts[..., 6:] >= 0
        other_boosts_embed = ((boosts[..., 6:] + 3) / 3) * other_boosts_mask + (
            (3 / (boosts[..., 6:] - 3).clamp(min=1)) * ~other_boosts_mask
        )
        return torch.cat((norm_boosts_embed, other_boosts_embed), dim=-1)

    def encode_volatiles(self, volatiles: torch.Tensor) -> torch.Tensor:
        return volatiles.float()

    def encode_side_conditions(self, side_conditions: torch.Tensor) -> torch.Tensor:
        T, B, *_ = side_conditions.shape

        stealthrock = side_conditions[..., -4].unsqueeze(-1)
        stickyweb = side_conditions[..., -1].unsqueeze(-1)

        toxicspikes_onehot = self.toxicspikes_onehot(side_conditions[..., -3])
        spikes_onehot = self.spikes_onehot(side_conditions[..., -2])

        _, min_dur, max_dur = (
            side_conditions[..., :-4].view(T, B, 2, -1, 3).chunk(3, -1)
        )

        side_condition_embed = torch.cat(
            [
                stealthrock,
                stickyweb,
                toxicspikes_onehot,
                spikes_onehot,
                self.min_dur(min_dur).flatten(3),
                self.max_dur(max_dur).flatten(3),
            ],
            dim=-1,
        )
        return side_condition_embed

    def forward(
        self,
        turn: torch.Tensor,
        # action_type_mask: torch.Tensor,
        # move_mask: torch.Tensor,
        # max_move_mask: torch.Tensor,
        # switch_mask: torch.Tensor,
        # flag_mask: torch.Tensor,
        # target_mask: torch.Tensor,
        n: torch.Tensor,
        total_pokemon: torch.Tensor,
        faint_counter: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
        side_conditions: torch.Tensor,
        wisher: torch.Tensor,
        weather: torch.Tensor,
        pseudoweather: torch.Tensor,
    ) -> TensorDict:
        turn = turn.clamp(min=0, max=DRAW_BY_TURNS)
        scalar_enc = [
            self.turn_sqrt_onehot(turn),
            (turn / DRAW_BY_TURNS).unsqueeze(-1),
            # action_type_mask,
            # move_mask,
            # switch_mask,
            # flag_mask,
            self.n_onehot(n).flatten(-2),
            self.n_onehot(total_pokemon.clamp(max=6)).flatten(-2),
            self.n_onehot(faint_counter).flatten(-2),
        ]
        if self.gen == 8:
            scalar_enc.append(max_move_mask)

        if self.n_active > 1:
            prev_choices_x = prev_choices + 1
            prev_choice_token = prev_choices_x[..., 0]
            prev_choice_index = prev_choices[..., 1] % 4
            prev_choice_index = prev_choice_index + 1
            prev_choice_index[prev_choices[..., 1] == -1] = 0
            prev_choice_flag_token = prev_choices_x[..., 2]
            prev_choice_targ_token = prev_choices_x[..., 3]
            scalar_enc += [
                self.choices_done_onehot(choices_done),
                self.prev_choice_token_onehot(prev_choice_token),
                self.prev_choice_index_onehot(prev_choice_index),
                self.prev_choice_flag_token_onehot(prev_choice_flag_token),
                self.prev_choice_targ_token_onehot(prev_choice_targ_token),
                target_mask,
            ]

        boosts = self.encode_boosts(boosts)
        volatiles = self.encode_volatiles(volatiles)
        side_conditions = self.encode_side_conditions(side_conditions)

        weather_token = weather[..., 0]
        time_left = weather[..., 1]
        min_time_left = weather[..., 2]

        weather_onehot = self.weather_onehot(weather_token + 1)
        time_left_onehot = self.time_left_onehot(time_left)
        min_time_left_onehot = self.min_time_left_onehot(min_time_left)

        pseudo_weather_x = pseudoweather + 1
        pw_min_time_left = pseudo_weather_x[..., 0]
        pw_max_time_left = pseudo_weather_x[..., 1]

        pw_min_time_left_onehot = self.pw_min_onehot(pw_min_time_left)
        pw_max_time_left_onehot = self.pw_max_onehot(pw_max_time_left)

        scalar_embeddings = [
            self.boosts_mlp(boosts.flatten(2)),
            self.volatiles_mlp(volatiles.flatten(2)),
            self.side_conditions_mlp(side_conditions.flatten(2)),
            self.weather_mlp(
                torch.cat(
                    (
                        weather_onehot,
                        time_left_onehot,
                        min_time_left_onehot,
                        pw_max_time_left_onehot,
                        pw_min_time_left_onehot,
                    ),
                    dim=-1,
                )
            ),
            self.scalar_mlp(torch.cat(scalar_enc, dim=-1)),
        ]
        scalar_embedding = sum(scalar_embeddings)

        return OrderedDict(
            scalar_embedding=self.encoder(scalar_embedding),
            boosts=boosts,
            volatiles=volatiles,
            side_conditions=side_conditions,
        )
