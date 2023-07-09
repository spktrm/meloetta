import torch
import torch.nn.functional as F

from torch import nn

from collections import OrderedDict
from typing import Tuple

from meloetta.frameworks.nash_ketchum.modelv2 import config
from meloetta.frameworks.nash_ketchum.modelv2.utils import (
    sqrt_one_hot_matrix,
    power_one_hot_matrix,
    binary_enc_matrix,
    linear_layer,
    gather_along_rows,
    MLP,
    ToVector,
    TransformerEncoder,
)
from meloetta.types import TensorDict
from meloetta.embeddings import (
    AbilityEmbedding,
    PokedexEmbedding,
    MoveEmbedding,
    ItemEmbedding,
)

from meloetta.data import (
    GENDERS,
    STATUS,
    TOKENIZED_SCHEMA,
    ITEM_EFFECTS,
    BattleTypeChart,
)


def _select_along_rows(src: torch.Tensor, index: torch.Tensor):
    index_expanded = index.unsqueeze(2).expand(-1, -1, src.shape[-1]).long()
    return torch.gather(src, 1, index_expanded)


class HistoryEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.SideEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.switch_action_embedding = nn.Parameter(
            torch.randn(1, 1, config.entity_embedding_dim)
        )

        self.action_embedding = MLP(
            [2 * config.entity_embedding_dim, config.output_dim]
        )

        def _conv_layer_init(*args, **kwargs):
            layer = nn.Conv1d(*args, **kwargs)
            nn.init.normal_(layer.weight, std=5e-3)
            nn.init.constant_(layer.bias, val=0)
            return layer

        self.in_conv = nn.Sequential(
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            _conv_layer_init(40, 256, 1, padding="same"),
        )

        num_layers = 2 * self.config.num_heads
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.output_dim),
                    nn.ReLU(),
                    _conv_layer_init(256, 256, 3, padding="same"),
                )
                for _ in range(num_layers)
            ]
        )

        self.out_conv = nn.Sequential(
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            _conv_layer_init(256, 1, 1, padding="same"),
        )

    def _preproc(
        self,
        history: torch.Tensor,
        pokemon_embeddings: torch.Tensor,
        move_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        T, B, S, N, *_ = history.shape

        # reshape everything
        player_id = history[..., 0].view(T * B, S * N)
        action_type_token = history[..., 1].view(T * B, S * N)
        move_index = history[..., 2].view(T * B, S * N)
        switch_index = history[..., 3].view(T * B, S * N)
        my_side_embeddings = pokemon_embeddings[:, :, 0].view(T * B, 12, -1)
        op_side_embeddings = pokemon_embeddings[:, :, -1].view(T * B, 12, -1)
        my_move_embeddings = move_embeddings[:, :, 0].view(T * B, 4 * 12, -1)
        op_move_embeddings = move_embeddings[:, :, -1].view(T * B, 4 * 12, -1)

        # select relevant embeddings along index dimension
        my_selected_side_embeddings = _select_along_rows(
            my_side_embeddings, switch_index.clamp(min=0)
        )
        op_selected_side_embeddings = _select_along_rows(
            op_side_embeddings, switch_index.clamp(min=0)
        )
        my_selected_move_embeddings = _select_along_rows(
            my_move_embeddings,
            (4 * switch_index.clamp(min=0) + move_index.clamp(min=0)),
        )
        op_selected_move_embeddings = _select_along_rows(
            op_move_embeddings,
            (4 * switch_index.clamp(min=0) + move_index.clamp(min=0)),
        )

        # Organise switch embedding
        switch_action_embedding = self.switch_action_embedding.repeat(T * B, S * N, 1)
        switch_action_mask = (action_type_token == 1).unsqueeze(-1)

        # select correct move/switch embedding
        my_selected_move_embeddings = torch.where(
            switch_action_mask, switch_action_embedding, my_selected_move_embeddings
        )
        op_selected_move_embeddings = torch.where(
            switch_action_mask, switch_action_embedding, op_selected_move_embeddings
        )

        # concatenate move to pokemon that used it
        my_action_embeddings = torch.cat(
            (my_selected_side_embeddings, my_selected_move_embeddings), dim=-1
        )
        op_action_embeddings = torch.cat(
            (op_selected_side_embeddings, op_selected_move_embeddings), dim=-1
        )

        # get correct order
        action_embeddings = torch.where(
            (player_id == 0).unsqueeze(-1), my_action_embeddings, op_action_embeddings
        )
        action_embeddings = self.action_embedding(action_embeddings)

        # reshape
        valid_action_mask = (action_type_token >= 0).unsqueeze(-1)
        action_embeddings = action_embeddings * valid_action_mask

        return action_embeddings

    def forward(
        self,
        history: torch.Tensor,
        pokemon_embeddings: torch.Tensor,
        move_embeddings: torch.Tensor,
    ) -> TensorDict:
        T, B, *_ = history.shape

        action_embeddings = self._preproc(history, pokemon_embeddings, move_embeddings)
        action_embeddings = self.in_conv(action_embeddings)

        for conv_block in self.conv_layers:
            residual = action_embeddings
            action_embeddings_ = conv_block(action_embeddings)
            action_embeddings = action_embeddings_ + residual

        action_embedding = self.out_conv(action_embeddings)
        return action_embedding.view(T, B, -1)
