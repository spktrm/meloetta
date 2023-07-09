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
    RandomBattlesEmbedding,
)

from meloetta.data import (
    GENDERS,
    STATUS,
    TOKENIZED_SCHEMA,
    ITEM_EFFECTS,
    BattleTypeChart,
)


class PokemonEmbedding(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        config: config.SideEncoderConfig = config.SideEncoderConfig(),
    ) -> None:
        super().__init__()

        self.gen = gen

        self.randbats_embedding = RandomBattlesEmbedding(gen)

        pokedex_onehot = PokedexEmbedding(gen=gen)
        ability_onehot = AbilityEmbedding(gen=gen)
        item_onehot = ItemEmbedding(gen=gen)
        move_onehot = MoveEmbedding(gen=gen)

        num_species = pokedex_onehot.num_embeddings
        self.pokedex_onehot = nn.Embedding.from_pretrained(
            torch.eye(num_species)[..., 1:]
        )
        self.species_embedding = linear_layer(
            num_species - 1,
            config.entity_embedding_dim,
            bias=False,
        )

        self.ability_onehot = nn.Embedding.from_pretrained(
            torch.eye(ability_onehot.num_embeddings)[..., 1:]
        )
        self.ability_embedding = linear_layer(
            ability_onehot.num_embeddings - 1, config.entity_embedding_dim, bias=False
        )

        self.item_onehot = nn.Embedding.from_pretrained(
            torch.eye(item_onehot.num_embeddings)[..., 1:]
        )
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)[..., 1:]
        )
        self.item_embedding = linear_layer(
            self.item_onehot.embedding_dim + self.item_effect_onehot.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )

        self.pp_bin_enc = nn.Embedding.from_pretrained(binary_enc_matrix(64))
        self.move_raw_onehot = nn.Embedding.from_pretrained(
            torch.eye(move_onehot.num_embeddings)[..., 1:]
        )
        self.move_embedding = linear_layer(
            self.move_raw_onehot.embedding_dim + self.pp_bin_enc.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )

        # self.last_move_embedding = linear_layer(
        #     move_onehot.embedding_dim + self.pp_bin_enc.embedding_dim,
        #     config.entity_embedding_dim,
        # )

        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.gender_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(GENDERS) + 1)[..., 1:]
        )

        status_dummy = torch.eye(len(STATUS) + 1)[..., 1:]
        self.status_onehot = nn.Embedding.from_pretrained(status_dummy)
        self.sleep_turns_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.toxic_turns_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(16)[..., 1:]
        )

        num_formes = len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1
        self.forme_embedding = nn.Embedding.from_pretrained(
            torch.eye(num_formes)[..., 1:]
        )
        self.level_onehot = nn.Embedding.from_pretrained(torch.eye(100))

        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(11)[..., 1:])
        self.stat_sqrt_onehot = nn.Embedding.from_pretrained(
            power_one_hot_matrix(512, 1 / 3)[..., 1:]
        )

        self.side_embedding = nn.Embedding(2, config.entity_embedding_dim)

        onehot_size = (
            # self.pokedex_onehot.embedding_dim
            # + self.ability_onehot.embedding_dim
            # + self.item_onehot.embedding_dim
            # + self.item_effect_onehot.embedding_dim
            +self.active_onehot.embedding_dim
            + self.fainted_onehot.embedding_dim
            + self.gender_onehot.embedding_dim
            + self.status_onehot.embedding_dim
            + self.sleep_turns_onehot.embedding_dim
            + self.toxic_turns_onehot.embedding_dim
            + self.forme_embedding.embedding_dim
            + self.level_onehot.embedding_dim
            + self.hp_onehot.embedding_dim
        )

        if gen == 9:
            self.commanding_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
            self.reviving_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
            self.tera_onehot = nn.Embedding.from_pretrained(torch.eye(2))

            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)[..., 1:]
            )
            self.teratype_lin = linear_layer(
                len(BattleTypeChart), config.entity_embedding_dim, bias=False
            )

            onehot_size += self.tera_onehot.embedding_dim

        elif gen == 8:
            self.can_gmax_embedding = nn.Embedding.from_pretrained(torch.eye(3))

            onehot_size += self.can_gmax_embedding.embedding_dim

        self.onehots_lin = linear_layer(onehot_size, config.entity_embedding_dim)

        self.mlp = MLP([config.entity_embedding_dim, config.entity_embedding_dim])

    def encode_species(self, species_token: torch.Tensor):
        species_token = species_token.clamp(min=0)
        (
            unknown_species_encoding,
            unknown_ability_encoding,
            unknown_item_encoding,
            unknown_moveset_encoding,
            unknown_teratype_encoding,
        ) = self.randbats_embedding(species_token)

        known_onehot = self.pokedex_onehot(species_token)
        team_onehot = known_onehot.sum(-2, keepdim=True).expand(-1, -1, -1, 12, -1)
        unknown_species_encoding = (
            torch.ones_like(unknown_species_encoding) - team_onehot
        )
        unknown_species_encoding = (
            unknown_species_encoding
            / unknown_species_encoding.sum(-1, keepdim=True).clamp(min=1)
        )

        known_mask = (species_token > 0).unsqueeze(-1)

        return (
            torch.where(known_mask, known_onehot, unknown_species_encoding),
            torch.where(
                known_mask,
                unknown_ability_encoding,
                unknown_species_encoding @ self.randbats_embedding.abilities.weight[1:],
            ),
            torch.where(
                known_mask,
                unknown_item_encoding,
                unknown_species_encoding @ self.randbats_embedding.items.weight[1:],
            ),
            torch.where(
                known_mask,
                unknown_moveset_encoding,
                unknown_species_encoding @ self.randbats_embedding.movesets.weight[1:],
            ),
            torch.where(
                known_mask,
                unknown_teratype_encoding,
                unknown_species_encoding @ self.randbats_embedding.teratypes.weight[1:],
            ),
        )

    def embed_ability(
        self, ability_token: torch.Tensor, unknown_ability_encoding: torch.Tensor
    ) -> torch.Tensor:
        known_mask = (ability_token > 0).unsqueeze(-1)
        ability_encoding = torch.where(
            known_mask,
            self.ability_onehot(ability_token),
            unknown_ability_encoding
            / unknown_ability_encoding.sum(-1, keepdim=True).clamp(min=1),
        )
        return self.ability_embedding(ability_encoding)

    def embed_item(
        self,
        item_token: torch.Tensor,
        item_effect_token: torch.Tensor,
        unknown_item_encoding: torch.Tensor,
    ) -> torch.Tensor:
        known_mask = (item_token > 0).unsqueeze(-1)
        item_dist = torch.where(
            known_mask,
            self.item_onehot(item_token),
            unknown_item_encoding
            / unknown_item_encoding.sum(-1, keepdim=True).clamp(min=1),
        )
        item_concat = torch.cat(
            (item_dist, self.item_effect_onehot(item_effect_token)),
            dim=-1,
        )
        return self.item_embedding(item_concat)

    def embed_moveset(
        self,
        move_tokens: torch.Tensor,
        pp_token: torch.Tensor,
        unknwown_moveset_encoding: torch.Tensor,
    ) -> torch.Tensor:
        known_mask = (move_tokens > 0).unsqueeze(-1)
        unknwown_moveset_encoding = unknwown_moveset_encoding.unsqueeze(-2)
        known_moveset_onehot = self.move_raw_onehot(move_tokens).sum(-2, keepdim=True)
        unknwown_moveset_encoding = (
            unknwown_moveset_encoding - known_moveset_onehot
        ).expand(-1, -1, -1, 12, -1, -1)

        move_dist = torch.where(
            known_mask,
            self.move_raw_onehot(move_tokens),
            unknwown_moveset_encoding
            / unknwown_moveset_encoding.sum(-1, keepdim=True).clamp(min=1),
        )
        move_concat = torch.cat(
            (move_dist, self.pp_bin_enc(pp_token)),
            dim=-1,
        )
        move_embedding = self.move_embedding(move_concat)

        return move_embedding.sum(-2), move_embedding

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        longs = (x + 1).long().clamp(min=0)

        species_token = longs[..., 0]
        forme_token = longs[..., 1]
        # slot = longs[..., 2]
        # hp = longs[..., 3]
        # maxhp = longs[..., 4]
        hp_ratio = x[..., 5].float()
        # stats = x[..., 6:11]
        fainted_token = longs[..., 11]
        active_token = longs[..., 12]
        level_token = x[..., 13].long()
        gender_token = longs[..., 14]
        ability_token = longs[..., 15]
        # base_ability = longs[..., 16]
        item_token = longs[..., 17]
        # prev_item = longs[..., 18]
        item_effect_token = longs[..., 19]
        # prev_item_effect = longs[..., 20]
        status_token = longs[..., 21]
        sleep_turns = longs[..., 22]
        toxic_turns = longs[..., 23]
        # last_move = longs[..., 24]
        # last_move_pp = longs[..., 25].clamp(max=63)
        move_tokens = longs[..., 26:30]
        pp_tokens = longs[..., 30:34].clamp(max=63)

        if self.gen == 9:
            terastallized = longs[..., 34]
            teratype = longs[..., 35]
            # times_attacked = longs[..., 36]

        side = (longs[..., 37] - 1).clamp(min=0, max=1)

        (
            species_onehot,
            unknown_ability_encoding,
            unknown_item_encoding,
            unknown_moveset_encoding,
            teratype_encoding,
        ) = self.encode_species(species_token)

        species_embedding = self.species_embedding(species_onehot)

        hp_embedding = self.hp_onehot((hp_ratio * 10).clamp(min=0, max=10).long())
        stat_enc = hp_embedding

        # stat_embedding = self.stat_lin(
        #     hp_ratio.unsqueeze(-1)
        #     # torch.cat(
        #     #     (, stats / 512 * (stats >= 0) + -1 * (stats < 0)),
        #     #     dim=-1,
        #     # )
        # )

        ability_embedding = self.embed_ability(ability_token, unknown_ability_encoding)
        # base_ability_emb = self.ability_embedding(base_ability)

        item_embedding = self.embed_item(
            item_token, item_effect_token, unknown_item_encoding
        )
        # prev_item_emb = self.embed_item(prev_item, prev_item_effect)

        status_onehot = self.status_onehot(status_token)
        sleep_turns_onehot = self.sleep_turns_onehot(sleep_turns)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns)

        moveset_embedding, move_embeddings = self.embed_moveset(
            move_tokens, pp_tokens, unknown_moveset_encoding
        )

        # last_move_emb = self.last_move_embedding(
        #     torch.cat(
        #         (self.move_embedding_ae(last_move), self.pp_bin_enc(last_move_pp)),
        #         dim=-1,
        #     )
        # )

        forme_enc = self.forme_embedding(forme_token)
        active_enc = self.active_onehot(active_token)
        fainted_enc = self.fainted_onehot(fainted_token)
        gender_enc = self.gender_onehot(gender_token)
        level_enc = self.level_onehot(level_token.clamp(min=1) - 1)
        status_enc = torch.cat(
            (status_onehot, sleep_turns_onehot, toxic_turns_onehot), dim=-1
        )
        side_embedding = self.side_embedding(side)

        onehots = [
            forme_enc,
            stat_enc,
            active_enc,
            fainted_enc,
            gender_enc,
            level_enc,
            status_enc,
        ]

        embeddings = [
            species_embedding,
            ability_embedding,
            item_embedding,
            moveset_embedding,
            # stat_embedding,
            side_embedding,
            # "last_move_emb": last_move_emb,
            # "base_ability_emb": base_ability_emb,
            # "prev_item_emb": prev_item_emb,
        ]

        if self.gen == 9:
            teratype_encoding = torch.where(
                (teratype > 0).unsqueeze(-1),
                self.teratype_onehot(teratype),
                teratype_encoding,
            )
            embeddings += [self.teratype_lin(teratype_encoding)]
            onehots += [self.tera_onehot((terastallized > 0).long())]

        onehots = torch.cat(onehots, dim=-1)
        onehots_embedding = self.onehots_lin(onehots)
        embeddings += [onehots_embedding]

        pokemon_emb = sum(embeddings)

        mask = species_token >= 0
        pokemon_emb = self.mlp(pokemon_emb)

        return pokemon_emb, mask, move_embeddings


class SideEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.SideEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.embedding = PokemonEmbedding(gen=gen, config=config)

        self.compress = MLP(
            [
                config.entity_embedding_dim,
                config.output_dim,
            ]
        )

        self.move_gate = MLP(
            [
                config.entity_embedding_dim,
                config.entity_embedding_dim,
            ]
        )

    def forward(self, side: torch.Tensor) -> TensorDict:
        (
            pokemon_embeddings,
            pokemon_mask,
            move_embeddings,
        ) = self.embedding.forward(side)

        T, B, N, S, *_ = pokemon_embeddings.shape
        active_pokemon_embedding = pokemon_embeddings[:, :, 0, 0]

        pokemon_embeddings = pokemon_embeddings.view(T * B * N, S, -1)
        pokemon_mask = pokemon_mask.view(T * B * N, S)

        active_move_gates = self.move_gate(move_embeddings[:, :, 0, 0])
        active_move_embeddings = torch.cat(
            (
                active_pokemon_embedding.unsqueeze(-2).repeat(1, 1, 4, 1),
                active_move_gates,
            ),
            dim=-1,
        )
        active_move_embeddings = F.glu(active_move_embeddings, dim=-1)

        switch_embeddings = pokemon_embeddings.view(T, B, N, S, -1)
        switch_embeddings = switch_embeddings[:, :, 0, :6]

        pokemon_embeddings = self.compress(pokemon_embeddings)
        pokemon_embeddings = torch.where(
            pokemon_mask.unsqueeze(-1), pokemon_embeddings, -1e2
        )
        pokemon_embedding = pokemon_embeddings.max(1).values

        pokemon_embedding = pokemon_embedding.view(T, B, N, -1)
        private_embedding, player_public, opponent_public = pokemon_embedding.chunk(
            3, 2
        )

        return OrderedDict(
            private_embedding=private_embedding.squeeze(2),
            player_public=player_public.squeeze(2),
            opponent_public=opponent_public.squeeze(2),
            switch_embeddings=switch_embeddings,
            active_move_embeddings=active_move_embeddings,
        )
