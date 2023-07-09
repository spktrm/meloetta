import torch
import torch.nn.functional as F

from torch import nn

from collections import OrderedDict
from typing import Tuple

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
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

        pokedex_onehot = PokedexEmbedding(gen=gen)
        num_species = pokedex_onehot.num_embeddings
        self.species_onehot = nn.Embedding.from_pretrained(
            torch.eye(num_species)[..., 1:]
        )
        self.pokedex_embedding = linear_layer(
            num_species - 1, config.entity_embedding_dim
        )

        self.ability_onehot = AbilityEmbedding(gen=gen)
        self.ability_embedding_known = nn.Embedding(
            self.ability_onehot.num_embeddings, config.entity_embedding_dim
        )
        self.ability_embedding_unknown = MLP(
            [config.entity_embedding_dim, config.entity_embedding_dim]
        )

        item_onehot = ItemEmbedding(gen=gen)
        self.item_onehot = nn.Embedding.from_pretrained(
            torch.eye(item_onehot.num_embeddings)[..., 1:]
        )
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)[..., 1:]
        )
        self.item_embedding_known = linear_layer(
            self.item_onehot.embedding_dim + self.item_effect_onehot.embedding_dim,
            config.entity_embedding_dim,
        )
        self.item_embedding_unknown = MLP(
            [config.entity_embedding_dim, config.entity_embedding_dim]
        )

        move_onehot = MoveEmbedding(gen=gen)
        self.pp_bin_enc = nn.Embedding.from_pretrained(binary_enc_matrix(64))
        self.move_raw_onehot = nn.Embedding.from_pretrained(
            torch.eye(move_onehot.num_embeddings)[..., 1:]
        )
        self.move_embedding = linear_layer(
            self.move_raw_onehot.embedding_dim + self.pp_bin_enc.embedding_dim,
            config.entity_embedding_dim,
        )
        self.moveset_onehot = linear_layer(
            self.move_raw_onehot.embedding_dim, config.entity_embedding_dim
        )
        self.unknown_move_embedding = MLP(
            [config.entity_embedding_dim, config.entity_embedding_dim]
        )

        self.last_move_embedding = linear_layer(
            move_onehot.embedding_dim + self.pp_bin_enc.embedding_dim,
            config.entity_embedding_dim,
        )

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

            teratype_onehot = torch.eye(len(BattleTypeChart) + 1)[..., 1:]
            teratype_onehot[0] = torch.ones_like(teratype_onehot[0]) / len(
                BattleTypeChart
            )
            teratype_onehot = nn.Embedding.from_pretrained(teratype_onehot)
            self.teratype_onehot = nn.Sequential(
                teratype_onehot,
                linear_layer(len(BattleTypeChart), config.entity_embedding_dim),
            )

            onehot_size += self.tera_onehot.embedding_dim

        elif gen == 8:
            self.can_gmax_embedding = nn.Embedding.from_pretrained(torch.eye(3))

            onehot_size += self.can_gmax_embedding.embedding_dim

        self.onehots_lin = linear_layer(onehot_size, config.entity_embedding_dim)

    def embed_species(self, species_token: torch.Tensor):
        unrevealed = (species_token[..., :6] == 0).sum(-1, keepdim=True).unsqueeze(-1)
        known_onehot = (
            self.species_onehot(species_token)
            .sum(-2, keepdim=True)
            .repeat(1, 1, 1, 12, 1)
        )
        unknown_probs = (1 - known_onehot) / unrevealed.clamp(min=1)
        known_probs = self.species_onehot(species_token)
        species_mask = (species_token > 0).unsqueeze(-1)
        species_onehot = torch.where(species_mask, known_probs, unknown_probs)
        return self.pokedex_embedding(species_onehot)

    def embed_ability(
        self, ability_token: torch.Tensor, species_embedding: torch.Tensor
    ):
        known = self.ability_embedding_known(ability_token)
        unknown = self.ability_embedding_unknown(species_embedding)
        unknown_mask = (ability_token > 0).unsqueeze(-1)
        return torch.where(unknown_mask, known, unknown)

    def embed_item(
        self,
        item_token: torch.Tensor,
        item_effect_token: torch.Tensor,
        species_embedding: torch.Tensor,
    ) -> torch.Tensor:
        item_concat = torch.cat(
            (
                self.item_onehot(item_token),
                self.item_effect_onehot(item_effect_token),
            ),
            dim=-1,
        )
        known_item_embedding = self.item_embedding_known(item_concat)
        unknown_item_embedding = self.item_embedding_unknown(species_embedding)
        known_mask = (item_token > 0).unsqueeze(-1)
        return torch.where(known_mask, known_item_embedding, unknown_item_embedding)

    def embed_moveset(
        self,
        move_tokens: torch.Tensor,
        pp_token: torch.Tensor,
        species_embedding: torch.Tensor,
    ) -> torch.Tensor:
        move_concat = torch.cat(
            (
                self.move_raw_onehot(move_tokens),
                self.pp_bin_enc(pp_token),
            ),
            dim=-1,
        )
        known_mask = (move_tokens > 0).unsqueeze(-1)

        known_move_embedding = self.move_embedding(move_concat)

        moveset_onehot = self.move_raw_onehot(move_tokens).sum(-2)
        moveset_onehot = self.moveset_onehot(moveset_onehot)
        unknown_move_embedding = self.unknown_move_embedding(
            moveset_onehot + species_embedding
        ).unsqueeze(-2)

        moveset_embedding = torch.where(
            known_mask, known_move_embedding, unknown_move_embedding
        )

        return moveset_embedding.sum(-2), known_move_embedding

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        longs = (x + 1).long()

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

        species_embedding = self.embed_species(species_token)

        hp_embedding = self.hp_onehot((hp_ratio * 10).clamp(min=0, max=10).long())
        stat_enc = hp_embedding

        # stat_embedding = self.stat_lin(
        #     hp_ratio.unsqueeze(-1)
        #     # torch.cat(
        #     #     (, stats / 512 * (stats >= 0) + -1 * (stats < 0)),
        #     #     dim=-1,
        #     # )
        # )

        ability_embedding = self.embed_ability(ability_token, species_embedding)
        # base_ability_emb = self.ability_embedding(base_ability)

        item_embedding = self.embed_item(
            item_token, item_effect_token, species_embedding
        )
        # prev_item_emb = self.embed_item(prev_item, prev_item_effect)

        status_onehot = self.status_onehot(status_token)
        sleep_turns_onehot = self.sleep_turns_onehot(sleep_turns)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns)

        moveset_embedding, move_embeddings = self.embed_moveset(
            move_tokens, pp_tokens, species_embedding
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
            embeddings += [self.teratype_onehot(teratype)]
            onehots += [self.tera_onehot((terastallized > 0).long())]

        onehots = torch.cat(onehots, dim=-1)
        onehots_embedding = self.onehots_lin(onehots)
        embeddings += [onehots_embedding]

        pokemon_emb = sum(embeddings)

        mask = torch.zeros_like(species_token, dtype=torch.bool)  # | (fainted == 2)
        # mask = species_token == 0  | (fainted_token == 2)
        mask[..., 6:] = True
        # mask[..., 0, :] = True
        mask = ~mask

        pokemon_emb = pokemon_emb * mask.unsqueeze(-1)

        return pokemon_emb, mask, move_embeddings


class SideEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.SideEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.embedding = PokemonEmbedding(gen=gen, config=config)

        self.transformer = TransformerEncoder(
            model_size=config.entity_embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            key_size=config.key_size,
            value_size=config.value_size,
            resblocks_num_before=config.resblocks_num_before,
            resblocks_num_after=config.resblocks_num_after,
            resblocks_hidden_size=config.resblocks_hidden_size,
        )

        self.output = ToVector(
            input_dim=config.entity_embedding_dim,
            hidden_dim=2 * config.entity_embedding_dim,
            output_dim=config.output_dim,
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

        pokemon_embeddings = pokemon_embeddings.view(T * B, N * S, -1)
        pokemon_mask = pokemon_mask.view(T * B, N * S)

        pokemon_embeddings = self.transformer(pokemon_embeddings, pokemon_mask)
        pokemon_embeddings = pokemon_embeddings * pokemon_mask.unsqueeze(-1)

        pokemon_embedding = self.output(pokemon_embeddings, pokemon_mask)

        pokemon_embeddings = pokemon_embeddings.view(T, B, N, S, -1)
        pokemon_embedding = pokemon_embedding.view(T, B, -1)

        active_pokemon_embedding = pokemon_embeddings[:, :, 0, 0]
        switch_embeddings = pokemon_embeddings[:, :, 0, :6]

        active_move_gates = self.move_gate(move_embeddings[:, :, 0, 0])
        active_move_embeddings = torch.cat(
            (
                active_pokemon_embedding.unsqueeze(-2).repeat(1, 1, 4, 1),
                active_move_gates,
            ),
            dim=-1,
        )
        active_move_embeddings = F.glu(active_move_embeddings, dim=-1)

        return OrderedDict(
            pokemon_embedding=pokemon_embedding,
            pokemon_embeddings=pokemon_embeddings,
            move_embeddings=move_embeddings,
            switch_embeddings=switch_embeddings,
            active_pokemon_embedding=active_pokemon_embedding,
            active_move_embeddings=active_move_embeddings,
        )
