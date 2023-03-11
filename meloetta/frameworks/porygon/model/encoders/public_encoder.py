import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.utils import (
    binary_enc_matrix,
    sqrt_one_hot_matrix,
    TransformerEncoder,
    PublicToVector,
)
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
    BOOSTS,
    SIDE_CONDITIONS,
    VOLATILES,
    ITEM_EFFECTS,
    BattleTypeChart,
)


class PublicPokemonEmbedding(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        n_active: int = 1,
        config: config.PublicEncoderConfig = None,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.gen = gen

        ability_embedding = AbilityEmbedding(gen=gen)
        self.ability_embedding = nn.Sequential(
            ability_embedding,
            nn.Linear(ability_embedding.embedding_dim, output_dim),
        )
        pokedex_embedding = PokedexEmbedding(gen=gen)
        self.pokedex_embedding = nn.Sequential(
            pokedex_embedding,
            nn.Linear(pokedex_embedding.embedding_dim, output_dim),
        )
        n_item_effects = len(ITEM_EFFECTS) + 1
        item_embedding = ItemEmbedding(gen=gen)
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(n_item_effects)
        )
        self.item_embedding = nn.Sequential(
            item_embedding,
            nn.Linear(item_embedding.embedding_dim, output_dim),
        )
        self.item_lin = nn.Linear(
            2 * (output_dim + self.item_effect_onehot.embedding_dim),
            output_dim,
        )

        self.move_embedding = MoveEmbedding(gen=gen)
        self.pp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(67))
        self.moveset_embedding = nn.Linear(
            self.pp_sqrt_onehot.embedding_dim + self.move_embedding.embedding_dim,
            output_dim,
        )
        self.move_embedding_prev = nn.Linear(
            self.move_embedding.embedding_dim, output_dim
        )

        n_genders = len(GENDERS) + 1
        n_formes = len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1

        self.sideid_embedding = nn.Embedding(3, config.entity_embedding_dim)
        # self.active_embedding = nn.Embedding.from_pretrained(torch.eye(2))
        # self.slot_embedding = nn.Embedding.from_pretrained(torch.eye(n_active + 1))
        self.fainted_embedding = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_embedding = nn.Embedding.from_pretrained(torch.eye(n_genders))
        self.forme_embedding = nn.Embedding.from_pretrained(torch.eye(n_formes))
        self.hp_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(101))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.toxic_onehot = nn.Embedding.from_pretrained(torch.eye(17))
        self.sleep_onehot = nn.Embedding.from_pretrained(torch.eye(5))

        # self.level_embedding = nn.Embedding.from_pretrained(binary_enc_matrix(102))

        other_stat_size = (
            # self.active_embedding.embedding_dim
            # + self.slot_embedding.embedding_dim
            self.fainted_embedding.embedding_dim
            + self.gender_embedding.embedding_dim
            + self.forme_embedding.embedding_dim
            + self.hp_onehot.embedding_dim
            + 1
            + self.status_onehot.embedding_dim
            + self.toxic_onehot.embedding_dim
            + self.sleep_onehot.embedding_dim
            + output_dim
        )

        if gen == 9:
            self.times_attacked_embedding = nn.Embedding.from_pretrained(torch.eye(8))
            self.teratype_embedding = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
            other_stat_size += (
                self.times_attacked_embedding.embedding_dim
                + self.teratype_embedding.embedding_dim
            )

        self.other_stats = nn.Sequential(
            nn.Linear(other_stat_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.embedding = nn.Sequential(
            nn.Linear(64 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, config.entity_embedding_dim),
        )

    def embed_hp(self, hp_ratio: torch.Tensor) -> torch.Tensor:
        hp_ratio = hp_ratio * (hp_ratio >= 0)
        hp_ratio = hp_ratio.clamp(min=0, max=1) * 100
        hp_emb = self.hp_onehot(hp_ratio.long())
        return torch.cat([hp_ratio.unsqueeze(-1), hp_emb], dim=-1)

    def embed_ability(self, current_and_base_ability: torch.Tensor) -> torch.Tensor:
        current_and_base_ability_embedding = self.ability_embedding(
            current_and_base_ability[..., 0]
        )
        return current_and_base_ability_embedding

    def embed_item(self, item: torch.Tensor) -> torch.Tensor:
        curr_item, curr_item_effect, prev_item, prev_item_effect = item.chunk(4, -1)
        item_embedding = torch.cat(
            (
                F.relu(self.item_embedding(curr_item)),
                self.item_effect_onehot(curr_item_effect),
                F.relu(self.item_embedding(prev_item)),
                self.item_effect_onehot(prev_item_effect),
            ),
            dim=-1,
        )
        return self.item_lin(item_embedding).squeeze(-2)

    def embed_status(
        self,
        status: torch.Tensor,
        status_stage: torch.Tensor,
        sleep_turns: torch.Tensor,
        toxic_turns: torch.Tensor,
    ) -> torch.Tensor:
        status_onehot = torch.cat(
            (
                self.status_onehot(status),
                self.sleep_onehot(sleep_turns),
                self.toxic_onehot(toxic_turns),
            ),
            dim=-1,
        )
        return status_onehot

    def embed_last_move(self, last_move: torch.Tensor) -> torch.Tensor:
        last_move = self.move_embedding(last_move)
        return self.move_embedding_prev(last_move)

    def embed_moveset(
        self, moves_id: torch.Tensor, moves_pp: torch.Tensor
    ) -> torch.Tensor:
        moves_id_emb = self.move_embedding(moves_id)
        moves_pp_emb = self.pp_sqrt_onehot(moves_pp)
        moves_emb = torch.cat((moves_id_emb, moves_pp_emb), dim=-1)
        moves = self.moveset_embedding(moves_emb)
        return moves.mean(-2)

    def embed_active(self, active: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        active = active.long()
        active_x = active + 1

        species = active_x[..., 0]
        forme = active_x[..., 1]
        slot = active_x[..., 2]
        hp = active[..., 3]
        fainted = active_x[..., 4]
        level = active_x[..., 5]
        gender = active_x[..., 6]
        current_and_base_ability = active_x[..., 7:9]
        item_features = active_x[..., 9:13]
        status = active_x[..., 14]
        status_stage = active_x[..., 15]
        last_move = active_x[..., 16]
        sleep_turns = active[..., 18].clamp(min=0)
        toxic_turns = active[..., 19].clamp(min=0)
        boosts = active[..., 20 : 20 + len(BOOSTS)] + 6
        volatiles = (
            active[..., 20 + len(BOOSTS) : 20 + len(BOOSTS) + len(VOLATILES)]
            .clamp(min=0)
            .float()
        )
        sideid = active_x[..., -17]

        moves = active_x[..., -16:]
        moves = moves.view(*moves.shape[:-1], 8, 2)
        moves_id = moves[..., 0]
        moves_pp = moves[..., 1]

        entity_embed = [
            self.pokedex_embedding(species),
            self.embed_ability(current_and_base_ability),
            self.embed_item(item_features),
            self.embed_moveset(moves_id, moves_pp),
        ]
        other_stats = [
            self.forme_embedding(forme),
            self.fainted_embedding(fainted),
            self.gender_embedding(gender),
            self.embed_hp(hp),
            self.embed_status(status, status_stage, sleep_turns, toxic_turns),
            self.embed_last_move(last_move),
        ]
        if self.gen == 9:
            terastallized = active_x[..., 13]
            times_attacked = active_x[..., 17]
            other_stats += [self.teratype_embedding(terastallized)]
            other_stats += [self.times_attacked_embedding(times_attacked)]

        mask = (species == 0) | (fainted == 2) | ~(slot).bool()

        other_stats = torch.cat(other_stats, dim=-1)
        entity_embed += [self.other_stats(other_stats)]
        entity_embed = torch.cat(entity_embed, dim=-1)

        return entity_embed, mask, sideid, boosts, volatiles

    def embed_reserve(self, reserve: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reserve = reserve.long()
        reserve_x = reserve + 1

        species = reserve_x[..., 0]
        forme = reserve_x[..., 1]
        slot = reserve_x[..., 2]
        hp = reserve[..., 3]
        fainted = reserve_x[..., 4]
        level = reserve_x[..., 5]
        gender = reserve_x[..., 6]
        current_and_base_ability = reserve_x[..., 7:9]
        item_features = reserve_x[..., 9:13]
        status = reserve_x[..., 14]
        status_stage = reserve_x[..., 15]
        last_move = reserve_x[..., 16]
        sleep_turns = reserve[..., 18].clamp(min=0)
        toxic_turns = reserve[..., 19].clamp(min=0)
        sideid = reserve_x[..., 20]

        moves = reserve_x[..., 21:]
        moves = moves.view(*moves.shape[:-1], 8, 2)
        moves_id = moves[..., 0]
        moves_pp = moves[..., 1]

        entity_embed = [
            self.pokedex_embedding(species),
            self.embed_ability(current_and_base_ability),
            self.embed_item(item_features),
            self.embed_moveset(moves_id, moves_pp),
        ]
        other_stats = [
            self.forme_embedding(forme),
            self.fainted_embedding(fainted),
            self.gender_embedding(gender),
            self.embed_hp(hp),
            self.embed_status(status, status_stage, sleep_turns, toxic_turns),
            self.embed_last_move(last_move),
        ]
        if self.gen == 9:
            terastallized = reserve_x[..., 13]
            times_attacked = reserve_x[..., 17]
            other_stats += [self.teratype_embedding(terastallized)]
            other_stats += [self.times_attacked_embedding(times_attacked)]

        mask = (species == 0) | (fainted == 2) | ~(slot).bool()

        other_stats = torch.cat(other_stats, dim=-1)
        entity_embed += [self.other_stats(other_stats)]
        entity_embed = torch.cat(entity_embed, dim=-1)

        return entity_embed, mask, sideid

    def forward(self, active: torch.Tensor, reserve: torch.Tensor):
        (
            active_entity_embed,
            active_mask,
            active_sideid,
            boosts,
            volatiles,
        ) = self.embed_active(active)
        reserve_entity_embed, reserve_mask, reserve_sideid = self.embed_reserve(reserve)

        entity_embeddings = torch.cat(
            [
                active_entity_embed,
                reserve_entity_embed,
            ],
            dim=-2,
        )

        T, B, S, L, *_ = entity_embeddings.shape

        mask = torch.cat(
            [
                active_mask,
                reserve_mask,
            ],
            dim=-1,
        )
        mask = mask.to(torch.bool)
        mask = mask.view(T * B, S * L)

        sideid = torch.cat(
            [
                active_sideid,
                reserve_sideid,
            ],
            dim=-1,
        )
        sideid = sideid.view(T * B, S * L)

        entity_embeddings = entity_embeddings.view(T * B, S * L, -1)
        entity_embeddings = self.embedding(entity_embeddings)
        entity_embeddings += self.sideid_embedding(sideid)

        return entity_embeddings, mask, boosts, volatiles


class PublicEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PublicEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        # scalar stuff
        self.n_onehot = nn.Embedding.from_pretrained(torch.eye(7))
        self.total_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        sc_min_dur_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.sc_min_dur_onehot = nn.Sequential(sc_min_dur_onehot, nn.Flatten(-2))

        sc_max_dur_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.sc_max_dur_onehot = nn.Sequential(sc_max_dur_onehot, nn.Flatten(-2))

        self.toxic_spikes_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4))
        self.stealthrock_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.stickyweb_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        self.boosts_emb = nn.Sequential(
            nn.Embedding.from_pretrained(torch.eye(13)),
            nn.Flatten(3),
        )

        n_special_sc = 4
        normal_sc = len(SIDE_CONDITIONS) - n_special_sc
        sc_lin_in = (
            self.n_onehot.embedding_dim
            + self.n_onehot.embedding_dim
            + self.total_onehot.embedding_dim
            + self.n_onehot.embedding_dim
            + normal_sc
            + (normal_sc * sc_min_dur_onehot.embedding_dim)
            + (normal_sc * sc_min_dur_onehot.embedding_dim)
            + self.toxic_spikes_onehot.embedding_dim
            + self.spikes_onehot.embedding_dim
            + self.stealthrock_onehot.embedding_dim
            + self.stickyweb_onehot.embedding_dim
        )
        self.scalar_lin = nn.Sequential(
            nn.Linear(1268, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.scalar_embedding_dim),
        )

        self.embedding = PublicPokemonEmbedding(
            gen=gen, n_active=n_active, config=config
        )

        self.transformer = TransformerEncoder(
            model_size=config.entity_embedding_dim,
            key_size=128,
            value_size=128,
            num_heads=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
            resblocks_num_before=config.resblocks_num_before,
            resblocks_num_after=config.resblocks_num_after,
        )
        self.output = PublicToVector(
            input_dim=config.entity_embedding_dim,
            output_dim=config.output_dim,
        )

    def embed_scalars(
        self,
        n: torch.Tensor,
        total_pokemon: torch.Tensor,
        faint_counter: torch.Tensor,
        side_conditions: torch.Tensor,
        wisher: torch.Tensor,
        stealthrock: torch.Tensor,
        spikes: torch.Tensor,
        toxicspikes: torch.Tensor,
        stickyweb: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
    ) -> torch.Tensor:
        side_conditions_x = side_conditions + 1
        sc_levels = side_conditions[..., 0]
        sc_min_dur = side_conditions_x[..., 1]
        sc_max_dur = side_conditions_x[..., 2]

        scalars = torch.cat(
            [
                self.n_onehot(n),
                self.total_onehot(total_pokemon),
                self.n_onehot(faint_counter),
                self.n_onehot(wisher + 1),
                sc_levels,
                self.sc_min_dur_onehot(sc_min_dur),
                self.sc_max_dur_onehot(sc_max_dur),
                self.toxic_spikes_onehot(toxicspikes),
                self.spikes_onehot(spikes),
                self.stealthrock_onehot(stealthrock),
                self.stickyweb_onehot(stickyweb),
                self.boosts_emb(boosts),
                volatiles.flatten(3),
            ],
            dim=-1,
        ).flatten(2)
        return scalars, self.scalar_lin(scalars)

    def forward(
        self,
        n: torch.Tensor,
        total_pokemon: torch.Tensor,
        faint_counter: torch.Tensor,
        side_conditions: torch.Tensor,
        wisher: torch.Tensor,
        active: torch.Tensor,
        reserve: torch.Tensor,
        stealthrock: torch.Tensor,
        spikes: torch.Tensor,
        toxicspikes: torch.Tensor,
        stickyweb: torch.Tensor,
    ):
        T, B, *_ = n.shape

        entity_embeddings, mask, boosts, volatiles = self.embedding(active, reserve)

        raw_scalars, scalar_emb = self.embed_scalars(
            n,
            total_pokemon,
            faint_counter,
            side_conditions,
            wisher,
            stealthrock,
            spikes,
            toxicspikes,
            stickyweb,
            boosts,
            volatiles,
        )

        mask = ~mask
        encoder_mask = mask.clone()
        encoder_mask[..., 0] = torch.ones_like(encoder_mask[..., 0])
        encoder_mask = encoder_mask.to(torch.bool)

        entity_embeddings = self.transformer(entity_embeddings, encoder_mask)

        entity_embedding = self.output(entity_embeddings, mask.unsqueeze(-1))
        entity_embedding = entity_embedding.view(T, B, -1)

        return (
            entity_embedding,
            scalar_emb,
            None,
            None,
            raw_scalars,
            mask,
        )
