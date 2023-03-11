import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    sqrt_one_hot_matrix,
    binary_enc_matrix,
    TransformerEncoder,
    ToVector,
)
from meloetta.frameworks.nash_ketchum.model.encoders import public_spatial
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
        self.scalar_lin = nn.Linear(sc_lin_in, config.scalar_embedding_dim)

        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.sideid_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.slot_onehot = nn.Embedding.from_pretrained(torch.eye(n_active + 1))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_onehot = nn.Embedding.from_pretrained(torch.eye(len(GENDERS) + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)
        )
        self.times_attacked_onehot = nn.Embedding.from_pretrained(torch.eye(8))

        self.forme_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1)
        )

        # binaries
        self.level_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(102))
        self.hp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(102))
        self.toxic_bin = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(17))
        self.sleep_bin = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(5))

        status_embedding_dim = (
            self.status_onehot.embedding_dim
            + self.toxic_bin.embedding_dim
            + self.sleep_bin.embedding_dim
            + 2
        )

        # precomputed embeddings
        self.ability_embedding = AbilityEmbedding(gen=gen)
        self.pokedex_embedding = PokedexEmbedding(gen=gen)
        self.item_embedding = ItemEmbedding(gen=gen)
        item_embedding_dim = (
            2 * self.item_embedding.embedding_dim
            + 2 * self.item_effect_onehot.embedding_dim
        )

        self.move_embedding = MoveEmbedding(gen=gen)

        self.pp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(66))
        self.move_lin = nn.Linear(
            self.move_embedding.embedding_dim + self.pp_sqrt_onehot.embedding_dim,
            config.entity_embedding_dim,
        )

        hp_embedding_dim = self.hp_sqrt_onehot.embedding_dim + 1
        level_embedding_dim = self.level_sqrt_onehot.embedding_dim + 1

        base_embedding_size = (
            self.pokedex_embedding.embedding_dim
            + self.active_onehot.embedding_dim
            + self.sideid_onehot.embedding_dim
            + self.forme_onehot.embedding_dim
            + self.slot_onehot.embedding_dim
            + hp_embedding_dim
            + self.fainted_onehot.embedding_dim
            + level_embedding_dim
            + self.gender_onehot.embedding_dim
            + 2 * self.ability_embedding.embedding_dim
            + item_embedding_dim
            + status_embedding_dim
            + self.move_embedding.embedding_dim
            + self.times_attacked_onehot.embedding_dim
        )
        if self.gen == 9:
            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
            base_embedding_size += self.teratype_onehot.embedding_dim

        self.entity_lin = nn.Linear(base_embedding_size, config.entity_embedding_dim)

        self.transformer = TransformerEncoder(
            key_size=config.entity_embedding_dim,
            value_size=config.entity_embedding_dim,
            num_heads=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
            resblocks_num_before=config.resblocks_num_before,
            resblocks_num_after=config.resblocks_num_after,
        )
        self.output = ToVector(
            input_dim=config.entity_embedding_dim,
            output_dim=2 * config.entity_embedding_dim,
        )
        self.spatial = public_spatial.PublicSpatialEncoder(
            gen=gen, n_active=n_active, config=config
        )

    def embed_hp(self, hp_ratio: torch.Tensor) -> torch.Tensor:
        hp_ratio = hp_ratio * (hp_ratio >= 0)
        hp_cat = hp_ratio.clamp(min=0) * 100
        hp_emb = self.hp_sqrt_onehot(hp_cat)
        hp_emb = torch.cat([hp_ratio.unsqueeze(-1), hp_emb], dim=-1)
        return hp_emb

    def embed_level(self, level: torch.Tensor) -> torch.Tensor:
        level_bin = self.level_sqrt_onehot(level)
        level_ratio = (level / 100).unsqueeze(-1)
        level_emb = torch.cat((level_bin, level_ratio), dim=-1)
        return level_emb

    def embed_pp(self, pp: torch.Tensor) -> torch.Tensor:
        return self.pp_sqrt_onehot(pp)

    def embed_ability(self, current_and_base_ability: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.ability_embedding(current_and_base_ability), -2)

    def embed_item(
        self,
        curr_item: torch.Tensor,
        curr_item_effect: torch.Tensor,
        prev_item: torch.Tensor,
        prev_item_effect: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = torch.cat(
            (
                self.item_embedding(curr_item),
                self.item_embedding(prev_item),
                self.item_effect_onehot(curr_item_effect),
                self.item_effect_onehot(prev_item_effect),
            ),
            dim=-1,
        )
        return item_emb

    def embed_status(
        self,
        status: torch.Tensor,
        status_stage: torch.Tensor,
        sleep_turns: torch.Tensor,
        toxic_turns: torch.Tensor,
    ) -> torch.Tensor:
        status_emb = torch.cat(
            (
                self.status_onehot(status),
                (sleep_turns / 5).unsqueeze(-1),
                self.sleep_bin(sleep_turns),
                (toxic_turns / 17).unsqueeze(-1),
                self.toxic_bin(toxic_turns),
            ),
            dim=-1,
        )
        return status_emb

    def embed_last_move(self, last_move: torch.Tensor) -> torch.Tensor:
        return self.move_embedding(last_move)

    def embed_moves(
        self, moves_id: torch.Tensor, moves_pp: torch.Tensor
    ) -> torch.Tensor:
        moves_id_emb = self.move_embedding(moves_id)
        moves_pp_emb = self.embed_pp(moves_pp)
        moves_emb = torch.cat((moves_id_emb, moves_pp_emb), dim=-1)
        return self.move_lin(moves_emb)

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
            ],
            dim=-1,
        )
        return self.scalar_lin(scalars)

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
        item = active_x[..., 9]
        item_effect = active_x[..., 10]
        prev_item = active_x[..., 11]
        prev_item_effect = active_x[..., 12]
        terastallized = active_x[..., 13]
        status = active_x[..., 14]
        status_stage = active_x[..., 15]
        last_move = active_x[..., 16]
        times_attacked = active_x[..., 17]
        sleep_turns = active_x[..., 18]
        toxic_turns = active_x[..., 19]
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
        moves_emb = self.embed_moves(moves_id, moves_pp)

        entity_embed = [
            self.pokedex_embedding(species),
            self.active_onehot(torch.zeros_like(slot)),
            self.sideid_onehot(sideid),
            self.forme_onehot(forme),
            self.slot_onehot(slot),
            self.embed_hp(hp),
            self.fainted_onehot(fainted),
            self.embed_level(level),
            self.gender_onehot(gender),
            self.embed_ability(current_and_base_ability),
            self.embed_item(
                item,
                item_effect,
                prev_item,
                prev_item_effect,
            ),
            self.embed_status(
                status,
                status_stage,
                sleep_turns,
                toxic_turns,
            ),
            self.embed_last_move(last_move),
            self.times_attacked_onehot(times_attacked),
        ]
        if self.gen == 9:
            entity_embed += [self.teratype_onehot(terastallized)]

        entity_embed = torch.cat(entity_embed, dim=-1)
        entity_embed = self.entity_lin(entity_embed)
        entity_embed = entity_embed + moves_emb.sum(-2)

        mask = (species == 0) | (fainted == 2)

        return entity_embed, mask, boosts, volatiles

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
        curr_item = reserve_x[..., 9]
        curr_item_effect = reserve_x[..., 10]
        prev_item = reserve_x[..., 11]
        prev_item_effect = reserve_x[..., 12]
        terastallized = reserve_x[..., 13]
        status = reserve_x[..., 14]
        status_stage = reserve_x[..., 15]
        last_move = reserve_x[..., 16]
        times_attacked = reserve_x[..., 17]
        sleep_turns = reserve_x[..., 18]
        toxic_turns = reserve_x[..., 18]
        sideid = reserve_x[..., 20]

        moves = reserve_x[..., 21:]
        moves = moves.view(*moves.shape[:-1], 8, 2)
        moves_id = moves[..., 0]
        moves_pp = moves[..., 1]
        moves_emb = self.embed_moves(moves_id, moves_pp)

        entity_embed = [
            self.pokedex_embedding(species),
            self.active_onehot(torch.ones_like(slot)),
            self.sideid_onehot(sideid),
            self.forme_onehot(forme),
            self.slot_onehot(slot),
            self.embed_hp(hp),
            self.fainted_onehot(fainted),
            self.embed_level(level),
            self.gender_onehot(gender),
            self.embed_ability(current_and_base_ability),
            self.embed_item(
                curr_item,
                curr_item_effect,
                prev_item,
                prev_item_effect,
            ),
            self.embed_status(
                status,
                status_stage,
                sleep_turns,
                toxic_turns,
            ),
            self.embed_last_move(last_move),
            self.times_attacked_onehot(times_attacked),
        ]
        if self.gen == 9:
            entity_embed += [self.teratype_onehot(terastallized)]

        entity_embed = torch.cat(entity_embed, dim=-1)
        entity_embed = self.entity_lin(entity_embed)
        entity_embed = entity_embed + moves_emb.sum(-2)

        mask = (species == 0) | (fainted == 2)

        return entity_embed, mask

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
        scalar_emb = self.embed_scalars(
            n,
            total_pokemon,
            faint_counter,
            side_conditions,
            wisher,
            stealthrock,
            spikes,
            toxicspikes,
            stickyweb,
        )

        active_entity_embed, active_mask, boosts, volatiles = self.embed_active(active)
        reserve_entity_embed, reserve_mask = self.embed_reserve(reserve)

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
        mask = mask.bool()
        mask = mask.view(T * B, S * L)

        encoder_mask = mask.clone()
        encoder_mask[..., 0] = False

        entity_embeddings: torch.Tensor = entity_embeddings.view(T * B, S * L, -1)
        entity_embeddings = self.transformer(entity_embeddings, encoder_mask)

        mask = ~mask.unsqueeze(-1)
        entity_embeddings = entity_embeddings * mask
        entity_embedding = self.output(entity_embeddings)

        entity_embeddings = entity_embeddings.view(T * B, S, L, -1)
        mask = mask.view(T * B, S, L, -1)

        spatial_embedding = self.spatial(
            entity_embeddings, mask, boosts, volatiles, scalar_emb
        )
        spatial_embedding = spatial_embedding.view(T, B, -1)

        entity_embedding = entity_embedding.view(T, B, -1)

        return entity_embedding, spatial_embedding


class PublicEncoderV2(nn.Module):
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
        self.scalar_lin = nn.Linear(sc_lin_in, config.scalar_embedding_dim)

        self.active_embedding = nn.Embedding(2, config.entity_embedding_dim)

        self.sideid_embedding = nn.Embedding(
            3, config.entity_embedding_dim, padding_idx=0
        )
        self.slot_embedding = nn.Embedding(n_active + 1, config.entity_embedding_dim)
        self.fainted_embedding = nn.Embedding(3, config.entity_embedding_dim)
        self.gender_embedding = nn.Embedding(
            len(GENDERS) + 1, config.entity_embedding_dim
        )

        self.forme_embedding = nn.Embedding(
            len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1,
            config.entity_embedding_dim,
            padding_idx=0,
        )

        # binaries
        self.level_embedding = nn.Embedding(102, config.entity_embedding_dim)

        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(101))
        self.hp_embedding = nn.Linear(
            self.hp_onehot.embedding_dim + 1, config.entity_embedding_dim, bias=False
        )

        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.toxic_onehot = nn.Embedding.from_pretrained(torch.eye(17))
        self.sleep_onehot = nn.Embedding.from_pretrained(torch.eye(5))
        self.status_embedding = nn.Linear(
            self.status_onehot.embedding_dim
            + self.toxic_onehot.embedding_dim
            + self.sleep_onehot.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )

        # ability embeddings
        self.ability_embedding = AbilityEmbedding(gen=gen)
        self.current_ability_lin = nn.Linear(
            self.ability_embedding.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )
        self.base_ability_lin = nn.Linear(
            self.ability_embedding.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )

        # pokedex embeddings
        pokedex_embedding = PokedexEmbedding(gen=gen)
        self.pokedex_embedding = nn.Sequential(
            pokedex_embedding,
            nn.Linear(
                pokedex_embedding.embedding_dim, config.entity_embedding_dim, bias=False
            ),
        )

        # item embeddings
        self.item_embedding = ItemEmbedding(gen=gen)
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)
        )
        self.curr_item_lin = nn.Linear(
            self.item_effect_onehot.embedding_dim + self.item_embedding.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )
        self.prev_item_lin = nn.Linear(
            self.item_effect_onehot.embedding_dim + self.item_embedding.embedding_dim,
            config.entity_embedding_dim,
            bias=False,
        )

        self.move_embedding = MoveEmbedding(gen=gen)

        self.pp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(67))
        self.moveset_lin = nn.Linear(
            self.move_embedding.embedding_dim + self.pp_bin.embedding_dim,
            config.entity_embedding_dim,
        )
        self.prev_move_lin = nn.Linear(
            self.move_embedding.embedding_dim,
            config.entity_embedding_dim,
        )

        if self.gen == 9:
            self.times_attacked_embedding = nn.Embedding(8, config.entity_embedding_dim)
            self.teratype_embedding = nn.Embedding(
                len(BattleTypeChart) + 1, config.entity_embedding_dim
            )

        self.transformer = TransformerEncoder(
            key_size=config.entity_embedding_dim,
            value_size=config.entity_embedding_dim,
            num_heads=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
            resblocks_num_before=config.resblocks_num_before,
            resblocks_num_after=config.resblocks_num_after,
        )
        self.output = ToVector(
            input_dim=config.entity_embedding_dim,
            output_dim=config.output_dim,
        )
        self.spatial = public_spatial.PublicSpatialEncoder(
            gen=gen, n_active=n_active, config=config
        )

    def embed_hp(self, hp_ratio: torch.Tensor) -> torch.Tensor:
        hp_ratio = hp_ratio * (hp_ratio >= 0)
        hp_ratio = hp_ratio.clamp(min=0, max=1) * 100
        hp_emb = self.hp_onehot(hp_ratio.long())
        hp_cat = torch.cat([hp_ratio.unsqueeze(-1), hp_emb], dim=-1)
        return self.hp_embedding(hp_cat)

    def embed_level(self, level: torch.Tensor) -> torch.Tensor:
        return self.level_embedding(level)

    def embed_pp(self, pp: torch.Tensor) -> torch.Tensor:
        return self.pp_bin(pp)

    def embed_ability(self, current_and_base_ability: torch.Tensor) -> torch.Tensor:
        current_ability_token, base_ability_token = torch.chunk(
            current_and_base_ability, 2, -1
        )
        current_ability_embedding, base_ability_embedding = torch.chunk(
            self.ability_embedding(current_and_base_ability), 2, -2
        )
        current_ability_embedding = self.current_ability_lin(
            current_ability_embedding.squeeze(-2)
        )
        current_ability_embedding *= current_ability_token > 0
        # base_ability_embedding = self.base_ability_lin(
        #     base_ability_embedding.squeeze(-2)
        # )
        # base_ability_embedding *= base_ability_token > 0

        return current_ability_embedding  # base_ability_embedding

    def embed_item(
        self,
        curr_item: torch.Tensor,
        curr_item_effect: torch.Tensor,
        prev_item: torch.Tensor,
        prev_item_effect: torch.Tensor,
    ) -> torch.Tensor:
        curr_item_cat = torch.cat(
            (self.item_embedding(curr_item), self.item_effect_onehot(curr_item_effect)),
            dim=-1,
        )
        curr_item_embedding = self.curr_item_lin(curr_item_cat)
        curr_item_embedding = curr_item_embedding * (curr_item > 0).unsqueeze(-1)
        prev_item_cat = torch.cat(
            (self.item_embedding(prev_item), self.item_effect_onehot(prev_item_effect)),
            dim=-1,
        )
        prev_item_embedding = self.prev_item_lin(prev_item_cat) * (
            prev_item > 0
        ).unsqueeze(-1)
        return curr_item_embedding + prev_item_embedding

    def embed_status(
        self,
        status: torch.Tensor,
        status_stage: torch.Tensor,
        sleep_turns: torch.Tensor,
        toxic_turns: torch.Tensor,
    ) -> torch.Tensor:
        status_concat_onehot = torch.cat(
            (
                self.status_onehot(status),
                self.sleep_onehot(sleep_turns),
                self.toxic_onehot(toxic_turns),
            ),
            dim=-1,
        )
        status_embedding = self.status_embedding(status_concat_onehot)
        return status_embedding * (status > 0).unsqueeze(-1)

    def embed_last_move(self, last_move: torch.Tensor) -> torch.Tensor:
        last_move_onehot = self.move_embedding(last_move)
        return self.prev_move_lin(last_move_onehot)

    def embed_moveset(
        self, moves_id: torch.Tensor, moves_pp: torch.Tensor
    ) -> torch.Tensor:
        moves_id_emb = self.move_embedding(moves_id)
        moves_pp_emb = self.embed_pp(moves_pp)
        moves_emb = torch.cat((moves_id_emb, moves_pp_emb), dim=-1)
        moves = self.moveset_lin(moves_emb)
        moves = moves * (moves_id > 0).unsqueeze(-1)
        return moves.sum(-2)

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
            ],
            dim=-1,
        )
        return self.scalar_lin(scalars)

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
        item = active_x[..., 9]
        item_effect = active_x[..., 10]
        prev_item = active_x[..., 11]
        prev_item_effect = active_x[..., 12]
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

        entity_embed = (
            self.pokedex_embedding(species)
            + self.active_embedding(torch.zeros_like(slot))
            + self.sideid_embedding(sideid)
            + self.forme_embedding(forme)
            + self.fainted_embedding(fainted)
            + self.gender_embedding(gender)
            + self.embed_hp(hp)
            + self.embed_level(level)
            + self.embed_ability(current_and_base_ability)
            + self.embed_item(item, item_effect, prev_item, prev_item_effect)
            + self.embed_status(status, status_stage, sleep_turns, toxic_turns)
            + self.embed_last_move(last_move)
            + self.embed_moveset(moves_id, moves_pp)
        )
        if self.gen == 9:
            terastallized = active_x[..., 13]
            times_attacked = active_x[..., 17]
            entity_embed += self.teratype_embedding(terastallized)
            entity_embed += self.times_attacked_embedding(times_attacked)

        entity_embed = entity_embed * slot.unsqueeze(-1)
        mask = (species == 0) | (fainted == 2) | ~(slot).bool()

        return entity_embed, mask, boosts, volatiles

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
        curr_item = reserve_x[..., 9]
        curr_item_effect = reserve_x[..., 10]
        prev_item = reserve_x[..., 11]
        prev_item_effect = reserve_x[..., 12]
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

        entity_embed = (
            self.pokedex_embedding(species)
            + self.active_embedding(torch.ones_like(slot))
            + self.sideid_embedding(sideid)
            + self.forme_embedding(forme)
            + self.fainted_embedding(fainted)
            + self.gender_embedding(gender)
            + self.embed_hp(hp)
            + self.embed_level(level)
            + self.embed_ability(current_and_base_ability)
            + self.embed_item(curr_item, curr_item_effect, prev_item, prev_item_effect)
            + self.embed_status(status, status_stage, sleep_turns, toxic_turns)
            + self.embed_last_move(last_move)
            + self.embed_moveset(moves_id, moves_pp)
        )

        if self.gen == 9:
            terastallized = reserve_x[..., 13]
            times_attacked = reserve_x[..., 17]
            entity_embed += self.teratype_embedding(terastallized)
            entity_embed += self.times_attacked_embedding(times_attacked)

        entity_embed = entity_embed * slot.unsqueeze(-1)
        mask = (species == 0) | (fainted == 2) | ~(slot).bool()

        return entity_embed, mask

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
        scalar_emb = self.embed_scalars(
            n,
            total_pokemon,
            faint_counter,
            side_conditions,
            wisher,
            stealthrock,
            spikes,
            toxicspikes,
            stickyweb,
        )

        active_entity_embed, active_mask, boosts, volatiles = self.embed_active(active)
        reserve_entity_embed, reserve_mask = self.embed_reserve(reserve)

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

        encoder_mask = mask.clone()
        encoder_mask[..., 0] = torch.zeros_like(encoder_mask[..., 0])
        encoder_mask = encoder_mask.to(torch.bool)

        entity_embeddings: torch.Tensor = entity_embeddings.view(T * B, S * L, -1)
        entity_embeddings = self.transformer(entity_embeddings, encoder_mask)

        mask = ~mask.unsqueeze(-1)
        entity_embeddings = entity_embeddings * mask
        entity_embedding = self.output(entity_embeddings)

        entity_embeddings = entity_embeddings.view(T * B, S, L, -1)
        mask = mask.view(T * B, S, L, -1)

        spatial_embedding = self.spatial(
            entity_embeddings, mask, boosts, volatiles, scalar_emb
        )
        spatial_embedding = spatial_embedding.view(T, B, -1)

        entity_embedding = entity_embedding.view(T, B, -1)

        return entity_embedding, spatial_embedding
