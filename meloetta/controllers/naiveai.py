import math
import json
import torch
import torch.nn as nn

from copy import deepcopy
from typing import NamedTuple, Optional, Literal, Tuple, Dict, Any

from meloetta.client import Client
from meloetta.room import BattleRoom

from meloetta.buffers.buffer import ReplayBuffer
from meloetta.controllers.base import Controller
from meloetta.controllers.types import State, Choices

from meloetta.embeddings import (
    AbilityEmbedding,
    PokedexEmbedding,
    MoveEmbedding,
    ItemEmbedding,
)
from meloetta.data import (
    CHOICE_FLAGS,
    CHOICE_TARGETS,
    CHOICE_TOKENS,
    WEATHERS,
    PSEUDOWEATHERS,
    GENDERS,
    STATUS,
    TOKENIZED_SCHEMA,
    BOOSTS,
    SIDE_CONDITIONS,
    VOLATILES,
    ITEM_EFFECTS,
    _STATE_FIELDS,
    BattleTypeChart,
)


def binary_enc_matrix(num_embeddings: int):
    bits = math.ceil(math.log2(num_embeddings))
    mask = 2 ** torch.arange(bits)
    x = torch.arange(mask.sum().item() + 1)
    embs = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return embs[..., :num_embeddings:, :]


def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdim=True).values
    logits = torch.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdim=True).values
    logits *= legal_actions
    exp_logits = torch.where(
        legal_actions, torch.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = torch.sum(exp_logits, axis=-1, keepdim=True)
    policy = exp_logits / exp_logits_sum
    return policy


def _log_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """Return the log of the policy on legal action, 0 on illegal action."""
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + torch.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = torch.exp(logits_masked)

    baseline = torch.log(torch.sum(exp_logits_masked, axis=-1, keepdim=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = torch.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


class Indices(NamedTuple):
    action_type_index: torch.Tensor
    move_index: torch.Tensor
    max_move_index: torch.Tensor
    switch_index: torch.Tensor
    flag_index: torch.Tensor
    target_index: torch.Tensor


class Logits(NamedTuple):
    action_type_logits: torch.Tensor
    move_logits: torch.Tensor
    max_move_logits: torch.Tensor
    switch_logits: torch.Tensor
    flag_logits: torch.Tensor
    target_logits: torch.Tensor


class Policy(NamedTuple):
    action_type_policy: torch.Tensor
    move_policy: torch.Tensor
    max_move_policy: torch.Tensor
    switch_policy: torch.Tensor
    flag_policy: torch.Tensor
    target_policy: torch.Tensor


class LogPolicy(NamedTuple):
    action_type_log_policy: torch.Tensor
    move_log_policy: torch.Tensor
    max_move_log_policy: torch.Tensor
    switch_log_policy: torch.Tensor
    flag_log_policy: torch.Tensor
    target_log_policy: torch.Tensor


class TrainingOutput(NamedTuple):
    indices: Indices
    policy: Policy
    log_policy: LogPolicy
    logits: Logits
    value: torch.Tensor


class EnvStep(NamedTuple):
    indices: Indices
    policy: Policy
    logits: Logits

    def to_store(self, state: State) -> Dict[str, torch.Tensor]:
        to_store = {
            k: v.squeeze() for k, v in state.items() if isinstance(v, torch.Tensor)
        }
        to_store.update(
            {
                k: v.squeeze()
                for k, v in self.indices._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        to_store.update(
            {
                k: v.squeeze()
                for k, v in self.policy._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        return to_store


class PostProcess(NamedTuple):
    data: Choices
    index: torch.Tensor


class PrivateEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, embedding_dim: int):
        super().__init__()

        self.gen = gen
        self.n_active = n_active
        self.embedding_dim = embedding_dim

        # onehots
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_onehot = nn.Embedding.from_pretrained(torch.eye(len(GENDERS) + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.move_slot_onehot = nn.Embedding.from_pretrained(torch.eye(5))
        self.forme_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1)
        )

        if gen == 9:
            self.commanding_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.reviving_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.tera_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
        elif gen == 8:
            self.can_gmax_onehot = nn.Embedding.from_pretrained(torch.eye(3))

        # binaries
        self.hp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(2048))
        self.level_bin = nn.Embedding.from_pretrained(binary_enc_matrix(102))
        self.atk_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.def_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.spa_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.spd_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.spe_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.pp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(64))

        # precomputed embeddings
        self.ability_embedding = AbilityEmbedding(gen=gen)
        self.pokedex_embedding = PokedexEmbedding(gen=gen)
        self.move_embedding = MoveEmbedding(gen=gen)
        self.item_embedding = ItemEmbedding(gen=gen)

        # linear layers
        mon_lin_in = (
            self.ability_embedding.embedding_dim
            + self.active_onehot.embedding_dim
            + self.fainted_onehot.embedding_dim
            + self.gender_onehot.embedding_dim
            + self.hp_bin.embedding_dim
            + self.item_embedding.embedding_dim
            + self.level_bin.embedding_dim
            + self.hp_bin.embedding_dim
            + self.pokedex_embedding.embedding_dim
            + self.forme_onehot.embedding_dim
            + self.atk_bin.embedding_dim
            + self.def_bin.embedding_dim
            + self.spa_bin.embedding_dim
            + self.spe_bin.embedding_dim
            + self.spd_bin.embedding_dim
            + self.status_onehot.embedding_dim
        )
        if gen == 9:
            mon_lin_in += (
                self.commanding_onehot.embedding_dim
                + self.reviving_onehot.embedding_dim
                + self.tera_onehot.embedding_dim
                + self.teratype_onehot.embedding_dim
            )
        elif gen == 8:
            mon_lin_in += self.can_gmax_onehot.embedding_dim

        move_lin_in = (
            self.move_embedding.embedding_dim
            + self.pp_bin.embedding_dim
            + self.move_slot_onehot.embedding_dim
        )

        self.mon_lin = nn.Linear(mon_lin_in, embedding_dim)
        self.move_lin = nn.Linear(move_lin_in, embedding_dim)

        self.entity_emb = nn.Parameter(torch.randn(1, 1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=2,
            dim_feedforward=2 * embedding_dim,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.glu = nn.GLU()

    def forward(self, private_reserve: torch.Tensor):
        private_reserve_x = private_reserve + 1
        ability = private_reserve_x[..., 0]
        active = private_reserve_x[..., 1]
        fainted = private_reserve_x[..., 2]
        gender = private_reserve_x[..., 3]
        hp = private_reserve_x[..., 4]
        item = private_reserve_x[..., 5]
        level = private_reserve_x[..., 6]
        maxhp = private_reserve_x[..., 7]
        name = private_reserve_x[..., 8]
        forme = private_reserve_x[..., 9]
        stat_atk = private_reserve_x[..., 10]
        stat_def = private_reserve_x[..., 11]
        stat_spa = private_reserve_x[..., 12]
        stat_spd = private_reserve_x[..., 13]
        stat_spe = private_reserve_x[..., 14]
        status = private_reserve_x[..., 15]

        if self.gen == 9:
            commanding = private_reserve_x[..., 16]
            reviving = private_reserve_x[..., 17]
            teraType = private_reserve_x[..., 18]
            terastallized = private_reserve_x[..., 19]

        elif self.gen == 8:
            canGmax = private_reserve_x[..., 16]

        moves = private_reserve_x[..., -8:]
        moves = moves.view(*moves.shape[:-1], 4, 2)
        move_tokens = moves[..., 0]
        move_used = moves[..., 1]

        ability_emb = self.ability_embedding(ability)
        active_emb = self.active_onehot(active)
        fainted_emb = self.fainted_onehot(fainted)
        gender_emb = self.gender_onehot(gender)
        hp_emb = self.hp_bin(hp)
        item_emb = self.item_embedding(item)
        level_emb = self.level_bin(level)
        maxhp_emb = self.hp_bin(maxhp)
        name_emb = self.pokedex_embedding(name)
        forme_emb = self.forme_onehot(forme)
        stat_atk_emb = self.atk_bin(stat_atk)
        stat_def_emb = self.def_bin(stat_def)
        stat_spa_emb = self.spa_bin(stat_spa)
        stat_spd_emb = self.spd_bin(stat_spd)
        stat_spe_emb = self.spe_bin(stat_spe)
        status_emb = self.status_onehot(status)

        mon_emb = [
            ability_emb,
            active_emb,
            fainted_emb,
            gender_emb,
            hp_emb,
            item_emb,
            level_emb,
            maxhp_emb,
            name_emb,
            forme_emb,
            stat_atk_emb,
            stat_def_emb,
            stat_spa_emb,
            stat_spd_emb,
            stat_spe_emb,
            status_emb,
        ]

        moves_emb = self.move_embedding(move_tokens)
        move_used_emb = self.pp_bin(move_used)
        move_slot = torch.ones_like(move_used)
        for i in range(4):
            move_slot[..., i] = i
        move_slot_emb = self.move_slot_onehot(move_used)
        moves_emb = [moves_emb, move_used_emb, move_slot_emb]

        if self.gen == 9:
            commanding_emb = self.commanding_onehot(commanding)
            reviving_emb = self.reviving_onehot(reviving)
            teraType_emb = self.teratype_onehot(teraType)
            terastallized_emb = self.tera_onehot(terastallized)
            mon_emb += [
                commanding_emb,
                reviving_emb,
                teraType_emb,
                terastallized_emb,
            ]

        elif self.gen == 8:
            can_gmax_emb = self.can_gmax_onehot(canGmax)
            mon_emb.append(can_gmax_emb)

        mon_emb = torch.cat(mon_emb, dim=-1)
        mon_emb = self.mon_lin(mon_emb)

        moves_emb = torch.cat(moves_emb, dim=-1)
        moves_emb = self.move_lin(moves_emb)
        move_emb = torch.sum(moves_emb, -2)

        entity_emb = mon_emb + move_emb
        entity_emb = torch.cat(
            (
                self.entity_emb,
                entity_emb,
            ),
            dim=2,
        )

        B, T, S, *_ = entity_emb.shape

        src_key_padding_mask = private_reserve[..., 0] == -1
        src_key_padding_mask = torch.cat(
            (
                torch.zeros_like(src_key_padding_mask[..., 0]).unsqueeze(-1),
                src_key_padding_mask,
            ),
            dim=2,
        )
        src_key_padding_mask = src_key_padding_mask.view(B * T, S)
        entity_emb = entity_emb.view(B * T, S, -1)

        entity_emb = self.encoder(entity_emb, src_key_padding_mask=src_key_padding_mask)
        entity_emb = entity_emb.view(B, T, S, -1)

        entity_embedding = entity_emb[..., 0, :]
        active: torch.Tensor = entity_emb[..., 1 : 1 + self.n_active, :]
        active = active.view(B, T, -1, self.embedding_dim)
        moves = self.glu(
            torch.cat(
                (
                    moves_emb[..., : self.n_active, :, :],
                    active.unsqueeze(-2).repeat_interleave(4, -2),
                ),
                dim=-1,
            )
        )
        switches = entity_emb[..., 1:, :]

        return entity_embedding, moves, switches


class PublicEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, embedding_dim: int):
        super().__init__()

        self.gen = gen
        self.n_active = n_active
        self.embedding_dim = embedding_dim

        # onehots
        self.n_onehot = nn.Embedding.from_pretrained(torch.eye(7))
        self.total_onehot = nn.Embedding.from_pretrained(torch.eye(6))
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.active_slot_onehot = nn.Embedding.from_pretrained(torch.eye(n_active + 1))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_onehot = nn.Embedding.from_pretrained(torch.eye(len(GENDERS) + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)
        )
        self.times_attacked_onehot = nn.Embedding.from_pretrained(torch.eye(8))
        self.boosts_onehot = nn.Sequential(
            nn.Embedding.from_pretrained(torch.eye(13)), nn.Flatten(-2)
        )
        self.forme_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1)
        )

        # side condition stuff
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
        self.scalar_lin = nn.Linear(2 * sc_lin_in, embedding_dim)

        self.teratype_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(BattleTypeChart) + 1)
        )

        # binaries
        self.level_bin = nn.Embedding.from_pretrained(binary_enc_matrix(102))
        self.hp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(102))
        self.pp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(64))
        self.toxic_bin = nn.Embedding.from_pretrained(binary_enc_matrix(17))
        self.sleep_bin = nn.Embedding.from_pretrained(binary_enc_matrix(5))

        # precomputed embeddings
        self.ability_embedding = AbilityEmbedding(gen=gen)
        self.pokedex_embedding = PokedexEmbedding(gen=gen)
        self.move_embedding = MoveEmbedding(gen=gen)
        self.item_embedding = ItemEmbedding(gen=gen)

        # linear layers
        reserve_mon_lin_in = (
            self.pokedex_embedding.embedding_dim
            + self.forme_onehot.embedding_dim
            + self.active_slot_onehot.embedding_dim
            + 1
            + self.hp_bin.embedding_dim
            + self.fainted_onehot.embedding_dim
            + self.level_bin.embedding_dim
            + self.gender_onehot.embedding_dim
            + self.ability_embedding.embedding_dim
            + self.ability_embedding.embedding_dim
            + self.item_embedding.embedding_dim
            + self.item_effect_onehot.embedding_dim
            + self.item_embedding.embedding_dim
            + self.item_effect_onehot.embedding_dim
            + self.teratype_onehot.embedding_dim
            + self.status_onehot.embedding_dim
            + self.active_onehot.embedding_dim
            + self.move_embedding.embedding_dim
            + self.times_attacked_onehot.embedding_dim
        )
        active_mon_lin_in = (
            reserve_mon_lin_in
            + self.sleep_bin.embedding_dim
            + self.toxic_bin.embedding_dim
            + len(BOOSTS) * 13
            + len(VOLATILES)
        )

        move_lin_in = self.move_embedding.embedding_dim + self.pp_bin.embedding_dim

        self.active_mon_lin = nn.Linear(active_mon_lin_in, embedding_dim)
        self.reserve_mon_lin = nn.Linear(reserve_mon_lin_in, embedding_dim)

        self.move_lin = nn.Linear(move_lin_in, embedding_dim)

        self.active_emb = nn.Embedding(2, embedding_dim)
        self.side_id_emb = nn.Embedding(2, embedding_dim)

        self.entity_emb = nn.Parameter(torch.randn(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=2,
            dim_feedforward=2 * embedding_dim,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.glu = nn.GLU()

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
        side_conditions_x = side_conditions + 1
        sc_levels = side_conditions[..., 0]
        sc_min_dur = side_conditions_x[..., 1]
        sc_max_dur = side_conditions_x[..., 2]

        scalars = torch.cat(
            [
                self.n_onehot(n),
                self.total_onehot(total_pokemon - 1),
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
        scalars = torch.flatten(scalars, -2)
        scalar_emb = self.scalar_lin(scalars)

        active = active.long()
        active_x = active + 1

        active_species = active_x[..., 0]
        active_forme = active_x[..., 1]
        active_slot = active_x[..., 2]
        active_hp = active[..., 3].unsqueeze(-1)
        active_hp_bin = active[..., 3].clamp(min=0) * 100
        active_fainted = active_x[..., 4]
        active_level = active_x[..., 5]
        active_gender = active_x[..., 6]
        active_ability = active_x[..., 7]
        active_base_ability = active_x[..., 8]
        active_item = active_x[..., 9]
        active_item_effect = active_x[..., 10]
        active_prev_item = active_x[..., 11]
        active_prev_item_effect = active_x[..., 12]
        active_terastallized = active_x[..., 13]
        active_status = active_x[..., 14]
        active_status_stage = active_x[..., 15]
        active_last_move = active_x[..., 16]
        active_times_attacked = active_x[..., 17]
        active_sleep_turns = active_x[..., 18]
        active_toxic_turns = active_x[..., 19]
        active_boosts = active[..., 20 : 20 + len(BOOSTS)] + 6
        active_volatiles = active[
            ..., 20 + len(BOOSTS) : 20 + len(BOOSTS) + len(VOLATILES)
        ].clamp(min=0)
        active_side_id = active[..., -17]
        active_moves = active_x[..., -16:]
        active_moves = active_moves.view(*active_moves.shape[:-1], 8, 2)
        active_moves_id = active_moves[..., 0]
        active_moves_pp = active_moves[..., 1]

        active_side_id_emb = self.side_id_emb(active_side_id)
        active_moves_id_emb = self.move_embedding(active_moves_id)
        active_moves_pp_emb = self.pp_bin(active_moves_pp)
        active_active_emb = self.active_emb(torch.ones_like(active_slot))

        active_mon_emb = torch.cat(
            [
                self.pokedex_embedding(active_species),
                self.forme_onehot(active_forme),
                self.active_slot_onehot(active_slot),
                active_hp,
                self.hp_bin(active_hp_bin),
                self.fainted_onehot(active_fainted),
                self.level_bin(active_level),
                self.gender_onehot(active_gender),
                self.ability_embedding(active_ability),
                self.ability_embedding(active_base_ability),
                self.item_embedding(active_item),
                self.item_effect_onehot(active_item_effect),
                self.item_embedding(active_prev_item),
                self.item_effect_onehot(active_prev_item_effect),
                self.teratype_onehot(active_terastallized),
                self.status_onehot(active_status),
                self.active_onehot(active_status_stage),
                self.move_embedding(active_last_move),
                self.times_attacked_onehot(active_times_attacked),
                self.sleep_bin(active_sleep_turns),
                self.toxic_bin(active_toxic_turns),
                self.boosts_onehot(active_boosts),
                active_volatiles,
            ],
            dim=-1,
        )
        active_moves_emb = torch.cat(
            [
                active_moves_id_emb,
                active_moves_pp_emb,
            ],
            dim=-1,
        )
        active_mon_emb = self.active_mon_lin(active_mon_emb)
        active_mon_emb += self.move_lin(active_moves_emb).sum(-2)
        active_mon_emb += active_side_id_emb
        active_mon_emb += active_active_emb

        reserve = reserve.long()
        reserve_x = reserve + 1
        reserve_species = reserve_x[..., 0]
        reserve_forme = reserve_x[..., 1]
        reserve_slot = reserve_x[..., 2]
        reserve_hp = reserve[..., 3].unsqueeze(-1)
        reserve_hp_bin = reserve[..., 3].clamp(min=0) * 100
        reserve_fainted = reserve_x[..., 4]
        reserve_level = reserve_x[..., 5]
        reserve_gender = reserve_x[..., 6]
        reserve_ability = reserve_x[..., 7]
        reserve_base_ability = reserve_x[..., 8]
        reserve_item = reserve_x[..., 9]
        reserve_item_effect = reserve_x[..., 10]
        reserve_prev_item = reserve_x[..., 11]
        reserve_prev_item_effect = reserve_x[..., 12]
        reserve_terastallized = reserve_x[..., 13]
        reserve_status = reserve_x[..., 14]
        reserve_status_stage = reserve_x[..., 15]
        reserve_last_move = reserve_x[..., 16]
        reserve_times_attacked = reserve_x[..., 17]
        reserve_side = reserve[..., 18]
        reserve_moves = reserve_x[..., 19:]

        reserve_moves = reserve_moves.view(*reserve_moves.shape[:-1], 8, 2)
        reserve_moves_id = reserve_moves[..., 0]
        reserve_moves_pp = reserve_moves[..., 1]

        reserve_side_emb = self.side_id_emb(reserve_side)
        reserve_moves_id_emb = self.move_embedding(reserve_moves_id)
        reserve_moves_pp_emb = self.pp_bin(reserve_moves_pp)
        reserve_active_emb = self.active_emb(torch.ones_like(reserve_slot))

        reserve_mon_emb = torch.cat(
            [
                self.pokedex_embedding(reserve_species),
                self.forme_onehot(reserve_forme),
                self.active_slot_onehot(reserve_slot),
                reserve_hp,
                self.hp_bin(reserve_hp_bin),
                self.fainted_onehot(reserve_fainted),
                self.level_bin(reserve_level),
                self.gender_onehot(reserve_gender),
                self.ability_embedding(reserve_ability),
                self.ability_embedding(reserve_base_ability),
                self.item_embedding(reserve_item),
                self.item_effect_onehot(reserve_item_effect),
                self.item_embedding(reserve_prev_item),
                self.item_effect_onehot(reserve_prev_item_effect),
                self.teratype_onehot(reserve_terastallized),
                self.status_onehot(reserve_status),
                self.active_onehot(reserve_status_stage),
                self.move_embedding(reserve_last_move),
                self.times_attacked_onehot(reserve_times_attacked),
            ],
            dim=-1,
        )

        reserve_moves_emb = torch.cat(
            [
                reserve_moves_id_emb,
                reserve_moves_pp_emb,
            ],
            dim=-1,
        )
        reserve_mon_emb = self.reserve_mon_lin(reserve_mon_emb)
        reserve_mon_emb += self.move_lin(reserve_moves_emb).sum(-2)
        reserve_mon_emb += reserve_side_emb
        reserve_mon_emb += reserve_active_emb

        entity_emb = torch.cat(
            [
                active_mon_emb,
                reserve_mon_emb,
            ],
            dim=-2,
        )

        B, T, S, L, *_ = entity_emb.shape
        entity_emb = entity_emb.view(B * T, S * L, -1)
        entity_emb = torch.cat((self.entity_emb, entity_emb), dim=1)

        public_reserve = torch.cat(
            [
                active[..., 0],
                reserve[..., 0],
            ],
            dim=-1,
        )
        src_key_padding_mask = public_reserve.view(B * T, S * L)
        src_key_padding_mask = torch.cat(
            (
                torch.zeros_like(src_key_padding_mask[..., 0]).unsqueeze(-1),
                src_key_padding_mask,
            ),
            dim=-1,
        )
        src_key_padding_mask = src_key_padding_mask == -1

        entity_emb = self.encoder(entity_emb, src_key_padding_mask=src_key_padding_mask)
        entity_emb = entity_emb[..., 0, :].view(B, T, -1)

        return entity_emb, scalar_emb


class WeatherEncoder(nn.Module):
    def __init__(self, gen: int, embedding_dim: int):
        super().__init__()

        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(len(WEATHERS) + 1))
        self.time_left_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.min_time_left_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        pw_min_onehot = nn.Embedding.from_pretrained(torch.eye(8))
        self.pw_min_onehot = nn.Sequential(pw_min_onehot, nn.Flatten(2))
        pw_max_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.pw_max_onehot = nn.Sequential(pw_max_onehot, nn.Flatten(2))

        lin_in = (
            self.weather_onehot.embedding_dim
            + self.time_left_onehot.embedding_dim
            + self.min_time_left_onehot.embedding_dim
            + pw_min_onehot.embedding_dim * len(PSEUDOWEATHERS)
            + pw_max_onehot.embedding_dim * len(PSEUDOWEATHERS)
        )
        self.lin = nn.Linear(lin_in, embedding_dim)

    def forward(
        self,
        weather: torch.Tensor,
        time_left: torch.Tensor,
        min_time_left: torch.Tensor,
        pseudo_weather: torch.Tensor,
    ):
        weather_onehot = self.weather_onehot(weather + 1)
        time_left_onehot = self.time_left_onehot(time_left)
        min_time_left_onehot = self.min_time_left_onehot(min_time_left)

        pseudo_weather_x = pseudo_weather + 1
        pw_min_time_left = pseudo_weather_x[..., 0]
        pw_max_time_left = pseudo_weather_x[..., 1]

        pw_min_time_left_onehot = self.pw_min_onehot(pw_min_time_left)
        pw_max_time_left_onehot = self.pw_max_onehot(pw_max_time_left)

        weather_emb = torch.cat(
            (
                weather_onehot,
                time_left_onehot,
                min_time_left_onehot,
                pw_max_time_left_onehot,
                pw_min_time_left_onehot,
            ),
            dim=-1,
        )
        weather_emb = self.lin(weather_emb)

        return weather_emb


class ScalarEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, embedding_dim: int):
        super().__init__()

        self.gen = gen
        self.turn_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1002))

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

        lin_in = self.turn_bin.embedding_dim + action_mask_size
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

        self.lin = nn.Linear(lin_in, embedding_dim // 2)

    def forward(
        self,
        turn: torch.Tensor,
        prev_choices: torch.Tensor,
        choices_done: torch.Tensor,
        action_type_mask: torch.Tensor,
        moves_mask: torch.Tensor,
        max_moves_mask: torch.Tensor,
        switches_mask: torch.Tensor,
        flags_mask: torch.Tensor,
        targets_mask: torch.Tensor,
    ):

        scalar_emb = [
            self.turn_bin(turn),
            action_type_mask,
            moves_mask,
            switches_mask,
            flags_mask,
        ]
        if self.gen == 8:
            scalar_emb.append(max_moves_mask)

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
                targets_mask,
            ]

        scalar_emb = torch.cat(scalar_emb, dim=-1)
        scalar_emb = self.lin(scalar_emb)

        return scalar_emb


class ResBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.lin(x)
        x = x + res
        return x


class PolicyHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_resblocks: int = 1):
        super().__init__()
        self.resblock_stack = nn.ModuleList(
            [ResBlock(in_features) for _ in range(num_resblocks)]
        )
        self.lin_out = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = self.lin_out(x)
        return x


class ActionTypeHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.lin_out = nn.Linear(in_features, out_features)

        self.glu = nn.GLU()
        self.action_type_embedding = nn.Embedding(3, in_features)

    def forward(self, state: torch.Tensor, action_type_mask: torch.Tensor):
        B, T, *_ = state.shape

        action_type_logits = self.lin_out(state)
        action_type_policy = _legal_policy(action_type_logits, action_type_mask)
        action_type_index = torch.multinomial(action_type_policy.view(B * T, -1), 1)
        action_type_index = action_type_index.view(B, T)

        action_type_embedding = self.action_type_embedding(action_type_index)
        autoregressive_embedding = torch.cat((action_type_embedding, state), dim=-1)
        autoregressive_embedding = self.glu(autoregressive_embedding)

        return (
            action_type_logits,
            action_type_policy,
            action_type_index,
            autoregressive_embedding,
        )


class MoveHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.query = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.keys = nn.Sequential(
            nn.Linear(in_features // 2, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.proj_move = nn.Linear(in_features // 2, in_features)

    def forward(
        self,
        action_type_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        moves: torch.Tensor,
        moves_mask: torch.Tensor,
    ):
        B, T, *_ = autoregressive_embedding.shape

        query = self.query(autoregressive_embedding)
        query = query.view(B, T, 1, 1, -1)
        keys = self.keys(moves)

        move_logits = query @ keys.transpose(-2, -1)
        move_logits = move_logits.view(B, T, -1)

        moves_mask = moves_mask.view(B * T, -1)
        moves_mask[moves_mask.sum() == 0] = True
        moves_mask = moves_mask.view(B, T, -1)

        move_policy = _legal_policy(move_logits, moves_mask)
        embedding_index = torch.multinomial(move_policy.view(B * T, -1), 1)
        move_index = embedding_index.view(B, T, -1)

        device = next(self.parameters()).device
        embedding_index = embedding_index + torch.arange(0, B * T, 4, device=device)
        move_embedding = moves.flatten(0, -2)[embedding_index]
        move_embedding = move_embedding.view(B, T, -1)
        projected_move_embedding = self.proj_move(move_embedding)

        valid_indices = action_type_index == 0
        autoregressive_embedding[valid_indices] = (
            autoregressive_embedding[valid_indices]
            + projected_move_embedding[valid_indices]
        )

        return (
            move_logits,
            move_policy,
            move_index,
            autoregressive_embedding,
            projected_move_embedding,
        )


class SwitchHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.query = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.keys = nn.Sequential(
            nn.Linear(in_features // 2, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.proj_switch = nn.Linear(in_features // 2, in_features)

    def forward(
        self,
        action_type_index: torch.Tensor,
        autoregressive_embedding: torch.Tensor,
        switches: torch.Tensor,
        switches_mask: torch.Tensor,
    ):
        B, T, *_ = autoregressive_embedding.shape

        query = self.query(autoregressive_embedding)
        keys = self.keys(switches)

        switch_logits = query @ keys.transpose(-2, -1)

        switches_mask = switches_mask.view(B * T, -1)
        switches_mask[switches_mask.sum() == 0] = True
        switches_mask = switches_mask.view(B, T, -1)

        switch_policy = _legal_policy(switch_logits, switches_mask)
        embedding_index = torch.multinomial(switch_policy.view(B * T, -1), 1)
        switch_index = embedding_index.view(B, T, -1)

        device = next(self.parameters()).device
        embedding_index = embedding_index + torch.arange(0, B * T, 6, device=device)
        switch_embedding = switches.flatten(0, -2)[embedding_index]
        switch_embedding = switch_embedding.view(B, T, -1)
        projected_move_embedding = self.proj_switch(switch_embedding)

        valid_indices = action_type_index == 1
        autoregressive_embedding[valid_indices] = (
            autoregressive_embedding[valid_indices]
            + projected_move_embedding[valid_indices]
        )

        return (
            switch_logits,
            switch_policy,
            switch_index,
            autoregressive_embedding,
        )


class MaxMoveHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.query = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.keys = nn.Sequential(
            nn.Linear(in_features // 2, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
        )
        self.proj_max_move = nn.Linear(in_features // 2, in_features)

    def forward(
        self,
        autoregressive_embedding: torch.Tensor,
        max_moves: torch.Tensor,
        max_moves_mask: torch.Tensor,
    ):
        B, T, *_ = autoregressive_embedding.shape

        query = self.query(autoregressive_embedding)
        query = query.view(B, T, 1, 1, -1)
        keys = self.keys(max_moves)

        max_move_logits = query @ keys.transpose(-2, -1)
        max_move_logits = max_move_logits.view(B, T, -1)

        max_moves_mask = max_moves_mask.view(B * T, -1)
        max_moves_mask[max_moves_mask.sum() == 0] = True
        max_moves_mask = max_moves_mask.view(B, T, -1)

        max_move_policy = _legal_policy(max_move_logits, max_moves_mask)
        embedding_index = torch.multinomial(max_move_policy.view(B * T, -1), 1)
        max_move_index = embedding_index.view(B, T, -1)

        return (
            max_move_logits,
            max_move_policy,
            max_move_index,
        )


class FlagsHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, len(CHOICE_FLAGS)),
        )

    def forward(
        self,
        autoregressive_embedding: torch.Tensor,
        move_embedding: torch.Tensor,
        flag_mask: torch.Tensor,
    ):
        B, T, *_ = autoregressive_embedding.shape

        autoregressive_embedding = self.mlp1(autoregressive_embedding + move_embedding)
        flag_logits = self.mlp2(autoregressive_embedding)
        flag_policy = _legal_policy(flag_logits, flag_mask)
        flag_index = torch.multinomial(flag_policy.view(B * T, -1), 1)
        flag_index = flag_index.view(B, T, -1)

        return (
            flag_logits,
            flag_policy,
            flag_index,
            autoregressive_embedding,
        )


class Torso(nn.Module):
    def __init__(self, embedding_dim: int, num_resblocks: int = 2):
        super().__init__()

        input_dim = 4 * embedding_dim + embedding_dim // 2
        self.lin_in = nn.Linear(input_dim, 2 * embedding_dim)
        self.resblock_stack = nn.ModuleList(
            [ResBlock(2 * embedding_dim) for _ in range(num_resblocks)]
        )

    def forward(self, x: torch.Tensor):
        x = self.lin_in(x)
        for resblock in self.resblock_stack:
            x = resblock(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, embedding_dim: int, num_resblocks: int = 1):
        super().__init__()
        self.lin_in = nn.Linear(2 * embedding_dim, embedding_dim)
        self.resblock_stack = nn.ModuleList(
            [ResBlock(embedding_dim) for _ in range(num_resblocks)]
        )
        self.lin_out = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.lin_in(x)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = self.lin_out(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        gen: int = 9,
        gametype: Literal["singles", "doubles", "triples"] = "singles",
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.gen = gen
        self.gametype = gametype
        n_active = {"singles": 1, "doubles": 2, "triples": 3}[gametype]

        self.private_encoder = PrivateEncoder(
            gen=gen, n_active=n_active, embedding_dim=embedding_dim
        )
        self.public_encoder = PublicEncoder(
            gen=gen, n_active=n_active, embedding_dim=embedding_dim
        )
        self.weather_encoder = WeatherEncoder(gen=gen, embedding_dim=embedding_dim)
        self.scalar_encoder = ScalarEncoder(
            gen=gen, n_active=n_active, embedding_dim=embedding_dim
        )

        self.action_type_head = ActionTypeHead(2 * embedding_dim, 3)
        self.move_head = MoveHead(2 * embedding_dim)

        self.state_fields = deepcopy(_STATE_FIELDS)

        if gen == 8:
            self.max_move_head = MaxMoveHead(2 * embedding_dim)
        else:
            self.state_fields.remove("max_moves_mask")

        self.switch_head = SwitchHead(2 * embedding_dim)
        self.flag_head = FlagsHead(2 * embedding_dim)

        if self.gametype != "singles":
            self.target_head = PolicyHead(2 * embedding_dim, 2 * n_active)
        else:
            self.state_fields.remove("targets_mask")
            self.state_fields.remove("prev_choices")

        self.torso = Torso(embedding_dim=embedding_dim)
        self.value_Head = ValueHead(embedding_dim=embedding_dim)

        learnable_params = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        print(f"learnabled params: {learnable_params:,}")

    def clean(self, state: State):
        return {k: state[k] for k in self.state_fields}

    def forward(
        self,
        state: State,
        choices: Optional[Dict[str, Any]] = None,
    ):
        # private info
        private_reserve = state["private_reserve"]

        # public info
        public_n = state["public_n"]
        public_total_pokemon = state["public_total_pokemon"]
        public_faint_counter = state["public_faint_counter"]
        public_side_conditions = state["public_side_conditions"]
        public_wisher = state["public_wisher"]
        public_active = state["public_active"]
        public_reserve = state["public_reserve"]
        public_stealthrock = state["public_stealthrock"]
        public_spikes = state["public_spikes"]
        public_toxicspikes = state["public_toxicspikes"]
        public_stickyweb = state["public_stickyweb"]

        # weather type stuff (still public)
        weather = state["weather"]
        weather_time_left = state["weather_time_left"]
        weather_min_time_left = state["weather_min_time_left"]
        pseudo_weather = state["pseudo_weather"]

        # scalar information
        turn = state["turn"]
        prev_choices = state["prev_choices"]
        choices_done = state["choices_done"]

        # action masks
        action_type_mask = state["action_type_mask"]
        moves_mask = state["moves_mask"]
        max_move_mask = state["max_moves_mask"]
        switches_mask = state["switches_mask"]
        flags_mask = state["flags_mask"]
        targets_mask = state["targets_mask"]

        private_entity_emb, moves, switches = self.private_encoder(private_reserve)

        public_entity_emb, public_scalar_emb = self.public_encoder(
            public_n,
            public_total_pokemon,
            public_faint_counter,
            public_side_conditions,
            public_wisher,
            public_active,
            public_reserve,
            public_stealthrock,
            public_spikes,
            public_toxicspikes,
            public_stickyweb,
        )

        weather_emb = self.weather_encoder(
            weather,
            weather_time_left,
            weather_min_time_left,
            pseudo_weather,
        )

        scalar_emb = self.scalar_encoder(
            turn,
            prev_choices,
            choices_done,
            action_type_mask,
            moves_mask,
            max_move_mask,
            switches_mask,
            flags_mask,
            targets_mask,
        )

        state_emb = torch.cat(
            (
                private_entity_emb,
                public_entity_emb,
                public_scalar_emb,
                weather_emb,
                scalar_emb,
            ),
            dim=-1,
        )

        state_emb = self.torso(state_emb)

        (
            action_type_logits,
            action_type_policy,
            action_type_index,
            at_autoregressive_embedding,
        ) = self.action_type_head(state_emb, action_type_mask)

        (
            move_logits,
            move_policy,
            move_index,
            autoregressive_embedding,
            move_embedding,
        ) = self.move_head(
            action_type_index,
            at_autoregressive_embedding,
            moves,
            moves_mask,
        )

        (
            switch_logits,
            switch_policy,
            switch_index,
            autoregressive_embedding,
        ) = self.switch_head(
            action_type_index,
            at_autoregressive_embedding,
            switches,
            switches_mask,
        )

        (
            flag_logits,
            flag_policy,
            flag_index,
            autoregressive_embedding,
        ) = self.flag_head(
            autoregressive_embedding,
            move_embedding,
            flags_mask,
        )

        if self.gen == 8:
            max_move_logits, max_move_policy, max_move_index = self.max_move_head(
                at_autoregressive_embedding,
                moves,
                max_move_mask,
            )
            max_move_log_policy = _log_policy(max_move_logits, max_move_mask)
        else:
            max_move_index = None
            max_move_logits = None
            max_move_policy = None
            max_move_log_policy = None

        if self.gametype != "singles":
            target_logits, target_policy, target_index = self.target_head(
                prev_choices,
                moves,
                switches,
                autoregressive_embedding,
            )
            target_log_policy = _log_policy(target_logits, targets_mask)
        else:
            target_index = None
            target_logits = None
            target_policy = None
            target_log_policy = None

        value = self.value_Head(state_emb)

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

        if not self.training:
            targeting = bool(state["targeting"])
            post_process = self.index_to_action(targeting, indices, choices)
            env_step = EnvStep(
                indices=indices,
                policy=policy,
                logits=logits,
            )
            output = (env_step, post_process)
        else:
            log_policy = LogPolicy(
                action_type_log_policy=_log_policy(
                    action_type_logits, action_type_mask
                ),
                move_log_policy=_log_policy(move_logits, moves_mask),
                max_move_log_policy=max_move_log_policy,
                switch_log_policy=_log_policy(switch_logits, switches_mask),
                flag_log_policy=_log_policy(flag_logits, flags_mask),
                target_log_policy=target_log_policy,
            )
            output = TrainingOutput(
                policy=policy,
                log_policy=log_policy,
                logits=logits,
                value=value,
            )
        return output

    def index_to_action(
        self,
        targeting: bool,
        indices: Indices,
        choices: Optional[Dict[str, Any]] = None,
    ):
        action_type_index = indices.action_type_index
        move_index = indices.move_index
        max_move_index = indices.max_move_index
        switch_index = indices.switch_index
        flag_index = indices.flag_index
        target_index = indices.target_index

        if not targeting:
            index = action_type_index

            if action_type_index == 0:
                if flag_index == 3:
                    index = max_move_index
                else:
                    index = move_index

                if choices is not None:
                    if flag_index == 0:
                        data = choices["moves"]
                    elif flag_index == 1:
                        data = choices["mega_moves"]
                    elif flag_index == 2:
                        data = choices["zmoves"]
                    elif flag_index == 3:
                        data = choices["max_moves"]
                    elif flag_index == 4:
                        data = choices["tera_moves"]
                else:
                    data = None

            elif action_type_index == 1:
                if choices is not None:
                    data = choices["switches"]
                else:
                    data = None
                index = switch_index

        else:
            if choices is not None:
                data = choices["targets"]
            else:
                data = None
            index = target_index

        return PostProcess(data, index)


class MewZeroController(Controller):
    def __init__(self, model: Model, replay_buffer: ReplayBuffer):

        self._model = model
        self._replay_buffer = replay_buffer

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        output: Tuple[EnvStep, PostProcess]
        with torch.no_grad():
            output = self._model(state, choices)

        env_step, postprocess = output

        state = self._model.clean(state)
        to_store = env_step.to_store(state)
        self._replay_buffer.store_sample(room.battle_tag, to_store)

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs

    def store_reward(self, room: BattleRoom, pid: int, reward: float = None):
        self._replay_buffer.append_reward(room.battle_tag, pid, reward)
        self._replay_buffer.register_done(room.battle_tag)
