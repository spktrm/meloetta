import math
import torch
import torch.nn as nn

from typing import NamedTuple, Optional, Dict, Any

from meloetta.room import BattleRoom

from meloetta.controllers.base import Controller
from meloetta.controllers.types import State, Choices

from meloetta.embeddings import (
    AbilityEmbedding,
    PokedexEmbedding,
    MoveEmbedding,
    ItemEmbedding,
)
from meloetta.data import GENDERS, STATUS, BattleTypeChart, TOKENIZED_SCHEMA


def binary_enc_matrix(num_embeddings: int):
    bits = math.ceil(math.log2(num_embeddings))
    mask = 2 ** torch.arange(bits)
    x = torch.arange(mask.sum().item() + 1)
    embs = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return embs[..., :num_embeddings:, :]


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
    policy: torch.Tensor
    log_policy: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor


class PostProcess(NamedTuple):
    logits: Dict[str, torch.Tensor]
    indices: Dict[str, torch.Tensor]
    data: Choices
    index: torch.Tensor


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


def _legal_log_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor
) -> torch.Tensor:
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


class PrivateEncoder(nn.Module):
    def __init__(self, gen: int, embedding_dim: int = 128):
        super().__init__()

        self.gen = gen

        # onehots
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.gender_onehot = nn.Embedding.from_pretrained(torch.eye(len(GENDERS) + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.move_slot_onehot = nn.Embedding.from_pretrained(torch.eye(4))
        self.forme_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1)
        )

        if gen == 9:
            self.commanding_onehot = nn.Embedding.from_pretrained(torch.eye(2))
            self.reviving_onehot = nn.Embedding.from_pretrained(torch.eye(2))
            self.tera_onehot = nn.Embedding.from_pretrained(torch.eye(2))
            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
        elif gen == 8:
            self.can_gmax_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        # binaries
        self.hp_bin = nn.Embedding.from_pretrained(binary_enc_matrix(1024))
        self.level_bin = nn.Embedding.from_pretrained(binary_enc_matrix(100))
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

    def forward(self, private_reserve: torch.Tensor):
        ability = private_reserve[..., 0]
        active = private_reserve[..., 1]
        fainted = private_reserve[..., 2]
        gender = private_reserve[..., 3] + 1
        hp = private_reserve[..., 4]
        item = private_reserve[..., 5]
        level = private_reserve[..., 6]
        maxhp = private_reserve[..., 7]
        name = private_reserve[..., 8]
        forme = private_reserve[..., 9] + 1
        stat_atk = private_reserve[..., 10]
        stat_def = private_reserve[..., 11]
        stat_spa = private_reserve[..., 12]
        stat_spd = private_reserve[..., 13]
        stat_spe = private_reserve[..., 14]
        status = private_reserve[..., 15] + 1

        if self.gen == 9:
            commanding = private_reserve[..., 16]
            reviving = private_reserve[..., 17]
            teraType = private_reserve[..., 18]
            terastallized = private_reserve[..., 19]

        elif self.gen == 8:
            canGmax = private_reserve[..., 16]

        moves = private_reserve[..., -8:]
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

        move_emb = self.move_embedding(move_tokens)
        move_used_emb = self.pp_bin(move_used)
        move_slot = torch.ones_like(move_used)
        for i in range(4):
            move_slot[..., i] = i
        move_slot_emb = self.move_slot_onehot(move_used)
        move_emb = [move_emb, move_used_emb, move_slot_emb]

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

        move_emb = torch.cat(move_emb, dim=-1)
        move_emb = self.move_lin(move_emb)
        move_emb = torch.sum(move_emb, -2)

        return mon_emb + move_emb


class PublicEncoder(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

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
        x = torch.cat(
            (
                n.flatten(2),
                total_pokemon.flatten(2),
                faint_counter.flatten(2),
                side_conditions.flatten(2),
                wisher.flatten(2),
                active.flatten(2),
                reserve.flatten(2),
                stealthrock.flatten(2),
                spikes.flatten(2),
                toxicspikes.flatten(2),
                stickyweb.flatten(2),
            ),
            dim=-1,
        )
        return x


class WeatherEncoder(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(
        self,
        weather: torch.Tensor,
        time_left: torch.Tensor,
        min_time_left: torch.Tensor,
        pseudo_weather: torch.Tensor,
    ):
        x = torch.cat(
            (
                weather.flatten(2),
                time_left.flatten(2),
                min_time_left.flatten(2),
                pseudo_weather.flatten(2),
            ),
            dim=-1,
        )
        return x


class ScalarEncoder(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

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
        x = torch.cat(
            (
                turn.flatten(2),
                prev_choices.flatten(2),
                choices_done.flatten(2),
                action_type_mask.flatten(2),
                moves_mask.flatten(2),
                max_moves_mask.flatten(2),
                switches_mask.flatten(2),
                flags_mask.flatten(2),
                targets_mask.flatten(2),
            ),
            dim=-1,
        )
        return x


class ActionTypeHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class MovesHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class MaxMovesHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class SwitchesHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class FlagsHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class TargetsHead(nn.Module):
    def __init__(self, gen: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class Torso(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor):
        return x


class Model(nn.Module):
    def __init__(self, gen: int = 9) -> None:
        super().__init__()

        self.gen = gen

        self.private_encoder = PrivateEncoder(gen=gen)
        self.public_encoder = PublicEncoder(gen=gen)
        self.weather_encoder = WeatherEncoder(gen=gen)
        self.scalar_encoder = ScalarEncoder(gen=gen)

        self.action_type_head = ActionTypeHead(gen=gen)
        self.move_head = MovesHead(gen=gen)
        if gen == 8:
            self.max_move_head = MaxMovesHead(gen=gen)
        self.switch_head = SwitchesHead(gen=gen)
        self.flag_head = FlagsHead(gen=gen)
        self.target_head = TargetsHead(gen=gen)

        self.torso = Torso()
        self.value_Head = ValueHead()

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
        max_moves_mask = state["max_moves_mask"]
        switches_mask = state["switches_mask"]
        flags_mask = state["flags_mask"]
        targets_mask = state["targets_mask"]

        private_emb = self.private_encoder(private_reserve)

        public_emb = self.public_encoder(
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
            max_moves_mask,
            switches_mask,
            flags_mask,
            targets_mask,
        )

        state_emb = torch.cat(
            (
                private_emb,
                public_emb,
                weather_emb,
                scalar_emb,
            ),
            dim=-1,
        )

        state_emb = self.torso(state_emb)

        action_type_logits = self.action_type_head(state_emb)
        moves_logits = self.move_head(state_emb)
        switches_logits = self.switch_head(state_emb)
        flags_logits = self.flag_head(state_emb)
        targets_logits = self.target_head(state_emb)

        if self.gen == 8:
            max_move_logits = self.max_move_head(state_emb)
            max_move_logits = torch.ones_like(max_moves_mask).to(torch.float32)
            max_move_policy = _legal_policy(
                max_move_logits,
                max_moves_mask,
            )
            max_move_log_policy = (
                _legal_log_policy(
                    max_move_logits,
                    max_moves_mask,
                ),
            )
        else:
            max_move_logits = None
            max_move_policy = None
            max_move_log_policy = None

        action_type_logits = torch.ones_like(action_type_mask).to(torch.float32)
        moves_logits = torch.ones_like(moves_mask).to(torch.float32)
        switches_logits = torch.ones_like(switches_mask).to(torch.float32)
        flags_logits = torch.ones_like(flags_mask).to(torch.float32)
        targets_logits = torch.ones_like(targets_mask).to(torch.float32)

        value = self.value_Head(state_emb)

        logits = Logits(
            action_type_logits=action_type_logits,
            move_logits=moves_logits,
            max_move_logits=max_move_logits,
            switch_logits=switches_logits,
            flag_logits=flags_logits,
            target_logits=targets_logits,
        )

        policy = Policy(
            action_type_policy=_legal_policy(
                logits.action_type_logits,
                action_type_mask,
            ),
            move_policy=_legal_policy(
                logits.move_logits,
                moves_mask,
            ),
            max_move_policy=max_move_policy,
            switch_policy=_legal_policy(
                logits.switch_logits,
                switches_mask,
            ),
            flag_policy=_legal_policy(
                logits.flag_logits,
                flags_mask,
            ),
            target_policy=_legal_policy(
                logits.target_logits,
                targets_mask,
            ),
        )
        log_policy = LogPolicy(
            action_type_log_policy=_legal_log_policy(
                logits.action_type_logits,
                action_type_mask,
            ),
            move_log_policy=_legal_log_policy(
                logits.move_logits,
                moves_mask,
            ),
            max_move_log_policy=max_move_log_policy,
            switch_log_policy=_legal_log_policy(
                logits.switch_logits,
                switches_mask,
            ),
            flag_log_policy=_legal_log_policy(
                logits.flag_logits,
                flags_mask,
            ),
            target_log_policy=_legal_log_policy(
                logits.target_logits,
                targets_mask,
            ),
        )

        if not self.training:
            targeting = bool(state["targeting"])
            output = self.postprocess(targeting, policy, choices)
        else:
            output = TrainingOutput(
                policy=policy,
                log_policy=log_policy,
                logits=logits,
                value=value,
            )
        return output

    def postprocess(
        self,
        targeting: bool,
        policy: Policy,
        choices: Optional[Dict[str, Any]] = None,
    ):
        action_type_policy = policy.action_type_policy.flatten()
        moves_policy = policy.move_policy.flatten()
        max_moves_policy = policy.max_move_policy.flatten()
        switches_policy = policy.switch_policy.flatten()
        flags_policy = policy.flag_policy.flatten()
        targets_policy = policy.target_policy.flatten()

        action_type_index = -torch.ones(action_type_policy.shape[:-1] + (1,))
        move_index = -torch.ones(moves_policy.shape[:-1] + (1,))
        max_move_index = -torch.ones(max_moves_policy.shape[:-1] + (1,))
        switch_index = -torch.ones(switches_policy.shape[:-1] + (1,))
        flag_index = -torch.ones(flags_policy.shape[:-1] + (1,))
        target_index = -torch.ones(targets_policy.shape[:-1] + (1,))

        if not targeting:
            action_type_index = torch.multinomial(action_type_policy, 1)
            index = action_type_index

            if action_type_index == 0:
                flag = torch.multinomial(flags_policy, 1)
                if flag == 3:
                    max_move_index = torch.multinomial(max_moves_policy, 1)
                    index = max_move_index
                else:
                    move_index = torch.multinomial(moves_policy, 1)
                    index = move_index
                flag_index = flag

                if choices is not None:
                    if flag == 0:
                        data = choices["moves"]
                    elif flag == 1:
                        data = choices["mega_moves"]
                    elif flag == 2:
                        data = choices["zmoves"]
                    elif flag == 3:
                        data = choices["max_moves"]
                    elif flag == 4:
                        data = choices["tera_moves"]
                else:
                    data = None

            elif action_type_index == 1:
                switch_index = torch.multinomial(switches_policy, 1)
                if choices is not None:
                    data = choices["switches"]
                else:
                    data = None
                index = switch_index

        else:
            target_index = torch.multinomial(targets_policy, 1)
            if choices is not None:
                data = choices["targets"]
            else:
                data = None
            index = target_index

        logits = {
            "action_type_logits": action_type_policy,
            "move_logits": moves_policy,
            "max_move_logits": max_moves_policy,
            "switch_logits": switches_policy,
            "flag_logits": flags_policy,
            "target_logits": targets_policy,
        }
        indices = {
            "action_type_index": action_type_index,
            "move_index": move_index,
            "max_move_index": max_move_index,
            "switch_index": switch_index,
            "flag_index": flag_index,
            "target_index": target_index,
        }

        return PostProcess(logits, indices, data, index)


class NaiveAIController(Controller):
    def __init__(self):
        self.model = Model(gen=8)
        self.model.eval()

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        postprocess: PostProcess = self.model(state, choices)
        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs
