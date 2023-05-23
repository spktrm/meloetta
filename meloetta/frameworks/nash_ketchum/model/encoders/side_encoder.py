import torch

from torch import nn
from torch.nn import functional as F

from typing import Tuple

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    sqrt_one_hot_matrix,
    power_one_hot_matrix,
    binary_enc_matrix,
    TransformerEncoder,
    MLP,
    ToVector,
    VectorMerge,
    GLU,
)
from meloetta.frameworks.nash_ketchum.model.interfaces import SideEncoderOutput
from meloetta.embeddings import (
    AbilityEmbedding,
    PokedexEmbedding,
    MoveEmbedding,
    ItemEmbedding,
)

from meloetta.data import (
    GENDERS,
    BOOSTS,
    VOLATILES,
    STATUS,
    TOKENIZED_SCHEMA,
    ITEM_EFFECTS,
    SIDE_CONDITIONS,
    BattleTypeChart,
)


class PokemonEmbedding(nn.Module):
    def __init__(self, gen: int = 9, output_dim: int = 128) -> None:
        super().__init__()

        self.gen = gen

        pokedex_embedding = PokedexEmbedding(gen=gen)
        self.pokedex_embedding = nn.Sequential(
            pokedex_embedding, nn.Linear(pokedex_embedding.embedding_dim, output_dim)
        )
        ability_embedding = AbilityEmbedding(gen=gen)
        self.ability_embedding = nn.Sequential(
            ability_embedding, nn.Linear(ability_embedding.embedding_dim, output_dim)
        )
        self.item_embedding = ItemEmbedding(gen=gen)
        self.item_effect_onehot = nn.Embedding.from_pretrained(
            torch.eye(len(ITEM_EFFECTS) + 1)[..., 1:]
        )
        self.item_lin = nn.Linear(
            self.item_embedding.embedding_dim + self.item_effect_onehot.embedding_dim,
            output_dim,
        )
        self.pp_bin_enc = nn.Embedding.from_pretrained(binary_enc_matrix(64))
        self.move_embedding_ae = MoveEmbedding(gen=gen)
        self.move_embedding = nn.Sequential(
            nn.Linear(
                self.move_embedding_ae.embedding_dim + self.pp_bin_enc.embedding_dim,
                output_dim,
            )
        )
        self.last_move_embedding = nn.Linear(
            self.move_embedding_ae.embedding_dim + self.pp_bin_enc.embedding_dim,
            output_dim,
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

        self.hp_sqrt_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(768 if gen != 8 else 1536)[..., 1:]
        )
        self.stat_sqrt_onehot = nn.Embedding.from_pretrained(
            power_one_hot_matrix(512, 1 / 3)[..., 1:]
        )

        self.side_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.known_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        onehot_size = (
            self.active_onehot.embedding_dim
            + self.fainted_onehot.embedding_dim
            + self.gender_onehot.embedding_dim
            + self.status_onehot.embedding_dim
            + self.sleep_turns_onehot.embedding_dim
            + self.toxic_turns_onehot.embedding_dim
            + self.forme_embedding.embedding_dim
            + self.level_onehot.embedding_dim
            + 2 * self.hp_sqrt_onehot.embedding_dim
            + 1
            + 5 * self.stat_sqrt_onehot.embedding_dim
            + self.side_onehot.embedding_dim
            + self.known_onehot.embedding_dim
        )

        if gen == 9:
            self.commanding_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
            self.reviving_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
            self.tera_onehot = nn.Embedding.from_pretrained(torch.eye(2))
            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)[..., 1:]
            )

            onehot_size += (
                self.tera_onehot.embedding_dim + self.teratype_onehot.embedding_dim
            )

        elif gen == 8:
            self.can_gmax_embedding = nn.Embedding.from_pretrained(torch.eye(3))

            onehot_size += self.can_gmax_embedding.embedding_dim

        self.onehots_lin = nn.Linear(onehot_size, output_dim)

    def embed_moves(
        self, move_tokens: torch.Tensor, move_used: torch.Tensor
    ) -> torch.Tensor:
        moves_emb = self.move_embedding_ae(move_tokens)
        move_used_emb = self.pp_bin_enc(move_used)
        moves_emb = torch.cat((moves_emb, move_used_emb), dim=-1)
        return self.move_embedding(moves_emb)

    def embed_item(
        self, item_token: torch.Tensor, item_effect_token: torch.Tensor
    ) -> torch.Tensor:
        item_cat = torch.cat(
            (
                self.item_embedding(item_token),
                self.item_effect_onehot(item_effect_token),
            ),
            dim=-1,
        )
        return self.item_lin(item_cat)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        longs = (x + 1).long()

        name = longs[..., 0]
        forme = longs[..., 1]
        # slot = longs[..., 2]
        hp = longs[..., 3]
        maxhp = longs[..., 4]
        hp_ratio = x[..., 5]
        stats = longs[..., 6:11]
        fainted = longs[..., 11]
        active = longs[..., 12]
        level = x[..., 13].long()
        gender = longs[..., 14]
        ability = longs[..., 15]
        # base_ability = longs[..., 16]
        item = longs[..., 17]
        # prev_item = longs[..., 18]
        item_effect = longs[..., 19]
        # prev_item_effect = longs[..., 20]
        status = longs[..., 21]
        sleep_turns = longs[..., 22]
        toxic_turns = longs[..., 23]

        last_move = longs[..., 24]
        last_move_pp = longs[..., 25].clamp(max=63)

        moves = longs[..., 26:30]
        pp = longs[..., 30:34].clamp(max=63)

        if self.gen == 9:
            terastallized = longs[..., 33]
            teratype = longs[..., 35]
            # times_attacked = longs[..., 31]

        name_emb = self.pokedex_embedding(name)

        hp_emb = self.hp_sqrt_onehot(hp)
        maxhp_emb = self.hp_sqrt_onehot(maxhp)
        hp_ratio = hp_ratio.unsqueeze(-1)
        stat_onehot = self.stat_sqrt_onehot(stats).flatten(-2)

        ability_emb = self.ability_embedding(ability)
        # base_ability_emb = self.ability_embedding(base_ability)

        item_emb = self.embed_item(item, item_effect)
        # prev_item_emb = self.embed_item(prev_item, prev_item_effect)

        status_onehot = self.status_onehot(status)
        sleep_turns_onehot = self.sleep_turns_onehot(sleep_turns)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns)

        moves_emb = self.move_embedding(
            torch.cat((self.move_embedding_ae(moves), self.pp_bin_enc(pp)), dim=-1)
        )
        moveset_emb = moves_emb.sum(-2)

        last_move_emb = self.last_move_embedding(
            torch.cat(
                (self.move_embedding_ae(last_move), self.pp_bin_enc(last_move_pp)),
                dim=-1,
            )
        )

        side = torch.ones_like(active)
        side[:, :, :2] = 0

        known = torch.zeros_like(active)
        known[:, :, 1:] = 0

        forme_enc = self.forme_embedding(forme)
        stat_enc = torch.cat((hp_emb, maxhp_emb, hp_ratio, stat_onehot), dim=-1)
        active_enc = self.active_onehot(active)
        fainted_enc = self.fainted_onehot(fainted)
        gender_enc = self.gender_onehot(gender)
        level_enc = self.level_onehot(level.clamp(min=1) - 1)
        status_enc = torch.cat(
            (status_onehot, sleep_turns_onehot, toxic_turns_onehot), dim=-1
        )
        side_enc = self.side_onehot(side)
        known_enc = self.known_onehot(known)

        onehots = [
            forme_enc,
            stat_enc,
            active_enc,
            fainted_enc,
            gender_enc,
            level_enc,
            status_enc,
            side_enc,
            known_enc,
        ]

        if self.gen == 9:
            onehots += [self.teratype_onehot(teratype)]
            onehots += [self.tera_onehot((terastallized > 0).long())]

        onehots = torch.cat(onehots, dim=-1)
        onehots_emb = self.onehots_lin(onehots)

        pokemon_dict_embs = {
            "name_emb": name_emb,
            "ability_emb": ability_emb,
            "item_emb": item_emb,
            "moveset_emb": moveset_emb,
            "onehots": onehots_emb,
            # "last_move_emb": last_move_emb,
            # "base_ability_emb": base_ability_emb,
            # "prev_item_emb": prev_item_emb,
        }

        pokemon_emb = sum(pokemon_dict_embs.values())

        mask = name == 0  # | (fainted == 2)
        mask = ~mask

        pokemon_emb = pokemon_emb * mask.unsqueeze(-1)
        pokemon_emb = pokemon_emb.flatten(2, 3)

        mask = mask.flatten(2)

        return pokemon_emb, mask, moves_emb


class SideEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.SideEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.embedding = PokemonEmbedding(
            gen=gen, output_dim=config.entity_embedding_dim  # , frozen=True
        )

        self.transformer = TransformerEncoder(
            model_size=config.model_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            key_size=config.key_size,
            value_size=config.value_size,
            resblocks_num_before=config.resblocks_num_before,
            resblocks_num_after=config.resblocks_num_after,
            resblocks_hidden_size=config.resblocks_hidden_size,
            use_layer_norm=config.use_layer_norm,
        )

        self.output = ToVector(
            input_dim=config.entity_embedding_dim,
            hidden_dim=2 * config.entity_embedding_dim,
            output_dim=config.output_dim,
        )

        self.boosts1_onehot = nn.Embedding.from_pretrained(
            torch.tensor(
                [2 / n for n in range(8, 1, -1)] + [n / 2 for n in range(3, 9)]
            ).view(-1, 1)
        )
        self.boosts2_onehot = nn.Embedding.from_pretrained(
            torch.tensor(
                [3 / n for n in range(9, 2, -1)] + [n / 3 for n in range(4, 10)]
            ).view(-1, 1)
        )
        self.boosts_mlp = MLP([2 * len(BOOSTS), config.output_dim, config.output_dim])

        self.volatiles_mlp = MLP(
            [2 * len(VOLATILES), config.output_dim, config.output_dim]
        )

        self.toxicspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])

        self.min_dur = nn.Embedding.from_pretrained(torch.eye(6)[..., 1:])
        self.max_dur = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])
        side_con_size = (
            2
            + self.toxicspikes_onehot.embedding_dim
            + self.spikes_onehot.embedding_dim
            + (len(SIDE_CONDITIONS) - 4)
            * (self.min_dur.weight.shape[-1] + self.max_dur.weight.shape[-1])
        )
        self.side_condt_mlp = MLP(
            [2 * side_con_size, config.output_dim, config.output_dim]
        )

        self.active_moves = GLU(
            config.entity_embedding_dim,
            config.entity_embedding_dim,
            config.entity_embedding_dim,
        )

    def encode_boosts(self, boosts: torch.Tensor):
        boosts = boosts + 6
        boosts1_embed = self.boosts1_onehot(boosts[..., :6]).squeeze(-1)
        boosts2_embed = self.boosts2_onehot(boosts[..., 6:]).squeeze(-1)
        return self.boosts_mlp(
            torch.cat((boosts1_embed, boosts2_embed), dim=-1).flatten(2)
        )

    def encode_volatiles(self, volatiles: torch.Tensor):
        return self.volatiles_mlp(volatiles.float().flatten(2))

    def encode_side_conditions(self, side_conditions: torch.Tensor):
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
        return self.side_condt_mlp(side_condition_embed.flatten(2))

    def moves_given_context(self, active: torch.Tensor, moves: torch.Tensor):
        active = active.unsqueeze(-2)  # .repeat_interleave(4, -2)
        moves = moves[..., : self.n_active, :, :]
        return self.active_moves(active, moves)

    def forward(
        self,
        side: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
        side_conditions: torch.Tensor,
        wisher: torch.Tensor,
    ) -> SideEncoderOutput:

        pokemon_enc, mask, moves_emb = self.embedding.forward(side)

        T, B, S, *_ = pokemon_enc.shape

        mask = mask.clone()
        empty_mask = (mask).sum(-1) == 0
        if torch.any(empty_mask):
            mask[empty_mask, 0] = 0

        pokemon_embeddings = pokemon_enc.view(T * B, S, -1)
        mask = mask.view(T * B, S)

        pokemon_embeddings = self.transformer(pokemon_embeddings, mask)
        pokemon_embeddings = pokemon_embeddings * mask.unsqueeze(-1)

        pokemon_embedding = self.output(pokemon_embeddings)

        pokemon_embeddings = pokemon_embeddings.view(T, B, 3 * 12, -1)
        pokemon_embedding = pokemon_embedding.view(T, B, -1)

        boosts = self.encode_boosts(boosts)
        volatiles = self.encode_volatiles(volatiles)
        side_conditions = self.encode_side_conditions(side_conditions)

        active = pokemon_embeddings[..., : self.n_active, :]
        moves = self.moves_given_context(active, moves_emb[:, :, 0])
        switches = pokemon_embeddings[:, :, :6]

        return SideEncoderOutput(
            pokemon_embedding=pokemon_embedding,
            boosts=boosts,
            volatiles=volatiles,
            side_conditions=side_conditions,
            moves=moves,
            switches=switches,
        )
