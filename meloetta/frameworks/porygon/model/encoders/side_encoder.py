import torch

from torch import nn
from torch.nn import functional as F

from typing import Tuple

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.utils import (
    sqrt_one_hot_matrix,
    power_one_hot_matrix,
    TransformerEncoder,
    ToVector,
)
from meloetta.frameworks.porygon.model.interfaces import SideEncoderOutput
from meloetta.embeddings import (
    AbilityEmbedding,
    PokedexEmbedding,
    MoveEmbedding,
    ItemEmbedding,
)

from meloetta.data import (
    GENDERS,
    VOLATILES,
    STATUS,
    TOKENIZED_SCHEMA,
    ITEM_EFFECTS,
    SIDE_CONDITIONS,
    BattleTypeChart,
)


class PrivatePokemonEmbedding(nn.Module):
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
        self.pp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(64))
        self.move_embedding_ae = MoveEmbedding(gen=gen)
        self.move_embedding = nn.Linear(
            self.move_embedding_ae.embedding_dim,  # +self.pp_sqrt_onehot.embedding_dim ,
            output_dim,
        )
        self.last_move_embedding = nn.Linear(
            self.move_embedding_ae.embedding_dim,  # +self.pp_sqrt_onehot.embedding_dim ,
            output_dim,
        )

        self.active_embedding = nn.Embedding(3, output_dim)
        self.fainted_embedding = nn.Embedding(3, output_dim)
        self.gender_embedding = nn.Embedding(len(GENDERS) + 1, output_dim)

        status_dummy = torch.eye(len(STATUS) + 1)[..., 1:]
        self.status_onehot = nn.Embedding.from_pretrained(status_dummy)
        self.sleep_turns_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.toxic_turns_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(16)[..., 1:]
        )
        self.status_lin = nn.Linear(
            self.status_onehot.embedding_dim
            + self.sleep_turns_onehot.embedding_dim
            + self.toxic_turns_onehot.embedding_dim,
            output_dim,
        )

        num_formes = len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1
        self.forme_embedding = nn.Embedding(num_formes, output_dim)
        self.level_embedding = nn.Embedding(102, output_dim)

        self.hp_sqrt_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(768 if gen != 8 else 1536)[..., 1:]
        )
        self.stat_sqrt_onehot = nn.Embedding.from_pretrained(
            power_one_hot_matrix(512, 1 / 3)[..., 1:]
        )
        self.stat_lin = nn.Linear(
            2 * self.hp_sqrt_onehot.embedding_dim
            + 1
            + 5 * self.stat_sqrt_onehot.embedding_dim,
            output_dim,
        )

        if gen == 9:
            self.commanding_embedding = nn.Embedding(3, output_dim)
            self.reviving_embedding = nn.Embedding(3, output_dim)
            self.tera_embedding = nn.Embedding(2, output_dim)
            self.teratype_embedding = nn.Embedding(
                len(BattleTypeChart) + 1, output_dim, padding_idx=0
            )

        elif gen == 8:
            self.can_gmax_embedding = nn.Embedding.from_pretrained(torch.eye(3))

        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def embed_moves(
        self, move_tokens: torch.Tensor, move_used: torch.Tensor
    ) -> torch.Tensor:
        moves_emb = self.move_embedding_ae(move_tokens)
        move_used_emb = self.pp_sqrt_onehot(move_used)
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
        slot = longs[..., 2]
        hp = longs[..., 3]
        maxhp = longs[..., 4]
        hp_ratio = x[..., 5]
        stats = longs[..., 6:11]
        fainted = longs[..., 11]
        active = longs[..., 12]
        level = longs[..., 13]
        gender = longs[..., 14]
        ability = longs[..., 15]
        base_ability = longs[..., 16]
        item = longs[..., 17]
        prev_item = longs[..., 18]
        item_effect = longs[..., 19]
        prev_item_effect = longs[..., 20]
        status = longs[..., 21]
        sleep_turns = longs[..., 22]
        toxic_turns = longs[..., 23]
        last_move = longs[..., 24]
        moves = longs[..., 25:29]

        if self.gen == 9:
            terastallized = longs[..., 29]
            teratype = longs[..., 30]
            times_attacked = longs[..., 31]

        name_emb = self.pokedex_embedding(name)
        forme_emb = self.forme_embedding(forme)

        hp_emb = self.hp_sqrt_onehot(hp)
        maxhp_emb = self.hp_sqrt_onehot(maxhp)
        hp_ratio = hp_ratio.unsqueeze(-1)
        stat_onehot = self.stat_sqrt_onehot(stats).flatten(-2)
        stat_emb = self.stat_lin(
            torch.cat((hp_emb, maxhp_emb, hp_ratio, stat_onehot), dim=-1)
        )

        active_emb = self.active_embedding(active)
        fainted_emb = self.fainted_embedding(fainted)
        gender_emb = self.gender_embedding(gender)
        level_emb = self.level_embedding(level)

        ability_emb = self.ability_embedding(ability)
        base_ability_emb = self.ability_embedding(base_ability)
        base_ability_emb *= ~(ability == base_ability).unsqueeze(-1)

        item_emb = self.embed_item(item, item_effect)
        prev_item_emb = self.embed_item(prev_item, prev_item_effect)
        prev_item_emb *= ~(item == prev_item).unsqueeze(-1)

        status_onehot = self.status_onehot(status)
        sleep_turns_onehot = self.sleep_turns_onehot(sleep_turns)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns)
        status_emb = self.status_lin(
            torch.cat((status_onehot, sleep_turns_onehot, toxic_turns_onehot), dim=-1)
        )

        moves_emb = self.move_embedding(self.move_embedding_ae(moves))
        moveset_emb = moves_emb.max(-2).values

        last_move_emb = self.last_move_embedding(self.move_embedding_ae(last_move))

        pokemon_emb = (
            name_emb
            + forme_emb
            + stat_emb
            + active_emb
            + fainted_emb
            + gender_emb
            + level_emb
            + ability_emb
            + base_ability_emb
            + item_emb
            + prev_item_emb
            + status_emb
            + moveset_emb
            + last_move_emb
        )

        if self.gen == 9:
            pokemon_emb += self.teratype_embedding(teratype)
            pokemon_emb += self.tera_embedding((terastallized > 0).long())

        mask = (name == 0) | (fainted == 2)

        pokemon_emb = self.encoder(pokemon_emb)

        return pokemon_emb, mask, moves_emb


class SideEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.SideEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.embedding = PrivatePokemonEmbedding(
            gen=gen, output_dim=config.entity_embedding_dim  # , frozen=True
        )

        # self.transformer = TransformerEncoder(
        #     model_size=config.entity_embedding_dim,
        #     key_size=config.entity_embedding_dim,
        #     value_size=config.entity_embedding_dim,
        #     num_heads=config.transformer_num_heads,
        #     num_layers=config.transformer_num_layers,
        #     resblocks_num_before=config.resblocks_num_before,
        #     resblocks_num_after=config.resblocks_num_after,
        # )

        self.output = ToVector(
            input_dim=config.entity_embedding_dim,
            output_dim=config.output_dim,
        )

        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))
        self.boosts_lin = nn.Linear(8 * 6, config.output_dim)

        self.volatiles_mlp = nn.Sequential(
            nn.Linear(len(VOLATILES), config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        self.volatiles_mlp = nn.Sequential(
            nn.Linear(len(VOLATILES), config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        self.toxicspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])

        self.sc_embedding = nn.Embedding(len(SIDE_CONDITIONS) - 4, 32)
        self.sc_max_time_emb = nn.Embedding(9, 32)
        self.sc_min_time_emb = nn.Embedding(9, 32)
        self.sc_mlp = nn.Sequential(
            nn.Linear(
                2
                + self.toxicspikes_onehot.embedding_dim
                + self.spikes_onehot.embedding_dim
                + self.sc_embedding.embedding_dim,
                config.output_dim,
            ),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        self.active_moves = nn.Sequential(
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
        )

        self.side_embedding_lin = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def encode_boosts(self, boosts: torch.Tensor):
        boosts_embed = self.boosts_onehot(boosts + 6)
        boosts_embed = -boosts_embed[..., :6] + boosts_embed[..., 7:]
        boosts_embed = self.boosts_lin(boosts_embed.flatten(-2))
        return boosts_embed

    def encode_volatiles(self, volatiles: torch.Tensor):
        return self.volatiles_mlp(volatiles.float())

    def encode_side_conditions(self, side_conditions: torch.Tensor):
        stealthrock = side_conditions[..., -4].unsqueeze(-1)
        stickyweb = side_conditions[..., -1].unsqueeze(-1)

        toxicspikes_onehot = self.toxicspikes_onehot(side_conditions[..., -3])
        spikes_onehot = self.spikes_onehot(side_conditions[..., -2])

        other_sc = side_conditions[..., :-4]

        other_sc_size = len(SIDE_CONDITIONS) - 4
        other_sc = other_sc.view(*other_sc.shape[:-1], other_sc_size, 3)
        num_other_sc = other_sc[..., 0].sum(-1, keepdim=True)

        other_sc_mask = other_sc[..., 0].unsqueeze(-1)
        other_sc_embed = self.sc_embedding.weight.data
        other_sc_embed = other_sc_embed + self.sc_max_time_emb(other_sc[..., 1])
        other_sc_embed = other_sc_embed + self.sc_min_time_emb(other_sc[..., 2])
        other_sc_embed = other_sc_embed * other_sc_mask
        other_sc_embed = other_sc_embed.sum(-2)
        other_sc_embed = other_sc_embed / num_other_sc.clamp(min=1)

        side_condition_embed = torch.cat(
            [
                stealthrock,
                stickyweb,
                toxicspikes_onehot,
                spikes_onehot,
                other_sc_embed,
            ],
            dim=-1,
        )
        return self.sc_mlp(side_condition_embed)

    def moves_given_context(self, active: torch.Tensor, moves: torch.Tensor):
        active = active.unsqueeze(-2).repeat_interleave(4, -2)
        moves = moves[..., : self.n_active, :, :]
        active_and_moves = torch.cat((moves, active), dim=-1)
        return F.glu(active_and_moves, dim=-1)

    def forward(
        self,
        side: torch.Tensor,
        boosts: torch.Tensor,
        volatiles: torch.Tensor,
        side_conditions: torch.Tensor,
        wisher: torch.Tensor,
    ) -> SideEncoderOutput:
        entity_embeddings, mask, moves_emb = self.embedding.forward(side)

        T, B, N, S, *_ = entity_embeddings.shape

        mask = mask.to(torch.bool)
        mask = ~mask.view(T * B * N, S)

        encoder_mask = mask.clone()
        encoder_mask[..., 0] = 1

        entity_embeddings = entity_embeddings.view(T * B * N, S, -1)
        # entity_embeddings = self.transformer(entity_embeddings, encoder_mask)

        entity_embedding = self.output(entity_embeddings, mask.unsqueeze(-1))
        entity_embedding = entity_embedding.view(T, B, N, -1)

        side_embeddings = [
            torch.cat(
                (
                    entity_embedding[:, :, :2].sum(2, keepdim=True),
                    entity_embedding[:, :, 2:],
                ),
                dim=2,
            ),
            self.encode_boosts(boosts),
            self.encode_volatiles(volatiles),
            self.encode_side_conditions(side_conditions),
        ]
        side_embeddings = torch.stack(side_embeddings, dim=2)
        side_embedding = torch.sum(side_embeddings, dim=2)
        side_embedding = self.side_embedding_lin(side_embedding)

        entity_embeddings = entity_embeddings.view(T, B, N, S, -1)

        active = entity_embeddings[..., 0, : self.n_active, :]
        moves = self.moves_given_context(active, moves_emb[:, :, 0])
        switches = entity_embeddings[:, :, 0, :6]

        private_player = None
        public_entity = None

        return SideEncoderOutput(
            side_embedding=side_embedding,
            private_entity=private_player,
            public_entity=public_entity,
            moves=moves,
            switches=switches,
        )
