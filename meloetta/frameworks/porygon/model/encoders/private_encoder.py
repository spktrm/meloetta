import torch

from torch import nn
from torch.nn import functional as F

from meloetta.frameworks.porygon.model import config
from meloetta.frameworks.porygon.model.utils import (
    sqrt_one_hot_matrix,
    power_one_hot_matrix,
    GLU,
    TransformerEncoder,
    PrivateToVector,
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
    BattleTypeChart,
)


class PrivatePokemonEmbedding(nn.Module):
    def __init__(self, gen: int = 9, output_dim: int = 64) -> None:
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
        item_embedding = ItemEmbedding(gen=gen)
        self.item_embedding = nn.Sequential(
            item_embedding, nn.Linear(item_embedding.embedding_dim, output_dim)
        )
        self.pp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(64))
        self.move_embedding_ae = MoveEmbedding(gen=gen)
        self.move_embedding = nn.Linear(
            self.pp_sqrt_onehot.embedding_dim + self.move_embedding_ae.embedding_dim,
            output_dim,
        )

        num_genders = torch.eye(len(GENDERS) + 1)
        self.active_embedding = nn.Embedding.from_pretrained(torch.eye(3))
        self.fainted_embedding = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_embedding = nn.Embedding.from_pretrained(num_genders)
        self.status_embedding = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))

        num_formes = len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1
        self.forme_embedding = nn.Embedding(num_formes, 32)
        # self.level_embedding = nn.Embedding.from_pretrained(torch.eye(102))
        self.hp_sqrt_onehot = nn.Embedding.from_pretrained(
            sqrt_one_hot_matrix(768 if gen != 8 else 1536)
        )
        self.stat_sqrt_onehot = nn.Embedding.from_pretrained(
            power_one_hot_matrix(512, 1 / 3)
        )

        other_stat_size = (
            # self.active_embedding.embedding_dim
            +self.fainted_embedding.embedding_dim
            + self.gender_embedding.embedding_dim
            + self.status_embedding.embedding_dim
            + self.forme_embedding.embedding_dim
            + self.hp_sqrt_onehot.embedding_dim
            + 1
            + 5 * self.stat_sqrt_onehot.embedding_dim
        )

        if gen == 9:
            self.commanding_embedding = nn.Embedding.from_pretrained(torch.eye(3))
            self.reviving_embedding = nn.Embedding.from_pretrained(torch.eye(3))
            self.tera_embedding = nn.Embedding.from_pretrained(torch.eye(3))
            self.teratype_embedding = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
            other_stat_size += (
                self.commanding_embedding.embedding_dim
                + self.reviving_embedding.embedding_dim
                + self.tera_embedding.embedding_dim
                + self.teratype_embedding.embedding_dim
            )

        elif gen == 8:
            self.can_gmax_embedding = nn.Embedding.from_pretrained(torch.eye(3))
            other_stat_size += self.can_gmax_embedding.embedding_dim

        self.other_stats = nn.Linear(other_stat_size, output_dim)

        self.encoder = nn.Sequential(
            nn.Linear(output_dim * 5, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def embed_hp(self, hp: torch.Tensor, maxhp: torch.Tensor) -> torch.Tensor:
        hp_ratio = (hp / maxhp.clamp(min=1)).unsqueeze(-1)
        hp_sqrt_onehot = self.hp_sqrt_onehot(hp)
        hp_sqrt_onehot = torch.cat([hp_ratio, hp_sqrt_onehot], dim=-1)
        return hp_sqrt_onehot

    def embed_moves(
        self, move_tokens: torch.Tensor, move_used: torch.Tensor
    ) -> torch.Tensor:
        moves_emb = self.move_embedding_ae(move_tokens)
        move_used_emb = self.pp_sqrt_onehot(move_used)
        moves_emb = torch.cat((moves_emb, move_used_emb), dim=-1)
        return self.move_embedding(moves_emb)

    def embed_stats(self, stats: torch.Tensor):
        _atk, _def, _spa, _spd, _spe = stats.chunk(5, -1)
        stat_embedding = torch.cat(
            (
                self.stat_sqrt_onehot(_atk),
                self.stat_sqrt_onehot(_def),
                self.stat_sqrt_onehot(_spa),
                self.stat_sqrt_onehot(_spd),
                self.stat_sqrt_onehot(_spe),
            ),
            dim=-1,
        )
        return stat_embedding.squeeze(-2)

    def forward(self, x: torch.Tensor):
        ability = x[..., 0]
        active = x[..., 1]
        fainted = x[..., 2]
        gender = x[..., 3]
        hp = x[..., 4]
        item = x[..., 5]
        level = x[..., 6]
        maxhp = x[..., 7]
        name = x[..., 8]
        forme = x[..., 9]
        stats = x[..., 10:15]
        status = x[..., 15]

        if self.gen == 9:
            commanding = x[..., 16]
            reviving = x[..., 17]
            teraType = x[..., 18]
            terastallized = x[..., 19]

        elif self.gen == 8:
            canGmax = x[..., 16]

        moves = x[..., -8:]
        moves = moves.view(*moves.shape[:-1], 4, 2)
        move_tokens = moves[..., 0]
        move_used = moves[..., 1]

        name_emb = self.pokedex_embedding(name)
        ability_emb = self.ability_embedding(ability)
        item_emb = self.item_embedding(item)
        moves_emb = self.embed_moves(move_tokens, move_used)

        # active_emb = self.active_embedding(active)
        fainted_emb = self.fainted_embedding(fainted)
        gender_emb = self.gender_embedding(gender)
        hp_emb = self.embed_hp(hp, maxhp)
        # level_emb = self.level_embedding(level)
        forme_emb = self.forme_embedding(forme)
        stat_emb = self.embed_stats(stats)
        status_emb = self.status_embedding(status)

        other_stats = [
            # active_emb,
            fainted_emb,
            gender_emb,
            hp_emb,
            forme_emb,
            stat_emb,
            status_emb,
        ]

        if self.gen == 9:
            other_stats += [self.commanding_embedding(commanding)]
            other_stats += [self.reviving_embedding(reviving)]
            other_stats += [self.teratype_embedding(teraType)]
            other_stats += [self.tera_embedding(terastallized)]

        elif self.gen == 8:
            other_stats += [self.can_gmax_embedding(canGmax)]

        mask = (name == 0) | (fainted == 2)

        other_stats = torch.cat(other_stats, dim=-1)
        other_stats = self.other_stats(other_stats)

        pokemon_raw = torch.cat(
            [
                name_emb,
                ability_emb,
                item_emb,
                moves_emb.mean(-2),
                other_stats,
            ],
            dim=-1,
        )
        pokemon_emb = self.encoder(pokemon_raw)
        return pokemon_emb, mask, moves_emb


class PrivateEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PrivateEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        self.embedding = PrivatePokemonEmbedding(
            gen=gen, output_dim=config.entity_embedding_dim  # , frozen=True
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
        self.output = PrivateToVector(
            input_dim=config.entity_embedding_dim,
            output_dim=config.output_dim,
        )

    def moves_given_context(self, active: torch.Tensor, moves: torch.Tensor):
        # active = active.unsqueeze(-2).repeat_interleave(4, -2)
        moves = moves[..., : self.n_active, :, :]
        return moves

    def forward(self, private_reserve: torch.Tensor):
        private_reserve_x = private_reserve + 1

        entity_embeddings, mask, moves_emb = self.embedding(private_reserve_x)

        T, B, S, *_ = entity_embeddings.shape

        mask = mask.to(torch.bool)
        mask = ~mask.view(T * B, S)

        encoder_mask = mask.clone()
        encoder_mask[..., 0] = torch.ones_like(encoder_mask[..., 0])
        encoder_mask = encoder_mask.to(torch.bool)

        entity_embeddings: torch.Tensor = entity_embeddings.view(T * B, S, -1)
        entity_embeddings = self.transformer(entity_embeddings, encoder_mask)

        entity_embeddings = entity_embeddings.view(T * B, S, -1)
        entity_embedding = self.output(entity_embeddings, mask.unsqueeze(-1))
        entity_embedding = entity_embedding.view(T, B, -1)

        mask = mask.view(T, B, S)
        entity_embeddings = entity_embeddings.view(T, B, S, -1)

        active = entity_embeddings[..., : self.n_active, :]
        moves = self.moves_given_context(active, moves_emb)

        return entity_embedding, None, moves, entity_embeddings, mask
