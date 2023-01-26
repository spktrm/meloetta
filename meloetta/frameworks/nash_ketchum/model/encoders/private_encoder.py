import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.nash_ketchum.model import config
from meloetta.frameworks.nash_ketchum.model.utils import (
    sqrt_one_hot_matrix,
    GLU,
    TransformerEncoder,
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


nn.MultiheadAttention


class PrivateEncoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: config.PrivateEncoderConfig):
        super().__init__()

        self.config = config
        self.gen = gen
        self.n_active = n_active

        # onehots
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(3))
        self.gender_onehot = nn.Embedding.from_pretrained(torch.eye(len(GENDERS) + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(len(STATUS) + 1))
        self.move_slot_onehot = nn.Embedding.from_pretrained(torch.eye(5))
        self.forme_embedding = nn.Embedding.from_pretrained(
            torch.eye(len(TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"]["forme"]) + 1)
        )

        self.hp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(2048))
        self.level_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(102))

        self.atk_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(1024))
        self.def_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(1024))
        self.spa_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(1024))
        self.spd_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(1024))
        self.spe_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(1024))

        self.pp_sqrt_onehot = nn.Embedding.from_pretrained(sqrt_one_hot_matrix(64))

        # precomputed embeddings
        ability_embedding = AbilityEmbedding(gen=gen)
        self.ability_embedding = nn.Sequential(
            ability_embedding,
            nn.Linear(ability_embedding.embedding_dim, config.embedding_dim),
            nn.ReLU(),
        )
        pokedex_embedding = PokedexEmbedding(gen=gen)
        self.pokedex_embedding = nn.Sequential(
            pokedex_embedding,
            nn.Linear(pokedex_embedding.embedding_dim, config.embedding_dim),
            nn.ReLU(),
        )
        self.move_embedding = MoveEmbedding(gen=gen)
        self.move_lin = nn.Linear(
            self.move_embedding.embedding_dim + 5 + self.pp_sqrt_onehot.embedding_dim,
            config.embedding_dim,
        )

        item_embedding = ItemEmbedding(gen=gen)
        self.item_embedding = nn.Sequential(
            item_embedding,
            nn.Linear(item_embedding.embedding_dim, config.embedding_dim),
            nn.ReLU(),
        )
        hp_embedding_dim = self.hp_sqrt_onehot.embedding_dim + 1
        level_embedding_dim = self.level_sqrt_onehot.embedding_dim + 1
        stat_embedding_dim = (
            self.atk_sqrt_onehot.embedding_dim
            + self.def_sqrt_onehot.embedding_dim
            + self.spa_sqrt_onehot.embedding_dim
            + self.spd_sqrt_onehot.embedding_dim
            + self.spe_sqrt_onehot.embedding_dim
        )
        entity_in = (
            config.embedding_dim
            + self.active_onehot.embedding_dim
            + self.fainted_onehot.embedding_dim
            + self.gender_onehot.embedding_dim
            + hp_embedding_dim
            + config.embedding_dim
            + level_embedding_dim
            + config.embedding_dim
            + self.forme_embedding.embedding_dim
            + stat_embedding_dim
            + self.status_onehot.embedding_dim
            + config.embedding_dim
        )
        if gen == 9:
            self.commanding_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.reviving_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.tera_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            self.teratype_onehot = nn.Embedding.from_pretrained(
                torch.eye(len(BattleTypeChart) + 1)
            )
            entity_in += (
                self.commanding_onehot.embedding_dim
                + self.reviving_onehot.embedding_dim
                + self.tera_onehot.embedding_dim
                + self.teratype_onehot.embedding_dim
            )

        elif gen == 8:
            self.can_gmax_onehot = nn.Embedding.from_pretrained(torch.eye(3))
            entity_in += self.can_gmax_onehot.embedding_dim

        self.entity_lin = nn.Linear(entity_in, config.entity_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.entity_embedding_dim,
            nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            enable_nested_tensor=False,
            num_layers=config.transformer_num_layers,
        )
        self.active_lin_out = nn.Linear(
            config.entity_embedding_dim, config.entity_embedding_dim
        )
        self.reserve_lin_out = nn.Linear(
            config.entity_embedding_dim, config.entity_embedding_dim
        )
        self.lin_out = nn.Linear(
            2 * config.entity_embedding_dim, config.entity_embedding_dim
        )
        self.glu = GLU(
            config.entity_embedding_dim,
            config.entity_embedding_dim,
            config.embedding_dim,
        )

    def embed_hp(self, hp: torch.Tensor, maxhp: torch.Tensor) -> torch.Tensor:
        hp_ratio = (hp / maxhp.clamp(min=1)).unsqueeze(-1)
        hp_sqrt_onehot = self.hp_sqrt_onehot(hp)
        hp_sqrt_onehot = torch.cat([hp_ratio, hp_sqrt_onehot], dim=-1)
        return hp_sqrt_onehot

    def embed_stat(
        self,
        stat_atk: torch.Tensor,
        stat_def: torch.Tensor,
        stat_spa: torch.Tensor,
        stat_spd: torch.Tensor,
        stat_spe: torch.Tensor,
    ) -> torch.Tensor:
        stat_atk_emb = self.atk_sqrt_onehot(stat_atk)
        stat_def_emb = self.def_sqrt_onehot(stat_def)
        stat_spa_emb = self.spa_sqrt_onehot(stat_spa)
        stat_spd_emb = self.spd_sqrt_onehot(stat_spd)
        stat_spe_emb = self.spe_sqrt_onehot(stat_spe)
        stat_emb = torch.cat(
            [
                stat_atk_emb,
                stat_def_emb,
                stat_spa_emb,
                stat_spd_emb,
                stat_spe_emb,
            ],
            dim=-1,
        )
        return stat_emb

    def embed_level(self, level: torch.Tensor) -> torch.Tensor:
        level_sqrt_onehot = self.level_sqrt_onehot(level)
        level_ratio = (level / 100).unsqueeze(-1)
        level_emb = torch.cat((level_sqrt_onehot, level_ratio), dim=-1)
        return level_emb

    def embed_pp(self, pp: torch.Tensor) -> torch.Tensor:
        pp_bin = self.pp_sqrt_onehot(pp)
        return pp_bin

    def embed_moves(
        self, move_tokens: torch.Tensor, move_used: torch.Tensor
    ) -> torch.Tensor:
        moves_emb = self.move_embedding(move_tokens)
        move_used_emb = self.embed_pp(move_used)
        move_slot = torch.ones_like(move_used)
        for i in range(4):
            move_slot[..., i] = i
        move_slot_emb = self.move_slot_onehot(move_used)
        moves_emb = torch.cat((moves_emb, move_used_emb, move_slot_emb), dim=-1)
        return F.relu(self.move_lin(moves_emb))

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

        hp_emb = self.embed_hp(hp, maxhp)

        item_emb = self.item_embedding(item)

        level_emb = self.embed_level(level)

        name_emb = self.pokedex_embedding(name)
        forme_emb = self.forme_embedding(forme)

        stat_emb = self.embed_stat(stat_atk, stat_def, stat_spa, stat_spd, stat_spe)

        status_emb = self.status_onehot(status)
        moves_emb = self.embed_moves(move_tokens, move_used)

        mon_emb = [
            ability_emb,
            active_emb,
            fainted_emb,
            gender_emb,
            hp_emb,
            item_emb,
            level_emb,
            name_emb,
            forme_emb,
            stat_emb,
            status_emb,
            moves_emb.sum(-2),
        ]

        if self.gen == 9:
            commanding_onehot = self.commanding_onehot(commanding)
            reviving_onehot = self.reviving_onehot(reviving)
            teraType_onehot = self.teratype_onehot(teraType)
            terastallized_onehot = self.tera_onehot(terastallized)
            mon_emb += [
                commanding_onehot,
                reviving_onehot,
                teraType_onehot,
                terastallized_onehot,
            ]

        elif self.gen == 8:
            can_gmax_emb = self.can_gmax_onehot(canGmax)
            mon_emb += [can_gmax_emb]

        entity_embeddings = torch.cat(mon_emb, dim=-1)
        entity_embeddings = F.relu(self.entity_lin(entity_embeddings))
        T, B, S, *_ = entity_embeddings.shape

        mask = name == 0
        mask = mask.view(T * B, S)

        encoder_mask = mask.clone()
        encoder_mask[..., 0] = False

        entity_embeddings = entity_embeddings.view(T * B, S, -1)
        entity_embeddings: torch.Tensor = self.encoder(
            entity_embeddings, src_key_padding_mask=encoder_mask
        )

        entity_embeddings = entity_embeddings.view(T * B, S, -1)
        mask = mask.view(T * B, S, -1)

        entity_embeddings = entity_embeddings * ~mask

        num_active_entities = (~mask)[..., : self.n_active, :].sum(-2)
        num_active_entities = num_active_entities.clamp(min=1)
        active_embedding = entity_embeddings[..., : self.n_active, :]
        active_embedding = active_embedding.sum(-2)
        active_embedding = active_embedding / num_active_entities
        active_embedding = F.relu(self.active_lin_out(active_embedding))

        num_reserve_entities = (~mask)[..., self.n_active :, :].sum(-2)
        num_reserve_entities = num_reserve_entities.clamp(min=1)
        reserve_embedding = entity_embeddings[..., self.n_active :, :]
        reserve_embedding = reserve_embedding.sum(-2)
        reserve_embedding = reserve_embedding / num_reserve_entities
        reserve_embedding = F.relu(self.reserve_lin_out(reserve_embedding))

        entity_embedding = torch.cat((active_embedding, reserve_embedding), dim=-1)
        entity_embedding = self.lin_out(entity_embedding)
        entity_embedding = entity_embedding.view(T, B, -1)

        entity_embeddings = entity_embeddings.view(T, B, S, -1)

        active = entity_embeddings[..., : self.n_active, :]
        moves = self.glu(
            active.unsqueeze(-2).repeat_interleave(4, -2),
            moves_emb[..., : self.n_active, :, :],
        )

        return entity_embedding, moves, entity_embeddings
