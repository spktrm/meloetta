import torch
import requests
import torch.nn.functional as F

from torch import nn

from typing import List, NamedTuple

from meloetta.data import (
    ROOT_DIR,
    BattleTypeChart,
    to_id,
    get_species_token,
    get_ability_token,
    get_item_token,
    get_move_token,
    get_type_token,
)


class PokedexEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"{ROOT_DIR}/pretrained/gen{gen}/pokedex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class AbilityEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"{ROOT_DIR}/pretrained/gen{gen}/abilitydex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class MoveEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"{ROOT_DIR}/pretrained/gen{gen}/movedex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class ItemEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        embeddings: torch.Tensor
        names, embeddings = torch.load(f"{ROOT_DIR}/pretrained/gen{gen}/itemdex.pt")
        embeddings = embeddings.to(dtype)
        self.names: List[str] = names
        self.num_embeddings = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[-1]
        self.emb = nn.Embedding.from_pretrained(embeddings)

    def get_name(self, x: torch.Tensor):
        return [self.names[token] for token in x.squeeze().tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class RandBatsOutput(NamedTuple):
    species: torch.Tensor
    abilities: torch.Tensor
    items: torch.Tensor
    moves: torch.Tensor
    teratypes: torch.Tensor = None


class RandomBattlesEmbedding(nn.Module):
    def __init__(self, gen: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        data = requests.get(
            f"https://raw.githubusercontent.com/pkmn/randbats/main/data/gen{gen}randombattle.json"
        ).json()

        self.gen = gen
        ability_embedding = AbilityEmbedding(gen)
        pokedex_embedding = PokedexEmbedding(gen)
        move_embedding = MoveEmbedding(gen)
        item_embedding = ItemEmbedding(gen)

        self.species = torch.zeros(
            pokedex_embedding.num_embeddings,
            pokedex_embedding.num_embeddings - 1,
            dtype=dtype,
        )
        self.abilities = torch.zeros(
            pokedex_embedding.num_embeddings,
            ability_embedding.num_embeddings - 1,
            dtype=dtype,
        )
        self.movesets = torch.zeros(
            pokedex_embedding.num_embeddings,
            move_embedding.num_embeddings - 1,
            dtype=dtype,
        )
        self.items = torch.zeros(
            pokedex_embedding.num_embeddings,
            item_embedding.num_embeddings - 1,
            dtype=dtype,
        )
        self.teratypes = torch.zeros(
            pokedex_embedding.num_embeddings,
            len(BattleTypeChart),
            dtype=dtype,
        )

        for species_, datum in data.items():
            species_token = get_species_token(9, "id", to_id(species_)) + 1
            assert species_token > 0, species_

            self.species[species_token][species_token - 1] = 1

            ability_tokens = torch.tensor(
                [
                    get_ability_token(9, "name", to_id(value))
                    for value in datum["abilities"]
                ]
            )
            self.abilities[species_token][...] = F.one_hot(
                ability_tokens, ability_embedding.num_embeddings - 1
            ).sum(0)

            species_moves = set()
            species_teratypes = set()
            for role in datum["roles"].values():
                species_moves.update(role["moves"])
                species_teratypes.update(role["teraTypes"])

            move_tokens = torch.tensor(
                [get_move_token(9, "id", to_id(value)) for value in species_moves]
            )
            self.movesets[species_token][...] = F.one_hot(
                move_tokens + 1, move_embedding.num_embeddings
            ).sum(0)[..., 1:]

            item_tokens = torch.tensor(
                [
                    get_item_token(9, "id", to_id(value))
                    for value in datum.get(
                        "items",
                    )
                    or [-1]
                ]
            )
            self.items[species_token][...] = F.one_hot(
                item_tokens + 1, item_embedding.num_embeddings
            ).sum(0)[..., 1:]

            teratype_tokens = torch.tensor(
                [get_type_token(9, to_id(value)) for value in species_teratypes]
            )
            if gen == 9:
                self.teratypes[species_token][...] = F.one_hot(
                    teratype_tokens, self.teratypes.shape[-1]
                ).sum(0)

        # self.species = self.species / (self.species.sum(-1, keepdim=True).clamp(min=1))
        # self.abilities = self.abilities / (
        #     self.abilities.sum(-1, keepdim=True).clamp(min=1)
        # )
        # self.movesets = self.movesets / (
        #     self.movesets.sum(-1, keepdim=True).clamp(min=1)
        # )
        # self.items = self.items / (self.items.sum(-1, keepdim=True).clamp(min=1))
        # self.teratypes = self.teratypes / (
        #     self.teratypes.sum(-1, keepdim=True).clamp(min=1)
        # )

        self.species = nn.Embedding.from_pretrained(self.species)
        self.items = nn.Embedding.from_pretrained(self.items)
        self.abilities = nn.Embedding.from_pretrained(self.abilities)
        self.movesets = nn.Embedding.from_pretrained(self.movesets)
        self.teratypes = nn.Embedding.from_pretrained(self.teratypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RandBatsOutput(
            species=self.species(x),
            abilities=self.abilities(x),
            items=self.items(x),
            moves=self.movesets(x),
            teratypes=(self.teratypes(x) if self.gen == 9 else None),
        )
