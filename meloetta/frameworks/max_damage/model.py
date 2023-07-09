import json
import random

import torch
import torch.nn as nn

from meloetta.types import State, Choices
from meloetta.embeddings import MoveEmbedding


with open("meloetta/pretrained/schema.json", "r") as f:
    schema = json.load(f)


class MaxDamageModel(nn.Module):
    def __init__(self, gen: int):
        super().__init__()

        self.schema = schema[f"gen{gen}"]
        self.offset = len(self.schema["movedex"]["basePower"])
        self.basePowers = torch.tensor(self.schema["movedex"]["basePower"])
        self.move_embedding = MoveEmbedding(gen=gen)

    def forward(
        self,
        state: State,
        choices: Choices,
    ):
        private_reserve = state["sides"][..., 0, :, :].squeeze()
        private_reserve_x = private_reserve + 1

        active = private_reserve_x[..., 12] == 2

        try:
            if choices["moves"] and active.sum():
                active_mons = private_reserve_x[active]

                move_tokens = active_mons[..., -12:-8].long()
                move_names = self.move_embedding.get_name(move_tokens)
                moves_emb = self.move_embedding(move_tokens)
                moves_basepower = moves_emb[..., : self.offset].squeeze()
                moves_basepower = (moves_basepower * self.basePowers).max(-1).values
                moves_basepower = torch.masked_fill(
                    moves_basepower, ~state["move_mask"].squeeze(), -1
                )

                index = torch.argmax(moves_basepower, -1).item()
                data = choices["moves"]
                func, args, kwargs = data[index]
            else:
                random_key = random.choice(
                    [key for key, value in choices.items() if value]
                )
                _, (func, args, kwargs) = random.choice(
                    list(choices[random_key].items())
                )
        except:
            random_key = random.choice([key for key, value in choices.items() if value])
            _, (func, args, kwargs) = random.choice(list(choices[random_key].items()))
        finally:
            return func, args, kwargs
