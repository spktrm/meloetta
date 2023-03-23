import os
import json
import torch
import traceback

from abc import ABC, abstractmethod

from typing import Any, Dict

from meloetta.vector import VectorizedState
from meloetta.room import BattleRoom
from meloetta.actors.types import State, Choices, Battle


Battle = Dict[str, Dict[str, Dict[str, Any]]]


class Actor(ABC):
    def __call__(
        self,
        env_output: State,
        room: BattleRoom,
        choices: Choices,
        *args,
        **kwargs,
    ) -> Any:
        try:
            return self.choose_action(env_output, room, choices)
        except Exception as e:
            battle_tag = room.battle_tag
            state = room.get_state()
            side = state["side"]
            print(f"{battle_tag}: Error Occured")
            datum = {
                "state": state,
                "choices": {
                    k: {sk: sv[1:] for sk, sv in v.items()} for k, v in choices.items()
                },
                "traceback": traceback.format_exc(),
            }
            if not os.path.exists("errors"):
                os.mkdir("errors")
            with open(f"errors/{battle_tag}-{side}.json", "w") as f:
                json.dump(datum, f)

    @abstractmethod
    def choose_action(
        self,
        env_output: State,
        room: BattleRoom,
        choices: Choices,
    ):
        raise NotImplementedError

    def post_match(self, room: BattleRoom):
        pass

    def get_vectorized_state(
        self, room: BattleRoom, battle: Battle
    ) -> Dict[str, torch.Tensor]:
        return VectorizedState.from_battle(room, battle).to_dict()
