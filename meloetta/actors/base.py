import torch
import traceback

from abc import ABC, abstractmethod

from typing import Any, Dict

from meloetta.vector import VectorizedState
from meloetta.room import BattleRoom


Battle = Dict[str, Dict[str, Dict[str, Any]]]


class Actor(ABC):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.choose_action(*args, **kwargs)
        except Exception as e:
            room = args[1]
            print(room.battle_tag)
            print(traceback.format_exc())

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError

    def post_match(self, room: BattleRoom):
        pass

    def get_vectorized_state(
        self, room: BattleRoom, battle: Battle
    ) -> Dict[str, torch.Tensor]:
        return VectorizedState.from_battle(room, battle).to_dict()
