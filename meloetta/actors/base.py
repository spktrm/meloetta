import ray
import torch

from abc import ABC, abstractmethod

from typing import Any, Dict

from meloetta.vector import VectorizedState
from meloetta.room import BattleRoom
from meloetta.types import State, Choices, Battle


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
        return self.choose_action(env_output, room, choices)

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
        return VectorizedState.from_battle(room, battle)
