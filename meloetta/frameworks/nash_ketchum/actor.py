import torch

from typing import Tuple

from meloetta.room import BattleRoom

from meloetta.frameworks.nash_ketchum import ReplayBuffer
from meloetta.frameworks.nash_ketchum.model import (
    NAshKetchumModel,
    EnvStep,
    PostProcess,
)

from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices
from meloetta.frameworks.nash_ketchum.model.interfaces import Indices


class NAshKetchumActor(Actor):
    def __init__(
        self,
        model: NAshKetchumModel,
        replay_buffer: ReplayBuffer,
    ):
        self._model = model
        self._replay_buffer = replay_buffer

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
        store_transition: bool = True,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        output: Tuple[EnvStep, PostProcess]
        with torch.no_grad():
            output = self._model(state, hidden_state, choices)

        env_step, postprocess, hidden_state = output
        if store_transition:
            self.store_transition(state, env_step, room)

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs, hidden_state

    def store_transition(self, state: State, env_step: EnvStep, room: BattleRoom):
        state = self._model.clean(state)
        to_store = env_step.to_store(state)
        self._replay_buffer.store_sample(room.battle_tag, to_store)

    def store_reward(
        self,
        room: BattleRoom,
        pid: int,
        reward: float = None,
        store_transition: bool = True,
    ):
        if store_transition:
            self._replay_buffer.append_reward(room.battle_tag, pid, reward)
            self._replay_buffer.register_done(room.battle_tag)
