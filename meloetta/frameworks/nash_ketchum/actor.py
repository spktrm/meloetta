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


class NAshKetchumActor(Actor):
    def __init__(
        self,
        model: NAshKetchumModel,
        replay_buffer: ReplayBuffer = None,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.hidden_state = model.core.initial_state(1)

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        output: Tuple[EnvStep, PostProcess]
        with torch.no_grad():
            output = self.model(state, self.hidden_state, choices)
        env_step, postprocess, self.hidden_state = output

        if self.storing_transition:
            self.store_transition(state, env_step, room)

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs

    def store_transition(self, state: State, env_step: EnvStep, room: BattleRoom):
        state = self.model.clean(state)
        to_store = env_step.to_store(state)
        self.replay_buffer.store_sample(room.battle_tag, to_store)

    def post_match(self, room: BattleRoom):
        if self.storing_transition:
            datum = room.get_reward()
            self.replay_buffer.append_reward(
                room.battle_tag, datum["pid"], datum["reward"]
            )
            self.replay_buffer.register_done(room.battle_tag)
