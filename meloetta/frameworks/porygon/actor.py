import torch

from typing import Tuple

from meloetta.room import BattleRoom

from meloetta.frameworks.porygon import ReplayBuffer
from meloetta.frameworks.porygon.model import (
    PorygonModel,
    ModelOutput,
    PostProcess,
)

from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices


class PorygonActor(Actor):
    def __init__(
        self,
        model: PorygonModel,
        replay_buffer: ReplayBuffer = None,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.hidden_state = model.core.initial_state(1)
        if replay_buffer is not None:
            self.buffer_index = self.replay_buffer._get_index()
        self.step_index = 0

        self.env_outputs = []
        self.model_outputs = []

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    def choose_action(
        self,
        env_output: State,
        room: BattleRoom,
        choices: Choices,
    ):
        output: Tuple[ModelOutput, PostProcess]
        with torch.no_grad():
            output = self.model(env_output, self.hidden_state, choices)
        model_output, postprocess, self.hidden_state = output

        if self.storing_transition:
            self._store_transition(env_output, model_output, room)

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs

    def _clean_env_output(self, env_output: State):
        return {
            k: env_output[k]
            for k in self.model.state_fields
            if isinstance(env_output[k], torch.Tensor)
        }

    def _clean_model_output(self, model_output: ModelOutput):
        to_store = {}
        to_store.update(
            {
                k: v.squeeze()
                for k, v in model_output.indices._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        to_store.update(
            {
                k: v.squeeze()
                for k, v in model_output._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        return to_store

    def _store_transition(
        self, env_output: State, model_output: ModelOutput, room: BattleRoom
    ):
        env_output = self._clean_env_output(env_output)
        model_output = self._clean_model_output(model_output)

        self.env_outputs.append(env_output)
        self.model_outputs.append(model_output)

    def _populate_buffer(self) -> int:
        for step_index, (env_output, model_output) in enumerate(
            zip(self.env_outputs[1:], self.model_outputs)
        ):
            to_store = {**env_output, **model_output}

            self.replay_buffer.store_sample(self.buffer_index, step_index, to_store)

        return step_index

    def post_match(self, room: BattleRoom):
        if self.storing_transition:

            datum = room.get_reward()
            reward = datum["reward"]

            final_turn = self._populate_buffer()

            self.replay_buffer.append_reward(self.buffer_index, final_turn, reward)
