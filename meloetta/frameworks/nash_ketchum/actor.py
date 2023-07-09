import torch

from typing import List

from meloetta.room import BattleRoom

from meloetta.frameworks.nash_ketchum.buffer import ReplayBuffer
from meloetta.frameworks.nash_ketchum.modelv2 import NAshKetchumModel

from meloetta.actors.base import Actor
from meloetta.types import State, Choices, TensorDict


class NAshKetchumActor(Actor):
    def __init__(
        self,
        model: NAshKetchumModel = None,
        replay_buffer: ReplayBuffer = None,
        pid: str = None,
    ):
        self.model = model
        self.gen = model.gen

        self.replay_buffer = replay_buffer

        self.step_index = 0
        self.index = None

        self.turn = 0
        self.pid = pid
        self.turns = []
        self.trajectory = []

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    def choose_action(self, state: State, room: BattleRoom, choices: Choices):
        model_output = self.foward_model(state)
        return self.post_process(state, model_output, room, choices)

    @torch.no_grad()
    def foward_model(self, state: State):
        return self.model.forward(state, compute_log_policy=False, compute_value=False)

    def post_process(
        self, state: State, model_output: TensorDict, room: BattleRoom, choices: Choices
    ):
        postprocess = self.model.postprocess(
            state=state,
            model_output=model_output,
            choices=choices,
        )

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index]

        if self.storing_transition:
            action_type = model_output["action_type_index"].item()
            if action_type == 0:
                policies_to_store = []
                for policy_select, policy_mask in (
                    (0, state["action_type_mask"]),
                    (1, state["flag_mask"]),
                    (2, state["move_mask"]),
                ):
                    # if (policy_mask.sum() > 1).item():
                    policies_to_store.append(policy_select)
                for i, policy_select in enumerate(policies_to_store):
                    model_output["policy_select"] = torch.tensor(
                        policy_select, dtype=torch.long
                    )
                    model_output["utc"] = torch.tensor(
                        self.turn + i / 100 + self.pid / 2, dtype=torch.float64
                    )
                    self.store_transition(state, model_output, room)
            elif action_type == 1:
                policies_to_store = []
                for policy_select, policy_mask in (
                    (0, state["action_type_mask"]),
                    (3, state["switch_mask"]),
                ):
                    # if (policy_mask.sum() > 1).item():
                    policies_to_store.append(policy_select)
                for i, policy_select in enumerate([0, 3]):
                    model_output["policy_select"] = torch.tensor(
                        policy_select, dtype=torch.long
                    )
                    model_output["utc"] = torch.tensor(
                        self.turn + i / 100 + self.pid / 2, dtype=torch.float64
                    )
                    self.store_transition(state, model_output, room)
            self.turn += 1

        return func, args, kwargs

    def get_index(self, battle_tag: str):
        if self.index is None:
            self.index = self.replay_buffer._get_index(battle_tag, self.pid)
        return self.index

    def store_transition(
        self, state: State, model_output: TensorDict, room: BattleRoom
    ):
        to_store = self.model.clean(state)
        to_store = {**to_store, **model_output}
        to_store = {
            k: v
            for k, v in to_store.items()
            if k in self.replay_buffer.buffers.keys() and v is not None
        }
        self.trajectory.append(to_store)

    def post_match(self, room: BattleRoom):
        if self.storing_transition:

            def _prepare_trajectory(trajectory: List[TensorDict]):
                return {
                    key: torch.stack([step[key] for step in trajectory]).squeeze()
                    for key in trajectory[0].keys()
                }

            trajectory = _prepare_trajectory(self.trajectory)
            self.replay_buffer.store_trajectory(
                self.get_index(room.battle_tag), self.pid, trajectory
            )

            datum = room.get_reward()

            battle_tag = room.battle_tag
            index = self.get_index(battle_tag)

            self.replay_buffer.append_reward(index, -1, self.pid, datum["reward"])
            self.replay_buffer.register_done(battle_tag, self.pid)
