import math
import torch

from typing import NamedTuple, Iterator, Tuple, Dict, List

from meloetta.actors.types import State, Choices


class EncoderOutput(NamedTuple):
    private_entity_emb: torch.Tensor
    moves: torch.Tensor
    switches: torch.Tensor
    public_entity_emb: torch.Tensor
    public_spatial: torch.Tensor
    private_spatial: torch.Tensor
    weather_emb: torch.Tensor
    scalar_emb: torch.Tensor


class Indices(NamedTuple):
    action_type_index: torch.Tensor
    move_index: torch.Tensor
    max_move_index: torch.Tensor
    switch_index: torch.Tensor
    flag_index: torch.Tensor
    target_index: torch.Tensor


class Logits(NamedTuple):
    action_type_logits: torch.Tensor = None
    move_logits: torch.Tensor = None
    max_move_logits: torch.Tensor = None
    switch_logits: torch.Tensor = None
    flag_logits: torch.Tensor = None
    target_logits: torch.Tensor = None


class Policy(NamedTuple):
    action_type_policy: torch.Tensor = None
    move_policy: torch.Tensor = None
    max_move_policy: torch.Tensor = None
    switch_policy: torch.Tensor = None
    flag_policy: torch.Tensor = None
    target_policy: torch.Tensor = None


class LogPolicy(NamedTuple):
    action_type_log_policy: torch.Tensor = None
    move_log_policy: torch.Tensor = None
    max_move_log_policy: torch.Tensor = None
    switch_log_policy: torch.Tensor = None
    flag_log_policy: torch.Tensor = None
    target_log_policy: torch.Tensor = None


class TrainingOutput(NamedTuple):
    pi: Policy
    v: torch.Tensor
    logit: Logits
    log_pi: LogPolicy = None


class EnvStep(NamedTuple):
    indices: Indices
    policy: Policy
    logits: Logits

    def to_store(self, state: State) -> Dict[str, torch.Tensor]:
        to_store = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        to_store.update(
            {
                k: v.squeeze()
                for k, v in self.indices._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        to_store.update(
            {
                k: v.squeeze()
                for k, v in self.policy._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        return to_store


class PostProcess(NamedTuple):
    data: Choices
    index: torch.Tensor


class Batch(NamedTuple):
    private_reserve: torch.Tensor = None
    public_n: torch.Tensor = None
    public_total_pokemon: torch.Tensor = None
    public_faint_counter: torch.Tensor = None
    public_side_conditions: torch.Tensor = None
    public_wisher: torch.Tensor = None
    public_active: torch.Tensor = None
    public_reserve: torch.Tensor = None
    public_stealthrock: torch.Tensor = None
    public_spikes: torch.Tensor = None
    public_toxicspikes: torch.Tensor = None
    public_stickyweb: torch.Tensor = None
    weather: torch.Tensor = None
    weather_time_left: torch.Tensor = None
    weather_min_time_left: torch.Tensor = None
    pseudo_weather: torch.Tensor = None
    turn: torch.Tensor = None
    turns_since_last_move: torch.Tensor = None
    action_type_mask: torch.Tensor = None
    move_mask: torch.Tensor = None
    max_move_mask: torch.Tensor = None
    switch_mask: torch.Tensor = None
    flag_mask: torch.Tensor = None
    rewards: torch.Tensor = None
    player_id: torch.Tensor = None
    action_type_policy: torch.Tensor = None
    move_policy: torch.Tensor = None
    max_move_policy: torch.Tensor = None
    switch_policy: torch.Tensor = None
    flag_policy: torch.Tensor = None
    action_type_index: torch.Tensor = None
    move_index: torch.Tensor = None
    max_move_index: torch.Tensor = None
    switch_index: torch.Tensor = None
    flag_index: torch.Tensor = None
    valid: torch.Tensor = None

    def get_index_from_policy(self, policy_field: str):
        return getattr(self, policy_field.replace("policy", "index"))

    def get_mask_from_policy(self, policy_field: str):
        return getattr(self, policy_field.replace("policy", "mask"))

    @property
    def batch_size(self):
        return self.valid.shape[1]

    @property
    def trajectory_lengths(self):
        return self.valid.sum(0)

    @property
    def trajectory_length(self):
        return self.valid.shape[0]

    def slice(self, start: int, end: int, max_length: int = None) -> "Batch":
        return Batch(
            **{
                key: value[:max_length, start:end].contiguous()
                for key, value in self._asdict().items()
                if value is not None
            }
        )


class Loss(NamedTuple):
    action_type_policy_loss: float = 0
    move_policy_loss: float = 0
    max_move_policy_loss: float = 0
    switch_policy_loss: float = 0
    flag_policy_loss: float = 0
    target_policy_loss: float = 0

    action_type_value_loss: float = 0
    move_value_loss: float = 0
    max_move_value_loss: float = 0
    switch_value_loss: float = 0
    flag_value_loss: float = 0
    target_value_loss: float = 0

    def to_log(self, batch: Batch):
        logs = {
            key: value for key, value in self._asdict().items() if value is not None
        }
        logs["traj_len"] = batch.valid.sum(0).float().mean().item()
        logs["final_turn"] = (
            (batch.turn * batch.valid).max(0).values.float().mean().item()
        )
        return logs

    def entropy(self):
        return


class Targets(NamedTuple):
    action_type_value_target: List[torch.Tensor] = None
    action_type_policy_target: List[torch.Tensor] = None
    action_type_has_played: List[torch.Tensor] = None

    move_value_target: List[torch.Tensor] = None
    move_policy_target: List[torch.Tensor] = None
    move_has_played: List[torch.Tensor] = None

    max_move_value_target: List[torch.Tensor] = None
    max_move_policy_target: List[torch.Tensor] = None
    max_move_has_played: List[torch.Tensor] = None

    switch_value_target: List[torch.Tensor] = None
    switch_policy_target: List[torch.Tensor] = None
    switch_has_played: List[torch.Tensor] = None

    flag_value_target: List[torch.Tensor] = None
    flag_policy_target: List[torch.Tensor] = None
    flag_has_played: List[torch.Tensor] = None

    target_value_target: List[torch.Tensor] = None
    target_policy_target: List[torch.Tensor] = None
    target_has_played: List[torch.Tensor] = None

    importance_sampling_corrections: List[torch.Tensor] = None

    @property
    def batch_size(self):
        return self.action_type_value_target[0].shape[1]

    @property
    def trajectory_length(self):
        return self.action_type_value_target[0].shape[0]

    def iterate(
        self, minibatch_size: int = 16, minitraj_size: int = 16
    ) -> Iterator["Targets"]:
        for batch_index in range(math.ceil(self.batch_size / minibatch_size)):
            for traj_index in range(math.ceil(self.trajectory_length / minitraj_size)):
                batch_start = minibatch_size * batch_index
                batch_end = minibatch_size * (batch_index + 1)

                traj_start = minitraj_size * traj_index
                traj_end = minitraj_size * (traj_index + 1)

                minibatch = {}
                for key, lst in self._asdict().items():
                    if lst is not None:
                        value1, value2 = lst
                        minibatch[key] = [
                            value1[
                                traj_start:traj_end, batch_start:batch_end
                            ].contiguous(),
                            value2[
                                traj_start:traj_end, batch_start:batch_end
                            ].contiguous(),
                        ]

                yield Targets(**minibatch)

    def get_value_target(self, policy_field: str):
        return getattr(self, policy_field.replace("_policy", "_value_target"))

    def get_policy_target(self, policy_field: str):
        return getattr(self, policy_field + "_target")

    def get_has_played(self, policy_field: str, mask: torch.Tensor):
        has_played = [
            hp * mask
            for hp in getattr(self, policy_field.replace("_policy", "_has_played"))
        ]
        return has_played

    def slice(self, start: int, end: int, max_length: int = None) -> "Targets":
        targets_dict = {}
        for key, lst in self._asdict().items():
            if lst is not None:
                value1, value2 = lst
                targets_dict[key] = [
                    value1[:max_length, start:end].contiguous(),
                    value2[:max_length, start:end].contiguous(),
                ]
        return Targets(**targets_dict)
