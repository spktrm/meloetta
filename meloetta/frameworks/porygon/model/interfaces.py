import math
import torch
import collections

from typing import NamedTuple, Iterator, Dict, List

from meloetta.actors.types import State, Choices


class SideEncoderOutput(NamedTuple):
    side_embedding: torch.Tensor
    private_entity: torch.Tensor
    public_entity: torch.Tensor
    moves: torch.Tensor
    switches: torch.Tensor


class EncoderOutput(NamedTuple):
    moves: torch.Tensor
    switches: torch.Tensor
    private_entity: torch.Tensor
    public_entity: torch.Tensor
    side_embedding: torch.Tensor
    weather_emb: torch.Tensor
    scalar_emb: torch.Tensor


class Indices(NamedTuple):
    action_index: torch.Tensor
    max_move_index: torch.Tensor
    flag_index: torch.Tensor
    target_index: torch.Tensor


class Logits(NamedTuple):
    action_logits: torch.Tensor = None
    max_move_logits: torch.Tensor = None
    flag_logits: torch.Tensor = None
    target_logits: torch.Tensor = None


class Policy(NamedTuple):
    action_policy: torch.Tensor = None
    max_move_policy: torch.Tensor = None
    flag_policy: torch.Tensor = None
    target_policy: torch.Tensor = None


class LogPolicy(NamedTuple):
    action_log_policy: torch.Tensor = None
    max_move_log_policy: torch.Tensor = None
    flag_log_policy: torch.Tensor = None
    target_log_policy: torch.Tensor = None


class TrainingOutput(NamedTuple):
    pi: Policy
    v: torch.Tensor
    logit: Logits
    log_pi: LogPolicy = None


class ModelOutput(NamedTuple):
    indices: Indices
    policy: Policy
    logits: Logits
    value: torch.Tensor

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
    sides: torch.Tensor = None
    boosts: torch.Tensor = None
    volatiles: torch.Tensor = None
    side_conditions: torch.Tensor = None
    pseudoweathers: torch.Tensor = None
    weather: torch.Tensor = None
    wisher: torch.Tensor = None
    turn: torch.Tensor = None
    n: torch.Tensor = None
    total_pokemon: torch.Tensor = None
    faint_counter: torch.Tensor = None
    turn: torch.Tensor = None
    turns_since_last_move: torch.Tensor = None
    action_type_mask: torch.Tensor = None
    move_mask: torch.Tensor = None
    max_move_mask: torch.Tensor = None
    switch_mask: torch.Tensor = None
    flag_mask: torch.Tensor = None
    rewards: torch.Tensor = None
    player_id: torch.Tensor = None
    action_policy: torch.Tensor = None
    max_move_policy: torch.Tensor = None
    flag_policy: torch.Tensor = None
    action_index: torch.Tensor = None
    max_move_index: torch.Tensor = None
    flag_index: torch.Tensor = None
    value: torch.Tensor = None
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
    action_policy_loss: float = 0
    max_move_policy_loss: float = 0
    flag_policy_loss: float = 0
    target_policy_loss: float = 0

    action_value_loss: float = 0
    max_move_value_loss: float = 0
    flag_value_loss: float = 0
    target_value_loss: float = 0

    action_entropy_loss: float = 0
    max_move_entropy_loss: float = 0
    flag_entropy_loss: float = 0
    target_entropy_loss: float = 0

    recon_loss: float = 0

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


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
        "kl_loss",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


class Targets(NamedTuple):
    action_vtrace_returns: VTraceFromLogitsReturns = None
    max_move_vtrace_returns: VTraceFromLogitsReturns = None
    flag_vtrace_returns: VTraceFromLogitsReturns = None
    target_vtrace_returns: VTraceFromLogitsReturns = None

    @property
    def batch_size(self):
        return self.action_vtrace_returns.vs.shape[1]

    @property
    def trajectory_length(self):
        return self.action_vtrace_returns.vs.shape[0]

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
                targets_dict[key] = VTraceFromLogitsReturns(
                    **{
                        key: value[:max_length, start:end].contiguous()
                        for key, value in lst._asdict().items()
                    }
                )
        return Targets(**targets_dict)
