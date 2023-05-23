import math
import torch

from typing import NamedTuple, Dict, List

from meloetta.actors.types import State, Choices


class SideEncoderOutput(NamedTuple):
    pokemon_embedding: torch.Tensor
    boosts: torch.Tensor
    volatiles: torch.Tensor
    side_conditions: torch.Tensor
    moves: torch.Tensor
    switches: torch.Tensor


class EncoderOutput(NamedTuple):
    pokemon_embedding: torch.Tensor
    boosts: torch.Tensor
    volatiles: torch.Tensor
    side_conditions: torch.Tensor
    moves: torch.Tensor
    switches: torch.Tensor
    weather_emb: torch.Tensor
    scalar_emb: torch.Tensor


class Indices(NamedTuple):
    action_type_index: torch.Tensor = None
    move_index: torch.Tensor = None
    max_move_index: torch.Tensor = None
    switch_index: torch.Tensor = None
    flag_index: torch.Tensor = None
    target_index: torch.Tensor = None

    @classmethod
    def from_list(self, outputs: List[Dict[str, torch.Tensor]]):
        buckets = {k: [] for k in self._fields}
        for output in outputs:
            for key in buckets:
                value = getattr(output, key, None)
                if value is not None:
                    buckets[key].append(value)
        for key, value in buckets.items():
            if buckets[key]:
                buckets[key] = torch.cat(buckets[key], dim=1)
            else:
                buckets[key] = None
        return self(**buckets)

    def to_json(self):
        return {
            k: v.squeeze().tolist() for k, v in self._asdict().items() if v is not None
        }

    def to(self, device):
        return Indices(
            **{k: v.to(device) for k, v in self._asdict().items() if v is not None}
        )

    def flatten(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {
            k: v.flatten(*args, *kwargs).unsqueeze(0)
            for k, v in self._asdict().items()
            if v is not None
        }

    def flatten_without_padding(self, t: torch.Tensor):
        return {
            k: torch.cat([v[: t[i], i] for i in range(v.shape[1])]).unsqueeze(0)
            for k, v in self._asdict().items()
            if v is not None
        }

    def view(self, t, b):
        return Indices(
            **{
                k: v.view(t, b, *v.shape[2:])
                for k, v in self._asdict().items()
                if v is not None
            }
        )


class Policy(NamedTuple):
    action_type: torch.Tensor = None
    move: torch.Tensor = None
    max_move: torch.Tensor = None
    switch: torch.Tensor = None
    flag: torch.Tensor = None
    target: torch.Tensor = None

    @classmethod
    def from_list(self, outputs: List[Dict[str, torch.Tensor]]):
        buckets = {k: [] for k in self._fields}
        for output in outputs:
            for key in buckets:
                value = getattr(output, key, None)
                if value is not None:
                    buckets[key].append(value)
        for key, value in buckets.items():
            if buckets[key]:
                buckets[key] = torch.cat(buckets[key], dim=1)
            else:
                buckets[key] = None
        return self(**buckets)

    def to_json(self):
        return {
            k: v.squeeze().tolist() for k, v in self._asdict().items() if v is not None
        }

    def to(self, device):
        return Policy(
            **{k: v.to(device) for k, v in self._asdict().items() if v is not None}
        )

    def view(self, t, b):
        return Policy(
            **{
                k: v.view(t, b, *v.shape[2:])
                for k, v in self._asdict().items()
                if v is not None
            }
        )


class ModelOutput(NamedTuple):
    pi: Policy
    logit: Policy
    indices: Indices
    v: torch.Tensor = None
    log_pi: Policy = None

    state_emb: torch.Tensor = None
    state_action_emb: torch.Tensor = None

    @classmethod
    def from_list(self, outputs: List[Dict[str, torch.Tensor]]):
        buckets = {k: [] for k in self._fields}
        for output in outputs:
            for key in buckets:
                value = getattr(output, key, None)
                if value is not None:
                    buckets[key].append(value)
        buckets["pi"] = Policy.from_list(buckets["pi"])
        buckets["logit"] = Policy.from_list(buckets["logit"])
        buckets["log_pi"] = Policy.from_list(buckets["log_pi"])
        buckets["indices"] = Indices.from_list(buckets["indices"])
        buckets["v"] = torch.cat(buckets["v"], dim=1).unsqueeze(-1)
        return self(**buckets)

    def to_store(self, state: State) -> Dict[str, torch.Tensor]:
        to_store = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        to_store.update(
            {k: v.squeeze() for k, v in self.indices._asdict().items() if v is not None}
        )
        to_store.update(
            {
                k + "_policy": v.squeeze()
                for k, v in self.pi._asdict().items()
                if v is not None
            }
        )
        return to_store

    def to(self, device):
        return ModelOutput(
            **{k: v.to(device) for k, v in self._asdict().items() if v is not None}
        )

    def view(self, t, b):
        return ModelOutput(
            pi=self.pi.view(t, b),
            logit=self.logit.view(t, b),
            log_pi=self.log_pi.view(t, b),
            indices=self.indices.view(t, b),
            v=self.v.view(t, b, -1),
        )


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
    scalars: torch.Tensor = None
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
    value: torch.Tensor = None
    valid: torch.Tensor = None

    @property
    def batch_size(self):
        return self.valid.shape[1]

    @property
    def trajectory_lengths(self):
        return self.valid.sum(0)

    @property
    def trajectory_length(self):
        return self.valid.shape[0]

    def to(self, device: str, non_blocking: bool = False):
        batch = {
            k: t.to(
                device=device,
                non_blocking=non_blocking,
            )
            for k, t in self._asdict().items()
            if t is not None
        }
        return Batch(**batch)

    def flatten(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {
            k: v.flatten(*args, *kwargs).unsqueeze(0)
            for k, v in self._asdict().items()
            if v is not None
        }

    def flatten_without_padding(self, t: torch.Tensor):
        return {
            k: torch.cat([v[: t[i], i] for i in range(v.shape[1])]).unsqueeze(0)
            for k, v in self._asdict().items()
            if v is not None
        }


class Loss(NamedTuple):
    action_type_policy_loss: float = 0
    move_policy_loss: float = 0
    max_move_policy_loss: float = 0
    switch_policy_loss: float = 0
    flag_policy_loss: float = 0
    target_policy_loss: float = 0

    value_loss: float = 0
    repr_loss: float = 0


class Targets(NamedTuple):
    value_targets: List[torch.Tensor] = None
    has_played: List[torch.Tensor] = None

    action_type_policy_target: torch.Tensor = None
    move_policy_target: torch.Tensor = None
    switch_policy_target: torch.Tensor = None
    max_move_policy_target: torch.Tensor = None
    flag_policy_target: torch.Tensor = None
    target_policy_target: torch.Tensor = None

    action_type_is: torch.Tensor = None
    move_is: torch.Tensor = None
    switch_is: torch.Tensor = None
    max_move_is: torch.Tensor = None
    flag_is: torch.Tensor = None
    target_is: torch.Tensor = None

    @property
    def batch_size(self):
        return self.value_targets[0].shape[1]

    @property
    def trajectory_length(self):
        return self.value_targets[0].shape[0]

    def flatten(self, *args, **kwargs) -> Dict[str, List[torch.Tensor]]:
        return {
            k: [
                [v.flatten(*args, *kwargs).unsqueeze(0) for v in vi]
                if isinstance(vi, list)
                else vi.flatten(*args, *kwargs).unsqueeze(0)
                for vi in vl
            ]
            for k, vl in self._asdict().items()
            if vl is not None
        }

    def flatten_without_padding(self, t: torch.Tensor):
        return {
            k: [
                torch.cat([v[: t[i], i] for i in range(v.shape[1])]).unsqueeze(0)
                for v in vt
            ]
            for k, vt in self._asdict().items()
            if vt is not None
        }

    def to(self, device: str, non_blocking: bool = False):
        batch = {
            k: t.to(
                device=device,
                non_blocking=non_blocking,
            )
            for k, t in self._asdict().items()
            if t is not None
        }
