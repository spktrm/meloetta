import re
import torch
import collections
import torch.nn.functional as F

from typing import Dict, List, Tuple

from meloetta.frameworks.porygon.model.utils import _log_policy, _legal_policy
from meloetta.data import CHOICE_FLAGS


def get_gen_and_gametype(battle_format: str) -> Tuple[int, str]:
    gen = int(re.search(r"gen([0-9])", battle_format).groups()[0])
    if "triples" in battle_format:
        gametype = "triples"
    if "doubles" in battle_format:
        gametype = "doubles"
    else:
        gametype = "singles"
    return gen, gametype


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


def _get_private_reserve_size(gen: int):
    if gen == 9:
        return 28
    elif gen == 8:
        return 25
    else:
        return 24


def get_buffer_specs(
    trajectory_length: int,
    gen: int,
    gametype: str,
    private_reserve_size: int,
):
    if gametype != "singles":
        if gametype == "doubles":
            n_active = 2
        else:
            n_active = 3
    else:
        n_active = 1

    buffer_specs = {
        "sides": {
            "size": (trajectory_length, 3, 12, 32),
            "dtype": torch.int64,
        },
        "boosts": {
            "size": (trajectory_length, 2, 8),
            "dtype": torch.int64,
        },
        "volatiles": {
            "size": (trajectory_length, 2, 113),
            "dtype": torch.int64,
        },
        "side_conditions": {
            "size": (trajectory_length, 2, 58),
            "dtype": torch.int64,
        },
        "pseudoweathers": {
            "size": (trajectory_length, 12, 2),
            "dtype": torch.int64,
        },
        "weather": {
            "size": (trajectory_length, 3),
            "dtype": torch.int64,
        },
        "wisher": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "turn": {
            "size": (trajectory_length,),
            "dtype": torch.int64,
        },
        "n": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "total_pokemon": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "faint_counter": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "turns_since_last_move": {
            "size": (trajectory_length,),
            "dtype": torch.int64,
        },
        "action_type_mask": {
            "size": (trajectory_length, 3),
            "dtype": torch.bool,
        },
        "move_mask": {
            "size": (trajectory_length, 4),
            "dtype": torch.bool,
        },
        "switch_mask": {
            "size": (trajectory_length, 6),
            "dtype": torch.bool,
        },
        "flag_mask": {
            "size": (trajectory_length, len(CHOICE_FLAGS)),
            "dtype": torch.bool,
        },
        "rewards": {
            "size": (trajectory_length,),
            "dtype": torch.float32,
        },
        "player_id": {
            "size": (trajectory_length,),
            "dtype": torch.long,
        },
    }
    if gametype != "singles":
        buffer_specs.update(
            {
                "target_mask": {
                    "size": (trajectory_length, 4),
                    "dtype": torch.bool,
                },
                "prev_choices": {
                    "size": (trajectory_length, 2, 4),
                    "dtype": torch.int64,
                },
                "choices_done": {
                    "size": (trajectory_length, 1),
                    "dtype": torch.int64,
                },
                "targeting": {
                    "size": (trajectory_length, 1),
                    "dtype": torch.int64,
                },
                "target_policy": {
                    "size": (trajectory_length, 2 * n_active),
                    "dtype": torch.float32,
                },
                "target_index": {
                    "size": (trajectory_length,),
                    "dtype": torch.int64,
                },
            }
        )

    # add policy...
    buffer_specs.update(
        {
            "action_policy": {
                "size": (trajectory_length, 10),
                "dtype": torch.float32,
            },
            "flag_policy": {
                "size": (trajectory_length, len(CHOICE_FLAGS)),
                "dtype": torch.float32,
            },
        }
    )

    # ...and indices
    buffer_specs.update(
        {
            "action_index": {
                "size": (trajectory_length,),
                "dtype": torch.int64,
            },
            "flag_index": {
                "size": (trajectory_length,),
                "dtype": torch.int64,
            },
        }
    )

    if gen == 8:
        buffer_specs.update(
            {
                "max_move_mask": {
                    "size": (trajectory_length, 4),
                    "dtype": torch.bool,
                },
                "max_move_policy": {
                    "size": (trajectory_length, 4),
                    "dtype": torch.float32,
                },
                "max_move_index": {
                    "size": (trajectory_length,),
                    "dtype": torch.int64,
                },
            }
        )

    buffer_specs.update(
        {
            "valid": {
                "size": (trajectory_length,),
                "dtype": torch.bool,
            },
            "value": {
                "size": (trajectory_length,),
                "dtype": torch.float32,
            },
        },
    )
    return buffer_specs


def create_buffers(num_buffers: int, trajectory_length: int, gen: int, gametype: str):
    """
    num_buffers: int
        the size of the replay buffer
    trajectory_length: int
        the max length of a replay
    battle_format: str
        the type of format
    """

    private_reserve_size = _get_private_reserve_size(gen)

    buffer_specs = get_buffer_specs(
        trajectory_length, gen, gametype, private_reserve_size
    )

    buffers: Dict[str, List[torch.Tensor]] = {key: [] for key in buffer_specs}
    for _ in range(num_buffers):
        for key in buffer_specs:
            if key.endswith("_mask"):
                buffers[key].append(torch.ones(**buffer_specs[key]).share_memory_())
            else:
                buffers[key].append(torch.zeros(**buffer_specs[key]).share_memory_())
    return buffers


def expand_bt(tensor: torch.Tensor, time: int = 1, batch: int = 1) -> torch.Tensor:
    shape = tensor.shape
    return tensor.view(time, batch, *(tensor.shape or shape))


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


def action_log_probs(policy_logits, actions, policy_mask):
    return torch.gather(
        _log_policy(policy_logits, policy_mask), -1, actions.unsqueeze(-1)
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    policy_mask,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(
        target_policy_logits, actions, policy_mask
    )
    behavior_action_log_probs = action_log_probs(
        behavior_policy_logits, actions, policy_mask
    )
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )

    target_dist = torch.distributions.Categorical(
        probs=_legal_policy(target_policy_logits, policy_mask)
    )
    behavior_dist = torch.distributions.Categorical(
        probs=_legal_policy(behavior_policy_logits, policy_mask)
    )
    kl_loss = torch.distributions.kl.kl_divergence(target_dist, behavior_dist)

    return VTraceFromLogitsReturns(
        kl_loss=kl_loss,
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
