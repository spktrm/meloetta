import torch

from typing import Dict, List

from meloetta.data import CHOICE_FLAGS


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
        "private_reserve": {
            "size": (trajectory_length, 6, private_reserve_size),
            "dtype": torch.int64,
        },
        "public_n": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_total_pokemon": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_faint_counter": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_side_conditions": {
            "size": (trajectory_length, 2, 18, 3),
            "dtype": torch.int64,
        },
        "public_wisher": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_active": {
            "size": (trajectory_length, 2, n_active, 158),
            "dtype": torch.int64,
        },
        "public_reserve": {
            "size": (trajectory_length, 2, 6, 37),
            "dtype": torch.int64,
        },
        "public_stealthrock": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_spikes": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_toxicspikes": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "public_stickyweb": {
            "size": (trajectory_length, 2),
            "dtype": torch.int64,
        },
        "weather": {
            "size": (trajectory_length,),
            "dtype": torch.int64,
        },
        "weather_time_left": {
            "size": (trajectory_length,),
            "dtype": torch.int64,
        },
        "weather_min_time_left": {
            "size": (trajectory_length,),
            "dtype": torch.int64,
        },
        "pseudo_weather": {
            "size": (trajectory_length, 12, 2),
            "dtype": torch.int64,
        },
        "turn": {
            "size": (trajectory_length,),
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
            "size": (trajectory_length, 2),
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
            "action_type_policy": {
                "size": (trajectory_length, 3),
                "dtype": torch.float32,
            },
            "move_policy": {
                "size": (trajectory_length, 4),
                "dtype": torch.float32,
            },
            "switch_policy": {
                "size": (trajectory_length, 6),
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
            "action_type_index": {
                "size": (trajectory_length,),
                "dtype": torch.int64,
            },
            "move_index": {
                "size": (trajectory_length,),
                "dtype": torch.int64,
            },
            "switch_index": {
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
            }
        }
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
