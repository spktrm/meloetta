import torch

from typing import Dict, List


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


def expand_bt(tensor: torch.Tensor, time: int = 1, batch: int = 1) -> torch.Tensor:
    shape = tensor.shape
    if not shape:
        shape = (1,)
    return tensor.view(time, batch, *(tensor.shape or shape))


def create_buffers(num_buffers: int, trajectory_length: int, battle_format: str):
    """
    num_buffers: int
        the size of the replay buffer
    trajectory_length: int
        the max length of a replay
    battle_format: str
        the type of format
    """

    buffer_specs = {
        "private_reserve": {
            "shape": (trajectory_length, 6, 36),
            "dtype": torch.int64,
        },
        "public_n": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_total_pokemon": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_faint_counter": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_side_conditions": {
            "shape": (trajectory_length, 2, 18, 3),
            "dtype": torch.int64,
        },
        "public_wisher": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_active": {
            "shape": (trajectory_length, 2, 2, 156),
            "dtype": torch.int64,
        },
        "public_reserve": {
            "shape": (trajectory_length, 2, 4, 33),
            "dtype": torch.int64,
        },
        "public_stealthrock": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_spikes": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_toxicspikes": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "public_stickyweb": {
            "shape": (trajectory_length, 2, 1),
            "dtype": torch.int64,
        },
        "weather": {
            "shape": (trajectory_length, 1),
            "dtype": torch.int64,
        },
        "weather_time_left": {
            "shape": (trajectory_length, 1),
            "dtype": torch.int64,
        },
        "weather_min_time_left": {
            "shape": (trajectory_length, 1),
            "dtype": torch.int64,
        },
        "pseudo_weather": {
            "shape": (trajectory_length, 12, 2),
            "dtype": torch.int64,
        },
        "turn": {
            "shape": (trajectory_length, 1),
            "dtype": torch.int64,
        },
        "action_type_mask": {
            "shape": (trajectory_length, 2),
            "dtype": torch.bool,
        },
        "moves_mask": {
            "shape": (trajectory_length, 4),
            "dtype": torch.bool,
        },
        "switches_mask": {
            "shape": (trajectory_length, 6),
            "dtype": torch.bool,
        },
        "flags_mask": {
            "shape": (trajectory_length, 4),
            "dtype": torch.bool,
        },
    }
    if battle_format != "singles":
        buffer_specs.update(
            {
                "targets_mask": {
                    "shape": (trajectory_length, 4),
                    "dtype": torch.bool,
                },
                "prev_choices": {
                    "shape": (trajectory_length, 2, 4),
                    "dtype": torch.int64,
                },
                "choices_done": {
                    "shape": (trajectory_length, 1),
                    "dtype": torch.int64,
                },
                "targeting": {
                    "shape": (trajectory_length, 1),
                    "dtype": torch.int64,
                },
            }
        )
    buffers: Dict[str, List[torch.Tensor]] = {key: [] for key in buffer_specs}
    for _ in range(num_buffers):
        for key in buffer_specs:
            buffers[key].append(torch.empty(**buffer_specs[key]).share_memory_())
    return buffers
