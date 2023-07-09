import torch
import random
import threading
import multiprocessing as mp

from typing import Dict, List

from meloetta.types import TensorDict
from meloetta.frameworks.nash_ketchum.utils import create_buffers


Buffers = Dict[str, List[torch.Tensor]]


class ReplayBuffer:
    def __init__(
        self,
        trajectory_length: int,
        gen: int,
        gametype: str,
        num_buffers: int = 512,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.trajectory_length = trajectory_length
        self.num_buffers = num_buffers
        self.device = device
        self.finish_queue = mp.Queue()

        manager = mp.Manager()
        self.index_cache = manager.dict()

        self.buffers = create_buffers(num_buffers, trajectory_length, gen, gametype)

        self.valid_masks = [field for field in self.buffers if field.endswith("_mask")]

        self.done_cache = [[0 for _ in range(num_buffers)] for _ in range(2)]

        self.full_queue = mp.Queue()
        self.free_queue = mp.Queue()

        for m in range(num_buffers):
            self.free_queue.put(m)

    def _get_index(self, battle_tag: str, pid: int = None) -> int:
        if battle_tag in self.index_cache:
            index = self.index_cache[battle_tag]
        else:
            index = self.free_queue.get()
            self._reset_index(index)
            self.index_cache[battle_tag] = index
            assert torch.all(~self.buffers["valid"][pid][index]).item()
        return index

    def store_sample(self, index: int, turn: int, pid: int, step: TensorDict):
        assert (~self.buffers["valid"][pid][index][turn]).item()
        for key, value in step.items():
            try:
                self.buffers[key][pid][index][turn][...] = value
            except Exception as e:
                raise e
        self.buffers["valid"][pid][index][turn][...] = 1

    def store_trajectory(self, index: int, pid: int, step: TensorDict):
        trajectory_length = step["utc"].shape[0]
        assert torch.all(~self.buffers["valid"][pid][index]).item()
        for key, values in step.items():
            try:
                self.buffers[key][pid][index][:trajectory_length][...] = values
            except Exception as e:
                raise e
        self.buffers["valid"][pid][index][:trajectory_length][...] = 1

    def append_reward(self, index: int, turn: int, pid: int, reward: int):
        self.buffers["rewards"][pid][index][turn][...] = reward

    def register_done(self, battle_tag: str, pid: int):
        index = self._get_index(battle_tag)
        self.done_cache[pid][index] = 1
        if sum([self.done_cache[k][index] for k in range(2)]) == 2:
            t_valid = torch.stack([self.buffers["valid"][k][index] for k in range(2)])
            length = t_valid.sum().item()
            self.finish_queue.put((None, length))
            self._clear_cache(battle_tag, index)

    def _clear_cache(self, battle_tag: str, index: int):
        self.full_queue.put(index)
        for k in range(2):
            self.done_cache[k][index] = 0
        self.index_cache.pop(battle_tag)

    def _reset_index(self, index: int):
        for pid in range(2):
            self.buffers["valid"][pid][index][...] = 0
            self.buffers["rewards"][pid][index][...] = 0
            self.buffers["scalars"][pid][index][...] = 0
            self.buffers["utc"][pid][index][...] = float("inf")
            for valid_mask in self.valid_masks:
                self.buffers[valid_mask][pid][index][...] = 1

    def get_batch(self, batch_size: int, lock=threading.Lock()) -> TensorDict:
        def _get_index():
            index = self.full_queue.get()
            return index

        with lock:
            indices = [_get_index() for _ in range(batch_size)]

        order1 = torch.stack(
            [
                torch.cat([self.buffers["utc"][k][index] for k in range(2)])
                for index in indices
            ],
            dim=1,
        ).argsort(0)
        orders1 = torch.chunk(order1, batch_size, 1)
        orders1 = [order.squeeze(-1) for order in orders1]

        max_len = (
            torch.stack(
                [
                    torch.cat([self.buffers["valid"][k][index] for k in range(2)])
                    for index in indices
                ],
                dim=1,
            )
            .sum(0)
            .max()
            .item()
        )

        batch = {
            key: torch.stack(
                [
                    torch.cat([self.buffers[key][k][batch_index] for k in range(2)])[
                        orders1[order_index]
                    ][:max_len]
                    for order_index, batch_index in enumerate(indices)
                ],
                dim=1,
            )
            for key in self.buffers
            if key != "rewards"
        }

        assert torch.all(batch["utc"][:-1] <= batch["utc"][1:]).item()
        batch.pop("utc")

        rewards = torch.zeros(max_len, batch_size, 2)
        final_reward = torch.stack(
            [
                torch.stack(
                    [self.buffers["rewards"][k][index] for k in range(2)], dim=-1
                )
                for index in indices
            ],
            dim=1,
        )
        final_reward = final_reward.sum(0)
        final_idx = batch["valid"].sum(0) - 1
        rewards[final_idx, torch.arange(batch_size)] = final_reward
        batch["rewards"] = rewards

        player_id = torch.zeros(
            2 * self.trajectory_length, batch_size, dtype=torch.long
        )
        player_id[self.trajectory_length :] = 1
        player_id = torch.stack(
            [
                ids.squeeze(-1)[orders1[index]][:max_len]
                for index, ids in enumerate(player_id.chunk(batch_size, 1))
            ],
            dim=1,
        )
        batch["player_id"] = player_id

        for m in indices:
            # if random.random() < 2 / 3:
            #     self.full_queue.put(m)
            # else:
            self.free_queue.put(m)

        return indices, batch
