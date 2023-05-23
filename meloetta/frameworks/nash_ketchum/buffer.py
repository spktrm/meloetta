import torch
import random
import threading
import multiprocessing as mp

from typing import Dict, List

from meloetta.frameworks.nash_ketchum.model.interfaces import Batch
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
        self.num_buffers = num_buffers
        self.device = device
        self.finish_queue = mp.Queue()

        manager = mp.Manager()
        self.index_cache = manager.dict()

        self.buffers = create_buffers(num_buffers, trajectory_length, gen, gametype)

        self.valid_masks = [field for field in self.buffers if field.endswith("_mask")]

        self.done_cache = [0 for _ in range(num_buffers)]
        self.turn_counters = [0 for _ in range(num_buffers)]

        self.full_queue = mp.Queue()
        self.free_queue = mp.Queue()

        for m in range(num_buffers):
            self.free_queue.put(m)

    def _get_index(self, battle_tag: str) -> int:
        if battle_tag in self.index_cache:
            index = self.index_cache[battle_tag]
        else:
            index = self.free_queue.get()
            self._reset_index(index)
            self.index_cache[battle_tag] = index
            assert torch.all(~self.buffers["valid"][index]).item()
            assert self.turn_counters[index] == 0
        return index

    def store_sample(
        self,
        index: int,
        step: Dict[str, torch.Tensor],
        pid: int,
    ):
        turn = self.turn_counters[index]
        assert (~self.buffers["valid"][index][turn]).item()
        for key, value in step.items():
            try:
                self.buffers[key][index][turn][...] = value
            except Exception as e:
                raise e
        self.buffers["player_id"][index][turn][...] = pid
        self.buffers["valid"][index][turn][...] = 1
        self.turn_counters[index] += 1

    def append_reward(self, index: int, pid: int, reward: int):
        turn = self.turn_counters[index] - 1
        self.buffers["rewards"][index][turn, pid][...] = reward
        # self.buffers["valid"][index][turn][...] = 0

    def register_done(self, battle_tag: str):
        index = self._get_index(battle_tag)
        self.done_cache[index] += 1
        if self.done_cache[index] == 2:
            length = sum(self.buffers["valid"][index]).item()
            self.finish_queue.put((None, length))
            self._clear_cache(battle_tag, index)

    def _clear_cache(self, battle_tag: str, index: int):
        self.turn_counters[index] = 0
        self.full_queue.put(index)
        self.done_cache[index] = 0
        self.index_cache.pop(battle_tag)

    def _reset_index(self, index: int):
        self.buffers["valid"][index][...] = 0
        self.buffers["rewards"][index][...] = 0
        self.buffers["scalars"][index][...] = 0
        for valid_mask in self.valid_masks:
            self.buffers[valid_mask][index][...] = 1

    def get_batch(
        self,
        batch_size: int,
        lock=threading.Lock(),
    ) -> Batch:
        def _get_index():
            index = self.full_queue.get()
            return index

        with lock:
            indices = [_get_index() for _ in range(batch_size)]

        # for m in indices:
        #     end = self.buffers["valid"][m].sum(-1)
        #     assert self.buffers["valid"][m][:end].sum() == end
        #     assert self.buffers["valid"][m][end:].sum() == 0

        rewards = torch.stack([self.buffers["rewards"][m] for m in indices])
        if not (torch.sum(rewards, dim=-1) == 0).all().item():
            print("Batch rewards are not zero-sum!")

        valids = torch.stack([self.buffers["valid"][m] for m in indices])
        lengths = valids.sum(-1)  # + 1

        max_length = lengths.max().item()

        indices = list(
            zip(
                *sorted(
                    list(zip(lengths.tolist(), indices)),
                    key=lambda x: x[0],
                    reverse=True,
                )
            )
        )[1]

        batch = {
            key: torch.stack(
                [self.buffers[key][index][:max_length] for index in indices],
                dim=1,
            )
            for key in self.buffers
        }

        for m in indices:
            # if random.random() < 0.75:
            #     self.full_queue.put(m)
            # else:
            self.free_queue.put(m)

        return Batch(**batch)
