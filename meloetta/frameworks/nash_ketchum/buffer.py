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

        self.done_cache = [mp.Value("i", 0) for i in range(num_buffers)]
        self.turn_counters = [mp.Value("i", 0) for i in range(num_buffers)]

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
        return index

    def store_sample(
        self,
        battle_tag: str,
        step: Dict[str, torch.Tensor],
    ):
        index = self._get_index(battle_tag)
        with self.turn_counters[index].get_lock():
            turn = self.turn_counters[index].value
            for key, value in step.items():
                try:
                    self.buffers[key][index][turn][...] = value
                except Exception as e:
                    raise e
            self.buffers["valid"][index][turn][...] = 1
            self.turn_counters[index].value += 1

    def append_reward(self, battle_tag: str, pid: int, reward: int):
        index = self._get_index(battle_tag)
        turn = self.turn_counters[index].value - 1
        self.buffers["rewards"][index][turn, pid][...] = reward

    def register_done(self, battle_tag: str):
        index = self._get_index(battle_tag)
        with self.done_cache[index].get_lock():
            self.done_cache[index].value += 1
            if self.done_cache[index].value >= 2:
                length = sum(self.buffers["valid"][index]).item()
                self.finish_queue.put((None, length))
                self._clear_cache(battle_tag, index)

    def _clear_cache(self, battle_tag: str, index: int):
        self.turn_counters[index].value = 0
        self.full_queue.put(index)
        self.done_cache[index].value = 0
        self.index_cache.pop(battle_tag)

    def _reset_index(self, index: int):
        self.buffers["valid"][index][...] = 0
        self.buffers["rewards"][index][...] = 0
        for valid_mask in self.valid_masks:
            self.buffers[valid_mask][index][...] = 1

    def get_batch(
        self,
        batch_size: int,
        lock=threading.Lock(),
    ) -> Batch:
        with lock:
            indices = [self.full_queue.get() for _ in range(batch_size)]

        valids = torch.stack([self.buffers["valid"][m] for m in indices])
        lengths = valids.sum(-1)

        if torch.any(lengths == 0):
            return None

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
            self.free_queue.put(m)

        batch = {
            k: t.to(device=self.device, non_blocking=True) for k, t in batch.items()
        }
        return Batch(**batch)
