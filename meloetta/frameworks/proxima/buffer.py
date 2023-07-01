import torch
import random
import threading
import multiprocessing as mp

from typing import Dict, List

from meloetta.actors.types import TensorDict
from meloetta.frameworks.proxima.utils import create_buffers


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
        return index

    def store_sample(
        self,
        index: int,
        turn: int,
        step: Dict[str, torch.Tensor],
    ):
        assert (~self.buffers["valid"][index][turn]).item()
        for key, value in step.items():
            try:
                self.buffers[key][index][turn][...] = value
            except Exception as e:
                raise e
        self.buffers["valid"][index][turn][...] = 1

    def append_reward(self, index: int, turn: int, reward: int):
        self.buffers["rewards"][index][turn][...] = reward

    def clear_cache(self, battle_tag: str, index: int):
        self.index_cache.pop(battle_tag)
        length = sum(self.buffers["valid"][index]).item()
        self.finish_queue.put((None, length))
        self.full_queue.put(index)

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
    ) -> TensorDict:
        def _get_index():
            index = self.full_queue.get()
            return index

        with lock:
            indices = [_get_index() for _ in range(batch_size)]

        valids = torch.stack([self.buffers["valid"][m] for m in indices])
        lengths = valids.sum(-1)  # + 1
        max_length = lengths.max().item()

        def _check_and_return(buffers, key, index, max_length: int = None):
            end = buffers["valid"][index].sum(-1)
            assert buffers["valid"][index][:end].sum() == end
            assert buffers["valid"][index][end:].sum() == 0
            return self.buffers[key][index][:max_length]

        batch = {
            key: torch.stack(
                [
                    _check_and_return(self.buffers, key, index, max_length)
                    for index in indices
                ],
                dim=1,
            )
            for key in self.buffers
        }

        for m in indices:
            if random.random() < 0.5:
                self.free_queue.put(m)
            else:
                self.full_queue.put(m)

        return indices, batch
