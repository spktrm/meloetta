import torch
import random
import multiprocessing as mp

from typing import Dict, List

from meloetta.utils import create_buffers


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

        self.lock = mp.Lock()

        self.full_queue = manager.list()
        self.free_queue = manager.list()

        for m in range(num_buffers):
            self.free_queue.append(m)

    def _get_index(self, battle_tag: str) -> int:
        if battle_tag in self.index_cache:
            index = self.index_cache[battle_tag]
        else:
            try:
                index = self.free_queue.pop(0)
            except:
                with self.lock:
                    while True:
                        try:
                            index = self.full_queue.pop(0)
                        except:
                            pass
                        else:
                            self.reset_index(index)
                            break
            self.index_cache[battle_tag] = index
        return index

    def store_sample(self, battle_tag: str, step: Dict[str, torch.Tensor]):
        index = self._get_index(battle_tag)
        turn = self.turn_counters[index].value
        for key, value in step.items():
            try:
                self.buffers[key][index][turn][...] = value
            except Exception as e:
                raise e
        self.buffers["valid"][index][turn][...] = 1
        self.buffers["rewards"][index][turn][...] = 0
        with self.turn_counters[index].get_lock():
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
            self.full_queue.append(index)
            self.index_cache.pop(battle_tag)
            self.done_cache[index].value = 0

    def reset_index(self, index: int):
        self.turn_counters[index].value = 0
        self.buffers["valid"][index][...] = 0
        for valid_mask in self.valid_masks:
            self.buffers[valid_mask][index][...] = 1

    def get_batch(self, batch_size: int):
        with self.lock:
            while True:
                if len(self.full_queue) >= batch_size:
                    break

            indices = random.sample(list(self.full_queue), k=batch_size)
            valids = torch.stack([self.buffers["valid"][m] for m in indices])
            lengths = valids.sum(-1)
            max_index = lengths.max().item()

            indices = list(
                zip(*sorted(list(zip(lengths.tolist(), indices)), key=lambda x: x[0]))
            )[1]

            batch = {
                key: torch.stack(
                    [self.buffers[key][index][:max_index] for index in indices],
                    dim=1,
                )
                for key in self.buffers
            }

        batch = {
            k: t.to(device=self.device, non_blocking=True) for k, t in batch.items()
        }
        return batch
