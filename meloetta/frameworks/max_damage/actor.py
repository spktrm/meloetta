import torch
import multiprocessing as mp

from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices

from meloetta.frameworks.max_damage.model import MaxDamageModel


class MaxDamageActor(Actor):
    def __init__(self, gen: int, queue: mp.Queue):
        self._model = MaxDamageModel(gen=gen)
        self._model.eval()

        self._queue = queue

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
        **kwargs,
    ):
        with torch.no_grad():
            func, args, kwargs = self._model(state, choices)

        return func, args, kwargs, None

    def store_reward(
        self,
        room: BattleRoom,
        pid: int,
        reward: float = None,
        store_transition: bool = True,
    ):
        if reward > 0:
            self._queue.put(("maxdmg", 1))
        elif reward < 0:
            self._queue.put(("maxdmg", 0))
