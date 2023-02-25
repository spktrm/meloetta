import torch
import multiprocessing as mp

from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices

from meloetta.frameworks.max_damage.model import MaxDamageModel


class MaxDamageActor(Actor):
    def __init__(self, gen: int = 9, queue: mp.Queue = None):
        self.model = MaxDamageModel(gen=gen)
        self.model.eval()

        self.queue = queue

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        with torch.no_grad():
            func, args, kwargs = self.model(state, choices)

        return func, args, kwargs

    def post_match(self, room: BattleRoom):
        if self.queue is not None:
            datum = room.get_reward()
            reward = datum["reward"]

            if reward > 0:
                self.queue.put(("maxdmg", 1))
            elif reward < 0:
                self.queue.put(("maxdmg", 0))
