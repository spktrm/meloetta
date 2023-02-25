import random
import multiprocessing as mp

from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices


class RandomActor(Actor):
    def __init__(self, queue: mp.Queue = None):
        self.queue = queue

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        random_key = random.choice([key for key, value in choices.items() if value])
        _, (func, args, kwargs) = random.choice(list(choices[random_key].items()))
        return func, args, kwargs

    def post_match(self, room: BattleRoom):
        if self.queue is not None:
            datum = room.get_reward()
            reward = datum["reward"]

            if reward > 0:
                self.queue.put(("random", 1))
            elif reward < 0:
                self.queue.put(("random", 0))
