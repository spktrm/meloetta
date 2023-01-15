import random

from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices


class RandomActor(Actor):
    def __init__(self):
        pass

    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        _, (func, args, kwargs) = random.choice(choices.items())
        return func, args, kwargs
