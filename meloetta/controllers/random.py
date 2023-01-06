import random

from meloetta.room import BattleRoom
from meloetta.controllers.base import Controller
from meloetta.controllers.types import State, Choices


class RandomController(Controller):
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
