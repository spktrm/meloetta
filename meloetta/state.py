from typing import NamedTuple


class Side(NamedTuple):
    pass


class VectorizedState:
    def __init__(self, battle, state):
        self.battle = battle
        self.state = state

        self.vectorize()

    def vectorize(self):
        self._vectorize_sides()

    def _vectorize_sides(self):
        self._vectorize_side("mySide")
        self._vectorize_side("farSide")

    def _vectorize_side(self, side_id: str):
        side = self.state[side_id]
        return
