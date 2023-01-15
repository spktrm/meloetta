from abc import ABC, abstractmethod

from typing import Any


class Actor(ABC):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.choose_action(*args, **kwds)

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError

    def store_reward(self, reward: float = None):
        pass
