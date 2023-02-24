import traceback

from abc import ABC, abstractmethod

from typing import Any


class Actor(ABC):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.choose_action(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError

    def store_reward(
        self, room, pid, reward: float = None, store_transition: bool = True
    ):
        pass
