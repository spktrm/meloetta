from abc import ABC, abstractmethod

from typing import Any


class Controller(ABC):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.choose_action(*args, **kwds)

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError
