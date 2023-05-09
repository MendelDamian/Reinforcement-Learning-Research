from abc import ABC, abstractmethod

from rl_tools.types import State, Action, Reward, Done


class Environment(ABC):
    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def step(self, action: Action) -> tuple[State, Reward, Done]:
        pass

    @abstractmethod
    def get_actions(self, state: State) -> tuple[Action, ...]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass
