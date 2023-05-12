from abc import ABC, abstractmethod

from rl_tools.types import State, Action, Reward, Done


class Agent(ABC):
    @abstractmethod
    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: Done) -> None:
        pass

    @abstractmethod
    def get_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        pass
