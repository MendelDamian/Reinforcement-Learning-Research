import random
from typing import Optional
import numpy as np

from rl_tools.env import Map2DStatic
from rl_tools.types import Action, State, Reward, Done


class Map2DDynamic(Map2DStatic):

    def __init__(self, shape: tuple[int, int], iterations: int) -> None:
        #TODO: create Map2D class with common stuff

        self._shape = shape
        self._state = self._get_random_point()
        self._goal = self._get_free_point()

        super().__init__(shape, self._state, self._goal, iterations)

    def reset(self) -> State:
        self._iteration = 0
        self._goal = self._get_free_point()

        self._rewards = np.full(self._shape, -0.1)
        self._rewards[self._goal] = 1

        return self._get_state()

    def step(self, action: Action) -> tuple[State, Reward, Done]:
        state, reward, done = super(Map2DDynamic, self).step(action)
        done = self._iteration >= self._iterations
        return state, reward, done

    def _get_random_point(self) -> tuple[int, int]:
        return random.randint(0, self._shape[0] - 1), random.randint(0, self._shape[1] - 1)

    def _get_free_point(self) -> Optional[tuple[int, int]]:
        free_point = (0, 0)

        while free_point != self._state:
            free_point = self._get_random_point()

        return free_point

    def _get_state(self) -> State:
        return self._state, self._goal
