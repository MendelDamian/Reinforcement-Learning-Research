import numpy as np

from rl_tools.env import Environment
from rl_tools.types import State, Action, Reward, Done


class Map2DStatic(Environment):
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, shape: tuple[int, int], start: tuple[int, int], goal: tuple[int, int], iterations: int) -> None:
        self._shape = shape
        self._start = start
        self._goal = goal
        self._iterations = iterations
        self._iteration = 0

        self._actions = (self.UP, self.DOWN, self.LEFT, self.RIGHT)
        self._state: tuple[int, int] = self.reset()

        self._rewards = np.full(shape, -0.1)
        self._rewards[self._goal] = 1

        self._pygame = {
            "init": False,
        }

    def reset(self) -> tuple[int, int]:
        self._state = self._start
        self._iteration = 0
        return self._start

    def step(self, action: Action) -> tuple[State, Reward, Done]:
        assert action in self._actions

        x, y = self._state
        if action == self.UP:
            y -= 1
        elif action == self.DOWN:
            y += 1
        elif action == self.LEFT:
            x -= 1
        elif action == self.RIGHT:
            x += 1

        self._iteration += 1
        self._state = self._clamp(x, y)
        done = self._state == self._goal or self._iteration >= self._iterations

        return self._state, self._rewards[self._state], done

    def get_actions(self, state: State) -> tuple[Action, ...]:
        return self._actions

    def render(self) -> None:
        import pygame

        if not self._pygame["init"]:
            pygame.init()
            pygame.display.set_caption("Playground")
            self._pygame["screen"] = pygame.display.set_mode((self._shape[0] * 50, self._shape[1] * 50))
            self._pygame["clock"] = pygame.time.Clock()
            self._pygame["init"] = True

        self._pygame["screen"].fill((0, 0, 0))

        for y in range(self._shape[0]):
            for x in range(self._shape[1]):
                color = (255, 255, 255)
                if (x, y) == self._start:
                    color = (0, 255, 0)
                elif (x, y) == self._goal:
                    color = (0, 0, 255)
                elif (x, y) == self._state:
                    color = (255, 0, 0)

                pygame.draw.rect(self._pygame["screen"], color, (x * 50, y * 50, 50, 50))

        pygame.display.flip()
        self._pygame["clock"].tick(15)

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        return max(0, min(x, self._shape[0] - 1)), max(0, min(y, self._shape[1] - 1))
