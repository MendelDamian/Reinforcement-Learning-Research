from typing import Dict
import random

from .agent import Agent
from ..types import State, Action, Reward, Done


class QAgent(Agent):
    def __init__(self, alpha: float, gamma: float, epsilon: float) -> None:
        self._alpha = alpha  # learning rate
        self._gamma = gamma  # discount factor
        self._epsilon = epsilon  # exploration rate

        self._Q: Dict[State, Dict[Action, float]] = dict()  # Q-table

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: Done) -> None:
        if done or next_state not in self._Q:
            self._Q[state][action] += self._alpha * (reward - self._Q[state][action])
        else:
            self._Q[state][action] += self._alpha * (
                reward + self._gamma * max(self._Q[next_state].values()) - self._Q[state][action]
            )

    def get_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        self._populate_q_table(state, actions)

        if self._epsilon > 0:
            if random.random() < self._epsilon:
                return random.choice(actions)

        return self.get_best_action(state, actions)

    def get_best_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        self._populate_q_table(state, actions)

        return max(self._Q[state], key=self._Q[state].get)

    def _populate_q_table(self, state: State, actions: tuple[Action, ...]) -> None:
        if state not in self._Q:
            self._Q[state] = {action: 0 for action in actions}
