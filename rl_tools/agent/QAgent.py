from typing import Dict
import random

from .agent import Agent
from rl_tools.types import State, Action, Reward, Done


class QAgent(Agent):
    """
    A class representing a Q-learning agent, which uses a Q-table to learn the optimal value function for a given environment.

    Attributes:
        _alpha (float): The learning rate, which determines the degree to which newly acquired information overrides old information.
        _gamma (float): The discount factor, which determines the importance of future rewards relative to immediate rewards.
        _epsilon (float): The exploration rate, which controls the degree to which the agent explores the environment rather than exploiting its current knowledge.
        _Q (Dict[State, Dict[Action, float]]): The Q-table, which maps states to dictionaries that map actions to their corresponding Q-values.

    The Q-value of a state-action pair is updated using the following formula:

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

    where:
        Q(s,a) is the current Q-value for state s and action a,
        alpha is the learning rate,
        r is the reward received for taking action a in state s,
        gamma is the discount factor,
        max(Q(s',a')) is the maximum Q-value for the next state s' and all possible actions a',
        Q(s',a') is the Q-value for the next state s' and action a'
    """

    def __init__(self, alpha: float, gamma: float, epsilon: float) -> None:
        """
        Initializes a new instance of the QAgent class.

        Args:
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            epsilon (float): The exploration rate.
        """

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self._Q: Dict[State, Dict[Action, float]] = dict()

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: Done) -> None:
        """
        Updates the Q-table based on the given transition.

        Args:
            state (State): The current state.
            action (Action): The action taken in the current state.
            reward (Reward): The reward received for taking the action in the current state.
            next_state (State): The resulting state after taking the action in the current state.
            done (Done): A flag indicating whether the episode has ended.
        """

        if done or next_state not in self._Q:
            self._Q[state][action] += self._alpha * (reward - self._Q[state][action])
        else:
            self._Q[state][action] += self._alpha * (
                reward + self._gamma * max(self._Q[next_state].values()) - self._Q[state][action]
            )

    def get_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        """
        Gets the action to take in the given state.

        Args:
            state (State): The current state.
            actions (tuple[Action, ...]): The available actions in the current state.

        Returns:
            Action: The action to take in the current state.
        """

        self._populate_q_table(state, actions)

        if self._epsilon > 0:
            if random.random() < self._epsilon:
                return random.choice(actions)

        return self.get_best_action(state, actions)

    def get_best_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        """
        Gets the best action to take in the given state using greedy policy.

        Args:
            state (State): The current state.
            actions (tuple[Action, ...]): The available actions in the current state.

        Returns:
            Action: The best action to take in the current state.
        """

        self._populate_q_table(state, actions)

        return max(self._Q[state], key=self._Q[state].get)

    def _populate_q_table(self, state: State, actions: tuple[Action, ...]) -> None:
        """
        Populates the Q-table with initial values for the given state and actions.

        Args:
            state (State): The state to populate the Q-table for.
            actions (tuple[Action, ...]): The actions to populate the Q-table for.
        """

        if state not in self._Q:
            self._Q[state] = {action: 0 for action in actions}
