from typing import Dict
import math
import random

from rl_tools.agent import QAgent
from rl_tools.types import State, Action, Reward, Done


class UCBQAgent(QAgent):
    """
    A class representing a Q-learning agent, which uses a Q-table to learn the optimal value function for a given environment.

    Attributes:
        _alpha (float): The learning rate, which determines the degree to which newly acquired information overrides old information.
        _gamma (float): The discount factor, which determines the importance of future rewards relative to immediate rewards.
        _c (float): The exploration rate, which controls the degree to which the agent explores the environment rather than exploiting its current knowledge.
        _Q (Dict[State, Dict[Action, float]]): The Q-table, which maps states to dictionaries that map actions to their corresponding Q-values.
        _N (Dict[State, Dict[Action, int]]): The visitation table that map actions to the number of times they have been taken.

    The Q-value of a state-action pair is updated using the following formula:

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

    where:
        Q(s,a) is the current Q-value for state s and action a,
        alpha is the learning rate,
        r is the reward received for taking action a in state s,
        gamma is the discount factor,
        max(Q(s',a')) is the maximum Q-value for the next state s' and all possible actions a',
        Q(s',a') is the Q-value for the next state s' and action a'

    The Agent differs from the QAgent in that it uses the Upper Confidence Bound (UCB) algorithm to select actions:

        a <- argmax(Q(s,a) + c * sqrt(ln(t) / N(s,a)))

    where:
        a is the action to take,
        Q(s,a) is the current Q-value for state s and action a,
        c is the exploration rate,
        t is the total number of times any action has been taken (sum of all N(s,a)),
        N(s,a) is the number of times action a has been taken in state s
    """

    def __init__(self, alpha: float, gamma: float, c: float) -> None:
        """
        Initializes a new instance of the QAgent class.

        Args:
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            c (float): The exploration rate.
        """

        super().__init__(alpha, gamma, epsilon=0.0)
        self._c = c

        self._N: Dict[State, Dict[Action, int]] = dict()

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

        super().update(state, action, reward, next_state, done)

        self._N[state][action] += 1

    def get_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        """
        Returns an action to take in the given state, using Upper Confidence Bound exploration.

        Args:
            state (State): The current state.
            actions (tuple[Action, ...]): The available actions in the current state.

        Returns:
            Action: The action to take.
        """

        self._populate_tables(state, actions)

        ucb_values = []
        for action in actions:
            if self._N[state][action] == 0:
                ucb_values.append(math.inf)
            else:
                exploration_bonus = self._c * math.sqrt(math.log(sum(self._N[state].values())) / self._N[state][action])
                ucb_value = self._Q[state][action] + exploration_bonus
                ucb_values.append(ucb_value)

        max_ucb_value = max(ucb_values)
        best_actions = [action for action, ucb_value in zip(actions, ucb_values) if ucb_value == max_ucb_value]
        return random.choice(best_actions)

    def _populate_tables(self, state: State, actions: tuple[Action, ...]) -> None:
        """
        Populates the Q-table with initial values for the given state and actions.

        Args:
            state (State): The state to populate the Q-table for.
            actions (tuple[Action, ...]): The actions to populate the Q-table for.
        """

        super()._populate_q_table(state, actions)

        if state not in self._N:
            self._N[state] = {action: 0 for action in actions}
