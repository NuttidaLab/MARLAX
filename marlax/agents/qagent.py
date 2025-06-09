"""
QAgent implementation using Q-learning for MARLAX.

This agent maintains a Q-table mapping global states to action values
and selects actions via an epsilon-greedy strategy.
"""

from marlax.abstracts import Agent

import random

class QAgent(Agent):
    """
    Q-learning agent that chooses actions based on a Q-table.

    Attributes:
        position (tuple): Agent's (x, y) position on the grid.
        actions (list): List of possible actions.
        q_table (dict): Maps state_key to a dict of action->value.
    """
    def __init__(self, init_position = None, actions = ['stay', 'up', 'down', 'left', 'right']):
        """
        Initialize an agent with a starting position and possible actions.

        Args:
            init_position (tuple, optional): The (x, y) starting coordinates. Defaults to None.
            actions (list of str, optional): Available actions. Defaults to ['stay', 'up', 'down', 'left', 'right'].
        """
        self.position = init_position  # Agent's (x, y) position on the grid.
        self.actions = actions # List of possible actions.
        # Q-table: maps global state (all agents' positions + active reward target) to action values.
        # default dict with partial that  defaults to 0.0
        self.q_table = {}

    def choose(self, possible_states, epsilon=0.1, agent_id = 0):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            possible_states (list): List of global state keys to evaluate.
            epsilon (float): Exploration probability. Defaults to 0.1.
            agent_id (int): Identifier for this agent among multiple agents. Defaults to 0.

        Returns:
            str: The chosen action.
        """
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            # for all the possible states, get the action with the highest q-value
            best_state = self.get_max_state(possible_states)
            best_possible_q_value = float('-inf')
            best_possible_action = None

            for action in self.actions:
                if self.q_table[best_state][action] > best_possible_q_value:
                    best_possible_action = action
                    best_possible_q_value = self.q_table[best_state][action]
            
            return best_possible_action
        
    def get_max_state(self, possible_states):
        """
        Identify the state with the highest max Q-value.

        This helper initializes missing table entries to zero.

        Args:
            possible_states (list): List of global state keys.

        Returns:
            any: The state_key with the highest action value.
        """
        best_possible_action = None
        best_possible_q_value = float('-inf')
        best_state = None
        
        for state_key in possible_states:
            
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in self.actions}
            
            best_action = None
            best_q_value = float('-inf')

            for action in self.actions:
                if self.q_table[state_key][action] > best_q_value:
                    best_action = action
                    best_q_value = self.q_table[state_key][action]
            
            if best_q_value > best_possible_q_value:
                best_possible_action = best_action
                best_possible_q_value = best_q_value
                best_state = state_key
        
        return best_state

    def update(self, state_key, action, reward, next_state_key, alpha=0.1, gamma=0.99):
        """
        Update the Q-table entry for a given state and action.

        Applies the Q-learning update rule:
        $$Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a Q(s',a) - Q(s,a))$$.

        Args:
            state_key (any): Current global state key.
            action (str): Action taken by the agent.
            reward (float): Reward received after action.
            next_state_key (any): Next global state key.
            alpha (float): Learning rate. Defaults to 0.1.
            gamma (float): Discount factor. Defaults to 0.99.
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}
        best_next_value = max(self.q_table[next_state_key].values())
        td_target = reward + gamma * best_next_value
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += alpha * td_error
