"""
QValueAgent implementation using Q-value functions for MARLAX.

This agent maintains a mapping from global states to scalar Q-values
and selects actions by inferring the best next position.
"""
from marlax.abstracts import Agent

import random
import numpy as np
from collections import defaultdict
from functools import partial

class QValueAgent(Agent):
    """
    Agent that selects actions based on scalar Q-values per global state.

    Attributes:
        position (tuple): Agent's (x, y) position in the grid.
        actions (list of str): Available actions.
        q_table (defaultdict): Maps state_key to a scalar Q-value.
        action_map (dict): Maps (dx, dy) offsets to action names.
    """
    def __init__(self,  init_position = None, actions = ['stay', 'up', 'down', 'left', 'right']):
        """
        Initialize agent with starting position and possible actions.

        Args:
            init_position (tuple, optional): The (x, y) starting coordinates. Defaults to None.
            actions (list of str, optional): Available actions. Defaults to ['stay', 'up', 'down', 'left', 'right'].
        """
        self.position = init_position
        self.actions = actions
        self.q_table = defaultdict(partial(int, 0))
        
        self.action_map = {
                (0, 0):'stay',
                (0, -1):'up',
                (0, 1):'down',
                (-1, 0):'left',
                (1, 0):'right',
            }
        
    def choose(self, possible_states, epsilon=0.1, agent_id = 0):
        """
        Select an action with epsilon-greedy exploration.

        Args:
            possible_states (list): List of global state keys to evaluate.
            epsilon (float): Exploration probability. Defaults to 0.1.
            agent_id (int): Identifier for this agent. Defaults to 0.

        Returns:
            str: The action corresponding to the move towards the best state.
        """
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
        # print(possible_states)
            max_state = self.get_max_state(possible_states)
            
            current_pos = self.position
            next_pos = max_state[0][agent_id]
            
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            action = self.action_map[(dx, dy)]
            return action
    
    def get_max_state(self, possible_states):
        """
        Identify the state with the highest scalar Q-value.

        Args:
            possible_states (list): List of global state keys.

        Returns:
            any: The state_key with the maximal Q-value.
        """
        max_state_id = np.argmax([self.q_table[state_key] for state_key in possible_states])
        max_state = possible_states[max_state_id]
        return max_state
    
    def update(self, state_key, action, reward, next_state_key, alpha=0.1, gamma=0.99):
        """
        Update the scalar Q-value for a state using a simple update rule.

        $$Q(s) <- (1-alpha)*Q(s) + alpha*(reward + gamma*Q(s'))$$.

        Args:
            state_key (any): Current global state key.
            action (str): Action taken (unused since value is state-based).
            reward (float): Reward received after action.
            next_state_key (any): Next global state key.
            alpha (float): Learning rate. Defaults to 0.1.
            gamma (float): Discount factor. Defaults to 0.99.
        """
        self.q_table[state_key] = (
            (1 - alpha) * self.q_table[state_key]
            + alpha * (reward + gamma * self.q_table[next_state_key])
        )