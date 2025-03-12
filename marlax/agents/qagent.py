from marlax.abstracts import Agent

import random

class QAgent(Agent):
    def __init__(self, init_position = None, actions = ['stay', 'up', 'down', 'left', 'right']):
        """
        Initialize an agent with a starting position and possible action set.
        
        Args:
            init_position (tuple): The (x, y) starting coordinates.
            actions (list): List of possible actions (e.g., ['stay', 'up','down','left','right']).
        """
        self.position = init_position  # Agent's (x, y) position on the grid.
        self.actions = actions # List of possible actions.
        # Q-table: maps global state (all agents' positions + active reward target) to action values.
        # default dict with partial that  defaults to 0.0
        self.q_table = {}

    def choose(self, possible_states, epsilon=0.1, agent_id = 0):
        """
        Choose an action using an epsilon-greedy policy based on the Q-table.
        
        Args:
            state_key: The key representing the current global state.
            epsilon (float): Exploration rate.
            
        Returns:
            action (str): Chosen action.
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
        Update Q-value for the given state and action using the Q-learning update rule.
        
        $Q(s, a) = Q(s, a) + \alpha * (r + \gamma * \max_a Q(s', a) - Q(s, a))$
        
        Args:
            state_key: Current global state key.
            action (str): Action taken.
            reward (float): Immediate reward received.
            next_state_key: Next global state key.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}
        best_next_value = max(self.q_table[next_state_key].values())
        td_target = reward + gamma * best_next_value
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += alpha * td_error
