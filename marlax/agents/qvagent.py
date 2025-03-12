from marlax.abstracts import Agent

import random
import numpy as np
from collections import defaultdict
from functools import partial

class QValueAgent(Agent):
    def __init__(self,  init_position = None, actions = ['stay', 'up', 'down', 'left', 'right']):
        
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
        
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
        # print(possible_states)
            max_state = self.get_max_state(possible_states)
            
            current_pos = self.position
            next_pos = max_state[0][agent_id]
            
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            # print("was here ", current_pos, "went here", next_pos, "dx, dy", dx, dy)
            action = self.action_map[(dx, dy)]
            return action
    
    def get_max_state(self, possible_states):
        max_state_id = np.argmax([self.q_table[state_key] for state_key in possible_states])
        max_state = possible_states[max_state_id]
        return max_state
    
    def update(self, state_key, action, reward, next_state_key, alpha=0.1, gamma=0.99):
        self.q_table[state_key] = (1 - alpha) * self.q_table[state_key] + alpha * (reward + gamma * self.q_table[next_state_key])