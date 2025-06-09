# tests/test_qagent.py
import numpy as np
from marlax.agents import QAgent
from marlax.envs import GridWorld_r0

def test_qagent_action_in_space():
    agent = QAgent()
    env = GridWorld_r0((5,5), [agent], [1.0], 0.1, 0.0)
    state = env.reset()