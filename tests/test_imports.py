# tests/test_imports.py
import pytest
import marlax

def test_can_import():
    # basic smoke‐test
    assert hasattr(marlax, "__version__")

# tests/test_qagent.py
import numpy as np
from marlax.agents import QAgent
from marlax.envs import GridWorld_r0

def test_qagent_action_in_space():
    agent = QAgent()
    env = GridWorld_r0((5,5), [agent], [1.0], 0.1, 0.0)
    state = env.reset()
    action = agent.act(state)
    # assuming discrete action space {0,1,2,3}
    assert isinstance(action, int)
    assert 0 <= action < env.n_actions