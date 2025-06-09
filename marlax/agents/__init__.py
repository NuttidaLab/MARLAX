"""
Agents Package

This package provides various agent implementations for the MARLAX framework,
including Q-learning agents, Q-value agents, independent agents, and deep Q-agents.
"""

from marlax.agents.qagent import QAgent
from marlax.agents.qvagent import QValueAgent


from marlax.agents.qagent import QAgent
from marlax.agents.qvagent import QValueAgent
# from marlax.agents.independent import IndependentAgent
# from marlax.agents.deepqagent import DeepQAgent

__all__ = [
    'QAgent',
    'QValueAgent',
    # 'IndependentAgent',
    # 'DeepQAgent',
]
