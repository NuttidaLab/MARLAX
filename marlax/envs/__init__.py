"""
Environments Package

This package provides environment implementations for the MARLAX framework.
Currently supports multi-agent grid-world environments under various reward regimes.
"""

from marlax.envs.gridworld.gridworld import (
    GridWorld,
    GridWorld_r0,
    GridWorld_r1,
    GridWorld_r2,
    GridWorld_r3,
    GridWorld_r4,
)

__all__ = [
    'GridWorld',
    'GridWorld_r0',
    'GridWorld_r1',
    'GridWorld_r2',
    'GridWorld_r3',
    'GridWorld_r4',
]
