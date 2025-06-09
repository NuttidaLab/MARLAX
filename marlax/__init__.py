"""
MARLAX: JAX-powered multi-agent reinforcement learning framework.

This package provides the core components required to build, train, and monitor
multi-agent RL experiments using JAX. It exposes:
    - Engine: Training and evaluation loop manager.
    - Tracer: Utilities for logging, checkpointing, and exporting agent data.
"""

from marlax.engines import Engine
from marlax.tracers import Tracer

__all__ = [
    'Engine',
    'Tracer',
]

__version__ = "0.1.0"
