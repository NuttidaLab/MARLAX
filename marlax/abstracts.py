
from abc import ABC, abstractmethod

class Agent(ABC):
    
    @abstractmethod
    def choose(self, state_key, epsilon=0.1):
        raise NotImplementedError("Choose method not implemented.")
    
    @abstractmethod
    def update(self, state_key, action, reward, next_state_key, alpha=0.1, gamma=0.99):
        raise NotImplementedError("Update method not implemented.")
    
class Environment(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Reset method not implemented.")
    
    @abstractmethod
    def step(self, action):
        raise NotImplementedError("Step method not implemented.")