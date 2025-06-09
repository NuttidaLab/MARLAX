"""Abstract base classes defining the interfaces for Agents and Environments."""


from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    Subclasses must implement decision-making and learning update methods.
    """
    
    @abstractmethod
    def choose(self, state_key, epsilon=0.1):
        """
        Select an action based on the current state and exploration factor.

        Args:
            state_key (any): A hashable identifier representing the current environment state.
            epsilon (float): Probability of choosing a random action for exploration.

        Returns:
            any: The action chosen by the agent.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Choose method not implemented.")
    
    @abstractmethod
    def update(self, state_key, action, reward, next_state_key, alpha=0.1, gamma=0.99):
        """
        Update the agent's knowledge after taking an action.

        Args:
            state_key (any): Identifier for the state where the action was taken.
            action (any): The action that was taken.
            reward (float): Reward received after the action.
            next_state_key (any): Identifier for the subsequent state after the action.
            alpha (float): Learning rate determining the step size for updates.
            gamma (float): Discount factor for future rewards.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Update method not implemented.")
    
class Environment(ABC):
    """
    Abstract base class for environments in which agents operate.

    Subclasses must implement state reset and transition logic.
    """
    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            any: The initial state key or representation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Reset method not implemented.")
    
    @abstractmethod
    def step(self, action):
        """
        Advance the environment by one time step given an action.

        Args:
            action (any): The action to apply in the environment.

        Returns:
            tuple: A tuple `(next_state_key, reward, done, info)`, where:
                next_state_key (any): Identifier for the next state.
                reward (float): Reward achieved by the action.
                done (bool): Boolean flag indicating episode termination.
                info (dict): Optional diagnostic information.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Step method not implemented.")