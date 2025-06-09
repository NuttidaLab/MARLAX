"""
Engine for running training and testing loops in the MARLAX framework.

Provides methods to train and evaluate agents in a given environment
with adjustable exploration schedules and logging hooks.
"""

from tqdm import tqdm

class Engine:
    """
    Core engine managing training and testing phases.

    Attributes:
        epsilon_start (float): Initial exploration rate.
        epsilon_end (float): Final exploration rate after decay.
        epsilon_test (float): Exploration rate during testing.
    """
    def __init__(self, epsilon_start, epsilon_end, epsilon_test = 0.0):
        """
        Initialize the engine with epsilon schedule parameters.

        Args:
            epsilon_start (float): Starting epsilon for exploration.
            epsilon_end (float): Ending epsilon after decay.
            epsilon_test (float, optional): Epsilon used in testing. Defaults to 0.0.
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_test = epsilon_test

    def train(self, env, logger, num_steps = 1_000_000, alpha=0.1, gamma=0.9, verbose=True, flush_every=1_000_000, regime_idx=0):
        """
        Run the training loop for a specified number of steps.

        Args:
            env (Environment): The environment instance to train on.
            logger (Tracer): Logger for recording training data (or None).
            num_steps (int, optional): Number of training iterations. Defaults to 1_000_000.
            alpha (float, optional): Learning rate for agent updates. Defaults to 0.1.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
            verbose (bool, optional): Display progress bar if True. Defaults to True.
            flush_every (int, optional): Log flush interval. Defaults to 1_000_000.
            regime_idx (int, optional): Identifier for environment regime. Defaults to 0.

        Returns:
            None
        """
        env.reset()
        if logger: logger._init_logger(flush_every, regime_idx, "training")
        for step in tqdm(range(num_steps), disable=not verbose, desc="Training"):
            # Linearly decay epsilon.
            epsilon = ((self.epsilon_end - self.epsilon_start) / num_steps) * step + self.epsilon_start
            
            possible_next_states = env.get_possible_states()
            actions = []
            # Each agent chooses an action based on the next possible states.
            for i, agent in enumerate(env.agents):
                actions.append(agent.choose(possible_next_states, epsilon, agent_id = i))
            
            # Environment processes the actions.
            state, rewards, info = env.step(actions)
            
            possible_next_states = env.get_possible_states()
            
            # Each agent updates its Q-table.
            for i, agent in enumerate(env.agents):
                agent.update(state, actions[i], rewards[i], agent.get_max_state(possible_next_states), alpha, gamma)
            
            # Checkout the tracks
            if logger: logger._log_frame(step, state, rewards, info)
        
        if logger: logger._flush_logger()
    
    def test(self, env, logger, num_steps = 100_000, verbose = True, flush_every=1_000_000, regime_idx=0):
        """
        Run the evaluation loop for a specified number of steps.

        Args:
            env (Environment): The environment instance to test on.
            logger (Tracer): Logger for recording test data (or None).
            num_steps (int, optional): Number of evaluation iterations. Defaults to 100_000.
            verbose (bool, optional): Display progress bar if True. Defaults to True.
            flush_every (int, optional): Log flush interval. Defaults to 1_000_000.
            regime_idx (int, optional): Identifier for environment regime. Defaults to 0.

        Returns:
            None
        """
        env.reset()
        if logger: logger._init_logger(flush_every, regime_idx, "testing")
        for step in tqdm(range(num_steps), disable=not verbose, desc="Testing"):
            
            actions = []
            for i, agent in enumerate(env.agents):
                actions.append(agent.choose(env.get_possible_states(), self.epsilon_test, agent_id = i))
            
            # Environment processes the actions.
            state, rewards, info = env.step(actions)
            
            if logger: logger._log_frame(step, state, rewards, info)
        if logger: logger._flush_logger()