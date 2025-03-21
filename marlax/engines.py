from tqdm import tqdm

class Engine:
    def __init__(self, epsilon_start, epsilon_end, epsilon_test = 0.0):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_test = epsilon_test

    def train(self, env, logger, num_steps = 1_000_000, alpha=0.1, gamma=0.9, verbose=True, flush_every=1_000_000, regime_idx=0):
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